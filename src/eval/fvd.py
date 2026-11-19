"""Fréchet Video Distance (FVD) between two sets of clips (brief §3b "Held-out set metrics", §10).

Protocol (the common StyleGAN-V / VideoGPT one): windows of 16 frames at 224x224, pixel values in [-1, 1], 400-d
features of the Kinetics-400 I3D (the ``i3d_torchscript.pt`` export used by StyleGAN-V, called with
``rescale=False, resize=False, return_features=True``), Fréchet distance between the two Gaussians.

Choices that matter when comparing anime shots with generated shots (all recorded in the output JSON):
  * both sets are resampled to the same frame rate first (default 16 fps = Wan's native rate), so a 16-frame window
    always covers 1.0 s;
  * 16:9 frames are resized to 224 px height and centre-cropped to 224x224 (``--resize squash`` to keep the sides);
  * up to ``--max-windows`` (default 4) non-overlapping windows are taken per clip, spread over the clip;
  * features are cached per clip, keyed by the sha256 of the file content + the protocol.

FVD needs many samples: with fewer windows than feature dimensions (400) the covariance is rank deficient and the
value is biased upwards, so only compare numbers computed with the same N. The output therefore also contains KVD
(unbiased polynomial-kernel MMD) and, with ``--baseline``, FVD between two random halves of the real set.

CLI
---
    python -m src.eval.fvd --real <dir|list> --gen <dir|list> --out fvd.json [--device auto|cuda|cpu] [--baseline]

API
---
    clip_features(path, ...)        -> np.ndarray [K, 400]   (cached)
    set_features(paths, ...)        -> (np.ndarray [N, 400], per-clip window counts)
    frechet_distance(feats_a, feats_b) -> float
    kernel_distance(feats_a, feats_b)  -> float
    fvd(real_paths, gen_paths, ...) -> dict
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from . import video_metrics as vm

CACHE_DIR = Path(os.environ.get("KOMI_EVAL_CACHE", vm.WORK / "cache" / "eval")) / "fvd"
FEATURE_DIM = 400


def _i3d(device: str):
    key = ("i3d", device)
    if key not in vm._MODELS:
        torch = vm._torch()
        vm._MODELS[key] = torch.jit.load(str(vm.ensure_weight("i3d")), map_location=device).eval()
    return vm._MODELS[key]


def clip_windows(path: str | Path, fps: float = 16.0, n_frames: int = 16, size: int = 224, max_windows: int = 4,
                 resize: str = "center_crop") -> np.ndarray:
    """uint8 [K, n_frames, size, size, 3]: up to ``max_windows`` non-overlapping windows spread over the clip.

    Clips shorter than ``n_frames`` (after resampling to ``fps``) are padded by repeating the last frame if they
    have at least n_frames // 2 frames, otherwise K = 0. Only the first ~64 s of a long video are considered.
    """
    cap = max(n_frames * max_windows * 4, int(fps * 64))        # bound memory: at most ~64 s of a long video is used
    if resize == "squash":
        fr = vm.load_frames(path, fps=fps, size=(size, size), max_frames=cap)
    elif resize == "center_crop":
        info = vm.video_info(path)
        w, h = info["width"], info["height"]
        if w >= h:
            fr = vm.load_frames(path, fps=fps, size=size, max_frames=cap)       # height = size
            x0 = (fr.shape[2] - size) // 2
            fr = fr[:, :, x0:x0 + size]
        else:
            tw, th = size, max(size, int(round(h * size / w / 2)) * 2)
            fr = vm.load_frames(path, fps=fps, size=(tw, th), max_frames=cap)
            y0 = (fr.shape[1] - size) // 2
            fr = fr[:, y0:y0 + size]
    else:
        raise ValueError("resize must be 'center_crop' or 'squash'")
    T = len(fr)
    if T < n_frames:
        if T < max(1, n_frames // 2):
            return np.zeros((0, n_frames, size, size, 3), np.uint8)
        fr = np.concatenate([fr, np.repeat(fr[-1:], n_frames - T, axis=0)], 0)
        T = n_frames
    k = max(1, min(max_windows, T // n_frames))
    starts = np.linspace(0, T - n_frames, k).round().astype(int) if k > 1 else np.array([(T - n_frames) // 2])
    return np.stack([fr[s:s + n_frames] for s in starts], 0)


def i3d_features(windows: np.ndarray, device: str | None = None, batch: int = 8) -> np.ndarray:
    """I3D features [K, 400] of uint8 windows [K, T, 224, 224, 3]."""
    if len(windows) == 0:
        return np.zeros((0, FEATURE_DIM), np.float64)
    torch = vm._torch()
    dev = vm.pick_device(device, need_gb=4.0)
    net = _i3d(dev)
    out = []
    with torch.inference_mode():
        for s in range(0, len(windows), batch):
            x = torch.from_numpy(windows[s:s + batch]).to(dev).float() / 127.5 - 1.0        # [B, T, H, W, 3]
            x = x.permute(0, 4, 1, 2, 3).contiguous()                                        # [B, 3, T, H, W]
            f = net(x, rescale=False, resize=False, return_features=True)
            out.append(f.double().cpu().numpy())
    return np.concatenate(out, 0)


def _proto_tag(fps, n_frames, size, max_windows, resize) -> str:
    return f"i3d_f{n_frames}_s{size}_fps{fps:g}_w{max_windows}_{resize}"


def clip_features(path: str | Path, fps: float = 16.0, n_frames: int = 16, size: int = 224, max_windows: int = 4,
                  resize: str = "center_crop", device: str | None = None, batch: int = 8,
                  cache_dir: str | Path | None = CACHE_DIR) -> np.ndarray:
    """Cached I3D features [K, 400] of one clip (cache key: sha256 of the file content + protocol)."""
    cpath = None
    if cache_dir is not None:
        cpath = Path(cache_dir) / f"{vm.sha256_file(path)[:32]}_{_proto_tag(fps, n_frames, size, max_windows, resize)}.npy"
        if cpath.exists():
            try:
                return np.load(cpath)
            except Exception:
                cpath.unlink(missing_ok=True)
    feats = i3d_features(clip_windows(path, fps, n_frames, size, max_windows, resize), device=device, batch=batch)
    if cpath is not None:
        cpath.parent.mkdir(parents=True, exist_ok=True)
        tmp = cpath.with_suffix(f".{os.getpid()}.tmp.npy")
        np.save(tmp, feats)
        os.replace(tmp, cpath)
    return feats


def set_features(paths: Sequence[str | Path], verbose: bool = False, **kw) -> tuple[np.ndarray, list[int], list[str]]:
    """Features of a clip set: ([N, 400], windows per clip, list of "path: error" for clips that failed)."""
    feats, counts, errors = [], [], []
    for i, p in enumerate(paths):
        try:
            f = clip_features(p, **kw)
        except Exception as e:  # unreadable clip: report, do not abort the whole set
            errors.append(f"{p}: {type(e).__name__}: {e}"[:300])
            f = np.zeros((0, FEATURE_DIM))
        feats.append(f)
        counts.append(int(len(f)))
        if verbose and (i + 1) % 20 == 0:
            print(f"[fvd] {i + 1}/{len(paths)} clips", file=sys.stderr, flush=True)
    allf = np.concatenate(feats, 0) if feats else np.zeros((0, FEATURE_DIM))
    return allf, counts, errors


def _sqrt_psd(m: np.ndarray) -> np.ndarray:
    w, v = np.linalg.eigh((m + m.T) / 2.0)
    return (v * np.sqrt(np.clip(w, 0.0, None))) @ v.T


def frechet_distance(feats_a: np.ndarray, feats_b: np.ndarray) -> float:
    """||mu_a - mu_b||^2 + Tr(S_a + S_b - 2 (S_a S_b)^(1/2)), computed through symmetric PSD square roots
    (stable for rank-deficient covariances, never complex)."""
    a, b = np.asarray(feats_a, np.float64), np.asarray(feats_b, np.float64)
    if len(a) < 2 or len(b) < 2:
        raise ValueError("need at least 2 feature vectors per set")
    mu_a, mu_b = a.mean(0), b.mean(0)
    s_a, s_b = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    ra = _sqrt_psd(s_a)
    tr_covmean = np.sqrt(np.clip(np.linalg.eigvalsh(ra @ s_b @ ra), 0.0, None)).sum()
    d = mu_a - mu_b
    return float(d @ d + np.trace(s_a) + np.trace(s_b) - 2.0 * tr_covmean)


def kernel_distance(feats_a: np.ndarray, feats_b: np.ndarray, subset: int = 1000, n_subsets: int = 50,
                    seed: int = 0) -> float:
    """KVD: unbiased MMD^2 with the cubic polynomial kernel (x.y / d + 1)^3, same scale as the ``kvd`` of the
    StyleGAN-V metrics code (not multiplied by 1000). Usable with small sets; can be slightly negative."""
    a, b = np.asarray(feats_a, np.float64), np.asarray(feats_b, np.float64)
    m = min(len(a), len(b), subset)
    if m < 2:
        raise ValueError("need at least 2 feature vectors per set")
    d = a.shape[1]
    rng = np.random.default_rng(seed)
    full = m == len(a) == len(b)
    vals = []
    for _ in range(1 if full else n_subsets):
        x = a if full else a[rng.choice(len(a), m, replace=False)]
        y = b if full else b[rng.choice(len(b), m, replace=False)]
        kxx = (x @ x.T / d + 1) ** 3
        kyy = (y @ y.T / d + 1) ** 3
        kxy = (x @ y.T / d + 1) ** 3
        vals.append((kxx.sum() - np.trace(kxx) + kyy.sum() - np.trace(kyy)) / (m * (m - 1)) - 2 * kxy.mean())
    return float(np.mean(vals))


def fvd(real_paths: Sequence[str | Path], gen_paths: Sequence[str | Path], fps: float = 16.0, n_frames: int = 16,
        size: int = 224, max_windows: int = 4, resize: str = "center_crop", device: str | None = None,
        batch: int = 8, cache_dir: str | Path | None = CACHE_DIR, baseline: bool = False, seed: int = 0,
        verbose: bool = False) -> dict:
    """FVD (+ KVD) between two clip sets. See the module docstring for the protocol and its limits."""
    kw = dict(fps=fps, n_frames=n_frames, size=size, max_windows=max_windows, resize=resize, device=device,
              batch=batch, cache_dir=cache_dir, verbose=verbose)
    t0 = time.time()
    fr, cr, er = set_features(real_paths, **kw)
    fg, cg, eg = set_features(gen_paths, **kw)
    warnings = []
    if er or eg:
        warnings.append(f"{len(er) + len(eg)} clip(s) could not be read (see 'errors')")
    n = min(len(fr), len(fg))
    if n < FEATURE_DIM:
        warnings.append(f"only {n} windows in the smaller set (< {FEATURE_DIM} feature dims): FVD is biased upwards; "
                        "compare only against values computed with the same N (see 'baseline' / KVD)")
    res = {
        "fvd": None, "kvd": None,
        "n_real_clips": len(real_paths), "n_gen_clips": len(gen_paths),
        "n_real_windows": int(len(fr)), "n_gen_windows": int(len(fg)),
        "protocol": {"model": "i3d_torchscript (Kinetics-400, 400-d)", "frames": n_frames, "size": size, "fps": fps,
                     "max_windows_per_clip": max_windows, "resize": resize},
        "warnings": warnings, "errors": er + eg,
    }
    if len(fr) >= 2 and len(fg) >= 2:
        res["fvd"] = frechet_distance(fr, fg)
        res["kvd"] = kernel_distance(fr, fg, seed=seed)
    else:
        warnings.append("not enough readable windows to compute FVD")
    if baseline and len(fr) >= 8:
        # noise floor: real vs real with the same number of windows as the real-vs-gen comparison allows
        rng = np.random.default_rng(seed)
        clip_ids = np.repeat(np.arange(len(cr)), cr)
        vals = []
        for _ in range(5):
            perm = rng.permutation(len(cr))                      # split by clip, so windows of a clip stay together
            half = set(perm[: len(perm) // 2].tolist())
            m = np.array([c in half for c in clip_ids])
            if m.sum() >= 2 and (~m).sum() >= 2:
                vals.append(frechet_distance(fr[m], fr[~m]))
        if vals:
            res["baseline_real_vs_real_halves"] = {"fvd_mean": float(np.mean(vals)), "fvd_std": float(np.std(vals)),
                                                   "windows_per_half": int(len(fr) // 2)}
    res["runtime_s"] = round(time.time() - t0, 2)
    return res


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Fréchet Video Distance between two clip sets (I3D, 16 frames, 224 px).")
    ap.add_argument("--real", nargs="+", required=True, help="clips, directories, .txt lists or labels.jsonl")
    ap.add_argument("--gen", nargs="+", required=True)
    ap.add_argument("--out", help="write JSON here (default: stdout)")
    ap.add_argument("--fps", type=float, default=16.0)
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--max-windows", type=int, default=4)
    ap.add_argument("--resize", default="center_crop", choices=["center_crop", "squash"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--cache-dir", default=str(CACHE_DIR))
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--baseline", action="store_true", help="also report FVD between random halves of the real set")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    os.environ.setdefault("KOMI_EVAL_THREADS", str(vm.n_threads()))
    real, gen = vm.list_clips(a.real), vm.list_clips(a.gen)
    res = fvd(real, gen, fps=a.fps, n_frames=a.frames, size=a.size, max_windows=a.max_windows, resize=a.resize,
              device=a.device, batch=a.batch, cache_dir=None if a.no_cache else a.cache_dir, baseline=a.baseline,
              verbose=not a.quiet)
    txt = json.dumps(res, indent=1)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(txt)
    if not a.quiet or not a.out:
        print(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
