"""Local video evidence metrics for the QA judge and the held-out evaluation (brief §3b, §8, §10).

Everything in this module is *evidence* for the judge; a metric alone never passes or fails a shot.
The single exception is :func:`technical_check` (corrupt file, wrong duration, black/frozen video,
silent audio), which is allowed to fail a shot automatically.

Public API
----------
    load_frames(path, fps=None, size=None, max_frames=None) -> np.ndarray[T, H, W, 3] uint8 RGB
    flicker_score(frames, fps=None)                          -> dict
    flow_stats(frames, method="raft_small", device=None)     -> dict of per-pair arrays (shared by the two below)
    warp_error(frames, ...)                                  -> float (mean error) | dict with return_details=True
    motion_stats(frames, ...)                                -> dict
    dover_score(path, device=None)                           -> {"technical", "aesthetic", "overall", ...}
    technical_check(path, expected_dur=None, need_audio=False) -> {"ok": bool, "reasons": [...], ...}
    analyze_clip(path, metrics=..., ...)                     -> dict with everything (one decode, one flow pass)

CLI
---
    python -m src.eval.video_metrics <clip-or-dir>... --out metrics.json \
        [--metrics technical,flicker,motion,warp,dover] [--device auto|cuda|cpu] [--flow raft_small|raft_large|farneback]

Design notes
------------
* Everything runs on CPU or GPU. ``device=None`` means "cuda if enough VRAM is free, else cpu"; the free-VRAM
  test uses ``nvidia-smi`` so that no CUDA context is created on a busy card.
* torch / torchvision / av are imported lazily, so importing this module is cheap (the web tools and the pure
  python tests use its ffmpeg helpers without paying for torch).
* Model weights live in ``$KOMI_WORK/models/eval`` (downloaded once, then fully offline).
"""
from __future__ import annotations

import contextlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

WORK = Path(os.environ.get("KOMI_WORK", "/home/user/komi-poc-work"))
MODELS_DIR = Path(os.environ.get("KOMI_EVAL_MODELS", WORK / "models" / "eval"))
DOVER_REPO = Path(os.environ.get("KOMI_DOVER_REPO", WORK / "third_party" / "DOVER"))
VIDEO_EXTS = (".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v")

# Official download locations (fetched once into MODELS_DIR; sha256 checked).
WEIGHTS = {
    "raft_small": ("raft_small_C_T_V2-01064c6d.pth",
                   "https://download.pytorch.org/models/raft_small_C_T_V2-01064c6d.pth",
                   "01064c6dba73b0fc9fc8edf772248560a00a3acfd62ac6677e9eeebad9680e27"),
    "raft_large": ("raft_large_C_T_SKHT_V2-ff5fadd5.pth",
                   "https://download.pytorch.org/models/raft_large_C_T_SKHT_V2-ff5fadd5.pth",
                   "ff5fadd56d26b40647388883af1547351ea17868b765c05b27231e72dd16a322"),
    "dover": ("DOVER.pth",
              "https://github.com/VQAssessment/DOVER/releases/download/v0.1.0/DOVER.pth",
              "f4a42c0bbc94c94dd7409e7f40887d44c5c30314d1d09e7edf03cc35813b4838"),
    "i3d": ("i3d_torchscript.pt",
            "https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1",
            "bec6519f66ea534e953026b4ae2c65553c17bf105611c746d904657e5860a5e2"),
}

# ---- thresholds (calibrated on ep01/ep02 proxy clips re-encoded with x264; see docs/eval_tools.md) ----
HOLD_MAD = 0.12        # mean |dY| (0..255, 180p luma) below which a frame pair can be a held (repeated) drawing
HOLD_BLOCK = 1.5       # ... and no 8x8 block may change by more than this on average (keeps lip flaps / slow pans "moving")
CUT_HIST = 0.35        # colour-histogram total-variation distance that marks a hard cut (with CUT_MAD)
CUT_MAD = 10.0
CUT_ZNCC = 0.45        # structural (zero-mean correlation) test for cuts between shots with the same palette
STATIC_FLOW_PX = 0.10  # mean flow magnitude (px at the analysis height) below which a pair counts as static


# ----------------------------------------------------------------------------------------------------------------
# small helpers: binaries, probing, devices, weights
# ----------------------------------------------------------------------------------------------------------------
def _find_bin(name: str) -> str:
    cands = [os.environ.get(f"KOMI_{name.upper()}"), shutil.which(name), str(WORK / "tools" / "ffenv" / "bin" / name)]
    for c in cands:
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    raise RuntimeError(f"{name} not found (set KOMI_{name.upper()} or source scripts/env.sh)")


def ffmpeg_bin() -> str:
    return _find_bin("ffmpeg")


def ffprobe_bin() -> str:
    return _find_bin("ffprobe")


def n_threads() -> int:
    """CPU threads this module may use (env KOMI_EVAL_THREADS, default 8)."""
    try:
        return max(1, int(os.environ.get("KOMI_EVAL_THREADS", "8")))
    except ValueError:
        return 8


class ProbeError(RuntimeError):
    pass


def probe(path: str | Path) -> dict:
    """ffprobe -> {"format": {...}, "streams": [...]}; raises ProbeError when the file cannot be parsed."""
    r = subprocess.run([ffprobe_bin(), "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise ProbeError((r.stderr or "ffprobe failed").strip().splitlines()[-1][:300] if r.stderr.strip() else "ffprobe failed")
    try:
        return json.loads(r.stdout)
    except json.JSONDecodeError as e:  # pragma: no cover
        raise ProbeError(f"ffprobe returned invalid json: {e}")


def _ratio(s: str | None) -> float:
    if not s or s in ("0/0", "N/A"):
        return 0.0
    if "/" in s:
        a, b = s.split("/")
        return float(a) / float(b) if float(b) else 0.0
    return float(s)


def video_info(path: str | Path) -> dict:
    """Flat summary of a media file (raises ProbeError if unreadable)."""
    p = probe(path)
    fmt = p.get("format", {})
    v = next((s for s in p.get("streams", []) if s.get("codec_type") == "video"
              and s.get("disposition", {}).get("attached_pic", 0) == 0), None)
    a = next((s for s in p.get("streams", []) if s.get("codec_type") == "audio"), None)

    def _f(x, default=0.0):
        try:
            return float(x)
        except (TypeError, ValueError):
            return default

    info: dict[str, Any] = {
        "path": str(path), "size_bytes": int(_f(fmt.get("size"), 0)), "container_duration": _f(fmt.get("duration")),
        "format_name": fmt.get("format_name"), "has_video": v is not None, "has_audio": a is not None,
    }
    if v is not None:
        fps = _ratio(v.get("avg_frame_rate")) or _ratio(v.get("r_frame_rate"))
        info.update(width=int(v.get("width", 0)), height=int(v.get("height", 0)), fps=fps,
                    video_codec=v.get("codec_name"), pix_fmt=v.get("pix_fmt"),
                    video_duration=_f(v.get("duration"), _f(fmt.get("duration"))),
                    nb_frames=int(_f(v.get("nb_frames"), 0)))
    if a is not None:
        info.update(audio_codec=a.get("codec_name"), audio_duration=_f(a.get("duration"), _f(fmt.get("duration"))),
                    sample_rate=int(_f(a.get("sample_rate"), 0)), channels=int(_f(a.get("channels"), 0)))
    return info


def list_clips(inputs: Iterable[str | Path]) -> list[Path]:
    """Expand files / directories (recursive) / text files with one path per line into a sorted clip list."""
    out: list[Path] = []
    for x in inputs:
        p = Path(x)
        if p.is_dir():
            out += sorted(q for q in p.rglob("*") if q.suffix.lower() in VIDEO_EXTS)
        elif p.suffix.lower() in (".txt", ".lst"):
            out += [Path(l.strip()) for l in p.read_text().splitlines() if l.strip() and not l.startswith("#")]
        elif p.suffix.lower() == ".jsonl":
            out += [Path(json.loads(l)["path"]) for l in p.read_text().splitlines() if l.strip()]
        else:
            out.append(p)
    return out


def list_clip_specs(inputs: Iterable[str | Path]) -> list[tuple[Path, dict]]:
    """Like list_clips, but keeps the per-clip spec found in .jsonl rows: expected_dur, need_audio, allow_static."""
    out: list[tuple[Path, dict]] = []
    for x in inputs:
        p = Path(x)
        if p.suffix.lower() == ".jsonl":
            for l in p.read_text().splitlines():
                if l.strip():
                    r = json.loads(l)
                    out.append((Path(r["path"]), {k: r[k] for k in ("expected_dur", "need_audio", "allow_static") if k in r}))
        else:
            out += [(c, {}) for c in list_clips([p])]
    return out


def free_vram_gb() -> float:
    """Free VRAM of GPU 0 according to nvidia-smi (0.0 when there is no GPU). Never creates a CUDA context."""
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) == "":
        return 0.0
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10)
        return float(r.stdout.strip().splitlines()[0]) / 1024.0
    except Exception:
        return 0.0


def pick_device(device: str | None = None, need_gb: float = 4.0) -> str:
    """Resolve a device string. None/"auto" -> "cuda" only if >= need_gb of VRAM is free right now, else "cpu".

    The env var KOMI_EVAL_DEVICE overrides the automatic choice (e.g. KOMI_EVAL_DEVICE=cpu while the card is busy).
    """
    if device in (None, "", "auto"):
        device = os.environ.get("KOMI_EVAL_DEVICE") or "auto"
    if device != "auto":
        return device
    if free_vram_gb() < need_gb:
        return "cpu"
    torch = _torch()
    return "cuda" if torch.cuda.is_available() else "cpu"


_TORCH = None


def _torch():
    """Lazy torch import; applies the CPU thread cap once."""
    global _TORCH
    if _TORCH is None:
        import torch

        if "KOMI_EVAL_THREADS" in os.environ or torch.get_num_threads() > n_threads():
            torch.set_num_threads(n_threads())
        _TORCH = torch
    return _TORCH


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def ensure_weight(key: str) -> Path:
    """Return the local path of a weight file, downloading it once from its official URL (sha256 verified)."""
    fname, url, sha = WEIGHTS[key]
    dst = MODELS_DIR / fname
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    import urllib.request

    print(f"[eval] downloading {fname} (once) ...", file=sys.stderr, flush=True)
    with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:  # noqa: S310 (fixed official URLs)
        shutil.copyfileobj(r, f)
    got = sha256_file(tmp)
    if got != sha:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"checksum mismatch for {fname}: {got} != {sha}")
    os.replace(tmp, dst)
    return dst


# ----------------------------------------------------------------------------------------------------------------
# frame loading
# ----------------------------------------------------------------------------------------------------------------
def _target_size(w: int, h: int, size: int | Sequence[int] | None) -> tuple[int, int]:
    """size=None keeps (w, h); int = target height (aspect kept, even width); (W, H) = exact."""
    if size is None:
        return w, h
    if isinstance(size, (int, np.integer)):
        th = int(size)
        tw = max(2, int(round(w * th / h / 2)) * 2)
        return tw, th
    tw, th = size
    return int(tw), int(th)


def load_frames(path: str | Path, fps: float | None = None, size: int | Sequence[int] | None = None,
                max_frames: int | None = None, start: float = 0.0, return_fps: bool = False):
    """Decode a clip into ``uint8 [T, H, W, 3]`` RGB.

    fps        resample to this constant frame rate (each output frame is the source frame displayed at k/fps,
               i.e. frames are dropped or repeated, never blended). None keeps every decoded frame.
    size       None = native; int = target height (aspect kept); (W, H) = exact size.
    max_frames stop after this many output frames.
    start      skip this many seconds first.
    return_fps also return the frame rate of the returned array (source fps, or ``fps`` when resampling).
    """
    import av

    out: list[np.ndarray] = []
    with av.open(str(path)) as c:
        if not c.streams.video:
            raise ValueError(f"no video stream in {path}")
        vs = c.streams.video[0]
        vs.thread_type = "AUTO"
        with contextlib.suppress(Exception):
            vs.codec_context.thread_count = min(n_threads(), 8)
        src_fps = float(vs.average_rate or vs.guessed_rate or 0) or 24.0
        tb = float(vs.time_base) if vs.time_base else 1.0 / src_fps
        tw, th = _target_size(vs.codec_context.width, vs.codec_context.height, size)
        resize = (tw, th) != (vs.codec_context.width, vs.codec_context.height)
        interp = "AREA" if tw * th < vs.codec_context.width * vs.codec_context.height else "BICUBIC"

        from av.video.reformatter import VideoReformatter

        rf = VideoReformatter()          # one shared swscale context (a per-frame one costs 2-3x the decode time)

        def conv(fr) -> np.ndarray:
            if resize:
                return rf.reformat(fr, width=tw, height=th, format="rgb24", interpolation=interp).to_ndarray()
            return rf.reformat(fr, format="rgb24").to_ndarray()

        t0 = None
        idx = 0
        prev = None          # (time, frame) most recently decoded and not yet "expired"
        k = 0                # next output index when resampling
        done = False
        for fr in c.decode(vs):
            t = fr.pts * tb if fr.pts is not None else idx / src_fps
            idx += 1
            if t0 is None:
                t0 = t
            t = t - t0 - start
            if t < -1e-6:
                continue
            if fps is None:
                out.append(conv(fr))
                if max_frames and len(out) >= max_frames:
                    done = True
                    break
                continue
            # resampling: output frame k shows the last source frame with time <= k / fps
            while prev is not None and k / fps < t - 1e-6:
                out.append(prev)
                k += 1
                if max_frames and len(out) >= max_frames:
                    done = True
                    break
            if done:
                break
            prev = conv(fr)
            last_t = t
        if fps is not None and not done and prev is not None:
            end_t = last_t + 1.0 / src_fps
            while k / fps < end_t - 1e-6:
                out.append(prev)
                k += 1
                if max_frames and len(out) >= max_frames:
                    break
    if not out:
        raise ValueError(f"no frames decoded from {path}")
    arr = np.stack(out, 0)
    return (arr, float(fps or src_fps)) if return_fps else arr


def count_frames(path: str | Path) -> int:
    """Number of decodable video frames (container metadata when present, else a counting decode)."""
    import av

    with av.open(str(path)) as c:
        if not c.streams.video:
            return 0
        vs = c.streams.video[0]
        if vs.frames and vs.frames > 0:
            return int(vs.frames)
        vs.thread_type = "AUTO"
        return sum(1 for _ in c.decode(vs))


def frames_at(path: str | Path, indices: Iterable[int]) -> dict[int, np.ndarray]:
    """Decode only the frames with the given indices (native size, RGB). Indices past the end map to the last frame."""
    import av

    want = sorted({int(i) for i in indices})
    out: dict[int, np.ndarray] = {}
    if not want:
        return out
    need = set(want)
    last_fr = None
    from av.video.reformatter import VideoReformatter

    rf = VideoReformatter()
    with av.open(str(path)) as c:
        vs = c.streams.video[0]
        vs.thread_type = "AUTO"
        with contextlib.suppress(Exception):
            vs.codec_context.thread_count = min(n_threads(), 8)
        for i, fr in enumerate(c.decode(vs)):
            if i > want[-1]:
                break
            last_fr = fr
            if i in need:
                out[i] = rf.reformat(fr, format="rgb24").to_ndarray()
        if last_fr is None:
            raise ValueError(f"no frames decoded from {path}")
        if len(out) < len(want):                    # the file is shorter than its metadata claims
            tail = rf.reformat(last_fr, format="rgb24").to_ndarray()
            for i in want:
                out.setdefault(i, tail)
    return out


def _small_luma(frames: np.ndarray, height: int = 180) -> np.ndarray:
    """float32 luma [T, h, w] (0..255) at a small working height (area-averaged: suppresses codec noise)."""
    import cv2

    T, H, W = frames.shape[:3]
    h = min(height, H)
    w = max(8, int(round(W * h / H / 8)) * 8)
    h = max(8, (h // 8) * 8)
    out = np.empty((T, h, w), np.float32)
    for i in range(T):
        g = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
        out[i] = cv2.resize(g, (w, h), interpolation=cv2.INTER_AREA)
    return out


def _pair_diffs(luma: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per consecutive pair: mean |dY| and the largest 8x8-block mean |dY|."""
    d = np.abs(np.diff(luma, axis=0))                                    # [T-1, h, w]
    mad = d.mean(axis=(1, 2))
    T1, h, w = d.shape
    blk = d.reshape(T1, h // 8, 8, w // 8, 8).mean(axis=(2, 4))
    return mad, blk.reshape(T1, -1).max(axis=1) if T1 else np.zeros(0, np.float32)


def held_pairs(frames: np.ndarray, mad_thr: float = HOLD_MAD, block_thr: float = HOLD_BLOCK) -> np.ndarray:
    """bool [T-1]: True where frame t+1 repeats frame t (a held drawing), up to codec noise."""
    if len(frames) < 2:
        return np.zeros(0, bool)
    mad, blk = _pair_diffs(_small_luma(frames))
    return (mad < mad_thr) & (blk < block_thr)


def _runs(mask: np.ndarray) -> list[int]:
    """Lengths of the runs of True in a boolean vector."""
    runs, n = [], 0
    for m in mask:
        if m:
            n += 1
        elif n:
            runs.append(n)
            n = 0
    if n:
        runs.append(n)
    return runs


# ----------------------------------------------------------------------------------------------------------------
# flicker / held frames
# ----------------------------------------------------------------------------------------------------------------
def flicker_score(frames: np.ndarray, fps: float | None = None, hold_mad: float = HOLD_MAD,
                  hold_block: float = HOLD_BLOCK) -> dict:
    """Frame-difference flicker statistics, aware of limited animation (drawings held for 2-3 frames).

    Keys (all pixel values on a 0..255 scale):
      mean_abs_diff          mean |I[t+1]-I[t]| over all pairs (VBench-style "temporal flicker" is 1 - this/255)
      mean_abs_diff_moving   same, only over pairs that are not held frames
      hf_energy              mean |I[t-1] - 2 I[t] + I[t+1]| / 2  (temporal high-pass, all frames)
      hf_energy_unique       same after removing held frames (a clean "on twos" shot scores low here)
      hf_ratio_unique        hf_energy_unique / mean diff of unique frames; ~0 smooth motion ... ~1+ = every
                             drawing differs from both neighbours in an uncorrelated way (shuffle, boiling)
      luma_flicker           mean |second difference| of the per-frame mean luma (global brightness pumping)
      luma_flicker_p95, chroma_flicker   same idea: 95th percentile / on mean Cr, Cb
      repeat_ratio           fraction of frame pairs that are exact repeats (held drawings)
      unique_frames, unique_ratio, effective_fps (if fps given)
      mean_hold, hold_hist   mean number of frames a drawing is shown; share of drawings shown 1/2/3/4+ frames
      n_frames
      held_frame_ratio, flicker_index   aliases of repeat_ratio and luma_flicker

    hold_mad / hold_block: thresholds of the held-frame test. The defaults suit good encodes (x264 crf <= ~20);
    heavy compression adds noise to held frames, which then count as changes (see docs/eval_tools.md, limits).
    """
    import cv2

    T = int(len(frames))
    res: dict[str, Any] = {"n_frames": T}
    if T < 2:
        res.update(mean_abs_diff=0.0, mean_abs_diff_moving=0.0, hf_energy=0.0, hf_energy_unique=0.0,
                   hf_ratio_unique=0.0, luma_flicker=0.0, luma_flicker_p95=0.0, chroma_flicker=0.0,
                   repeat_ratio=0.0, unique_frames=T, unique_ratio=1.0, mean_hold=1.0,
                   hold_hist={"1": 1.0, "2": 0.0, "3": 0.0, "4+": 0.0}, vbench_temporal_flicker=1.0,
                   held_frame_ratio=0.0, flicker_index=0.0)
        if fps:
            res["effective_fps"] = float(fps)
        return res
    luma = _small_luma(frames)
    mad, blk = _pair_diffs(luma)
    held = (mad < hold_mad) & (blk < hold_block)
    # drawings = maximal runs of held pairs; hold length = run + 1
    holds: list[int] = []
    n = 1
    for hflag in held:
        if hflag:
            n += 1
        else:
            holds.append(n)
            n = 1
    holds.append(n)
    uniq_idx = np.concatenate([[0], np.nonzero(~held)[0] + 1])
    uniq = luma[uniq_idx]

    def _hf(x: np.ndarray) -> float:
        if len(x) < 3:
            return 0.0
        return float(np.abs(x[:-2] - 2 * x[1:-1] + x[2:]).mean() / 2.0)

    d_uniq = float(np.abs(np.diff(uniq, axis=0)).mean()) if len(uniq) > 1 else 0.0
    hf_u = _hf(uniq)

    # global brightness / colour pumping
    small = np.stack([cv2.resize(f, (64, 36), interpolation=cv2.INTER_AREA) for f in frames]).astype(np.float32)
    ycc = np.stack([cv2.cvtColor(s.astype(np.uint8), cv2.COLOR_RGB2YCrCb).reshape(-1, 3).mean(0) for s in small])
    if T >= 3:
        d2 = np.abs(ycc[:-2] - 2 * ycc[1:-1] + ycc[2:])
        luma_f, luma_p95 = float(d2[:, 0].mean()), float(np.percentile(d2[:, 0], 95))
        chroma_f = float(d2[:, 1:].mean())
    else:
        luma_f = luma_p95 = chroma_f = 0.0

    hl = np.asarray(holds)
    res.update(
        mean_abs_diff=float(mad.mean()),
        mean_abs_diff_moving=float(mad[~held].mean()) if (~held).any() else 0.0,
        vbench_temporal_flicker=float(1.0 - mad.mean() / 255.0),
        hf_energy=_hf(luma),
        hf_energy_unique=hf_u,
        hf_ratio_unique=float(hf_u / d_uniq) if d_uniq > 1e-6 else 0.0,
        luma_flicker=luma_f, luma_flicker_p95=luma_p95, chroma_flicker=chroma_f,
        repeat_ratio=float(held.mean()),
        unique_frames=int(len(uniq_idx)),
        unique_ratio=float(len(uniq_idx) / T),
        mean_hold=float(hl.mean()),
        hold_hist={"1": float((hl == 1).mean()), "2": float((hl == 2).mean()),
                   "3": float((hl == 3).mean()), "4+": float((hl >= 4).mean())},
    )
    # aliases used by src/eval/episode_eval.py, src/eval/ab_judge.py and the judge prompt
    res["held_frame_ratio"] = res["repeat_ratio"]
    res["flicker_index"] = res["luma_flicker"]
    if fps:
        res["effective_fps"] = float(fps * len(uniq_idx) / T)
    return res


# ----------------------------------------------------------------------------------------------------------------
# hard cuts (no optical flow needed)
# ----------------------------------------------------------------------------------------------------------------
def detect_cuts(frames: np.ndarray, return_scores: bool = False):
    """Indices i such that frame i is the first frame of a new shot (hard cut between i-1 and i).

    A pair is a cut when the picture changes a lot (mean |dY| > CUT_MAD) and either the colour histogram changes
    (total-variation distance > CUT_HIST) or the structure is uncorrelated (ZNCC of 64x36 luma < CUT_ZNCC), and the
    change is isolated in time (much larger than its neighbours: flashes/flicker/fast pans are not cuts).
    """
    import cv2

    T = len(frames)
    if T < 2:
        return ([], {}) if return_scores else []
    small = np.stack([cv2.resize(f, (64, 36), interpolation=cv2.INTER_AREA) for f in frames])
    g = small.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)          # [T, 36, 64]
    mad = np.abs(np.diff(g, axis=0)).mean(axis=(1, 2))
    q = (small // 32).astype(np.int32)                                                     # 8 levels / channel
    codes = (q[..., 0] * 64 + q[..., 1] * 8 + q[..., 2]).reshape(T, -1)
    hist = np.stack([np.bincount(c, minlength=512) for c in codes]).astype(np.float32) / codes.shape[1]
    hist_d = 0.5 * np.abs(np.diff(hist, axis=0)).sum(axis=1)
    z = g - g.mean(axis=(1, 2), keepdims=True)
    zn = np.sqrt((z ** 2).sum(axis=(1, 2))) + 1e-6
    zncc = (z[:-1] * z[1:]).sum(axis=(1, 2)) / (zn[:-1] * zn[1:])
    flat = (zn[:-1] < 64) | (zn[1:] < 64)             # (near) uniform frames: correlation is meaningless
    zncc = np.where(flat, 1.0, zncc)
    cand = (mad > CUT_MAD) & ((hist_d > CUT_HIST) | (zncc < CUT_ZNCC))
    cuts = []
    for i in np.nonzero(cand)[0]:
        lo, hi = max(0, i - 3), min(len(mad), i + 4)
        neigh = np.concatenate([mad[lo:i], mad[i + 1:hi]])
        if len(neigh) == 0 or mad[i] > 2.5 * np.median(neigh) + 4.0:
            cuts.append(int(i + 1))
    if return_scores:
        return cuts, {"mad": mad, "hist_d": hist_d, "zncc": zncc}
    return cuts


# ----------------------------------------------------------------------------------------------------------------
# optical flow (RAFT / Farneback) -> per-pair statistics
# ----------------------------------------------------------------------------------------------------------------
_MODELS: dict[tuple[str, str], Any] = {}


def _raft(name: str, device: str):
    key = (name, device)
    if key not in _MODELS:
        torch = _torch()
        from torchvision.models.optical_flow import raft_large, raft_small

        net = (raft_small if name == "raft_small" else raft_large)(weights=None)
        sd = torch.load(ensure_weight(name), map_location="cpu", weights_only=True)
        net.load_state_dict(sd)
        _MODELS[key] = net.eval().to(device)
    return _MODELS[key]


def unload_models() -> None:
    """Drop cached models (RAFT, DOVER, I3D) and free VRAM."""
    _MODELS.clear()
    if _TORCH is not None and _TORCH.cuda.is_available() and _TORCH.cuda.is_initialized():
        _TORCH.cuda.empty_cache()


def _warp(x, flow):
    """Sample x at (grid + flow). x: [B, C, H, W], flow: [B, 2, H, W] in px. Returns (warped, in_bounds_mask)."""
    torch = _torch()
    B, _, H, W = x.shape
    ys, xs = torch.meshgrid(torch.arange(H, device=x.device, dtype=x.dtype),
                            torch.arange(W, device=x.device, dtype=x.dtype), indexing="ij")
    gx = xs[None] + flow[:, 0]
    gy = ys[None] + flow[:, 1]
    inb = (gx >= 0) & (gx <= W - 1) & (gy >= 0) & (gy <= H - 1)
    grid = torch.stack([2 * gx / (W - 1) - 1, 2 * gy / (H - 1) - 1], -1)
    out = torch.nn.functional.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=True)
    return out, inb


def flow_stats(frames: np.ndarray, method: str = "raft_small", device: str | None = None, height: int = 360,
               iters: int = 12, batch: int | None = None, max_pairs: int | None = None,
               skip_held: bool = True) -> dict:
    """Bidirectional optical flow between consecutive frames, reduced to per-pair statistics.

    method     "raft_small" | "raft_large" (torchvision weights) | "farneback" (OpenCV, CPU, no weights)
    height     analysis height in px (width keeps the aspect; both rounded to multiples of 8)
    max_pairs  evaluate at most this many pairs, evenly spread over the clip (None = all) - for CPU runs
    skip_held  pairs that are exact repeats (held drawings) get zero flow without running the network

    Returns a dict of float arrays of length n_pairs (plus "pair_index", "held", "height", "width", "method"):
      mag_mean, mag_p95      forward-flow magnitude (px at the analysis size)
      moving_frac            share of pixels moving by more than 1 px
      global_dx, global_dy   median flow vector (camera pan estimate); local_mag = mean |flow - median|
      occ_frac               share of pixels failing the forward-backward consistency test (occluded / unmatched)
      warp_l1, warp_mse      masked error between frame t and frame t+1 warped back to t (pixel values in 0..1)
      raw_l1                 the same error without warping (for reference)
    """
    torch = _torch()
    import cv2

    T = len(frames)
    keys = ("mag_mean", "mag_p95", "moving_frac", "global_dx", "global_dy", "local_mag", "occ_frac",
            "warp_l1", "warp_mse", "raw_l1")
    H0, W0 = frames.shape[1:3]
    h = max(64, int(round(min(height, H0) / 8)) * 8)
    w = max(64, int(round(W0 * h / H0 / 8)) * 8)
    empty = {k: np.zeros(0, np.float32) for k in keys}
    empty.update(pair_index=np.zeros(0, np.int64), held=np.zeros(0, bool), height=h, width=w, method=method)
    if T < 2:
        return empty
    pairs = np.arange(T - 1)
    if max_pairs and len(pairs) > max_pairs:
        pairs = np.unique(np.linspace(0, T - 2, max_pairs).round().astype(np.int64))
    held_all = held_pairs(frames)
    dev = "cpu" if method == "farneback" else pick_device(device, need_gb=4.0 if method == "raft_small" else 6.0)
    if batch is None:
        batch = 8 if dev.startswith("cuda") else 2
    small = np.stack([cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA) for f in frames])
    out = {k: np.zeros(len(pairs), np.float32) for k in keys}

    def _stats(i_out: list[int], I1, I2, fw, bw):
        """I*: [B,3,h,w] in 0..1, fw/bw: [B,2,h,w]."""
        I2w, inb = _warp(I2, fw)
        bww, _ = _warp(bw, fw)
        fb = ((fw + bww) ** 2).sum(1)
        occ = fb > 0.01 * ((fw ** 2).sum(1) + (bww ** 2).sum(1)) + 0.5
        valid = (~occ) & inb
        vf = valid.float()
        nvalid = vf.sum((1, 2)).clamp(min=1.0)
        err = (I1 - I2w).abs().mean(1)
        sq = ((I1 - I2w) ** 2).mean(1)
        mag = (fw ** 2).sum(1).sqrt()
        med = fw.flatten(2).median(dim=2).values                                   # [B, 2]
        loc = ((fw - med[:, :, None, None]) ** 2).sum(1).sqrt()
        res = {
            "mag_mean": mag.mean((1, 2)), "mag_p95": torch.quantile(mag.flatten(1), 0.95, dim=1),
            "moving_frac": (mag > 1.0).float().mean((1, 2)), "global_dx": med[:, 0], "global_dy": med[:, 1],
            "local_mag": loc.mean((1, 2)), "occ_frac": 1.0 - vf.mean((1, 2)),
            "warp_l1": (err * vf).sum((1, 2)) / nvalid, "warp_mse": (sq * vf).sum((1, 2)) / nvalid,
            "raw_l1": (I1 - I2).abs().mean((1, 2, 3)),
        }
        for k, v in res.items():
            out[k][i_out] = v.float().cpu().numpy()

    todo = []
    for j, p in enumerate(pairs):
        if skip_held and held_all[p]:
            a = small[p].astype(np.float32) / 255.0
            b = small[p + 1].astype(np.float32) / 255.0
            d = np.abs(a - b)
            out["warp_l1"][j] = out["raw_l1"][j] = d.mean()
            out["warp_mse"][j] = ((a - b) ** 2).mean()
        else:
            todo.append(j)

    def _t(idx):  # uint8 [B,h,w,3] -> float [B,3,h,w] 0..1
        return torch.from_numpy(small[idx]).to(dev).permute(0, 3, 1, 2).float() / 255.0

    with torch.inference_mode():
        if method == "farneback":
            gray = [cv2.cvtColor(s, cv2.COLOR_RGB2GRAY) for s in small]
            for j in todo:
                p = int(pairs[j])
                fw = cv2.calcOpticalFlowFarneback(gray[p], gray[p + 1], None, 0.5, 4, 21, 3, 7, 1.5, 0)
                bw = cv2.calcOpticalFlowFarneback(gray[p + 1], gray[p], None, 0.5, 4, 21, 3, 7, 1.5, 0)
                _stats([j], _t([p]), _t([p + 1]), torch.from_numpy(fw).permute(2, 0, 1)[None],
                       torch.from_numpy(bw).permute(2, 0, 1)[None])
        elif method in ("raft_small", "raft_large"):
            net = _raft(method, dev)
            for s in range(0, len(todo), batch):
                js = todo[s:s + batch]
                ps = pairs[js]
                I1, I2 = _t(ps), _t(ps + 1)
                a, b = I1 * 2 - 1, I2 * 2 - 1
                flows = net(torch.cat([a, b]), torch.cat([b, a]), num_flow_updates=iters)[-1]
                _stats(js, I1, I2, flows[:len(js)], flows[len(js):])
        else:
            raise ValueError(f"unknown flow method {method!r}")
    out.update(pair_index=pairs, held=held_all[pairs], height=h, width=w, method=method, device=dev)
    return out


def warp_error(frames: np.ndarray, method: str = "raft_small", device: str | None = None, height: int = 360,
               max_pairs: int | None = None, flow: dict | None = None, return_details: bool = False):
    """Optical-flow warp error (temporal stability): frame t+1 is warped back onto frame t with the estimated
    flow and compared on non-occluded pixels (forward-backward consistency mask).

    Returns the mean masked L1 error on a 0..255 scale over all frame pairs that are not hard cuts
    (``return_details=True`` returns a dict with the mean, the Lai et al. E_warp (masked MSE x 1000), the mean over
    moving pairs only, percentiles, occlusion ratio, ...). ``flow`` can be a precomputed :func:`flow_stats` result.
    """
    fs = flow if flow is not None else flow_stats(frames, method=method, device=device, height=height,
                                                  max_pairs=max_pairs)
    n = len(fs["warp_l1"])
    if n == 0:
        det = {"warp_error": 0.0, "mean": 0.0, "warp_mse_x1000": 0.0, "warp_error_moving": 0.0, "warp_error_p95": 0.0,
               "warp_error_max": 0.0, "occlusion_ratio": 0.0, "residual_ratio": None, "n_pairs": 0, "n_cut_pairs": 0,
               "method": fs.get("method", method)}
        return det if return_details else 0.0
    cuts = set(detect_cuts(frames))
    is_cut = np.array([(int(p) + 1) in cuts for p in fs["pair_index"]])
    ok = ~is_cut
    if not ok.any():
        ok = np.ones(n, bool)
    l1 = fs["warp_l1"][ok] * 255.0
    moving = ok & ~fs["held"]
    raw = fs["raw_l1"][moving] * 255.0
    det = {
        "warp_error": float(l1.mean()),
        "warp_mse_x1000": float(fs["warp_mse"][ok].mean() * 1000.0),
        "warp_error_moving": float(fs["warp_l1"][moving].mean() * 255.0) if moving.any() else 0.0,
        "warp_error_p95": float(np.percentile(l1, 95)),
        "warp_error_max": float(l1.max()),
        "occlusion_ratio": float(fs["occ_frac"][ok].mean()),
        "occlusion_ratio_moving": float(fs["occ_frac"][moving].mean()) if moving.any() else 0.0,
        # share of the raw frame difference that motion compensation does NOT explain (1 = flow explains nothing)
        # (None when the frames barely change: the ratio of two tiny numbers means nothing)
        "residual_ratio": float(fs["warp_l1"][moving].sum() * 255.0 / raw.sum()) if moving.any() and raw.mean() >= 0.5 else None,
        "n_pairs": int(n), "n_cut_pairs": int(is_cut.sum()), "method": fs.get("method", method),
        "flow_height": int(fs["height"]),
    }
    det["mean"] = det["warp_error"]                 # alias
    return det if return_details else det["warp_error"]


def motion_stats(frames: np.ndarray, fps: float | None = None, method: str = "raft_small",
                 device: str | None = None, height: int = 360, max_pairs: int | None = None,
                 flow: dict | None = None) -> dict:
    """How much moves, and how: mean flow magnitude, static-frame ratio, camera vs local motion, hard cuts.

    Keys: mean_flow_mag (px per frame at ``flow_height``), mean_flow_mag_rel (% of the frame height per frame),
    mean_flow_mag_moving, flow_mag_p95, static_frame_ratio (pairs with mean flow < STATIC_FLOW_PX or exact repeats),
    moving_area_ratio (share of pixels moving > 1 px, on non-static pairs), global_motion_mag (median flow vector =
    pan/tilt estimate), local_motion_mag (what remains after removing it), hard_cuts (frame indices that start a
    new shot), n_hard_cuts, n_frames, [flow_mag_rel_per_sec if fps]. Aliases: mean_flow, static_ratio.
    """
    cuts = detect_cuts(frames)
    fs = flow if flow is not None else flow_stats(frames, method=method, device=device, height=height,
                                                  max_pairs=max_pairs)
    n = len(fs["mag_mean"])
    res: dict[str, Any] = {"n_frames": int(len(frames)), "hard_cuts": cuts, "n_hard_cuts": len(cuts),
                           "method": fs.get("method", method), "flow_height": int(fs["height"])}
    if n == 0:
        res.update(mean_flow_mag=0.0, mean_flow_mag_rel=0.0, mean_flow_mag_moving=0.0, flow_mag_p95=0.0,
                   static_frame_ratio=1.0, moving_area_ratio=0.0, global_motion_mag=0.0, local_motion_mag=0.0,
                   mean_flow=0.0, static_ratio=1.0)
        return res
    cutset = set(cuts)
    ok = np.array([(int(p) + 1) not in cutset for p in fs["pair_index"]])
    if not ok.any():
        ok = np.ones(n, bool)
    mag = fs["mag_mean"][ok]
    static = fs["held"][ok] | (mag < STATIC_FLOW_PX)
    mov = ~static
    gmag = np.sqrt(fs["global_dx"][ok] ** 2 + fs["global_dy"][ok] ** 2)
    res.update(
        mean_flow_mag=float(mag.mean()),
        mean_flow_mag_rel=float(mag.mean() / fs["height"] * 100.0),
        mean_flow_mag_moving=float(mag[mov].mean()) if mov.any() else 0.0,
        flow_mag_p95=float(np.percentile(fs["mag_p95"][ok], 95)),
        static_frame_ratio=float(static.mean()),
        moving_area_ratio=float(fs["moving_frac"][ok][mov].mean()) if mov.any() else 0.0,
        global_motion_mag=float(gmag.mean()),
        local_motion_mag=float(fs["local_mag"][ok].mean()),
    )
    res["mean_flow"], res["static_ratio"] = res["mean_flow_mag"], res["static_frame_ratio"]      # aliases
    if fps:
        res["flow_mag_rel_per_sec"] = float(res["mean_flow_mag_rel"] * fps)
    return res


# ----------------------------------------------------------------------------------------------------------------
# DOVER (no-reference video quality: technical + aesthetic)
# ----------------------------------------------------------------------------------------------------------------
_DOVER_SAMPLE = {
    "technical": dict(fragments_h=7, fragments_w=7, fsize_h=32, fsize_w=32, aligned=32, clip_len=32,
                      frame_interval=2, num_clips=3),
    "aesthetic": dict(size_h=224, size_w=224, clip_len=32, frame_interval=2, t_frag=32, num_clips=1),
}
_DOVER_MODEL_ARGS = dict(
    backbone={"technical": {"type": "swin_tiny_grpb", "checkpoint": True, "pretrained": None},
              "aesthetic": {"type": "conv_tiny"}},
    backbone_preserve_keys="technical,aesthetic", divide_head=True,
    vqa_head={"in_channels": 768, "hidden_channels": 64},
)


def _import_dover():
    """Import the DOVER code from the pinned clone without its heavy optional deps (decord, scikit-video)."""
    import importlib
    import importlib.machinery
    import types

    if not (DOVER_REPO / "dover" / "models" / "evaluator.py").exists():
        raise RuntimeError(
            f"DOVER code not found at {DOVER_REPO}. Run once: git clone https://github.com/VQAssessment/DOVER {DOVER_REPO}")
    if str(DOVER_REPO) not in sys.path:
        sys.path.insert(0, str(DOVER_REPO))
    for name in ("decord", "skvideo", "skvideo.io"):          # only used by DOVER's own file readers (we decode with PyAV)
        if name in sys.modules:
            continue
        try:
            importlib.import_module(name)
        except Exception:
            m = types.ModuleType(name)
            # a real spec, so later `importlib.util.find_spec(name)` calls (e.g. inside transformers) do not raise
            m.__spec__ = importlib.machinery.ModuleSpec(name, None)
            if name == "decord":
                m.VideoReader = m.cpu = m.gpu = None
                m.bridge = types.SimpleNamespace(set_bridge=lambda *_a, **_k: None)
            sys.modules[name] = m
            if name == "skvideo.io":
                sys.modules["skvideo"].io = m
    import warnings

    with warnings.catch_warnings(), contextlib.redirect_stdout(open(os.devnull, "w")):
        warnings.simplefilter("ignore")
        ds = importlib.import_module("dover.datasets.dover_datasets")
        models = importlib.import_module("dover.models")
    return ds, models


def _dover(device: str):
    key = ("dover", device)
    if key not in _MODELS:
        torch = _torch()
        ds, models = _import_dover()
        import warnings

        import importlib

        ev = importlib.import_module("dover.models.evaluator")
        orig = ev.convnext_3d_tiny
        # DOVER() hard-codes pretrained=True (an ImageNet ConvNeXt download that DOVER.pth overwrites anyway)
        ev.convnext_3d_tiny = lambda pretrained=False, **k: orig(pretrained=False, **k)
        try:
            with warnings.catch_warnings(), contextlib.redirect_stdout(open(os.devnull, "w")):
                warnings.simplefilter("ignore")
                net = models.DOVER(**_DOVER_MODEL_ARGS)
        finally:
            ev.convnext_3d_tiny = orig
        sd = torch.load(ensure_weight("dover"), map_location="cpu", weights_only=True)
        net.load_state_dict(sd)
        _MODELS[key] = (net.eval().to(device), ds)
    return _MODELS[key]


def dover_score(path: str | Path, device: str | None = None, seed: int = 0, python: str | None = None) -> dict:
    """DOVER no-reference quality scores of a clip (https://github.com/VQAssessment/DOVER, weights DOVER.pth).

    Returns {"technical", "aesthetic", "overall"} in 0..1, higher is better, plus the raw branch outputs and the
    z-scores: technical_z = (raw + 0.08285) / 0.03774, aesthetic_z = (raw - 0.1107) / 0.07355 (mean / std of each
    branch on LIVE-VQC), score = sigmoid(z), overall = sigmoid(0.3896 technical_z + 0.6104 aesthetic_z).

    Two deviations from running the repo scripts as they are today (both verified, see docs/eval_tools.md):
      * the aesthetic view is resized WITHOUT antialiasing, as torchvision did when DOVER was released; current
        torchvision antialiases by default, which shifts the aesthetic output by more than one std;
      * the repo's fuse_results() pairs the normalisation constants with the wrong branch (technical always ~0.06,
        aesthetic always ~0.95); here each branch uses the constants that match its own output distribution.
    DOVER was trained on natural user-generated video: use it to compare clips of the same kind (anime vs
    generated anime), not as an absolute quality scale. Sampling is seeded -> deterministic per clip.

    python: run in a subprocess with that interpreter instead (for callers whose own venv cannot import torch/timm);
            the env var KOMI_EVAL_PYTHON has the same effect.
    """
    python = python or os.environ.get("KOMI_EVAL_PYTHON")
    if python and os.path.realpath(python) != os.path.realpath(sys.executable):
        return run_in_subprocess([path], ["dover"], python=python, device=device)[str(path)]["dover"]
    torch = _torch()
    dev = pick_device(device, need_gb=4.0)
    net, ds = _dover(dev)
    T = count_frames(path)
    if T <= 0:
        raise ValueError(f"no frames in {path}")
    rs_np, rs_t = np.random.get_state(), torch.random.get_rng_state()
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        inds_by_type = {}
        for stype, sopt in _DOVER_SAMPLE.items():
            if "t_frag" not in sopt:
                sampler = ds.UnifiedFrameSampler(sopt["clip_len"], sopt["num_clips"], sopt["frame_interval"])
            else:
                sampler = ds.UnifiedFrameSampler(sopt["clip_len"] // sopt["t_frag"], sopt["t_frag"],
                                                 sopt["frame_interval"], sopt["num_clips"])
            inds_by_type[stype] = sampler(T, False)
        # native size (the technical branch crops raw 32x32 patches); only the sampled frames are kept in memory
        got = frames_at(path, np.unique(np.concatenate(list(inds_by_type.values()))))
        views = {}
        for stype, sopt in _DOVER_SAMPLE.items():
            inds = inds_by_type[stype]
            vid = torch.from_numpy(np.stack([got[int(i)] for i in inds])).permute(3, 0, 1, 2)   # [C, T, H, W] uint8
            if stype == "aesthetic":
                x = torch.nn.functional.interpolate(vid.permute(1, 0, 2, 3).float(), size=(sopt["size_h"], sopt["size_w"]),
                                                    mode="bilinear", align_corners=False, antialias=False)
                views[stype] = x.round().clamp(0, 255).permute(1, 0, 2, 3)
            else:
                views[stype] = ds.get_single_view(vid, stype, **sopt)
    finally:
        np.random.set_state(rs_np)
        torch.random.set_rng_state(rs_t)
    mean = torch.tensor([123.675, 116.28, 103.53])
    std = torch.tensor([58.395, 57.12, 57.375])
    raw = {}
    import warnings

    with torch.inference_mode(), warnings.catch_warnings():
        warnings.simplefilter("ignore")              # torch.utils.checkpoint chatter from the Swin backbone
        for stype, v in views.items():
            nclips = _DOVER_SAMPLE[stype].get("num_clips", 1)
            x = ((v.float().permute(1, 2, 3, 0) - mean) / std).permute(3, 0, 1, 2)
            x = x.reshape(x.shape[0], nclips, -1, *x.shape[2:]).transpose(0, 1).to(dev)
            raw[stype] = float(net({stype: x})[0].mean().item())
    t = (raw["technical"] + 0.08285) / 0.03774
    a = (raw["aesthetic"] - 0.1107) / 0.07355
    x = 0.3896 * t + 0.6104 * a
    sig = lambda z: float(1.0 / (1.0 + math.exp(-z)))  # noqa: E731
    return {"technical": sig(t), "aesthetic": sig(a), "overall": sig(x), "technical_z": t, "aesthetic_z": a,
            "technical_raw": raw["technical"], "aesthetic_raw": raw["aesthetic"], "n_frames": int(T), "device": dev}


# ----------------------------------------------------------------------------------------------------------------
# hard technical failures
# ----------------------------------------------------------------------------------------------------------------
def _decode_small_gray(path: str | Path, w: int = 320, h: int = 176):
    """Decode the whole first video stream at w x h gray through ffmpeg; yields frames, stores stderr/returncode."""
    err = tempfile.TemporaryFile()
    cmd = [ffmpeg_bin(), "-v", "error", "-nostdin", "-threads", str(min(4, n_threads())), "-i", str(path),
           "-map", "0:v:0", "-vf", f"scale={w}:{h}:flags=area,format=gray", "-fps_mode", "passthrough",
           "-f", "rawvideo", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err)
    state = {"proc": p}
    n = w * h

    def gen():
        try:
            while True:
                buf = p.stdout.read(n)
                if len(buf) < n:
                    break
                yield np.frombuffer(buf, np.uint8).reshape(h, w)
        finally:
            p.stdout.close()
            state["returncode"] = p.wait()
            err.seek(0)
            state["stderr"] = err.read().decode("utf-8", "replace")
            err.close()

    return gen(), state


def _audio_levels(path: str | Path, ignore_codec: str | None = None) -> dict:
    """Decode the first audio stream to 16 kHz mono and measure levels (dBFS). ``ignore_codec``: name of the video
    codec, whose messages (emitted while ffmpeg probes the file) are not audio errors."""
    sr = 16000
    r = subprocess.run([ffmpeg_bin(), "-v", "error", "-nostdin", "-i", str(path), "-vn", "-map", "0:a:0", "-ac", "1",
                        "-ar", str(sr), "-f", "s16le", "-"], capture_output=True)
    x = np.frombuffer(r.stdout[: len(r.stdout) // 2 * 2], np.int16).astype(np.float32) / 32768.0
    lines = [l for l in r.stderr.decode("utf-8", "replace").splitlines() if l.strip()
             and not (ignore_codec and l.startswith(f"[{ignore_codec} @"))]
    if r.returncode and not lines:
        lines = [f"ffmpeg exit code {r.returncode}"]
    out: dict[str, Any] = {"audio_decode_errors": "\n".join(lines)[:300]}
    if len(x) == 0:
        out.update(audio_samples=0, audio_duration=0.0, audio_rms_db=-120.0, audio_peak_db=-120.0, audio_silence_frac=1.0)
        return out
    rms = float(np.sqrt(np.mean(x ** 2)))
    peak = float(np.abs(x).max())
    win = sr // 10
    nwin = len(x) // win
    if nwin:
        wr = np.sqrt((x[: nwin * win].reshape(nwin, win) ** 2).mean(1))
        sil = float((20 * np.log10(wr + 1e-9) < -60).mean())
    else:
        sil = float(rms < 1e-3)
    out.update(audio_samples=int(len(x)), audio_duration=len(x) / sr,
               audio_rms_db=20 * math.log10(rms + 1e-9), audio_peak_db=20 * math.log10(peak + 1e-9),
               audio_silence_frac=sil)
    return out


def technical_check(path: str | Path, expected_dur: float | None = None, need_audio: bool = False, *,
                    dur_tol: float | None = None, expected_fps: float | None = None,
                    expected_size: Sequence[int] | None = None, allow_static: bool = False,
                    frozen_frac: float = 0.98, frozen_min_dur: float = 1.0, black_frac: float = 0.95,
                    silence_db: float = -60.0) -> dict:
    """Hard technical failures that fail a shot without a judge call.

    Checks: file missing / empty / unreadable, no video stream, decode errors (corrupt bitstream), no frames,
    container duration disagreeing with the decodable frames (truncated file), wrong duration vs ``expected_dur``
    (tolerance ``dur_tol``, default max(0.25 s, 5 %)), wrong fps / size when given, black or blank (uniform) video,
    frozen video (>= ``frozen_frac`` of the frame pairs identical, no cut, on a clip of >= ``frozen_min_dur`` s;
    pass ``allow_static=True`` for shots that are meant to be a still - a few percent of real anime shots are -
    and the finding becomes a warning), and - when ``need_audio`` - missing audio stream,
    silent audio (RMS < ``silence_db`` dBFS), undecodable audio, audio much shorter/longer than the video.

    Returns {"ok": bool, "reasons": ["code: detail", ...], "warnings": [...], "info": {...}}. Every reason starts
    with a stable code: missing_file, empty_file, unreadable, no_video_stream, decode_errors, no_frames,
    truncated_stream, wrong_duration, wrong_fps, wrong_size, black_video, blank_video, frozen_video,
    no_audio_stream, silent_audio, audio_decode_errors, audio_duration_mismatch.
    """
    reasons: list[str] = []
    warnings: list[str] = []
    info: dict[str, Any] = {}

    def done():
        return {"ok": not reasons, "reasons": reasons, "warnings": warnings, "info": info}

    p = Path(path)
    if not p.is_file():
        reasons.append(f"missing_file: {p}")
        return done()
    if p.stat().st_size == 0:
        reasons.append("empty_file: 0 bytes")
        return done()
    try:
        vi = video_info(p)
    except ProbeError as e:
        reasons.append(f"unreadable: {e}")
        return done()
    info.update({k: vi[k] for k in ("container_duration", "has_audio", "size_bytes") if k in vi})
    if not vi["has_video"]:
        reasons.append("no_video_stream: container has no video stream")
        return done()
    fps = vi.get("fps") or 0.0
    info.update(width=vi["width"], height=vi["height"], fps=round(fps, 4), video_codec=vi.get("video_codec"),
                pix_fmt=vi.get("pix_fmt"))

    # ---- full decode at thumbnail size: frame count, brightness, frame differences
    gen, st = _decode_small_gray(p)
    n = 0
    prev = None
    means, stds, diffs, blkmax = [], [], [], []
    for g in gen:
        gf = g.astype(np.float32)
        means.append(float(gf.mean()))
        stds.append(float(gf.std()))
        if prev is not None:
            d = np.abs(gf - prev)
            diffs.append(float(d.mean()))
            blkmax.append(float(d.reshape(22, 8, 40, 8).mean(axis=(1, 3)).max()))
        prev = gf
        n += 1
    info["frames_decoded"] = n
    err_txt = (st.get("stderr") or "").strip()
    if err_txt or st.get("returncode"):
        first = err_txt.splitlines()[0][:200] if err_txt else f"ffmpeg exit code {st.get('returncode')}"
        reasons.append(f"decode_errors: {first}")
        info["decode_error_lines"] = len(err_txt.splitlines())
    if n == 0:
        reasons.append("no_frames: no video frame could be decoded")
        return done()
    dur = n / fps if fps > 0 else vi.get("container_duration", 0.0)
    info["duration"] = round(dur, 4)
    cdur = vi.get("video_duration") or vi.get("container_duration") or 0.0
    if cdur > 0 and abs(cdur - dur) > max(0.5, 0.1 * cdur):
        reasons.append(f"truncated_stream: container says {cdur:.2f}s but only {dur:.2f}s decodes")
    if expected_dur is not None:
        tol = dur_tol if dur_tol is not None else max(0.25, 0.05 * expected_dur)
        if abs(dur - expected_dur) > tol:
            reasons.append(f"wrong_duration: {dur:.2f}s vs expected {expected_dur:.2f}s (tol {tol:.2f}s)")
    if expected_fps is not None and abs(fps - expected_fps) > 0.05:
        reasons.append(f"wrong_fps: {fps:.3f} vs expected {expected_fps:.3f}")
    if expected_size is not None and (vi["width"], vi["height"]) != tuple(expected_size):
        reasons.append(f"wrong_size: {vi['width']}x{vi['height']} vs expected {expected_size[0]}x{expected_size[1]}")

    means_a, stds_a = np.asarray(means), np.asarray(stds)
    blank = stds_a < 3.0
    black = blank & (means_a < 24.0)
    info.update(mean_luma=round(float(means_a.mean()), 2), black_frame_frac=round(float(black.mean()), 4),
                blank_frame_frac=round(float(blank.mean()), 4))
    if black.mean() >= black_frac:
        reasons.append(f"black_video: {black.mean() * 100:.0f}% of frames are black")
    elif blank.mean() >= black_frac:
        reasons.append(f"blank_video: {blank.mean() * 100:.0f}% of frames are a uniform colour (mean luma {means_a.mean():.0f})")
    elif black.mean() >= 0.5:
        warnings.append(f"mostly_black: {black.mean() * 100:.0f}% of frames are black")
    if n >= 2:
        static = (np.asarray(diffs) < HOLD_MAD) & (np.asarray(blkmax) < HOLD_BLOCK)
        runs = _runs(static)
        longest = (max(runs) if runs else 0)
        info.update(static_pair_frac=round(float(static.mean()), 4),
                    longest_static_run_s=round(longest / fps, 3) if fps > 0 else None)
        already = any(r.startswith(("black_video", "blank_video")) for r in reasons)
        has_cut = max(diffs) >= CUT_MAD            # two stills joined by a cut are not a frozen video
        if static.mean() >= frozen_frac and dur >= frozen_min_dur and not already and not has_cut:
            msg = f"frozen_video: {static.mean() * 100:.0f}% of frame pairs are identical over {dur:.2f}s"
            (warnings if allow_static else reasons).append(msg)
        elif fps > 0 and longest / fps >= max(2.0, 0.5 * dur) and not already:
            warnings.append(f"long_static_run: {longest / fps:.2f}s without any change")

    # ---- audio
    if vi["has_audio"]:
        al = _audio_levels(p, ignore_codec=vi.get("video_codec"))
        info.update({k: (round(v, 3) if isinstance(v, float) else v) for k, v in al.items() if k != "audio_decode_errors"})
        silent = al["audio_rms_db"] < silence_db
        if al["audio_decode_errors"]:
            (reasons if need_audio else warnings).append(f"audio_decode_errors: {al['audio_decode_errors'].splitlines()[0][:200]}")
        if silent:
            (reasons if need_audio else warnings).append(
                f"silent_audio: RMS {al['audio_rms_db']:.1f} dBFS, peak {al['audio_peak_db']:.1f} dBFS")
        if al["audio_samples"] and abs(al["audio_duration"] - dur) > max(0.5, 0.1 * dur):
            (reasons if need_audio else warnings).append(
                f"audio_duration_mismatch: audio {al['audio_duration']:.2f}s vs video {dur:.2f}s")
    elif need_audio:
        reasons.append("no_audio_stream: audio is required but the file has no audio stream")
    return done()


# ----------------------------------------------------------------------------------------------------------------
# one-call evidence bundle + CLI
# ----------------------------------------------------------------------------------------------------------------
ALL_METRICS = ("technical", "flicker", "motion", "warp", "dover")


def analyze_clip(path: str | Path, metrics: Sequence[str] = ALL_METRICS, *, expected_dur: float | None = None,
                 need_audio: bool = False, device: str | None = None, flow: str = "raft_small",
                 flow_height: int = 360, fps: float | None = None, height: int | None = None,
                 max_frames: int | None = None, max_pairs: int | None = None, allow_static: bool = False) -> dict:
    """Every requested metric for one clip (frames decoded once, optical flow computed once).

    Returns {"path", "technical": {...}, "flicker": {...}, "motion": {...}, "warp": {...}, "dover": {...},
    "runtime_s": {...}}. If the technical check finds the file unreadable the other metrics are skipped.
    A failing metric yields {"error": "..."} for that key instead of raising.
    """
    res: dict[str, Any] = {"path": str(path)}
    rt: dict[str, float] = {}
    metrics = [m.strip() for m in metrics if m.strip()]
    unknown = set(metrics) - set(ALL_METRICS)
    if unknown:
        raise ValueError(f"unknown metrics {sorted(unknown)}; choose from {ALL_METRICS}")
    if "technical" in metrics:
        t = time.time()
        res["technical"] = technical_check(path, expected_dur=expected_dur, need_audio=need_audio,
                                           allow_static=allow_static)
        rt["technical"] = time.time() - t
        fatal = {"missing_file", "empty_file", "unreadable", "no_video_stream", "no_frames"}
        if any(r.split(":")[0] in fatal for r in res["technical"]["reasons"]):
            res["runtime_s"] = {k: round(v, 3) for k, v in rt.items()}
            return res
    frames = None
    if any(m in metrics for m in ("flicker", "motion", "warp")):
        t = time.time()
        try:
            frames, clip_fps = load_frames(path, fps=fps, size=height, max_frames=max_frames, return_fps=True)
        except Exception as e:  # corrupt bitstreams can still raise here
            for m in ("flicker", "motion", "warp"):
                if m in metrics:
                    res[m] = {"error": f"{type(e).__name__}: {e}"[:300]}
        rt["decode"] = time.time() - t
    if frames is not None:
        res["n_frames"], res["fps"] = int(len(frames)), round(clip_fps, 4)
        if "flicker" in metrics:
            t = time.time()
            try:
                res["flicker"] = flicker_score(frames, fps=clip_fps)
            except Exception as e:
                res["flicker"] = {"error": f"{type(e).__name__}: {e}"[:300]}
            rt["flicker"] = time.time() - t
        if "motion" in metrics or "warp" in metrics:
            t = time.time()
            try:
                fs = flow_stats(frames, method=flow, device=device, height=flow_height, max_pairs=max_pairs)
                if "motion" in metrics:
                    res["motion"] = motion_stats(frames, fps=clip_fps, flow=fs)
                if "warp" in metrics:
                    res["warp"] = warp_error(frames, flow=fs, return_details=True)
            except Exception as e:
                for m in ("motion", "warp"):
                    if m in metrics:
                        res[m] = {"error": f"{type(e).__name__}: {e}"[:300]}
            rt["flow"] = time.time() - t
    if "dover" in metrics:
        t = time.time()
        try:
            res["dover"] = dover_score(path, device=device)
        except Exception as e:
            res["dover"] = {"error": f"{type(e).__name__}: {e}"[:300]}
        rt["dover"] = time.time() - t
    res["runtime_s"] = {k: round(v, 3) for k, v in rt.items()}
    return res


def run_in_subprocess(paths: Sequence[str | Path], metrics: Sequence[str] = ALL_METRICS, python: str | None = None,
                      device: str | None = None, extra_args: Sequence[str] = ()) -> dict:
    """Run this module's CLI with another interpreter (e.g. venvs/eval) and return {path: result}."""
    python = python or os.environ.get("KOMI_EVAL_PYTHON") or sys.executable
    repo = str(Path(__file__).resolve().parents[2])
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR") or None) as td:
        out = Path(td) / "metrics.json"
        cmd = [python, "-m", "src.eval.video_metrics", *map(str, paths), "--metrics", ",".join(metrics),
               "--out", str(out), "--quiet", *extra_args]
        if device:
            cmd += ["--device", device]
        env = dict(os.environ)
        env["PYTHONPATH"] = repo + os.pathsep + env.get("PYTHONPATH", "")
        env.pop("KOMI_EVAL_PYTHON", None)
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=repo, env=env)
        if r.returncode != 0 or not out.exists():
            raise RuntimeError(f"video_metrics subprocess failed ({r.returncode}): {r.stderr[-800:]}")
        return json.loads(out.read_text())["clips"]


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    return o


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Video evidence metrics -> JSON (see module docstring).")
    ap.add_argument("inputs", nargs="+", help="clips, directories (recursive), .txt lists, or .jsonl files with a "
                    "'path' per row (optional per-row expected_dur / need_audio / allow_static override the flags)")
    ap.add_argument("--out", help="write JSON here (default: print to stdout)")
    ap.add_argument("--metrics", default=",".join(ALL_METRICS), help=f"comma list from {ALL_METRICS}")
    ap.add_argument("--device", default="auto", help="auto (cuda only if enough VRAM is free) | cuda | cpu")
    ap.add_argument("--flow", default="raft_small", choices=["raft_small", "raft_large", "farneback"])
    ap.add_argument("--flow-height", type=int, default=360)
    ap.add_argument("--max-pairs", type=int, default=None, help="cap the number of flow pairs per clip (CPU runs)")
    ap.add_argument("--fps", type=float, default=None, help="resample clips to this fps before flicker/flow")
    ap.add_argument("--height", type=int, default=None, help="resize clips to this height before flicker/flow")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--expected-dur", type=float, default=None)
    ap.add_argument("--need-audio", action="store_true")
    ap.add_argument("--allow-static", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)

    os.environ.setdefault("KOMI_EVAL_THREADS", str(n_threads()))
    clips = list_clip_specs(a.inputs)
    metrics = [m for m in a.metrics.split(",") if m]
    t0 = time.time()
    results: dict[str, Any] = {}
    for i, (c, spec) in enumerate(clips):
        try:
            r = analyze_clip(c, metrics, expected_dur=spec.get("expected_dur", a.expected_dur),
                             need_audio=bool(spec.get("need_audio", a.need_audio)), device=a.device,
                             flow=a.flow, flow_height=a.flow_height, fps=a.fps, height=a.height,
                             max_frames=a.max_frames, max_pairs=a.max_pairs,
                             allow_static=bool(spec.get("allow_static", a.allow_static)))
        except Exception as e:  # never let one clip kill a batch
            r = {"path": str(c), "error": f"{type(e).__name__}: {e}"[:300]}
        results[str(c)] = r
        if not a.quiet:
            tc = r.get("technical", {})
            print(f"[{i + 1}/{len(clips)}] {Path(c).name}: ok={tc.get('ok', '-')} "
                  f"warp={r.get('warp', {}).get('warp_error', '-')} "
                  f"dover={r.get('dover', {}).get('overall', '-')} t={sum(r.get('runtime_s', {}).values()):.1f}s",
                  file=sys.stderr, flush=True)
    payload = _jsonable({
        "clips": results,
        "config": {"metrics": metrics, "device": a.device, "flow": a.flow, "flow_height": a.flow_height,
                   "max_pairs": a.max_pairs, "fps": a.fps, "height": a.height, "max_frames": a.max_frames},
        "runtime_s": round(time.time() - t0, 2),
    })
    txt = json.dumps(payload, indent=1, ensure_ascii=False)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(txt)
    else:
        print(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
