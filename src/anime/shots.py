"""Shot boundary detection (TransNetV2, fallback: histogram/SSIM cut detector) + keyframe extraction.

Outputs:
  data/interim/anime/epNN/shots_raw.json   [{shot, start, end, f0, f1}]  (times on the 23.976 fps proxy)
  data/interim/anime/epNN/keyframes/sXXXX_{a,m,b}.jpg  (first+2, middle, last-2 frame of every shot, 854x480)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from src.common.config import P
from src.common.io import log, read_json, write_json

FPS = 24000 / 1001


def _read_small(path: str, w: int = 48, h: int = 27) -> np.ndarray:
    """Decode the whole proxy at TransNetV2 input size (48x27 RGB). ~35k frames -> ~140 MB."""
    import subprocess
    from src.common.config import CFG

    cmd = [CFG.paths.ffmpeg, "-loglevel", "error", "-i", path, "-vf", f"scale={w}:{h}", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)


def detect_transnet(frames: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Per-frame cut probability with TransNetV2 (pytorch port). Returns bool array of 'last frame of a shot'."""
    import torch
    from transnetv2_pytorch import TransNetV2

    model = TransNetV2()
    # the pip package ships weights; API differs slightly across versions -> handle both
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(dev).eval()
    n = len(frames)
    pad_start = np.repeat(frames[:1], 25, 0)
    pad_end = np.repeat(frames[-1:], 25 + (50 - (n % 50 if n % 50 else 50)), 0)
    padded = np.concatenate([pad_start, frames, pad_end], 0)
    preds = []
    with torch.no_grad():
        for i in range(0, len(padded) - 50, 50):
            window = torch.from_numpy(padded[i : i + 100][None]).to(dev)
            out = model(window)
            single = out[0] if isinstance(out, (tuple, list)) else out
            p = torch.sigmoid(single)[0, 25:75, 0].float().cpu().numpy()
            preds.append(p)
    prob = np.concatenate(preds)[:n]
    return prob


def detect_fallback(frames: np.ndarray) -> np.ndarray:
    """Cheap cut score: colour-histogram distance + mean abs diff, robustly normalised. Returns pseudo-probability."""
    f = frames.astype(np.float32)
    mad = np.abs(f[1:] - f[:-1]).mean((1, 2, 3)) / 255.0
    hists = np.stack([np.concatenate([np.histogram(fr[..., c], bins=16, range=(0, 255))[0] for c in range(3)]) for fr in frames]).astype(np.float32)
    hists /= hists.sum(1, keepdims=True)
    hd = 0.5 * np.abs(hists[1:] - hists[:-1]).sum(1)
    score = 0.5 * np.clip(mad / 0.25, 0, 1) + 0.5 * np.clip(hd / 0.6, 0, 1)
    # local-contrast: a cut is a spike relative to its neighbourhood
    k = 6
    pad = np.pad(score, k, mode="edge")
    local = np.array([np.median(np.r_[pad[i : i + k], pad[i + k + 1 : i + 2 * k + 1]]) for i in range(len(score))])
    prob = np.clip((score - local) * 2.5, 0, 1)
    return np.r_[prob, 0.0]


def boundaries(prob: np.ndarray, threshold: float = 0.5, min_len: int = 8) -> list[tuple[int, int]]:
    cuts = np.where(prob > threshold)[0]
    shots, start = [], 0
    for c in cuts:
        if c + 1 - start >= min_len:
            shots.append((start, int(c)))
            start = int(c) + 1
    if len(prob) - start >= 1:
        shots.append((start, len(prob) - 1))
    return shots


def extract_keyframes(proxy: Path, shots: list[tuple[int, int]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted: dict[int, list[Path]] = {}
    for i, (a, b) in enumerate(shots):
        n = b - a + 1
        picks = {"a": a + min(2, n // 4), "m": a + n // 2, "b": b - min(2, n // 4)}
        for tag, fr in picks.items():
            f = out_dir / f"s{i:04d}_{tag}.jpg"
            if not f.exists():
                wanted.setdefault(fr, []).append(f)
    if not wanted:
        return
    cap = cv2.VideoCapture(str(proxy))
    idx = 0
    last = max(wanted)
    while idx <= last:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in wanted:
            for f in wanted[idx]:
                cv2.imwrite(str(f), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        idx += 1
    cap.release()


def run_episode(ep: int, threshold: float = 0.5) -> list[dict]:
    d = P("interim") / "anime" / f"ep{ep:02d}"
    out = d / "shots_raw.json"
    proxy = d / "proxy.mp4"
    if out.exists():
        shots = read_json(out)
    else:
        frames = _read_small(str(proxy))
        try:
            prob = detect_transnet(frames, threshold)
            method = "transnetv2"
        except Exception as e:
            log(f"TransNetV2 unavailable ({str(e)[:120]}); using fallback detector")
            prob = detect_fallback(frames)
            method = "fallback"
        np.save(d / "cut_prob.npy", prob.astype(np.float16))
        b = boundaries(prob, threshold)
        shots = [{"shot": i, "f0": a, "f1": e, "start": round(a / FPS, 3), "end": round((e + 1) / FPS, 3), "method": method}
                 for i, (a, e) in enumerate(b)]
        write_json(out, shots)
    extract_keyframes(proxy, [(s["f0"], s["f1"]) for s in shots], d / "keyframes")
    return shots


def run_fallback(ep: int, threshold: float = 0.45) -> list[dict]:
    """CPU-only provisional shots (shots_fb.json) used while the GPU is busy; TransNetV2 output supersedes it."""
    d = P("interim") / "anime" / f"ep{ep:02d}"
    out = d / "shots_fb.json"
    if out.exists():
        return read_json(out)
    prob = detect_fallback(_read_small(str(d / "proxy.mp4")))
    shots = [{"shot": i, "f0": a, "f1": e, "start": round(a / FPS, 3), "end": round((e + 1) / FPS, 3), "method": "fallback"}
             for i, (a, e) in enumerate(boundaries(prob, threshold))]
    write_json(out, shots)
    return shots


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", default="all")
    ap.add_argument("--fallback-only", action="store_true")
    a = ap.parse_args()
    eps = list(range(1, 25)) if a.episodes == "all" else [int(x) for x in a.episodes.split(",")]
    if a.fallback_only:
        for ep in eps:
            s = run_fallback(ep)
            log(f"ep{ep:02d}: {len(s)} provisional shots (fallback detector)")
        raise SystemExit(0)
    for ep in eps:
        if not (P("interim") / "anime" / f"ep{ep:02d}" / "proxy.mp4").exists():
            log(f"ep{ep:02d}: proxy missing, skipping")
            continue
        s = run_episode(ep)
        durs = np.array([x["end"] - x["start"] for x in s])
        log(f"ep{ep:02d}: {len(s)} shots, median {np.median(durs):.2f}s, mean {durs.mean():.2f}s ({s[0].get('method')})")
