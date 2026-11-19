"""Frame stage against the real episode: how far is the set of generated first frames from the set of real frames?

  python -m src.eval.frame_style --runs ep22_v2,ep22_v3,ep22_v3b,ep22_v3c,ep22_v3d --episode 22

FVD needs rendered clips; this needs only the first frames, so a new way of drawing can be measured before three hours of
rendering. Features: DINOv2-base (the model that aligned panels to shots), CLS token, L2-normalised. Distance: kernel
inception distance (unbiased MMD^2 with the cubic polynomial kernel, x1000; lower = closer), which stays honest on a few
hundred pictures where a Frechet distance does not. Reference set: frames of the real held-out episode (evaluation only).
For scale: frames of the real TRAINING episodes against the same reference (what "another episode of the show" scores).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.common.config import P
from src.common.io import log, read_json, read_jsonl, write_json

MODEL_ID = "facebook/dinov2-base"


def features(paths: list[Path], device: str = "cpu", batch: int = 16) -> np.ndarray:
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel

    cache = P("processed") / "eval" / "frame_style_feats"
    cache.mkdir(parents=True, exist_ok=True)
    proc = model = None
    out = []
    todo = []
    for p in paths:
        st = p.stat()
        f = cache / f"{abs(hash((str(p), st.st_size, int(st.st_mtime)))):x}.npy"
        todo.append((p, f))
    missing = [(p, f) for p, f in todo if not f.exists()]
    for i in range(0, len(missing), batch):
        if model is None:
            proc = AutoImageProcessor.from_pretrained(MODEL_ID)
            model = AutoModel.from_pretrained(MODEL_ID).to(device).eval()
        part = missing[i : i + batch]
        with torch.no_grad():
            x = proc(images=[Image.open(p).convert("RGB") for p, _ in part], return_tensors="pt").to(device)
            z = model(**x).last_hidden_state[:, 0]
            z = torch.nn.functional.normalize(z, dim=-1).cpu().numpy()
        for (_, f), v in zip(part, z):
            np.save(f, v)
    for _, f in todo:
        out.append(np.load(f))
    return np.stack(out) if out else np.zeros((0, 768), dtype=np.float32)


def kid(a: np.ndarray, b: np.ndarray, subsets: int = 50, size: int = 150, seed: int = 0) -> tuple[float, float]:
    """Unbiased MMD^2, polynomial kernel (x.y/d + 1)^3, mean and std over random subsets; x1000."""
    rng = np.random.default_rng(seed)
    d = a.shape[1]
    m = min(size, len(a), len(b))
    vals = []
    for _ in range(subsets):
        x, y = a[rng.choice(len(a), m, replace=False)], b[rng.choice(len(b), m, replace=False)]
        kxx, kyy, kxy = (x @ x.T / d + 1) ** 3, (y @ y.T / d + 1) ** 3, (x @ y.T / d + 1) ** 3
        vals.append((kxx.sum() - np.trace(kxx)) / (m * (m - 1)) + (kyy.sum() - np.trace(kyy)) / (m * (m - 1)) - 2 * kxy.mean())
    return float(np.mean(vals) * 1000), float(np.std(vals) * 1000)


def look_stats(paths: list[Path]) -> dict:
    """Plain numbers for the look, per set (means over frames at 384x216): brightness and contrast (L of Lab, 0-100), colourfulness
    (chroma of Lab), edge density (share of pixels with a strong gradient - how much fine detail is drawn) and sharpness
    (variance of the Laplacian). They name what a distance only counts."""
    import cv2

    rows = []
    for p in paths:
        im = cv2.imread(str(p))
        if im is None:
            continue
        im = cv2.resize(im, (384, 216), interpolation=cv2.INTER_AREA)
        lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB).astype(np.float32)
        L = lab[..., 0] * 100 / 255
        chroma = np.hypot(lab[..., 1] - 128, lab[..., 2] - 128)
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)
        mag = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))
        rows.append((L.mean(), L.std(), chroma.mean(), float((mag > 60).mean()), float(cv2.Laplacian(g, cv2.CV_32F).var())))
    a = np.array(rows)
    return {"frames": len(rows), "brightness": round(float(a[:, 0].mean()), 1), "contrast": round(float(a[:, 1].mean()), 1), "colourfulness": round(float(a[:, 2].mean()), 1),
            "edge_density": round(float(a[:, 3].mean()), 3), "sharpness": round(float(np.median(a[:, 4])), 0)}


def real_frames(episode: int) -> list[Path]:
    rows = [r for r in read_jsonl(P("processed") / "anime" / "shots.jsonl") if int(r["episode"]) == episode and r.get("content") and len(r.get("keyframes") or []) >= 2]
    return [Path(r["keyframes"][1]) for r in rows if Path(r["keyframes"][1]).exists()]


def run_frames(run: str) -> list[Path]:
    rd = P("runs") / run
    shots = (read_json(rd / "shotlist.json") or {}).get("shots", [])
    return [rd / "shots" / s["shot_id"] / "kf.png" for s in shots if s.get("start", "panel") in ("panel", "new") and "TITLE_CARD" not in str(s.get("notes") or "")
            and (rd / "shots" / s["shot_id"] / "kf.png").exists()]


def evaluate(runs: list[str], episode: int, device: str = "cpu") -> dict:
    ref = features(real_frames(episode), device)
    train = [Path(b["path"]) for b in (read_json(P("processed") / "v3" / "style_bank.json") or []) if Path(b["path"]).exists()]
    out = {"episode": episode, "model": MODEL_ID, "reference_frames": len(ref), "sets": {}}
    k, sd = kid(features(train, device), ref)
    out["sets"]["real training episodes (floor)"] = {"frames": len(train), "kid_x1000": round(k, 2), "std": round(sd, 2), "look": look_stats(train)}
    out["reference_look"] = look_stats(real_frames(episode))
    log(f"look, real episode {episode}: {out['reference_look']}")
    log(f"look, real training frames: {out['sets']['real training episodes (floor)']['look']}")
    log(f"frame style: real training frames vs real episode {episode}: KID {k:.2f} +- {sd:.2f} ({len(train)} vs {len(ref)} frames)")
    for run in runs:
        fr = run_frames(run)
        if len(fr) < 30:
            log(f"frame style: {run}: only {len(fr)} frames, skipped")
            continue
        k, sd = kid(features(fr, device), ref)
        out["sets"][run] = {"frames": len(fr), "kid_x1000": round(k, 2), "std": round(sd, 2), "look": look_stats(fr)}
        log(f"look, {run}: {out['sets'][run]['look']}")
        log(f"frame style: {run} vs real episode {episode}: KID {k:.2f} +- {sd:.2f} ({len(fr)} frames)")
    (P("runs") / "eval").mkdir(parents=True, exist_ok=True)
    write_json(P("runs") / "eval" / f"frame_style_ep{episode}.json", out)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--episode", type=int, default=22)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    evaluate(a.runs.split(","), a.episode, a.device)
