"""Wan 2.2 I2V LoRA dataset builder (brief §6): ~1,000–1,500 single-shot clips, 81 frames @16 fps, 832x480, + captions.

TRAIN episodes only (1–21). Excludes OP/ED/previews (audio fingerprint), title cards, and clips with an internal cut.

  python -m src.train.build_wan_dataset select   [--n 1200]     # choose shots (CPU)
  python -m src.train.build_wan_dataset cut                     # cut clips + caption frames from the 1080p source (CPU)
  python -m src.train.build_wan_dataset caption                 # Qwen captions (needs the local LLM server)
  python -m src.train.build_wan_dataset finalize                # write <id>.txt next to <id>.mp4 + manifest
"""
from __future__ import annotations

import argparse
import random
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from src.anime.oped import is_content, non_content
from src.common.config import CFG, P
from src.common.io import ffmpeg, log, read_json, read_jsonl, write_json, write_jsonl

TRIGGER = "komi_anime_style"
FPS = int(CFG.anime.train_fps)
NFRAMES = int(CFG.anime.train_frames)
CLIP_SEC = NFRAMES / FPS
OUT = P("processed") / "train" / "wan"


def shots_for(ep: int) -> list[dict]:
    d = P("interim") / "anime" / f"ep{ep:02d}"
    return read_json(d / "shots_raw.json") or read_json(d / "shots_fb.json") or []


def motion_score(proxy: str, start: float, n: int = 6) -> tuple[float, float]:
    """(mean abs diff between sampled frames, max consecutive-frame diff) on the proxy — cheap static / cut check."""
    cap = cv2.VideoCapture(proxy)
    frames = []
    for i in range(n):
        cap.set(cv2.CAP_PROP_POS_MSEC, (start + (i + 0.5) * CLIP_SEC / n) * 1000)
        ok, f = cap.read()
        if ok:
            frames.append(cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (160, 90)).astype(np.float32))
    cap.release()
    if len(frames) < 2:
        return 0.0, 0.0
    d = [float(np.abs(a - b).mean()) for a, b in zip(frames[:-1], frames[1:])]
    return float(np.mean(d)), float(np.max(d))


def select(n_target: int, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    cands = []
    for ep in CFG.split.train_episodes:
        shots = shots_for(ep)
        if not shots:
            log(f"ep{ep:02d}: no shots yet")
            continue
        nc = non_content(ep)
        subs = read_json(P("processed") / "anime" / "subs" / f"ep{ep:02d}.json", [])
        titles = [(e["start"] - 0.5, e["end"] + 0.5) for e in subs if e["kind"] == "title"]
        signs = [(e["start"], e["end"]) for e in subs if e["kind"] in ("sign", "notebook")]   # typeset on-screen text
        proxy = str(P("interim") / "anime" / f"ep{ep:02d}" / "proxy.mp4")
        for s in shots:
            dur = s["end"] - s["start"]
            if dur < CLIP_SEC + 0.25:
                continue
            start = s["start"] + 0.10          # skip the cut frame
            end = start + CLIP_SEC
            if not is_content(ep, start, end, nc) or any(min(end, b) - max(start, a) > 0 for a, b in titles):
                continue
            if s["start"] < 3.0:
                continue
            # on-screen writing teaches the video model to draw glyphs (-> garbled text): skip clips where it shows > 1 s
            if sum(max(0.0, min(end, b) - max(start, a)) for a, b in signs) > 1.0:
                continue
            m, mx = motion_score(proxy, start)
            cands.append({"id": f"ep{ep:02d}_s{s['shot']:04d}", "ep": ep, "shot": s["shot"], "start": round(start, 3),
                          "shot_dur": round(dur, 2), "motion": round(m, 3), "max_diff": round(mx, 3)})
        log(f"ep{ep:02d}: {sum(1 for c in cands if c['ep'] == ep)} candidate shots >= {CLIP_SEC:.2f}s")
    # drop probable internal cuts (very large jump between sampled frames) — verified again after cutting
    cands = [c for c in cands if c["max_diff"] < 60]
    static = [c for c in cands if c["motion"] < 1.0]
    moving = [c for c in cands if c["motion"] >= 1.0]
    rng.shuffle(static)
    rng.shuffle(moving)
    n_static = min(len(static), int(n_target * 0.2))
    chosen = moving[: n_target - n_static] + static[:n_static]
    # balance across episodes (cap per episode)
    cap = max(20, int(1.6 * n_target / max(1, len(CFG.split.train_episodes))))
    out, per = [], {}
    for c in chosen:
        if per.get(c["ep"], 0) < cap:
            out.append(c)
            per[c["ep"]] = per.get(c["ep"], 0) + 1
    out.sort(key=lambda c: c["id"])
    write_jsonl(OUT / "selected.jsonl", out)
    log(f"selected {len(out)} clips from {len(cands)} candidates ({len(moving)} moving / {len(static)} static)")
    return out


def _cut_one(c: dict) -> dict:
    clip = OUT / "clips" / f"{c['id']}.mp4"
    fdir = OUT / "frames" / c["id"]
    src = next((P("raw") / "anime").glob(f"ep{c['ep']:02d}_*.mkv"))
    if not clip.exists():
        clip.parent.mkdir(parents=True, exist_ok=True)
        tmp = clip.with_suffix(".tmp.mp4")
        ffmpeg("-ss", f"{c['start']:.3f}", "-i", str(src), "-t", f"{CLIP_SEC + 0.2:.3f}", "-map", "0:v:0",
               "-vf", f"fps={FPS},scale=-2:480:flags=lanczos,crop=832:480,format=yuv420p", "-frames:v", str(NFRAMES),
               "-c:v", "libx264", "-preset", "slow", "-crf", "14", "-an", str(tmp))
        tmp.rename(clip)
    cap = cv2.VideoCapture(str(clip))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames, prev, max_jump = [], None, 0.0
    i = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        g = cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (160, 90)).astype(np.float32)
        if prev is not None:
            max_jump = max(max_jump, float(np.abs(g - prev).mean()))
        prev = g
        if i in (0, NFRAMES // 2, NFRAMES - 1):
            frames.append(f)
        i += 1
    cap.release()
    fdir.mkdir(parents=True, exist_ok=True)
    for tag, f in zip("amb", frames):
        cv2.imwrite(str(fdir / f"{tag}.jpg"), f, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return {**c, "n_frames": i, "max_jump": round(max_jump, 2), "ok": i == NFRAMES and max_jump < 45}


def cut(workers: int = 6) -> list[dict]:
    sel = read_jsonl(OUT / "selected.jsonl")
    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(_cut_one, sel))
    bad = [r for r in res if not r["ok"]]
    for r in bad:  # internal cut or wrong length: drop the files
        (OUT / "clips" / f"{r['id']}.mp4").unlink(missing_ok=True)
    good = [r for r in res if r["ok"]]
    write_jsonl(OUT / "cut.jsonl", good)
    log(f"cut {len(good)} clips ({len(bad)} rejected: internal cut / wrong length)")
    return good


def purge(cache_dir: str | None = None) -> int:
    """Re-check every cut clip against TransNetV2 shot boundaries (clips were selected with the CPU fallback detector)
    and drop the ones that contain a real cut: clip, caption frames, caption text and any musubi cache files."""
    rows = read_jsonl(OUT / "cut.jsonl")
    cache = Path(cache_dir) if cache_dir else OUT / "cache" / "wan_832x480_f81"
    keep, bad = [], []
    shots_by_ep: dict[int, list[dict]] = {}
    for r in rows:
        ep = r["ep"]
        if ep not in shots_by_ep:
            shots_by_ep[ep] = read_json(P("interim") / "anime" / f"ep{ep:02d}" / "shots_raw.json") or []
        cuts = [s["end"] for s in shots_by_ep[ep][:-1]]
        if not shots_by_ep[ep]:
            keep.append(r)          # TransNetV2 not available for this episode: keep (frame-diff check already passed)
            continue
        inside = [c for c in cuts if r["start"] + 0.25 < c < r["start"] + CLIP_SEC - 0.25]
        (bad if inside else keep).append(r)
    for r in bad:
        for f in [OUT / "clips" / f"{r['id']}.mp4", OUT / "clips" / f"{r['id']}.txt"]:
            f.unlink(missing_ok=True)
        if cache.exists():
            for f in cache.glob(f"{r['id']}_*"):
                f.unlink(missing_ok=True)
    write_jsonl(OUT / "cut.jsonl", keep)
    write_jsonl(OUT / "purged.jsonl", bad)
    log(f"purge: removed {len(bad)} clips with an internal TransNetV2 cut; {len(keep)} clips remain")
    return len(bad)


def caption(include_extra: bool = False) -> None:
    """Captions for the clips in clips/ (and, with include_extra, for clips parked in clips_extra/)."""
    from src.anime.caption import caption_many

    rows = read_jsonl(OUT / "cut.jsonl")
    done = {r["id"]: r for r in read_jsonl(OUT / "captions.jsonl")}
    todo = [r for r in rows if r["id"] not in done and (include_extra or (OUT / "clips" / f"{r['id']}.mp4").exists())]
    subs = {}
    B = 32
    for i in range(0, len(todo), B):
        chunk = todo[i : i + B]
        items = []
        for r in chunk:
            if r["ep"] not in subs:
                subs[r["ep"]] = read_json(P("processed") / "anime" / "subs" / f"ep{r['ep']:02d}.json", [])
            line = " / ".join(e["text"] for e in subs[r["ep"]] if e["kind"] == "dialogue" and e["start"] < r["start"] + CLIP_SEC and e["end"] > r["start"])
            fd = OUT / "frames" / r["id"]
            items.append({"frames": [fd / "a.jpg", fd / "m.jpg", fd / "b.jpg"], "subtitle": line})
        caps = caption_many(items)
        for r, c in zip(chunk, caps):
            if c and "error" not in c:
                done[r["id"]] = {**r, **c}
        write_jsonl(OUT / "captions.jsonl", list(done.values()))
        log(f"captioned {min(i + B, len(todo))}/{len(todo)}")


def make_caption_text(c: dict) -> str:
    chars = [x for x in c.get("characters", []) if not x.lower().startswith("unknown")]
    cam = c.get("camera", "static")
    parts = [TRIGGER + ", anime", f"{c.get('shot_size', 'medium')} shot"]
    if chars:
        parts.append(", ".join(chars))
    body = c["caption"].strip().rstrip(".")
    motion = c.get("action", "").strip().rstrip(".")
    txt = ", ".join(parts) + ". " + body + "."
    if motion and motion.lower() not in body.lower():
        txt += " " + motion + "."
    txt += f" Camera: {cam}."
    return txt


def finalize(max_text_fraction: float = 1.0) -> None:
    caps = read_jsonl(OUT / "captions.jsonl")
    keep = []
    for c in caps:
        clip = OUT / "clips" / f"{c['id']}.mp4"
        if not clip.exists() or c.get("is_title_card"):
            (OUT / "clips" / f"{c['id']}.txt").unlink(missing_ok=True)
            continue
        text = make_caption_text(c)
        (OUT / "clips" / f"{c['id']}.txt").write_text(text)
        keep.append({**c, "text": text})
    # clips without captions/txt are not part of the dataset: move them aside so the trainer does not see them
    have = {k["id"] for k in keep}
    skipped = OUT / "clips_skipped"
    for f in (OUT / "clips").glob("*.mp4"):
        if f.stem not in have:
            skipped.mkdir(exist_ok=True)
            f.rename(skipped / f.name)
            (OUT / "clips" / f"{f.stem}.txt").unlink(missing_ok=True)
            for c in (OUT / "cache").glob(f"*/{f.stem}_*"):
                c.unlink(missing_ok=True)
    write_jsonl(OUT / "manifest.jsonl", keep)
    log(f"dataset ready: {len(keep)} clips with captions in {OUT / 'clips'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["select", "cut", "purge", "caption", "finalize"])
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--extra", action="store_true", help="caption: also caption clips parked in clips_extra/")
    a = ap.parse_args()
    if a.cmd == "select":
        select(a.n)
    elif a.cmd == "cut":
        cut(a.workers)
    elif a.cmd == "purge":
        purge()
    elif a.cmd == "caption":
        caption(a.extra)
    else:
        finalize()
