"""Panel -> keyframe edit-LoRA dataset (brief §6): aligned (manga panel -> matched anime frame) pairs, TRAIN episodes only.

  python -m src.train.build_kf_dataset [--n 1200] [--min-conf 0.6]
Layout (contract with src/train/kf_lora.py):
  data/processed/train/kf/control/<id>.png   manga panel
  data/processed/train/kf/target/<id>.png    anime frame, 16:9 (1024x576)
  data/processed/train/kf/target/<id>.txt    instruction caption (same template as inference, with the trigger)
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.align.build import kf_paths, load_shots, panel_embs, shot_embs
from src.anime.oped import is_content, non_content
from src.common.config import CFG, P
from src.common.io import ffmpeg, log, read_json, read_jsonl, write_jsonl
from src.generate.keyframes import KF_H, KF_W, compose_cover
from src.generate.prompts import KF_TRIGGER, colorize_prompt

OUT = P("processed") / "train" / "kf"


def build(n: int = 1200, min_conf: float = 0.6, min_vis: float = 0.40, seed: int = 0) -> list[dict]:
    train = set(CFG.split.train_episodes)
    rows = [r for r in read_jsonl(P("processed") / "align" / "alignment.jsonl")
            if r["match"] == "dialogue" and r["confidence"] >= min_conf and (r.get("visual_sim") or 0) >= min_vis
            and r["shots"] and r["shots"][0]["episode"] in train]
    panels = {(p["chapter"], p["order"]): p for p in read_jsonl(P("processed") / "manga" / "panels.jsonl")}
    rng = random.Random(seed)
    rng.shuffle(rows)
    # character names for the hair-colour hints: from captioned training shots when the panel itself has no VLM caption
    named_chars: dict[tuple, list[str]] = {}
    for ep in train:
        for r in read_jsonl(P("processed") / "anime" / "shot_captions" / f"ep{ep:02d}.jsonl"):
            named_chars[(ep, r["shot"])] = [c for c in r.get("characters", []) if not str(c).lower().startswith("unknown")]
    subs_cache, nc_cache, out = {}, {}, []
    for r in rows:
        if len(out) >= n:
            break
        p = panels.get((r["chapter"], r["panel_order"]))
        if not p or p["bbox"][2] < 120 or p["bbox"][3] < 120:
            continue
        ep, shot_id = r["shots"][0]["episode"], r["shots"][0]["shot"]
        shots = {s["shot"]: s for s in load_shots(ep)}
        s = shots.get(shot_id)
        if not s:
            continue
        if ep not in nc_cache:
            nc_cache[ep] = non_content(ep)
            subs_cache[ep] = read_json(P("processed") / "anime" / "subs" / f"ep{ep:02d}.json", [])
        if not is_content(ep, s["start"], s["end"], nc_cache[ep]):
            continue
        if any(e["kind"] in ("title", "sign") and e["start"] < s["end"] and e["end"] > s["start"] for e in subs_cache[ep]):
            continue      # typeset text on screen: not a clean target frame
        # pick the keyframe (start / middle / end of the shot) that looks most like the panel
        try:
            pe = panel_embs([p])[0]
            se = shot_embs(ep, [shot_id])[0]
            k = int(np.argmax(se @ pe))
        except Exception:
            k = 1
        n_fr = s["f1"] - s["f0"] + 1
        frame = [s["f0"] + min(2, n_fr // 4), s["f0"] + n_fr // 2, s["f1"] - min(2, n_fr // 4)][k]
        t = frame / (24000 / 1001)
        pid = f"c{r['chapter']:03d}_p{r['panel_order']:03d}_ep{ep:02d}s{shot_id:04d}"
        tgt = OUT / "target" / f"{pid}.png"
        ctl = OUT / "control" / f"{pid}.png"
        if not tgt.exists():
            tgt.parent.mkdir(parents=True, exist_ok=True)
            src = next((P("raw") / "anime").glob(f"ep{ep:02d}_*.mkv"))
            try:
                ffmpeg("-ss", f"{t:.3f}", "-i", str(src), "-frames:v", "1", "-vf", f"scale={KF_W}:{KF_H}:flags=lanczos", str(tgt))
            except Exception:
                continue
        if not ctl.exists():
            compose_cover(p, ctl)        # exactly the inference input: 16:9 crop, dialogue lettering blanked
        chars = p.get("characters") or sorted({n for n in (named_chars.get((ep, shot_id)) or [])})
        text = KF_TRIGGER + ". " + colorize_prompt(chars)
        (OUT / "target" / f"{pid}.txt").write_text(text)
        out.append({"id": pid, "chapter": r["chapter"], "panel_order": r["panel_order"], "episode": ep, "shot": shot_id,
                    "confidence": r["confidence"], "visual_sim": r.get("visual_sim"), "control": str(ctl), "target": str(tgt), "text": text})
    write_jsonl(OUT / "manifest.jsonl", out)
    log(f"keyframe dataset: {len(out)} pairs (from {len(rows)} candidate aligned panels)")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--min-conf", type=float, default=0.6)
    a = ap.parse_args()
    build(a.n, a.min_conf)
