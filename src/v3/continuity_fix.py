"""v3 step 5c — CONTINUITY REPAIR of the first frames (D-055): the script supervisor's remarks are acted on before the clips are made.

  python -m src.v3.continuity_fix --run ep22_v3b [--rounds 2]
1. The continuity check (src/eval/continuity.py) looks at the first frames of every scene, eight consecutive shots at a time.
2. For every error it reports between two frames of a scene ("Komi's scarf is red in frame 3 and beige in frame 5"), the later
   frame is drawn again by the image model with the earlier frame as continuity reference and the remark as instruction
   (the scene's anchor frame is never the one that is redrawn).
3. Check again; repeat up to --rounds times. Writes runs/<run>/eval/continuity_frames_r<N>.json and continuity_fix.json.
Needs the Gemini frame backend (src/v3/frames_gemini.py).
"""
from __future__ import annotations

import argparse

from src.common.config import P
from src.common.io import log, read_json, write_json
from src.eval.continuity import evaluate
from src.v3.frames_gemini import redraw, scene_anchors


def first_frames(run: str) -> list[dict]:
    rd = P("runs") / run
    return [{"id": s["shot_id"], "group": s["scene_id"], "scene": s["scene_id"], "path": str(rd / "shots" / s["shot_id"] / "kf.png")}
            for s in read_json(rd / "shotlist.json")["shots"] if s["start"] in ("panel", "new") and (rd / "shots" / s["shot_id"] / "kf.png").exists()]


def repair(run: str, rounds: int = 2) -> dict:
    rd = P("runs") / run
    shots = read_json(rd / "shotlist.json")["shots"]
    order = {s["shot_id"]: i for i, s in enumerate(shots)}
    anchors = {a["shot_id"] for a in scene_anchors(shots).values()}
    hist = []
    for r in range(rounds + 1):
        out = rd / "eval" / f"continuity_frames_r{r}.json"
        summ = evaluate(first_frames(run), out, f"{run}_frames_r{r}")
        hist.append({k: summ[k] for k in ("cuts", "same_scene_cuts", "errors", "errors_by_kind", "people_followed", "people_inconsistent", "mean_score")})
        if r == rounds:
            break
        targets: dict[str, tuple[str, str]] = {}
        for w in read_json(out)["windows"]:
            for e in w["errors"]:
                a, b = sorted({e["shots"][0], e["shots"][-1]}, key=lambda x: order[x])[:2] if len(set(e["shots"])) > 1 else (None, None)
                if a is None:
                    continue
                tgt, ref = (b, a) if b not in anchors else (a, b)
                targets.setdefault(tgt, (ref, f"{e['kind']} - {e['note']}"))
        if not targets:
            break
        log(f"continuity repair round {r + 1}: {len(targets)} frames to redraw")
        redraw(rd, shots, set(targets), hints={t: v[1] for t, v in targets.items()}, match={t: v[0] for t, v in targets.items()},
               label=f"continuity repair {r + 1}", threads=10)
        hist[-1]["redrawn"] = sorted(targets)
    write_json(rd / "eval" / "continuity_fix.json", {"run": run, "rounds": hist})
    log(f"continuity repair {run}: errors per round {[h['errors'] for h in hist]}, same-scene cuts {[h['same_scene_cuts'] for h in hist]}")
    return {"rounds": hist}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--rounds", type=int, default=2)
    a = ap.parse_args()
    repair(a.run, a.rounds)
