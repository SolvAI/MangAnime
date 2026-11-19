"""Known-answer controls for the blind A/B judge on ITS OWN question: "animated clip vs pan/zoom of its first frame".

  python -m src.eval.ab_controls build        # CPU: two control runs from the judge-calibration clips (held-out, eval only)
  python -m src.eval.ab_controls report
Control runs (same layout as a generated run, so src.eval.ab_judge works unchanged; label "wan" = the animated side):
  abctl_real    clip = REAL anime shot            animatic = pan/zoom of that shot's first frame
                -> a valid judge prefers the real animation, at least for shots that really move
  abctl_broken  clip = deliberately BROKEN shot   animatic = pan/zoom of the CLEAN first frame
                (flicker, jitter, melt, ghost, noise/blur, colour shift, garbled text, shuffled frames, hard cut, freeze)
                -> a valid judge prefers the clean still
Identical clips (the fallback shots inside every real run) are the third control: they must come out as ties.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from src.common.config import P
from src.common.io import ffmpeg, log, media_duration, read_json, read_jsonl, write_json

SETS = ("judge_calibration", "judge_calibration_val")


def build() -> None:
    from src.eval.ab_judge import _keep_metrics
    from src.generate.video import still_clip

    for name, want in (("abctl_real", "real"), ("abctl_broken", "broken")):
        rd = P("runs") / name
        shots, state, primed, info = [], {}, {}, {}
        for cs in SETS:
            ev = read_json(P("runs") / cs / "evidence_metrics.json") or {}
            for r in read_jsonl(P("runs") / cs / "labels.jsonl"):
                if r["label"] != want or not Path(r["path"]).exists():
                    continue
                sid = r["id"]
                sd = rd / "shots" / sid
                sd.mkdir(parents=True, exist_ok=True)
                clip, kf, ani = sd / "clip.mp4", sd / "kf.png", sd / "animatic.mp4"
                try:
                    if not clip.exists():
                        shutil.copyfile(r["path"], clip)
                    if not kf.exists():
                        ffmpeg("-i", r["src"], "-frames:v", "1", str(kf))        # first frame of the CLEAN shot
                    dur = media_duration(clip)
                    if not ani.exists():
                        still_clip(kf, ani, dur, "static")                           # the pipeline's default animatic move
                except Exception as e:
                    log(f"control {sid}: {str(e)[:100]}")
                    continue
                shots.append({"shot_id": sid, "duration": round(dur, 2), "shot_size": "medium", "notes": "", "characters": [], "dialogue": []})
                state[sid] = {"final": "wan", "verdict": "pass", "vid_attempt": 0}
                m = (ev.get(sid) or {}).get("metrics") or {}
                k = _keep_metrics(m) if m else {}
                if k and "error" not in k:
                    st = clip.stat()
                    primed[f"{clip}|{st.st_size}|{int(st.st_mtime)}"] = k               # measured already by the calibration run
                info[sid] = {"kind": r["kind"], "mean_flow": k.get("mean_flow"), "static_ratio": k.get("static_ratio")}
        write_json(rd / "shotlist.json", {"shots": shots, "control": name})
        write_json(rd / "state.json", state)
        (rd / "eval").mkdir(exist_ok=True)
        old = read_json(rd / "eval" / "ab_metrics_files.json") or {}
        old.update(primed)
        write_json(rd / "eval" / "ab_metrics_files.json", old)
        write_json(rd / "eval" / "control_info.json", info)
        log(f"{name}: {len(shots)} pairs ({len(primed)} clip measurements reused)")


def report() -> dict:
    import collections

    from tools.ab_page.stats import fetch_ratings, sign_test_p, wilson_ci

    db = P("runs") / "ab" / "results.duckdb"
    out = {}
    for name in ("abctl_real", "abctl_broken"):
        info = read_json(P("runs") / name / "eval" / "control_info.json") or {}
        rows = [r for r in fetch_ratings(db, f"{name}_wan_vs_animatic") if r["criterion"] == "overall"]
        flows = sorted(v["mean_flow"] for v in info.values() if isinstance(v.get("mean_flow"), (int, float)))
        med = flows[len(flows) // 2] if flows else 0.0
        res = {}
        for rater in sorted({r["rater"] for r in rows}):
            rs = [r for r in rows if r["rater"] == rater]
            groups = {"all": rs}
            if name == "abctl_real":
                groups["real shots that move (mean flow above the median)"] = [r for r in rs if (info.get(r["pair_id"], {}).get("mean_flow") or 0) > med]
                groups["real shots that barely move"] = [r for r in rs if (info.get(r["pair_id"], {}).get("mean_flow") or 0) <= med]
            else:
                for kind in sorted({v["kind"] for v in info.values()}):
                    groups[kind] = [r for r in rs if info.get(r["pair_id"], {}).get("kind") == kind]
            res[rater] = {}
            for g, gr in groups.items():
                c = collections.Counter(r["winner_label"] or "tie" for r in gr)
                a, b, t = c.get("wan", 0), c.get("animatic", 0), c.get("tie", 0)
                lo, hi = wilson_ci(a, a + b) if a + b else (None, None)
                res[rater][g] = {"animated": a, "still": b, "tie": t, "animated_share": round(a / (a + b), 3) if a + b else None,
                                 "ci95": [round(lo, 2), round(hi, 2)] if lo is not None else None, "p": round(sign_test_p(max(a, b), a + b), 4) if a + b else None}
        out[name] = res
    write_json(P("runs") / "ab" / "judge_controls.json", out)
    for name, res in out.items():
        print(name)
        for rater, gs in res.items():
            for g, v in gs.items():
                print(f"   {rater:34s} {g:52s} animated {v['animated']:3d} : still {v['still']:3d}  tie {v['tie']:2d}  share {v['animated_share']}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "report"])
    a = ap.parse_args()
    build() if a.cmd == "build" else report()
