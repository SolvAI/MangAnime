"""Pair every generated shot with the real anime shot that adapted the same manga panel (brief §10, evaluation only).

  python -m src.eval.real_pairs --run ep22 --episode 22 [--min-conf 0.5]
Uses the held-out rows of alignment.jsonl (panel -> real shot). Writes per shot
  runs/<run>/shots/<id>/real.mp4     the real shot, encoded like the generated clips (832x480, 23.976 fps, no audio)
  runs/<run>/shots/<id>/real_kf.jpg  its middle frame (1024x576)
and runs/<run>/eval/real_pairs.json. The real episode is only ever read here and in src/eval — never by the generator.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor

from src.common.config import P
from src.common.io import ffmpeg, log, read_json, read_jsonl, write_json


def build(run: str, episode: int, min_conf: float = 0.5) -> list[dict]:
    rd = P("runs") / run
    shots = (read_json(rd / "shotlist.json") or {}).get("shots", [])
    al = {(r["chapter"], r["panel_order"]): r for r in read_jsonl(P("processed") / "align" / "alignment.jsonl")
          if r.get("shots") and r["shots"][0]["episode"] == episode and r["confidence"] >= min_conf}
    real = {s["shot"]: s for s in read_json(P("interim") / "anime" / f"ep{episode:02d}" / "shots_raw.json") or []}
    src = next((P("raw") / "anime").glob(f"ep{episode:02d}_*.mkv"))
    jobs = []
    for s in shots:
        if "TITLE_CARD" in (s.get("notes") or "") or not s.get("source_panels"):
            continue
        r = al.get((s["chapter"], s["source_panels"][0]))
        if not r:
            continue
        rs = real.get(r["shots"][0]["shot"])
        if not rs or rs["end"] - rs["start"] < 0.5:
            continue
        jobs.append({"shot_id": s["shot_id"], "real_shot": rs["shot"], "start": rs["start"], "end": rs["end"], "match": r["match"],
                     "confidence": r["confidence"], "generated_duration": s["duration"]})

    def cut(j):
        sd = rd / "shots" / j["shot_id"]
        sd.mkdir(parents=True, exist_ok=True)
        v, k = sd / "real.mp4", sd / "real_kf.jpg"
        dur = min(j["end"] - j["start"] - 0.1, 10.0)
        try:
            if not v.exists():
                ffmpeg("-ss", f"{j['start'] + 0.05:.3f}", "-i", str(src), "-t", f"{dur:.3f}", "-map", "0:v:0",
                       "-vf", "scale=-2:480:flags=lanczos,crop=832:480,fps=24000/1001,format=yuv420p", "-c:v", "libx264", "-crf", "14", "-an", str(v))
            if not k.exists():
                ffmpeg("-ss", f"{(j['start'] + j['end']) / 2:.3f}", "-i", str(src), "-frames:v", "1", "-vf", "scale=1024:576:flags=lanczos", "-q:v", "2", str(k))
            return j
        except Exception as e:
            log(f"real pair {j['shot_id']}: {str(e)[:100]}")
            return None

    with ThreadPoolExecutor(6) as ex:
        done = [j for j in ex.map(cut, jobs) if j]
    write_json(rd / "eval" / "real_pairs.json", {"episode": episode, "min_conf": min_conf, "n_shots": len(shots), "pairs": done})
    log(f"real pairs: {len(done)} of {len(shots)} generated shots have a real counterpart (episode {episode})")
    return done


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--min-conf", type=float, default=0.5)
    a = ap.parse_args()
    build(a.run, a.episode, a.min_conf)
