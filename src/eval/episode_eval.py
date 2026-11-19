"""Stage-by-stage evaluation of a generated held-out episode against the real one (brief §10).

  python -m src.eval.episode_eval --run ep23_base --episode 23
Writes runs/<run>/eval/episode_eval.json and a Markdown block (eval/episode_eval.md) for RESULTS.md.

Stages: director (shot count / durations / shot sizes / coverage), keyframes (identity vs the real frames),
video (FVD + stability + quality vs real, and vs the animatic baseline), voice (speaker similarity, CER), cost.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np

from src.anime.oped import is_content, non_content
from src.common.config import P
from src.common.io import ffmpeg, log, read_json, read_jsonl, write_json


def real_clips(episode: int, out_dir: Path, n: int = 150, seed: int = 0) -> list[Path]:
    """Real shots of the held-out episode, encoded like the generated clips (832x480, 23.976 fps)."""
    shots = read_json(P("interim") / "anime" / f"ep{episode:02d}" / "shots_raw.json") or []
    nc = non_content(episode)
    shots = [s for s in shots if 1.2 <= s["end"] - s["start"] <= 8 and is_content(episode, s["start"], s["end"], nc)]
    random.Random(seed).shuffle(shots)
    src = next((P("raw") / "anime").glob(f"ep{episode:02d}_*.mkv"))
    out = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for s in shots[:n]:
        f = out_dir / f"ep{episode:02d}_s{s['shot']:04d}.mp4"
        if not f.exists():
            try:
                ffmpeg("-ss", f"{s['start'] + 0.05:.3f}", "-i", str(src), "-t", f"{s['end'] - s['start'] - 0.1:.3f}", "-map", "0:v:0",
                       "-vf", "scale=-2:480:flags=lanczos,crop=832:480,fps=24000/1001,format=yuv420p", "-c:v", "libx264", "-crf", "14", "-an", str(f))
            except Exception:
                continue
        out.append(f)
    return out


def clip_stats(paths: list[Path], cache: Path, max_n: int = 120) -> dict:
    """Mean stability / quality numbers over a set of clips (cached per file; same settings as the QA evidence)."""
    import os

    from src.eval import video_metrics as vm

    max_n = int(os.environ.get("EVAL_MAX_N", max_n))
    if len(paths) > max_n:          # spread over the episode instead of the first N shots
        paths = [paths[i] for i in sorted(set(int(round(x)) for x in np.linspace(0, len(paths) - 1, max_n)))]
    store = read_json(cache) or {}
    for p in paths[:max_n]:
        if str(p) in store:
            continue
        try:
            m = vm.analyze_clip(p, metrics=["flicker", "motion", "warp", "dover"], max_pairs=12, flow_height=288)
            fl, mo, do, wp = m.get("flicker") or {}, m.get("motion") or {}, m.get("dover") or {}, m.get("warp")
            store[str(p)] = {"held_frame_ratio": fl.get("held_frame_ratio"), "effective_fps": fl.get("effective_fps"),
                             "mean_abs_diff": fl.get("mean_abs_diff"), "mean_flow": mo.get("mean_flow"), "static_ratio": mo.get("static_ratio"),
                             "warp_error": wp if isinstance(wp, (int, float)) else (wp or {}).get("mean") if isinstance(wp, dict) else None,
                             "dover_overall": do.get("overall"), "dover_technical": do.get("technical"), "dover_aesthetic": do.get("aesthetic")}
        except Exception as e:
            store[str(p)] = {"error": str(e)[:100]}
    write_json(cache, store)
    agg = {}
    for k in ("held_frame_ratio", "effective_fps", "mean_abs_diff", "mean_flow", "static_ratio", "warp_error", "dover_overall", "dover_technical", "dover_aesthetic"):
        vals = [float(v[k]) for p in paths[:max_n] if (v := store.get(str(p))) and isinstance(v.get(k), (int, float))]
        if vals:
            agg[k] = round(float(np.mean(vals)), 4)
    agg["n"] = len(paths[:max_n])
    return agg


def identity_rate(frames: list[tuple[str, list[str]]]) -> dict:
    """Share of frames (with at least one expected named character) where CCIP recognises an expected character."""
    from src.anime.characters import identify

    hit = tot = faces = 0
    for path, expected in frames:
        if not expected:
            continue
        try:
            r = identify(path, expected)
        except Exception:
            continue
        if not r["faces"]:
            continue
        faces += 1
        tot += 1
        hit += int(any(f["same"] and f["best"] in expected for f in r["faces"]))
    return {"frames_with_faces": faces, "match_rate": round(hit / max(1, tot), 4)}


def evaluate(run: str, episode: int) -> dict:
    from src.eval.director_eval import evaluate as director_eval
    from src.eval.fvd import fvd

    rd = P("runs") / run
    ed = rd / "eval"
    ed.mkdir(parents=True, exist_ok=True)
    shots = (read_json(rd / "shotlist.json") or {}).get("shots", [])
    state = read_json(rd / "state.json") or read_json(rd / "clips_state.json") or {}      # v1/v2 | v3
    res: dict = {"run": run, "episode": episode}
    # ---------------- director
    try:
        res["director"] = director_eval(run)["summary"]
    except Exception as e:
        res["director"] = {"error": str(e)[:200]}
    # ---------------- clips
    gen = [rd / "shots" / s["shot_id"] / "clip.mp4" for s in shots if "TITLE_CARD" not in (s.get("notes") or "")]
    gen_wan = [p for s, p in zip([s for s in shots if "TITLE_CARD" not in (s.get("notes") or "")], gen) if (state.get(s["shot_id"]) or {}).get("final") == "wan" and p.exists()]
    ani = [rd / "shots" / s["shot_id"] / "animatic.mp4" for s in shots if "TITLE_CARD" not in (s.get("notes") or "")]
    gen, ani = [p for p in gen if p.exists()], [p for p in ani if p.exists()]
    real = real_clips(episode, P("runs") / "eval_real" / f"ep{episode:02d}")
    train_ref = real_clips(20, P("runs") / "eval_real" / "ep20", n=150, seed=1)      # real-vs-real floor (a training episode)
    video = {"n_generated": len(gen), "n_generated_wan_only": len(gen_wan), "n_animatic": len(ani), "n_real": len(real)}
    for name, a, b in (("fvd_generated_vs_real", real, gen), ("fvd_animatic_vs_real", real, ani), ("fvd_real_ep20_vs_real", real, train_ref)):
        try:
            r = fvd([str(x) for x in a], [str(x) for x in b])
            video[name] = {k: (round(float(v), 3) if isinstance(v, (int, float)) else v) for k, v in r.items() if k in ("fvd", "kvd", "n_real_windows", "n_gen_windows")}
        except Exception as e:
            video[name] = {"error": str(e)[:160]}
    video["stats_generated"] = clip_stats(gen, ed / "stats_gen.json")
    video["stats_animatic"] = clip_stats(ani, ed / "stats_animatic.json")
    video["stats_real"] = clip_stats(real, P("runs") / "eval_real" / f"stats_ep{episode:02d}.json")
    for f in ("ab_summary_animatic.json", "ab_summary_real.json"):
        if (ed / f).exists():
            video[f.replace(".json", "")] = read_json(ed / f)
    res["video"] = video
    # ---------------- keyframes: identity vs real frames
    try:
        kf = [(str(rd / "shots" / s["shot_id"] / "kf.png"), [c for c in s.get("characters", []) if c not in ("Narrator", "Classmates")])
              for s in shots if (rd / "shots" / s["shot_id"] / "kf.png").exists() and "TITLE_CARD" not in (s.get("notes") or "")]
        caps = {r["shot"]: r for r in read_jsonl(P("processed") / "anime" / "shot_captions" / f"ep{episode:02d}.jsonl")}
        kdir = P("interim") / "anime" / f"ep{episode:02d}" / "keyframes"
        real_kf = [(str(kdir / f"s{sid:04d}_m.jpg"), [c for c in c_.get("characters", []) if not str(c).lower().startswith("unknown")]) for sid, c_ in caps.items()]
        res["keyframes"] = {"identity_generated": identity_rate(kf), "identity_real": identity_rate(real_kf),
                            "keyframe_first_pass_rate": (read_json(rd / "report.json") or {}).get("keyframe_first_pass_rate")}
    except Exception as e:
        res["keyframes"] = {"error": str(e)[:200]}
    # ---------------- voice
    vmj = read_json(rd / "tts" / "voice_metrics.json") or {}
    res["voice"] = {"aggregate": vmj.get("aggregate"), "calibration": vmj.get("calibration"),
                    "n_lines": len(vmj.get("items", []))}
    # ---------------- cost
    rep = read_json(rd / "report.json") or {}
    res["cost"] = {k: rep.get(k) for k in ("gpu_hours", "timing_sec", "qa_retry_rate", "video_attempts_total", "video_first_pass_rate",
                                           "video_final_pass_rate", "final_wan", "final_fallback", "final_rung_counts", "duration_sec", "n_shots",
                                           "start", "frames_checked", "frames_ok", "frames_first_attempt_ok", "clips", "clips_final", "spoken_lines",
                                           "api_spend_total_usd") if k in rep}
    write_json(ed / "episode_eval.json", res)
    log(f"episode eval written: {ed / 'episode_eval.json'}")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--episode", type=int, required=True)
    a = ap.parse_args()
    evaluate(a.run, a.episode)
