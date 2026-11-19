"""Judge calibration (brief §3b): ~50 REAL held-out shots (label pass) + ~50 deliberately BROKEN ones (label fail).

Labels are known by construction, so no human is needed. Require >= 85 % agreement before trusting the judge in the
loop; otherwise improve the evidence bundle / prompt and re-test.

  python -m src.qa.calibrate build   # cut real held-out shots, make broken variants (CPU)
  python -m src.qa.calibrate run     # evidence (metrics + Qwen) + judge -> runs/judge_calibration/report.json
"""
from __future__ import annotations

import argparse
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.anime.oped import is_content, non_content
from src.common import servers
from src.common.config import CFG, P
from src.common.io import ffmpeg, log, read_json, read_jsonl, write_json, write_jsonl

import os

OUT = P("runs") / ("judge_calibration" + ("_" + os.environ["CALIB_SET"] if os.environ.get("CALIB_SET") else ""))
BROKEN_KINDS = ["flicker", "frame_shuffle", "melt", "ghost", "noise_blur", "color_shift", "garbled_text_overlay", "jitter", "freeze", "hard_cut"]


def build(n_real: int = 50, n_broken: int = 50, seed: int = 0) -> list[dict]:
    """Real clips: held-out shots of 2-6 s, re-encoded like generated clips (832x480, 23.976 fps, no audio)."""
    from src.eval import broken as B

    rng = random.Random(seed)
    cands = []
    for ep in CFG.split.heldout_episodes:
        shots = read_json(P("interim") / "anime" / f"ep{ep:02d}" / "shots_raw.json") or read_json(P("interim") / "anime" / f"ep{ep:02d}" / "shots_fb.json") or []
        nc = non_content(ep)
        subs = read_json(P("processed") / "anime" / "subs" / f"ep{ep:02d}.json", [])
        for s in shots:
            d = s["end"] - s["start"]
            if 2.0 <= d <= 6.0 and is_content(ep, s["start"], s["end"], nc) and not any(
                    e["kind"] in ("title",) and e["start"] < s["end"] and e["end"] > s["start"] for e in subs):
                cands.append({"ep": ep, **s})
    if os.environ.get("CALIB_SET"):      # a second (validation) set never reuses shots of the first one
        used = {r["id"][:10] for r in read_jsonl(P("runs") / "judge_calibration" / "labels.jsonl")}
        cands = [c for c in cands if f"ep{c['ep']:02d}_s{c['shot']:04d}" not in used]
    rng.shuffle(cands)
    picked = cands[: n_real + n_broken]
    rows = []
    for i, s in enumerate(picked):
        cid = f"ep{s['ep']:02d}_s{s['shot']:04d}"
        clip = OUT / "src" / f"{cid}.mp4"
        if not clip.exists():
            clip.parent.mkdir(parents=True, exist_ok=True)
            src = next((P("raw") / "anime").glob(f"ep{s['ep']:02d}_*.mkv"))
            ffmpeg("-ss", f"{s['start'] + 0.05:.3f}", "-i", str(src), "-t", f"{s['end'] - s['start'] - 0.1:.3f}", "-map", "0:v:0",
                   "-vf", "scale=-2:480:flags=lanczos,crop=832:480,fps=24000/1001,format=yuv420p", "-c:v", "libx264", "-crf", "14", "-an", str(clip))
        if i < n_real:
            rows.append({"id": cid, "label": "real", "kind": "real", "path": str(clip), "src": str(clip), "ep": s["ep"], "shot": s["shot"],
                         "start": s["start"], "end": s["end"]})
        else:
            kind = BROKEN_KINDS[(i - n_real) % len(BROKEN_KINDS)]
            out = OUT / "broken" / f"{cid}_{kind}.mp4"
            if not out.exists():
                out.parent.mkdir(parents=True, exist_ok=True)
                row = None
                for kk in [kind] + [k for k in ("flicker", "noise_blur", "melt", "color_shift") if k != kind]:
                    # a broken clip must be visibly/measurably broken (e.g. freezing a still shot changes nothing)
                    try:
                        kw = {"other": rows[0]["path"]} if kk == "hard_cut" and rows else {}
                        cand = OUT / "broken" / f"{cid}_{kk}.mp4"
                        row = B.make_broken(clip, kk, cand, seed=seed + i, **kw)
                        if B.effect_ok(row):
                            kind, out = kk, cand
                            break
                        cand.unlink(missing_ok=True)
                        row = None
                    except Exception as e:
                        log(f"broken {cid}/{kk} failed: {str(e)[:120]}")
                        row = None
                if row is None:
                    continue
            rows.append({"id": f"{cid}_{kind}", "label": "broken", "kind": kind, "path": str(out), "src": str(clip), "ep": s["ep"],
                         "shot": s["shot"], "start": s["start"], "end": s["end"]})
    write_jsonl(OUT / "labels.jsonl", rows)
    log(f"calibration set: {sum(r['label'] == 'real' for r in rows)} real + {sum(r['label'] == 'broken' for r in rows)} broken")
    return rows


def _wait_qwen(max_min: float = 240) -> None:
    """keep-servers mode: the pipeline owns the GPU servers; wait until its Qwen phase is up."""
    import time

    t0 = time.time()
    while not servers.is_up("qwen"):
        if time.time() - t0 > max_min * 60:
            raise TimeoutError("qwen server never came up")
        time.sleep(10)


def run(send_frames: bool = True, phase: str = "all", keep_servers: bool = False) -> dict:
    """Spec = what the local VLM sees in the CLEAN source clip (+ subtitles); evidence from the clip under test.

    phase "metrics": only the small-model numbers (resumable, may run on CPU while the GPU is busy);
    phase "judge": VLM evidence + judge, reusing cached metrics. keep_servers: never start/stop the GPU servers
    (piggy-back on the Qwen phases of a running pipeline; every VLM result is cached, so a rerun continues)."""
    from src.anime.caption import caption_shot
    from src.anime.characters import identify
    from src.eval import video_metrics as vm
    from src.llm.client import LLM
    from src.qa.evidence import _compact, hard_fail, sample_frames, visual_evidence
    from src.qa.judge import judge_shot

    rows = read_jsonl(OUT / "labels.jsonl")
    if not keep_servers:
        servers.stop("comfy")
        servers.stop("qwen")
    # ---- small-model evidence first (GPU otherwise idle)
    mfile, vfile = OUT / "evidence_metrics.json", OUT / "evidence_visual.json"
    evs = read_json(mfile) or {}
    vcache = read_json(vfile) or {}
    for i, r in enumerate(rows):
        if r["id"] in evs or phase == "judge":
            continue
        ev = {"shot_id": r["id"]}
        try:
            # real anime contains genuine stills: frozen video is not a hard failure in the calibration set
            m = vm.analyze_clip(r["path"], metrics=["technical", "flicker", "motion", "warp", "dover"], expected_dur=r["end"] - r["start"] - 0.1,
                                allow_static=True, max_pairs=12, flow_height=288)
        except Exception as e:
            m = {"error": str(e)[:200]}
        ev["metrics"] = _compact(m)
        frames = sample_frames(Path(r["path"]), OUT / "frames" / r["id"], 3) if not m.get("error") else []
        ev["frames"] = [str(f) for f in frames]
        evs[r["id"]] = ev
        if i % 10 == 9:
            write_json(mfile, evs)
            log(f"calibration metrics {i + 1}/{len(rows)}")
    write_json(mfile, evs)
    try:
        vm.unload_models()
    except Exception:
        pass
    if phase == "metrics":
        return {"n_metrics": len(evs)}
    rows = [r for r in rows if r["id"] in evs]
    if keep_servers:
        if any(r["id"] not in vcache for r in rows):      # cached VLM evidence needs no server at all
            _wait_qwen()
    else:
        servers.only("qwen")
    qwen = LLM("local", role="calibration")
    subs = {ep: read_json(P("processed") / "anime" / "subs" / f"ep{ep:02d}.json", []) for ep in CFG.split.heldout_episodes}

    def spec_of(r):
        src_frames = sample_frames(Path(r["src"]), OUT / "frames_src" / r["id"], 3)
        cap = caption_shot(qwen, src_frames)
        dlg = [{"speaker": "?", "kind": "speech", "text": e["text"], "text_ja": "", "emotion": "neutral"} for e in subs[r["ep"]]
               if e["kind"] == "dialogue" and r["start"] <= (e["start"] + e["end"]) / 2 < r["end"]]
        return {"shot_id": r["id"], "duration": round(r["end"] - r["start"] - 0.1, 2), "shot_size": cap.get("shot_size"), "camera": cap.get("camera"),
                "characters": [c for c in cap.get("characters", []) if not str(c).lower().startswith("unknown")],
                "location": cap.get("setting", ""), "time_of_day": "", "first_frame": cap.get("caption", ""), "action": cap.get("action", ""),
                "notes": "", "dialogue": []}       # audio is not part of this calibration (clips are silent)

    import threading

    lock = threading.Lock()

    def prep(r):
        ev = evs[r["id"]]
        if r["id"] in vcache:
            ev.update({k: v for k, v in vcache[r["id"]]["ev"].items() if k in ("visual", "identity")})
            return r["id"], vcache[r["id"]]["spec"]
        for attempt in range(6 if keep_servers else 1):
            try:
                if keep_servers:
                    _wait_qwen()
                rid, spec = _prep(r)
                with lock:
                    vcache[rid] = {"spec": spec, "ev": {k: ev.get(k) for k in ("visual", "identity")}}
                    write_json(vfile, vcache)
                return rid, spec
            except Exception as e:
                err = str(e)[:200]
        return r["id"], {"shot_id": r["id"], "duration": r["end"] - r["start"], "characters": [], "dialogue": [], "error": err}

    def _prep(r):
        if True:
            spec = spec_of(r)
            ev = evs[r["id"]]
            if ev["frames"]:
                ev["visual"] = visual_evidence(qwen, ev["frames"], spec)
                ident = []
                for f in ev["frames"]:
                    try:
                        x = identify(f, spec["characters"])
                        ident.append({"frame": Path(f).name, "faces": [{k: y[k] for k in ("best", "diff", "same", "expected_diffs")} for y in x["faces"]], "threshold": x["threshold"]})
                    except Exception:
                        pass
                ev["identity"] = {"expected": spec["characters"], "frames": ident}
            return r["id"], spec

    with ThreadPoolExecutor(int(CFG.llm.local.parallel)) as ex:
        specs = dict(ex.map(prep, rows))

    def jd(r):
        ev = evs[r["id"]]
        if specs[r["id"]].get("error"):
            return r["id"], {"verdict": "error", "failure_reasons": [specs[r["id"]]["error"]]}
        hf = hard_fail(ev)
        if hf:
            return r["id"], {"verdict": "fail", "failure_reasons": hf, "judge_endpoint": "hard_fail"}
        try:
            return r["id"], judge_shot(specs[r["id"]], ev, stage="video", send_frames=send_frames)
        except Exception as e:
            return r["id"], {"verdict": "error", "failure_reasons": [str(e)[:200]]}

    with ThreadPoolExecutor(6) as ex:
        verdicts = dict(ex.map(jd, rows))
    res, by_kind = [], {}
    for r in rows:
        v = verdicts[r["id"]]
        want = "pass" if r["label"] == "real" else "fail"
        ok = v["verdict"] == want
        res.append({"id": r["id"], "label": r["label"], "kind": r["kind"], "verdict": v["verdict"], "agree": ok,
                    "reasons": v.get("failure_reasons", []), "endpoint": v.get("judge_endpoint")})
        k = by_kind.setdefault(r["kind"], [0, 0])
        k[0] += int(ok)
        k[1] += 1
    valid = [x for x in res if x["verdict"] in ("pass", "fail")]
    rep = {"n": len(valid), "agreement": round(sum(x["agree"] for x in valid) / max(1, len(valid)), 4),
           "real_pass_rate": round(sum(x["agree"] for x in valid if x["label"] == "real") / max(1, sum(x["label"] == "real" for x in valid)), 4),
           "broken_fail_rate": round(sum(x["agree"] for x in valid if x["label"] == "broken") / max(1, sum(x["label"] == "broken" for x in valid)), 4),
           "by_kind": {k: f"{a}/{b}" for k, (a, b) in sorted(by_kind.items())}, "frames_sent": send_frames,
           "threshold": 0.85, "trusted": None, "items": res}
    rep["trusted"] = rep["agreement"] >= 0.85
    write_json(OUT / "report.json", rep)
    log(f"judge calibration: agreement {rep['agreement']:.3f} (real pass {rep['real_pass_rate']:.3f}, broken fail {rep['broken_fail_rate']:.3f}) trusted={rep['trusted']}")
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "run"])
    ap.add_argument("--no-frames", action="store_true")
    ap.add_argument("--phase", default="all", choices=["all", "metrics", "judge"])
    ap.add_argument("--seed", type=int, default=0, help="build: which shots / defects (use with CALIB_SET=<name> for a second set)")
    ap.add_argument("--keep-servers", action="store_true")
    a = ap.parse_args()
    build(seed=a.seed) if a.cmd == "build" else run(send_frames=not a.no_frames, phase=a.phase, keep_servers=a.keep_servers)
