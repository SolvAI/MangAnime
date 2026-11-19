"""Blind A/B (brief §10): generated video vs (a) the animatic baseline and (b) the real episode.

Provisional automatic raters, written into the SAME DuckDB the human rating page uses (tools/ab_page):
  rater "judge:deepseek-flash" - sees two frame strips (A / B, random order, labels hidden) + local metrics per side
  rater "metric:dover"         - picks the higher DOVER overall score (local evidence only; reported separately)
and builds manifests for the human page (shot-level and ~20 s segment-level pairs).

  python -m src.eval.ab_judge --run ep23_base [--limit 0] [--versus animatic|real]
"""
from __future__ import annotations

import argparse
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from src.common.config import P
from src.common.io import ffmpeg, log, media_duration, read_json, sha, write_json
from src.llm.client import BudgetExceeded, remote_or_local, vision


def judge_llm():
    """The automatic rater: DeepSeek (v1/v2 runs) or, with KOMI_AB_JUDGE=gemini, the visual model of the v3 pipeline (D-044)."""
    import os

    if os.environ.get("KOMI_AB_JUDGE", "") == "gemini":
        return vision("judge"), "judge:gemini-flash"
    llm = remote_or_local("judge")
    return llm, ("judge:deepseek-flash" if llm.endpoint == "deepseek" else "judge:qwen-local")     # KOMI_FORCE_LOCAL=1 -> second, independent judge


def run_state(rd: Path) -> dict:
    """Per-shot final state: state.json (v1/v2 pipeline) or clips_state.json (v3)."""
    return read_json(rd / "state.json") or read_json(rd / "clips_state.json") or {}

CRITERIA = ["overall", "looks like real anime", "motion", "character consistency"]
SCHEMA = {"type": "object", "required": ["overall", "looks_like_real_anime", "motion", "character_consistency", "reason"],
          "properties": {k: {"type": "string"} for k in ("overall", "looks_like_real_anime", "motion", "character_consistency", "reason")}}

SYSTEM = """You compare two versions (A and B) of the same shot from an anime episode. Each version is shown as ONE image:
a horizontal strip of 4 frames sampled at the same times (left = start, right = end of the shot). You also get a few
local measurements per version. Judge like a viewer of TV anime: a good shot keeps the characters on-model in every
frame, has purposeful limited animation (mouth flaps, blinks, gestures, hair/cloth movement, camera moves) and no
morphing, flicker, melting faces or warped anatomy. A still drawing with a slow pan/zoom is a legitimate TV-anime
technique; motion only counts in a version's favour when it is purposeful and keeps the drawing intact, and a shot
whose motion damages the drawing is worse than a clean still. Do not guess how either version was made.
Answer for each criterion with "A", "B" or "tie". Return ONE JSON object:
{"overall": "...", "looks_like_real_anime": "...", "motion": "...", "character_consistency": "...", "reason": "one sentence"}"""


def strip(video: Path, out: Path, n: int = 4, w: int = 384) -> Path | None:
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        return None
    frames = []
    for i in np.linspace(0, total - 1, n).round().astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            h = int(f.shape[0] * w / f.shape[1])
            frames.append(cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA))
    cap.release()
    if not frames:
        return None
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.concatenate(frames, 1), [cv2.IMWRITE_JPEG_QUALITY, 88])
    return out


def quick_metrics(video: Path) -> dict:
    """Cheap per-clip numbers given to the judge as evidence (and used by the metric rater)."""
    try:
        from src.eval import video_metrics as vm

        m = vm.analyze_clip(video, metrics=["flicker", "motion", "warp", "dover"], max_pairs=12, flow_height=288)   # same settings as the QA evidence
        return _keep_metrics(m)
    except Exception as e:
        return {"error": str(e)[:100]}


def _keep_metrics(m: dict) -> dict:
    """The few numbers shown to the A/B judge, from an analyze_clip result or a QA evidence bundle's `metrics`."""
    try:
        fl, mo, do = m.get("flicker") or {}, m.get("motion") or {}, m.get("dover") or {}
        keep = {}
        for src, keys in ((fl, ("mean_abs_diff", "held_frame_ratio", "effective_fps", "flicker_index")),
                          (mo, ("mean_flow", "static_ratio")), (do, ("overall", "technical", "aesthetic"))):
            for k in keys:
                if isinstance(src, dict) and isinstance(src.get(k), (int, float)):
                    keep[k] = round(float(src[k]), 4)
        if isinstance(m.get("warp"), (int, float)):
            keep["warp_error"] = round(float(m["warp"]), 4)
        elif isinstance(m.get("warp"), dict) and isinstance(m["warp"].get("mean"), (int, float)):
            keep["warp_error"] = round(float(m["warp"]["mean"]), 4)
        return keep
    except Exception as e:
        return {"error": str(e)[:100]}


class FileMetrics:
    """quick_metrics cached per clip file (path + size + mtime), so the real / animatic sides can be measured ahead of time."""

    def __init__(self, rd: Path):
        self.path = rd / "eval" / "ab_metrics_files.json"
        self.d = read_json(self.path) or {}
        self.n = 0

    def from_qa(self, f: Path) -> dict | None:
        """Final Wan clips were already measured by the QA stage with the same settings: reuse that evidence."""
        sid = f.parent.name
        if f.name == "clip.mp4":
            rdir = self.path.parent.parent
        elif (P("runs") / f.stem / "state.json").exists():      # "<other run>.mp4": that run's final clip, copied next to ours
            rdir = P("runs") / f.stem
        else:
            return None
        if not hasattr(self, "states"):
            self.states = {}
        if rdir not in self.states:
            self.states[rdir] = read_json(rdir / "state.json") or {}
        st = self.states[rdir].get(sid) or {}
        if st.get("final") != "wan":
            return None
        q = read_json(rdir / "shots" / sid / f"qa_video_a{st.get('vid_attempt', 0)}.json") or {}
        m = (q.get("evidence") or {}).get("metrics") or {}
        k = _keep_metrics(m) if m else {}
        return k if k.get("overall") is not None and "error" not in k else None

    def get(self, f: Path) -> dict:
        st = f.stat()
        key = f"{f}|{st.st_size}|{int(st.st_mtime)}"
        if key not in self.d or "error" in self.d[key]:
            self.d[key] = self.from_qa(f) or quick_metrics(f)
            self.n += 1
            if self.n % 20 == 0:
                self.save()
        return self.d[key]

    def save(self) -> None:
        write_json(self.path, self.d)


def precompute(run: str, name: str) -> int:
    """Measure shots/*/<name>.mp4 ahead of the A/B (CPU is fine)."""
    rd = P("runs") / run
    fm = FileMetrics(rd)
    fs = sorted((rd / "shots").glob(f"*/{name}.mp4"))
    for i, f in enumerate(fs):
        fm.get(f)
        if i % 25 == 24:
            log(f"precompute {name}: {i + 1}/{len(fs)}")
    fm.save()
    return len(fs)


def pairs_for(run: str, versus: str = "animatic") -> list[dict]:
    rd = P("runs") / run
    shots = (read_json(rd / "shotlist.json") or {}).get("shots", [])
    state = run_state(rd)
    out = []
    for s in shots:
        if "TITLE_CARD" in (s.get("notes") or ""):
            continue
        sd = rd / "shots" / s["shot_id"]
        a = sd / "clip.mp4"
        b = sd / f"{versus}.mp4"            # animatic.mp4 | real.mp4 | base.mp4 (LoRA-vs-base variant runs)
        if a.exists() and b.exists():
            out.append({"pair_id": s["shot_id"], "a": str(a), "b": str(b), "label_a": "wan", "label_b": versus,
                        "meta": {"final": (state.get(s["shot_id"]) or {}).get("final"), "duration": s["duration"], "shot_size": s.get("shot_size")}})
    return out


def judge_pairs(run: str, versus: str = "animatic", limit: int = 0, seed: int = 0, with_metrics: bool = True, tag: str = "") -> dict:
    """tag: suffix of the rater name for a control pass (e.g. "frames-only" with with_metrics=False), so it is stored
    next to the main rating of the same pairs instead of being skipped as already rated."""
    from tools.ab_page.stats import rated_pairs, record_rating, summarize

    rd = P("runs") / run
    db = P("runs") / "ab" / "results.duckdb"
    db.parent.mkdir(parents=True, exist_ok=True)
    exp = f"{run}_wan_vs_{versus}"
    pairs = pairs_for(run, versus)
    write_json(rd / "eval" / f"ab_manifest_{versus}.json", {"experiment": exp, "pairs": pairs, "criteria": CRITERIA})
    rng = random.Random(seed)
    if limit:
        pairs = rng.sample(pairs, min(limit, len(pairs)))
    llm, rater = judge_llm()
    if tag:
        rater += ":" + tag
    try:
        done = rated_pairs(db, exp, rater)
    except Exception:
        done = set()
    todo = [p for p in pairs if p["pair_id"] not in done]
    tmp = rd / "eval" / "strips"
    metrics_cache = read_json(rd / "eval" / f"ab_metrics_{versus}.json") or {}
    files = FileMetrics(rd)

    def prep(p):
        sa, sb = strip(Path(p["a"]), tmp / f"{p['pair_id']}_wan.jpg"), strip(Path(p["b"]), tmp / f"{p['pair_id']}_{versus}.jpg")
        if with_metrics and p["pair_id"] not in metrics_cache:
            metrics_cache[p["pair_id"]] = {"wan": files.get(Path(p["a"])), versus: files.get(Path(p["b"]))}
        return sa, sb

    preps = [prep(p) for p in todo]            # sequential: metrics models are not thread-safe
    files.save()
    write_json(rd / "eval" / f"ab_metrics_{versus}.json", metrics_cache)

    def one(args):
        p, (sa, sb) = args
        if sa is None or sb is None:
            return p, None, None
        flip = int(sha(p["pair_id"] + str(seed), 8), 16) % 2 == 1      # which version is shown as "A"
        first, second = (sb, sa) if flip else (sa, sb)
        m = metrics_cache.get(p["pair_id"], {}) if with_metrics else {}
        ma, mb = (m.get(versus), m.get("wan")) if flip else (m.get("wan"), m.get(versus))
        user = (f"Shot spec: {p['meta'].get('shot_size')} shot, {p['meta'].get('duration')} s.\n"
                f"Image 1 = version A, image 2 = version B.\n")
        if with_metrics:
            user += (f"Local measurements A: {ma}\nLocal measurements B: {mb}\n"
                     "(held_frame_ratio of 0.5-0.7 is normal for animation on twos / threes; high warp_error/flicker means unstable drawing.)")
        else:
            user += "No measurements are available: judge from the frames only."
        try:
            v = llm.chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}], schema=SCHEMA, images=[first, second],
                         max_tokens=300, temperature=0.0, thinking=False, image_max_side=1536, tag=f"ab:{versus}:{p['pair_id']}")
        except BudgetExceeded:
            return p, None, None
        except Exception as e:
            log(f"ab judge {p['pair_id']}: {str(e)[:120]}")
            return p, None, None

        def unflip(x):
            x = str(x).strip().upper()
            if x not in ("A", "B"):
                return "tie"
            shown_wan = "B" if flip else "A"
            return "a" if x == shown_wan else "b"            # 'a' = label_a = wan

        choices = {"overall": unflip(v["overall"]), "looks like real anime": unflip(v["looks_like_real_anime"]),
                   "motion": unflip(v["motion"]), "character consistency": unflip(v["character_consistency"])}
        return p, choices, {"reason": v.get("reason", ""), "flip": flip}

    with ThreadPoolExecutor(6) as ex:
        results = list(ex.map(one, zip(todo, preps)))
    n = 0
    for p, choices, meta in results:
        if not choices:
            continue
        record_rating(db, exp, p["pair_id"], rater, choices, "wan", versus, left_side="b" if meta["flip"] else "a",
                      note=meta["reason"][:300], meta=p["meta"])
        n += 1
    # metric-only rater (evidence, not a verdict): higher DOVER overall wins, tie within 2 %
    try:
        done_m = rated_pairs(db, exp, "metric:dover")
    except Exception:
        done_m = set()
    for p in pairs:
        m = metrics_cache.get(p["pair_id"])
        if not m or p["pair_id"] in done_m:
            continue
        a, b = (m.get("wan") or {}).get("overall"), (m.get(versus) or {}).get("overall")
        if a is None or b is None:
            continue
        ch = "tie" if abs(a - b) <= 0.02 * max(abs(a), abs(b), 1e-6) else ("a" if a > b else "b")
        record_rating(db, exp, p["pair_id"], "metric:dover", {"overall": ch}, "wan", versus, meta=p["meta"])
    summ = summarize(db, exp)
    write_json(rd / "eval" / f"ab_summary_{versus}.json", summ)
    log(f"A/B {exp}: judged {n} new pairs (total pairs {len(pairs)})")
    return summ


FRAME_SYSTEM = """You compare two still frames (A and B) that show the same story moment of the TV anime
"Komi Can't Communicate". One or both may be machine-generated. Judge like a viewer: clean line art, flat cel colours,
on-model faces and uniforms, a finished painted background, no leftover manga lettering / speech balloons / screentone,
no distorted anatomy. The two frames may be framed differently; do not reward or punish framing.
Answer each criterion with "A", "B" or "tie". Return ONE JSON object:
{"overall": "...", "looks_like_real_anime": "...", "character_consistency": "...", "reason": "one sentence"}"""
FRAME_SCHEMA = {"type": "object", "required": ["overall", "looks_like_real_anime", "character_consistency", "reason"],
                "properties": {k: {"type": "string"} for k in ("overall", "looks_like_real_anime", "character_consistency", "reason")}}


def judge_frames(run: str, limit: int = 0, seed: int = 0, other_run: str = "") -> dict:
    """Keyframe A/B (brief §10): generated first frame vs the middle frame of the real shot that adapted the same panel.
    other_run: compare instead against the keyframes of another run of the same shot list (label = that run's name)."""
    from tools.ab_page.stats import rated_pairs, record_rating, summarize

    rd = P("runs") / run
    db = P("runs") / "ab" / "results.duckdb"
    other = other_run or "real"
    exp = f"{run}_keyframe_vs_{other}"
    pairs = []
    if other_run:
        for s in (read_json(rd / "shotlist.json") or {}).get("shots", []):
            a, b = rd / "shots" / s["shot_id"] / "kf.png", P("runs") / other_run / "shots" / s["shot_id"] / "kf.png"
            if "TITLE_CARD" not in (s.get("notes") or "") and a.exists() and b.exists():
                pairs.append({"pair_id": s["shot_id"], "a": str(a), "b": str(b), "label_a": "generated", "label_b": other, "meta": {}})
    else:
        for j in (read_json(rd / "eval" / "real_pairs.json") or {}).get("pairs", []):
            sd = rd / "shots" / j["shot_id"]
            if (sd / "kf.png").exists() and (sd / "real_kf.jpg").exists():
                pairs.append({"pair_id": j["shot_id"], "a": str(sd / "kf.png"), "b": str(sd / "real_kf.jpg"), "label_a": "generated", "label_b": "real",
                              "meta": {"match": j["match"], "confidence": j["confidence"]}})
    suffix = "keyframes" if not other_run else f"keyframes_{other}"
    write_json(rd / "eval" / f"ab_manifest_{suffix}.json", {"experiment": exp, "pairs": pairs, "criteria": ["overall", "looks like real anime", "character consistency"]})
    if limit:
        pairs = random.Random(seed).sample(pairs, min(limit, len(pairs)))
    llm, rater = judge_llm()
    try:
        done = rated_pairs(db, exp, rater)
    except Exception:
        done = set()

    def one(p):
        flip = int(sha(p["pair_id"] + "kf" + str(seed), 8), 16) % 2 == 1
        first, second = (p["b"], p["a"]) if flip else (p["a"], p["b"])
        try:
            v = llm.chat([{"role": "system", "content": FRAME_SYSTEM}, {"role": "user", "content": "Image 1 = frame A, image 2 = frame B."}],
                         schema=FRAME_SCHEMA, images=[first, second], max_tokens=250, temperature=0.0, thinking=False, image_max_side=512,
                         tag=f"abkf:{p['pair_id']}")
        except Exception as e:
            log(f"ab keyframe {p['pair_id']}: {str(e)[:120]}")
            return p, None, None

        def unflip(x):
            x = str(x).strip().upper()
            if x not in ("A", "B"):
                return "tie"
            return "a" if x == ("B" if flip else "A") else "b"

        return p, {"overall": unflip(v["overall"]), "looks like real anime": unflip(v["looks_like_real_anime"]),
                   "character consistency": unflip(v["character_consistency"])}, {"reason": v.get("reason", ""), "flip": flip}

    with ThreadPoolExecutor(6) as ex:
        results = list(ex.map(one, [p for p in pairs if p["pair_id"] not in done]))
    n = 0
    for p, choices, meta in results:
        if choices:
            record_rating(db, exp, p["pair_id"], rater, choices, "generated", other, left_side="b" if meta["flip"] else "a",
                          note=meta["reason"][:300], meta=p["meta"])
            n += 1
    summ = summarize(db, exp)
    write_json(rd / "eval" / f"ab_summary_{suffix}.json", summ)
    log(f"A/B {exp}: judged {n} new pairs (total pairs {len(pairs)})")
    return summ


def segment_manifest(run: str, seg_sec: float = 20.0, other_run: str = "") -> dict:
    """Segment-level pairs with audio for the human page: the two full episodes cut at the same shot boundaries.
    Default: this run's Wan cut vs its animatic; other_run: vs the Wan cut of another run of the same shot list."""
    from src.generate.audio import timeline

    rd = P("runs") / run
    shots = (read_json(rd / "shotlist.json") or {}).get("shots", [])
    tl = timeline(shots)
    cuts, t0 = [], 0.0
    for (s, e) in tl:
        if e - t0 >= seg_sec:
            cuts.append((t0, e))
            t0 = e
    if tl and tl[-1][1] - t0 > 5:
        cuts.append((t0, tl[-1][1]))
    pairs = []
    for i, (a, b) in enumerate(cuts):
        outs = {}
        oname = other_run or "animatic"
        for name, src in (("wan", rd / "episode.mp4"), (oname, (P("runs") / other_run / "episode.mp4") if other_run else rd / "animatic" / "episode.mp4")):
            o = rd / "eval" / "segments" / f"seg{i:03d}_{name}.mp4"
            if not o.exists() and src.exists():
                o.parent.mkdir(parents=True, exist_ok=True)
                ffmpeg("-ss", f"{a:.3f}", "-i", str(src), "-t", f"{b - a:.3f}", "-c:v", "libx264", "-crf", "18", "-c:a", "aac", "-b:a", "160k", str(o))
            outs[name] = str(o)
        pairs.append({"pair_id": f"seg{i:03d}", "a": outs["wan"], "b": outs[oname], "label_a": "wan", "label_b": oname,
                      "meta": {"start": round(a, 2), "end": round(b, 2)}})
    man = {"experiment": f"{run}_segments_wan_vs_{oname}", "pairs": pairs, "criteria": CRITERIA}
    write_json(rd / "eval" / ("ab_manifest_segments.json" if not other_run else f"ab_manifest_segments_{other_run}.json"), man)
    return man


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--versus", default="animatic")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--segments", action="store_true")
    ap.add_argument("--frames", action="store_true", help="keyframe A/B: generated first frame vs real frame")
    ap.add_argument("--no-metrics", action="store_true")
    ap.add_argument("--tag", default="", help="rater-name suffix for a control pass, e.g. frames-only")
    ap.add_argument("--other-run", default="", help="with --frames / --segments: compare against another run of the same shot list")
    ap.add_argument("--precompute", default=None, help="measure shots/*/<name>.mp4 (real | animatic | clip) and exit")
    a = ap.parse_args()
    if a.precompute:
        print(precompute(a.run, a.precompute), "clips measured")
    elif a.segments:
        print(len(segment_manifest(a.run, other_run=a.other_run)["pairs"]), "segment pairs")
    elif a.frames:
        s = judge_frames(a.run, a.limit, other_run=a.other_run)
        print({k: v for k, v in s.items() if k != "pairs"})
    else:
        s = judge_pairs(a.run, a.versus, a.limit, with_metrics=not a.no_metrics, tag=a.tag)
        print({k: v for k, v in s.items() if k != "pairs"})
