"""Continuity between shots: does a run of consecutive shots read as the same scene with the same people?

  python -m src.eval.continuity --run ep22_v3            # a generated run: middle frame of every clip (first frame if no clip)
  python -m src.eval.continuity --real 22                # the real episode: middle frame of every detected shot (content only)
The visual model (Gemini, D-044) sees WINDOW consecutive shots, one frame each, in story order, and does not know whether
they are real or generated. Like a script supervisor it marks every cut as "same scene" or "scene change" and lists the
continuity errors inside a scene: a person whose design or outfit changes, a place that changes, lighting / time of day
that changes, a drawing style that changes. The same procedure on the real episode gives the reference level.
Writes runs/<run>/eval/continuity.json (or runs/eval_real/continuity_ep<NN>.json).
"""
from __future__ import annotations

import argparse
import collections
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.common.config import P
from src.common.io import ffmpeg, log, media_duration, read_json, write_json
from src.llm.client import encode_image, vision

KINDS = ["character design", "outfit", "location", "lighting", "art style", "other"]
SCHEMA = {"type": "object", "required": ["cuts", "errors", "score"], "properties": {
    "cuts": {"type": "array", "items": {"type": "object", "required": ["from", "to", "same_scene"], "properties": {
        "from": {"type": "integer"}, "to": {"type": "integer"}, "same_scene": {"type": "boolean"}}}},
    "people": {"type": "array", "items": {"type": "object", "properties": {
        "who": {"type": "string"}, "frames": {"type": "array", "items": {"type": "integer"}}, "consistent": {"type": "boolean"}}}},
    "errors": {"type": "array", "items": {"type": "object", "required": ["frames", "kind", "note"], "properties": {
        "frames": {"type": "array", "items": {"type": "integer"}}, "kind": {"type": "string"}, "note": {"type": "string"}}}},
    "score": {"type": "integer"}}}

SYSTEM = """You are the script supervisor of a TV anime. You get consecutive shots of one episode in story order, one frame per
shot ("Frame 1" is the earliest). Your job is continuity.
1. For every cut (frame k to frame k+1) decide whether both frames belong to the SAME SCENE (same place, same moment of the
   story) or whether the story moved to another scene (a new place or time, a flashback, an imagined image, a title card).
   A close-up, an insert of an object or a reverse angle is still the same scene.
2. List the people you can follow across frames ("girl with long black hair and a red bow"), the frames each appears in, and
   whether that person looks the same in all of them.
3. List CONTINUITY ERRORS between frames of the same scene only. Kinds:
   "character design" - the same person has a different hair colour, hair style, eye colour, age or face from one frame to another;
   "outfit"           - the same person wears different clothes without a scene change;
   "location"         - the room or place changes although it is the same scene (other furniture, other walls, indoors/outdoors);
   "lighting"         - time of day or colour of the light jumps inside a scene;
   "art style"        - a frame is drawn in a clearly different style from its neighbours;
   "other"            - anything else a script supervisor would flag (a prop that vanishes, a person who appears from nowhere).
   Deliberate anime devices are NOT errors: simplified / chibi faces for comedy, abstract or patterned backgrounds in reaction
   shots, speed lines, a change of framing, an insert shot. Do not report differences between different scenes.
   Name the two frames that contradict each other, e.g. [3, 5].
4. Give one overall continuity score from 1 (unwatchable: people and places change from shot to shot) to 10 (flawless).
Return ONE JSON object:
{"cuts": [{"from": 1, "to": 2, "same_scene": true}], "people": [{"who": "...", "frames": [1, 2], "consistent": true}],
 "errors": [{"frames": [3, 5], "kind": "outfit", "note": "one short sentence"}], "score": 8}"""


def mid_frame(video: Path, out: Path) -> Path | None:
    if out.exists():
        return out
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        ffmpeg("-ss", f"{media_duration(video) / 2:.3f}", "-i", str(video), "-frames:v", "1", "-vf", "scale=768:-2:flags=lanczos", "-q:v", "3", str(out))
        return out
    except Exception:
        return None


def frames_of_run(run: str) -> list[dict]:
    rd = P("runs") / run
    out = []
    for s in (read_json(rd / "shotlist.json") or {}).get("shots", []):
        if "TITLE_CARD" in (s.get("notes") or ""):
            continue
        sd = rd / "shots" / s["shot_id"]
        f = mid_frame(sd / "clip.mp4", rd / "eval" / "continuity_frames" / f"{s['shot_id']}.jpg") if (sd / "clip.mp4").exists() else None
        if f is None and (sd / "kf.png").exists():
            f = sd / "kf.png"
        if f is not None:
            out.append({"id": s["shot_id"], "group": s["chapter"], "path": str(f), "scene": s.get("scene_id") or s.get("location")})
    return out


def frames_of_real(episode: int) -> list[dict]:
    from src.anime.oped import is_content, non_content

    nc = non_content(episode)
    kdir = P("interim") / "anime" / f"ep{episode:02d}" / "keyframes"
    segs = [s for s in read_json(P("processed") / "align" / "chapter_segments.json", []) if s.get("episode") == episode]
    out = []
    for s in read_json(P("interim") / "anime" / f"ep{episode:02d}" / "shots_raw.json") or []:
        f = kdir / f"s{s['shot']:04d}_m.jpg"
        if not f.exists() or s["end"] - s["start"] < 0.5 or not is_content(episode, s["start"], s["end"], nc):
            continue
        ch = next((g["chapter"] for g in segs if s["start"] >= g["start"] - 0.5 and s["end"] <= g["end"] + 0.5), None)
        if ch is None:
            continue
        out.append({"id": f"s{s['shot']:04d}", "group": ch, "path": str(f)})
    return out


def windows(frames: list[dict], n: int) -> list[list[dict]]:
    """Consecutive runs of n shots inside one chapter (a shorter tail joins the window before it when it is under 4 shots)."""
    out: list[list[dict]] = []
    by = collections.OrderedDict()
    for f in frames:
        by.setdefault(f["group"], []).append(f)
    for fs in by.values():
        parts = [fs[i : i + n] for i in range(0, len(fs), n)]
        if len(parts) > 1 and len(parts[-1]) < 4:
            tail = parts.pop()                       # (not `parts[-2] += parts.pop()`: that writes to the wrong window, or fails with two parts)
            parts[-1] = parts[-1] + tail
        out += [p for p in parts if len(p) >= 2]
    return out


def judge(win: list[dict], tag: str) -> dict | None:
    content: list[dict] = []
    for k, f in enumerate(win):
        content.append({"type": "text", "text": f"Frame {k + 1}:"})
        content.append({"type": "image_url", "image_url": {"url": encode_image(f["path"], 768, 85)}})
    content.append({"type": "text", "text": f"{len(win)} frames, {len(win) - 1} cuts. Return the JSON object."})
    try:
        r = vision("continuity").chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}], schema=SCHEMA,
                                      max_tokens=6000, effort="low", media_resolution="medium", tag=tag)
    except Exception as e:
        log(f"continuity {tag}: {str(e)[:140]}")
        return None
    n = len(win)
    cuts = {(c["from"], c["to"]): bool(c["same_scene"]) for c in r.get("cuts", []) if isinstance(c.get("from"), int) and c.get("to") == c.get("from") + 1 and 1 <= c["from"] < n}
    errs = []
    for e in r.get("errors", []):
        fr = sorted({int(x) for x in (e.get("frames") or []) if isinstance(x, int) and 1 <= x <= n})
        if len(fr) < 2:
            continue
        kind = e.get("kind") if e.get("kind") in KINDS else "other"
        errs.append({"frames": fr, "shots": [win[i - 1]["id"] for i in fr], "kind": kind, "note": str(e.get("note", ""))[:200]})
    people = [p for p in r.get("people", []) if isinstance(p, dict) and len(p.get("frames") or []) >= 2]
    same = [cuts.get((k, k + 1), True) for k in range(1, n)]
    scripted = [win[k - 1].get("scene") is not None and win[k - 1].get("scene") == win[k].get("scene") for k in range(1, n)]   # the script says: same scene
    return {"shots": [f["id"] for f in win], "n_cuts": n - 1, "same_scene_cuts": sum(same), "same_scene": same,
            "scripted_same_scene_cuts": sum(scripted), "scripted_same_seen_same": sum(1 for a, b in zip(scripted, same) if a and b),
            "errors": errs, "score": r.get("score"), "people_followed": len(people), "people_inconsistent": sum(1 for p in people if p.get("consistent") is False)}


def evaluate(frames: list[dict], out: Path, label: str, window: int = 8) -> dict:
    wins = windows(frames, window)
    with ThreadPoolExecutor(6) as ex:
        res = [r for r in ex.map(lambda iw: judge(iw[1], f"continuity:{label}:{iw[0]}"), enumerate(wins)) if r]
    cuts = sum(r["n_cuts"] for r in res)
    same = sum(r["same_scene_cuts"] for r in res)
    kinds = collections.Counter(e["kind"] for r in res for e in r["errors"])
    scores = [r["score"] for r in res if isinstance(r.get("score"), (int, float))]
    pf, pi = sum(r["people_followed"] for r in res), sum(r["people_inconsistent"] for r in res)
    sc, ss = sum(r["scripted_same_scene_cuts"] for r in res), sum(r["scripted_same_seen_same"] for r in res)
    summ = {"label": label, "shots": len(frames), "windows": len(res), "cuts": cuts, "same_scene_cuts": same,
            "same_scene_share": round(same / max(1, cuts), 3),
            **({"scripted_same_scene_cuts": sc, "of_those_seen_as_same_scene": ss, "scripted_scene_held_share": round(ss / sc, 3)} if sc else {}),
            "errors": sum(kinds.values()), "errors_per_100_same_scene_cuts": round(100 * sum(kinds.values()) / max(1, same), 1),
            "errors_by_kind": dict(kinds.most_common()),
            "errors_by_kind_per_100_same_scene_cuts": {k: round(100 * v / max(1, same), 1) for k, v in kinds.most_common()},
            "windows_without_error": sum(1 for r in res if not r["errors"]), "mean_score": round(sum(scores) / max(1, len(scores)), 2),
            "people_followed": pf, "people_inconsistent": pi, "people_inconsistent_share": round(pi / max(1, pf), 3)}
    write_json(out, {"summary": summ, "windows": res})
    log(f"continuity {label}: {same}/{cuts} cuts seen as the same scene" + (f" (scripted same-scene cuts held: {ss}/{sc})" if sc else "") + "; "
        f"{summ['errors']} errors on {same} same-scene cuts ({summ['errors_per_100_same_scene_cuts']}/100), "
        f"score {summ['mean_score']}, people inconsistent {pi}/{pf}; by kind {summ['errors_by_kind']}")
    return summ


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="")
    ap.add_argument("--real", type=int, default=0, help="episode number of the real anime")
    ap.add_argument("--window", type=int, default=8)
    a = ap.parse_args()
    if a.real:
        evaluate(frames_of_real(a.real), P("runs") / "eval_real" / f"continuity_ep{a.real:02d}.json", f"real_ep{a.real:02d}", a.window)
    if a.run:
        evaluate(frames_of_run(a.run), P("runs") / a.run / "eval" / "continuity.json", a.run, a.window)
