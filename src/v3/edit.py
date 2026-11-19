"""Editor pass (D-059): cut on the line, the way the show does.

  python -m src.v3.edit plan  --src ep22_v3d --dst ep22_v3e     # propose the cuts (DeepSeek), write the new shot list and run folder
  python -m src.v3.edit clips --src ep22_v3d --dst ep22_v3e     # once the source run has its clips: reuse them (trimmed where a shot was cut)

Measured on the 21 training episodes (6,618 shots): the show's median shot lasts 3.25 s (mean 4.04), 45 % of its shots are
3 s or shorter and 11 % run over 7 s; only 7 % of its subtitle lines run over a cut. The v3 shot plans hold several lines
in one two-shot instead (median 4.0 s, 32 % at 3 s or less, 16 % over 7 s). This pass takes a finished shot list WITH its
voice lines, finds the shots that hold two or more spoken lines and run long, and asks the writer model where to cut each
one at a line boundary and what the second half shows (the next speaker in a single, a listener's reaction, an insert) -
from an unused manga panel when one shows that moment. Nothing is re-voiced and the script is untouched: the first half
keeps the shot's id, picture and (trimmed) clip, the second half gets a new first frame and a new clip.
"""
from __future__ import annotations

import argparse
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.common.config import P
from src.common.io import ffmpeg, log, read_json, write_json
from src.generate.audio import GAP, LEAD, SPOKEN, TAIL
from src.llm.client import writer
from src.v3.cast import canon
from src.v3.shots import FRAMINGS

MIN_LONG = 5.0           # a shot shorter than this is left alone
RHYTHM = "median shot 3.25 s, mean 4.0 s; 45 % of shots are 3 s or shorter; only 11 % run over 7 s; 29 % of shots carry no line"

SYS = """You are the film editor of the TV anime "Komi Can't Communicate". The storyboard below is already voiced. It holds
too many lines per shot: the show itself cuts on the line ({rhythm}).

For each shot listed under SHOTS TO CUT decide whether to cut it in two at a line boundary, and if so what the SECOND half
shows. Cut when the lines belong to different speakers, when a long speech deserves a look at the listener, or when a
thought or narration line follows a spoken one. Do NOT cut a held moment whose point is that nothing moves, and do not cut
when both halves would show the same thing.

The second half is a different picture of the same moment, in the same place:
- the speaker of its first line in a single (close-up or medium close-up), or
- the listener reacting while the line plays over them, or
- an insert of the object the line is about.
It must differ from the first half in subject or clearly in framing, and should not repeat the framing of the shot that
follows. Keep everyone on their side of the frame (a person on the left of the two-shot looks right in their single).
Only people of the scene; at most three in "visible", normally one. If one of the UNUSED PANELS of the scene shows this
moment, use it: "start": "panel" with its number; otherwise "start": "new".
All ages: the cast are school students and children; no frame dwells on hips, chest or legs, nobody touches another
person's backside, chest or thighs.

Return ONE JSON object:
{{"cuts": [{{"shot": "<shot id>", "after_line": <index of the last line that stays in the first half, counting from 0>,
            "start": "panel|new", "panel": <n or null>, "framing": "{framings}", "camera": "static camera",
            "visible": [{{"id": "...", "position": "left|center|right", "pose": "...", "expression": "..."}}],
            "image_tags": "8-16 lowercase Danbooru-style tags: view, pose, expression, held objects (no names, hair, uniform or background)",
            "image_sentence": "one plain sentence, at most 25 words, that says exactly what the first frame shows",
            "motion": "1-2 sentences: what visibly moves, who is talking, the camera"}}]}}
List only the shots you cut."""

SCHEMA = {"type": "object", "required": ["cuts"], "properties": {"cuts": {"type": "array", "items": {
    "type": "object", "required": ["shot", "after_line", "start", "framing", "visible", "image_sentence", "motion"]}}}}


def _need(durs: list[float]) -> float:
    return LEAD + sum(durs) + GAP * (len(durs) - 1) + TAIL if durs else 0.0


def _fit(shot: dict, ks: list[int]) -> float:
    """Seconds a shot needs for the dialogue lines ks (indices into shot['dialogue']): the longer of the two voice takes."""
    need = 0.0
    for key in ("tts", "tts_clone"):
        need = max(need, _need([float((shot.get(key) or {}).get(f"l{k}", 0)) for k in ks if (shot.get(key) or {}).get(f"l{k}")]))
    return need


def candidates(shots: list[dict]) -> list[dict]:
    out = []
    for i, s in enumerate(shots):
        nxt = shots[i + 1] if i + 1 < len(shots) else None
        spoken = [k for k, d in enumerate(s["dialogue"]) if d["kind"] in SPOKEN and (s.get("tts") or {}).get(f"l{k}")]
        if s["start"] in ("panel", "new") and len(spoken) >= 2 and float(s["duration"]) > MIN_LONG and not (nxt and nxt["start"] == "chain"):
            out.append(s)
    return out


def propose(src: str) -> list[dict]:
    from src.v3.script import reading

    rd = P("runs") / src
    out = rd / "edit_cuts.json"
    if out.exists():
        return read_json(out)
    shots = read_json(rd / "shotlist.json")["shots"]
    cand = candidates(shots)
    by_scene: dict[str, list[dict]] = {}
    for s in cand:
        by_scene.setdefault(str(s["scene_id"]), []).append(s)
    idx = {s["shot_id"]: i for i, s in enumerate(shots)}

    def view(s: dict) -> dict:
        i = idx[s["shot_id"]]
        nb = lambda j: (f"{shots[j]['framing']}: {shots[j].get('image_sentence', '')}" if 0 <= j < len(shots) and shots[j]["scene_id"] == s["scene_id"] else None)      # noqa: E731
        return {"shot": s["shot_id"], "seconds": s["duration"], "framing": s["framing"], "shows": s.get("image_sentence"),
                "visible": [f"{v['id']} ({v.get('position', 'center')})" for v in s.get("visible", [])],
                "lines": [{"n": k, "speaker": d["speaker"], "kind": d["kind"], "text": d["text"], "seconds": (s.get("tts") or {}).get(f"l{k}")}
                          for k, d in enumerate(s["dialogue"])],
                "previous_shot": nb(i - 1), "next_shot": nb(i + 1)}

    def one(item: tuple[str, list[dict]]) -> list[dict]:
        sid, part = item
        ch = part[0]["chapter"]
        r = reading(ch) or {}
        scene_shots = [s for s in shots if str(s["scene_id"]) == sid]
        used = {s.get("panel") for s in scene_shots if s.get("panel") is not None}
        script = read_json(rd / "script" / f"c{ch:03d}.json") or {}
        sc = next((x for x in script.get("scenes", []) if str(x.get("scene_id")) == sid), {})
        wanted = {int(n) for b in sc.get("beats", []) for n in (b.get("from_panels") or []) if str(n).lstrip("-").isdigit()}
        unused = [{"n": p["n"], "framing": p.get("framing"), "visible": [f"{v.get('id')} ({v.get('where', '')}): {v.get('doing', '')}" for v in p.get("visible", [])],
                   "shows": p.get("action")} for p in r.get("panels", []) if p["n"] in wanted and p["n"] not in used]
        cast = sorted({v["id"] for s in scene_shots for v in s.get("visible", [])})
        user = (f"SCENE {sid}: {part[0].get('slug', '')}\nPEOPLE OF THE SCENE: {', '.join(cast)}\n\nUNUSED PANELS OF THE SCENE\n"
                + json.dumps(unused, ensure_ascii=False) + "\n\nSHOTS TO CUT\n" + json.dumps([view(s) for s in part], ensure_ascii=False))
        try:
            res = writer("edit").chat([{"role": "system", "content": SYS.format(rhythm=RHYTHM, framings="|".join(FRAMINGS))}, {"role": "user", "content": user}],
                                      schema=SCHEMA, max_tokens=20000, thinking=True, effort="medium", tag=f"edit:{src}:{sid}")
        except Exception as e:                                             # noqa: BLE001
            log(f"edit {sid}: {str(e)[:160]}")
            return []
        ok_panels = {p["n"] for p in unused}
        good = []
        for c in res.get("cuts", []):
            s = next((x for x in part if x["shot_id"] == c.get("shot")), None)
            try:
                j = int(c.get("after_line"))
            except Exception:
                continue
            if s is None or not (0 <= j < len(s["dialogue"]) - 1):
                continue
            a, b = list(range(0, j + 1)), list(range(j + 1, len(s["dialogue"])))
            if not any(s["dialogue"][k]["kind"] in SPOKEN for k in a) or not any(s["dialogue"][k]["kind"] in SPOKEN for k in b):
                continue                                                   # each half must carry a spoken line
            start = "panel" if str(c.get("start")).lower() == "panel" and c.get("panel") in ok_panels else "new"
            framing = str(c.get("framing") or "medium close-up").lower()
            vis = [{**v, "id": canon(v["id"])} for v in (c.get("visible") or []) if isinstance(v, dict) and v.get("id")][:3]
            vis = [v for v in vis if v["id"] in cast]
            good.append({"shot": s["shot_id"], "after_line": j, "start": start, "panel": c.get("panel") if start == "panel" else None,
                         "framing": framing if framing in FRAMINGS else "medium close-up", "camera": str(c.get("camera") or "static camera"), "visible": vis,
                         "image_tags": str(c.get("image_tags") or ""), "image_sentence": str(c.get("image_sentence") or ""), "motion": str(c.get("motion") or "")})
        return good

    with ThreadPoolExecutor(6) as ex:
        cuts = [c for part in ex.map(one, by_scene.items()) for c in part]
    write_json(out, cuts)
    log(f"edit {src}: {len(cand)} long shots with two or more spoken lines in {len(by_scene)} scenes; {len(cuts)} cut on a line")
    return cuts


def apply(src: str, dst: str) -> dict:
    """New run folder: the shot list with every cut shot as two shots, the voice files of the second halves under their new
    ids, and every finished first frame of the source run. First frames of the new halves are drawn by the frames stage."""
    rs, rd = P("runs") / src, P("runs") / dst
    shots = read_json(rs / "shotlist.json")
    if src == dst and shots.get("edited_from"):
        return {"new_shots": (read_json(rd / "edit.json") or {}).get("new_shots", [])}        # in place (pipeline stage "edit"): once only
    cuts = {c["shot"]: c for c in propose(src)}
    (rd / "shots").mkdir(parents=True, exist_ok=True)
    for name in ("script", "tts", "tts_clone"):
        if (rs / name).exists() and not (rd / name).exists():
            shutil.copytree(rs / name, rd / name)
    for name in ("wardrobe.json",):
        if (rs / name).exists():
            shutil.copyfile(rs / name, rd / name)
    new, added = [], []
    for s in shots["shots"]:
        c = cuts.get(s["shot_id"])
        if not c:
            new.append(s)
            continue
        j = c["after_line"]
        ka, kb = list(range(0, j + 1)), list(range(j + 1, len(s["dialogue"])))
        first = {**s, "dialogue": [s["dialogue"][k] for k in ka], "lines": [s["lines"][k] for k in ka] if len(s.get("lines", [])) == len(s["dialogue"]) else s.get("lines", []),
                 "tts": {f"l{k}": v for k in ka if (v := (s.get("tts") or {}).get(f"l{k}"))}, "tts_clone": {f"l{k}": v for k in ka if (v := (s.get("tts_clone") or {}).get(f"l{k}"))},
                 "duration": round(max(1.2, _fit(s, ka)), 3), "take": {"shot": s["shot_id"], "seconds": s["duration"]}, "edit": "first half"}
        first["planned_duration"] = first["duration"]
        sid2 = s["shot_id"] + "b"
        vis = c["visible"]
        second = {**{k: s[k] for k in ("chapter", "scene_id", "beats", "background_tags", "location", "slug", "bgm_cue") if k in s},
                  "shot_id": sid2, "start": c["start"], "panel": c["panel"], "framing": c["framing"], "shot_size": c["framing"], "camera": c["camera"],
                  "visible": vis, "characters": [v["id"] for v in vis], "image_tags": c["image_tags"], "image_sentence": c["image_sentence"],
                  "motion": c["motion"], "action": c["motion"],
                  "outfits": {k: v for k, v in (s.get("outfits") or {}).items()}, "wardrobe": s.get("wardrobe"),
                  "dialogue": [s["dialogue"][k] for k in kb], "lines": [s["lines"][k] for k in kb] if len(s.get("lines", [])) == len(s["dialogue"]) else [],
                  "tts": {f"l{n}": v for n, k in enumerate(kb) if (v := (s.get("tts") or {}).get(f"l{k}"))},
                  "tts_clone": {f"l{n}": v for n, k in enumerate(kb) if (v := (s.get("tts_clone") or {}).get(f"l{k}"))},
                  "sfx": [], "source_panels": [c["panel"]] if c["panel"] is not None else [], "notes": "", "off_cast": [], "edit": f"second half of {s['shot_id']}"}
        second["duration"] = round(max(1.2, _fit(s, kb)), 3)
        second["planned_duration"] = second["duration"]
        for sub in ("tts", "tts_clone"):
            for n, k in enumerate(kb):
                f = rs / sub / f"{s['shot_id']}_l{k}.wav"
                if f.exists():
                    shutil.copyfile(f, rd / sub / f"{sid2}_l{n}.wav")
        new += [first, second]
        added.append(sid2)
    write_json(rd / "shotlist.json", {**shots, "shots": new, "edited_from": src})
    # finished first frames and their records
    st = read_json(rs / "frames_state.json") or {}
    for s in new:
        a, b = rs / "shots" / s["shot_id"], rd / "shots" / s["shot_id"]
        if not a.exists() or rs == rd:
            continue
        b.mkdir(parents=True, exist_ok=True)
        for f in a.glob("*.png"):
            if not (b / f.name).exists():
                shutil.copyfile(f, b / f.name)
        for f in a.glob("kf*.json"):
            shutil.copyfile(f, b / f.name)
    if rs != rd:
        write_json(rd / "frames_state.json", {k: v for k, v in st.items()})
    write_json(rd / "edit.json", {"from": src, "cuts": len(added), "new_shots": added})
    d0 = [float(s["duration"]) for s in shots["shots"] if s["start"] != "title"]
    d1 = [float(s["duration"]) for s in new if s["start"] != "title"]
    med = lambda x: sorted(x)[len(x) // 2]          # noqa: E731
    log(f"edit {src} -> {dst}: {len(added)} shots cut in two; shots {len(d0)} -> {len(d1)}, total {sum(d0):.0f} -> {sum(d1):.0f} s, "
        f"mean {sum(d0) / len(d0):.2f} -> {sum(d1) / len(d1):.2f} s, median {med(d0):.2f} -> {med(d1):.2f} s, "
        f"3 s or shorter {sum(x <= 3 for x in d0) / len(d0):.0%} -> {sum(x <= 3 for x in d1) / len(d1):.0%}, over 7 s {sum(x > 7 for x in d0) / len(d0):.0%} -> {sum(x > 7 for x in d1) / len(d1):.0%}")
    return {"new_shots": added}


def clips(src: str, dst: str) -> None:
    """Reuse the source run's finished clips: unchanged shots as they are, the first half of a cut shot trimmed to its new length."""
    rs, rd = P("runs") / src, P("runs") / dst
    shots = read_json(rd / "shotlist.json")["shots"]
    cs = read_json(rs / "clips_state.json") or {}
    n = t = 0
    for s in shots:
        a, b = rs / "shots" / s["shot_id"] / "clip.mp4", rd / "shots" / s["shot_id"] / "clip.mp4"
        if not a.exists() or b.exists():
            continue
        b.parent.mkdir(parents=True, exist_ok=True)
        for name in ("kf.png", "kf_wide.png"):          # the frame the clip was rendered from (it may have been reframed since the plan was made)
            if (a.parent / name).exists():
                shutil.copyfile(a.parent / name, b.parent / name)
        if s.get("take"):
            ffmpeg("-i", str(a), "-t", f"{float(s['duration']):.3f}", "-c:v", "libx264", "-crf", "14", "-pix_fmt", "yuv420p", "-an", str(b))
            t += 1
        else:
            shutil.copyfile(a, b)
        n += 1
    write_json(rd / "clips_state.json", {k: v for k, v in cs.items() if any(s["shot_id"] == k for s in shots)})
    log(f"edit clips {src} -> {dst}: {n} clips reused ({t} trimmed); {sum(1 for s in shots if not (rd / 'shots' / s['shot_id'] / 'clip.mp4').exists())} still to render")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "clips"])
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    a = ap.parse_args()
    apply(a.src, a.dst) if a.cmd == "plan" else clips(a.src, a.dst)
