"""v3 step 4 — SHOT PLAN (D-043): the movie script becomes shots with image prompts, video prompts and voice lines (DeepSeek).

  python -m src.v3.shots --run ep22_v3 --chapters 114,116,117,118
Output: runs/<run>/script/c<ch>_shots.json and runs/<run>/shotlist.json (superset of the v1/v2 shot schema, so the
video, mix and assembly code can read it).

Every shot says how its first frame is made:
  start = "panel"  the frame follows manga panel N (panel crop as layout for the image model)
          "new"    no panel shows this: the frame is drawn from the prompt alone
          "chain"  no new frame: the clip starts on the last frame of the previous clip (the picture runs on)
          "title"  title card
"""
from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor

from src.common.config import P
from src.common.io import log, read_json, write_json
from src.llm.client import writer
from src.v3.cast import canon
from src.v3.script import SHOW, cards_text, chapter_ids, reading

FRAMINGS = ["extreme close-up", "close-up", "medium close-up", "medium", "medium wide", "wide", "insert"]
MAX_CHAIN = 2            # at most this many chained clips after a drawn frame (image-to-video drifts when chained further)
MAX_SEC, MIN_SEC = 10.0, 1.0

SYS = """You are the storyboard director of the TV anime "Komi Can't Communicate". You turn scenes of a movie script into
a SHOT LIST for an automatic production line:
  - an anime IMAGE model draws the first frame of a shot from tags (optionally following the layout of a manga panel),
  - an image-to-VIDEO model animates that frame for up to 10 seconds from a short motion description,
  - a VOICE model speaks each line in the character's voice from a short acting instruction.

{show}

You get: the scenes to board (set, cast, beats with action, lines and seconds), the manga panels those beats come from
(framing, who is visible where), the locations of the episode, and the character cards.

HOW TO BOARD
1. Cover every beat in order. A beat is usually 1-3 shots; a held moment can be one long shot; a quick exchange can be
   a two-shot that holds over several lines instead of cutting on every line. Every line of the script appears in exactly
   one shot, in script order. Put a line on a shot that shows its speaker — or deliberately on the listener's reaction.
   Thoughts and narration can play over anything that fits.
2. Scene geography: open a scene (and re-open after a long stretch of close-ups) with a shot that shows the room and where
   people are. Keep everyone on the same side of frame for the whole scene: if Komi is on the left of Tadano in the wide
   shot, she stays on the left in every two-shot and looks right in her close-ups.
3. start — how the first frame is made:
   "panel": a manga panel already shows this composition (give its number in "panel"). Prefer this when the script beat
            comes from a panel and you want the same framing and staging.
   "new":   nothing in the manga shows it (establishing shots, added connective action, reactions the manga skipped).
   "chain": the picture runs on from the previous shot without a cut — same framing, same people, the action or the
            talking continues. The video model then starts from the last frame of the previous clip. Use it for long
            lines, held reactions, an action that takes more than one clip. Never as the first shot of a scene, and never
            when anything in the frame (people, framing, place) differs from the previous shot.
4. seconds: 1 to 10 per shot. Fit the lines (about 2.5 words per second plus a breath per line). A continuous take longer
   than 10 seconds is two shots, the second with start "chain".
5. visible: the people whose faces the frame shows, most prominent first, each with position (left | center | right |
   background), pose and expression in plain words. Only people from the scene's cast. For an object or text insert: [].
   THE IMAGE MODEL CAN KEEP ONLY TWO DIFFERENT NAMED PEOPLE RECOGNISABLE IN ONE FRAME. Board like limited TV animation:
   singles and two-shots carry the scene; a third person is added only in a wide or medium wide shot where figures are
   small; never more than three named people in "visible". When a manga panel shows a group, split it into a wide shot
   for the geography and singles / pairs for the faces and the lines. People who are only there as backs, shoulders or
   a blurred background are not listed in "visible" — mention them in image_sentence ("classmates in the background").
6. image_tags: 8-16 lowercase Danbooru-style tags for what the first frame shows: view (from behind, from side, profile,
   from above, looking at another, looking at viewer), poses and gestures, expressions, held objects, and anything in the
   foreground. Do NOT put character names, hair or eye colours, uniforms, framing words or background tags there: those are
   added automatically from the cards, the scene and "framing". All-ages content only.
7. image_sentence: one plain sentence, at most 25 words, that says exactly what the first frame shows.
8. motion: 1-2 sentences for the video model: what moves during the shot, in simple visible terms (limited TV animation:
   talking mouths, blinks, a head turn, a gesture, a few steps, hair or cloth moving), who is talking, and the camera
   ("static camera", "slow push-in", "slow pan right" ...). No cuts inside a shot.
9. lines: the lines spoken in this shot: speaker, kind, text (unchanged from the script), and "instruct" — an acting
   instruction for the voice model in one sentence: emotion, energy, pace, volume, any stammer or pause
   (e.g. "Flustered and too loud, words tumbling out fast, voice cracking on the name.").
10. bgm: one mood tag for music retrieval (gentle_comedy, awkward_silence, heartwarming, energetic_comedy, tense_comedic,
    melancholic, triumphant, none); change it only when the mood changes. sfx: short sound tags.
11. ALL AGES - the cast are school students and children. Slapstick aimed at someone's body (the "kancho" prank, a slap, a
    grab) is staged OFF-CONTACT: show the wind-up (the prankster crouching with a grin, a step away) or the reaction (the
    yelp, the jump, the bystander's shock) - never a hand at another person's backside, chest or thighs, in the frame
    description, the poses or the motion. No frame dwells on hips, chest or legs.

Also return, once per scene, the look every frame of that scene shares:
  "background_tags": 4-8 Danbooru-style tags for the place and light (classroom, desk, chalkboard, window, indoors, morning ...)
  "outfits": {{"<id>": "Danbooru-style clothing tags for this scene"}} for every cast member of the scene.

RETURN ONE JSON OBJECT
{{"scenes": [{{"scene_id": "...", "background_tags": "...", "outfits": {{...}},
              "shots": [{{"beats": [<beat numbers>], "start": "panel|new|chain|title", "panel": <n or null>,
                         "seconds": <number>, "framing": "{framings}", "camera": "...",
                         "visible": [{{"id": "...", "position": "...", "pose": "...", "expression": "..."}}],
                         "image_tags": "...", "image_sentence": "...", "motion": "...",
                         "lines": [{{"speaker": "...", "kind": "...", "text": "...", "instruct": "..."}}],
                         "bgm": "...", "sfx": ["..."]}}]}}]}}
The title-card beat is one shot with start "title", no people, 3 seconds."""

SCHEMA = {"type": "object", "required": ["scenes"], "properties": {"scenes": {"type": "array", "minItems": 1, "items": {
    "type": "object", "required": ["scene_id", "shots"], "properties": {"shots": {"type": "array", "minItems": 1, "items": {
        "type": "object", "required": ["start", "seconds", "framing", "visible", "image_tags", "motion", "lines"]}}}}}}}


def _chunks(scenes: list[dict], max_beats: int = 34) -> list[list[dict]]:
    out, cur, n = [], [], 0
    for s in scenes:
        nb = len(s.get("beats", []))
        if cur and n + nb > max_beats:
            out.append(cur)
            cur, n = [], 0
        cur.append(s)
        n += nb
    if cur:
        out.append(cur)
    return out


def _panel_view(r: dict, wanted: set[int]) -> list[dict]:
    return [{"n": p["n"], "framing": p.get("framing"), "people_count": p.get("people_count"),
             "visible": [f"{v.get('id')} ({v.get('where', '')}): {v.get('doing', '')}" for v in p.get("visible", [])],
             "shows": p.get("action")} for p in r.get("panels", []) if p["n"] in wanted]


def board_chunk(chapter: int, scenes: list[dict], bible: dict, r: dict, part: int) -> list[dict]:
    wanted = {int(n) for s in scenes for b in s.get("beats", []) for n in (b.get("from_panels") or []) if str(n).lstrip("-").isdigit()}
    loc_ids = {s.get("location_id") for s in scenes}
    user = ("EPISODE CONTINUITY\n" + json.dumps(bible.get("continuity", {}), ensure_ascii=False)
            + "\n\nLOCATIONS\n" + json.dumps([l for l in bible.get("locations", []) if l.get("id") in loc_ids], ensure_ascii=False)
            + "\n\nCHARACTER CARDS\n" + cards_text(chapter_ids(r))
            + "\n\nMANGA PANELS THESE SCENES DRAW ON\n" + json.dumps(_panel_view(r, wanted), ensure_ascii=False)
            + "\n\nSCENES TO BOARD\n" + json.dumps(scenes, ensure_ascii=False))
    res = writer("shots").chat([{"role": "system", "content": SYS.format(show=SHOW, framings="|".join(FRAMINGS))}, {"role": "user", "content": user}],
                               schema=SCHEMA, max_tokens=60000, thinking=True, effort="medium", tag=f"shots:c{chapter}:{part}")
    return res["scenes"]


def _words(t: str) -> int:
    return len(re.findall(r"[A-Za-z0-9']+", t or ""))


def _norm(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (t or "").lower()).strip()


def finish(chapter: int, script: dict, boarded: list[dict], r: dict) -> list[dict]:
    """Validate and normalise the model's shot list; returns shots in the pipeline schema."""
    panels = {p["n"] for p in r.get("panels", [])}
    by_scene = {s["scene_id"]: s for s in script["scenes"]}
    shots, k = [], 0
    for bs in boarded:
        sc = by_scene.get(bs.get("scene_id")) or {}
        cast = {canon(c.get("id")) for c in sc.get("cast", [])} | {"Narrator", "Classmates"}
        chain_run, first = 0, True
        for s in bs.get("shots", []):
            start = str(s.get("start") or "new").lower()
            if start not in ("panel", "new", "chain", "title"):
                start = "new"
            panel = s.get("panel")
            try:
                panel = int(panel) if panel is not None else None
            except Exception:
                panel = None
            if start == "title" and shots and shots[-1]["chapter"] == chapter:
                start = "new"                                   # one title card per segment; later cards are ordinary inserts
            if start == "panel" and panel not in panels:
                start, panel = "new", None
            if start == "chain" and (first or chain_run >= MAX_CHAIN or (shots and shots[-1]["start"] == "title")):
                start = "new"                                   # re-anchor on a drawn frame
            chain_run = chain_run + 1 if start == "chain" else 0
            lines = []
            for l in s.get("lines") or []:
                if not isinstance(l, dict) or not str(l.get("text") or "").strip():
                    continue
                kind = str(l.get("kind") or "speech").lower()
                kind = kind if kind in ("speech", "thought", "narration", "notebook", "sign") else "speech"
                lines.append({"speaker": canon(l.get("speaker")), "kind": kind, "text": str(l["text"]).strip(),
                              "instruct": str(l.get("instruct") or "").strip()})
            spoken = [l for l in lines if l["kind"] in ("speech", "thought", "narration")]
            need = sum(_words(l["text"]) / 2.6 + 0.35 for l in spoken) + (0.3 if spoken else 0.0)
            try:
                sec = float(s.get("seconds") or 3.0)
            except Exception:
                sec = 3.0
            sec = min(MAX_SEC, max(MIN_SEC, sec, need))
            framing = str(s.get("framing") or "medium").lower()
            framing = framing if framing in FRAMINGS else "medium"
            vis = [{**v, "id": canon(v["id"])} for v in (s.get("visible") or []) if isinstance(v, dict) and v.get("id")][:4]
            title = start == "title"
            shot = {
                "shot_id": f"c{chapter:03d}_s{k:03d}", "chapter": chapter, "scene_id": bs.get("scene_id"), "beats": s.get("beats") or [],
                "start": start, "panel": panel if start == "panel" else None, "duration": round(sec, 2),
                "framing": framing, "shot_size": framing, "camera": str(s.get("camera") or "static camera"),
                "visible": vis, "characters": [v["id"] for v in vis],
                "image_tags": str(s.get("image_tags") or ""), "image_sentence": str(s.get("image_sentence") or ""),
                "motion": str(s.get("motion") or ""), "action": str(s.get("motion") or ""),
                "background_tags": str(bs.get("background_tags") or ""),
                "outfits": {canon(k): str(v) for k, v in (bs.get("outfits") or {}).items()},
                "location": sc.get("location_id") or "", "slug": sc.get("slug") or "",
                "lines": lines, "bgm_cue": str(s.get("bgm") or "none"), "sfx": s.get("sfx") if isinstance(s.get("sfx"), list) else [],
                "source_panels": [panel] if start == "panel" and panel is not None else [],
                "notes": f"TITLE_CARD: {script.get('segment_title', '')}" if title else "",
                "off_cast": sorted({v["id"] for v in vis} - cast),
            }
            # the keys the mix / subtitle code reads (v1 schema)
            shot["dialogue"] = [{"speaker": l["speaker"], "kind": "sfx_text" if l["kind"] == "sign" else l["kind"], "text": l["text"],
                                 "emotion": "neutral", "instruct": l["instruct"]} for l in lines]
            shots.append(shot)
            k += 1
            first = False
    return shots


def coverage(script: dict, shots: list[dict]) -> dict:
    want = [_norm(l.get("text")) for s in script["scenes"] for b in s["beats"] for l in (b.get("lines") or []) if _norm(l.get("text"))]
    have = [_norm(l["text"]) for s in shots for l in s["lines"]]
    missing = [w for w in want if w not in have]
    beats_want = {(s["scene_id"], b.get("beat")) for s in script["scenes"] for b in s["beats"]}
    beats_have = {(s["scene_id"], b) for s in shots for b in s["beats"]}
    return {"script_lines": len(want), "lines_in_shots": len(have), "lines_missing": len(missing), "missing_examples": missing[:5],
            "beats": len(beats_want), "beats_covered": len(beats_want & beats_have)}


def plan_chapter(run: str, chapter: int, redo: bool = False) -> list[dict]:
    d = P("runs") / run / "script"
    out = d / f"c{chapter:03d}_shots.json"
    if out.exists() and not redo:
        return read_json(out)["shots"]
    script, bible, r = read_json(d / f"c{chapter:03d}.json"), read_json(d / "bible.json") or {}, reading(chapter)
    if not script:
        raise FileNotFoundError(f"no revised script for chapter {chapter} in {d}")
    parts = _chunks(script["scenes"])
    with ThreadPoolExecutor(4) as ex:
        boarded = [s for part in ex.map(lambda ip: board_chunk(chapter, ip[1], bible, r, ip[0]), list(enumerate(parts))) for s in part]
    shots = finish(chapter, script, boarded, r)
    cov = coverage(script, shots)
    write_json(out, {"chapter": chapter, "coverage": cov, "shots": shots})
    st = {k: sum(1 for s in shots if s["start"] == k) for k in ("panel", "new", "chain", "title")}
    log(f"shot plan chapter {chapter}: {len(shots)} shots, {sum(s['duration'] for s in shots):.0f} s, start {st}, "
        f"lines {cov['lines_in_shots']}/{cov['script_lines']} (missing {cov['lines_missing']}), beats {cov['beats_covered']}/{cov['beats']}")
    return shots


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--chapters", required=True)
    ap.add_argument("--redo", action="store_true")
    a = ap.parse_args()
    chs = [int(c) for c in a.chapters.split(",") if c]
    with ThreadPoolExecutor(4) as ex:
        per = list(ex.map(lambda c: plan_chapter(a.run, c, a.redo), chs))
    shots = [s for part in per for s in part]
    write_json(P("runs") / a.run / "shotlist.json", {"chapters": chs, "pipeline": "v3", "shots": shots})
    log(f"shotlist: {len(shots)} shots, {sum(s['duration'] for s in shots):.0f} s")


if __name__ == "__main__":
    main()
