"""Scene tags for the Illustrious-XL keyframe prompt (local Qwen, text only; brief: everything but director/judge is local).

  tag_shots(run_dir, shots) -> runs/<run>/kf_tags.json  {shot_id: {"visible": [names], "tags": [danbooru tags]}}
The cast, uniform, framing and quality tags are added deterministically in prompts.illustrious_prompt; the LLM only
describes pose, expression, props and background, so a bad answer cannot change who is in the frame.
"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.common.config import CFG, P
from src.common.io import log, read_json, read_jsonl, write_json

SCHEMA = {"type": "object", "required": ["visible", "tags"],
          "properties": {"visible": {"type": "array", "items": {"type": "string"}}, "tags": {"type": "array", "items": {"type": "string"}}}}

SYSTEM = """You write Danbooru-style tags for ONE frame of an all-ages school-comedy TV anime (everyday scenes only).
You get the shot description and what the source manga panel shows. Return ONE JSON object:
{"visible": [names from the given character list that are actually in the frame, most prominent first, at most 3],
 "tags": [8 to 16 lowercase tags]}
The tags describe, in this order: view (from behind / from side / profile / looking at viewer) if relevant; pose and
gesture; facial expression and gaze; objects held or shown; background and location; lighting or time of day.
Rules:
- no character names, no hair or eye colours, no shot-size tags (close-up, upper body...), no school-uniform tags: those
  are added automatically. Add clothing only when it is NOT the school uniform (apron, scarf, coat, sweater, pajamas).
- use tags that exist on Danbooru, e.g. smile, closed eyes, open mouth, blush, sweatdrop, surprised, nervous, v-shaped
  eyebrows, hand on own chin, arms at sides, waving, pointing, holding notebook, holding gift box, sitting, standing,
  walking, looking at another, classroom, desk, chalkboard, school hallway, locker, kitchen, indoors, outdoors, window,
  day, sunset, cherry blossoms, simple background.
- for an insert of an object or text, give the object tags and "no humans".
- never any suggestive or unsafe tag."""

_BAD = re.compile(r"nsfw|nude|naked|panties|underwear|cleavage|sexy|breast|ass\b|lingerie|swimsuit|bikini|thigh|skirt lift|upskirt", re.I)
_SKIP = re.compile(r"hair|eyes\b|uniform|blazer|bowtie|necktie|close-up|upper body|cowboy shot|full body|wide shot|portrait|1girl|1boy|2girls|solo", re.I)


def clean(tags: list) -> list[str]:
    out = []
    for t in tags or []:
        t = str(t).strip().lower().replace("_", " ")
        if not t or len(t) > 40 or _BAD.search(t) or _SKIP.search(t) or t in out:
            continue
        out.append(t)
    return out[:16]


def tag_shots(run_dir: Path, shots: list[dict], only: set[str] | None = None) -> dict:
    from src.llm.client import LLM

    path = run_dir / "kf_tags.json"
    store = read_json(path) or {}
    panels = {(p["chapter"], p["order"]): p for p in read_jsonl(P("processed") / "manga" / "panels.jsonl")}
    llm = LLM("local", role="kf_tags")
    todo = [s for s in shots if "TITLE_CARD" not in (s.get("notes") or "") and (s["shot_id"] not in store or (only and s["shot_id"] in only))]

    def one(s):
        pn = panels.get((s["chapter"], (s.get("source_panels") or [None])[0])) or {}
        chars = [c for c in s.get("characters", []) if c != "Narrator"]
        user = {"shot_size": s.get("shot_size"), "characters": chars, "location": s.get("location"), "time_of_day": s.get("time_of_day"),
                "first_frame": s.get("first_frame"), "action": s.get("action"), "source_panel_shows": pn.get("caption"),
                "characters_seen_in_panel": pn.get("characters")}
        try:
            import json

            r = llm.chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": json.dumps(user, ensure_ascii=False)}],
                         schema=SCHEMA, max_tokens=300, temperature=0.2, tag="kf_tags")
            vis = [c for c in r.get("visible", []) if c in chars][:3] or chars[:3]
            return s["shot_id"], {"visible": vis, "tags": clean(r.get("tags"))}
        except Exception as e:
            return s["shot_id"], {"visible": chars[:3], "tags": [], "error": str(e)[:120]}

    with ThreadPoolExecutor(int(CFG.llm.local.parallel)) as ex:
        for sid, r in ex.map(one, todo):
            store[sid] = r
    write_json(path, store)
    log(f"keyframe tags: {len(todo)} shots tagged ({sum(1 for s in todo if store[s['shot_id']].get('error'))} errors)")
    return store


# ----------------------------------------------------------------------------- v2: the VLM LOOKS at the panel
VSCHEMA = {"type": "object", "required": ["people", "n_people_total", "view", "action", "background", "situation"],
           "properties": {"people": {"type": "array", "items": {"type": "object", "required": ["name", "gender", "position", "look", "pose", "expression"],
                                                              "properties": {k: {"type": "string"} for k in ("name", "confidence", "gender", "position", "look", "pose", "expression", "clothing")}}},
                          "n_people_total": {"type": "integer"}, "view": {"type": "string"}, "action": {"type": "string"},
                          "background": {"type": "string"}, "situation": {"type": "string"}}}

VSYSTEM = """You look at ONE black-and-white manga panel (lettering removed) from "Komi Can't Communicate" and write the
prompt data for an anime image model that will redraw THIS moment as a colour TV-anime frame. You also get the director's
shot description (who is supposed to be in the shot and what happens).
- For WHAT IS VISIBLE (how many people, where, poses, objects, background) trust your eyes, not the description.
- For WHAT IS HAPPENING use the description.
- Name a person only when the drawing clearly matches the cast reference below; give "confidence": "high" only then.
  If unsure, use name "unknown" and describe the look. Never name someone just because the description lists them.
Cast reference (how they are drawn):
{roster}
Return ONE JSON object:
{{"people": [   // the people actually drawn, most prominent first, at most 4
    {{"name": "<cast name or unknown>", "confidence": "high|low", "gender": "girl|boy|woman|man",
      "position": "left|center|right|background",
      "look": "danbooru tags for what is drawn: hair length and style, dark or light hair, glasses, braids, ponytail, ahoge ...",
      "pose": "danbooru tags: standing, sitting, from behind, arms crossed, hand on own chin, pointing, holding notebook ...",
      "expression": "danbooru tags: smile, open mouth, closed eyes, blush, sweatdrop, surprised, nervous, annoyed ...",
      "clothing": "danbooru tags ONLY if it is not the standard school uniform (apron, scarf, coat, sweater, pajamas, gym uniform); else empty"}}],
 "n_people_total": <how many people are drawn in total, 0 for an object or text insert>,
 "view": "danbooru framing tags that match the panel: close-up | portrait | upper body | cowboy shot | full body | wide shot, plus from behind / from side / profile / from above / from below if true",
 "action": "4-8 danbooru tags for the interaction and objects (looking at another, handing over gift box, writing in notebook, whisking, eating ...)",
 "background": "3-6 danbooru tags (classroom, desk, chalkboard, school hallway, locker, kitchen, street, indoors, outdoors, window, day, sunset) or 'simple background'",
 "situation": "one plain English sentence, at most 22 words, saying exactly what is happening in this frame"}}
All-ages content only: never any suggestive tag."""


def _tags(x) -> list[str]:
    out = []
    for t in str(x or "").replace(";", ",").split(","):
        t = t.strip().lower().replace("_", " ")
        if t and len(t) <= 40 and not _BAD.search(t) and t not in out and t not in ("unknown", "partially visible", "none", "n/a", "medium shot"):
            out.append(t)
    return out


def tag_shots_vlm(run_dir: Path, shots: list[dict], only: set[str] | None = None) -> dict:
    """Panel-aware prompt data (D-039). Needs shots/<id>/panel_canvas.png (written by the keyframe stage / prepare_canvases)."""
    import json

    from src.anime.caption import roster_block
    from src.llm.client import LLM

    path = run_dir / "kf_tags_v2.json"
    store = read_json(path) or {}
    panels = {(p["chapter"], p["order"]): p for p in read_jsonl(P("processed") / "manga" / "panels.jsonl")}
    llm = LLM("local", role="kf_tags")
    sys_txt = VSYSTEM.format(roster=roster_block())
    todo = [s for s in shots if "TITLE_CARD" not in (s.get("notes") or "") and (s["shot_id"] not in store or (only and s["shot_id"] in only))]

    def one(s):
        img = run_dir / "shots" / s["shot_id"] / "panel_canvas.png"
        pn = panels.get((s["chapter"], (s.get("source_panels") or [None])[0])) or {}
        chars = [c for c in s.get("characters", []) if c != "Narrator"]
        user = {"director_characters": chars, "shot_size": s.get("shot_size"), "location": s.get("location"), "time_of_day": s.get("time_of_day"),
                "first_frame": s.get("first_frame"), "action": s.get("action"),
                "dialogue": [f"{d['speaker']}: {d['text']}" for d in s.get("dialogue", [])][:3]}
        if not img.exists():
            return s["shot_id"], {"no_panel": True}
        for attempt in range(2):
            try:
                r = llm.chat([{"role": "system", "content": sys_txt}, {"role": "user", "content": "Director's shot description:\n" + json.dumps(user, ensure_ascii=False)}],
                             schema=VSCHEMA, images=[str(img)], max_tokens=700, temperature=0.1, image_max_side=768, tag="kf_tags_v2")
                r["director_characters"] = chars
                r["panel_characters"] = pn.get("characters") or []          # the earlier, independent caption pass
                return s["shot_id"], r
            except Exception as e:
                err = str(e)[:120]
        return s["shot_id"], {"error": err, "director_characters": chars}

    done = 0
    with ThreadPoolExecutor(int(CFG.llm.local.parallel)) as ex:
        for sid, r in ex.map(one, todo):
            store[sid] = r
            done += 1
            if done % 40 == 0:
                write_json(path, store)
                log(f"panel-aware tags {done}/{len(todo)}")
    write_json(path, store)
    log(f"panel-aware keyframe tags: {len(todo)} shots ({sum(1 for s in todo if store[s['shot_id']].get('error'))} errors)")
    return store
