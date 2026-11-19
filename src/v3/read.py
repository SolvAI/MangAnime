"""v3 step 1 — READING (D-043): Gemini reads a whole manga chapter and reports what happens, panel by panel.

One call per chapter: the cast reference pictures (character cards) + every page in reading order, each panel outlined
with a number. Because the model sees the whole chapter and real pictures of the cast, a person keeps one identity across
pages, which the old page-by-page pass could not do.

  python -m src.v3.read --chapters 114,116,117,118 [--effort medium] [--redo]
Output: data/processed/v3/reading/c<chapter>.json
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from src.common.config import CFG, P
from src.common.io import log, read_json, read_jsonl, write_json
from src.llm.client import encode_image, vision
from src.v3.cast import canon

OUT = P("processed") / "v3" / "reading"
CARDS = P("processed") / "cards"
KINDS = ["speech", "thought", "narration", "notebook", "sfx", "sign"]

SCHEMA = {
    "type": "object", "required": ["summary", "characters", "scenes", "panels"],
    "properties": {
        "title": {"type": "string"}, "summary": {"type": "string"},
        "characters": {"type": "array", "items": {"type": "object", "required": ["id"]}},
        "scenes": {"type": "array", "items": {"type": "object", "required": ["scene", "location", "what_happens"]}},
        "panels": {"type": "array", "items": {"type": "object", "required": ["n", "scene", "visible", "action", "lines"]}},
    },
}

SYSTEM = """You are the script supervisor of an anime studio. You read one chapter of the manga "Komi Can't Communicate"
(English edition) so that writers can adapt it. Be exact: the writers never see the pages, only your report.

WHAT YOU GET
1. CAST REFERENCE: for each known character a picture (manga drawing on the left, anime design on the right, the name
   printed underneath) and a short description.
2. THE CHAPTER: its pages in order. It is a Japanese manga: read each page right-to-left, top-to-bottom. Every panel is
   outlined in red and carries a red number in its top-left corner; the numbers follow the reading order.

HOW TO IDENTIFY PEOPLE
- Compare the drawing with the reference pictures: hair shape and tone, fringe, accessories, glasses, height, build,
  uniform details. Use the story as well: who was just addressed by name, who is answering, who the scene is about.
- The same person keeps the same id for the whole chapter. Simplified / chibi comedy faces are the same people.
- Someone who matches no reference gets an id such as "girl_A", "boy_B", "teacher_A" with a description, and that same id
  every time they appear. Do NOT turn an unknown classmate into a main character because they stand in the same room.
- Background crowds are "crowd".

HOW TO READ TEXT
- Copy every text exactly as printed (keep the wording; normal sentence case is fine instead of all capitals).
- kind: speech (balloon with a tail), thought (cloud / borderless inner voice), narration (box; the narrator or a character's
  inner narration), notebook (words Komi or someone wrote on paper, a board or a phone and shows), sfx (sound-effect
  lettering), sign (labels, signs, captions that are part of the scene).
- speaker: decide from the balloon tail, the position, who is addressed and how the conversation flows. Komi almost never
  speaks aloud: her spoken lines are stammers or single words; she writes in her notebook instead. Explanations about
  "people with a communication disorder" are the Narrator.
- If you are not sure who speaks, give your best reading and set "sure": false.

RETURN ONE JSON OBJECT
{"title": "chapter title as printed, or empty",
 "summary": "4-6 sentences: what the chapter is about, start to end",
 "characters": [{"id": "<cast name or girl_A ...>", "known": true|false,
                 "seen_as": "what they look like in THIS chapter (clothes, hair, anything unusual)",
                 "how_identified": "short reason", "sure": true|false}],
 "scenes": [{"scene": 1, "location": "specific place", "time_of_day": "...", "present": ["ids"],
             "kind": "story|front_matter|preview"  (covers, contents pages, author notes and next-chapter teasers are not story),
             "what_happens": "2-3 sentences", "mood": "few words", "first_panel": <n>, "last_panel": <n>}],
 "panels": [{"n": <panel number>, "scene": <scene number>,
             "framing": "extreme close-up|close-up|medium close-up|medium|medium wide|wide|insert",
             "visible": [{"id": "...", "where": "left|center|right|background", "doing": "pose, gesture, expression"}],
             "people_count": <how many people are drawn, 0 for an object or text panel>,
             "action": "one sentence: what this panel shows and what changes since the previous one",
             "lines": [{"speaker": "<id|Narrator|crowd>", "kind": "<kind>", "text": "...", "to": "<id or empty>",
                        "delivery": "how it is said, 2-5 words", "sure": true|false}]}]}
Cover EVERY numbered panel exactly once, in order, and every text on the pages."""


def cards() -> list[dict]:
    return read_json(P("processed") / "cards" / "cards.json") or []


def cast_for(chapter: int, max_n: int = 64) -> list[dict]:
    """Cards shown to the reader: main / recurring / family cast that exists by this chapter (plus minor ones while room is left)."""
    rank = {"main": 0, "recurring": 1, "family": 2, "minor": 3, "other": 4}
    out = []
    for c in cards():
        fc = c.get("first_chapter")
        if isinstance(fc, int) and fc > chapter:
            continue
        if not (c.get("ref_sheet") or c.get("look_manga") or c.get("look_anime")):
            continue
        out.append(c)
    out.sort(key=lambda c: (rank.get(c.get("role"), 5), c.get("first_chapter") or 999))
    return out[:max_n]


def roster_fallback() -> list[dict]:
    """Before the cards exist: the hand-written roster (text only)."""
    return [{"id": c["name"], "full_name": c["full"], "look_manga": c["manga"], "look_anime": c["look"], "role": c["role"]}
            for c in CFG.characters.roster if c["role"] not in ("crowd", "voice_only")]


def chapter_pages(chapter: int) -> list[dict]:
    """[{volume, page, path, panels:[panel rows]}] in reading order."""
    rows = sorted((p for p in read_jsonl(P("processed") / "manga" / "panels.jsonl") if p["chapter"] == chapter), key=lambda p: p["order"])
    paths = {(r["volume"], r["page"]): r["path"] for r in read_jsonl(P("interim") / "manga" / "pages.jsonl")}
    pages: dict[tuple, dict] = {}
    for p in rows:
        k = (p["volume"], p["page"])
        pages.setdefault(k, {"volume": k[0], "page": k[1], "path": paths[k], "panels": []})["panels"].append(p)
    return [pages[k] for k in sorted(pages)]


def mark_page(pg: dict, scale: float = 1.5) -> Path:
    out = P("cache") / "v3_marked" / f"v{pg['volume']:03d}_p{pg['page']:03d}_{len(pg['panels'])}_{pg['panels'][0]['order']}.jpg"
    if out.exists():
        return out
    img = Image.open(pg["path"]).convert("RGB")
    img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
    d = ImageDraw.Draw(img)
    fp = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    font = ImageFont.truetype(fp, 30) if Path(fp).exists() else ImageFont.load_default()
    for p in pg["panels"]:
        x, y, w, h = [v * scale for v in p["bbox"]]
        d.rectangle([x, y, x + w, y + h], outline=(230, 0, 0), width=3)
        label = str(p["order"])
        tw = d.textlength(label, font=font)
        d.rectangle([x + 3, y + 3, x + 13 + tw, y + 39], fill=(255, 255, 255), outline=(230, 0, 0), width=2)
        d.text((x + 8, y + 4), label, fill=(230, 0, 0), font=font)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out, quality=90)
    return out


def _img(path: str | Path, side: int) -> dict:
    return {"type": "image_url", "image_url": {"url": encode_image(path, side, 88)}}


def cast_picture(c: dict) -> Path | None:
    """The card's reference sheet (manga | anime picture) if it is on disk. Card paths are relative to the cards folder."""
    ref = c.get("ref_sheet")
    return CARDS / ref if ref and (CARDS / ref).exists() else None


def build_messages(chapter: int, pages: list[dict], cast: list[dict], pictures: bool = True) -> list[dict]:
    content: list[dict] = [{"type": "text", "text": "CAST REFERENCE"}]
    for c in cast:
        desc = f"{c['id']} ({c.get('full_name', '')}) — in the manga: {c.get('look_manga') or 'n/a'}. In the anime: {c.get('look_anime') or 'n/a'}."
        content.append({"type": "text", "text": desc})
        pic = cast_picture(c) if pictures else None
        if pic:
            content.append(_img(pic, 768))
    n_panels = sum(len(pg["panels"]) for pg in pages)
    first, last = pages[0]["panels"][0]["order"], pages[-1]["panels"][-1]["order"]
    content.append({"type": "text", "text": f"THE CHAPTER — chapter {chapter}: {len(pages)} pages, panels {first} to {last} ({n_panels} panels)."})
    for i, pg in enumerate(pages):
        nums = ", ".join(str(p["order"]) for p in pg["panels"])
        content.append({"type": "text", "text": f"Page {i + 1} of {len(pages)} (panels {nums}):"})
        content.append(_img(mark_page(pg), 1800))
    content.append({"type": "text", "text": "Now write the report as ONE JSON object. Every panel from "
                                            f"{first} to {last} must appear exactly once in \"panels\"."})
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]


def normalise(chapter: int, res: dict, pages: list[dict], cast: list[dict]) -> dict:
    valid = {p["order"]: (pg["volume"], pg["page"]) for pg in pages for p in pg["panels"]}
    known = {c["id"] for c in cast}
    got = {}
    for p in res.get("panels", []):
        try:
            n = int(p.get("n"))
        except Exception:
            continue
        if n not in valid or n in got:
            continue
        p["n"] = n
        p["visible"] = [{**v, "id": canon(v["id"])} for v in (p.get("visible") or []) if isinstance(v, dict) and v.get("id")]
        lines = []
        for l in p.get("lines") or []:
            if not isinstance(l, dict) or not str(l.get("text") or "").strip():
                continue
            l["kind"] = l.get("kind") if l.get("kind") in KINDS else "speech"
            l["speaker"] = canon(l.get("speaker") or "unknown")
            if l.get("to"):
                l["to"] = canon(l["to"])
            l["sure"] = bool(l.get("sure", True))
            lines.append(l)
        p["lines"] = lines
        p["volume"], p["page"] = valid[n]
        got[n] = p
    res["panels"] = [got[n] for n in sorted(got)]
    res["chapter"] = chapter
    res["missing_panels"] = sorted(set(valid) - set(got))
    res["cast_shown"] = sorted(known)
    for c in res.get("characters", []):
        c["id"] = canon(c.get("id"))
        c["known"] = c.get("id") in known
    for sc in res.get("scenes", []):
        sc["present"] = [canon(x) for x in (sc.get("present") or [])]
    return res


def read_chapter(chapter: int, *, effort: str = "medium", redo: bool = False, pictures: bool = True, out_dir: Path | None = None) -> dict:
    out = (out_dir or OUT) / f"c{chapter:03d}.json"
    if out.exists() and not redo:
        return read_json(out)
    pages = chapter_pages(chapter)
    if not pages:
        raise ValueError(f"chapter {chapter} has no parsed panels")
    cast = cast_for(chapter) or roster_fallback()
    msgs = build_messages(chapter, pages, cast, pictures)
    llm = vision("read")
    res = llm.chat(msgs, schema=SCHEMA, max_tokens=60000, effort=effort, media_resolution="high", tag=f"read:c{chapter}")
    res = normalise(chapter, res, pages, cast)
    res["cast_pictures"] = sum(1 for c in cast if pictures and cast_picture(c))          # pictures really attached to the request
    res["with_reference_pictures"] = res["cast_pictures"] > 0
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, res)
    log(f"read chapter {chapter}: {len(res['panels'])} panels ({len(res['missing_panels'])} missing), {len(res.get('scenes', []))} scenes, "
        f"{sum(len(p['lines']) for p in res['panels'])} lines, {len(res.get('characters', []))} characters")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chapters", required=True)
    ap.add_argument("--effort", default="medium", choices=["low", "medium", "high"])
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--no-pictures", action="store_true", help="cast described in words only (to measure what the pictures add)")
    ap.add_argument("--out", default="", help="write to data/processed/v3/<out>/ instead of reading/")
    a = ap.parse_args()
    chs = [int(c) for c in a.chapters.split(",") if c]
    out_dir = OUT.parent / a.out if a.out else None
    with ThreadPoolExecutor(4) as ex:
        list(ex.map(lambda c: read_chapter(c, effort=a.effort, redo=a.redo, pictures=not a.no_pictures, out_dir=out_dir), chs))


if __name__ == "__main__":
    main()
