"""Set references (D-062): real frames of the school's rooms, so that a scene set in the classroom is drawn in THE classroom.

  python -m src.v3.sets label        # one more label pass over the school frames of the style bank (Gemini Flash)
  python -m src.v3.sets sheet        # runs/style_test/set_plates.jpg: the frames that can serve as set references, per place

Blind raters name "stock sets" first when they tell a generated frame from the show (RESULTS M6, ep22_v3e): a pale classroom
with a wall clock where the show has a teal blackboard, salmon curtains, grey-green lockers and blooming windows. The style
references of D-057 could not fix that: the painter is told NOT to take the place from them, and the two frames a classroom
scene got did not show the room anyway. The school is one building in every episode, so its rooms can be shown: this module
marks, among the style bank's frames of school places (training episodes only), those that show the room itself on an
ordinary school day, and picks up to two per shot - different sides of the room - as SET REFERENCE for the painter
(src/v3/frames_gemini.py, on with KOMI_SET_REFS=<n> or llm.gemini_image.set_refs). Homes, streets and shops are not
handled: nothing says that a home in a new chapter is a home the show has drawn before.
"""
from __future__ import annotations

import argparse
import collections
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageDraw

from src.common.config import P
from src.common.io import log, read_json, write_json
from src.llm.client import vision
from src.v3.style_bank import BANK, place_of, time_of

SCHOOL = ["classroom", "school hallway", "school entrance or shoe lockers", "school stairs", "school rooftop", "school grounds or gate"]
SIDES = ["blackboard wall", "window side", "corridor side or doors", "back wall or lockers", "floor or desks from above", "whole room", "outside view", "other"]
SYS = ("You sort still frames of a TV anime set in a high school; they will serve as set references for background painters. "
       "For each numbered frame give: room (0-3: how much of the room or place itself the frame shows - 0 nothing but a person or a "
       "plain surface, 1 a little, 2 a good part of the room is clearly visible, 3 the room is the subject of the frame), side (which "
       "part of the place is behind the people or in view: one of the list), and everyday (true when this is the place on an ordinary "
       "school day: usual uniforms or gym clothes, no festival decoration, stage, costumes, party, and no special-effect or symbolic "
       "background).\n"
       'Answer with one JSON object: {"frames": [{"n": 1, "room": 2, "side": "...", "everyday": true}, ...]} - one entry per frame.')
SCHEMA = {"type": "object", "required": ["frames"], "properties": {"frames": {"type": "array", "items": {
    "type": "object", "required": ["n", "room", "side", "everyday"], "properties": {
        "n": {"type": "integer"}, "room": {"type": "integer"}, "side": {"type": "string", "enum": SIDES}, "everyday": {"type": "boolean"}}}}}}


def _usable(b: dict) -> bool:
    return bool(b.get("clean") and b.get("plain") and not b.get("letterbox") and not b.get("dark") and b.get("black", 0) <= 0.25 and Path(b["path"]).exists())


def label(batch: int = 8) -> int:
    bank = read_json(BANK) or []
    assert all(int(b["episode"]) <= 21 for b in bank), "held-out episodes must never be in the bank"
    todo = [b for b in bank if b["place"] in SCHOOL and _usable(b) and "room" not in b]

    def one(part: list[dict]) -> list[dict]:
        try:
            r = vision("sets").chat([{"role": "system", "content": SYS},
                                     {"role": "user", "content": "Sides: " + "; ".join(SIDES) + f".\n{len(part)} frames follow, numbered 1-{len(part)} in order."}],
                                    schema=SCHEMA, images=[p["path"] for p in part], max_tokens=2000, effort="low", media_resolution="medium",
                                    tag=f"sets:{part[0]['episode']}:{part[0]['shot']}")
        except Exception as e:                                             # noqa: BLE001
            log(f"sets batch failed: {str(e)[:120]}")
            return []
        return [{**part[x["n"] - 1], "room": int(x["room"]), "side": x["side"], "everyday": bool(x["everyday"])}
                for x in r.get("frames", []) if isinstance(x.get("n"), int) and 1 <= x["n"] <= len(part)]

    parts = [todo[i : i + batch] for i in range(0, len(todo), batch)]
    with ThreadPoolExecutor(8) as ex:
        seen = {(x["episode"], x["shot"]): x for res in ex.map(one, parts) for x in res}
    for b in bank:
        x = seen.get((b["episode"], b["shot"]))
        if x:
            b.update(room=x["room"], side=x["side"], everyday=x["everyday"])
    write_json(BANK, bank)
    pl = plates(bank)
    log(f"sets: {len(todo)} school frames looked at, {len(seen)} labelled; set references: "
        + ", ".join(f"{p} {sum(1 for b in pl if b['place'] == p)}" for p in SCHOOL))
    return len(seen)


def plates(bank: list[dict] | None = None) -> list[dict]:
    """Frames that show a school room well enough to stand for it."""
    bank = bank if bank is not None else (read_json(BANK) or [])
    return [b for b in bank if b["place"] in SCHOOL and _usable(b) and b.get("everyday") and int(b.get("room") or 0) >= 2 and b["people"] <= 4]


def place_for(shot: dict) -> str:
    slug = str(shot.get("slug") or "")
    place = place_of(slug + " " + str(shot.get("location") or "").replace("_", " "))
    own = " ".join([" ".join(map(str, shot.get("background_tags") or [])), str(shot.get("image_sentence") or "")])
    if place == "other":
        place = place_of(" ".join([slug, own]))
    # a shot of a scene "in the classroom" that looks at something else (the school from outside, a memory of another place)
    if place in SCHOOL and place != "school grounds or gate" and re.search(r"\b(outside|exterior|outdoors|sky|rooftop|school gate|front gate|street|veranda|snow|garden|house|home|kitchen|bedroom)\b", own, flags=re.I):
        return "elsewhere"
    return place


def plates_for(shot: dict, bank: list[dict] | None = None, k: int = 2) -> list[Path]:
    """Up to k set references for a shot in a school place, of different sides of the room, the same per scene."""
    place = place_for(shot)
    if place not in SCHOOL:
        return []
    pool = [b for b in plates(bank) if b["place"] == place]
    if not pool:
        return []
    tm = time_of(str(shot.get("slug") or ""))
    # the room at its plainest first: room is the subject, few people, the scene's time of day when the bank has it
    pool.sort(key=lambda b: (b["time"] != tm and b["time"] != "day", -int(b["room"]), b["people"], b["episode"], b["shot"]))
    key = sum(map(ord, str(shot.get("scene_id") or shot.get("shot_id") or "")))
    by_side: dict[str, list[dict]] = collections.OrderedDict()
    for b in pool:
        by_side.setdefault(b["side"], []).append(b)
    out = []
    for i, rows in enumerate(by_side.values()):
        top = rows[:3]                                       # among the three best of that side, by scene
        out.append(top[(key + i) % len(top)])
        if len(out) >= k:
            break
    if len(out) < k:
        out += [b for b in pool if b not in out][: k - len(out)]
    return [Path(b["path"]) for b in out[:k]]


BIBLE = P("processed") / "v3" / "set_bible.json"
BIBLE_SYS = ("You are the art director of a TV anime. The frames show ONE recurring set of the series (named below) in several shots. "
             "Write the set note that background painters work from, so that the place looks the same in every shot: in 70 to 100 words, "
             "the fixed things and their exact colours - walls (and any coloured band or panelling), floor, ceiling and lamps, windows "
             "and curtains, blackboard or boards, doors, lockers or shelves, furniture - and how the place is lit by day. Only what the "
             "frames show; no people, no story, no writing on boards. Name colours plainly (e.g. 'pale grey-white walls', 'teal-green "
             'blackboard\'). Answer with one JSON object: {"note": "..."}.')


def bible(place: str) -> str:
    """The set note of a school place: its fixed features and colours in words, written once from its best frames and repeated
    to the painter with every shot - the rater's second complaint after 'stock rooms' was that floor, lockers and wall colour
    change from shot to shot, and that the pink of the curtains spreads to the walls."""
    cache = read_json(BIBLE) or {}
    if place in cache:
        return cache[place]
    pool = sorted([b for b in plates() if b["place"] == place and b["time"] in ("day", "unclear")], key=lambda b: (-int(b["room"]), b["people"], b["episode"], b["shot"]))
    by_side: dict[str, list[dict]] = collections.OrderedDict()
    for b in pool:
        by_side.setdefault(b["side"], []).append(b)
    pick = [rows[i] for i in range(3) for rows in by_side.values() if i < len(rows)][:8]          # the best of every side first
    if len(pick) < 2:
        return ""
    try:
        r = vision("sets").chat([{"role": "system", "content": BIBLE_SYS}, {"role": "user", "content": f"The set: the {place} of the high school. {len(pick)} frames follow."}],
                                schema={"type": "object", "required": ["note"], "properties": {"note": {"type": "string"}}},
                                images=[b["path"] for b in pick], max_tokens=1500, effort="medium", media_resolution="high", tag=f"set_bible:{place}")
    except Exception as e:                                             # noqa: BLE001
        log(f"set bible {place}: {str(e)[:120]}")
        return ""
    cache = read_json(BIBLE) or {}
    cache[place] = " ".join(str(r.get("note") or "").split())
    write_json(BIBLE, cache)
    return cache[place]


def sheet() -> Path:
    pl = plates()
    w, h = 320, 180
    rows = []
    for p in SCHOOL:
        part = sorted([b for b in pl if b["place"] == p], key=lambda b: (b["side"], -int(b["room"]), b["people"]))[:24]
        rows += [(p, b) for b in part]
    cols = 8
    im = Image.new("RGB", (cols * w, h * -(-len(rows) // cols)), (20, 20, 20))
    d = ImageDraw.Draw(im)
    for i, (p, b) in enumerate(rows):
        x, y = (i % cols) * w, (i // cols) * h
        im.paste(Image.open(b["path"]).convert("RGB").resize((w, h)), (x, y))
        d.rectangle((x, y, x + w, y + 13), fill=(0, 0, 0))
        d.text((x + 3, y + 1), f"{p[:22]} | {b['side'][:20]} | room {b['room']} | ep{b['episode']}", fill=(255, 255, 0))
    out = P("runs") / "style_test" / "set_plates.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    im.save(out, quality=86)
    log(f"sets: {len(rows)} set references on {out}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["label", "sheet", "bible"])
    a = ap.parse_args()
    if a.cmd == "bible":
        for p in SCHOOL:
            log(f"set note, {p}: {bible(p) or '(too few frames)'}")
    else:
        label() if a.cmd == "label" else sheet()
