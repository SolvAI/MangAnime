"""Real frames of the TRAINING episodes (1-21) as art-style references for the frame painter.

  python -m src.v3.style_bank build [--per-episode 40]     # sample + label with Gemini Flash -> data/processed/v3/style_bank.json
  python -m src.v3.style_bank sheet                        # contact sheet of what the bank holds per place

Two blind raters told the generated frames from the real ones 80 times out of 80 (RESULTS M6): backgrounds too detailed and
too sharp, glossy shading, a generic "anime" look. The fix tried here is the obvious one: show the painter what the show
looks like. For every scene of a shot list, `refs_for` returns up to two real frames of the same kind of place and light
- one that shows the place, one that shows how people are drawn in it. Held-out episodes (22-24) are never sampled.
"""
from __future__ import annotations

import argparse
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.common.config import CFG, P
from src.common.io import log, read_json, read_jsonl, write_json
from src.llm.client import vision

BANK = P("processed") / "v3" / "style_bank.json"
PLACES = ["classroom", "school hallway", "school entrance or shoe lockers", "school stairs", "school rooftop", "school grounds or gate",
          "gym or sports field", "other school room", "street", "park", "shop or cafe or restaurant", "train or station",
          "home living room", "home kitchen", "home bedroom", "home entrance or bathroom", "festival or event", "nature or outdoors", "other"]
SYS = ("You label still frames of a TV anime for an art-reference library. For each numbered frame give: place (one of the list), "
       "time (day, sunset, night, or unclear), shot_size (wide, medium wide, medium, medium close-up, close-up, insert), people (number of "
       "people clearly visible, 0-9), background (true when the real place behind the people is clearly drawn and visible), and clean "
       "(true ONLY for an ordinary scene frame in the show's normal drawing style: no captions, text boxes, lettering, speed lines, "
       "abstract or patterned background, chibi or simplified gag faces, split screen, flashback tint, title card or credits).\n"
       'Answer with one JSON object: {"frames": [{"n": 1, "place": "...", "time": "...", "shot_size": "...", "people": 0, "background": true, '
       '"clean": true}, ...]} - one entry per frame, "n" is the frame number.')
SCHEMA = {"type": "object", "required": ["frames"], "properties": {"frames": {"type": "array", "items": {
    "type": "object", "required": ["n", "place", "time", "shot_size", "people", "background", "clean"], "properties": {
        "n": {"type": "integer"}, "place": {"type": "string", "enum": PLACES}, "time": {"type": "string", "enum": ["day", "sunset", "night", "unclear"]},
        "shot_size": {"type": "string"}, "people": {"type": "integer"}, "background": {"type": "boolean"}, "clean": {"type": "boolean"}}}}}}


def build(per_episode: int = 40, batch: int = 8) -> list[dict]:
    held = set(int(e) for e in CFG.split.heldout_episodes) if "split" in CFG and "heldout_episodes" in CFG.split else {22, 23, 24}
    rows = [r for r in read_jsonl(P("processed") / "anime" / "shots.jsonl")
            if r.get("content") and int(r["episode"]) <= 21 and int(r["episode"]) not in held and r["end"] - r["start"] >= 1.5 and len(r.get("keyframes") or []) >= 2]
    picks = []
    for ep in sorted({r["episode"] for r in rows}):
        er = [r for r in rows if r["episode"] == ep]
        step = max(1, len(er) // per_episode)
        picks += [{"episode": ep, "shot": r["shot"], "path": r["keyframes"][1], "start": r["start"]} for r in er[step // 2 :: step][:per_episode]]
    picks = [p for p in picks if Path(p["path"]).exists()]
    assert all(p["episode"] <= 21 for p in picks)
    have = read_json(BANK) or []
    done = {b["path"] for b in have}
    picks = [p for p in picks if p["path"] not in done]
    log(f"style bank: {len(picks)} frames of {len({p['episode'] for p in picks})} training episodes to label ({len(have)} already labelled)")

    def one(part: list[dict]) -> list[dict]:
        content = "Places: " + "; ".join(PLACES) + f".\n{len(part)} frames follow, numbered 1-{len(part)} in order."
        try:
            r = vision("style_bank").chat([{"role": "system", "content": SYS}, {"role": "user", "content": content}], schema=SCHEMA,
                                          images=[p["path"] for p in part], max_tokens=2500, effort="low", media_resolution="medium",
                                          tag=f"style_bank:{part[0]['episode']}:{part[0]['shot']}")
        except Exception as e:                                             # noqa: BLE001
            log(f"style bank batch failed: {str(e)[:120]}")
            return []
        out = []
        for x in r.get("frames", []):
            if isinstance(x.get("n"), int) and 1 <= x["n"] <= len(part):
                out.append({**part[x["n"] - 1], **{k: x[k] for k in ("place", "time", "shot_size", "people", "background", "clean")}})
        return out

    parts = [picks[i : i + batch] for i in range(0, len(picks), batch)]
    with ThreadPoolExecutor(8) as ex:
        bank = have + [x for res in ex.map(one, parts) for x in res]
    bank.sort(key=lambda b: (b["episode"], b["shot"]))
    BANK.parent.mkdir(parents=True, exist_ok=True)
    write_json(BANK, bank)
    clean = [b for b in bank if b["clean"]]
    log(f"style bank: {len(bank)} labelled, {len(clean)} clean; places " + ", ".join(f"{p} {sum(1 for b in clean if b['place'] == p)}" for p in PLACES if any(b["place"] == p for b in clean)))
    return bank


_HOME = r"residence|apartment|\bhome\b|\bhouse\b|family"
_KEYS = [("street", r"\bEXT\b.*(street|road|sidewalk|alley|neighbou?rhood)|^.*streets? "),
         ("home kitchen", r"kitchen"), ("home bedroom", r"bedroom|\bbed\b|her room|his room"),
         ("home entrance or bathroom", rf"({_HOME}).*(entry|entrance|hallway|door|genkan|walkway)|front door|doorstep"),
         ("home living room", rf"living|kotatsu|tatami|dining|sofa|{_HOME}"),
         ("classroom", r"class\s?room|\bclass\b|homeroom|blackboard"), ("school hallway", r"hallway|corridor"),
         ("school entrance or shoe lockers", r"locker|shoe|school entrance|front gate"), ("school stairs", r"stair"),
         ("school rooftop", r"rooftop|\broof\b"), ("gym or sports field", r"\bgym|sports|track|field"),
         ("school grounds or gate", r"school (gate|yard|ground|court)|courtyard|school exterior|campus"),
         ("park", r"\bpark\b|swing|playground|bench"),
         ("shop or cafe or restaurant", r"shop|store|cafe|restaurant|mall|supermarket"), ("train or station", r"train|station|platform"),
         ("street", r"street|road|sidewalk|alley|crossing|neighbou?rhood"), ("nature or outdoors", r"river|beach|forest|mountain")]
_NEAR = {"school entrance or shoe lockers": "school hallway", "school hallway": "classroom", "school stairs": "school hallway", "home kitchen": "home living room",
         "home entrance or bathroom": "home living room", "home bedroom": "home living room", "park": "street", "other": "street"}


def screen_bank(batch: int = 10) -> int:
    """All-ages screen of the usable frames: a frame with swimwear, underwear, bathing or undress is never a reference."""
    bank = read_json(BANK) or []
    todo = [b for b in bank if b.get("clean") and "plain" not in b]
    sys_ = ("You screen reference frames for an all-ages project. List the numbers of the frames in which anyone is in swimwear, underwear, "
            "a towel, a bath or shower, partly undressed, or in which the framing dwells on hips, chest, thighs or underwear. "
            'Answer with one JSON object: {"flagged": [numbers]} - an empty list when none.')
    schema = {"type": "object", "required": ["flagged"], "properties": {"flagged": {"type": "array", "items": {"type": "integer"}}}}

    def one(part: list[dict]) -> list[dict]:
        try:
            r = vision("style_bank").chat([{"role": "system", "content": sys_}, {"role": "user", "content": f"{len(part)} frames follow, numbered 1-{len(part)}."}],
                                          schema=schema, images=[p["path"] for p in part], max_tokens=1000, effort="low", media_resolution="medium",
                                          tag=f"style_bank_screen:{part[0]['episode']}:{part[0]['shot']}")
        except Exception as e:                                             # noqa: BLE001
            log(f"style bank screen failed: {str(e)[:120]}")
            return []
        bad = {n for n in r.get("flagged", []) if isinstance(n, int)}
        return [{**p, "plain": (i + 1) not in bad} for i, p in enumerate(part)]

    parts = [todo[i : i + batch] for i in range(0, len(todo), batch)]
    with ThreadPoolExecutor(8) as ex:
        seen = {(x["episode"], x["shot"]): x["plain"] for res in ex.map(one, parts) for x in res}
    for b in bank:
        if (b["episode"], b["shot"]) in seen:
            b["plain"] = seen[(b["episode"], b["shot"])]
        elif b.get("clean") and "plain" not in b:
            b["plain"] = False                                              # not screened = not used
    write_json(BANK, bank)
    n = sum(1 for b in bank if b.get("clean") and b.get("plain") is False)
    log(f"style bank screen: {len(todo)} frames screened, {n} set aside, {sum(1 for b in bank if b.get('clean') and b.get('plain'))} usable")
    return n


def mark_letterbox() -> int:
    """Frames with black bars (flashbacks, wide-screen gags) are not what an ordinary shot looks like: never a reference."""
    import numpy as np
    from PIL import Image

    bank = read_json(BANK) or []
    for b in bank:
        if "black" in b or not Path(b["path"]).exists():
            continue
        g = np.asarray(Image.open(b["path"]).convert("L").resize((160, 90)), dtype=np.float32)
        b["letterbox"] = bool(g[:6].mean() < 14 and g[-6:].mean() < 14) or bool(g[:, :6].mean() < 14 and g[:, -6:].mean() < 14)
        b["dark"] = bool(g.mean() < 40)
        b["black"] = round(float((g < 22).mean()), 3)
    write_json(BANK, bank)
    return sum(1 for b in bank if b.get("letterbox"))


def place_of(text: str) -> str:
    for place, pat in _KEYS:
        if re.search(pat, text, flags=re.I):
            return place
    return "other"


def time_of(text: str) -> str:
    t = text.lower()
    if re.search(r"night|evening dark|after dark|streetlamp|moon", t):
        return "night"
    if re.search(r"sunset|dusk|golden|late afternoon|evening|orange light|twilight", t):
        return "sunset"
    return "day"


def refs_for(shot: dict, bank: list[dict] | None = None, k: int = 2) -> list[Path]:
    """Up to k real training frames for this shot's place and light: first one that shows the place (few people, background
    visible, wide or medium), then one that shows people drawn in it (medium or closer). Deterministic per scene."""
    bank = bank if bank is not None else (read_json(BANK) or [])
    slug = str(shot.get("slug") or "")
    desc = " ".join([slug, str(shot.get("location") or "").replace("_", " "), " ".join(map(str, shot.get("background_tags") or []))])
    place = place_of(slug + " " + str(shot.get("location") or "").replace("_", " "))
    place = place if place != "other" else place_of(desc)
    own = " ".join([" ".join(map(str, shot.get("background_tags") or [])), str(shot.get("image_sentence") or "")])
    if re.search(r"EXT\./INT\.|INT\./EXT\.", slug) and place_of(own) != "other":
        place = place_of(own)                     # a scene that moves between outside and inside: the shot itself decides
    elif re.search(r"\b(outside|exterior|outdoors|sky|school gate|front gate)\b", own, flags=re.I) and place in ("classroom", "school hallway", "school entrance or shoe lockers"):
        place = "school grounds or gate"
    tm = time_of(slug) if slug else time_of(desc)
    clean = [b for b in bank if b.get("clean") and b.get("plain") is not False and not b.get("letterbox") and b.get("everyday") is not False      # everyday: src/v3/sets.py
             and not ((b.get("dark") or b.get("black", 0) > 0.25) and tm != "night") and Path(b["path"]).exists()]
    pool = [b for b in clean if b["place"] == place] or [b for b in clean if b["place"] == _NEAR.get(place, "classroom" if "school" in desc.lower() else "street")] or clean
    if not pool:
        return []
    near = [b for b in clean if b["place"] == _NEAR.get(place)] if len(pool) < 4 else []
    key = sum(map(ord, str(shot.get("scene_id") or shot.get("shot_id") or "")))

    def best(cands: list[dict], off: int) -> dict | None:
        cands = sorted(cands, key=lambda b: (b["time"] != tm, b["episode"], b["shot"]))
        same = [b for b in cands if b["time"] == tm] or cands
        return same[(key + off) % len(same)] if same else None

    wide = lambda rows: [b for b in rows if b["background"] and b["people"] <= 2 and b["shot_size"] in ("wide", "medium wide", "medium")]      # noqa: E731
    a = best(wide(pool) or wide([b for b in clean if b["place"] == _NEAR.get(place)]) or [b for b in pool if b["background"]] or pool, 0)
    ppl = lambda rows, sizes: [b for b in rows if 1 <= b["people"] <= 3 and b["shot_size"] in sizes and b is not a]      # noqa: E731
    mid = ("medium", "medium close-up")
    b_ = best(ppl(pool, mid) or ppl(near, mid) or ppl(pool, ("close-up",)) or [b for b in pool + near if b is not a], 1)
    out = [Path(x["path"]) for x in (a, b_) if x]
    return out[:k]


def sheet() -> Path:
    from PIL import Image, ImageDraw
    bank = [b for b in (read_json(BANK) or []) if b.get("clean") and b.get("plain") is not False and not b.get("letterbox")]
    places = [p for p in PLACES if any(b["place"] == p for b in bank)]
    tw, th = 256, 144
    im = Image.new("RGB", (8 * tw, len(places) * (th + 16)), (30, 30, 30))
    d = ImageDraw.Draw(im)
    for r, p in enumerate(places):
        rows = [b for b in bank if b["place"] == p]
        d.text((4, r * (th + 16) + 2), f"{p}: {len(rows)} clean frames", fill=(255, 255, 255))
        for c, b in enumerate(rows[:: max(1, len(rows) // 8)][:8]):
            im.paste(Image.open(b["path"]).convert("RGB").resize((tw, th)), (c * tw, r * (th + 16) + 16))
    out = BANK.parent / "style_bank_sheet.jpg"
    im.save(out, quality=88)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "sheet"])
    ap.add_argument("--per-episode", type=int, default=40)
    a = ap.parse_args()
    if a.cmd == "build":
        build(a.per_episode)
        screen_bank()
        mark_letterbox()
    print(sheet())
