"""v3 step 4b — WARDROBE (D-052): one outfit per character per scene, every garment with a colour, top layer only.

Why: the continuity check (src/eval/continuity.py) found that inside a scene the same person changes clothes from shot to
shot. The shot plan named garments without a colour ("puffer jacket, knit scarf", "apron") or stacked layers ("navy blazer
... puffer jacket"), and the image model resolved that differently in every frame. A costume supervisor pass (DeepSeek) now
fixes ONE outfit string per character and scene; every shot of the scene carries it word for word.

  python -m src.v3.wardrobe --run ep22_v3 [--out ep22_v3b]      # writes runs/<out>/shotlist.json (+ wardrobe.json)
"""
from __future__ import annotations

import argparse
import collections
import json

from src.common.config import P
from src.common.io import log, read_json, write_json
from src.generate.tags import _tags
from src.llm.client import vision, writer
from src.v3.cast import card, gender
from src.v3.script import SHOW

SYS = """You are the costume supervisor of the TV anime "Komi Can't Communicate". The storyboard is finished; before the frames
are drawn you fix what everybody wears, scene by scene, so that no one changes clothes between two shots of the same scene.

{show}

You get the scenes of one chapter in story order. For every scene: the place, the background, and per character the standard
look from the character card and the outfit descriptions the storyboard used in that scene's shots (they may differ from shot
to shot, lack colours, or pile an indoor outfit under an outdoor one).

For every character of every scene write ONE outfit as Danbooru-style tags for an image model:
1. Only what is VISIBLE as the outer layer. Someone in a winter coat: the coat, scarf, gloves, skirt or trousers, legwear -
   not the blazer underneath. Someone in an apron: the apron and what shows around it.
2. EVERY garment and accessory gets a definite colour, plus pattern or material where it helps: "olive green puffer jacket",
   "beige knit scarf", "red checkered apron", "light brown cardigan". Never a bare "jacket", "scarf", "apron", "pajamas".
   Choose plain, common colours that suit the character and differ between characters who share the frame.
3. At school the uniform is fixed: girls "navy blue blazer, white shirt, red striped bow, red plaid pleated skirt" (Komi adds
   "black tights"), boys "navy blue blazer, white shirt, red striped necktie, grey trousers". A character's own way of
   wearing it ALWAYS stays and is written the same way every time: "worn_in_the_show" says what that person wears in the
   show's own character picture - a cardigan instead of the blazer, the blazer open, NO necktie and an open collar, an
   eyepatch, a headband. If it shows a long-sleeved outfit without a necktie, the school outfit has none either (write "open
   collar, no necktie"). A picture in the SUMMER uniform (short sleeves, no blazer) only tells you the accessories and
   the neckwear: in a cold-season chapter that person wears the blazer like everybody else.
   "cold_season_outdoors_in_the_show", when given, is what that person wears over the school uniform on the way to and from
   school in winter in the show itself: use it, in those words, whenever they are outdoors in uniform in a cold-season scene
   (and indoors as long as they have not taken it off). Private clothes at home or on a day out are yours to choose.
4. The same person keeps the same clothes in consecutive scenes of the same day: REPEAT YOUR OWN WORDING EXACTLY. Change it
   only when the story does (coat on to go outside, apron on to cook, changed at home).
5. A hairstyle tag only when it differs from the standard look in this scene (ponytail for cooking).
6. 4 to 9 tags per person. No pose, no expression, no hair or eye colour, nothing held in the hands, no bag.
Return ONE JSON object: {{"scenes": [{{"scene_id": "...", "outfits": {{"<character id>": "tag, tag, ..."}}}}]}}
Include every scene and every character you were given, with the ids exactly as given."""

WORN = P("processed") / "v3" / "as_worn.json"
WORN_SYS = ("You are a costume supervisor's assistant. Describe ONLY what the person in this official character picture of a TV anime "
            "wears, as short comma-separated tags with colours, outer layer first: every garment, how it is worn (blazer open or "
            "buttoned, sleeves rolled, shirt untucked), neckwear (say 'no necktie, open collar' or 'no bow' when there is none), legwear, "
            "and accessories that are always on the person (glasses, eyepatch, band-aid, headband, ribbon). No hair, eyes, face, pose or "
            'background. Answer with one JSON object: {"wears": "tag, tag, ..."}.')


def as_worn(cid: str) -> str:
    """What the character wears in the show's own picture (the anime design from the wiki, never one that debuts in a held-out
    episode), read once by the vision model and cached. Why (D-062): the costume supervisor wrote the standard uniform for
    everybody, so Katai got a necktie the show never gives him - blind raters name such details as a giveaway."""
    from src.v3.frames_gemini import character_picture

    cache = read_json(WORN) or {}
    if cid in cache:
        return cache[cid]
    pic, anime = character_picture(cid)
    worn = ""
    if pic is not None and anime:
        try:
            r = vision("wardrobe").chat([{"role": "system", "content": WORN_SYS}, {"role": "user", "content": f"The character: {cid}."}],
                                        schema={"type": "object", "required": ["wears"], "properties": {"wears": {"type": "string"}}},
                                        images=[str(pic)], max_tokens=600, effort="low", media_resolution="medium", tag=f"as_worn:{cid}")
            worn = ", ".join([w for w in _tags(str(r.get("wears") or "")) if not any(x in w.lower() for x in ("chest", "exposed", "bare ", "cleavage", "midriff"))][:12])
        except Exception as e:                                             # noqa: BLE001
            log(f"as_worn {cid}: {str(e)[:100]}")
            return ""
    cache = read_json(WORN) or {}
    cache[cid] = worn
    write_json(WORN, cache)
    return worn


WINTER = P("processed") / "v3" / "as_worn_winter.json"
OUTDOORS = ("street", "park", "train or station", "nature or outdoors", "school grounds or gate", "shop or cafe or restaurant", "festival or event",
            "school entrance or shoe lockers")


def winter_worn(cids: list[str]) -> dict[str, str]:
    """The cold-season outdoor clothes of the given characters as the show's training episodes draw them (coat, scarf, gloves,
    with colours): Gemini Flash looks at the style bank's outdoor frames beside each character's picture, a second call
    writes the consensus. Cached. Why (D-062): the costume supervisor invented a red scarf for Komi; the show gives her another."""
    from concurrent.futures import ThreadPoolExecutor

    from src.v3.frames_gemini import character_picture

    cache = read_json(WINTER) or {}
    todo = [c for c in cids if c not in cache and character_picture(c)[1]]
    if not todo:
        return {c: cache.get(c, "") for c in cids}
    bank = [b for b in (read_json(P("processed") / "v3" / "style_bank.json") or [])
            if b.get("clean") and b.get("plain") and b["place"] in OUTDOORS and 1 <= b["people"] <= 3 and int(b["episode"]) <= 21]
    pics = [str(character_picture(c)[0]) for c in todo]
    sys_ = ("You are a costume supervisor's assistant for a TV anime. First come the official pictures of " + ", ".join(todo) + " (in that order), then "
            "numbered frames of the show. For each frame say whether it is a COLD-SEASON outdoor moment (people in coats, scarves or gloves), and "
            "for each of those characters who is clearly in the frame, the outdoor clothes they wear there as short tags with colours and "
            "patterns (coat or jacket, scarf, gloves or mittens, hat, legwear), and whether they wear it WITH THE SCHOOL UNIFORM (on the way to or "
            "from school) or as private clothes. Leave a character out when they are not in the frame or you are not sure it is them. "
            'Answer with one JSON object: {"frames": [{"n": 1, "cold": true, "wears": {"<character>": {"clothes": "tag, tag", "school_uniform": true}}}]}.')
    schema = {"type": "object", "required": ["frames"], "properties": {"frames": {"type": "array", "items": {
        "type": "object", "required": ["n", "cold"], "properties": {"n": {"type": "integer"}, "cold": {"type": "boolean"}, "wears": {"type": "object"}}}}}}

    def one(part: list[dict]) -> list[dict]:
        try:
            r = vision("wardrobe").chat([{"role": "system", "content": sys_}, {"role": "user", "content": f"{len(todo)} character pictures, then {len(part)} frames numbered 1-{len(part)}."}],
                                        schema=schema, images=pics + [b["path"] for b in part], max_tokens=2500, effort="low", media_resolution="medium",
                                        tag=f"winter_worn:{part[0]['episode']}:{part[0]['shot']}")
        except Exception as e:                                             # noqa: BLE001
            log(f"winter_worn batch failed: {str(e)[:100]}")
            return []
        return [{"episode": part[x["n"] - 1]["episode"], "wears": x.get("wears") or {}} for x in r.get("frames", [])
                if isinstance(x.get("n"), int) and 1 <= x["n"] <= len(part) and x.get("cold")]

    parts = [bank[i : i + 6] for i in range(0, len(bank), 6)]
    with ThreadPoolExecutor(8) as ex:
        seen = [x for res in ex.map(one, parts) for x in res]
    for c in todo:
        notes = [f"episode {x['episode']}: {w['clothes']}" for x in seen for w in [x["wears"].get(c)]
                 if isinstance(w, dict) and w.get("school_uniform") and str(w.get("clothes") or "").strip()]
        worn = ""
        if len(notes) >= 3:
            try:
                r = writer("wardrobe").chat([{"role": "system", "content": "These notes describe what ONE anime character wears outdoors in the cold season on the way to or from "
                                              "school, over the school uniform, frame by frame; they mostly show the same outfit seen from different sides. Write "
                                              "that one usual outfit as 3 to 7 short tags with colours and patterns (coat or jacket if any, scarf, gloves, legwear), "
                                              'keeping only what most notes agree on. Answer with one JSON object: {"wears": "tag, tag, ..."}.'},
                                             {"role": "user", "content": f"{c}\n" + "\n".join(notes[:60])}],
                                            schema={"type": "object", "required": ["wears"], "properties": {"wears": {"type": "string"}}}, max_tokens=1500, tag=f"winter_worn:{c}")
                seen_kind, keep = set(), []
                for w in _tags(str(r.get("wears") or "")):                 # one scarf, one coat: the first wording of each garment wins
                    kind = next((k for k in ("scarf", "coat", "jacket", "cardigan", "gloves", "mittens", "hat", "beanie", "earmuffs", "tights") if k in w.lower()), w.lower())
                    if kind not in seen_kind:
                        seen_kind.add(kind)
                        keep.append(w)
                worn = ", ".join(keep[:7])
            except Exception as e:                                         # noqa: BLE001
                log(f"winter_worn {c}: {str(e)[:100]}")
        cache[c] = worn
        log(f"winter clothes in the show, {c}: {worn or '(not seen often enough)'} ({len(notes)} frames)")
    write_json(WINTER, cache)
    return {c: cache.get(c, "") for c in cids}


SCHEMA = {"type": "object", "required": ["scenes"], "properties": {"scenes": {"type": "array", "minItems": 1, "items": {
    "type": "object", "required": ["scene_id", "outfits"], "properties": {"scene_id": {"type": "string"}, "outfits": {"type": "object"}}}}}}


def chapter_wardrobe(chapter: int, shots: list[dict]) -> dict[str, dict[str, str]]:
    """{scene_id: {character id: outfit tags}} for one chapter."""
    scenes: dict[str, dict] = collections.OrderedDict()
    for s in shots:
        if s["chapter"] != chapter or s["start"] == "title":
            continue
        sc = scenes.setdefault(s["scene_id"], {"scene_id": s["scene_id"], "place": s.get("location"), "background": ", ".join(_tags(s.get("background_tags"))[:8]), "people": {}})
        for v in s.get("visible", []):
            cid = v["id"]
            if cid == "Classmates":
                continue
            c = card(cid) or {}
            p = sc["people"].setdefault(cid, {"id": cid, "gender": gender(cid, str(v.get("pose", ""))), "shots": 0, "storyboard_outfits": [],
                                              "standard_look": (c.get("look_anime") or c.get("look_manga") or v.get("look") or "")[:220]})
            if "worn_in_the_show" not in p and as_worn(cid):
                p["worn_in_the_show"] = as_worn(cid)
            if (read_json(WINTER) or {}).get(cid):
                p["cold_season_outdoors_in_the_show"] = (read_json(WINTER) or {})[cid]
            p["shots"] += 1
            o = str((s.get("outfits") or {}).get(cid) or "").strip()
            if o and o not in p["storyboard_outfits"]:
                p["storyboard_outfits"].append(o)
    if not scenes:
        return {}
    user = "CHAPTER %d — scenes in story order:\n%s" % (chapter, json.dumps(
        [{**sc, "people": list(sc["people"].values())} for sc in scenes.values()], ensure_ascii=False, indent=1))
    res = writer("wardrobe").chat([{"role": "system", "content": SYS.format(show=SHOW)}, {"role": "user", "content": user}],
                                  schema=SCHEMA, max_tokens=12000, thinking=True, effort="low", tag=f"wardrobe:c{chapter}")
    out: dict[str, dict[str, str]] = {}
    for sc in res.get("scenes", []):
        sid = str(sc.get("scene_id"))
        if sid not in scenes:
            continue
        out[sid] = {cid: ", ".join(_tags(o)[:9]) for cid, o in (sc.get("outfits") or {}).items() if cid in scenes[sid]["people"] and str(o).strip()}
    return out


def apply(shots: list[dict], wardrobe: dict[str, dict[str, str]]) -> tuple[list[dict], list[str]]:
    """Every shot gets the scene's outfits word for word. Returns (shots, ids of the shots whose drawn people changed outfit text)."""
    changed = []
    for s in shots:
        w = wardrobe.get(s.get("scene_id")) or {}
        if not w:
            continue
        before = {v["id"]: str((s.get("outfits") or {}).get(v["id"]) or "") for v in s.get("visible", [])}
        s["outfits"] = {**(s.get("outfits") or {}), **w}
        s["wardrobe"] = True
        if any(w.get(cid) and w[cid] != o for cid, o in before.items()):
            changed.append(s["shot_id"])
    return shots, changed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    src, dst = P("runs") / a.run, P("runs") / (a.out or a.run)
    sl = read_json(src / "shotlist.json")
    wardrobe: dict[str, dict[str, str]] = {}
    for ch in sl.get("chapters") or sorted({s["chapter"] for s in sl["shots"]}):
        w = chapter_wardrobe(ch, sl["shots"])
        wardrobe.update(w)
        log(f"wardrobe chapter {ch}: {len(w)} scenes, {sum(len(v) for v in w.values())} outfits")
    shots, changed = apply(sl["shots"], wardrobe)
    dst.mkdir(parents=True, exist_ok=True)
    write_json(dst / "wardrobe.json", {"from_run": a.run, "outfits": wardrobe, "shots_changed": changed})
    write_json(dst / "shotlist.json", {**sl, "shots": shots, "wardrobe": True})
    log(f"wardrobe: {len(changed)} of {len(shots)} shots get a different outfit text -> {dst / 'shotlist.json'}")


if __name__ == "__main__":
    main()
