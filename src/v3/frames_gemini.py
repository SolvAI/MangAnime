"""v3 step 5a, second backend — FIRST FRAMES drawn by Gemini's image model from reference pictures (D-053).

Why: frames with two or three named people were the weakest measured step (Illustrious: 34 of 117 two-person frames accepted,
side characters off-model). On 24 of those rejected frames the image model, given one picture per character and the manga
panel as layout, was accepted 24 times out of 24 by the same check (RESULTS M6).

What the model gets for one shot:
  - the shot in words: what happens, framing, place, and per person position / pose / expression / the scene's fixed outfit
    (src/v3/wardrobe.py);
  - one reference picture per named character (the card's anime picture; only the manga picture plus colours in words for
    characters who first appear in a held-out episode, D-050);
  - for a shot that starts from a manga panel: the panel with its balloons removed, as layout reference;
  - for every shot but the first of its scene: the scene's first accepted frame, so the room, the light and the clothes stay.
Each frame is then checked by src/v3/frames.check_frame (Gemini 3.8 Flash against the cast pictures); a rejected frame is
drawn once more with the check's remarks. State file and file names are the same as the Illustrious backend's.

  python -m src.v3.frames_gemini --run ep22_v3b [--only c116_s020,c116_s021] [--rounds 1]
"""
from __future__ import annotations

import argparse
import base64
import collections
import io
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
from PIL import Image

from src.common.config import CFG, P
from src.common.io import append_jsonl, log, read_json, write_json
from src.generate.tags import _tags
from src.llm.client import COST_LOG, api_spend
from src.v3.cast import card
from src.v3.frames import _panels, check_frame

CARDS = P("processed") / "cards"
CLOSE_CROP = {"medium close-up": 0.55, "close-up": 0.42, "extreme close-up": 0.32, "insert": 0.4}     # share of the scene frame handed to a close shot
ANCHOR_ORDER = {"wide": 0, "medium wide": 1, "medium": 2, "medium close-up": 3, "close-up": 4, "extreme close-up": 5, "insert": 6}


def _cfg():
    return CFG.llm.gemini_image


def _b64(img: Image.Image, side: int, quality: int = 88) -> str:
    img = img.convert("RGB")
    img.thumbnail((side, side), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def _flat(path: Path) -> Image.Image:
    im = Image.open(path)
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, "white")
        bg.alpha_composite(im)
        im = bg
    return im.convert("RGB")


def character_picture(cid: str) -> tuple[Path | None, bool]:
    """(picture, is it the anime design). Anime pictures of held-out-debut characters are never used."""
    c = card(cid) or {}
    if c.get("img_anime") and not c.get("img_anime_heldout") and (CARDS / c["img_anime"]).exists():
        return CARDS / c["img_anime"], True
    if c.get("img_manga") and (CARDS / c["img_manga"]).exists():
        return CARDS / c["img_manga"], False
    return None, False


def panel_reference(run_dir: Path, shot: dict, panel: dict | None) -> Path | None:
    """The manga panel on a 16:9 canvas with its speech balloons painted out (same preparation as the Illustrious backend)."""
    if shot["start"] != "panel" or panel is None:
        return None
    from src.generate.keyframes import compose_cover, prep_panel, remove_balloons

    sdir = run_dir / "shots" / shot["shot_id"]
    out = sdir / "panel_clean.png"
    if not out.exists():
        prep_panel(panel["image"], sdir / "panel.png")
        src = compose_cover(panel, sdir / "panel_canvas.png")
        remove_balloons(src, out)
    return out


def style_references(shot: dict) -> list[Path]:
    """Real frames of the training episodes as art-style references (src/v3/style_bank.py); on with llm.gemini_image.style_refs
    or KOMI_STYLE_REFS=<how many>, off by default so that earlier runs stay reproducible."""
    n = int(os.environ.get("KOMI_STYLE_REFS") or _cfg().get("style_refs", 0) or 0)
    if n <= 0:
        return []
    from src.v3.style_bank import refs_for

    return refs_for(shot, k=n)


def set_references(shot: dict) -> list[Path]:
    """Real frames of the training episodes that show the school room this shot is set in (src/v3/sets.py, D-062); on with
    llm.gemini_image.set_refs or KOMI_SET_REFS=<how many>."""
    n = int(os.environ.get("KOMI_SET_REFS") or _cfg().get("set_refs", 0) or 0)
    if n <= 0:
        return []
    from src.v3.sets import plates_for

    return plates_for(shot, k=n)


def build_parts(shot: dict, panel_ref: Path | None, scene_ref: Path | None, hint: str = "", match_ref: Path | None = None) -> tuple[list[dict], dict]:
    vis = [v for v in shot.get("visible", []) if v["id"] != "Classmates"][:4]
    crowd = len(vis) != len(shot.get("visible", []))
    who = ("exactly %d %s" % (len(vis), "person" if len(vis) == 1 else "people")) if vis else "nobody (no person in the frame)"
    if crowd:
        who += ", plus unnamed students in school uniform in the background" if vis else " in particular: unnamed students in school uniform"
    elif vis:
        who += " - nobody else in the frame"
    style = style_references(shot)
    sets = set_references(shot)
    if sets and len(style) > 1:
        style = style[1:]                 # the set references show the place; of the style references keep the one that shows people
    lines = ["Draw ONE frame of the TV anime \"Komi Can't Communicate\" as a finished full-colour anime screenshot, 16:9: clean line art, flat cel "
             "shading, painted background, the look of the anime reference pictures. All-ages school comedy.",
             f"WHAT THE FRAME SHOWS: {shot.get('image_sentence', '')}",
             f"FRAMING: {shot.get('framing')}" + (f", camera {shot['camera']}" if shot.get("camera") else "") + f". PLACE: {', '.join(_tags(shot.get('background_tags'))[:8])}.",
             f"PEOPLE: {who}."]
    for v in vis:
        outfit = (shot.get("outfits") or {}).get(v["id"]) or ""
        lines.append(f"- {v['id']} ({v.get('position', 'center')}): {v.get('pose', '')}; {v.get('expression', '')}." + (f" Wears: {outfit}." if outfit else ""))
    from src.v3.staging import camera_line

    cam = camera_line(shot)
    if cam:
        lines.insert(3, f"CAMERA POSITION: {cam} This is a decision of the director of photography: follow it even where the layout reference "
                        "shows the moment from another side.")
    parts: list[dict] = [{"text": "\n".join(lines)}]
    used = []
    for v in vis:
        c = card(v["id"]) or {}
        pic, anime = character_picture(v["id"])
        if pic is None:
            parts.append({"text": f"CHARACTER {v['id']}: no picture. Looks: {c.get('look_anime') or v.get('look') or 'as described above'}."})
            continue
        note = "" if anime else f" (black-and-white manga drawing; colours: {c.get('danbooru_tags') or c.get('look_anime') or 'natural'})"
        parts.append({"text": f"CHARACTER REFERENCE - {v['id']}{note}. Keep this face, hair style and hair colour; the clothes are the ones listed above."})
        parts.append({"inlineData": {"mimeType": "image/jpeg", "data": _b64(_flat(pic), 768)}})
        used.append(str(pic))
    if style:
        lines[0] = ("Draw ONE frame of the TV anime \"Komi Can't Communicate\", 16:9, so that it could be cut into the broadcast episode unnoticed. "
                    "The ART STYLE REFERENCE frames below are real frames of the show: match their look exactly - thin even outlines, flat two-tone "
                    "cel shading without gradients or glossy highlights, soft pastel and slightly washed-out colours, simple faces (small mouth, no "
                    "drawn teeth, light blush only when the script asks for it), and backgrounds painted simply and softly, with LESS detail and "
                    "contrast than the characters, often a little out of focus behind them. Not a detailed illustration, not photorealistic, no "
                    "dramatic lighting. All-ages school comedy.")
        if os.environ.get("KOMI_STYLE_WORDING") == "2":     # wording after the look statistics (edge density 0.29 against the show's 0.18)
            lines[0] = ("Draw ONE frame of the TV anime \"Komi Can't Communicate\", 16:9, so that it could be cut into the broadcast episode unnoticed. "
                        "The ART STYLE REFERENCE frames below are real frames of the show; match their look exactly. The show is drawn with VERY FEW "
                        "LINES: thin outlines in a dark tint of the local colour (never heavy black), hair as large flat shapes with two or three "
                        "strand lines at most, clothes almost without fold lines, faces of a few strokes (small mouth, no drawn teeth, blush only "
                        "when the script asks). Shading is one flat shadow tone, no gradients, no gloss. Colours are soft, pastel, slightly washed "
                        "out. BACKGROUNDS are simple colour shapes with hardly any outlines or texture, and they are OUT OF FOCUS behind the people "
                        "(strongly so in medium and close shots); windows and lamps bloom softly. Draw less: fewer objects, fewer lines, fewer "
                        "details than you would by default. Not a detailed illustration, not photorealistic. All-ages school comedy.")
        parts[0] = {"text": "\n".join(lines)}
        parts.append({"text": f"ART STYLE REFERENCE - {len(style)} real frame{'s' if len(style) > 1 else ''} of the show from other episodes, same kind of place. "
                              "Copy the drawing style, the colour palette, the amount of background detail and softness. Do NOT copy the people, their "
                              "clothes, the poses, the composition, the time of day or any writing in them, and do not move the frame to their "
                              "location: the place of THIS frame is the one named under PLACE."})
        for f in style:
            parts.append({"inlineData": {"mimeType": "image/jpeg", "data": _b64(Image.open(f), 768)}})
    if sets:
        from src.v3.sets import bible, place_for

        note = bible(place_for(shot))
        parts.append({"text": (f"THE SET, as the show's art direction fixes it - the same in every shot of every scene: {note}\n" if note else "")
                              + f"SET REFERENCE - {len(sets)} real frame{'s' if len(sets) > 1 else ''} of the show that show the actual {place_for(shot)} of the "
                              "series, from other episodes (the school is the same building in every episode). THIS frame happens in that very place: "
                              "take its architecture and furniture - walls, blackboard, windows and curtains, lockers, doors, floor, desks - with "
                              "their colours, and the soft bloom of the light through the windows; show the part of it that lies behind the people "
                              "of this shot, from this shot's camera angle. Do NOT copy the people in these frames, their poses, the composition "
                              "or any writing; the time of day and the weather are those of THIS frame."})
        for f in sets:
            parts.append({"inlineData": {"mimeType": "image/jpeg", "data": _b64(Image.open(f), 768)}})
    if scene_ref is not None and scene_ref.exists():
        ref_img = Image.open(scene_ref).convert("RGB")
        frac = CLOSE_CROP.get(shot.get("framing"))
        if frac:
            # a close shot sees only a part of the place: hand over that part, enlarged, instead of the whole view (the model
            # otherwise repeats the reference's background, lamp for lamp, behind every close-up)
            pos = str((vis[0].get("position") if vis else "center") or "center").lower()
            cx = {"left": 0.3, "right": 0.7}.get(pos, 0.5)
            W0, H0 = ref_img.size
            w, h = int(W0 * frac), int(H0 * frac)
            x0 = int(min(max(cx * W0 - w / 2, 0), W0 - w))
            y0 = int(min(max(0.42 * H0 - h / 2, 0), H0 - h))
            ref_img = ref_img.crop((x0, y0, x0 + w, y0 + h)).resize((W0, H0), Image.LANCZOS)
            parts.append({"text": "PLACE REFERENCE - a part of the place where this scene happens, enlarged: this is roughly what lies behind the subject of "
                                  "this close shot. Keep its light, colours and time of day; it may be soft or out of focus. The subject fills the frame. "
                                  "People who also appear in other shots of the scene keep the clothes listed above."})
        else:
            parts.append({"text": "SCENE REFERENCE - an earlier frame of the SAME scene. Keep the same room or place, the same furniture, light and time of "
                                  "day, and for people who appear in both frames the same clothes and colours. Do NOT copy its composition, its poses or "
                                  "the people who are not listed above. The camera is somewhere else now: show the part of that place that lies behind "
                                  "the people of THIS shot, from this shot's angle, not the same view again."})
        if shot.get("scene_note"):
            parts[-1]["text"] += f" THE SET OF THIS SCENE, fixed by its first frame - the same in every shot: {shot['scene_note']}"
        parts.append({"inlineData": {"mimeType": "image/jpeg", "data": _b64(ref_img, 896)}})
    if match_ref is not None and match_ref.exists():
        parts.append({"text": "CONTINUITY REFERENCE - another shot of this scene, a moment apart. A script supervisor found that this frame contradicted "
                              "it. People who appear in both must look and be dressed exactly as here, and the place must be the same place."})
        parts.append({"inlineData": {"mimeType": "image/jpeg", "data": _b64(Image.open(match_ref), 896)}})
    if panel_ref is not None and panel_ref.exists():
        parts.append({"text": ("LAYOUT REFERENCE - the manga panel this frame adapts. Use it for who is there, who stands on which side, the poses and the "
                               "expressions; the camera position is the one given under CAMERA POSITION. " if cam else
                               "LAYOUT REFERENCE - the manga panel this frame adapts. Use it for the composition, who stands where and the poses. ")
                              + "Do not copy lettering, speech balloons, sound effects, screentone or the panel border; do not keep it black-and-white."})
        parts.append({"inlineData": {"mimeType": "image/jpeg", "data": _b64(_flat(panel_ref), 1024)}})
    notes = [str(d.get("text") or "").strip() for d in shot.get("dialogue", []) if d.get("kind") == "notebook" and str(d.get("text") or "").strip()]
    written = ""
    if notes and any(w in str(shot.get("image_sentence", "")).lower() for w in ("notebook", "note", "writes", "written", "page", "letter")):
        written = f"The handwritten text that can be read in the frame is in ENGLISH and says: \"{notes[0][:120]}\". "
    tail = (written + "Output: the single frame. No speech balloons, no captions, no title or name lettering, no sound-effect lettering, no border, no split panels, no watermark; "
            "no Japanese writing anywhere (signs and papers are blank or in English). All-ages: nobody touches another person's backside, chest "
            "or thighs, and the framing does not dwell on them.")
    if hint:
        tail = f"A first attempt was rejected for: {hint.strip()[:300]}. Fix exactly that.\n" + tail
    parts.append({"text": tail})
    return parts, {"prompt": "\n".join(lines) + ("\nFIX: " + hint if hint else ""), "character_pictures": used, "style_references": [str(f) for f in style], "set_references": [str(f) for f in sets],
                   "scene_reference": str(scene_ref) if scene_ref else None, "panel_reference": str(panel_ref) if panel_ref else None,
                   "continuity_reference": str(match_ref) if match_ref else None}


def draw_frame(run_dir: Path, shot: dict, panel: dict | None, *, attempt: int = 0, scene_ref: Path | None = None, hint: str = "",
               match_ref: Path | None = None) -> Path | None:
    from src.generate.keyframes import KF_H, KF_W

    sdir = run_dir / "shots" / shot["shot_id"]
    sdir.mkdir(parents=True, exist_ok=True)
    out = sdir / f"kf_a{attempt}.png"
    if out.exists() and (sdir / f"kf_a{attempt}.json").exists():
        return out
    cfg = _cfg()
    if scene_ref is not None and "scene_note" not in shot:
        note = (read_json(run_dir / "scene_notes.json") or {}).get(str(shot.get("scene_id")))
        if note:
            shot = {**shot, "scene_note": note}
    parts, meta = build_parts(shot, panel_reference(run_dir, shot, panel), scene_ref, hint, match_ref)
    body = {"contents": [{"role": "user", "parts": parts}],
            "generationConfig": {"responseModalities": ["IMAGE"], "imageConfig": {"aspectRatio": "16:9", "imageSize": str(cfg.image_size)}}}
    url = f"{CFG.llm.gemini.base_url}/models/{cfg.model}:generateContent"
    headers = {"Content-Type": "application/json", "x-goog-api-key": os.environ["GEMINI_API_KEY"]}
    cap = min(float(CFG.budget.api_usd), float(CFG.budget.api_usd_effective))
    if api_spend() >= cap * float(CFG.budget.api_soft_fraction):
        raise RuntimeError(f"API budget: ${api_spend():.2f} spent, soft limit ${cap * float(CFG.budget.api_soft_fraction):.0f}")
    for k in range(6):
        try:
            r = httpx.post(url, json=body, headers=headers, timeout=300)
        except httpx.HTTPError as e:
            log(f"gemini frame {shot['shot_id']}: {str(e)[:100]}")
            time.sleep(min(60, 5 * 2**k))
            continue
        if r.status_code in (429, 500, 502, 503, 504, 529):
            time.sleep(min(90, 5 * 2**k))
            continue
        if r.status_code >= 400:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
        d = r.json()
        u = d.get("usageMetadata") or {}
        # interim "thought images" are not charged (image-generation docs); text and image output are both counted at the image rate here
        cost = (u.get("promptTokenCount", 0) * float(cfg.price_in) + u.get("candidatesTokenCount", 0) * float(cfg.price_out)) / 1e6
        append_jsonl(COST_LOG(), {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "endpoint": "gemini", "model": cfg.model, "role": "frame_image",
                                  "tag": f"frame:{shot['shot_id']}:a{attempt}", "prompt_tokens": u.get("promptTokenCount", 0),
                                  "completion_tokens": u.get("candidatesTokenCount", 0), "cost_usd": round(cost, 6)})
        cands = d.get("candidates") or []
        imgs = [p["inlineData"]["data"] for p in ((cands[0].get("content") or {}).get("parts") or []) if p.get("inlineData") and not p.get("thought")] if cands else []
        if not imgs:
            why = (d.get("promptFeedback") or {}).get("blockReason") or (cands[0].get("finishReason") if cands else "no candidates")
            log(f"gemini frame {shot['shot_id']} a{attempt}: no image ({why})")
            if k >= 1:
                return None
            continue
        im = Image.open(io.BytesIO(base64.b64decode(imgs[-1]))).convert("RGB")
        w, h = im.size
        t = KF_W / KF_H
        if w / h > t:
            x = int((w - h * t) / 2)
            im = im.crop((x, 0, w - x, h))
        elif w / h < t:
            y = int((h - w / t) / 2)
            im = im.crop((0, y, w, h - y))
        im.resize((KF_W, KF_H), Image.LANCZOS).save(out)
        write_json(sdir / f"kf_a{attempt}.json", {**meta, "backend": "gemini", "model": cfg.model, "attempt": attempt, "start": shot["start"],
                                                   "panel": shot.get("panel"), "native_size": [w, h], "cost_usd": round(cost, 4)})
        return out
    return None


def run_frames_gemini(run_dir: Path, shots: list[dict], *, rounds: int = 1, only: set[str] | None = None, threads: int = 6) -> dict:
    """Scene by scene: the scene's first frame, then every other frame with that one as scene reference; check; one redraw."""
    todo = [s for s in shots if s["start"] in ("panel", "new") and (only is None or s["shot_id"] in only)]
    state = read_json(run_dir / "frames_state.json") or {}
    panels = {ch: _panels(ch) for ch in {s["chapter"] for s in todo}}
    by_scene: dict[str, list[dict]] = collections.OrderedDict()
    for s in shots:
        if s["start"] in ("panel", "new"):
            by_scene.setdefault(s["scene_id"], []).append(s)

    def anchor_of(sc: list[dict]) -> dict:
        """The frame the rest of the scene is matched to: an early shot that shows the place (widest framing among the first five)."""
        return min(sc[:5], key=lambda s: (ANCHOR_ORDER.get(s.get("framing"), 3), sc.index(s)))

    anchors = {sid: anchor_of(sc) for sid, sc in by_scene.items()}

    def kf_of(shot: dict) -> Path:
        return run_dir / "shots" / shot["shot_id"] / "kf.png"

    def one(shot: dict, attempt: int, hint: str = "") -> dict | None:
        a = anchors[shot["scene_id"]]
        ref = kf_of(a) if a["shot_id"] != shot["shot_id"] and kf_of(a).exists() else None
        try:
            f = draw_frame(run_dir, shot, panels[shot["chapter"]].get(shot.get("panel")) if shot["start"] == "panel" else None, attempt=attempt, scene_ref=ref, hint=hint)
        except Exception as e:                                             # noqa: BLE001
            log(f"gemini frame {shot['shot_id']} failed: {str(e)[:200]}")
            return None
        if f is None:
            return None
        return check_frame(run_dir, shot, attempt)

    def record(shot: dict, attempt: int, c: dict | None) -> None:
        st = state.setdefault(shot["shot_id"], {"attempt": -1, "checks": []})
        st["attempt"] = max(st["attempt"], attempt)
        if c is None:
            return
        st["checks"] = [x for x in st["checks"] if x.get("attempt") != attempt] + [
            {k: c.get(k) for k in ("attempt", "verdict", "score", "people_drawn", "problems", "unsuitable", "extra_people")}]
        st["hint"] = "; ".join(c.get("problems") or [])[:300] if c.get("verdict") == "redo" else ""
        good = [x for x in st["checks"] if not x.get("unsuitable")]
        if good:
            best = max(good, key=lambda x: (x.get("verdict") == "ok", x.get("score") or 0, -x["attempt"]))
            st["final_attempt"], st["final_verdict"], st["final_score"] = best["attempt"], best.get("verdict"), best.get("score")
            shutil.copyfile(run_dir / "shots" / shot["shot_id"] / f"kf_a{best['attempt']}.png", kf_of(shot))

    def batch(jobs: list[tuple[dict, int, str]], label: str) -> None:
        if not jobs:
            return
        t0 = time.time()
        with ThreadPoolExecutor(threads) as ex:
            res = list(ex.map(lambda j: one(*j), jobs))
        for (shot, attempt, _), c in zip(jobs, res):
            record(shot, attempt, c)
        write_json(run_dir / "frames_state.json", state)
        ok = sum(1 for (s, _, _) in jobs if (state.get(s["shot_id"]) or {}).get("final_verdict") == "ok")
        log(f"gemini frames, {label}: {len(jobs)} drawn in {(time.time() - t0) / 60:.1f} min, {ok} accepted; API spend ${api_spend():.2f}")

    need = lambda s: not (state.get(s["shot_id"]) or {}).get("final_verdict") or not kf_of(s).exists()      # noqa: E731
    wanted = {s["shot_id"] for s in todo}
    first = [a for a in anchors.values() if need(a) and (a["shot_id"] in wanted or any(x["shot_id"] in wanted for x in by_scene[a["scene_id"]]))]
    batch([(s, 0, "") for s in first], "scene anchors")
    for rnd in range(rounds):                                               # an anchor must be right before the scene is matched to it
        redo = [a for a in first if (state.get(a["shot_id"]) or {}).get("final_verdict") == "redo" and state[a["shot_id"]]["attempt"] == rnd]
        batch([(s, rnd + 1, state[s["shot_id"]].get("hint", "")) for s in redo], f"scene anchors, redraw {rnd + 1}")
    scene_notes(run_dir, {sid: a for sid, a in anchors.items() if kf_of(a).exists()})
    rest = [s for s in todo if need(s)]
    batch([(s, 0, "") for s in rest], "other frames")
    for rnd in range(rounds):
        redo = [s for s in rest if (state.get(s["shot_id"]) or {}).get("final_verdict") == "redo" and state[s["shot_id"]]["attempt"] == rnd]
        batch([(s, rnd + 1, state[s["shot_id"]].get("hint", "")) for s in redo], f"other frames, redraw {rnd + 1}")
    fin = [state[s["shot_id"]] for s in todo if (state.get(s["shot_id"]) or {}).get("final_verdict")]
    log(f"gemini frames: {sum(1 for x in fin if x['final_verdict'] == 'ok')}/{len(fin)} accepted "
        f"(first attempt {sum(1 for x in fin if x['checks'] and x['checks'][0].get('verdict') == 'ok')})")
    return state


NOTE_SYS = ("You are the art director of a TV anime. This finished frame opens a scene; every other shot of the scene must show the "
            "same place. Write the set note the background painters work from: in 50 to 80 words, the fixed things of the place and "
            "their exact colours - walls, floor or ground, furniture and its position, windows, doors, large objects, trees or "
            "buildings behind - and the light and time of day. No people, no clothes, no story. Name colours plainly. "
            'Answer with one JSON object: {"note": "..."}.')


def scene_notes(run_dir: Path, anchors: dict[str, dict]) -> dict:
    """A written set note per scene from its first accepted frame (D-064), for scenes outside the school (those have the
    show's own set notes, src/v3/sets.py): raters saw furniture, floors and playground equipment change between the shots of
    one scene although every shot gets the first frame as picture. On with KOMI_SCENE_NOTES=1 or llm.gemini_image.scene_notes."""
    if not int(os.environ.get("KOMI_SCENE_NOTES") or _cfg().get("scene_notes", 0) or 0):
        return {}
    from src.llm.client import vision
    from src.v3.sets import SCHOOL, place_for

    notes = read_json(run_dir / "scene_notes.json") or {}
    todo = [(str(sid), a) for sid, a in anchors.items() if str(sid) not in notes and place_for(a) not in SCHOOL]

    def one(item: tuple[str, dict]) -> tuple[str, str]:
        sid, a = item
        try:
            r = vision("sets").chat([{"role": "system", "content": NOTE_SYS}, {"role": "user", "content": f"The scene: {a.get('slug') or a.get('location') or ''}."}],
                                    schema={"type": "object", "required": ["note"], "properties": {"note": {"type": "string"}}},
                                    images=[str(run_dir / "shots" / a["shot_id"] / "kf.png")], max_tokens=1200, effort="low", media_resolution="high",
                                    tag=f"scene_note:{run_dir.name}:{sid}")
            return sid, " ".join(str(r.get("note") or "").split())
        except Exception as e:                                             # noqa: BLE001
            log(f"scene note {sid}: {str(e)[:100]}")
            return sid, ""

    with ThreadPoolExecutor(6) as ex:
        for sid, note in ex.map(one, todo):
            if note:
                notes[sid] = note
    if todo:
        write_json(run_dir / "scene_notes.json", notes)
        log(f"scene notes: {len(notes)} scenes outside the school have a written set note")
    return notes


def scene_anchors(shots: list[dict]) -> dict[str, dict]:
    by_scene: dict[str, list[dict]] = collections.OrderedDict()
    for s in shots:
        if s["start"] in ("panel", "new"):
            by_scene.setdefault(s["scene_id"], []).append(s)
    return {sid: min(sc[:5], key=lambda s: (ANCHOR_ORDER.get(s.get("framing"), 3), sc.index(s))) for sid, sc in by_scene.items()}


def redraw(run_dir: Path, shots: list[dict], ids: set[str], *, hints: dict[str, str] | None = None, match: dict[str, str] | None = None,
           label: str = "redraw", threads: int = 6) -> dict:
    """Draw the given frames once more as a new attempt. The new picture replaces the old one when the check accepts it
    (a redraw is asked for a reason the check's score does not see: a repeated background view, a continuity error).
    hints: {shot id: what to fix}; match: {shot id: id of the shot whose frame it must agree with}."""
    state = read_json(run_dir / "frames_state.json") or {}
    todo = [s for s in shots if s["shot_id"] in ids and s["start"] in ("panel", "new")]
    panels = {ch: _panels(ch) for ch in {s["chapter"] for s in todo}}
    anchors = scene_anchors(shots)
    kf = lambda sid: run_dir / "shots" / sid / "kf.png"          # noqa: E731

    def one(shot: dict) -> tuple[int, dict | None]:
        st = state.get(shot["shot_id"]) or {"attempt": -1, "checks": []}
        n = max([st.get("attempt", -1)] + [c.get("attempt", -1) for c in st.get("checks", [])]) + 1
        a = anchors[shot["scene_id"]]
        ref = kf(a["shot_id"]) if a["shot_id"] != shot["shot_id"] and kf(a["shot_id"]).exists() else None
        m = (match or {}).get(shot["shot_id"])
        try:
            f = draw_frame(run_dir, shot, panels[shot["chapter"]].get(shot.get("panel")) if shot["start"] == "panel" else None, attempt=n,
                           scene_ref=ref, hint=(hints or {}).get(shot["shot_id"], ""), match_ref=kf(m) if m and kf(m).exists() else None)
        except Exception as e:                                             # noqa: BLE001
            log(f"redraw {shot['shot_id']} failed: {str(e)[:200]}")
            return n, None
        return n, (check_frame(run_dir, shot, n) if f else None)

    t0 = time.time()
    with ThreadPoolExecutor(threads) as ex:
        res = list(ex.map(one, todo))
    kept = 0
    for shot, (n, c) in zip(todo, res):
        st = state.setdefault(shot["shot_id"], {"attempt": -1, "checks": []})
        st["attempt"] = max(st.get("attempt", -1), n)
        if c is None:
            continue
        st["checks"].append({**{k: c.get(k) for k in ("attempt", "verdict", "score", "people_drawn", "problems", "unsuitable", "extra_people")}, "why": label})
        if c.get("verdict") == "ok" and not c.get("unsuitable"):
            st["final_attempt"], st["final_verdict"], st["final_score"] = n, "ok", c.get("score")
            shutil.copyfile(run_dir / "shots" / shot["shot_id"] / f"kf_a{n}.png", kf(shot["shot_id"]))
            kept += 1
    write_json(run_dir / "frames_state.json", state)
    log(f"gemini frames, {label}: {len(todo)} redrawn in {(time.time() - t0) / 60:.1f} min, {kept} accepted and kept; API spend ${api_spend():.2f}")
    return state


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--only", default="")
    ap.add_argument("--rounds", type=int, default=1)
    a = ap.parse_args()
    rd = P("runs") / a.run
    run_frames_gemini(rd, read_json(rd / "shotlist.json")["shots"], rounds=a.rounds, only=set(a.only.split(",")) if a.only else None)
