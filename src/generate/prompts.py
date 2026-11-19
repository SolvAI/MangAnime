"""Prompt builders: shot spec -> keyframe edit prompt (Qwen-Image-Edit) and video prompt (Wan 2.2 I2V).

Deterministic templates first; the repair ladder may ask the local Qwen to rewrite them (src/qa/repair.py).
"""
from __future__ import annotations

from src.common.config import CFG

TRIGGER = "komi_anime_style"
KF_TRIGGER = "komi_keyframe"

STYLE = ("clean TV anime frame, flat cel shading, thin dark outlines, soft pastel colours, bright even school lighting, "
         "detailed painted background")
REMOVE = ("Remove every speech bubble, caption box, sound-effect lettering, screentone dots, speed lines and panel border; "
          "no text anywhere in the image")


# Hair colours are the only design hints given to the editor: longer descriptions (outfits, poses) make it redraw the
# panel instead of colouring it (measured in the M1 keyframe experiments, see RESULTS.md).
HAIR = {
    "Komi": "Komi has long dark purple-black hair", "Tadano": "Tadano has short black hair",
    "Najimi": "Najimi has short lavender-brown hair", "Yamai": "Yamai has orange-brown hair",
    "Agari": "Agari has brown braided hair", "Nakanaka": "Nakanaka has long dark hair", "Katai": "Katai has light brown hair",
    "Onemine": "Onemine has brown hair", "Otori": "Otori has light hair", "Yadano": "Yadano has black hair",
    "Tadano's sister": "Tadano's sister has short black hair", "Komi's mother": "Komi's mother has dark purple-black hair",
    "Komi's father": "Komi's father has dark hair", "Shousuke": "Shousuke has dark purple-black hair",
}


SCHOOL = ("class", "school", "hallway", "corridor", "locker", "cafeteria", "gym", "rooftop", "library", "staff room", "infirmary")
# A colour PALETTE, not a content demand. Without it every shot gets a different blazer colour (grey / lavender / navy).
# "Everyone wears the uniform ..." made the editor zoom out and redraw close-ups to show the uniform (tmp/kf_exp5, kf_exp6:
# panel | no hint | uniform sentence | palette); the palette wording keeps the composition in 6 of 8 test shots. It is left out
# for close-ups and inserts, where it still triggered redraws and the uniform is barely visible anyway.
PALETTE = "Colour palette: school blazers dark navy blue, bows and neckties red with white stripes, shirts white"
BG = "pastel background colours, no checkerboard"
TIGHT = ("close-up", "extreme close-up", "insert")


def is_school(location: str) -> bool:
    loc = (location or "").lower()
    return any(k in loc for k in SCHOOL)


def colorize_prompt(characters: list[str], hint: str = "", location: str = "", shot_size: str = "") -> str:
    """Panel -> anime frame as a faithful colourisation (Picture 1 = 16:9 crop of the panel, lettering blanked)."""
    names = [n for n in characters if n in HAIR][:3]
    txt = ("Colorize this black-and-white manga panel as a frame of the TV anime \"Komi Can't Communicate\": clean flat cel shading, "
           "thin dark outlines, soft pastel colours, bright even lighting. "
           "Keep exactly the same composition, characters, clothes, poses, facial expressions and camera angle as Picture 1. "
           "Erase every speech balloon, caption box, sound-effect lettering and screentone and paint the scene behind them. "
           "No text, no borders, no white areas.")
    if names:
        txt += " " + "; ".join(HAIR[n] for n in names) + "."
    tight = (shot_size or "").lower() in TIGHT
    if is_school(location) and not tight:
        txt += " " + PALETTE + "; " + BG + "."
    elif not tight:
        txt += " " + BG[0].upper() + BG[1:] + "."
    if hint:
        txt += " " + hint.rstrip(".") + "."
    return txt


# ----------------------------------------------------------------------------- Illustrious-XL (Danbooru tags)
ILL_QUALITY = "masterpiece, best quality, amazing quality, absurdres, anime screencap, anime coloring, official style"
ILL_SERIES = "komi-san wa komyushou desu"
# Strictly safe-for-work: the cast are school students. Rating tag in the positive, everything else in the negative.
ILL_NEG = ("nsfw, explicit, sensitive, nude, cleavage, underwear, panties, swimsuit, lowres, worst quality, bad quality, bad anatomy, "
           "bad hands, extra digits, extra arms, jpeg artifacts, signature, watermark, artist name, text, speech bubble, thought bubble, "
           "comic, manga, monochrome, greyscale, border, letterboxed, multiple views, sketch, ai-generated, pale color, 3d, realistic, "
           "clone, multiple persona, kiss, couple, romance")
# name -> (danbooru character tag + stable appearance tags, gender for the count tag)
ILL_CAST = {
    "Komi": ("komi shouko, long hair, black hair, purple eyes", "f"),
    "Tadano": ("tadano hitohito, short hair, black hair, ahoge", "m"),
    "Najimi": ("osana najimi, short hair, purple hair, androgynous", "o"),
    "Yamai": ("yamai ren, long hair, brown hair, ponytail", "f"),
    "Agari": ("agari himiko, brown hair, twin braids, glasses", "f"),
    "Nakanaka": ("nakanaka omoharu, long hair, black hair, hair over one eye", "f"),
    "Katai": ("katai makoto, short hair, spiked hair, blonde hair, thick eyebrows, sanpaku, broad shoulders", "m"),
    "Onemine": ("onemine nene, long hair, brown hair", "f"),
    "Otori": ("otori kaede, long hair, light brown hair", "f"),
    "Yadano": ("yadano makeru, short hair, black hair, bob cut", "f"),
    "Inaka": ("inaka nokoko, brown hair, low twintails", "f"),
    "Tadano's sister": ("tadano hitomi, short hair, black hair, ahoge", "f"),
    "Komi's mother": ("komi shuuko, long hair, black hair, mature female", "f"),
    "Komi's father": ("komi masayoshi, short hair, black hair, mature male", "m"),
    "Shousuke": ("komi shousuke, short hair, black hair", "m"),
}
ILL_UNIFORM_F = "school uniform, blazer, blue jacket, long sleeves, white shirt, red bowtie, striped bowtie, plaid skirt"
ILL_UNIFORM_M = "school uniform, blazer, blue jacket, long sleeves, white shirt, red necktie, striped necktie"
ILL_FRAMING = {"extreme close-up": "close-up, face focus", "close-up": "portrait, close-up", "medium close-up": "upper body",
               "medium": "cowboy shot", "medium wide": "full body", "wide": "wide shot, scenery", "insert": "close-up, still life"}


def _count_tags(genders: list[str]) -> str:
    def n(k, one, many):
        c = genders.count(k)
        return [] if c == 0 else [f"{c}{one}" if c == 1 else (f"{c}{many}" if c <= 3 else f"multiple {many}")]

    tags = n("f", "girl", "girls") + n("m", "boy", "boys") + n("o", "other", "others")
    if len(genders) == 1:
        tags.append("solo")
    if genders and set(genders) == {"m"}:
        tags.append("male focus")
    return ", ".join(tags)


def illustrious_prompt(shot: dict, scene_tags: list[str] | str = "", visible: list[str] | None = None, hint: str = "") -> str:
    """Tag prompt for the Illustrious keyframe: quality + series + cast (Danbooru names) + uniform + scene tags."""
    names = [c for c in (visible if visible is not None else shot.get("characters", [])) if c in ILL_CAST][:4]
    crowd = "Classmates" in (shot.get("characters") or [])
    parts = [ILL_QUALITY, ILL_SERIES]
    cnt = _count_tags([ILL_CAST[n][1] for n in names])
    if cnt:
        parts.append(cnt)
    parts += [ILL_CAST[n][0] for n in names]
    if is_school(shot.get("location", "")) and names:
        g = {ILL_CAST[n][1] for n in names}
        parts.append(ILL_UNIFORM_F if g != {"m"} else ILL_UNIFORM_M)
    if crowd:
        parts.append("multiple girls, multiple boys, crowd" if not names else "people in background")
    fr = ILL_FRAMING.get((shot.get("shot_size") or "").lower())
    if fr:
        parts.append(fr)
    st = scene_tags if isinstance(scene_tags, str) else ", ".join(t for t in scene_tags if t)
    if st:
        parts.append(st)
    parts.append("general")
    if hint:
        parts.append(hint.strip().rstrip(".")[:220])
    return ", ".join(p for p in parts if p)


def illustrious_prompt_v2(shot: dict, t: dict, hint: str = "") -> tuple[str, dict]:
    """Panel-aware prompt (D-039). `t` = tags.tag_shots_vlm output for the shot. Returns (prompt, {"named": [...], "described": n}).

    A cast member is drawn BY NAME only when the VLM recognised them in this panel crop AND the director's script or the
    earlier caption pass also lists them. Everyone else is drawn from the look the VLM actually saw, so a mislabel in the
    script no longer turns a side character into Komi."""
    from src.generate.tags import _tags

    script = set(t.get("director_characters") or shot.get("characters") or []) | set(t.get("panel_characters") or [])
    people = [p for p in (t.get("people") or []) if isinstance(p, dict)][:4]
    parts = [ILL_QUALITY, ILL_SERIES]
    genders, named, body = [], [], []
    for p in people:
        g = {"girl": "f", "woman": "f", "boy": "m", "man": "m"}.get(str(p.get("gender", "")).lower(), "o")
        genders.append(g)
        name = str(p.get("name", "unknown"))
        # two independent sources must agree: this look at the panel AND (the director's script or the earlier caption pass)
        ok = name in ILL_CAST and name in script and name not in named
        if ok:
            named.append(name)
            who = ILL_CAST[name][0]
        else:
            who = ", ".join(_tags(p.get("look"))[:6]) + (", mature female" if p.get("gender") == "woman" else ", mature male" if p.get("gender") == "man" else "")
        cloth = _tags(p.get("clothing"))[:4]
        seg, seen = [], set()
        for x in ", ".join([who] + cloth + _tags(p.get("pose"))[:4] + _tags(p.get("expression"))[:4]).split(", "):
            if x and x not in seen:
                seen.add(x)
                seg.append(x)
        body.append(", ".join(seg))
    n_total = int(t.get("n_people_total") or len(people))
    if not people and n_total == 0:
        parts.append("no humans")
    else:
        cnt = _count_tags(genders)
        if cnt:
            parts.append(cnt)
        parts += body
        if is_school(shot.get("location", "")) and not any(_tags(p.get("clothing")) for p in people):
            parts.append(ILL_UNIFORM_F if set(genders) != {"m"} else ILL_UNIFORM_M)
        if n_total > len(people) + 1:
            parts.append("people in background, crowd")
    for k, n in (("view", 4), ("action", 8), ("background", 6)):
        parts.append(", ".join(_tags(t.get(k))[:n]))
    parts.append("general")
    sit = str(t.get("situation") or "").strip().rstrip(".")
    if sit:
        parts.append(sit[:200])
    if hint:
        parts.append(hint.strip().rstrip(".")[:200])
    return ", ".join(p for p in parts if p), {"named": named, "described": len(people) - len(named), "n_people_total": n_total}


def _look(name: str) -> str:
    for c in CFG.characters.roster:
        if c["name"] == name:
            return c["look"]
    return ""


def keyframe_prompt(shot: dict, has_panel: bool, ref_names: list[str], hint: str = "") -> str:
    """Picture 1 = manga panel (if any); Picture 2.. = anime reference frames of `ref_names` (same order)."""
    chars = [c for c in shot.get("characters", []) if c not in ("Narrator",)]
    parts = []
    if has_panel:
        parts.append("Turn the black-and-white manga panel in Picture 1 into a finished full-colour frame of the TV anime "
                     "\"Komi Can't Communicate\" in 16:9 widescreen.")
        parts.append("Keep the characters, poses, facial expressions and camera angle of Picture 1, and paint in the missing "
                     "background so the wide frame is completely filled (no white borders, no letterbox).")
    else:
        parts.append("Create a finished full-colour 16:9 frame of the TV anime \"Komi Can't Communicate\".")
    for i, n in enumerate(ref_names):
        pic = i + (2 if has_panel else 1)
        parts.append(f"{n} must look exactly like the character in Picture {pic} (same hair, face, colours and uniform).")
    for n in chars:
        if n not in ref_names and _look(n):
            parts.append(f"{n}: {_look(n)}.")
    if shot.get("first_frame"):
        parts.append("Shot: " + shot["first_frame"].rstrip(".") + ".")
    if shot.get("shot_size"):
        parts.append(f"Framing: {shot['shot_size']} shot.")
    loc = ", ".join(x for x in (shot.get("location", ""), shot.get("time_of_day", "")) if x)
    if loc:
        parts.append(f"Setting: {loc}.")
    parts.append(f"Style: {STYLE}.")
    parts.append(REMOVE + ".")
    if hint:
        parts.append(hint.rstrip(".") + ".")
    return " ".join(parts)


def video_prompt(shot: dict, hint: str = "", trigger: bool | str = True, style: str = "") -> str:
    """trigger: True -> this project's LoRA trigger; a string -> that LoRA's own trigger words; False -> none.
    style "lively": the wording that asks for visible character animation instead of "subtle" motion (motion test, D-057)."""
    import os
    import re

    style = style or os.environ.get("KOMI_VIDEO_PROMPT", "")
    chars = [c for c in shot.get("characters", []) if c != "Narrator"]
    # a "line" made of dots or marks ("......", "!?") is a silent reaction, not speech: no mouth flaps for it
    speaking = [d["speaker"] for d in shot.get("dialogue", []) if d["kind"] == "speech" and d["speaker"] in chars and re.search(r"[A-Za-z]{2,}", str(d.get("text") or ""))]
    if style == "lively":
        head = ["anime", f"{shot.get('shot_size', 'medium')} shot"] + ([", ".join(chars)] if chars else [])
        body = shot.get("action", "").strip().rstrip(".")
        body = re.sub(r"\s*Static camera$", "", body, flags=re.I).rstrip(". ")
        txt = ", ".join(head) + ". " + (body + "." if body else "")
        if speaking:
            who = " and ".join(dict.fromkeys(speaking))
            txt += f" {who} talks the whole time with clear mouth movement, changing expression and small gestures."
            if len(chars) > len(set(speaking)):
                txt += " The listeners react: they blink, shift and turn their heads."
        elif chars:
            txt += " Nobody speaks, mouths closed. The characters stay alive: they blink, breathe, shift their weight and their hair and clothes move."
        cam = shot.get("camera", "static") or "static"
        txt += f" Camera: {cam}."
        if chars:
            txt += " Expressive TV anime character animation with clear, continuous movement; characters stay on-model, clean linework, flat colours."
        else:       # an insert or an empty place: nothing may start moving on its own (motion test: 2 of 2 such clips rejected with the lively wording)
            txt += " Calm shot: only the small movement described, everything else holds still; stable picture, clean linework, flat colours."
        if hint:
            txt += " " + hint.rstrip(".") + "."
        return txt
    trig = trigger if isinstance(trigger, str) else (TRIGGER if trigger else "")
    head = [(trig + ", anime") if trig else "anime", f"{shot.get('shot_size', 'medium')} shot"]
    if chars:
        head.append(", ".join(chars))
    body = shot.get("action", "").strip().rstrip(".")
    txt = ", ".join(head) + ". " + (body + "." if body else "")
    if speaking:
        who = " and ".join(dict.fromkeys(speaking))
        txt += f" {who} is talking, mouth opening and closing naturally."
    elif chars:
        txt += " Mouths stay closed; subtle blinking and breathing."
    cam = shot.get("camera", "static") or "static"
    txt += f" Camera: {cam}."
    txt += " Limited TV animation, consistent character design, stable linework, flat colours."
    if hint:
        txt += " " + hint.rstrip(".") + "."
    return txt
