"""Keyframes (brief §7.2): manga panel -> anime first frame with Qwen-Image-Edit-2511 (+ character references).

Per shot: Picture 1 = source panel crop (first of `source_panels`), Picture 2/3 = reference frames of the shot's
characters from the character bank (train episodes only). Shots without a usable panel are generated from the
references + description, or re-framed from the previous shot's keyframe.
Outputs runs/<run>/shots/<shot_id>/kf_a<attempt>.png (1280x720) and kf.png (current pick).
"""
from __future__ import annotations

import shutil
from pathlib import Path

from PIL import Image

from src.common.config import CFG, P
from src.common.io import log, read_json, read_jsonl, sha, write_json
from src.generate.comfy import Comfy, wf_qwen_edit
from src.generate.prompts import ILL_NEG, KF_TRIGGER, colorize_prompt, illustrious_prompt, keyframe_prompt

ERASE_TEXT = True            # white-out detected dialogue text before editing
LAYOUT = "canvas"            # "canvas": panel pre-placed on a 16:9 canvas (outpaint margins) | "raw": the bare panel crop
KF_W, KF_H = 1024, 576      # exact 16:9, multiple of 16; Wan consumes 832x480 anyway (fewer tokens = faster edits)


def character_refs(names: list[str], k: int = 2) -> list[tuple[str, str]]:
    """[(name, path)] reference frames for up to k characters, main cast first."""
    bank = read_json(P("processed") / "characters" / "bank.json") or {}
    order = {c["name"]: i for i, c in enumerate(CFG.characters.roster)}
    out = []
    for n in sorted(dict.fromkeys(names), key=lambda x: order.get(x, 99)):
        refs = (bank.get(n) or {}).get("refs") or []
        if refs and Path(refs[0]).exists():
            out.append((n, refs[0]))
        if len(out) >= k:
            break
    return out


def prep_panel(panel_img: str, out: Path) -> Path:
    """Pad very small panels and convert to RGB PNG (the editor sees the whole panel; it reframes to 16:9 itself)."""
    im = Image.open(panel_img).convert("RGB")
    if min(im.size) < 384:
        s = 384 / min(im.size)
        im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    out.parent.mkdir(parents=True, exist_ok=True)
    im.save(out)
    return out


def compose_canvas(panel: dict, out: Path, w: int = KF_W, h: int = KF_H) -> Path:
    """Put the panel on a 16:9 canvas so the edit model does not have to invent the reframing:
    wide panels are cropped around the characters, tall panels are cropped to the characters' extent and the rest is
    left as white margins for the model to paint in (outpainting)."""
    im = Image.open(panel["image"]).convert("RGB")
    pw, ph = im.size
    px, py = panel["bbox"][0], panel["bbox"][1]
    if ERASE_TEXT:      # blank the lettering inside balloons / caption boxes (they are white inside): no glyphs left to garble
        from PIL import ImageDraw

        d = ImageDraw.Draw(im)
        for t in panel.get("dialogue") or []:
            if t.get("kind") != "sfx" and t.get("box"):
                x0, y0, x1, y1 = t["box"]
                d.rectangle([x0 - px - 2, y0 - py - 2, x1 - px + 2, y1 - py + 2], fill=(255, 255, 255))
    boxes = [[b[0] - px, b[1] - py, b[2] - px, b[3] - py] for b in panel.get("char_boxes") or []]
    boxes = [b for b in boxes if b[2] > 0 and b[3] > 0 and b[0] < pw and b[1] < ph]
    if boxes:
        ux0, uy0 = max(0, min(b[0] for b in boxes)), max(0, min(b[1] for b in boxes))
        ux1, uy1 = min(pw, max(b[2] for b in boxes)), min(ph, max(b[3] for b in boxes))
    else:
        ux0, uy0, ux1, uy1 = 0, 0, pw, ph
    target = w / h
    if pw / ph >= target:                     # wide panel: full height, crop the width around the characters
        cw = ph * target
        cx = (ux0 + ux1) / 2
        x0 = min(max(0, cx - cw / 2), pw - cw)
        crop = im.crop((int(x0), 0, int(x0 + cw), ph))
    else:                                     # narrower than 16:9: keep the characters (heads first), pad the sides
        ch = min(ph, max(pw / target, (uy1 - uy0) * 1.15))
        y0 = min(max(0, uy0 - 0.08 * ch), ph - ch) if boxes else (ph - ch) / 2
        crop = im.crop((0, int(y0), pw, int(y0 + ch)))
    s = min(w / crop.width, h / crop.height)
    crop = crop.resize((max(1, round(crop.width * s)), max(1, round(crop.height * s))), Image.LANCZOS)
    canvas = Image.new("RGB", (w, h), (255, 255, 255))
    canvas.paste(crop, ((w - crop.width) // 2, (h - crop.height) // 2))
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    return out


def compose_cover(panel: dict, out: Path, w: int = KF_W, h: int = KF_H, zoom: float = 1.0) -> Path:
    """Full-bleed 16:9 crop of the panel centred on the characters (heads kept), dialogue lettering blanked.
    `zoom` < 1 punches in on the characters (used when consecutive shots come from the same panel)."""
    from PIL import ImageDraw

    im = Image.open(panel["image"]).convert("RGB")
    pw, ph = im.size
    px, py = panel["bbox"][0], panel["bbox"][1]
    d = ImageDraw.Draw(im)
    tmask = Image.new("L", im.size, 0)          # 255 = dialogue lettering box, 128 = sound-effect lettering box
    dm = ImageDraw.Draw(tmask)
    for t in panel.get("dialogue") or []:
        if not t.get("box"):
            continue
        x0, y0, x1, y1 = t["box"]
        if t.get("kind") != "sfx":
            d.rectangle([x0 - px - 2, y0 - py - 2, x1 - px + 2, y1 - py + 2], fill=(255, 255, 255))
            dm.rectangle([x0 - px - 2, y0 - py - 2, x1 - px + 2, y1 - py + 2], fill=255)
        else:
            dm.rectangle([x0 - px, y0 - py, x1 - px, y1 - py], fill=128)
    boxes = [[b[0] - px, b[1] - py, b[2] - px, b[3] - py] for b in panel.get("char_boxes") or []]
    boxes = [b for b in boxes if b[2] > 0 and b[3] > 0 and b[0] < pw and b[1] < ph]
    if boxes:
        ux0, uy0 = max(0, min(b[0] for b in boxes)), max(0, min(b[1] for b in boxes))
        ux1, uy1 = min(pw, max(b[2] for b in boxes)), min(ph, max(b[3] for b in boxes))
    else:
        ux0, uy0, ux1, uy1 = 0, 0, pw, ph
    target = w / h
    if pw / ph >= target:
        cw, ch = ph * target, float(ph)
    else:
        cw, ch = float(pw), pw / target
    cw, ch = cw * zoom, ch * zoom
    cx = (ux0 + ux1) / 2
    x0 = min(max(0.0, cx - cw / 2), pw - cw)
    y0 = min(max(0.0, uy0 - 0.06 * ch), ph - ch) if boxes else (ph - ch) * 0.3     # keep the heads
    crop = im.crop((int(x0), int(y0), int(x0 + cw), int(y0 + ch))).resize((w, h), Image.LANCZOS)
    out.parent.mkdir(parents=True, exist_ok=True)
    crop.save(out)
    tmask.crop((int(x0), int(y0), int(x0 + cw), int(y0 + ch))).resize((w, h), Image.NEAREST).save(out.with_name(out.stem + "_text.png"))
    return out


def remove_balloons(canvas: Path, out: Path) -> tuple[Path, float]:
    """Paint speech balloons and sound-effect lettering out of the panel crop (for the image model's layout inputs).

    A balloon = the white region around a dialogue lettering box (flood from the box over near-white pixels), unless that
    region is the open white background (> 30 % of the frame). Sound-effect boxes are removed as rectangles. Returns the
    inpainted crop and the share of the frame that was removed."""
    import cv2
    import numpy as np

    img = cv2.imread(str(canvas))
    tpath = canvas.with_name(canvas.stem + "_text.png")
    if img is None or not tpath.exists():
        return canvas, 0.0
    tm = cv2.imread(str(tpath), cv2.IMREAD_GRAYSCALE)
    H, W = img.shape[:2]
    white = (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) > 225).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(white, 8)
    mask = np.zeros((H, W), np.uint8)
    for i in set(np.unique(labels[tm == 255]).tolist()) - {0}:
        if stats[i, cv2.CC_STAT_AREA] < 0.30 * H * W:
            mask[labels == i] = 255
    k = max(3, int(15 * W / 1024)) | 1
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))       # take the balloon outline too
    mask[tm == 128] = 255
    frac = float((mask > 0).mean())
    if frac < 0.002 or frac > 0.6:
        return canvas, 0.0
    cv2.imwrite(str(out), cv2.inpaint(img, mask, 7, cv2.INPAINT_TELEA))
    cv2.imwrite(str(out.with_name(out.stem + "_mask.png")), mask)          # lineart_control drops every line inside it
    return out, frac


def title_card(text: str, out: Path, subtitle: str = "") -> Path:
    """Simple title card (the real ones are excluded from training; this is a typographic stand-in)."""
    from PIL import ImageDraw, ImageFont

    im = Image.new("RGB", (KF_W, KF_H), (24, 46, 38))  # blackboard green
    d = ImageDraw.Draw(im)
    font_path = next((p for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                                  "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf") if Path(p).exists()), None)

    def fit(t: str, size: int):
        """Largest font up to `size` at which the line fits the card with a margin (a long title used to run off both edges)."""
        if not font_path:
            return ImageFont.load_default()
        while size > 18 and d.textlength(t, font=ImageFont.truetype(font_path, size)) > KF_W * 0.88:
            size -= 2
        return ImageFont.truetype(font_path, size)

    if not subtitle and ": " in text and font_path and d.textlength(text, font=ImageFont.truetype(font_path, 64)) > KF_W * 0.88:
        text, subtitle = [p.strip() for p in text.split(": ", 1)]          # "Communication 117: Valentine's Day" -> two lines
    for i, (t, size) in enumerate(((text, 64), (subtitle, 40))):
        if not t:
            continue
        f = fit(t, size)
        w = d.textlength(t, font=f)
        d.text(((KF_W - w) / 2, KF_H / 2 - 70 + i * 90), t, font=f, fill=(240, 240, 232))
    out.parent.mkdir(parents=True, exist_ok=True)
    im.save(out)
    return out


def make_keyframe(comfy: Comfy, run_dir: Path, shot: dict, panels: dict[int, dict], *, attempt: int = 0, seed: int | None = None,
                  hint: str = "", loras: list[tuple[str, float]] | None = None, prev_kf: Path | None = None,
                  max_chars: int | None = None) -> Path:
    sdir = run_dir / "shots" / shot["shot_id"]
    sdir.mkdir(parents=True, exist_ok=True)
    out = sdir / f"kf_a{attempt}.png"
    if "TITLE_CARD" in (shot.get("notes") or ""):
        import re

        title = shot["notes"].split("TITLE_CARD:", 1)[-1].strip() if "TITLE_CARD:" in shot["notes"] else ""
        title = re.sub(r"^\s*comm(unication)?\.?\s*\d+\s*[:\-–—]?\s*", "", title, flags=re.I).strip(' "“”')
        title_card(f"Communication {shot['chapter']}", out, title)
        shutil.copyfile(out, sdir / "kf.png")
        return out
    chars = [c for c in shot.get("characters", []) if c not in ("Narrator",)]
    if max_chars is not None:
        chars = chars[:max_chars]
    panel = next((panels[o] for o in shot.get("source_panels", []) if o in panels), None)
    images, names = [], []
    has_panel = panel is not None
    use_prev = (not has_panel) and prev_kf is not None and Path(prev_kf).exists()
    if backend() == "illustrious":
        return _make_keyframe_illustrious(comfy, run_dir, shot, panel, attempt=attempt, seed=seed, hint=hint, prev_kf=prev_kf if use_prev else None,
                                          max_chars=max_chars)
    if has_panel:
        # faithful colourisation of a 16:9 crop of the panel; no reference images (they get copied wholesale)
        prep_panel(panel["image"], sdir / "panel.png")                      # kept for the fidelity evidence
        zoom = float(shot.get("_kf_zoom", 1.0)) * (0.92 if attempt % 2 == 1 else 1.0)
        src_img = compose_cover(panel, sdir / "panel_canvas.png", zoom=zoom)
        images.append(comfy.put_input(src_img, f"{run_dir.name}_{shot['shot_id']}_panel.png"))
        pchars = [c for c in (panel.get("characters") or []) if c in chars] or chars or (panel.get("characters") or [])
        size = shot.get("shot_size", "") if float(shot.get("_kf_zoom", 1.0)) >= 0.99 else "close-up"      # punch-ins are tight
        prompt = colorize_prompt(pchars, hint, location=shot.get("location", ""), shot_size=size)
    elif use_prev:
        # no panel for this shot (director-added reaction/insert): re-frame from the previous keyframe
        images.append(comfy.put_input(prev_kf, f"{run_dir.name}_{shot['shot_id']}_prev.png"))
        spec = dict(shot)
        spec["characters"] = chars
        prompt = keyframe_prompt(spec, True, [], "Picture 1 is the previous shot of the same scene: keep the same location, lighting, "
                                 "art style and character designs, but change the framing and pose as described. " + hint)
    else:
        spec = dict(shot)
        spec["characters"] = chars
        prompt = keyframe_prompt(spec, False, [], hint)
    if loras:
        prompt = KF_TRIGGER + ". " + prompt
    seed = seed if seed is not None else int(sha(shot["shot_id"] + str(attempt), 8), 16) % (2**31)
    meta = {"prompt": prompt, "images": images, "seed": seed, "attempt": attempt, "has_panel": has_panel, "refs": names, "loras": loras or [],
            "layout": "cover" if has_panel else ("prev" if use_prev else "text")}
    if not images:  # nothing to condition on at all -> use a neutral grey canvas as Picture 1
        grey = sdir / "blank.png"
        Image.new("RGB", (KF_W, KF_H), (128, 128, 128)).save(grey)
        images.append(comfy.put_input(grey, f"{run_dir.name}_{shot['shot_id']}_blank.png"))
    files = comfy.run(wf_qwen_edit(images, prompt, width=KF_W, height=KF_H, seed=seed, loras=loras,
                                   prefix=f"{run_dir.name}/{shot['shot_id']}_kf_a{attempt}"), timeout=900)
    if not files:
        raise RuntimeError("keyframe workflow produced no image")
    shutil.copyfile(files[0], out)
    shutil.copyfile(out, sdir / "kf.png")
    write_json(sdir / f"kf_a{attempt}.json", meta)
    return out


def backend() -> str:
    """KOMI_KF_BACKEND: "qwen_edit" (colourised panel, D-026) or "illustrious" (anime image model, panel as layout, D-038)."""
    import os

    return os.environ.get("KOMI_KF_BACKEND", "qwen_edit")


def soften_layout(src: Path, out: Path, sigma: float = 7.0, tint=(240, 236, 232), keep: float = 0.6) -> Path:
    """Layout reference for img2img: the panel crop blurred and washed with a pale warm tint.

    A raw black-and-white panel pulls the image model back to monochrome line art with screentone and balloons unless the
    denoise is so high that the layout is lost. Blurring removes line / screentone / lettering detail and keeps the masses
    (where the figures and the dark hair are); a near-neutral wash lifts it away from pure black-and-white (a warm wash gave every scene a sunset cast)."""
    import cv2
    import numpy as np

    im = cv2.imread(str(src))
    k = sigma * im.shape[1] / 1024.0
    im = cv2.GaussianBlur(im, (0, 0), k)
    im = cv2.addWeighted(im, keep, np.full_like(im, tint), 1.0 - keep, 0)        # tint is BGR
    cv2.imwrite(str(out), im)
    return out


def lineart_control(src: Path, out: Path) -> Path:
    """White-on-black line drawing of the panel crop for the line-art ControlNet: thin lines + outlines of solid black areas,
    screentone and grey shading removed (they would be read as texture)."""
    import cv2
    import numpy as np

    g = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
    k = max(1.0, g.shape[1] / 1024.0)
    dark = (cv2.GaussianBlur(g, (0, 0), 1.3 * k) < 105).astype(np.uint8) * 255
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(7 * k) | 1, int(7 * k) | 1))
    solid = cv2.morphologyEx(dark, cv2.MORPH_OPEN, ker)                       # filled black regions (hair, night skies)
    lines = cv2.bitwise_and(dark, cv2.bitwise_not(cv2.dilate(solid, ker)))    # thin strokes
    outline = cv2.morphologyEx(solid, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    ctl = cv2.bitwise_or(lines, outline)
    mpath = Path(src).with_name(Path(src).stem + "_mask.png")            # balloons / sound words painted out by remove_balloons
    if mpath.exists():
        m = cv2.imread(str(mpath), cv2.IMREAD_GRAYSCALE)
        if m is not None and m.shape == ctl.shape:
            ctl[cv2.dilate(m, np.ones((5, 5), np.uint8)) > 0] = 0
    cv2.imwrite(str(out), ctl)
    return out


def mean_saturation(path) -> float:
    """Mean HSV saturation (0-255); finished anime frames are ~60-110, a monochrome manga-like result is < 25."""
    import cv2

    im = cv2.imread(str(path))
    return float(cv2.cvtColor(im, cv2.COLOR_BGR2HSV)[..., 1].mean()) if im is not None else 0.0


ILL_DENOISE_CN = (0.85, 0.88, 0.9, 0.92)    # with the line-art ControlNet (D-039)
ILL_DENOISE = (0.78, 0.84, 0.88, 0.9)       # per attempt: a retry moves further away from the panel's layout
# measured on 8 shots (tmp/ill_test*/sheet.jpg): raw panel at 0.6-0.72 stays monochrome with balloons; softened panel at 0.78
# gives full-colour screencap frames that still follow the panel's masses; 0.85 is cleaner but looser.


def _make_keyframe_illustrious(comfy: Comfy, run_dir: Path, shot: dict, panel: dict | None, *, attempt: int = 0, seed: int | None = None,
                               hint: str = "", prev_kf: Path | None = None, max_chars: int | None = None) -> Path:
    """Illustrious-XL + screencap LoRA + detail slider. Picture = 16:9 crop of the manga panel (lettering blanked) as an
    img2img layout reference; the prompt is Danbooru tags (cast by name, uniform, pose / expression / background)."""
    import os

    from src.generate.comfy import wf_illustrious

    sdir = run_dir / "shots" / shot["shot_id"]
    out = sdir / f"kf_a{attempt}.png"
    v2 = (read_json(run_dir / "kf_tags_v2.json") or {}).get(shot["shot_id"]) or {}
    tags = (read_json(run_dir / "kf_tags.json") or {}).get(shot["shot_id"]) or {}
    info: dict = {}
    if v2.get("people") is not None and not v2.get("error") and panel is not None:
        from src.generate.prompts import illustrious_prompt_v2

        if max_chars is not None:
            v2 = {**v2, "people": (v2.get("people") or [])[:max_chars]}
        prompt, info = illustrious_prompt_v2(shot, v2, hint=hint)          # panel-aware prompt (D-039)
    else:
        visible = tags.get("visible")
        if max_chars is not None and visible:
            visible = visible[:max_chars]
        prompt = illustrious_prompt(shot, tags.get("tags", []), visible=visible, hint=hint)
    cn = os.environ.get("KOMI_ILL_CN", "0.5:0.6")              # "<strength>:<end>" of the line-art ControlNet; "" or "0" = off
    cn_s, cn_e = (float(x) for x in cn.split(":")) if cn and cn != "0" else (0.0, 0.0)
    sched = ILL_DENOISE_CN if cn_s > 0 else ILL_DENOISE        # with the ControlNet holding the layout the init can be looser
    denoise = float(os.environ.get("KOMI_ILL_DENOISE", 0)) or sched[min(attempt, len(sched) - 1)]
    image, layout, control = None, "text", None
    if panel is not None:
        prep_panel(panel["image"], sdir / "panel.png")
        zoom = float(shot.get("_kf_zoom", 1.0)) * (0.92 if attempt % 2 == 1 else 1.0)
        src_img = compose_cover(panel, sdir / "panel_canvas.png", zoom=zoom)
        src_img, info["balloons_removed"] = remove_balloons(src_img, sdir / "panel_clean.png")     # no balloons / sound words in the layout inputs
        if cn_s > 0:
            control = comfy.put_input(lineart_control(src_img, sdir / "panel_lines.png"), f"{run_dir.name}_{shot['shot_id']}_lines.png")
            cn_s = max(0.25, cn_s - 0.1 * attempt)          # a retry loosens the grip of the panel's lines
        if os.environ.get("KOMI_ILL_INIT", "soft") == "soft":
            src_img = soften_layout(src_img, sdir / "panel_layout.png")
        image, layout = comfy.put_input(src_img, f"{run_dir.name}_{shot['shot_id']}_panel.png"), "panel_i2i"
    elif prev_kf is not None:
        image, layout, denoise = comfy.put_input(prev_kf, f"{run_dir.name}_{shot['shot_id']}_prev.png"), "prev_i2i", max(denoise, 0.8)
    seed = seed if seed is not None else int(sha(shot["shot_id"] + "ill" + str(attempt), 8), 16) % (2**31)
    retries = 0
    while True:
        files = comfy.run(wf_illustrious(image, prompt, ILL_NEG, seed=seed + retries, denoise=denoise, out_size=(KF_W, KF_H),
                                         control_image=control, control_strength=cn_s, control_end=cn_e,
                                         prefix=f"{run_dir.name}/{shot['shot_id']}_kf_a{attempt}"), timeout=900)
        if not files:
            raise RuntimeError("keyframe workflow produced no image")
        sat = mean_saturation(files[0])
        if sat >= 28 or retries >= 2:
            break
        # busy black-and-white panels (sound-effect lettering, screentone) can pull the result back to monochrome:
        # retry further from the init, with a weaker grip of the lines and an explicit colour tag
        retries += 1
        denoise, cn_s = min(0.96, denoise + 0.06), cn_s * 0.7
        if "colorful" not in prompt:
            prompt = prompt.replace(", general", ", colorful, vibrant colors, general", 1)
    info["saturation"], info["colour_retries"] = round(sat, 1), retries
    shutil.copyfile(files[0], out)
    shutil.copyfile(out, sdir / "kf.png")
    write_json(sdir / f"kf_a{attempt}.json", {"backend": "illustrious", "prompt": prompt, "negative": ILL_NEG, "seed": seed, "attempt": attempt,
                                               "denoise": denoise, "has_panel": panel is not None, "layout": layout, "tags": v2 or tags,
                                               "controlnet": {"strength": cn_s, "end": cn_e} if control else None, **info})
    return out


def run_keyframes(run_dir: Path, shots: list[dict], *, loras: list[tuple[str, float]] | None = None, only: set[str] | None = None,
                  attempt: int = 0, hints: dict[str, str] | None = None) -> None:
    panels_all = read_jsonl(P("processed") / "manga" / "panels.jsonl")
    comfy = Comfy()
    prev: Path | None = None
    last_panel, repeat = None, 0
    for s in shots:
        key = (s["chapter"], (s.get("source_panels") or [None])[0])
        repeat = repeat + 1 if key == last_panel and key[1] is not None else 0
        last_panel = key
        s["_kf_zoom"] = (1.0, 0.72, 0.86)[repeat % 3]
        sdir = run_dir / "shots" / s["shot_id"]
        cur = sdir / "kf.png"
        if (only is None and cur.exists()) or (only is not None and s["shot_id"] not in only):
            prev = cur if cur.exists() else prev
            continue
        panels = {p["order"]: p for p in panels_all if p["chapter"] == s["chapter"]}
        try:
            make_keyframe(comfy, run_dir, s, panels, attempt=attempt, loras=loras, prev_kf=prev, hint=(hints or {}).get(s["shot_id"], ""))
            log(f"keyframe {s['shot_id']} ok")
        except Exception as e:
            log(f"keyframe {s['shot_id']} FAILED: {str(e)[:300]}")
        prev = cur if cur.exists() else prev
