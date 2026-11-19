"""Test: first frames of multi-person shots drawn by Gemini's image model from reference pictures (D-053).

  python scripts/test_gemini_frames.py --run ep22_v3 draw [--n 24] [--model gemini-nano-banana-2.1]
  python scripts/test_gemini_frames.py --run ep22_v3 check
Same frames as scripts/test_ipa_redraw.py (frames of ep22_v3 whose best Illustrious attempt failed the check with 2-3 named
people). The model gets: the shot description (people, positions, poses, the scene's fixed outfits from the wardrobe pass,
place), one reference picture per character (the card's anime picture; for characters who first appear in a held-out
episode only the manga picture plus the colours in words), and the manga panel as layout reference.
Results: shots/<id>/kf_a22.png, checked like every other frame; eval/gemini_frames_test.json and gemini_frames_N.jpg.
Every call is cost-logged in runs/api_cost.jsonl (counts against the API cap). Nothing of the run is changed.
"""
import argparse
import base64
import io
import os
import sys
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.config import CFG, P  # noqa: E402
from src.common.io import append_jsonl, log, read_json, write_json  # noqa: E402
from src.generate.tags import _tags  # noqa: E402
from src.llm.client import COST_LOG, api_spend  # noqa: E402
from src.v3.cast import card  # noqa: E402
from src.v3.frames import check_frame  # noqa: E402

ATT = 22
CARDS = P("processed") / "cards"
PRICE_IN, PRICE_OUT = 1.50, 30.00                     # USD per 1M tokens, gemini-nano-banana-2.1 (pricing page, 2026-11-12); 1K image = 1,120 output tokens
ap = argparse.ArgumentParser()
ap.add_argument("step", choices=["draw", "check"])
ap.add_argument("--run", default="ep22_v3")
ap.add_argument("--outfits-from", default="ep22_v3b", help="run whose shot list carries the wardrobe outfits")
ap.add_argument("--n", type=int, default=24)
ap.add_argument("--model", default="gemini-nano-banana-2.1")
a = ap.parse_args()
rd = P("runs") / a.run
ev = rd / "eval"
shots = {s["shot_id"]: s for s in read_json(P("runs") / a.outfits_from / "shotlist.json")["shots"]}
test = read_json(ev / "ipa_test.json")["frames"][: a.n]          # the same frames as the reference-picture redraw test


def b64(img: Image.Image, side: int = 1024, quality: int = 88) -> str:
    img = img.convert("RGB")
    img.thumbnail((side, side), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def flat(path: Path) -> Image.Image:
    im = Image.open(path)
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, "white")
        bg.alpha_composite(im)
        im = bg
    return im.convert("RGB")


def request(shot: dict) -> list[dict]:
    vis = [v for v in shot.get("visible", []) if v["id"] != "Classmates"][:4]
    crowd = len(vis) != len(shot.get("visible", []))
    lines = [f"Draw ONE frame of the TV anime \"Komi Can't Communicate\" as a finished full-colour anime screenshot, 16:9: clean line art, flat cel "
             f"shading, painted background, the look of the anime reference pictures below. All-ages school comedy.",
             f"WHAT THE FRAME SHOWS: {shot.get('image_sentence', '')}",
             f"FRAMING: {shot.get('framing')}. PLACE: {', '.join(_tags(shot.get('background_tags'))[:8])}.",
             f"PEOPLE: exactly {len(vis)} " + ("in the foreground, plus a few unnamed students far in the background." if crowd else "- nobody else in the frame.")]
    parts: list[dict] = []
    for v in vis:
        c = card(v["id"]) or {}
        outfit = (shot.get("outfits") or {}).get(v["id"]) or ""
        lines.append(f"- {v['id']} ({v.get('position', 'center')}): {v.get('pose', '')}; {v.get('expression', '')}. Wears: {outfit}.")
    parts.append({"text": "\n".join(lines)})
    for v in vis:
        c = card(v["id"]) or {}
        anime = c.get("img_anime") and not c.get("img_anime_heldout") and (CARDS / c["img_anime"]).exists()
        pic = CARDS / c["img_anime"] if anime else (CARDS / c["img_manga"] if c.get("img_manga") and (CARDS / c["img_manga"]).exists() else None)
        if pic is None:
            parts.append({"text": f"CHARACTER {v['id']}: no picture. Looks: {c.get('look_anime') or v.get('look') or 'see the description above'}."})
            continue
        note = "" if anime else f" (black-and-white manga drawing; colours: {c.get('danbooru_tags') or c.get('look_anime') or ''})"
        parts.append({"text": f"CHARACTER REFERENCE — {v['id']}{note}. Keep this face, hair style and hair colour; the clothes are the ones listed above."})
        parts.append({"inlineData": {"mimeType": "image/jpeg", "data": b64(flat(pic), 768)}})
    panel = rd / "shots" / shot["shot_id"] / "panel_clean.png"
    if shot["start"] == "panel" and panel.exists():
        parts.append({"text": "LAYOUT REFERENCE — the manga panel this frame adapts. Use it for the composition, who stands where and the poses. "
                              "Do not copy its lettering, speech balloons, sound effects, screentone or frame border; do not keep it black-and-white."})
        parts.append({"inlineData": {"mimeType": "image/jpeg", "data": b64(flat(panel), 1024)}})
    parts.append({"text": "Output: the single frame. No text, no speech balloons, no captions, no border, no split panels, no watermark."})
    return parts


def draw(m: dict) -> dict:
    sid = m["shot_id"]
    out = rd / "shots" / sid / f"kf_a{ATT}.png"
    if out.exists():
        return {"shot_id": sid, "ok": True, "cached": True}
    body = {"contents": [{"role": "user", "parts": request(shots[sid])}],
            "generationConfig": {"responseModalities": ["IMAGE"], "imageConfig": {"aspectRatio": "16:9", "imageSize": "1K"}}}
    url = f"{CFG.llm.gemini.base_url}/models/{a.model}:generateContent"
    headers = {"Content-Type": "application/json", "x-goog-api-key": os.environ["GEMINI_API_KEY"]}
    for attempt in range(5):
        r = httpx.post(url, json=body, headers=headers, timeout=300)
        if r.status_code in (429, 500, 502, 503, 504, 529):
            time.sleep(min(60, 5 * 2**attempt))
            continue
        if r.status_code >= 400:
            return {"shot_id": sid, "ok": False, "error": f"HTTP {r.status_code}: {r.text[:300]}"}
        d = r.json()
        u = d.get("usageMetadata") or {}
        cost = (u.get("promptTokenCount", 0) * PRICE_IN + (u.get("candidatesTokenCount", 0) + u.get("thoughtsTokenCount", 0)) * PRICE_OUT) / 1e6
        append_jsonl(COST_LOG(), {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "endpoint": "gemini", "model": a.model, "role": "frame_image", "tag": f"geminiframe:{sid}",
                                  "prompt_tokens": u.get("promptTokenCount", 0), "completion_tokens": u.get("candidatesTokenCount", 0) + u.get("thoughtsTokenCount", 0),
                                  "cost_usd": round(cost, 6)})
        cands = d.get("candidates") or []
        imgs = [p["inlineData"]["data"] for p in ((cands[0].get("content") or {}).get("parts") or []) if p.get("inlineData") and not p.get("thought")] if cands else []
        if not imgs:
            why = (d.get("promptFeedback") or {}).get("blockReason") or (cands[0].get("finishReason") if cands else "no candidates")
            return {"shot_id": sid, "ok": False, "error": f"no image ({why})", "cost": cost}
        im = Image.open(io.BytesIO(base64.b64decode(imgs[-1]))).convert("RGB")
        from src.generate.keyframes import KF_H, KF_W

        w, h = im.size                                  # centre-crop to the frame's aspect, then the frame size every other step expects
        t = KF_W / KF_H
        if w / h > t:
            x = int((w - h * t) / 2)
            im = im.crop((x, 0, w - x, h))
        else:
            y = int((h - w / t) / 2)
            im = im.crop((0, y, w, h - y))
        im.resize((KF_W, KF_H), Image.LANCZOS).save(out)
        return {"shot_id": sid, "ok": True, "cost": round(cost, 4), "size": [w, h], "usage": u}
    return {"shot_id": sid, "ok": False, "error": "rate-limited / server errors"}


if a.step == "draw":
    assert "GEMINI_API_KEY" in os.environ, "source scripts/env.sh first"
    cap = min(float(CFG.budget.api_usd), float(CFG.budget.api_usd_effective)) * float(CFG.budget.api_soft_fraction)
    assert api_spend() + 0.08 * len(test) < cap, "API budget"
    first = draw(test[0])                                # one call first: fail early on a wrong request format
    print("first:", {k: v for k, v in first.items() if k != "usage"}, first.get("usage"))
    if first["ok"]:
        with ThreadPoolExecutor(4) as ex:
            res = [first] + list(ex.map(draw, test[1:]))
        write_json(ev / "gemini_frames_test.json", {"model": a.model, "frames": res})
        log(f"gemini frames: {sum(r['ok'] for r in res)}/{len(res)} drawn, ${sum(r.get('cost', 0) for r in res):.2f}; failures: {[r for r in res if not r['ok']][:5]}")
else:
    t = read_json(ev / "gemini_frames_test.json")
    ipa = {m["shot_id"]: m for m in read_json(ev / "ipa_test.json")["frames"]}
    done = [r for r in t["frames"] if r["ok"] and (rd / "shots" / r["shot_id"] / f"kf_a{ATT}.png").exists()]
    with ThreadPoolExecutor(8) as ex:
        checks = list(ex.map(lambda r: check_frame(rd, shots[r["shot_id"]], ATT), done))
    for r, c in zip(done, checks):
        r["check"] = {k: c.get(k) for k in ("verdict", "score", "problems", "unsuitable")} | {
            "cast": [{k: x.get(k) for k in ("id", "present", "looks_right", "note")} for x in (c.get("cast") or []) if isinstance(x, dict)]}
    summ = {"frames": len(done), "accepted": sum(r["check"]["verdict"] == "ok" for r in done), "unsuitable": sum(bool(r["check"].get("unsuitable")) for r in done),
            "mean_score": round(float(np.mean([r["check"]["score"] or 0 for r in done])), 2),
            "mean_score_illustrious_best": round(float(np.mean([ipa[r["shot_id"]]["base_score"] or 0 for r in done])), 2),
            "people_right": sum(1 for r in done for x in r["check"]["cast"] if x.get("present") and x.get("looks_right")),
            "people": sum(len(r["check"]["cast"]) for r in done), "cost_usd": round(sum(r.get("cost", 0) for r in t["frames"]), 2)}
    t["summary"] = summ
    write_json(ev / "gemini_frames_test.json", t)
    print(summ)
    f1 = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    fb = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    TW, TH = 448, 256
    for si in range(0, len(done), 8):
        part = done[si : si + 8]
        sheet = Image.new("RGB", (TW * 3 + 20, (TH + 40) * len(part) + 30), "white")
        d = ImageDraw.Draw(sheet)
        for x, tx in ((5, "manga panel"), (TW + 10, "Illustrious, best attempt (failed the check)"), (TW * 2 + 15, f"{t['model']} from reference pictures")):
            d.text((x, 6), tx, fill="black", font=fb)
        for k, r in enumerate(part):
            y = 28 + k * (TH + 40)
            sd = rd / "shots" / r["shot_id"]
            for x, f in ((5, sd / "panel.png"), (TW + 10, sd / f"kf_a{ipa[r['shot_id']]['base_attempt']}.png"), (TW * 2 + 15, sd / f"kf_a{ATT}.png")):
                if f.exists():
                    im = Image.open(f).convert("RGB")
                    im.thumbnail((TW, TH))
                    sheet.paste(im, (x, y))
            s = shots[r["shot_id"]]
            d.text((5, y + TH + 2), f"{r['shot_id']}: {', '.join(v['id'] for v in s['visible'])}"[:62], fill=(90, 90, 90), font=f1)
            c = r["check"]
            for li, ln in enumerate(textwrap.wrap(f"{c['verdict']} {c['score']} " + "; ".join(c.get("problems") or []), 64)[:2]):
                d.text((TW * 2 + 15, y + TH + 2 + li * 15), ln, fill=(0, 110, 0) if c["verdict"] == "ok" else (170, 0, 0), font=f1)
        sheet.save(ev / f"gemini_frames_{si // 8 + 1}.jpg", quality=88)
    print(f"sheets: {ev}/gemini_frames_*.jpg")
