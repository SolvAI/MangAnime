"""OCR for every detected text region with PaddleOCR-VL 1.6 (GGUF, served locally by llama.cpp on :8189).

Used for manga bubbles / notebook / narration / SFX and for on-screen text in anime frames (brief §3b).
Results are cached by image-content hash, so reruns are free and offline replays work.
"""
from __future__ import annotations

import base64
import hashlib
import io
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
from PIL import Image, ImageOps

from src.common.config import P
from src.common.io import read_json, write_json

OCR_URL = "http://127.0.0.1:8189/v1/chat/completions"
_CACHE = P("cache") / "ocr"


def prep_crop(img: Image.Image, box, pad: float = 0.06, min_side: int = 96, max_side: int = 1024) -> Image.Image:
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    px, py = w * pad + 2, h * pad + 2
    crop = img.crop((max(0, int(x1 - px)), max(0, int(y1 - py)), min(img.width, int(x2 + px)), min(img.height, int(y2 + py))))
    crop = crop.convert("RGB")
    # low-resolution scans: upscale small regions so glyphs are >= ~20 px tall
    scale = 2.0 if max(crop.size) < 500 else 1.0
    if min(crop.size) * scale < min_side:
        scale = min_side / max(1, min(crop.size))
    if max(crop.size) * scale > max_side:
        scale = max_side / max(crop.size)
    if abs(scale - 1.0) > 1e-3:
        crop = crop.resize((max(8, round(crop.width * scale)), max(8, round(crop.height * scale))), Image.LANCZOS)
    return ImageOps.expand(crop, border=8, fill="white")


def collapse_repeats(t: str, max_run: int = 3) -> str:
    """SFX lettering makes the OCR model loop ("DOOM DOOM DOOM ..."): keep at most `max_run` repeats of a token."""
    out, run = [], 0
    for w in t.split():
        if out and w.lower() == out[-1].lower():
            run += 1
            if run >= max_run:
                continue
        else:
            run = 0
        out.append(w)
    return " ".join(out)


def _one(crop: Image.Image, timeout: float = 300.0, max_tokens: int = 128) -> str:
    buf = io.BytesIO()
    crop.save(buf, "PNG")
    raw = buf.getvalue()
    key = hashlib.sha256(raw).hexdigest()[:32]
    cpath = _CACHE / key[:2] / f"{key}.json"
    hit = read_json(cpath)
    if hit is not None:
        return hit["text"]
    body = {
        "model": "paddleocr-vl",
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(raw).decode()}},
            {"type": "text", "text": "OCR:"},
        ]}],
        "max_tokens": max_tokens, "temperature": 0.0,
    }
    last = None
    for _ in range(3):
        try:
            r = httpx.post(OCR_URL, json=body, timeout=timeout)
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"] or ""
            text = clean_text(text)
            write_json(cpath, {"text": text}, indent=None)
            return text
        except Exception as e:  # retry transient server errors
            last = e
    raise RuntimeError(f"OCR failed: {last}")


def clean_text(t: str) -> str:
    t = t.replace("\r", "\n")
    # join hyphenated line breaks ("EMBARRAS-\nING" -> "EMBARRASING"), then collapse whitespace
    t = re.sub(r"(\w)-\n(\w)", r"\1\2", t)
    t = re.sub(r"\s*\n\s*", " ", t)
    t = re.sub(r"([・.…·\-~!?])\1{4,}", r"\1\1\1", t)      # dotted leaders / long repeats
    t = re.sub(r"\s+", " ", t).strip()
    return collapse_repeats(t)


def ocr_crops(crops: list[Image.Image], workers: int = 8, max_tokens: list[int] | None = None) -> list[str]:
    if not crops:
        return []
    mt = max_tokens or [128] * len(crops)
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(lambda a: _one(a[0], max_tokens=a[1]), zip(crops, mt)))


def ocr_page(page_path: str | Path, boxes: list[list[float]], workers: int = 8, essential: list[bool] | None = None) -> list[str]:
    img = Image.open(page_path).convert("RGB")
    mt = [160 if (essential is None or e) else 32 for e in (essential or [True] * len(boxes))]
    return ocr_crops([prep_crop(img, b) for b in boxes], workers, mt)


def ocr_region(page_path: str | Path, box: list[float], max_tokens: int = 48) -> str:
    """OCR an arbitrary rectangle (used for the 'Comm. N Title' chapter banners)."""
    img = Image.open(page_path).convert("RGB")
    return _one(prep_crop(img, box, pad=0.0), max_tokens=max_tokens)
