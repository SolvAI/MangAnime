"""Assemble parsed manga: Magi detections + PaddleOCR-VL text -> pages_parsed, chapters.json, panels.jsonl.

  python -m src.manga.build --volumes 1,2
Requires: magi_detect outputs (data/interim/manga/magi) and the OCR server (python -m src.common.servers start ocr).
Captions / character names / speakers are added later by src.manga.caption (VLM) into the same panels.jsonl schema.
"""
from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

from PIL import Image

from src.common.config import P
from src.common.io import log, read_json, read_jsonl, write_json, write_jsonl
from src.manga.ocr import ocr_page, ocr_region
from src.manga.pages import load_pages

BANNER_HEAD = re.compile(r"^comm(unication)?\s*[.:]?\s*\d{0,3}\s*[.:]?$", re.I)
BANNER_RE = re.compile(r"comm(?:unication)?\s*[.:]?\s*(\d{1,3})\s*[.:\-–—]?\s*(.*)$", re.I)
START_RE = re.compile(r"\bcomm?unication\s*[:#.]?\s*(\d{1,3})\b", re.I)
END_RE = re.compile(r"\bcomm(?:unication)?\s*[.:]?\s*(\d{1,3})\s*[-–—:.]*\s*(?:the\s+)?end\b", re.I)


def parse_page(page: dict, magi_dir: Path, out_dir: Path) -> dict | None:
    f = out_dir / f"v{page['volume']:03d}" / f"p{page['page']:03d}.json"
    hit = read_json(f)
    if hit is not None:
        return hit
    m = read_json(magi_dir / f"v{page['volume']:03d}" / f"p{page['page']:03d}.json")
    if m is None:
        return None
    texts = ocr_page(page["path"], m["texts"], essential=m.get("essential")) if m["texts"] else []
    rec = dict(m)
    rec["path"] = page["path"]
    rec["ocr"] = texts
    # chapter banners are pills like "Comm. (2) Peaceful" / "Comm. (1) END": the digit sits in a black circle and is
    # missed by the text detector, so re-read a strip to the right of every "Comm." box as one line.
    banners = []
    for b, t in zip(m["texts"], texts):
        if BANNER_HEAD.match(t.strip()):
            x1, y1, x2, y2 = b
            w, h = x2 - x1, y2 - y1
            strip = [max(0, x1 - 0.3 * w), max(0, y1 - 0.6 * h), min(page["w"], x1 + 7.0 * w), min(page["h"], y2 + 0.6 * h)]
            try:
                banners.append({"box": strip, "text": ocr_region(page["path"], strip)})
            except Exception:
                pass
    rec["banners"] = banners
    write_json(f, rec, indent=None)
    return rec


def detect_chapters(parsed: list[dict]) -> list[dict]:
    """Find chapter spans from 'Communication N' headers and 'Comm. N END' banners (monotonic in N)."""
    starts: dict[int, tuple[int, int]] = {}
    ends: dict[int, tuple[int, int]] = {}
    titles: dict[int, str] = {}
    seq = sorted(parsed, key=lambda r: (r["volume"], r["page"]))
    pos = {(r["volume"], r["page"]): i for i, r in enumerate(seq)}
    for r in seq:
        for bn in r.get("banners", []):
            mb = BANNER_RE.search(bn["text"])
            if not mb:
                continue
            n, rest = int(mb.group(1)), mb.group(2).strip()
            if re.search(r"\bend\b", rest, re.I):
                ends.setdefault(n, (r["volume"], r["page"]))
            else:
                starts.setdefault(n, (r["volume"], r["page"]))
                titles.setdefault(n, rest)
        for t in r.get("ocr", []):
            me = END_RE.search(t)
            if me:
                ends.setdefault(int(me.group(1)), (r["volume"], r["page"]))
                continue
            ms = START_RE.search(t)
            if ms and len(t) < 80:
                n = int(ms.group(1))
                starts.setdefault(n, (r["volume"], r["page"]))
    # unnumbered end markers: an "End" box next to a "Comm." mention whose digit/box was merged with other lettering.
    # The page gets the number of the next numbered END minus one (if that chapter has no end yet).
    numbered_pages = {v for v in list(starts.values()) + list(ends.values())}
    loose_ends = []
    for r in seq:
        key = (r["volume"], r["page"])
        if key in numbered_pages:
            continue
        texts = [t.strip().lower() for t in r.get("ocr", [])]
        if any(t in ("end", "end.") for t in texts) and any(re.search(r"\bcomm\b\.?", t) for t in texts):
            loose_ends.append(key)
    for key in loose_ends:
        later = sorted((pos[v], n) for n, v in ends.items() if pos[v] > pos[key])
        if later and (later[0][1] - 1) not in ends and all(not (pos[key] < pos[v] < later[0][0]) for v in starts.values()):
            ends[later[0][1] - 1] = key
    nums = sorted(set(starts) | set(ends))
    chapters = []
    for n in nums:
        s, e = starts.get(n), ends.get(n)
        if s is None:  # start = page after the previous chapter's end (or previous start)
            prev_e = ends.get(n - 1)
            if prev_e is not None and pos[prev_e] + 1 < len(seq):
                r = seq[pos[prev_e] + 1]
                s = (r["volume"], r["page"])
        if e is None:
            nxt = starts.get(n + 1)
            if nxt is not None and pos[nxt] - 1 >= 0:
                r = seq[pos[nxt] - 1]
                e = (r["volume"], r["page"])
        if s is None and e is not None:
            # no start marker and no previous end (e.g. first chapter of a volume): walk back over story pages
            k = pos[e]
            marked = {v for m, v in list(starts.items()) + list(ends.items()) if m != n}
            while k - 1 >= 0 and pos[e] - (k - 1) <= 24 and seq[k - 1]["volume"] == e[0] and (seq[k - 1]["volume"], seq[k - 1]["page"]) not in marked \
                    and (len(seq[k - 1].get("panels", [])) >= 2      # story page (front matter = single full-page "panel")
                         or sum(1 for t, es in zip(seq[k - 1].get("ocr", []), seq[k - 1].get("essential", [])) if es and len(t) > 3) >= 1):
                k -= 1
            if k < pos[e]:
                s = (seq[k]["volume"], seq[k]["page"])
        if s is None or e is None or pos[e] < pos[s] or pos[e] - pos[s] > 40:
            continue
        # prologue pages before the banner belong to the chapter (e.g. the colour opening of chapter 1)
        prev_e = ends.get(n - 1)
        if prev_e is not None and 0 < pos[s] - pos[prev_e] - 1 <= 3:
            r0 = seq[pos[prev_e] + 1]
            s = (r0["volume"], r0["page"])
        elif prev_e is None:
            k = pos[s]
            while k - 1 >= 0 and pos[s] - (k - 1) <= 3 and sum(1 for t, es in zip(seq[k - 1].get("ocr", []), seq[k - 1].get("essential", [])) if es and len(t) > 12) >= 2:
                k -= 1
            s = (seq[k]["volume"], seq[k]["page"])
        chapters.append({"chapter": n, "volume": s[0], "start_page": s[1], "end_volume": e[0], "end_page": e[1],
                         "n_pages": pos[e] - pos[s] + 1, "title": titles.get(n, ""), "start_found": n in starts, "end_found": n in ends})
    # drop non-monotonic outliers (OCR misreads of the number)
    clean, last_pos = [], -1
    for c in chapters:
        p0 = pos[(c["volume"], c["start_page"])]
        if p0 > last_pos:
            clean.append(c)
            last_pos = pos[(c["end_volume"], c["end_page"])]
    return clean


def text_kind(rec: dict, ti: int) -> str:
    if not rec["essential"][ti]:
        return "sfx"
    if str(ti) in rec["text_char"] or str(ti) in rec.get("text_tail", {}):
        return "speech"
    return "narration"


def build_panels(parsed: list[dict], chapters: list[dict]) -> list[dict]:
    by_key = {(r["volume"], r["page"]): r for r in parsed}
    seq = sorted(by_key)
    idx = {k: i for i, k in enumerate(seq)}
    img_root = P("processed") / "manga" / "panel_img"
    rows = []
    for ch in chapters:
        order = 0
        i0, i1 = idx[(ch["volume"], ch["start_page"])], idx[(ch["end_volume"], ch["end_page"])]
        for k in seq[i0 : i1 + 1]:
            r = by_key[k]
            page_img = None
            texts_by_panel = defaultdict(list)
            for ti, pi in enumerate(r["text_panel"]):
                texts_by_panel[pi].append(ti)
            chars_by_panel = defaultdict(list)
            for ci, pi in enumerate(r["char_panel"]):
                chars_by_panel[pi].append(ci)
            for pi, box in enumerate(r["panels"]):
                x1, y1, x2, y2 = box
                out = img_root / f"v{r['volume']:03d}" / f"p{r['page']:03d}_{pi}.jpg"
                if not out.exists():
                    if page_img is None:
                        page_img = Image.open(r["path"]).convert("RGB")
                    out.parent.mkdir(parents=True, exist_ok=True)
                    page_img.crop((int(x1), int(y1), int(x2), int(y2))).save(out, quality=92)
                dialogue = []
                for ti in texts_by_panel.get(pi, []):
                    txt = r["ocr"][ti] if ti < len(r["ocr"]) else ""
                    if not txt or BANNER_HEAD.match(txt.strip()) or txt.strip().lower() in ("end", (ch.get("title") or "").strip().lower()):
                        continue                                  # chapter banner pieces are not dialogue
                    dialogue.append({"speaker": None, "text": txt, "kind": text_kind(r, ti), "box": [round(v, 1) for v in r["texts"][ti]],
                                     "char_idx": r["text_char"].get(str(ti)), "text_idx": ti})
                rows.append({
                    "chapter": ch["chapter"], "volume": r["volume"], "page": r["page"], "panel": pi, "order": order,
                    "bbox": [round(x1, 1), round(y1, 1), round(x2 - x1, 1), round(y2 - y1, 1)],
                    "characters": [], "char_boxes": [[round(v, 1) for v in r["characters"][ci]] for ci in chars_by_panel.get(pi, [])],
                    "dialogue": dialogue, "caption": None, "image": str(out),
                })
                order += 1
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--volumes", default="")
    ap.add_argument("--assemble-only", action="store_true", help="skip OCR; rebuild chapters/panels from parsed pages")
    a = ap.parse_args()
    vols = [int(v) for v in a.volumes.split(",") if v]
    magi_dir = P("interim") / "manga" / "magi"
    out_dir = P("interim") / "manga" / "parsed"
    parsed = []
    pages = [] if a.assemble_only else load_pages(vols)
    for i, pg in enumerate(pages):
        rec = parse_page(pg, magi_dir, out_dir)
        if rec is not None:
            parsed.append(rec)
        if (i + 1) % 50 == 0:
            log(f"ocr {i + 1}/{len(pages)} pages")
    # chapters over everything parsed so far (not only the requested volumes)
    all_parsed = []
    for f in sorted(out_dir.glob("v*/p*.json")):
        all_parsed.append(read_json(f))
    chapters = detect_chapters(all_parsed)
    write_json(P("processed") / "manga" / "chapters.json", chapters)
    panels = build_panels(all_parsed, chapters)
    # keep captions that already exist (caption stage is expensive)
    old = {(r["chapter"], r["order"]): r for r in read_jsonl(P("processed") / "manga" / "panels.jsonl")}
    for r in panels:
        o = old.get((r["chapter"], r["order"]))
        if o and o.get("caption") and o["image"] == r["image"] and [d["text"] for d in o["dialogue"]] == [d["text"] for d in r["dialogue"]]:
            r.update({k: o[k] for k in ("characters", "dialogue", "caption") if k in o})
            for k in ("shot_size", "location", "notes"):
                if k in o:
                    r[k] = o[k]
    write_jsonl(P("processed") / "manga" / "panels.jsonl", panels)
    log(f"chapters: {len(chapters)} ({chapters[0]['chapter'] if chapters else '-'}..{chapters[-1]['chapter'] if chapters else '-'}), panels: {len(panels)}")


if __name__ == "__main__":
    main()
