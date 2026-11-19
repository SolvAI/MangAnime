"""VLM pass over parsed manga pages (brief §5.1): characters, expressions/action/framing captions, speaker attribution,
notebook detection. One call per page with numbered panel boxes drawn on the page (set-of-marks), local Qwen 3.8 27B.

  python -m src.manga.caption --chapters 1,2      (needs: python -m src.common.servers only qwen)
Updates data/processed/manga/panels.jsonl in place (characters, caption, shot_size, location, dialogue[].speaker/kind).
"""
from __future__ import annotations

import argparse
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from src.common.config import CFG, P
from src.common.io import log, read_json, read_jsonl, write_jsonl
from src.llm.client import LLM

KINDS = ["speech", "thought", "narration", "notebook", "sfx", "sign"]
SCHEMA = {"type": "object", "required": ["panels"], "properties": {"panels": {"type": "array", "items": {
    "type": "object", "required": ["id", "characters", "caption"],
    "properties": {"id": {"type": ["integer", "string"]}, "characters": {"type": "array"}, "caption": {"type": "string"},
                   "shot_size": {"type": "string"}, "location": {"type": "string"}, "texts": {"type": "array"}}}}}}


def roster_manga() -> str:
    return "\n".join(f"- {c['name']}: {c['manga']}" for c in CFG.characters.roster if c["role"] not in ("crowd",))


SYSTEM = (
    "You read pages of the manga \"Komi Can't Communicate\" (English scanlation, read RIGHT-TO-LEFT) for an anime adaptation. "
    "On the page image every panel has a red box with a red number (its id). For each numbered panel you also get the "
    "OCR'd texts with an index. Use the picture to decide who is in each panel and who says each text.\n"
    "Characters (name them only when the look or the context clearly matches; otherwise 'Classmates' or 'unknown'):\n{roster}\n"
    "Text kinds: speech (balloon with a tail), thought (cloud/soft balloon or inner monologue), narration (rectangular box, "
    "narrator or a character's inner narration), notebook (words Komi wrote in her notebook / on a board / on paper), "
    "sfx (sound-effect lettering), sign (labels, signs, on-screen notes).\n"
    "Return ONE JSON object: {{\"panels\": [{{\"id\": <panel number>, \"characters\": [names visible in the panel], "
    "\"caption\": \"1-2 sentences: who is visible, expressions, poses, action, background\", "
    "\"shot_size\": \"close-up|medium close-up|medium|medium wide|wide|insert\", \"location\": \"few words\", "
    "\"texts\": [{{\"i\": <text index>, \"speaker\": \"<name|Narrator|Classmates|unknown>\", \"kind\": \"<kind>\"}}]}}]}}\n"
    "Cover every numbered panel and every listed text index. Notebook text is always spoken by its writer (usually Komi)."
)


_PAGES: dict = {}


def page_file(volume: int, page: int) -> str:
    if not _PAGES:
        for r in read_jsonl(P("interim") / "manga" / "pages.jsonl"):
            _PAGES[(r["volume"], r["page"])] = r["path"]
    return _PAGES[(volume, page)]


def annotate_page(page_path: str, panels: list[dict], out: Path, scale: float = 1.5) -> Path:
    img = Image.open(page_path).convert("RGB")
    img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
    d = ImageDraw.Draw(img)
    fp = next((p for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",) if Path(p).exists()), None)
    font = ImageFont.truetype(fp, 34) if fp else ImageFont.load_default()
    for p in panels:
        x, y, w, h = [v * scale for v in p["bbox"]]
        d.rectangle([x, y, x + w, y + h], outline=(230, 0, 0), width=4)
        label = str(p["order"])
        tw = d.textlength(label, font=font)
        d.rectangle([x + 4, y + 4, x + 14 + tw, y + 44], fill=(255, 255, 255), outline=(230, 0, 0), width=2)
        d.text((x + 9, y + 5), label, fill=(230, 0, 0), font=font)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out, quality=90)
    return out


def caption_page(llm: LLM, page_panels: list[dict], title: str, prev_summary: str) -> dict:
    src = page_file(page_panels[0]["volume"], page_panels[0]["page"])
    ann = annotate_page(str(src), page_panels, P("cache") / "manga_annot" / f"v{page_panels[0]['volume']:03d}_p{page_panels[0]['page']:03d}.jpg")
    lines = [f"Chapter {page_panels[0]['chapter']}" + (f" \"{title}\"" if title else "") + f", page {page_panels[0]['page']}."]
    if prev_summary:
        lines.append("Previous page (context): " + prev_summary[:500])
    lines.append("Panels and OCR texts:")
    for p in page_panels:
        lines.append(f"Panel {p['order']}:")
        for i, t in enumerate(p["dialogue"]):
            lines.append(f"   [{i}] \"{t['text'][:200]}\"")
        if not p["dialogue"]:
            lines.append("   (no text)")
    return llm.chat([{"role": "system", "content": SYSTEM.format(roster=roster_manga())}, {"role": "user", "content": "\n".join(lines)}],
                    schema=SCHEMA, images=[ann], max_tokens=1800, temperature=0.1, image_max_side=1568, tag="manga_caption")


def run(chapters: list[int], workers: int | None = None) -> None:
    path = P("processed") / "manga" / "panels.jsonl"
    rows = read_jsonl(path)
    titles = {c["chapter"]: c.get("title", "") for c in read_json(P("processed") / "manga" / "chapters.json", [])}
    llm = LLM("local", role="manga_caption")
    names = {c["name"] for c in CFG.characters.roster} | {"Narrator", "Classmates"}
    alias = {a.lower(): c["name"] for c in CFG.characters.roster for a in [c["name"], c["full"], *c.get("aliases", [])]}
    by_page: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        if r["chapter"] in chapters and not r.get("caption"):
            by_page[(r["chapter"], r["volume"], r["page"])].append(r)
    keys = sorted(by_page)
    log(f"manga caption: {len(keys)} pages to do")

    def one(k):
        prev = ""
        try:
            return k, caption_page(llm, by_page[k], titles.get(k[0], ""), prev)
        except Exception as e:
            return k, {"error": str(e)[:200]}

    with ThreadPoolExecutor(workers or int(CFG.llm.local.parallel)) as ex:
        results = list(ex.map(one, keys))
    n_ok = 0
    for k, res in results:
        if "error" in res:
            log(f"page {k}: {res['error']}")
            continue
        got = {}
        for p in res.get("panels", []):
            m = re.search(r"\d+", str(p.get("id")))
            if m:
                got[int(m.group())] = p
        for r in by_page[k]:
            g = got.get(r["order"])
            if not g:
                continue
            chars = []
            for c in g.get("characters") or []:
                c = alias.get(str(c).lower().strip(), str(c).strip())
                if c and c.lower() not in ("unknown", "none") and c not in chars:
                    chars.append(c)
            r["characters"] = chars
            r["caption"] = str(g.get("caption") or "").strip()
            r["shot_size"] = str(g.get("shot_size") or "")
            r["location"] = str(g.get("location") or "")
            for t in g.get("texts") or []:
                try:
                    i = int(t.get("i"))
                except Exception:
                    continue
                if 0 <= i < len(r["dialogue"]):
                    sp = alias.get(str(t.get("speaker") or "").lower().strip(), str(t.get("speaker") or "").strip())
                    r["dialogue"][i]["speaker"] = sp if sp and sp.lower() != "unknown" else None
                    kd = str(t.get("kind") or "").lower()
                    if kd in KINDS:
                        r["dialogue"][i]["kind"] = "sfx" if kd == "sign" and len(r["dialogue"][i]["text"]) < 4 else kd
            n_ok += 1
    write_jsonl(path, rows)
    log(f"manga caption: updated {n_ok} panels")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--chapters", required=True)
    a = ap.parse_args()
    run([int(c) for c in a.chapters.split(",")])
