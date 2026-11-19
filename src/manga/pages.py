"""Manga page extraction: CBZ -> data/interim/manga/pages/vNN/pNNN.<ext> + pages.jsonl (originals untouched)."""
from __future__ import annotations

import argparse
import io
import re
import zipfile
from pathlib import Path

from PIL import Image

from src.common.config import P
from src.common.io import log, read_jsonl, write_jsonl

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}


def _nat(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def volume_id(name: str) -> int | None:
    m = re.search(r"Volume (\d+)", name)
    if m:
        return int(m.group(1))
    m = re.match(r"(\d+)-(\d+)\.cbz", name)  # loose chapter pack, e.g. 285-302.cbz -> volume 900+
    return 900 + int(m.group(1)) if m else None


def extract(volumes: list[int] | None = None) -> list[dict]:
    out_root = P("interim") / "manga" / "pages"
    index_path = P("interim") / "manga" / "pages.jsonl"
    rows = {(r["volume"], r["page"]): r for r in read_jsonl(index_path)}
    for cbz in sorted((P("raw") / "manga").glob("*.cbz"), key=lambda p: _nat(p.name)):
        vol = volume_id(cbz.name)
        if vol is None or (volumes and vol not in volumes):
            continue
        d = out_root / f"v{vol:03d}"
        if (d / ".done").exists():
            continue
        d.mkdir(parents=True, exist_ok=True)
        z = zipfile.ZipFile(cbz)
        names = sorted([n for n in z.namelist() if Path(n).suffix.lower() in IMG_EXT], key=_nat)
        for i, n in enumerate(names, 1):
            data = z.read(n)
            ext = Path(n).suffix.lower().replace(".jpeg", ".jpg")
            f = d / f"p{i:03d}{ext}"
            f.write_bytes(data)
            with Image.open(io.BytesIO(data)) as im:
                w, h = im.size
            rows[(vol, i)] = {"volume": vol, "page": i, "path": str(f), "w": w, "h": h, "src": f"{cbz.name}:{n}"}
        (d / ".done").write_text(str(len(names)))
        log(f"volume {vol}: {len(names)} pages")
    out = [rows[k] for k in sorted(rows)]
    write_jsonl(index_path, out)
    return out


def load_pages(volumes: list[int] | None = None) -> list[dict]:
    rows = read_jsonl(P("interim") / "manga" / "pages.jsonl")
    return [r for r in rows if not volumes or r["volume"] in volumes]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--volumes", default=None, help="comma list, default all")
    a = ap.parse_args()
    rows = extract([int(v) for v in a.volumes.split(",")] if a.volumes else None)
    log(f"{len(rows)} pages indexed")
