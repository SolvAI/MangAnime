"""Tiny synthetic fixture for the labeling page (schemas of panels.jsonl / shots.jsonl / alignment.jsonl).

    python -m tools.align_label.fixture --out /home/user/komi-poc-work/tmp/align_fixture
    python -m tools.align_label.app --processed /home/user/komi-poc-work/tmp/align_fixture --port 8302
"""
from __future__ import annotations

import json
from pathlib import Path


def make_fixture(out: str | Path, n_chapters: int = 2, pages: int = 2, panels_per_page: int = 4, n_shots: int = 24,
                 episode: int = 1) -> Path:
    from PIL import Image, ImageDraw

    out = Path(out)
    for d in ("manga/pages", "anime/kf", "align/gold"):
        (out / d).mkdir(parents=True, exist_ok=True)
    panels, order_rows = [], []
    for ch in range(1, n_chapters + 1):
        order = 0
        for pg in range(1, pages + 1):
            page = Image.new("L", (800, 1200), 255)
            d = ImageDraw.Draw(page)
            path = out / "manga" / "pages" / f"c{ch:03d}_p{pg:02d}.png"
            for k in range(panels_per_page):
                x, y, w, h = 30 + (k % 2) * 380, 30 + (k // 2) * 580, 360, 550
                d.rectangle((x, y, x + w, y + h), outline=0, width=5)
                d.ellipse((x + 90, y + 120, x + 270, y + 330), outline=0, width=4)
                d.text((x + 20, y + 20), f"ch{ch} p{pg} panel {k + 1}", fill=0)
                order += 1
                panels.append({"chapter": ch, "page": pg, "panel": k + 1, "order": order, "bbox": [x, y, w, h],
                               "image": str(path), "characters": ["Komi"] if k % 2 else ["Tadano"],
                               "dialogue": [{"speaker": "Tadano" if k % 2 == 0 else "Komi", "text": f"line {order} of chapter {ch}",
                                             "kind": "speech" if k % 2 == 0 else "notebook"}] if k != 3 else [],
                               "caption": f"synthetic panel {order}"})
                order_rows.append((ch, order))
            page.save(path)
    shots = []
    t = 0.0
    for s in range(1, n_shots + 1):
        kfs = []
        for k in range(2):
            p = out / "anime" / "kf" / f"ep{episode:02d}_s{s:04d}_{k}.jpg"
            im = Image.new("RGB", (427, 240), ((37 * s) % 255, (91 * s) % 255, (53 * s + 80 * k) % 255))
            d = ImageDraw.Draw(im)
            d.ellipse((150 + 20 * k, 50, 280 + 20 * k, 190), fill=(245, 225, 205))
            d.text((10, 10), f"ep{episode} shot {s} kf{k}", fill=(255, 255, 255))
            im.save(p, quality=88)
            kfs.append(str(p))
        dur = 2.0 + (s % 4)
        shots.append({"episode": episode, "shot": s, "start": round(t, 2), "end": round(t + dur, 2), "keyframes": kfs,
                      "dialogue": [{"speaker": "Tadano", "text": f"subtitle of shot {s}", "start": t + 0.2, "end": t + 1.5}] if s % 3 else []})
        t += dur
    align = []
    for i, (ch, order) in enumerate(order_rows):
        if i % 3 == 2:
            continue                                   # some panels have no automatic match
        s = 1 + (i * n_shots) // len(order_rows)
        align.append({"chapter": ch, "panel_order": order, "shots": [{"episode": episode, "shot": s}] + ([{"episode": episode, "shot": s + 1}] if i % 4 == 0 and s < n_shots else []),
                      "match": "dialogue" if i % 2 == 0 else "visual", "confidence": round(0.55 + 0.4 * ((i * 7) % 10) / 10, 2)})

    def dump(path: Path, rows):
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))

    dump(out / "manga" / "panels.jsonl", panels)
    dump(out / "anime" / "shots.jsonl", shots)
    dump(out / "align" / "alignment.jsonl", align)
    (out / "align" / "gold" / "queue.json").write_text(json.dumps({"episodes": [episode]}))
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    print(make_fixture(ap.parse_args().out))
