"""Two readings of the same chapters side by side: who each one sees in every panel, and who it gives each line to.

  python scripts/compare_readings.py --a reading --b reading_pics --chapters 114,116,117,118 [--sheets 2] [--name pics_vs_text]
Readings live in data/processed/v3/<name>/c<chapter>.json. Prints the agreement and writes contact sheets of the panels where
the named cast differs (panel | reading A | reading B) to runs/ep22_v3/eval/reading_<name>_N.jpg, plus a JSON list.
"""
import argparse
import random
import sys
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.config import P  # noqa: E402
from src.common.io import read_json, read_jsonl, write_json  # noqa: E402
from src.v3.cast import is_known  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--a", default="reading")
ap.add_argument("--b", default="reading_pics")
ap.add_argument("--chapters", default="114,116,117,118")
ap.add_argument("--sheets", type=int, default=2)
ap.add_argument("--rows", type=int, default=12)
ap.add_argument("--name", default="pics_vs_text")
ap.add_argument("--run", default="ep22_v3")
a = ap.parse_args()
chapters = [int(c) for c in a.chapters.split(",") if c]


def named(p: dict) -> frozenset:
    return frozenset(v["id"] for v in p.get("visible", []) if is_known(v["id"]))


def speakers(p: dict) -> dict:
    return {" ".join(str(l["text"]).lower().split())[:40]: l["speaker"] for l in p.get("lines", []) if l.get("kind") in ("speech", "thought")}


panels = {(p["chapter"], p["order"]): p for p in read_jsonl(P("processed") / "manga" / "panels.jsonl") if p["chapter"] in chapters}
paths = {(r["volume"], r["page"]): r["path"] for r in read_jsonl(P("interim") / "manga" / "pages.jsonl")}
diff, tot = [], {"panels": 0, "same_cast": 0, "lines": 0, "same_speaker": 0, "named_a": 0, "named_b": 0}
for ch in chapters:
    ra, rb = (read_json(P("processed") / "v3" / d / f"c{ch:03d}.json") for d in (a.a, a.b))
    if not ra or not rb:
        print(f"chapter {ch}: missing in {a.a if not ra else a.b}")
        continue
    pa, pb = {p["n"]: p for p in ra["panels"]}, {p["n"]: p for p in rb["panels"]}
    story = lambda r: {n for sc in r.get("scenes", []) if sc.get("kind", "story") == "story" for n in range(int(sc.get("first_panel") or 0), int(sc.get("last_panel") or -1) + 1)}  # noqa: E731
    keep = (story(ra) & story(rb)) or (set(pa) & set(pb))
    c = {"panels": 0, "same_cast": 0, "lines": 0, "same_speaker": 0}
    for n in sorted(set(pa) & set(pb) & keep):
        A, B = named(pa[n]), named(pb[n])
        c["panels"] += 1
        c["same_cast"] += A == B
        tot["named_a"] += len(A)
        tot["named_b"] += len(B)
        sa, sb = speakers(pa[n]), speakers(pb[n])
        for k in set(sa) & set(sb):
            c["lines"] += 1
            c["same_speaker"] += sa[k] == sb[k]
        if A != B:
            diff.append({"chapter": ch, "panel": n, "a": sorted(A), "b": sorted(B), "a_txt": pa[n].get("what") or pa[n].get("action") or "", "b_txt": pb[n].get("what") or pb[n].get("action") or ""})
    for k in c:
        tot[k] += c[k]
    print(f"chapter {ch}: same named cast in {c['same_cast']}/{c['panels']} story panels; same speaker on {c['same_speaker']}/{c['lines']} shared lines")
print(f"TOTAL {a.a} vs {a.b}: same named cast {tot['same_cast']}/{tot['panels']} panels ({tot['same_cast'] / max(1, tot['panels']):.0%}); "
      f"same speaker {tot['same_speaker']}/{tot['lines']} lines; named people per panel {tot['named_a'] / max(1, tot['panels']):.2f} vs {tot['named_b'] / max(1, tot['panels']):.2f}")
out = P("runs") / a.run / "eval"
out.mkdir(parents=True, exist_ok=True)
write_json(out / f"reading_{a.name}.json", {"a": a.a, "b": a.b, "total": tot, "different": diff})
if a.sheets and diff:
    random.Random(0).shuffle(diff)
    f1 = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    fb = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    TW, TH, CW = 420, 300, 430
    pick = diff[: a.rows * a.sheets]
    for si in range(0, len(pick), a.rows):
        part = pick[si : si + a.rows]
        cols = 2                                           # two entries per line of the sheet
        lines = -(-len(part) // cols)
        sheet = Image.new("RGB", ((TW + CW + 15) * cols, (TH + 12) * lines + 28), "white")
        d = ImageDraw.Draw(sheet)
        d.text((5, 5), f"panels where the named cast differs — A = {a.a}, B = {a.b}  (letters only; judge by eye)", fill=(0, 0, 0), font=fb)
        for k, r in enumerate(part):
            x0, y0 = (k % cols) * (TW + CW + 15) + 5, 28 + (k // cols) * (TH + 12)
            p = panels.get((r["chapter"], r["panel"]))
            if p:
                x, y, w, h = [int(v) for v in p["bbox"]]
                im = Image.open(paths[(p["volume"], p["page"])]).convert("RGB").crop((x, y, x + w, y + h))
                im.thumbnail((TW, TH))
                sheet.paste(im, (x0, y0))
            txt = [f"#{si + k + 1}  ch {r['chapter']} panel {r['panel']}", "A: " + (", ".join(r["a"]) or "nobody named"), "B: " + (", ".join(r["b"]) or "nobody named")]
            yy = y0
            for j, t in enumerate(txt):
                for ln in textwrap.wrap(t, 52):
                    d.text((x0 + TW + 8, yy), ln, fill=(0, 0, 0) if j == 0 else ((0, 70, 170) if j == 1 else (170, 60, 0)), font=fb if j == 0 else f1)
                    yy += 18
                yy += 6
        sheet.save(out / f"reading_{a.name}_{si // a.rows + 1}.jpg", quality=88)
    print(f"{len(diff)} differing panels; sheets: {out}/reading_{a.name}_*.jpg")
