"""Image-to-image comparison for the keyframe stage: one line per shot =
input panel crop | prompt | Qwen-Image-Edit (v1) | Illustrious first try (short prompt) | Illustrious panel-aware + line-art ControlNet.

  python scripts/make_i2i_mosaic.py [--rows 24] [--per-sheet 8]
Writes runs/ep22_v2/eval/i2i_comparison_v2_<n>.jpg and i2i_comparison_v2.html (all shots; local file, relative image paths).
"""
import argparse
import html
import json
import os
import textwrap

from PIL import Image, ImageDraw, ImageFont

ap = argparse.ArgumentParser()
ap.add_argument("--rows", type=int, default=24)
ap.add_argument("--per-sheet", type=int, default=8)
a = ap.parse_args()
R = "/home/user/komi-poc-work/runs"
V1, FIRST, V2 = f"{R}/ep22/shots", f"{R}/ep22_v2_kf_first", f"{R}/ep22_v2/shots"
font_p = next((p for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf") if os.path.exists(p)), None)
F = lambda n: ImageFont.truetype(font_p, n) if font_p else ImageFont.load_default()  # noqa: E731
sids = [s for s in sorted(os.listdir(V2)) if all(os.path.exists(p) for p in (f"{V2}/{s}/kf_a0.png", f"{V2}/{s}/panel_canvas.png", f"{V1}/{s}/kf_a0.png", f"{FIRST}/{s}/kf_a0.png"))]


def meta(d, sid):
    try:
        return json.load(open(f"{d}/{sid}/kf_a0.json"))
    except Exception:
        return {}


CW, CH, PW = 400, 225, 600


def cell(path):
    return Image.open(path).convert("RGB").resize((CW, CH), Image.LANCZOS)


def prompt_cell(sid):
    im = Image.new("RGB", (PW, CH), (250, 250, 250))
    d = ImageDraw.Draw(im)
    m2 = meta(V2, sid)
    p2 = m2.get("prompt", "")
    p2 = p2.split("komi-san wa komyushou desu, ", 1)[-1]
    y = 4
    d.text((8, y), f"{sid}   drawn by name: {', '.join(m2.get('named') or []) or '-'}   described by look: {m2.get('described', '-')}", fill=(200, 0, 0), font=F(12))
    y += 17
    d.text((8, y), "panel-aware prompt (after quality + series tags):", fill=(0, 90, 0), font=F(11))
    y += 14
    for line in textwrap.wrap(p2, 100)[:13]:
        d.text((8, y), line, fill=(20, 20, 20), font=F(11))
        y += 14
    return im


def header():
    im = Image.new("RGB", (CW * 4 + PW, 30), (30, 30, 30))
    d = ImageDraw.Draw(im)
    for x, t in ((6, "INPUT: manga panel crop"), (CW + 6, "PROMPT (new)"), (CW + PW + 6, "Qwen-Image-Edit 2511, 4-step"), (2 * CW + PW + 6, "Illustrious, short prompt (first try)"),
                 (3 * CW + PW + 6, "Illustrious, panel-aware prompt + line-art ControlNet")):
        d.text((x, 7), t, fill=(255, 255, 255), font=F(13))
    return im


def sheet(rows, out):
    im = Image.new("RGB", (CW * 4 + PW, 30 + len(rows) * (CH + 4)), (255, 255, 255))
    im.paste(header(), (0, 0))
    y = 30
    for sid in rows:
        im.paste(cell(f"{V2}/{sid}/panel_canvas.png"), (0, y))
        im.paste(prompt_cell(sid), (CW, y))
        im.paste(cell(f"{V1}/{sid}/kf_a0.png"), (CW + PW, y))
        im.paste(cell(f"{FIRST}/{sid}/kf_a0.png"), (2 * CW + PW, y))
        im.paste(cell(f"{V2}/{sid}/kf_a0.png"), (3 * CW + PW, y))
        y += CH + 4
    im.save(out, quality=88)
    return out


step = max(1, len(sids) // a.rows)
pick = sids[2::step][: a.rows]
outs = [sheet(pick[i : i + a.per_sheet], f"{R}/ep22_v2/eval/i2i_comparison_v2_{i // a.per_sheet + 1}.jpg") for i in range(0, len(pick), a.per_sheet)]
rows = []
for sid in sids:
    m1, m2 = meta(V1, sid), meta(V2, sid)
    rows.append(f"<tr><td><b>{sid}</b><br><img src='../shots/{sid}/panel_canvas.png'></td><td class='p'><b>Illustrious, panel-aware (named: {html.escape(', '.join(m2.get('named') or []) or '-')})</b><br>"
                f"{html.escape(m2.get('prompt', ''))}<br><br><b>Qwen-Edit</b><br>{html.escape(m1.get('prompt', ''))}</td><td><img src='../../ep22/shots/{sid}/kf_a0.png'></td>"
                f"<td><img src='../../ep22_v2_kf_first/{sid}/kf_a0.png'></td><td><img src='../shots/{sid}/kf_a0.png'></td></tr>")
open(f"{R}/ep22_v2/eval/i2i_comparison_v2.html", "w").write(
    "<!doctype html><meta charset='utf-8'><title>I2I comparison v2</title><style>body{font:13px sans-serif;background:#111;color:#ddd}img{width:380px}td{vertical-align:top;padding:4px}"
    ".p{width:400px;font-size:12px;color:#bbb}th{position:sticky;top:0;background:#222;padding:6px}</style><table><tr><th>input panel crop</th><th>prompts</th><th>Qwen-Image-Edit (v1)</th>"
    "<th>Illustrious, short prompt</th><th>Illustrious, panel-aware + ControlNet</th></tr>" + "".join(rows) + "</table>")
print(len(sids), "shots;", outs)
