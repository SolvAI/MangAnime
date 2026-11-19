"""Comparison sheets for the v3 first frames: one line per shot = manga panel | v2 frame from the same panel | v3 prompt | v3 frame.

  python scripts/make_v3_mosaic.py [--run ep22_v3] [--old ep22_v2] [--rows 14] [--sheets 4]
Writes runs/<run>/eval/frames_v3_{1..}.jpg and frames_v3.html (all shots with a drawn frame).
"""
import argparse
import html
import sys
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.config import P  # noqa: E402
from src.common.io import read_json  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="ep22_v3")
ap.add_argument("--old", default="ep22_v2")
ap.add_argument("--rows", type=int, default=14)
ap.add_argument("--sheets", type=int, default=4)
a = ap.parse_args()
run, old = P("runs") / a.run, P("runs") / a.old
shots = read_json(run / "shotlist.json")["shots"]
state = read_json(run / "frames_state.json") or {}
old_by_panel = {}
for s in (read_json(old / "shotlist.json") or {}).get("shots", []):
    if s.get("source_panels"):
        old_by_panel.setdefault((s["chapter"], s["source_panels"][0]), s)
rows = []
for s in shots:
    sd = run / "shots" / s["shot_id"]
    if s["start"] not in ("panel", "new") or not (sd / "kf.png").exists():
        continue
    st = state.get(s["shot_id"]) or {}
    att = st.get("final_attempt", 0)
    meta = read_json(sd / f"kf_a{att}.json") or {}
    chk = next((c for c in st.get("checks", []) if c.get("attempt") == att), {})
    o = old_by_panel.get((s["chapter"], s.get("panel"))) if s["start"] == "panel" else None
    okf = old / "shots" / o["shot_id"] / "kf.png" if o else None
    rows.append({"shot": s, "panel": sd / "panel_canvas.png" if (sd / "panel_canvas.png").exists() else None, "old_kf": okf if okf and okf.exists() else None,
                 "old_cast": ", ".join(o.get("characters", [])) if o else "", "kf": sd / "kf.png", "prompt": meta.get("prompt", ""),
                 "regions": meta.get("regions") or [], "check": chk, "attempts": len(st.get("checks", []))})
out = run / "eval"
out.mkdir(parents=True, exist_ok=True)
f1 = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
fb = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
TW, TH, PW = 384, 216, 430
step = max(1, len(rows) // (a.rows * a.sheets))
pick = rows[::step][: a.rows * a.sheets]
for si in range(0, len(pick), a.rows):
    part = pick[si : si + a.rows]
    sheet = Image.new("RGB", (TW * 3 + PW + 25, (TH + 34) * len(part) + 30), "white")
    d = ImageDraw.Draw(sheet)
    for x, t in ((5, "manga panel (layout input)"), (TW + 10, "v2 frame from the same panel"), (TW * 2 + 15, "v3 prompt"), (TW * 2 + PW + 20, "v3 frame")):
        d.text((x, 6), t, fill=(0, 0, 0), font=fb)
    for k, r in enumerate(part):
        y = 28 + k * (TH + 34)
        s = r["shot"]
        for x, pth in ((5, r["panel"]), (TW + 10, r["old_kf"]), (TW * 2 + PW + 20, r["kf"])):
            if pth:
                im = Image.open(pth).convert("RGB")
                im.thumbnail((TW, TH))
                sheet.paste(im, (x, y))
        if r["old_cast"]:
            d.text((TW + 10, y + TH + 2), ("v2 cast: " + r["old_cast"])[:58], fill=(150, 0, 0), font=f1)
        txt = f"{s['shot_id']} [{s['start']}] {s['framing']} — " + ", ".join(v["id"] for v in s["visible"]) + "\n" + s.get("image_sentence", "") + "\n" + r["prompt"][70:420]
        d.multiline_text((TW * 2 + 15, y), "\n".join(textwrap.wrap(txt, 66)[:15]), fill=(0, 0, 0), font=f1, spacing=1)
        c = r["check"]
        d.text((TW * 2 + PW + 20, y + TH + 2), (f"check: {c.get('verdict')} {c.get('score')} (attempts {r['attempts']}) " + "; ".join(c.get("problems") or []))[:62],
               fill=(0, 110, 0) if c.get("verdict") == "ok" else (170, 0, 0), font=f1)
        d.text((5, y + TH + 2), f"ch {s['chapter']} panel {s.get('panel')}" if s["start"] == "panel" else "no panel (new frame)", fill=(90, 90, 90), font=f1)
    sheet.save(out / f"frames_v3_{si // a.rows + 1}.jpg", quality=88)
trs = []
for r in rows:
    s, c = r["shot"], r["check"]
    img = lambda p: f"<img src='file://{p}'>" if p else ""          # noqa: E731
    trs.append(f"<tr><td>{img(r['panel'])}<br><small>{'ch %s panel %s' % (s['chapter'], s.get('panel')) if s['start'] == 'panel' else 'no panel (new frame)'}</small></td>"
               f"<td>{img(r['old_kf'])}<br><small>{html.escape(r['old_cast'])}</small></td>"
               f"<td><b>{s['shot_id']}</b> [{s['start']}] {s['framing']}, {s['duration']} s<br><b>{html.escape(', '.join(v['id'] for v in s['visible']))}</b><br>"
               f"<i>{html.escape(s.get('image_sentence', ''))}</i><br><small>{html.escape(r['prompt'])}"
               + "".join(f"<br>region: {html.escape(t)}" for t in r["regions"]) + f"</small><br>motion: {html.escape(s.get('motion', ''))}<br>"
               + "<br>".join(f"<b>{html.escape(l['speaker'])}</b>: {html.escape(l['text'])} <small>[{html.escape(l.get('instruct', ''))}]</small>" for l in s["lines"]) + "</td>"
               f"<td>{img(r['kf'])}<br><small class={'ok' if c.get('verdict') == 'ok' else 'bad'}>check: {c.get('verdict')} {c.get('score')} — {html.escape('; '.join(c.get('problems') or []))}</small></td></tr>")
ok = sum(1 for r in rows if r["check"].get("verdict") == "ok")
(out / "frames_v3.html").write_text(
    "<!doctype html><meta charset=utf-8><title>v3 first frames</title><style>body{font:13px sans-serif;margin:14px}table{border-collapse:collapse}"
    "td,th{border:1px solid #ccc;padding:5px;vertical-align:top}img{width:400px}td:nth-child(3){width:460px}.ok{color:#070}.bad{color:#b00}</style>"
    f"<h2>v3 first frames — {len(rows)} drawn frames, {ok} accepted by the Gemini check</h2>"
    "<table><tr><th>manga panel</th><th>v2 frame from the same panel</th><th>v3 shot, prompt, motion, lines</th><th>v3 frame</th></tr>" + "".join(trs) + "</table>")
print(f"{len(rows)} frames, {ok} accepted; sheets: {out}/frames_v3_*.jpg, {out}/frames_v3.html")
