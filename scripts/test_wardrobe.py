"""Test of the wardrobe pass (D-052): do fixed, coloured outfits per scene stop people changing clothes between shots?

  python scripts/test_wardrobe.py --src ep22_v3 --dst ep22_v3b --scenes 118.4,116.4,117.2 draw    # first frames with the new outfits (ComfyUI)
  python scripts/test_wardrobe.py --src ep22_v3 --dst ep22_v3b --scenes 118.4,116.4,117.2 check   # continuity check, before vs after
Both sides use the FIRST attempt of each frame (kf_a0.png), so the only difference is the outfit text of the prompt.
Writes runs/<dst>/eval/wardrobe_test.json and wardrobe_test_N.jpg (before | after, shot by shot).
"""
import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.config import P  # noqa: E402
from src.common.io import read_json, write_json  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("step", choices=["draw", "check"])
ap.add_argument("--src", default="ep22_v3")
ap.add_argument("--dst", default="ep22_v3b")
ap.add_argument("--scenes", default="118.4,116.4,117.2")
a = ap.parse_args()
src, dst = P("runs") / a.src, P("runs") / a.dst
scenes = a.scenes.split(",")
new = [s for s in read_json(dst / "shotlist.json")["shots"] if s.get("scene_id") in scenes and s["start"] in ("panel", "new")]
ids = {s["shot_id"] for s in new}

if a.step == "draw":
    from src.v3.frames import run_frames

    st = run_frames(dst, read_json(dst / "shotlist.json")["shots"], rounds=0, only=ids)
    fin = [st[i] for i in ids if st.get(i, {}).get("final_verdict")]
    print(f"drawn {len(fin)} frames, {sum(1 for x in fin if x['final_verdict'] == 'ok')} accepted by the frame check")
else:
    from src.eval.continuity import evaluate

    res = {}
    for label, rd in (("before", src), ("after", dst)):
        frames = [{"id": s["shot_id"], "group": s["scene_id"], "scene": s["scene_id"], "path": str(rd / "shots" / s["shot_id"] / "kf_a0.png")}
                  for s in new if (src / "shots" / s["shot_id"] / "kf_a0.png").exists() and (dst / "shots" / s["shot_id"] / "kf_a0.png").exists()]
        res[label] = evaluate(frames, dst / "eval" / f"wardrobe_test_{label}.json", f"wardrobe_{label}")
    st_old, st_new = read_json(src / "frames_state.json") or {}, read_json(dst / "frames_state.json") or {}
    first_ok = lambda st: sum(1 for i in ids if (st.get(i) or {}).get("checks") and st[i]["checks"][0].get("verdict") == "ok")  # noqa: E731
    res["frame_check_first_attempt_ok"] = {"before": first_ok(st_old), "after": first_ok(st_new), "frames": len(ids)}
    write_json(dst / "eval" / "wardrobe_test.json", {"scenes": scenes, **res})
    for k in ("before", "after"):
        s = res[k]
        print(f"{k}: {s['errors']} continuity errors on {s['same_scene_cuts']} same-scene cuts ({s['errors_per_100_same_scene_cuts']}/100), "
              f"outfit {s['errors_by_kind'].get('outfit', 0)}, people who change look {s['people_inconsistent']}/{s['people_followed']}, score {s['mean_score']}")
    print("frame check, first attempt accepted:", res["frame_check_first_attempt_ok"])
    fb = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)
    TW, TH = 336, 192
    for sc in scenes:
        part = [s for s in new if s["scene_id"] == sc][:14]
        sheet = Image.new("RGB", (TW * len(part) + 5 * (len(part) + 1), TH * 2 + 60), "white")
        d = ImageDraw.Draw(sheet)
        d.text((5, 4), f"scene {sc} — top: before (outfits as the storyboard wrote them)   bottom: after the wardrobe pass", fill="black", font=fb)
        for k, s in enumerate(part):
            x = 5 + k * (TW + 5)
            for row, rd in enumerate((src, dst)):
                f = rd / "shots" / s["shot_id"] / "kf_a0.png"
                if f.exists():
                    im = Image.open(f).convert("RGB")
                    im.thumbnail((TW, TH))
                    sheet.paste(im, (x, 24 + row * (TH + 6)))
            d.text((x, 24 + 2 * TH + 10), s["shot_id"] + " " + ",".join(v["id"] for v in s["visible"])[:30], fill=(90, 90, 90), font=fb)
        sheet.save(dst / "eval" / f"wardrobe_test_{sc.replace('.', '_')}.jpg", quality=88)
    print(f"sheets: {dst}/eval/wardrobe_test_*.jpg")
