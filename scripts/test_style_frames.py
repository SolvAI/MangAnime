"""Style test (D-057): do real frames of the training episodes, given to the painter as art-style references, bring the
first frames closer to the show? Same shots, same shot list, same character / scene / panel references as ep22_v3c; the only
difference is the two style reference frames and the style wording.

  python scripts/test_style_frames.py draw      # 5 first shots of 4 scenes -> runs/style_test (KOMI_STYLE_REFS=2)
  python scripts/test_style_frames.py sheet     # old | new sheets + a blind old-vs-new pair set with a real-frame reference sheet
  python scripts/test_style_frames.py score     # after a rater wrote runs/style_test/blind/answers_*.json

The same harness tests the set references of D-062 (real frames of the school's rooms):
  STYLE_TEST_DIR=set_test STYLE_TEST_SRC=ep22_v3e STYLE_TEST_SCENES=114.1,114.3,117.2,117.5 KOMI_SET_REFS=2 python scripts/test_style_frames.py draw
"""
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image, ImageDraw  # noqa: E402

from src.common.config import P  # noqa: E402
from src.common.io import log, read_json, write_json  # noqa: E402

SRC = os.environ.get("STYLE_TEST_SRC", "ep22_v3c")
OUT = P("runs") / os.environ.get("STYLE_TEST_DIR", "style_test")
SCENES = (os.environ.get("STYLE_TEST_SCENES") or "114.1,116.4,118.1,118.4").split(",")
PER_SCENE = int(os.environ.get("STYLE_TEST_PER_SCENE", "5"))


def old_frame(sid: str) -> Path:
    """The frame as the painter drew it in the source run (before any reframing crop)."""
    d = P("runs") / os.environ.get("STYLE_TEST_OLD", SRC) / "shots" / sid
    return d / "kf_wide.png" if (d / "kf_wide.png").exists() else d / "kf.png"


def subset() -> list[dict]:
    shots = read_json(P("runs") / SRC / "shotlist.json")["shots"]
    out = []
    for sc in SCENES:
        out += [s for s in shots if str(s["scene_id"]) == sc and s["start"] in ("panel", "new")][:PER_SCENE]
    return out


def draw() -> None:
    os.environ.setdefault("KOMI_STYLE_REFS", "2")
    from src.v3.frames_gemini import run_frames_gemini

    OUT.mkdir(parents=True, exist_ok=True)
    shots = subset()
    write_json(OUT / "shotlist.json", {"shots": shots})
    run_frames_gemini(OUT, shots, rounds=1)


def _lab(im: Image.Image, text: str) -> Image.Image:
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 10 + 7 * len(text), 16], fill=(0, 0, 0))
    d.text((4, 2), text, fill=(255, 255, 0))
    return im


def sheet() -> None:
    shots = subset()
    w, h = 480, 270
    rows = []
    for s in shots:
        old, new = old_frame(s["shot_id"]), OUT / "shots" / s["shot_id"] / "kf.png"
        if not (old.exists() and new.exists()):
            continue
        meta = read_json(OUT / "shots" / s["shot_id"] / f"kf_a{(read_json(OUT / 'frames_state.json') or {}).get(s['shot_id'], {}).get('final_attempt', 0)}.json") or {}
        row = Image.new("RGB", (4 * w, h), (30, 30, 30))
        row.paste(_lab(Image.open(old).convert("RGB").resize((w, h)), f"{s['shot_id']} {os.environ.get('STYLE_TEST_OLD', SRC)}"), (0, 0))
        row.paste(_lab(Image.open(new).convert("RGB").resize((w, h)), "new"), (w, 0))
        for i, f in enumerate(((meta.get("set_references") or []) + (meta.get("style_references") or []))[:2]):
            row.paste(_lab(Image.open(f).convert("RGB").resize((w, h)), "real frame given (training episode)"), ((2 + i) * w, 0))
        rows.append(row)
    for k in range(0, len(rows), 10):
        part = rows[k : k + 10]
        im = Image.new("RGB", (4 * w, h * len(part)))
        for i, r in enumerate(part):
            im.paste(r, (0, i * h))
        im.save(OUT / f"style_test_{k // 10 + 1}.jpg", quality=88)
    # blind old-vs-new pairs + a sheet of real frames that were NOT given to the painter
    rng = random.Random(5)
    bd = OUT / "blind"
    bd.mkdir(exist_ok=True)
    key = {}
    done = [s for s in shots if (OUT / "shots" / s["shot_id"] / "kf.png").exists()]
    rng.shuffle(done)
    for i, s in enumerate(done, 1):
        old = Image.open(old_frame(s["shot_id"])).convert("RGB").resize((768, 432))
        new = Image.open(OUT / "shots" / s["shot_id"] / "kf.png").convert("RGB").resize((768, 432))
        flip = rng.random() < 0.5
        a, b = (new, old) if flip else (old, new)
        im = Image.new("RGB", (768 * 2 + 12, 432), (40, 40, 40))
        im.paste(_lab(a, "L"), (0, 0))
        im.paste(_lab(b, "R"), (780, 0))
        im.save(bd / f"pair_{i:02d}.jpg", quality=90)
        key[f"pair_{i:02d}"] = {"shot_id": s["shot_id"], "new": "L" if flip else "R"}
    write_json(OUT / "blind_key.json", key)
    used = {f for s in shots for m in [read_json(OUT / "shots" / s["shot_id"] / "kf_a0.json") or {}] for f in (m.get("style_references") or []) + (m.get("set_references") or [])}
    bank = [b for b in read_json(P("processed") / "v3" / "style_bank.json") if b.get("clean") and b.get("plain") and b["path"] not in used
            and b["place"] in ("classroom", "home living room", "home kitchen", "street", "school hallway") and b["people"] >= 1]
    pick = rng.sample(bank, 12)
    ref = Image.new("RGB", (4 * 480, 3 * 270))
    for i, b in enumerate(pick):
        ref.paste(Image.open(b["path"]).convert("RGB").resize((480, 270)), ((i % 4) * 480, (i // 4) * 270))
    ref.save(bd / "real_reference.jpg", quality=90)
    log(f"style test: {len(rows)} rows on sheets, {len(key)} blind pairs in {bd}")


def score() -> None:
    key = read_json(OUT / "blind_key.json")
    for f in sorted((OUT / "blind").glob("answers_*.json")):
        ans = read_json(f) or {}
        c = {"new": 0, "old": 0, "tie": 0}
        for n, k in key.items():
            v = str((ans.get(n) or {}).get("closer", "")).strip().upper()
            c["tie" if v not in ("L", "R") else ("new" if v == k["new"] else "old")] += 1
        log(f"style test {f.stem}: closer to the show: {c}")
        write_json(OUT / f"score_{f.stem}.json", c)


if __name__ == "__main__":
    {"draw": draw, "sheet": sheet, "score": score}[sys.argv[1]]()
