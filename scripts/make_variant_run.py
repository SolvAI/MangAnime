"""Clone part of a finished run so another video model can re-render the SAME keyframes (LoRA vs base, twos vs threes).

  python scripts/make_variant_run.py --src ep22 --dst ep22_lora [--every 3] [--offset 1]
Copies the shot list (every N-th non-title shot), each shot's kf.png, and the source run's clips as
  shots/<id>/base.mp4 (the source run's final clip), animatic.mp4, real.mp4
so that `python -m src.eval.ab_judge --run <dst> --versus base|animatic|real` compares the new render against them.
Then: python -m src.generate.pipeline --run <dst> --stages video --mode threes --wan-lora-high <f> --wan-lora-low <f>
"""
import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.config import P  # noqa: E402
from src.common.io import read_json, write_json  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--dst", required=True)
ap.add_argument("--every", type=int, default=3)
ap.add_argument("--offset", type=int, default=1)
ap.add_argument("--only-wan", action="store_true", help="only shots whose final clip in the source run is a Wan clip")
a = ap.parse_args()

src, dst = P("runs") / a.src, P("runs") / a.dst
sl = read_json(src / "shotlist.json")
state = read_json(src / "state.json") or {}
shots = [s for s in sl["shots"] if "TITLE_CARD" not in (s.get("notes") or "")]
if a.only_wan:
    shots = [s for s in shots if (state.get(s["shot_id"]) or {}).get("final") == "wan"]
pick = shots[a.offset % a.every :: a.every]
n = 0
for s in pick:
    sd, dd = src / "shots" / s["shot_id"], dst / "shots" / s["shot_id"]
    if not (sd / "kf.png").exists():
        continue
    dd.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(sd / "kf.png", dd / "kf.png")
    for a_name, b_name in (("clip.mp4", "base.mp4"), ("animatic.mp4", "animatic.mp4"), ("real.mp4", "real.mp4")):
        if (sd / a_name).exists():
            shutil.copyfile(sd / a_name, dd / b_name)
    n += 1
out = dict(sl)
out["shots"] = pick
out["variant_of"] = a.src
write_json(dst / "shotlist.json", out)
print(f"{a.dst}: {n} shots (every {a.every} of {len(shots)}) from {a.src}")
