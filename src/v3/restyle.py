"""Style pass over finished first frames (D-058): the show's own line and colour through a LoRA trained on training episodes.

  python -m src.v3.restyle --run style_test --strength 0.4            # draws shots/<id>/kf_restyle_s40.png, prints the look numbers
  python -m src.v3.restyle --run ep22_v3e --strength 0.4 --apply      # ... and replaces kf.png where the frame check accepts the result

The worker (src/generate/restyle.py) runs in venvs/sdxl on the GPU. The picture that goes in is the frame as the image model
drew it (kept as kf_gemini.png); a restyled frame replaces it only if the same frame check that accepted the original
accepts it too (people present and on-model, all-ages), so the pass cannot trade the cast for the look.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.common.config import CFG, P
from src.common.io import log, read_json, write_json
from src.v3.style_bank import place_of, time_of

PY = Path("/home/user/komi-poc-work/venvs/sdxl/bin/python")
TRIGGER = "komistyle"


def prompt_for(shot: dict) -> str:
    slug = str(shot.get("slug") or "")
    desc = " ".join([slug, str(shot.get("location") or "").replace("_", " "), " ".join(map(str, shot.get("background_tags") or []))])
    place = place_of(desc)
    n = len([v for v in shot.get("visible", []) if v.get("id") != "Classmates"])
    who = "no humans" if n == 0 else ("1 person" if n == 1 else f"{min(n, 6)} people")
    return f"{TRIGGER}, anime screencap, {place if place != 'other' else 'room'}, {time_of(slug or desc)}, {shot.get('framing') or 'medium'} shot, {who}"


def run(run: str, lora: str = "komi_style_v1", strength: float = 0.4, scale: float = 1.0, only: set[str] | None = None, apply: bool = False) -> dict:
    from src.eval.frame_style import look_stats
    from src.v3.frames import check_frame

    rd = P("runs") / run
    shots = [s for s in read_json(rd / "shotlist.json")["shots"] if s["start"] in ("panel", "new") and (only is None or s["shot_id"] in only)]
    tag = f"restyle_s{int(round(strength * 100))}" + ("" if scale == 1.0 else f"_w{int(round(scale * 100))}")
    jobs = []
    for s in shots:
        sd = rd / "shots" / s["shot_id"]
        if not (sd / "kf.png").exists():
            continue
        if not (sd / "kf_gemini.png").exists():
            shutil.copyfile(sd / "kf_sharp.png" if (sd / "kf_sharp.png").exists() else sd / "kf.png", sd / "kf_gemini.png")
        jobs.append({"id": s["shot_id"], "image": str(sd / "kf_gemini.png"), "prompt": prompt_for(s), "out": str(sd / f"kf_{tag}.png"),
                     "seed": sum(map(ord, s["shot_id"]))})
    jf = rd / f"{tag}_jobs.jsonl"
    jf.write_text("\n".join(__import__("json").dumps(j) for j in jobs) + "\n")
    r = subprocess.run([str(PY), str(Path(CFG.paths.repo) / "src" / "generate" / "restyle.py"), "--jobs", str(jf), "--lora", lora, "--strength", str(strength),
                        "--scale", str(scale)], capture_output=True, text=True)
    log((r.stdout.strip().splitlines() or ["restyle worker: no output"])[-1])
    if r.returncode != 0:
        log("restyle worker failed: " + r.stderr.strip()[-400:])
        return {}
    done = [j for j in jobs if Path(j["out"]).exists()]
    out = {"run": run, "tag": tag, "frames": len(done), "look_before": look_stats([Path(j["image"]) for j in done]), "look_after": look_stats([Path(j["out"]) for j in done])}
    log(f"restyle {run} {tag}: look before {out['look_before']}")
    log(f"restyle {run} {tag}: look after  {out['look_after']}")
    if apply:
        by_id = {s["shot_id"]: s for s in shots}

        def one(j: dict) -> tuple[str, dict]:
            sd = rd / "shots" / j["id"]
            shutil.copyfile(j["out"], sd / "kf_a91.png")
            write_json(sd / "kf_a91.json", {"backend": "restyle", "lora": lora, "strength": strength, "from": "kf_gemini.png", "prompt": j["prompt"]})
            c = check_frame(rd, by_id[j["id"]], 91)
            ok = c.get("verdict") == "ok" and not c.get("unsuitable")
            if ok:
                shutil.copyfile(j["out"], sd / "kf.png")
            return j["id"], {"kept": ok, "score": c.get("score"), "problems": (c.get("problems") or [])[:3]}

        with ThreadPoolExecutor(8) as ex:
            out["checks"] = dict(ex.map(one, done))
        out["kept"] = sum(1 for v in out["checks"].values() if v["kept"])
        log(f"restyle {run} {tag}: {out['kept']} of {len(done)} restyled frames accepted by the frame check and kept")
    write_json(rd / f"{tag}.json", out)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--lora", default="komi_style_v1")
    ap.add_argument("--strength", type=float, default=0.4)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--only", default="")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    run(a.run, a.lora, a.strength, a.scale, set(a.only.split(",")) if a.only else None, a.apply)
