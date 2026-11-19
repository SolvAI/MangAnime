"""M3 benchmark of the Wan LoRA on held-out-style prompts (brief §3b step 7, §9 M3 "sample grid vs real frames").

Protocol: take real first frames of HELD-OUT shots (never trained on), caption them with the same template as the
training captions, and let each variant continue the shot. Because the first frame is real, differences come from
the video model only, and the real continuation of the shot is the reference:
  variants = base+lightx2v, base+lightx2v+LoRA (smooth 16 fps), ... as given on the command line
Outputs runs/lora_grid/<name>/: clips, a contact sheet per shot (rows = real + variants, columns = time), metrics.json
(held-frame ratio, flow, warp error, DOVER per variant vs real; FVD vs the real continuations).

  python -m src.eval.lora_grid build --n 16                       # pick shots, cut the real 5 s clips (CPU)
  python -m src.eval.lora_grid caption                            # Qwen captions of the real clips (needs qwen)
  python -m src.eval.lora_grid run --variant base                 # needs comfy
  python -m src.eval.lora_grid run --variant lora1000 --high wan22-i2v-high-v1.safetensors --low wan22-i2v-low-v1.safetensors
  python -m src.eval.lora_grid report
"""
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

from src.anime.oped import is_content, non_content
from src.common.config import CFG, P
from src.common.io import ffmpeg, log, read_json, read_jsonl, write_json, write_jsonl

OUT = P("runs") / "lora_grid"
FPS, NF = 16, 81


def build(n: int = 16, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    cands = []
    for ep in CFG.split.heldout_episodes:
        shots = read_json(P("interim") / "anime" / f"ep{ep:02d}" / "shots_raw.json") or []
        nc = non_content(ep)
        subs = read_json(P("processed") / "anime" / "subs" / f"ep{ep:02d}.json", [])
        for s in shots:
            if s["end"] - s["start"] >= NF / FPS + 0.3 and is_content(ep, s["start"], s["end"], nc) and not any(
                    e["kind"] in ("title", "sign") and e["start"] < s["end"] and e["end"] > s["start"] for e in subs):
                cands.append({"ep": ep, **s})
    rng.shuffle(cands)
    rows = []
    for s in cands[:n]:
        cid = f"ep{s['ep']:02d}_s{s['shot']:04d}"
        d = OUT / "real"
        d.mkdir(parents=True, exist_ok=True)
        clip = d / f"{cid}.mp4"
        if not clip.exists():
            src = next((P("raw") / "anime").glob(f"ep{s['ep']:02d}_*.mkv"))
            ffmpeg("-ss", f"{s['start'] + 0.10:.3f}", "-i", str(src), "-t", f"{NF / FPS + 0.2:.3f}", "-map", "0:v:0",
                   "-vf", f"fps={FPS},scale=-2:480:flags=lanczos,crop=832:480,format=yuv420p", "-frames:v", str(NF),
                   "-c:v", "libx264", "-crf", "12", "-an", str(clip))
        cap = cv2.VideoCapture(str(clip))
        frames = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
        cap.release()
        if len(frames) != NF:
            continue
        cv2.imwrite(str(d / f"{cid}_first.png"), frames[0])
        for tag, i in (("a", 0), ("m", NF // 2), ("b", NF - 1)):
            cv2.imwrite(str(d / f"{cid}_{tag}.jpg"), frames[i])
        rows.append({"id": cid, "ep": s["ep"], "shot": s["shot"], "start": s["start"], "real": str(clip), "first": str(d / f"{cid}_first.png")})
    write_jsonl(OUT / "shots.jsonl", rows)
    log(f"lora grid: {len(rows)} held-out shots")
    return rows


def caption() -> None:
    from src.anime.caption import caption_many
    from src.train.build_wan_dataset import make_caption_text

    rows = read_jsonl(OUT / "shots.jsonl")
    items = [{"frames": [OUT / "real" / f"{r['id']}_{t}.jpg" for t in "amb"]} for r in rows]
    for r, c in zip(rows, caption_many(items)):
        if c and "error" not in c:
            r["caption"] = c
            r["prompt"] = make_caption_text(c)
    write_jsonl(OUT / "shots.jsonl", rows)


def run(variant: str, high: str | None, low: str | None, strength: float = 1.0, use_lightx2v: bool = True, steps: int = 4,
        keep_servers: bool = False) -> None:
    """Two batched passes (all high-noise halves, then all low-noise halves), like the episode renderer."""
    from src.common import servers
    from src.generate.comfy import Comfy, wf_wan_stage

    if not keep_servers:
        servers.only("comfy")
    comfy = Comfy()
    rows = read_jsonl(OUT / "shots.jsonl")
    d = OUT / variant
    d.mkdir(parents=True, exist_ok=True)
    import time

    todo = [r for r in rows if not (d / f"{r['id']}.mp4").exists()]
    t0 = time.time()
    lat, names, prompts = {}, {}, {}
    lh = [(high, strength)] if high else None
    ll = [(low, strength)] if low else None
    for stage in ("high", "low"):
        for r in todo:
            rid = r["id"]
            try:
                if stage == "high":
                    prompt = r.get("prompt") or "komi_anime_style, anime."
                    if not (high or low):
                        prompt = prompt.replace("komi_anime_style, ", "")
                    prompts[rid] = prompt
                    names[rid] = comfy.put_input(r["first"], f"grid_{rid}.png")
                    files = comfy.run(wf_wan_stage("high", names[rid], prompt, length=NF, seed=1234, steps=steps, split=steps // 2,
                                                   use_lightx2v=use_lightx2v, cfg=1.0 if use_lightx2v else 3.5, loras_high=lh,
                                                   prefix=f"latents/grid_{variant}_{rid}"), timeout=3600)
                    lf = [f for f in files if f.suffix == ".latent"]
                    lat[rid] = comfy.put_input(lf[0], f"grid_{variant}_{rid}.latent")
                    lf[0].unlink(missing_ok=True)
                elif rid in lat:
                    files = comfy.run(wf_wan_stage("low", names[rid], prompts[rid], length=NF, seed=1234, steps=steps, split=steps // 2,
                                                   use_lightx2v=use_lightx2v, cfg=1.0 if use_lightx2v else 3.5, loras_low=ll,
                                                   latent_name=lat[rid], prefix=f"lora_grid/{variant}_{rid}"), timeout=3600)
                    vids = [f for f in files if f.suffix == ".mp4"]
                    if vids:
                        shutil.copyfile(vids[0], d / f"{rid}.mp4")
                    (comfy.io / "input" / lat[rid]).unlink(missing_ok=True)
            except Exception as e:
                log(f"lora grid {variant} {rid} {stage}: {str(e)[:200]}")
    n = sum(1 for r in todo if (d / f"{r['id']}.mp4").exists())
    write_json(d / "meta.json", {"variant": variant, "high": high, "low": low, "strength": strength, "lightx2v": use_lightx2v, "steps": steps,
                                  "sec_per_clip": round((time.time() - t0) / max(1, n), 1), "n": n})
    log(f"lora grid variant {variant}: {n} clips, {(time.time() - t0) / max(1, n):.0f}s/clip")


def _frames(path: str, idx=(0, 20, 40, 60, 80), w: int = 320) -> list[np.ndarray]:
    cap = cv2.VideoCapture(path)
    out = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, f = cap.read()
        if ok:
            out.append(cv2.resize(f, (w, int(f.shape[0] * w / f.shape[1]))))
    cap.release()
    return out


def report() -> dict:
    from src.eval.episode_eval import clip_stats
    from src.eval.fvd import fvd

    rows = read_jsonl(OUT / "shots.jsonl")
    variants = sorted(p.name for p in OUT.iterdir() if p.is_dir() and p.name not in ("real", "sheets") and (p / "meta.json").exists())
    (OUT / "sheets").mkdir(exist_ok=True)
    for r in rows:
        strips = []
        for name, path in [("real", r["real"])] + [(v, str(OUT / v / f"{r['id']}.mp4")) for v in variants]:
            if not Path(path).exists():
                continue
            fr = _frames(path)
            if len(fr) < 5:
                continue
            strip = np.concatenate(fr, 1)
            cv2.putText(strip, name, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
            cv2.putText(strip, name, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            strips.append(strip)
        if strips:
            cv2.imwrite(str(OUT / "sheets" / f"{r['id']}.jpg"), np.concatenate(strips, 0), [cv2.IMWRITE_JPEG_QUALITY, 85])
    real = [Path(r["real"]) for r in rows]
    res = {"n_shots": len(rows), "real": clip_stats(real, OUT / "stats_real.json"), "variants": {}}
    for v in variants:
        clips = [OUT / v / f"{r['id']}.mp4" for r in rows if (OUT / v / f"{r['id']}.mp4").exists()]
        st = clip_stats(clips, OUT / f"stats_{v}.json")
        try:
            f = fvd([str(x) for x in real], [str(x) for x in clips])
            st["fvd_vs_real"] = round(float(f["fvd"]), 2)
            st["kvd_vs_real"] = round(float(f.get("kvd", float("nan"))), 3)
        except Exception as e:
            st["fvd_error"] = str(e)[:120]
        st["meta"] = read_json(OUT / v / "meta.json")
        res["variants"][v] = st
    write_json(OUT / "metrics.json", res)
    log(f"lora grid report: {OUT / 'metrics.json'}; sheets in {OUT / 'sheets'}")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "caption", "run", "report"])
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--variant", default="base")
    ap.add_argument("--high", default=None)
    ap.add_argument("--low", default=None)
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--no-lightx2v", action="store_true")
    ap.add_argument("--steps", type=int, default=4)
    a = ap.parse_args()
    if a.cmd == "build":
        build(a.n)
    elif a.cmd == "caption":
        caption()
    elif a.cmd == "run":
        run(a.variant, a.high, a.low, a.strength, not a.no_lightx2v, a.steps)
    else:
        report()
