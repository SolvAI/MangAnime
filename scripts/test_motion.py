"""Motion test (D-057): the clips move a sixth as much as the real show (mean optical flow 0.13 vs 0.73) while the real show's
camera is static in 91 % of the captioned training shots, so the gap is character animation, not camera work.

  python scripts/test_motion.py render      # GPU: 10 shots of ep22_v3b x 4 variants (same first frames, same seeds)
  python scripts/test_motion.py score       # flow / warp per clip, Gemini clip check, one sheet per shot

Variants against A = the clip as rendered in ep22_v3b (4-step distilled sampler, "subtle" wording):
  P   livelier wording of the motion prompt, same sampler
  S   P + sampler shift 8 instead of 5 (more of the schedule at high noise)
  T1  P + the first of the 4 steps without the distill LoRA and with guidance 3.5
  T2  P + 6 steps: two without the distill LoRA at guidance 3.5, one distilled high-noise, three low-noise
"""
import random
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.config import P  # noqa: E402
from src.common.io import log, read_json, sha, write_json  # noqa: E402

SRC = "ep22_v3b"
OUT = P("runs") / "motion_test"
VARIANTS = {
    "P": dict(prompt="lively", steps=4, pre=0, split=2, shift=5.0),
    "S": dict(prompt="lively", steps=4, pre=0, split=2, shift=8.0),
    "T1": dict(prompt="lively", steps=4, pre=1, split=2, shift=5.0, pre_cfg=3.5),
    "T2": dict(prompt="lively", steps=6, pre=2, split=3, shift=5.0, pre_cfg=3.5),
    # the same as P, but every Wan frame is shown at Wan's own 16 fps instead of being held as one of 12 drawings per second
    "P16": dict(prompt="lively", steps=4, pre=0, split=2, shift=5.0, mode="smooth"),
}


def pick_shots(n_by=(("none", 3), ("subtle", 4), ("clear", 3))) -> list[dict]:
    rd = P("runs") / SRC
    shots = {s["shot_id"]: s for s in read_json(rd / "shotlist.json")["shots"]}
    st = read_json(rd / "clips_state.json")
    rng = random.Random(3)
    out = []
    for motion, n in n_by:
        pool = sorted(sid for sid, c in st.items() if c.get("final") == "wan" and len(c.get("checks", [])) == 1 and c["checks"][0].get("motion") == motion
                      and shots[sid]["start"] in ("panel", "new") and 3.0 <= float(shots[sid]["duration"]) <= 6.6
                      and (rd / "shots" / sid / "raw_a0_0.mp4").exists() and not (rd / "shots" / sid / "raw_a0_1.mp4").exists())
        out += [shots[sid] for sid in rng.sample(pool, min(n, len(pool)))]
    return out


def render() -> None:
    from src.common import servers
    from src.generate.comfy import Comfy, wf_wan_stage
    from src.generate.prompts import video_prompt
    from src.generate.video import DRAW_FPS, H, MAX_LEN, W, wan_length

    rd = P("runs") / SRC
    shots = pick_shots()
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "shots.json", [s["shot_id"] for s in shots])
    (OUT / "A").mkdir(exist_ok=True)
    for s in shots:
        shutil.copyfile(rd / "shots" / s["shot_id"] / "raw_a0_0.mp4", OUT / "A" / f"{s['shot_id']}.mp4")
    servers.only("comfy")
    comfy = Comfy()
    timing = read_json(OUT / "timing.json") or {}
    for name, v in VARIANTS.items():
        vdir = OUT / name
        vdir.mkdir(exist_ok=True)
        todo = [s for s in shots if not (vdir / f"{s['shot_id']}.mp4").exists()]
        if not todo:
            continue
        t0 = time.time()
        job = {}
        for s in todo:
            sid = s["shot_id"]
            job[sid] = {"img": comfy.put_input(rd / "shots" / sid / "kf.png", f"mt_{sid}.png"), "prompt": video_prompt(s, trigger=False, style=v["prompt"]),
                        "seed": int(sha(sid + "v0", 8), 16) % (2**31),
                        "length": wan_length(min(float(s["duration"]), (MAX_LEN - 1) / DRAW_FPS[v.get("mode", "twos")]), v.get("mode", "twos"))}
        kw = dict(width=W, height=H, steps=v["steps"], split=v["split"], shift=v["shift"])

        def latent_of(files, tag):
            lf = [f for f in files if f.suffix == ".latent"]
            if not lf:
                raise RuntimeError("no latent")
            n = comfy.put_input(lf[0], f"{tag}.latent")
            lf[0].unlink(missing_ok=True)
            return n

        try:
            if v["pre"]:
                for sid, j in job.items():
                    j["lat0"] = latent_of(comfy.run(wf_wan_stage("high", j["img"], j["prompt"], length=j["length"], seed=j["seed"], cfg=v["pre_cfg"], use_lightx2v=False,
                                                                 start_step=0, end_step=v["pre"], prefix=f"latents/mt_{name}_{sid}_pre", **kw), timeout=1800), f"mt_{name}_{sid}_pre")
                for sid, j in job.items():
                    j["lat"] = latent_of(comfy.run(wf_wan_stage("high", j["img"], j["prompt"], length=j["length"], seed=j["seed"], latent_name=j["lat0"],
                                                                start_step=v["pre"], end_step=v["split"], prefix=f"latents/mt_{name}_{sid}", **kw), timeout=1800), f"mt_{name}_{sid}")
            else:
                for sid, j in job.items():
                    j["lat"] = latent_of(comfy.run(wf_wan_stage("high", j["img"], j["prompt"], length=j["length"], seed=j["seed"], prefix=f"latents/mt_{name}_{sid}", **kw),
                                                   timeout=1800), f"mt_{name}_{sid}")
            for sid, j in job.items():
                files = comfy.run(wf_wan_stage("low", j["img"], j["prompt"], length=j["length"], seed=j["seed"], latent_name=j["lat"], prefix=f"motion_test/{name}_{sid}", **kw), timeout=1800)
                vids = [f for f in files if f.suffix == ".mp4"]
                shutil.copyfile(vids[0], vdir / f"{sid}.mp4")
                for k in ("lat0", "lat"):
                    if j.get(k):
                        (comfy.io / "input" / j[k]).unlink(missing_ok=True)
        except Exception as e:                                             # noqa: BLE001
            log(f"motion test {name}: {str(e)[:300]}")
            continue
        timing[name] = {"clips": len(todo), "seconds": round(time.time() - t0, 1), "per_clip": round((time.time() - t0) / len(todo), 1)}
        write_json(OUT / "timing.json", timing)
        write_json(vdir / "prompts.json", {sid: j["prompt"] for sid, j in job.items()})
        log(f"motion test {name}: {len(todo)} clips in {(time.time() - t0) / 60:.1f} min ({timing[name]['per_clip']} s per clip)")
    log("MOTION_TEST_RENDERED")


def score() -> None:
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np
    from PIL import Image, ImageDraw

    from src.eval import video_metrics as vm
    from src.llm.client import encode_image, vision
    from src.qa.evidence import sample_frames
    from src.v3.check import SCHEMA, SYS

    shots = {s["shot_id"]: s for s in read_json(P("runs") / SRC / "shotlist.json")["shots"]}
    ids = read_json(OUT / "shots.json")
    names = ["A"] + [n for n in VARIANTS if (OUT / n).exists()]
    res: dict = {}

    def check(name: str, sid: str) -> dict:
        s = shots[sid]
        frames = sample_frames(OUT / name / f"{sid}.mp4", OUT / "check" / name / sid, n=4, tag="f")
        content = [{"type": "text", "text": f"INTENDED: {s.get('image_sentence', '')} MOTION: {s.get('motion', '')} "
                                            f"People: {', '.join(v['id'] for v in s.get('visible', [])) or 'none'}. Four frames follow."}]
        content += [{"type": "image_url", "image_url": {"url": encode_image(f, 768, 85)}} for f in frames]
        try:
            return vision("clip_check").chat([{"role": "system", "content": SYS}, {"role": "user", "content": content}], schema=SCHEMA,
                                             max_tokens=2000, effort="low", media_resolution="medium", tag=f"motiontest:{name}:{sid}")
        except Exception as e:                                             # noqa: BLE001
            return {"verdict": "?", "motion": "?", "error": str(e)[:100]}

    from src.generate.video import DRAW_FPS, MAX_LEN, retime

    for name in names:
        clips = [(sid, OUT / name / f"{sid}.mp4") for sid in ids if (OUT / name / f"{sid}.mp4").exists()]
        mode = "threes" if name == "A" else (VARIANTS.get(name) or {}).get("mode", "twos")      # ep22_v3b was rendered on threes (D-033)
        rows = {}
        for sid, p in clips:
            m = vm.analyze_clip(p, metrics=["motion", "warp"], max_pairs=24, flow_height=288)
            rows[sid] = {"mean_flow": (m.get("motion") or {}).get("mean_flow"), "static_ratio": (m.get("motion") or {}).get("static_ratio"),
                         "warp_error": (m.get("warp") or {}).get("warp_error")}
            # the clip as it enters the episode (23.976 fps, held drawings), first 5 s of the shot so that all variants cover the same time
            rt = OUT / "retimed" / name / f"{sid}.mp4"
            rt.parent.mkdir(parents=True, exist_ok=True)
            if not rt.exists():
                retime([p], rt, min(float(shots[sid]["duration"]), (MAX_LEN - 1) / DRAW_FPS["smooth"]), mode)
            m2 = vm.analyze_clip(rt, metrics=["flicker", "motion"], max_pairs=24, flow_height=288)
            rows[sid].update({"out_flow": (m2.get("motion") or {}).get("mean_flow"), "out_static": (m2.get("motion") or {}).get("static_ratio"),
                              "out_held": (m2.get("flicker") or {}).get("held_frame_ratio"), "out_fps": (m2.get("flicker") or {}).get("effective_fps")})
        with ThreadPoolExecutor(6) as ex:
            for (sid, _), c in zip(clips, ex.map(lambda x: check(name, x[0]), clips)):
                rows[sid].update({"verdict": c.get("verdict"), "seen_motion": c.get("motion"), "score": c.get("score"), "problems": c.get("problems")})
        res[name] = rows
        f = [r["mean_flow"] for r in rows.values() if r["mean_flow"] is not None]
        log(f"motion test {name}: mean flow {np.mean(f):.3f} (median {np.median(f):.3f}), static pairs {np.mean([r['static_ratio'] for r in rows.values()]):.0%}, "
            f"warp {np.mean([r['warp_error'] for r in rows.values()]):.2f}, clip check ok {sum(1 for r in rows.values() if r['verdict'] == 'ok')}/{len(rows)}, "
            f"motion seen {dict((k, sum(1 for r in rows.values() if r['seen_motion'] == k)) for k in ('none', 'subtle', 'clear', 'chaotic'))}; "
            f"as cut into the episode: flow {np.mean([r['out_flow'] for r in rows.values()]):.3f} (median {np.median([r['out_flow'] for r in rows.values()]):.3f}), "
            f"static pairs {np.mean([r['out_static'] for r in rows.values()]):.0%}, effective fps {np.mean([r['out_fps'] or 0 for r in rows.values()]):.1f}")
    # cadence alone: the SAME raw Wan frames of variant P shown on threes (8 drawings/s), on twos (12) and at Wan's 16 fps
    cad = {}
    if (OUT / "P").exists():
        import cv2
        for mode in ("threes", "twos", "smooth"):
            vals = []
            for sid in ids:
                raw = OUT / "P" / f"{sid}.mp4"
                if not raw.exists():
                    continue
                cap = cv2.VideoCapture(str(raw)); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
                rt = OUT / "retimed" / f"P_as_{mode}" / f"{sid}.mp4"
                rt.parent.mkdir(parents=True, exist_ok=True)
                if not rt.exists():
                    retime([raw], rt, (n - 1) / DRAW_FPS[mode], mode)
                m2 = vm.analyze_clip(rt, metrics=["flicker", "motion"], max_pairs=24, flow_height=288)
                vals.append(((m2.get("motion") or {}).get("mean_flow"), (m2.get("motion") or {}).get("static_ratio"), (m2.get("flicker") or {}).get("effective_fps") or 0,
                             (m2.get("flicker") or {}).get("held_frame_ratio") or 0))
            cad[mode] = {"mean_flow": float(np.mean([v[0] for v in vals])), "static_ratio": float(np.mean([v[1] for v in vals])),
                         "effective_fps": float(np.mean([v[2] for v in vals])), "held_frame_ratio": float(np.mean([v[3] for v in vals]))}
            log(f"cadence, same frames of P shown {mode}: flow {cad[mode]['mean_flow']:.3f}, static pairs {cad[mode]['static_ratio']:.0%}, "
                f"effective fps {cad[mode]['effective_fps']:.1f}, held frames {cad[mode]['held_frame_ratio']:.0%}")
    write_json(OUT / "score.json", {"variants": res, "timing": read_json(OUT / "timing.json"), "cadence": cad})
    print("shot        " + "".join(f"{n:>22s}" for n in names))
    for sid in ids:
        print(f"{sid:12s}" + "".join((f"{res[n][sid]['mean_flow']:8.2f} {str(res[n][sid]['verdict']):>4s}/{str(res[n][sid]['seen_motion']):8s}" if sid in res[n] else " " * 22) for n in names))
    # one sheet per shot: a row of 6 frames per variant
    (OUT / "sheets").mkdir(exist_ok=True)
    for sid in ids:
        rows = []
        for name in names:
            p = OUT / name / f"{sid}.mp4"
            if not p.exists() or sid not in res[name]:
                continue
            fr = sample_frames(p, OUT / "check" / name / sid, n=6, tag="s")
            row = Image.new("RGB", (6 * 320, 184), (0, 0, 0))
            for i, f in enumerate(fr):
                row.paste(Image.open(f).convert("RGB").resize((320, 184)), (i * 320, 0))
            d = ImageDraw.Draw(row)
            r = res[name][sid]
            lab = f"{name}  flow {r['mean_flow']:.2f}  {r['verdict']} / {r['seen_motion']}"
            d.rectangle([0, 0, 12 + 7 * len(lab), 16], fill=(0, 0, 0))
            d.text((4, 2), lab, fill=(255, 255, 0))
            rows.append(row)
        sheet = Image.new("RGB", (6 * 320, 184 * len(rows)), (0, 0, 0))
        for i, r in enumerate(rows):
            sheet.paste(r, (0, i * 184))
        sheet.save(OUT / "sheets" / f"{sid}.jpg", quality=88)
    log(f"sheets: {OUT / 'sheets'}")


if __name__ == "__main__":
    {"render": render, "score": score}[sys.argv[1]]()
