"""Video (brief §7.3): Wan 2.2 I2V (+LoRA) per shot, chained for long shots, re-timed to animate "on twos".

raw clip:   runs/<run>/shots/<id>/raw_a<attempt>[_k].mp4   (832x480 @16 fps, straight from Wan)
final clip: runs/<run>/shots/<id>/clip.mp4                 (exact shot duration, 23.976 fps, held frames)
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

from src.common.config import CFG
from src.common.io import ffmpeg, log, media_duration, sha, write_json
from src.generate.comfy import Comfy, wf_wan_i2v
from src.generate.prompts import video_prompt

WAN_FPS = 16
OUT_FPS = 24000 / 1001
W, H = 832, 480
MAX_LEN = 81


DRAW_FPS = {"twos": 12.0, "threes": 8.0, "smooth": 16.0}   # drawings per second that Wan frames are mapped to


def wan_length(seconds: float, mode: str = "twos") -> int:
    """Frames to request from Wan. In 'twos'/'threes' every generated frame is ONE drawing held for 2/3 output
    frames (12 / 8 drawings per second), so a 4 s shot needs 49 frames instead of 65: real TV-anime cadence, and
    25-50 % less generation time than rendering at Wan's native 16 fps and throwing frames away."""
    n = int(np.ceil(seconds * DRAW_FPS[mode])) + 1
    n = ((max(17, n) - 1 + 3) // 4) * 4 + 1          # Wan needs 4k+1 frames
    return min(MAX_LEN, n)


def last_frame(video: Path, out: Path) -> Path:
    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, n - 1))
    ok, f = cap.read()
    if not ok:  # some containers mis-report the count: read sequentially
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while True:
            ok2, f2 = cap.read()
            if not ok2:
                break
            f = f2
    cap.release()
    cv2.imwrite(str(out), f)
    return out


def retime(raw: list[Path], out: Path, duration: float, mode: str = "twos") -> Path:
    """Concatenate raw 16 fps segments, then resample to the output rate with held frames.

    twos   -> every generated frame is a drawing held for 2 output frames (12 drawings/s, classic TV-anime cadence)
    threes -> held for 3 output frames (8 drawings/s)
    smooth -> Wan's native 16 fps resampled to 23.976 fps (no holds)
    """
    frames = []
    for i, r in enumerate(raw):
        cap = cv2.VideoCapture(str(r))
        k = 0
        while True:
            ok, f = cap.read()
            if not ok:
                break
            if not (i > 0 and k == 0):  # chained segments start on the previous segment's last frame
                frames.append(f)
            k += 1
        cap.release()
    if not frames:
        raise RuntimeError("no frames decoded from raw clip")
    n_out = max(1, int(round(duration * OUT_FPS)))
    step = {"twos": 2, "threes": 3, "smooth": 1}[mode]
    idx = []
    for j in range(n_out):
        if mode == "smooth":
            k = int(round(j / OUT_FPS * WAN_FPS))
        else:
            k = j // step                             # one generated frame = one drawing, held `step` output frames
        idx.append(min(len(frames) - 1, k))
    out.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    proc = subprocess.Popen(
        [CFG.paths.ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
         "-r", "24000/1001", "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "14", "-pix_fmt", "yuv420p", str(out)],
        stdin=subprocess.PIPE)
    for i in idx:
        proc.stdin.write(frames[i].tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError("ffmpeg retime failed")
    return out


def make_video(comfy: Comfy, run_dir: Path, shot: dict, *, attempt: int = 0, seed: int | None = None, hint: str = "",
               loras_high: list[tuple[str, float]] | None = None, loras_low: list[tuple[str, float]] | None = None,
               mode: str = "twos", steps: int = 4, use_lightx2v: bool = True, trigger: bool = True) -> Path:
    sdir = run_dir / "shots" / shot["shot_id"]
    kf = sdir / "kf.png"
    if not kf.exists():
        raise FileNotFoundError(f"no keyframe for {shot['shot_id']}")
    dur = float(shot["duration"])
    prompt = video_prompt(shot, hint, trigger=trigger and bool(loras_high or loras_low))
    seed = seed if seed is not None else int(sha(shot["shot_id"] + f"v{attempt}", 8), 16) % (2**31)
    raws: list[Path] = []
    remaining, start_img, k = dur, kf, 0
    while remaining > 0.05 and k < 3:
        seg = min(remaining, (MAX_LEN - 1) / DRAW_FPS[mode])
        length = wan_length(seg, mode)
        name = comfy.put_input(start_img, f"{run_dir.name}_{shot['shot_id']}_v{attempt}_{k}.png")
        files = comfy.run(wf_wan_i2v(name, prompt, width=W, height=H, length=length, seed=seed + k, steps=steps,
                                     split=steps // 2, loras_high=loras_high, loras_low=loras_low, use_lightx2v=use_lightx2v,
                                     prefix=f"{run_dir.name}/{shot['shot_id']}_a{attempt}_{k}"), timeout=1800)
        vids = [f for f in files if f.suffix == ".mp4"]
        if not vids:
            raise RuntimeError("Wan workflow produced no video")
        raw = sdir / f"raw_a{attempt}_{k}.mp4"
        shutil.copyfile(vids[0], raw)
        raws.append(raw)
        remaining -= (length - 1) / DRAW_FPS[mode]
        if remaining > 0.05:
            start_img = last_frame(raw, sdir / f"chain_a{attempt}_{k}.png")
        k += 1
    clip = retime(raws, sdir / f"clip_a{attempt}.mp4", dur, mode)
    shutil.copyfile(clip, sdir / "clip.mp4")
    write_json(sdir / f"video_a{attempt}.json", {"prompt": prompt, "seed": seed, "attempt": attempt, "segments": len(raws),
                                                  "duration": dur, "mode": mode, "loras_high": loras_high or [], "loras_low": loras_low or []})
    return clip


WAN_PROFILES = {   # sampling of the clips; chosen by the motion test (scripts/test_motion.py, D-057). KOMI_WAN_PROFILE selects one.
    "base": dict(prompt="", steps=4, pre=0, split=2, shift=5.0, pre_cfg=1.0),          # 4-step distilled sampler, "subtle" wording (v1-v3b)
    "P": dict(prompt="lively", steps=4, pre=0, split=2, shift=5.0, pre_cfg=1.0),       # livelier wording only
    "S": dict(prompt="lively", steps=4, pre=0, split=2, shift=8.0, pre_cfg=1.0),       # + more of the schedule at high noise
    "T1": dict(prompt="lively", steps=4, pre=1, split=2, shift=5.0, pre_cfg=3.5),      # + first step without the distill LoRA, with guidance
    "T2": dict(prompt="lively", steps=6, pre=2, split=3, shift=5.0, pre_cfg=3.5),      # + six steps, two of them guided
}


def wan_profile() -> dict:
    import os

    name = os.environ.get("KOMI_WAN_PROFILE", "base")
    return {"name": name, **WAN_PROFILES[name]}


def make_videos_batched(comfy: Comfy, run_dir: Path, jobs: list[dict], *, mode: str = "twos", steps: int = 4, use_lightx2v: bool = True,
                        loras_high: list[tuple[str, float]] | None = None, loras_low: list[tuple[str, float]] | None = None,
                        trigger: bool | str = True, chunk: int = 24, log_fn=None) -> dict[str, Path | Exception]:
    """Render many shots with ONE expert swap per chunk: high-noise pass for the whole chunk (latents saved), then the
    low-noise pass + decode. jobs: [{"shot": spec, "attempt": int, "hint": str, "seed": int | None}]. Shots longer than one
    Wan segment are chained in a second round from the last decoded frame. Returns {shot_id: clip path | Exception}."""
    from src.generate.comfy import wf_wan_stage

    seg_max = (MAX_LEN - 1) / DRAW_FPS[mode]
    results: dict[str, Path | Exception] = {}
    state = {}
    prof = wan_profile()
    if prof["name"] != "base":
        steps = prof["steps"]
    split = prof["split"] if prof["name"] != "base" else steps // 2
    kw = dict(width=W, height=H, steps=steps, split=split, shift=prof["shift"], use_lightx2v=use_lightx2v)
    for j in jobs:
        s = j["shot"]
        sdir = run_dir / "shots" / s["shot_id"]
        if not (sdir / "kf.png").exists():
            results[s["shot_id"]] = FileNotFoundError("no keyframe")
            continue
        att = j.get("attempt", 0)
        seed = j.get("seed")
        seed = seed if seed is not None else int(sha(s["shot_id"] + f"v{att}", 8), 16) % (2**31)
        state[s["shot_id"]] = {"shot": s, "sdir": sdir, "att": att, "seed": seed, "remaining": float(s["duration"]), "start": sdir / "kf.png",
                               "raws": [], "prompt": video_prompt(s, j.get("hint", ""), trigger=trigger if (loras_high or loras_low) else False, style=prof["prompt"])}
    rnd = 0
    while True:
        todo = [sid for sid, st in state.items() if st["remaining"] > 0.05 and sid not in results and rnd < 3]
        if not todo:
            break
        for c0 in range(0, len(todo), chunk):
            part = todo[c0 : c0 + chunk]
            lat: dict[str, str] = {}
            pre: dict[str, str] = {}
            for stage in (("pre",) if prof["pre"] else ()) + ("high", "low"):
                for sid in part:
                    if sid in results:
                        continue
                    st = state[sid]
                    seg = min(st["remaining"], seg_max)
                    length = wan_length(seg, mode)
                    tag = f"{run_dir.name}_{sid}_a{st['att']}_{rnd}"
                    try:
                        name = comfy.put_input(st["start"], f"{tag}.png")
                        if stage == "pre":      # first step(s) of the high-noise expert without the distill LoRA, with real guidance
                            files = comfy.run(wf_wan_stage("high", name, st["prompt"], length=length, seed=st["seed"] + rnd, **{**kw, "use_lightx2v": False},
                                                           cfg=prof["pre_cfg"], loras_high=loras_high, start_step=0, end_step=prof["pre"],
                                                           prefix=f"latents/{tag}_pre"), timeout=1800)
                            lf = [f for f in files if f.suffix == ".latent"]
                            if not lf:
                                raise RuntimeError("guided pre-pass produced no latent")
                            pre[sid] = comfy.put_input(lf[0], f"{tag}_pre.latent")
                            lf[0].unlink(missing_ok=True)
                        elif stage == "high":
                            more = dict(latent_name=pre[sid], start_step=prof["pre"], end_step=split) if prof["pre"] else {}
                            files = comfy.run(wf_wan_stage("high", name, st["prompt"], length=length, seed=st["seed"] + rnd, **kw, loras_high=loras_high,
                                                           prefix=f"latents/{tag}", **more), timeout=1800)
                            lf = [f for f in files if f.suffix == ".latent"]
                            if not lf:
                                raise RuntimeError("high-noise stage produced no latent")
                            lat[sid] = comfy.put_input(lf[0], f"{tag}.latent")
                            lf[0].unlink(missing_ok=True)
                            if sid in pre:
                                (comfy.io / "input" / pre.pop(sid)).unlink(missing_ok=True)
                        else:
                            files = comfy.run(wf_wan_stage("low", name, st["prompt"], length=length, seed=st["seed"] + rnd, **kw, loras_low=loras_low,
                                                           latent_name=lat[sid], prefix=f"{run_dir.name}/{sid}_a{st['att']}_{rnd}"), timeout=1800)
                            vids = [f for f in files if f.suffix == ".mp4"]
                            if not vids:
                                raise RuntimeError("low-noise stage produced no video")
                            raw = st["sdir"] / f"raw_a{st['att']}_{rnd}.mp4"
                            shutil.copyfile(vids[0], raw)
                            st["raws"].append(raw)
                            st["remaining"] -= (length - 1) / DRAW_FPS[mode]
                            if st["remaining"] > 0.05:
                                st["start"] = last_frame(raw, st["sdir"] / f"chain_a{st['att']}_{rnd}.png")
                            (comfy.io / "input" / lat[sid]).unlink(missing_ok=True)
                    except Exception as e:
                        results[sid] = e
                        if "out of memory" in str(e).lower():
                            comfy.free()
            if log_fn:
                log_fn(f"wan batch round {rnd}: {min(c0 + chunk, len(todo))}/{len(todo)} shots")
        rnd += 1
    for sid, st in state.items():
        if sid in results:
            continue
        try:
            clip = retime(st["raws"], st["sdir"] / f"clip_a{st['att']}.mp4", float(st["shot"]["duration"]), mode)
            shutil.copyfile(clip, st["sdir"] / "clip.mp4")
            write_json(st["sdir"] / f"video_a{st['att']}.json", {"prompt": st["prompt"], "seed": st["seed"], "attempt": st["att"], "segments": len(st["raws"]),
                                                                  "duration": st["shot"]["duration"], "mode": mode, "loras_high": loras_high or [], "loras_low": loras_low or [],
                                                                  "profile": prof})
            results[sid] = clip
        except Exception as e:
            results[sid] = e
    return results


def still_clip(image: Path, out: Path, duration: float, camera: str = "static", w: int = W, h: int = H) -> Path:
    """Keyframe + pan/zoom (the animatic baseline and the last rung of the repair ladder)."""
    n = max(1, int(round(duration * OUT_FPS)))
    cam = (camera or "static").lower()
    z0, z1, x0, x1, y0, y1 = 1.0, 1.04, 0.5, 0.5, 0.5, 0.5        # default: very slow push-in
    if "push" in cam or "zoom in" in cam or "zoom-in" in cam:
        z0, z1 = 1.0, 1.12
    elif "pull" in cam or "zoom out" in cam or "zoom-out" in cam:
        z0, z1 = 1.12, 1.0
    elif "pan left" in cam:
        z0 = z1 = 1.10; x0, x1 = 0.65, 0.35
    elif "pan right" in cam or "pan" in cam or "track" in cam:
        z0 = z1 = 1.10; x0, x1 = 0.35, 0.65
    elif "tilt up" in cam:
        z0 = z1 = 1.10; y0, y1 = 0.65, 0.35
    elif "tilt down" in cam or "tilt" in cam:
        z0 = z1 = 1.10; y0, y1 = 0.35, 0.65
    img = cv2.imread(str(image))
    big = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)
    out.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        [CFG.paths.ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
         "-r", "24000/1001", "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "14", "-pix_fmt", "yuv420p", str(out)],
        stdin=subprocess.PIPE)
    BH, BW = big.shape[:2]
    for j in range(n):
        a = j / max(1, n - 1)
        a = a * a * (3 - 2 * a)                                 # ease in/out
        z = z0 + (z1 - z0) * a
        cw, ch = BW / z, BH / z
        cx = (x0 + (x1 - x0) * a) * BW
        cy = (y0 + (y1 - y0) * a) * BH
        x = float(np.clip(cx - cw / 2, 0, BW - cw))
        y = float(np.clip(cy - ch / 2, 0, BH - ch))
        M = np.array([[w / cw, 0, -x * w / cw], [0, h / ch, -y * h / ch]], dtype=np.float32)   # sub-pixel accurate crop
        proc.stdin.write(cv2.warpAffine(big, M, (w, h), flags=cv2.INTER_AREA).tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError("ffmpeg still_clip failed")
    return out
