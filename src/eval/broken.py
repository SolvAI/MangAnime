"""Deliberately broken variants of real shots, for QA-judge calibration (brief §3b "Judge calibration").

Labels are known by construction: a clip written by :func:`make_real` is a real shot (pass), a clip written by
:func:`make_broken` is broken (fail). Both go through exactly the same decode -> x264/AAC encode path, so the encoder
itself carries no label information.

API
---
    KINDS                                      tuple of all failure kinds
    KIND_INFO[kind]                            description, requirements, which evidence should reveal it
    make_broken(src_clip, kind, out_path, **kw) -> dict   (metadata incl. "effect_mean" / "effect_local")
    make_real(src_clip, out_path)               -> dict   (same encode path, untouched content)
    build_set(src_dir, out_dir, n_broken, n_real, kinds=KINDS, seed=0) -> list of label rows (+ labels.jsonl)

CLI
---
    python -m src.eval.broken --src <dir of real clips> --out <dir> --n 50 [--n-real 50] [--kinds a,b,c] [--seed 0]

writes ``<out>/clips/c_0000.mp4 ...`` (opaque names, shuffled) and ``<out>/labels.jsonl`` with rows
``{"path", "label": "real|broken", "kind", "src", "params", "expected_dur", "need_audio", "allow_static",
"hard_fail", "technical_ok", "technical_reasons", "effect_mean", "effect_local"}``.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import tempfile
from fractions import Fraction
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from . import video_metrics as vm

# kind -> description / requirements / expected detection
#   needs:      "motion" = the source shot must contain visible motion, "audio" = non-silent audio track,
#               "other" = a second clip (kw other=...)
#   hard_fail:  technical_check(path, expected_dur=<label expected_dur>, need_audio=<label need_audio>) must fail it
#   evidence:   which evidence in the judge bundle should reveal it
KIND_INFO: dict[str, dict[str, Any]] = {
    "flicker": dict(desc="random brightness / colour jumps between frames", needs=[], hard_fail=False,
                    evidence=["flicker.luma_flicker", "flicker.chroma_flicker", "warp.warp_error"]),
    "frame_shuffle": dict(desc="frames shuffled inside short windows (temporal order destroyed)", needs=["motion"],
                          hard_fail=False, evidence=["flicker.hf_ratio_unique", "warp.warp_error", "captions"]),
    "freeze": dict(desc="video frozen on one frame while the audio continues", needs=["motion"], hard_fail=True,
                   evidence=["technical.frozen_video", "flicker.repeat_ratio", "motion.static_frame_ratio"]),
    "melt": dict(desc="progressive spatial warping of a region (melting face / limbs)", needs=[], hard_fail=False,
                 evidence=["captions/artifact checklist", "identity", "warp.warp_error", "frames"]),
    "ghost": dict(desc="double exposure with a time-shifted, slightly offset copy", needs=[], hard_fail=False,
                  evidence=["captions/artifact checklist", "dover.technical", "frames"]),
    "noise_blur": dict(desc="heavy blur plus per-frame noise", needs=[], hard_fail=False,
                       evidence=["dover.technical", "flicker.mean_abs_diff", "frames"]),
    "color_shift": dict(desc="wrong palette (hue rotation / saturation change)", needs=[], hard_fail=False,
                        evidence=["captions (hair/uniform colours)", "identity", "frames"]),
    "wrong_audio": dict(desc="audio replaced by the audio of another clip", needs=["audio", "other"], hard_fail=False,
                        evidence=["audio transcript / WER vs script", "speaker similarity"]),
    "silent_audio": dict(desc="audio track is digital silence (or missing with drop=True)", needs=["audio"],
                         hard_fail=True, evidence=["technical.silent_audio"]),
    "truncated": dict(desc="clip cut short: wrong duration", needs=[], hard_fail=True,
                      evidence=["technical.wrong_duration"]),
    "corrupt_file": dict(desc="damaged container / bitstream", needs=[], hard_fail=True,
                         evidence=["technical.unreadable|decode_errors|truncated_stream"]),
    "garbled_text_overlay": dict(desc="unreadable pseudo-text (morphing glyph salad) on a sign and as subtitles",
                                 needs=[], hard_fail=False, evidence=["captions/artifact checklist", "OCR", "frames"]),
    "jitter": dict(desc="frame-to-frame shake (random translation / rotation)", needs=[], hard_fail=False,
                   evidence=["motion.global_motion_mag", "flicker.repeat_ratio", "warp.warp_error"]),
    "hard_cut": dict(desc="second part of the shot replaced by another shot (cut inside a single shot)",
                     needs=["other"], hard_fail=False, evidence=["motion.n_hard_cuts", "captions"]),
    "black": dict(desc="black video with the original audio", needs=[], hard_fail=True,
                  evidence=["technical.black_video"]),
}
KINDS: tuple[str, ...] = tuple(KIND_INFO)

# minimal measured effect for a broken clip to count as broken (0..255 grey levels; see _effect)
_MIN_EFFECT = {
    "flicker": ("mean", 3.0), "frame_shuffle": ("mean", 1.5), "freeze": ("mean", 1.5), "ghost": ("mean", 2.0),
    "noise_blur": ("mean", 3.0), "color_shift": ("mean", 4.0), "jitter": ("mean", 2.0), "hard_cut": ("mean", 5.0),
    "black": ("mean", 5.0), "melt": ("local", 20.0), "garbled_text_overlay": ("local", 20.0),
}


# ----------------------------------------------------------------------------------------------------------------
# encoding
# ----------------------------------------------------------------------------------------------------------------
def write_video(frames: np.ndarray, out_path: str | Path, fps: float, audio: str | Path | None = None,
                silent: bool = False, audio_loop: bool = False, crf: int = 18, faststart: bool = True) -> Path:
    """Encode uint8 RGB frames [T, H, W, 3] to H.264 yuv420p (+ AAC audio).

    audio      file whose first audio stream is muxed in (trimmed / padded with silence to the video length)
    silent     mux digital silence instead
    audio_loop loop the audio input if it is shorter than the video
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    T, H, W = frames.shape[:3]
    H2, W2 = H - H % 2, W - W % 2
    if (H2, W2) != (H, W):
        frames = frames[:, :H2, :W2]
    rate = Fraction(fps).limit_denominator(1001)
    dur = T / fps
    cmd = [vm.ffmpeg_bin(), "-v", "error", "-nostdin", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W2}x{H2}", "-r", f"{rate.numerator}/{rate.denominator}", "-i", "pipe:0"]
    if silent:
        cmd += ["-f", "lavfi", "-t", f"{dur:.4f}", "-i", "anullsrc=r=44100:cl=stereo"]
    elif audio is not None:
        if audio_loop:
            cmd += ["-stream_loop", "-1"]
        cmd += ["-i", str(audio)]
    cmd += ["-map", "0:v:0"]
    if silent or audio is not None:
        cmd += ["-map", "1:a:0", "-af", f"apad=whole_dur={dur:.4f}", "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2"]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf), "-pix_fmt", "yuv420p",
            "-threads", str(min(4, vm.n_threads())), "-t", f"{dur:.4f}"]
    if faststart:
        cmd += ["-movflags", "+faststart"]
    cmd += [str(out_path)]
    with tempfile.TemporaryFile() as err:
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=err)
        try:
            for f in frames:
                p.stdin.write(np.ascontiguousarray(f).tobytes())
            p.stdin.close()
        except BrokenPipeError:
            pass
        rc = p.wait()
        if rc != 0:
            err.seek(0)
            raise RuntimeError(f"ffmpeg failed ({rc}): {err.read().decode('utf-8', 'replace')[-600:]}")
    return out_path


def _effect(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """(mean |a-b|, mean over frames of the most-changed 8x8 block) on 320x176 RGB; frame counts may differ."""
    import cv2

    n = min(len(a), len(b))
    if n == 0:
        return 0.0, 0.0
    if a.shape[1:] != b.shape[1:]:
        return 255.0, 255.0
    idx = np.unique(np.linspace(0, n - 1, min(n, 24)).round().astype(int))

    def small(x):
        return np.stack([cv2.resize(f, (320, 176), interpolation=cv2.INTER_AREA) for f in x[idx]]).astype(np.float32)

    d = np.abs(small(a) - small(b)).mean(axis=-1)
    t, h, w = d.shape
    blk = d.reshape(t, h // 8, 8, w // 8, 8).mean(axis=(2, 4)).reshape(t, -1).max(axis=1)
    return float(d.mean()), float(blk.mean())


# ----------------------------------------------------------------------------------------------------------------
# pixel-space corruptions (frames in, frames out)
# ----------------------------------------------------------------------------------------------------------------
def _flicker(frames, rng, strength: float = 0.35, rate: float = 0.7, color: bool = True):
    out = np.empty_like(frames)
    for i, f in enumerate(frames):
        if rng.random() < rate:
            gain = np.full(3, 1.0 + rng.uniform(-strength, strength), np.float32)
            if color:
                gain = gain * (1.0 + rng.uniform(-strength, strength, 3).astype(np.float32) * 0.6)
            off = float(rng.uniform(-strength, strength) * 60.0)
            out[i] = np.clip(f.astype(np.float32) * gain + off, 0, 255).astype(np.uint8)
        else:
            out[i] = f
    return out, dict(strength=strength, rate=rate, color=color)


def _frame_shuffle(frames, rng, window: int = 8):
    T = len(frames)
    idx = np.arange(T)
    if window <= 0 or window >= T:
        while T > 1 and np.array_equal(idx, np.arange(T)):
            rng.shuffle(idx)
    else:
        for s in range(0, T, window):
            seg = idx[s:s + window].copy()
            if len(seg) > 1:
                for _ in range(8):
                    perm = rng.permutation(seg)
                    if not np.array_equal(perm, seg):
                        break
                idx[s:s + window] = perm
    return frames[idx], dict(window=window)


def _freeze(frames, rng, start_frac: float = 0.0):
    T = len(frames)
    k = int(np.clip(round(start_frac * T), 0, T - 1))
    out = frames.copy()
    out[k:] = frames[k]
    return out, dict(start_frac=start_frac, frozen_from_frame=k)


def _focus_point(frames) -> tuple[float, float]:
    """(cx, cy) in 0..1 of the most 'active' region: temporal variance first, spatial detail as fallback."""
    import cv2

    luma = vm._small_luma(frames[:: max(1, len(frames) // 16)])
    act = luma.std(axis=0)
    if act.max() < 2.0:
        g = luma[len(luma) // 2]
        act = np.abs(cv2.Laplacian(g, cv2.CV_32F))
    act = cv2.GaussianBlur(act, (0, 0), act.shape[0] * 0.12)
    y, x = np.unravel_index(int(act.argmax()), act.shape)
    return float(x / act.shape[1]), float(y / act.shape[0])


def _face_point(frames) -> tuple[float, float] | None:
    """Centre of the largest anime face in the middle frame via dghs-imgutils, if that package is available."""
    try:
        from imgutils.detect import detect_faces
        from PIL import Image

        f = frames[len(frames) // 2]
        dets = detect_faces(Image.fromarray(f))
        if not dets:
            return None
        (x0, y0, x1, y1), _, _ = max(dets, key=lambda d: (d[0][2] - d[0][0]) * (d[0][3] - d[0][1]))
        return float((x0 + x1) / 2 / f.shape[1]), float((y0 + y1) / 2 / f.shape[0])
    except Exception:
        return None


def _melt(frames, rng, strength: float = 1.0, region: str | Sequence[float] = "auto", radius: float = 0.30):
    """region: "auto" (most active area), "face" (anime face detector, falls back to auto), "full", or (cx, cy) in 0..1."""
    import cv2

    T, H, W = frames.shape[:3]
    if region == "full":
        mask = np.ones((H, W), np.float32)
        centre = None
    else:
        if isinstance(region, str):
            centre = (_face_point(frames) if region == "face" else None) or _focus_point(frames)
        else:
            centre = (float(region[0]), float(region[1]))
        ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
        sig = radius * H
        mask = np.exp(-(((xs - centre[0] * W) ** 2 + (ys - centre[1] * H) ** 2) / (2 * sig * sig))).astype(np.float32)

    def field():
        n = rng.standard_normal((6, 10)).astype(np.float32)
        return cv2.resize(n, (W, H), interpolation=cv2.INTER_CUBIC)

    ax, ay, bx, by, drip = field(), field(), field(), field(), np.abs(field())
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    amp_max = strength * 0.07 * H
    out = np.empty_like(frames)
    for t in range(T):
        prog = t / max(T - 1, 1)
        a = (0.25 + 0.75 * prog) * amp_max
        ph = 2 * math.pi * 1.5 * prog
        dx = a * mask * (ax * math.cos(ph) + bx * math.sin(ph))
        dy = a * mask * (ay * math.cos(ph) + by * math.sin(ph)) - a * 0.8 * prog * mask * drip   # sag downwards
        out[t] = cv2.remap(frames[t], xs + dx, ys + dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return out, dict(strength=strength, region=region if isinstance(region, str) else list(region),
                     centre=[round(c, 3) for c in centre] if centre else None, radius=radius)


def _ghost(frames, rng, shift: int = 5, alpha: float = 0.45, offset: Sequence[int] = (10, 4)):
    T = len(frames)
    out = np.empty_like(frames)
    for t in range(T):
        other = np.roll(frames[max(t - shift, 0)], (int(offset[1]), int(offset[0])), axis=(0, 1))
        out[t] = (frames[t].astype(np.float32) * (1 - alpha) + other.astype(np.float32) * alpha).astype(np.uint8)
    return out, dict(shift=shift, alpha=alpha, offset=list(offset))


def _noise_blur(frames, rng, blur: float = 2.2, noise: float = 14.0):
    import cv2

    out = np.empty_like(frames)
    for i, f in enumerate(frames):
        g = cv2.GaussianBlur(f, (0, 0), blur).astype(np.float32) if blur > 0 else f.astype(np.float32)
        if noise > 0:
            g = g + rng.normal(0.0, noise, f.shape).astype(np.float32)
        out[i] = np.clip(g, 0, 255).astype(np.uint8)
    return out, dict(blur=blur, noise=noise)


def _color_shift(frames, rng, hue: float | None = None, sat: float = 1.3, mode: str = "hue"):
    """mode "hue": rotate the hue by ``hue`` degrees (random 60..180 if None) and scale saturation;
    "swap": permute the RGB channels."""
    import cv2

    if mode == "swap":
        perm = [2, 0, 1]
        return frames[..., perm].copy(), dict(mode=mode, perm=perm)
    if hue is None:
        hue = float(rng.uniform(60, 180) * rng.choice([-1, 1]))
    out = np.empty_like(frames)
    for i, f in enumerate(frames):
        hsv = cv2.cvtColor(f, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[..., 0] = (hsv[..., 0] + hue / 2.0) % 180.0
        hsv[..., 1] = np.clip(hsv[..., 1] * sat, 0, 255)
        out[i] = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return out, dict(mode=mode, hue=round(float(hue), 1), sat=sat)


def _jitter(frames, rng, amp: float = 0.012, rot: float = 0.4):
    import cv2

    T, H, W = frames.shape[:3]
    out = np.empty_like(frames)
    for t in range(T):
        dx, dy = rng.normal(0, amp * H, 2)
        m = cv2.getRotationMatrix2D((W / 2, H / 2), float(rng.normal(0, rot)), 1.02)
        m[:, 2] += (dx, dy)
        out[t] = cv2.warpAffine(frames[t], m, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return out, dict(amp=amp, rot=rot)


def _hard_cut(frames, rng, other: str | Path | None = None, at: float = 0.5):
    if other is None:
        raise ValueError("hard_cut needs other=<path of another clip>")
    T, H, W = frames.shape[:3]
    k = int(np.clip(round(at * T), 1, T - 1))
    oth = vm.load_frames(other, size=(W, H), max_frames=T - k)
    reps = int(math.ceil((T - k) / len(oth)))
    oth = np.concatenate([oth] * reps, 0)[: T - k]
    out = frames.copy()
    out[k:] = oth
    return out, dict(other=str(other), at=at, cut_frame=k)


def _black(frames, rng):
    return np.zeros_like(frames), {}


class _Glyphs:
    """Pseudo-characters made of random strokes; every glyph keeps morphing, like text in diffusion video."""

    def __init__(self, rng, n: int):
        self.rng = rng
        self.strokes = [self._new() for _ in range(n)]

    def _new(self):
        r = self.rng
        strokes = []
        for _ in range(int(r.integers(3, 7))):
            kind = r.choice(["h", "v", "d", "arc", "dot"], p=[0.28, 0.28, 0.22, 0.17, 0.05])
            a = r.uniform(0.08, 0.92, 4)
            if kind == "h":
                pts = [(a[0], a[1]), (a[2], a[1] + r.uniform(-0.08, 0.08))]
            elif kind == "v":
                pts = [(a[0], a[1]), (a[0] + r.uniform(-0.08, 0.08), a[3])]
            elif kind == "d":
                pts = [(a[0], a[1]), ((a[0] + a[2]) / 2 + r.uniform(-0.15, 0.15), (a[1] + a[3]) / 2), (a[2], a[3])]
            elif kind == "arc":
                c, rad = (a[0], a[1]), r.uniform(0.15, 0.4)
                t0, t1 = r.uniform(0, 6.28), r.uniform(1.5, 4.5)
                pts = [(c[0] + rad * math.cos(t0 + t1 * k / 5), c[1] + rad * math.sin(t0 + t1 * k / 5)) for k in range(6)]
            else:
                pts = [(a[0], a[1]), (a[0] + 0.04, a[1] + 0.04)]
            strokes.append(np.array(pts, np.float32))
        return strokes

    def step(self, p_new: float = 0.25, wobble: float = 0.035):
        r = self.rng
        for i, g in enumerate(self.strokes):
            if r.random() < p_new:
                self.strokes[i] = self._new()
            else:
                self.strokes[i] = [s + r.normal(0, wobble, s.shape).astype(np.float32) for s in g]

    def draw(self, img, x0: float, y0: float, cell: float, color, thick: int, outline=None):
        import cv2

        x = x0
        for g in self.strokes:
            for s in g:
                pts = np.round(s * cell * np.array([0.82, 1.0], np.float32) + np.array([x, y0], np.float32)).astype(np.int32)
                if outline is not None:
                    cv2.polylines(img, [pts], False, outline, thick + 3, cv2.LINE_AA)
            for s in g:
                pts = np.round(s * cell * np.array([0.82, 1.0], np.float32) + np.array([x, y0], np.float32)).astype(np.int32)
                cv2.polylines(img, [pts], False, color, thick, cv2.LINE_AA)
            x += cell * 0.9


def _garbled_text_overlay(frames, rng, style: str = "both", morph_every: int = 3, scale: float = 1.0):
    """style: "subtitle" (white outlined gibberish at the bottom), "sign" (paper card with dark gibberish), "both"."""
    import cv2

    T, H, W = frames.shape[:3]
    out = frames.copy()
    cell_sub = 0.075 * H * scale
    n_sub = int(rng.integers(8, 14))
    sub = _Glyphs(rng, n_sub)
    cell_sign = 0.085 * H * scale
    rows = [_Glyphs(rng, int(rng.integers(4, 7))) for _ in range(2)]
    sign_w = cell_sign * 0.9 * max(len(r.strokes) for r in rows) + cell_sign * 0.6
    sign_h = cell_sign * 2.5
    sx = float(rng.uniform(0.05 * W, max(0.06 * W, W - sign_w - 0.05 * W)))
    sy = float(rng.uniform(0.06 * H, 0.35 * H))
    thick = max(2, int(round(H / 190 * scale)))
    for t in range(T):
        if t and t % morph_every == 0:
            sub.step()
            for r in rows:
                r.step()
        img = out[t]
        if style in ("sign", "both"):
            ov = img.copy()
            cv2.rectangle(ov, (int(sx), int(sy)), (int(sx + sign_w), int(sy + sign_h)), (246, 242, 228), -1)
            cv2.rectangle(ov, (int(sx), int(sy)), (int(sx + sign_w), int(sy + sign_h)), (60, 60, 60), max(1, thick - 1))
            for k, r in enumerate(rows):
                r.draw(ov, sx + cell_sign * 0.3, sy + cell_sign * (0.15 + 1.15 * k), cell_sign, (25, 25, 30), thick)
            img[:] = cv2.addWeighted(ov, 0.93, img, 0.07, 0)
        if style in ("subtitle", "both"):
            x0 = (W - n_sub * cell_sub * 0.9) / 2
            sub.draw(img, x0, H * 0.985 - cell_sub * 1.25, cell_sub, (250, 250, 250), thick, outline=(10, 10, 10))
    return out, dict(style=style, morph_every=morph_every, scale=scale)


_PIXEL_KINDS = {
    "flicker": _flicker, "frame_shuffle": _frame_shuffle, "freeze": _freeze, "melt": _melt, "ghost": _ghost,
    "noise_blur": _noise_blur, "color_shift": _color_shift, "jitter": _jitter, "hard_cut": _hard_cut,
    "black": _black, "garbled_text_overlay": _garbled_text_overlay,
}


# ----------------------------------------------------------------------------------------------------------------
# file-level corruption
# ----------------------------------------------------------------------------------------------------------------
def _corrupt_bytes(clean: Path, out_path: Path, rng, mode: str = "garble", n_spans: int = 24, span: int = 1500,
                   frac: float = 0.5) -> dict:
    """mode: "garble" (random bytes over spans of the media data: decode errors), "cut" (file truncated at ``frac``;
    the index (moov) is lost: unreadable), "header" (first 4 KiB destroyed: unreadable), "empty" (0 bytes)."""
    data = bytearray(clean.read_bytes())
    n = len(data)
    if mode == "empty":
        data = bytearray()
    elif mode == "cut":
        data = data[: max(64, int(n * frac))]
    elif mode == "header":
        data[: min(4096, n)] = rng.integers(0, 256, min(4096, n), dtype=np.uint8).tobytes()
    elif mode == "garble":
        i = data.find(b"mdat")
        j = data.find(b"moov")
        lo = i + 8 if i >= 0 else int(n * 0.1)
        hi = j - 8 if j > lo else n
        lo, hi = lo + (hi - lo) // 20, hi - (hi - lo) // 20
        for _ in range(n_spans):
            s = int(rng.integers(lo, max(lo + 1, hi - span)))
            data[s:s + span] = rng.integers(0, 256, span, dtype=np.uint8).tobytes()
    else:
        raise ValueError(f"unknown corrupt_file mode {mode!r}")
    out_path.write_bytes(bytes(data))
    return dict(mode=mode, n_spans=n_spans, span=span, frac=frac) if mode in ("garble", "cut") else dict(mode=mode)


# ----------------------------------------------------------------------------------------------------------------
# public API
# ----------------------------------------------------------------------------------------------------------------
def _has_sound(path: str | Path, info: dict | None = None) -> bool:
    info = info or vm.video_info(path)
    if not info.get("has_audio"):
        return False
    return vm._audio_levels(path)["audio_rms_db"] > -50.0


def _spec_fields(src_clip: str | Path, frames: np.ndarray, info: dict) -> dict:
    """What a shot spec would say about the *source* shot: does it have sound, is it meant to be a still.
    Pass them to technical_check(need_audio=..., allow_static=...) for real and broken clips alike."""
    static = vm.technical_check(src_clip)["info"].get("static_pair_frac", 0.0)
    # 0.90, not the 0.98 of the frozen test: re-encoding an almost-still shot can remove its last small changes
    return {"need_audio": _has_sound(src_clip, info), "allow_static": bool(static >= 0.90)}


def make_real(src_clip: str | Path, out_path: str | Path) -> dict:
    """Re-encode a real clip through the same path as the broken ones (content untouched). Label: real."""
    info = vm.video_info(src_clip)
    frames, fps = vm.load_frames(src_clip, return_fps=True)
    write_video(frames, out_path, fps, audio=src_clip if info.get("has_audio") else None)
    return {"path": str(out_path), "label": "real", "kind": "none", "src": str(src_clip), "params": {},
            "expected_dur": round(len(frames) / fps, 4), **_spec_fields(src_clip, frames, info),
            "hard_fail": False, "effect_mean": 0.0, "effect_local": 0.0}


def make_broken(src_clip: str | Path, kind: str, out_path: str | Path, seed: int = 0, **kw) -> dict:
    """Write a deliberately broken variant of ``src_clip`` to ``out_path``.

    kind (see KIND_INFO) and its keyword arguments:
      flicker(strength=0.35, rate=0.7, color=True)      frame_shuffle(window=8; 0 = shuffle the whole clip)
      freeze(start_frac=0.0)                            melt(strength=1.0, region="auto"|"face"|"full"|(cx, cy), radius=0.3)
      ghost(shift=5, alpha=0.45, offset=(10, 4))        noise_blur(blur=2.2, noise=14)
      color_shift(hue=None, sat=1.3, mode="hue"|"swap") wrong_audio(other=<clip>)
      silent_audio(drop=False)                          truncated(keep=0.5)
      corrupt_file(mode="garble"|"cut"|"header"|"empty") garbled_text_overlay(style="both"|"subtitle"|"sign", morph_every=3)
      jitter(amp=0.012, rot=0.4)                        hard_cut(other=<clip>, at=0.5)
      black()

    Returns the label row: {"path", "label": "broken", "kind", "src", "params", "expected_dur" (duration of the
    source = what the shot should last), "need_audio" (the source has sound), "allow_static" (the source is an
    (almost) still shot), "hard_fail", "effect_mean", "effect_local"}. expected_dur / need_audio / allow_static play the role
    of the shot spec: pass them to ``technical_check`` for real and broken clips alike.
    ``effect_*`` measure how much the picture really changed (0..255); a freeze of an already static shot has
    effect 0 and is not a valid "broken" example - :func:`build_set` filters on this.
    """
    if kind not in KIND_INFO:
        raise ValueError(f"unknown kind {kind!r}; choose from {KINDS}")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    info = vm.video_info(src_clip)
    has_audio = bool(info.get("has_audio"))
    frames, fps = vm.load_frames(src_clip, return_fps=True)
    src_dur = len(frames) / fps
    audio = src_clip if has_audio else None
    spec = _spec_fields(src_clip, frames, info)
    eff = (None, None)
    hard_fail = bool(KIND_INFO[kind]["hard_fail"])

    if kind in _PIXEL_KINDS:
        out, params = _PIXEL_KINDS[kind](frames, rng, **kw)
        eff = _effect(frames, out)
        write_video(out, out_path, fps, audio=audio)
        if kind == "freeze":
            hard_fail = params["start_frac"] <= 0.02          # a partial freeze is only a warning in technical_check
    elif kind == "wrong_audio":
        other = kw.pop("other", None)
        if other is None:
            raise ValueError("wrong_audio needs other=<path of another clip with audio>")
        write_video(frames, out_path, fps, audio=other, audio_loop=True)
        params = dict(other=str(other))
    elif kind == "silent_audio":
        drop = bool(kw.pop("drop", False))
        write_video(frames, out_path, fps, audio=None, silent=not drop)
        params = dict(drop=drop)
    elif kind == "truncated":
        keep = float(kw.pop("keep", 0.5))
        n = int(np.clip(round(len(frames) * keep), 2, len(frames) - 1))
        write_video(frames[:n], out_path, fps, audio=audio)
        params = dict(keep=keep, frames_kept=n)
    elif kind == "corrupt_file":
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR") or None) as td:
            clean = Path(td) / "clean.mp4"
            write_video(frames, clean, fps, audio=audio, faststart=False)      # index at the end: "cut" loses it
            params = _corrupt_bytes(clean, out_path, rng, **kw)
        kw = {}
    else:  # pragma: no cover
        raise AssertionError(kind)
    if kind not in _PIXEL_KINDS and kw:
        raise TypeError(f"unexpected arguments for {kind}: {sorted(kw)}")
    return {"path": str(out_path), "label": "broken", "kind": kind, "src": str(src_clip), "params": params,
            "expected_dur": round(src_dur, 4), **spec, "hard_fail": hard_fail,
            "effect_mean": None if eff[0] is None else round(eff[0], 3),
            "effect_local": None if eff[1] is None else round(eff[1], 3)}


def effect_ok(row: dict) -> bool:
    """True if a broken clip is visibly / measurably broken (see _MIN_EFFECT); non-pixel kinds are always ok."""
    rule = _MIN_EFFECT.get(row["kind"])
    if rule is None or row.get("effect_mean") is None:
        return True
    which, thr = rule
    return (row["effect_mean"] if which == "mean" else row["effect_local"]) >= thr


def _random_kwargs(kind: str, rng) -> dict:
    """Parameter variety for build_set (still clearly broken)."""
    if kind == "flicker":
        return dict(strength=float(rng.uniform(0.3, 0.5)), rate=float(rng.uniform(0.5, 0.9)))
    if kind == "frame_shuffle":
        return dict(window=int(rng.choice([6, 8, 12, 0])))
    if kind == "freeze":
        return dict(start_frac=0.0 if rng.random() < 0.6 else float(rng.uniform(0.2, 0.45)))
    if kind == "melt":
        return dict(strength=float(rng.uniform(0.9, 1.5)), region="auto" if rng.random() < 0.75 else "full")
    if kind == "ghost":
        return dict(shift=int(rng.integers(4, 9)), alpha=float(rng.uniform(0.4, 0.5)))
    if kind == "noise_blur":
        return dict(blur=float(rng.uniform(1.8, 3.0)), noise=float(rng.uniform(10, 22)))
    if kind == "color_shift":
        return dict(mode="hue" if rng.random() < 0.8 else "swap")
    if kind == "truncated":
        return dict(keep=float(rng.uniform(0.3, 0.65)))
    if kind == "corrupt_file":
        return dict(mode=str(rng.choice(["garble", "garble", "cut", "header", "empty"])))
    if kind == "garbled_text_overlay":
        return dict(style=str(rng.choice(["both", "subtitle", "sign"])), morph_every=int(rng.integers(2, 5)))
    if kind == "jitter":
        return dict(amp=float(rng.uniform(0.01, 0.02)))
    if kind == "hard_cut":
        return dict(at=float(rng.uniform(0.35, 0.65)))
    if kind == "silent_audio":
        return dict(drop=bool(rng.random() < 0.25))
    return {}


def _tech(row: dict) -> dict:
    """Run technical_check with the row's spec fields; stored in the label row for the calibration report."""
    tc = vm.technical_check(row["path"], expected_dur=row["expected_dur"], need_audio=row["need_audio"],
                            allow_static=row["allow_static"])
    return {"technical_ok": tc["ok"], "technical_reasons": [r.split(":")[0] for r in tc["reasons"]]}


def build_set(src: str | Path | Sequence[str | Path], out_dir: str | Path, n_broken: int = 50,
              n_real: int | None = None, kinds: Sequence[str] = KINDS, seed: int = 0, verbose: bool = True) -> list[dict]:
    """Write ``n_broken`` broken clips (kinds balanced round-robin) and ``n_real`` re-encoded real clips with opaque
    file names into ``out_dir/clips`` plus ``out_dir/labels.jsonl``. Sources that cannot show a given failure (e.g.
    freezing a static shot, silencing a silent clip) are skipped for that kind; returns the label rows."""
    out_dir = Path(out_dir)
    clips_dir = out_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    srcs = vm.list_clips([src] if isinstance(src, (str, Path)) else list(src))
    if not srcs:
        raise ValueError(f"no clips found in {src}")
    bad = [k for k in kinds if k not in KIND_INFO]
    if bad:
        raise ValueError(f"unknown kinds {bad}")
    rng = np.random.default_rng(seed)
    n_real = n_broken if n_real is None else n_real
    sound: dict[str, bool] = {}

    def has_sound(p):
        if str(p) not in sound:
            sound[str(p)] = _has_sound(p)
        return sound[str(p)]

    jobs: list[tuple[str, str]] = [("broken", kinds[i % len(kinds)]) for i in range(n_broken)] + [("real", "none")] * n_real
    names = [f"c_{i:04d}.mp4" for i in rng.permutation(len(jobs))]
    rows: list[dict] = []
    real_order = [srcs[i] for i in rng.permutation(len(srcs))]
    r_i = 0
    for (label, kind), name in zip(jobs, names):
        out_path = clips_dir / name
        if label == "real":
            row = make_real(real_order[r_i % len(real_order)], out_path)
            row.update(_tech(row))
            r_i += 1
        else:
            row = None
            needs = KIND_INFO[kind]["needs"]
            for attempt, si in enumerate(rng.permutation(len(srcs))[:8]):
                s = srcs[si]
                if "audio" in needs and not has_sound(s):
                    continue
                kw = _random_kwargs(kind, rng)
                if "other" in needs:
                    others = [o for o in srcs if o != s and (kind != "wrong_audio" or has_sound(o))]
                    if not others:
                        continue
                    kw["other"] = others[int(rng.integers(len(others)))]
                cand = make_broken(s, kind, out_path, seed=int(rng.integers(1 << 31)), **kw)
                if not effect_ok(cand):
                    continue
                cand.update(_tech(cand))
                if cand["hard_fail"] and cand["technical_ok"]:
                    continue                      # e.g. garbled bytes that happen to decode cleanly: try again
                row = cand
                break
            if row is None:
                out_path.unlink(missing_ok=True)
                if verbose:
                    print(f"[broken] skipped one '{kind}': no source clip can show this failure")
                continue
        rows.append(row)
        if verbose:
            print(f"[broken] {len(rows)}/{len(jobs)} {name} {row['label']}:{row['kind']}", flush=True)
    rows.sort(key=lambda r: r["path"])
    with open(out_dir / "labels.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Make deliberately broken clips (+ re-encoded real ones) with labels.jsonl")
    ap.add_argument("--src", nargs="+", help="directory / list of real single-shot clips")
    ap.add_argument("--out")
    ap.add_argument("--n", type=int, default=50, help="number of broken clips")
    ap.add_argument("--n-real", type=int, default=None, help="number of real (re-encoded) clips; default = --n")
    ap.add_argument("--kinds", default=",".join(KINDS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--list-kinds", action="store_true")
    a = ap.parse_args(argv)
    if a.list_kinds:
        for k, v in KIND_INFO.items():
            print(f"{k:22s} hard_fail={str(v['hard_fail']):5s} needs={','.join(v['needs']) or '-':12s} {v['desc']}")
        return 0
    if not a.src or not a.out:
        ap.error("--src and --out are required")
    rows = build_set(a.src if len(a.src) > 1 else a.src[0], a.out, n_broken=a.n, n_real=a.n_real,
                     kinds=[k for k in a.kinds.split(",") if k], seed=a.seed)
    nb = sum(r["label"] == "broken" for r in rows)
    print(f"wrote {nb} broken + {len(rows) - nb} real clips -> {Path(a.out) / 'labels.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
