"""Synthetic clips for the eval tests (no real assets needed). Not a test module itself."""
from __future__ import annotations

import math
import struct
import wave
from pathlib import Path

import numpy as np

W, H, FPS = 320, 176, 24.0


def have_ffmpeg() -> bool:
    try:
        from src.eval import video_metrics as vm

        vm.ffmpeg_bin()
        vm.ffprobe_bin()
        return True
    except Exception:
        return False


def background(seed: int = 0) -> np.ndarray:
    """Smooth colourful static background, uint8 [H, W, 3]."""
    import cv2

    rng = np.random.default_rng(seed)
    small = rng.integers(40, 215, (6, 10, 3)).astype(np.uint8)
    return cv2.resize(small, (W, H), interpolation=cv2.INTER_CUBIC)


def moving_frames(n: int = 48, hold: int = 1, seed: int = 0, speed: float = 4.0) -> np.ndarray:
    """A bright textured square moving over a static background; ``hold`` = frames per drawing (2 = "on twos")."""
    bg = background(seed)
    rng = np.random.default_rng(seed + 1)
    tex = rng.integers(150, 255, (40, 40, 3)).astype(np.uint8)
    out = np.empty((n, H, W, 3), np.uint8)
    for t in range(n):
        k = (t // hold) * hold
        x = int(20 + speed * k) % (W - 60)
        y = int(60 + 30 * math.sin(k / 6.0))
        f = bg.copy()
        f[y:y + 40, x:x + 40] = tex
        out[t] = f
    return out


def tone_wav(path: Path, dur: float = 2.0, freq: float = 440.0, amp: int = 8000, sr: int = 44100) -> Path:
    n = int(dur * sr)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(b"".join(struct.pack("<h", int(amp * math.sin(2 * math.pi * freq * i / sr))) for i in range(n)))
    return path


def write_clip(path: Path, frames: np.ndarray, audio: bool = True, freq: float = 440.0, fps: float = FPS) -> Path:
    from src.eval import broken

    wav = None
    if audio:
        wav = tone_wav(Path(path).with_suffix(".wav"), dur=len(frames) / fps, freq=freq)
    broken.write_video(frames, path, fps, audio=wav)
    return Path(path)
