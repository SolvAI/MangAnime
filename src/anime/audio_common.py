"""Shared helpers for the anime audio stack (separation, ASR, speaker embeddings, BGM cues).

Only numpy / soundfile / stdlib here so every audio module (and the main venv) can import it.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from src.common.config import CFG, P
from src.common.io import read_json



def audio_cfg():
    """The `audio` config block, whichever way src/common/config.py assembles the yaml files
    (namespaced under the file stem -> CFG.audio, or a top-level `audio:` key, or a flat merge)."""
    c = CFG.get("audio")
    if c is not None and "separation" in c:
        return c
    if c is not None and "audio" in c and "separation" in c.audio:
        return c.audio
    if "separation" in CFG and "spk" in CFG:
        return CFG
    raise RuntimeError("configs/audio.yaml is not loaded (CFG.audio.separation missing)")


ACFG = audio_cfg()
TRAIN_EPS = [int(e) for e in CFG.split.train_episodes]
HELDOUT_EPS = [int(e) for e in CFG.split.heldout_episodes]
ALL_EPS = sorted(TRAIN_EPS + HELDOUT_EPS)


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def parse_episodes(spec: str | None) -> list[int]:
    """'all' | 'train' | 'heldout' | '1,2,5' | '1-21' | '1,2,22-24,3-21' -> episode numbers IN THE GIVEN ORDER (deduplicated)."""
    if spec is None or spec == "all":
        return list(ALL_EPS)
    out: list[int] = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if part == "all":
            cand = list(ALL_EPS)
        elif part == "train":
            cand = list(TRAIN_EPS)
        elif part == "heldout":
            cand = list(HELDOUT_EPS)
        elif "-" in part:
            a, b = part.split("-", 1)
            cand = list(range(int(a), int(b) + 1))
        else:
            cand = [int(part)]
        for e in cand:
            if e not in out:
                out.append(e)
    return out


def ep_name(ep: int) -> str:
    return f"ep{int(ep):02d}"


def ep_dir(ep: int) -> Path:
    return P("interim") / "anime" / ep_name(ep)


def subs_path(ep: int) -> Path:
    return P("processed") / "anime" / "subs" / f"{ep_name(ep)}.json"


def load_subs(ep: int, generate: bool = True) -> list[dict]:
    """Subtitle events of an episode (list of {start,end,text,kind,style}); generated through ingest if missing."""
    p = subs_path(ep)
    if not p.exists() and generate:
        # the ingest module only needs pysubs2 + omegaconf (both in the audio venv); audio extraction is skipped if present
        log(f"{ep_name(ep)}: subtitle JSON missing -> python -m src.anime.ingest --episodes {ep} --no-proxy")
        subprocess.run([sys.executable, "-m", "src.anime.ingest", "--episodes", str(ep), "--no-proxy"],
                       cwd=str(CFG.paths.repo), check=False)
    if not p.exists():
        raise FileNotFoundError(f"subtitle events missing: {p}")
    return read_json(p)


def dialogue_events(subs: list[dict]) -> list[tuple[int, dict]]:
    return [(i, e) for i, e in enumerate(subs) if e.get("kind") == "dialogue"]


def merge_spans(spans: list[tuple[float, float]], gap: float = 0.0) -> list[tuple[float, float]]:
    out: list[list[float]] = []
    for s, e in sorted(spans):
        if e <= s:
            continue
        if out and s - out[-1][1] <= gap:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(a, b) for a, b in out]


def spans_mask(spans: list[tuple[float, float]], n: int, rate: float, pad: float = 0.0) -> np.ndarray:
    """Boolean mask of length n (frames at `rate` Hz) that is True inside any span (optionally padded)."""
    m = np.zeros(n, dtype=bool)
    for s, e in spans:
        a = max(0, int(np.floor((s - pad) * rate)))
        b = min(n, int(np.ceil((e + pad) * rate)))
        if b > a:
            m[a:b] = True
    return m


def song_spans(subs: list[dict], pad: float = 4.0, gap: float = 20.0) -> list[tuple[float, float]]:
    """OP/ED/insert-song spans from `kind == "song"` subtitle events (lyric lines merged across short gaps)."""
    spans = [(e["start"], e["end"]) for e in subs if e.get("kind") == "song"]
    return [(max(0.0, s - pad), e + pad) for s, e in merge_spans(spans, gap=gap)]


def oped_spans(ep: int, pad: float = 0.0) -> list[tuple[float, float]]:
    """OP/ED/preview spans provided by the lead in data/processed/anime/oped.json.

    Accepted shapes per episode: [[start, end], ...], [[start, end, "op"|"ed"|"preview"], ...] or
    [{"start": .., "end": ..}, ...]. Malformed entries are ignored.
    """
    try:
        d = read_json(P("processed") / "anime" / "oped.json", default={}) or {}
    except Exception:
        return []
    out = []
    for it in d.get(ep_name(ep), []) or []:
        try:
            a, b = (it["start"], it["end"]) if isinstance(it, dict) else (it[0], it[1])
            a, b = float(a), float(b)
        except Exception:
            continue
        if b > a:
            out.append((max(0.0, a - pad), b + pad))
    return out


def read_wav(path: str | Path, start: float = 0.0, dur: float | None = None, mono: bool = False) -> tuple[np.ndarray, int]:
    """Read (part of) a wav as float32 [T, C] (or [T] when mono=True)."""
    import soundfile as sf

    with sf.SoundFile(str(path)) as f:
        sr = f.samplerate
        a = int(round(start * sr))
        f.seek(min(a, len(f)))
        n = -1 if dur is None else int(round(dur * sr))
        x = f.read(n, dtype="float32", always_2d=True)
    if mono:
        x = x.mean(axis=1)
    return x, sr


def write_wav_atomic(path: str | Path, x: np.ndarray, sr: int, subtype: str = "PCM_16") -> None:
    import soundfile as sf

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp" + path.suffix)
    sf.write(str(tmp), x, sr, subtype=subtype)
    os.replace(tmp, path)


def wav_info(path: str | Path) -> tuple[float, int, int]:
    """(duration_s, samplerate, channels) without decoding."""
    import soundfile as sf

    i = sf.info(str(path))
    return i.frames / i.samplerate, i.samplerate, i.channels


def frame_rms_db(x: np.ndarray, sr: int, hop_s: float = 0.02, win_s: float = 0.04) -> tuple[np.ndarray, float]:
    """Short-time RMS in dBFS of a mono signal; returns (db[n_frames], frame_rate_hz)."""
    hop, win = max(1, int(round(hop_s * sr))), max(1, int(round(win_s * sr)))
    p = np.concatenate([[0.0], np.cumsum(x.astype(np.float64) ** 2)])
    idx = np.arange(0, max(1, len(x) - win + 1), hop)
    e = (p[np.minimum(idx + win, len(x))] - p[idx]) / win
    return 10.0 * np.log10(np.maximum(e, 1e-12)), sr / hop


def db(x: float) -> float:
    return float(10.0 * np.log10(max(float(x), 1e-12)))


def to_16k_mono(src: str | Path, dst: str | Path) -> Path:
    """ffmpeg resample to 16 kHz mono PCM16 (idempotent, atomic)."""
    src, dst = Path(src), Path(dst)
    if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime and dst.stat().st_size > 1000:
        return dst
    tmp = dst.with_name(dst.stem + ".tmp" + dst.suffix)
    subprocess.run([str(CFG.paths.ffmpeg), "-hide_banner", "-loglevel", "error", "-y", "-i", str(src), "-ac", "1",
                    "-ar", "16000", "-c:a", "pcm_s16le", str(tmp)], check=True)
    os.replace(tmp, dst)
    return dst


def preload_cuda_libs() -> list[str]:
    """Make the CUDA 12 / cuDNN 9 libraries of the torch wheel visible to ctranslate2 (faster-whisper).

    ctranslate2 dlopen()s libcublas.so.12 / libcudnn*.so.9 by name; they live in site-packages/nvidia/*/lib, which is
    not on the loader path. Loading them once by absolute path (RTLD_GLOBAL) lets later by-name lookups succeed.
    No GPU memory is touched. Returns the libraries loaded.
    """
    import ctypes
    import glob
    import site

    done = []
    roots = [Path(p) / "nvidia" for p in site.getsitepackages()]
    for pat in ("cuda_runtime/lib/libcudart.so.*", "cublas/lib/libcublasLt.so.*", "cublas/lib/libcublas.so.*",
                "cudnn/lib/libcudnn.so.*", "cudnn/lib/libcudnn_*.so.*"):
        for root in roots:
            for f in sorted(glob.glob(str(root / pat))):
                try:
                    ctypes.CDLL(f, mode=ctypes.RTLD_GLOBAL)
                    done.append(Path(f).name)
                except OSError:
                    pass
    return done


# ----------------------------------------------------------------------------- VRAM budget helpers
def vram_budget_gb(default: float | None = None) -> float:
    """Separation VRAM budget in GB: env AUDIO_VRAM_GB, else CFG.audio.separation.vram_gb."""
    v = os.environ.get("AUDIO_VRAM_GB")
    return float(v) if v else float(default if default is not None else ACFG.separation.vram_gb)


def cap_torch_vram(gb: float, reserve_gb: float = 0.7) -> float | None:
    """Hard-cap this process's torch CUDA allocator at (gb - reserve_gb): the job fails with an OOM of its own instead
    of squeezing a neighbour on the shared card. reserve_gb covers the CUDA context / cuDNN / cuBLAS handles that live
    outside the allocator. Returns the cap in GB (None without CUDA)."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        total = torch.cuda.get_device_properties(0).total_memory
        cap = max(0.5, gb - reserve_gb) * 1024 ** 3
        torch.cuda.set_per_process_memory_fraction(min(1.0, cap / total), 0)
        return round(cap / 1024 ** 3, 2)
    except Exception as e:
        log(f"could not set the VRAM cap: {e!r}")
        return None


def is_oom(e: BaseException) -> bool:
    return "out of memory" in str(e).lower() or type(e).__name__ == "OutOfMemoryError"


class VramMonitor:
    """Peak GPU memory of THIS process as nvidia-smi sees it (context + allocator + non-torch libraries), polled in a
    background thread. Use as a context manager; `.peak_mb` is 0 when the process never touched the GPU."""

    def __init__(self, interval: float = 0.5):
        self.interval, self.peak_mb, self._stop, self._t = interval, 0, None, None

    def _poll(self) -> None:
        pid = str(os.getpid())
        while not self._stop.is_set():
            try:
                out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=5).stdout
                for line in out.splitlines():
                    a = [x.strip() for x in line.split(",")]
                    if len(a) >= 2 and a[0] == pid and a[1].isdigit():
                        self.peak_mb = max(self.peak_mb, int(a[1]))
            except Exception:
                pass
            self._stop.wait(self.interval)

    def __enter__(self):
        import threading

        self._stop = threading.Event()
        self._t = threading.Thread(target=self._poll, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self._t.join(timeout=6)

    def torch_peaks(self) -> dict:
        try:
            import torch

            if torch.cuda.is_available() and torch.cuda.is_initialized():
                return {"torch_peak_allocated_mb": int(torch.cuda.max_memory_allocated() / 2 ** 20),
                        "torch_peak_reserved_mb": int(torch.cuda.max_memory_reserved() / 2 ** 20)}
        except Exception:
            pass
        return {}

    def stats(self) -> dict:
        return {"peak_vram_mb": int(self.peak_mb), **self.torch_peaks()}

