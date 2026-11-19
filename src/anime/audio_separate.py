"""Vocal / instrumental separation of the Japanese episode audio (brief §3b, §5.2).

Library: python-audio-separator (`audio_separator.separator.Separator`), the library the UVR5_UI Space wraps.
Model:   --model > data/interim/anime/separation_choice.json (written by audio_sepcompare) > CFG.audio.separation.model.

Outputs (44.1 kHz stereo PCM16):
    data/interim/anime/epNN/vocals.wav
    data/interim/anime/epNN/instrumental.wav
    data/interim/anime/epNN/separation.json   (model, settings, runtime, peak VRAM, residual check)

VRAM budget: env AUDIO_VRAM_GB (default CFG.audio.separation.vram_gb = 5). The torch allocator of this process is
hard-capped at budget - 0.7 GB, so the job can only run itself out of memory, never a neighbour on the shared card.
The model's own chunk length (dim_t, 801 frames = 8 s) is tried first; on a CUDA OOM the next smaller segment of
CFG.audio.separation.segment_ladder is used (shorter context = slightly lower quality, measured by audio_sepcompare)
and remembered in data/interim/anime/separation_settings.json so all episodes of a run use the same setting.
The overlap-add buffers of the library live on the CPU, so VRAM does not grow with the episode length.

Idempotent / resumable: an episode is skipped when both stems exist with the duration of the source; stems are written to
a temp dir and renamed.

    . scripts/env.sh
    AUDIO_VRAM_GB=5 scripts/gpu_run.sh 5 $KOMI_WORK/venvs/audio/bin/python -m src.anime.audio_separate --episodes 1,2,22-24
"""
from __future__ import annotations

import argparse
import gc
import glob
import logging
import os
import shutil
import time
from pathlib import Path

import numpy as np

from src.anime.audio_common import (ACFG, VramMonitor, cap_torch_vram, ep_dir, ep_name, is_oom, log, parse_episodes,
                                    read_wav, vram_budget_gb, wav_info)
from src.common.config import CFG
from src.common.io import read_json, write_json

SCFG = ACFG.separation
CHOICE_FILE = Path(CFG.paths.interim) / "anime" / "separation_choice.json"
SETTINGS_FILE = Path(CFG.paths.interim) / "anime" / "separation_settings.json"


def chosen_model(cli_model: str | None = None) -> str:
    """--model > data/interim/anime/separation_choice.json (written by audio_sepcompare --write-choice) > config."""
    if cli_model:
        return cli_model
    if CHOICE_FILE.exists():
        try:
            return str(read_json(CHOICE_FILE)["model"])
        except Exception:
            pass
    return str(SCFG.model)


def ensure_local_weights() -> Path:
    """Symlink checkpoints that already sit in the HF cache into the separator's model dir (no re-download)."""
    mdir = Path(SCFG.model_dir)
    mdir.mkdir(parents=True, exist_ok=True)
    for name, pattern in dict(SCFG.local_weights).items():
        dst = mdir / name
        if dst.exists():
            continue
        hits = sorted(glob.glob(os.path.expanduser(str(pattern))))
        if hits:
            if dst.is_symlink():
                dst.unlink()
            dst.symlink_to(os.path.realpath(hits[-1]))
    return mdir


def make_separator(model: str | None = None, output_dir: str | Path | None = None, overlap: int | None = None,
                   segment: int | None = None):
    """Build an audio-separator `Separator` and load `model`.

    segment=None uses the model's own chunk length (dim_t); an int overrides it (smaller = less VRAM, less context).
    """
    from audio_separator.separator import Separator

    mdir = ensure_local_weights()
    out = Path(output_dir or (Path(CFG.paths.tmp) / "separate"))
    out.mkdir(parents=True, exist_ok=True)
    mdx = {
        "segment_size": int(segment or SCFG.segment_size),
        "override_model_segment_size": bool(segment) or bool(SCFG.override_model_segment_size),
        "batch_size": int(SCFG.batch_size),      # not used by the library for Roformer models (one chunk at a time)
        "overlap": int(overlap or SCFG.overlap),
        "pitch_shift": 0,
    }
    sep = Separator(
        log_level=logging.WARNING,
        model_file_dir=str(mdir),
        output_dir=str(out),
        output_format="WAV",
        # keep the stems at the gain of the mix (vocals + instrumental == mix); only scale down if a stem would clip
        normalization_threshold=1.0,
        amplification_threshold=0.0,
        sample_rate=int(SCFG.sample_rate),
        use_soundfile=True,
        use_autocast=bool(SCFG.use_autocast),
        mdxc_params=mdx,
        demucs_params={"segment_size": "Default", "shifts": 2, "overlap": 0.25, "segments_enabled": True},
    )
    sep.load_model(model_filename=model or chosen_model())
    return sep


def model_dim_t(sep) -> int | None:
    """The loaded Roformer/MDXC model's own chunk length in STFT frames (None for other architectures)."""
    try:
        return int(sep.model_instance.model_data_cfgdict.inference.dim_t)
    except Exception:
        return None


def _pick(files: list[str], out_dir: Path, key: str) -> Path | None:
    for f in files:
        if key.lower() in Path(f).name.lower():
            p = Path(f)
            return p if p.is_absolute() else out_dir / p
    return None


def separate_file(sep, src: Path, vocals_out: Path, inst_out: Path, tag: str) -> dict:
    """Separate one file with a loaded separator; atomically place the two stems. Returns timing info."""
    out_dir = Path(sep.output_dir)
    names = {"Vocals": f"{tag}_vocals", "Instrumental": f"{tag}_instrumental", "Other": f"{tag}_other",
             "Drums": f"{tag}_drums", "Bass": f"{tag}_bass"}
    t0 = time.time()
    files = sep.separate(str(src), custom_output_names=names)
    dt = time.time() - t0
    voc = _pick(files, out_dir, "_vocals")
    inst = _pick(files, out_dir, "_instrumental")
    if voc is None:
        raise RuntimeError(f"no vocal stem produced for {src}: {files}")
    if inst is None:
        # 4-stem models (Demucs) / models that call the second stem "other": instrumental = mix - vocals
        import soundfile as sf

        mix, sr = read_wav(src)
        v, _ = read_wav(voc)
        n = min(len(mix), len(v))
        inst = out_dir / f"{tag}_instrumental.wav"
        sf.write(str(inst), np.clip(mix[:n] - v[:n], -1.0, 1.0), sr, subtype="PCM_16")
        for f in files:
            p = Path(f) if Path(f).is_absolute() else out_dir / f
            if p != voc and p.exists():
                p.unlink()
    for a, b in ((voc, vocals_out), (inst, inst_out)):
        b.parent.mkdir(parents=True, exist_ok=True)
        tmp = b.with_name(b.stem + ".tmp" + b.suffix)
        shutil.move(str(a), str(tmp))
        os.replace(tmp, b)
    return {"seconds": round(dt, 2)}


class BudgetedSeparator:
    """A separator that stays inside a VRAM budget: model chunk first, smaller segments after a CUDA OOM.

    `separate()` returns {"seconds", "segment", "dim_t_default", "peak_vram_mb", "torch_peak_reserved_mb", ...}.
    """

    def __init__(self, model: str, output_dir: str | Path | None = None, overlap: int | None = None,
                 vram_gb: float | None = None, remember: bool = True):
        self.model, self.output_dir, self.overlap = model, output_dir, overlap
        self.vram_gb = vram_budget_gb() if vram_gb is None else float(vram_gb)
        self.cap_gb = cap_torch_vram(self.vram_gb)
        self.remember = remember
        self.key = f"{model}|{self.vram_gb:g}GB|overlap{int(overlap or SCFG.overlap)}"
        saved = (read_json(SETTINGS_FILE, default={}) or {}).get(self.key, {}) if remember else {}
        self.segment: int | None = saved.get("segment")
        self.dim_t: int | None = saved.get("dim_t_default")
        self.sep = None
        self.oom_events: list[int | None] = []

    def close(self) -> None:
        self.sep = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _next_segment(self) -> int | None:
        cur = self.segment or self.dim_t or 10 ** 9
        smaller = [int(x) for x in SCFG.segment_ladder if int(x) < cur]
        return max(smaller) if smaller else None

    def _save(self) -> None:
        if not self.remember:
            return
        d = read_json(SETTINGS_FILE, default={}) or {}
        d[self.key] = {"segment": self.segment, "dim_t_default": self.dim_t, "updated": time.strftime("%Y-%m-%d %H:%M:%S")}
        write_json(SETTINGS_FILE, d)

    def separate(self, src: Path, vocals_out: Path, inst_out: Path, tag: str) -> dict:
        while True:
            try:
                if self.sep is None:
                    self.sep = make_separator(self.model, self.output_dir, self.overlap, segment=self.segment)
                    self.dim_t = model_dim_t(self.sep) or self.dim_t
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.reset_peak_memory_stats()
                except Exception:
                    pass
                with VramMonitor() as mon:
                    info = separate_file(self.sep, src, vocals_out, inst_out, tag)
                info.update(mon.stats())
                info.update({"segment": self.segment or self.dim_t, "dim_t_default": self.dim_t, "vram_budget_gb": self.vram_gb,
                             "vram_cap_gb": self.cap_gb, "oom_fallbacks": list(self.oom_events)})
                self._save()
                return info
            except Exception as e:
                if not is_oom(e):
                    raise
                failed = self.segment or self.dim_t
                msg = str(e)[:160]
                del e
                self.close()
                nxt = self._next_segment()
                if nxt is None:
                    raise RuntimeError(f"out of memory even at the smallest segment ({failed}) within {self.vram_gb} GB: {msg}")
                log(f"CUDA OOM at segment {failed} within the {self.vram_gb} GB budget -> retrying with segment {nxt}")
                self.oom_events.append(failed)
                self.segment = nxt


def residual_db(mix_path: Path, voc_path: Path, inst_path: Path, start: float = 300.0, dur: float = 120.0) -> float:
    """10*log10( ||mix - (vocals+instrumental)||^2 / ||mix||^2 ) on a 2-minute window (reconstruction check)."""
    mix, _ = read_wav(mix_path, start, dur)
    v, _ = read_wav(voc_path, start, dur)
    i, _ = read_wav(inst_path, start, dur)
    n = min(len(mix), len(v), len(i))
    if n == 0:
        return float("nan")
    r = mix[:n] - v[:n] - i[:n]
    return float(10 * np.log10((np.mean(r ** 2) + 1e-12) / (np.mean(mix[:n] ** 2) + 1e-12)))


def stems_ok(ep: int) -> bool:
    d = ep_dir(ep)
    src, voc, inst = d / "audio_jpn.wav", d / "vocals.wav", d / "instrumental.wav"
    if not (voc.exists() and inst.exists() and src.exists()):
        return False
    try:
        ds, dv, di = wav_info(src)[0], wav_info(voc)[0], wav_info(inst)[0]
    except Exception:
        return False
    return abs(ds - dv) < 0.5 and abs(ds - di) < 0.5


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", default="all", help="all | train | heldout | 1,2,5 | 1-21 (processed in the given order)")
    ap.add_argument("--model", default=None, help="audio-separator model filename (default: choice file, then config)")
    ap.add_argument("--vram-gb", type=float, default=None, help="VRAM budget in GB (default: env AUDIO_VRAM_GB, then config)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    eps = parse_episodes(args.episodes)
    model = chosen_model(args.model)
    todo = [ep for ep in eps if args.force or not stems_ok(ep)]
    budget = vram_budget_gb() if args.vram_gb is None else args.vram_gb
    log(f"separation: model={model} vram_budget={budget} GB overlap={int(SCFG.overlap)} episodes={eps} todo={todo}")
    if not todo:
        return
    bs = BudgetedSeparator(model, vram_gb=budget)
    failed: list[int] = []
    for ep in todo:
        d = ep_dir(ep)
        src = d / "audio_jpn.wav"
        if not src.exists():
            log(f"{ep_name(ep)}: {src} not extracted yet, skipping (re-run later)")
            failed.append(ep)
            continue
        try:
            info = bs.separate(src, d / "vocals.wav", d / "instrumental.wav", tag=ep_name(ep))
        except Exception as e:  # keep going with the next episode; the caller can simply re-run
            log(f"{ep_name(ep)}: FAILED {e!r}")
            failed.append(ep)
            bs.close()
            continue
        dur = wav_info(src)[0]
        res = residual_db(src, d / "vocals.wav", d / "instrumental.wav")
        meta = {"episode": ep, "model": model, "library": "audio-separator", "overlap": int(SCFG.overlap),
                "audio_seconds": round(dur, 2), "runtime_seconds": info["seconds"],
                "realtime_factor": round(dur / max(info["seconds"], 1e-6), 2), "residual_db": round(res, 2),
                **{k: v for k, v in info.items() if k != "seconds"}, "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
        write_json(d / "separation.json", meta)
        log(f"{ep_name(ep)}: done in {info['seconds']:.0f}s ({meta['realtime_factor']}x realtime), segment {info['segment']}, "
            f"peak VRAM {info.get('peak_vram_mb')} MiB, residual {res:.1f} dB")
    if failed:
        log(f"separation: FAILED/SKIPPED episodes {failed}")
        raise SystemExit(1)
    log("separation: all requested episodes processed")


if __name__ == "__main__":
    main()
