"""Japanese ASR with word timestamps on the separated vocal stems (faster-whisper large-v3).

Output: data/processed/anime/asr/epNN.json = list of
    {"start", "end", "text", "words": [{"w", "start", "end", "prob"}], "avg_logprob", "no_speech_prob", ["song": true]}
(times in seconds on the episode timeline; "song" marks segments inside OP/ED/insert-song spans, i.e. lyrics) and data/processed/anime/asr/summary.json with per-episode sanity numbers:
    line_recall   share of subtitle dialogue lines (>= 0.4 s) that contain at least one ASR word (+-0.3 s tolerance)
    word_in_dlg   share of ASR words whose midpoint lies inside a subtitle dialogue/song span (+-0.5 s) or an OP/ED span

Idempotent (skips existing epNN.json), atomic writes. The silero VAD shipped with faster-whisper drops music-only
stretches before decoding; `condition_on_previous_text=False` avoids repetition loops.

    . scripts/env.sh
    scripts/gpu_run.sh 4 $KOMI_WORK/venvs/audio/bin/python -m src.anime.audio_asr --episodes all
    # compute type: int8_float16 (~3.1 GB) by default; AUDIO_ASR_VRAM_GB=6 (or --compute-type float16) -> float16 (~4.8 GB)
    # CPU smoke test: ... -m src.anime.audio_asr --episodes 2 --device cpu --start 600 --dur 60 --out /tmp/x.json
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np

from src.anime.audio_common import (ACFG, VramMonitor, dialogue_events, ep_dir, ep_name, load_subs, log, oped_spans, parse_episodes,
                                    song_spans, wav_info)
from src.common.config import CFG
from src.common.io import read_json, write_json

AC = ACFG.asr
OUT_DIR = Path(CFG.paths.processed) / "anime" / "asr"
# Whisper's classic Japanese hallucinations on music / silence (dropped when they make up a whole segment)
HALLUCINATIONS = ("ご視聴ありがとうございました", "ご視聴ありがとうございます", "チャンネル登録", "最後までご視聴", "字幕視聴ありがとう",
                  "おやすみなさい。おやすみなさい。", "次回予告")


def load_model(device: str = "auto", compute_type: str | None = None, cpu_threads: int = 8):
    """faster-whisper large-v3 from the local model dir. device: auto|cuda|cpu."""
    if device == "auto":
        try:
            import torch  # also puts the CUDA 12 / cuDNN 9 libraries of the torch wheel into the process

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
    elif device == "cuda":
        try:
            import torch  # noqa: F401  (pre-loads libcublas / libcudnn for ctranslate2)

            torch.cuda.is_available()
        except Exception:
            pass
    if device == "cuda":
        from src.anime.audio_common import preload_cuda_libs

        preload_cuda_libs()
    from faster_whisper import WhisperModel

    if compute_type:
        ct = compute_type
    elif device != "cuda":
        ct = "int8"
    else:  # float16 (~4.8 GB) only when the ASR budget allows it, else int8_float16 (~3.1 GB)
        budget = float(os.environ.get("AUDIO_ASR_VRAM_GB") or AC.vram_gb)
        ct = str(AC.compute_type) if budget >= 6 else str(AC.get("compute_type_small", "int8_float16"))
    return WhisperModel(str(AC.model_dir), device=device, compute_type=ct, cpu_threads=cpu_threads, local_files_only=True)


def transcribe_array(model, audio: np.ndarray | str, offset: float = 0.0) -> list[dict]:
    """Transcribe 16 kHz mono float32 audio (or a file path); returns segments with word timestamps (+offset)."""
    segments, _info = model.transcribe(
        audio,
        language=str(AC.language),
        beam_size=int(AC.beam_size),
        word_timestamps=True,
        vad_filter=bool(AC.vad_filter),
        vad_parameters={"min_silence_duration_ms": int(AC.vad_min_silence_ms), "speech_pad_ms": int(AC.vad_speech_pad_ms)},
        condition_on_previous_text=bool(AC.condition_on_previous_text),
        no_speech_threshold=float(AC.no_speech_threshold),
        hotwords=str(AC.hotwords) if AC.get("hotwords") else None,
    )
    out = []
    for s in segments:
        text = (s.text or "").strip()
        words = [{"w": w.word, "start": round(float(w.start) + offset, 3), "end": round(float(w.end) + offset, 3),
                  "prob": round(float(w.probability), 4)} for w in (s.words or [])]
        if not text or not words:
            continue
        if any(h in text for h in HALLUCINATIONS) and len(text) < 40:
            continue
        out.append({"start": round(float(s.start) + offset, 3), "end": round(float(s.end) + offset, 3), "text": text, "words": words,
                    "avg_logprob": round(float(s.avg_logprob), 4), "no_speech_prob": round(float(s.no_speech_prob), 4)})
    return out


def transcribe_file(model, path: str | Path, start: float = 0.0, dur: float | None = None) -> list[dict]:
    """Transcribe (part of) an audio file. Stereo / 44.1 kHz inputs are decoded to 16 kHz mono by faster-whisper."""
    if start == 0.0 and dur is None:
        return transcribe_array(model, str(path))
    from faster_whisper.audio import decode_audio

    audio = decode_audio(str(path), sampling_rate=16000)
    a = int(start * 16000)
    b = len(audio) if dur is None else min(len(audio), a + int(dur * 16000))
    return transcribe_array(model, audio[a:b], offset=start)


def sanity(asr: list[dict], subs: list[dict], t0: float = 0.0, t1: float | None = None,
           extra_spans: list[tuple[float, float]] | None = None) -> dict:
    words = [w for s in asr for w in s["words"]]
    mids = np.array([(w["start"] + w["end"]) / 2 for w in words]) if words else np.zeros(0)
    tot = hit = 0
    for _, e in dialogue_events(subs):
        if e["end"] - e["start"] < 0.4 or e["start"] < t0 or (t1 is not None and e["end"] > t1):
            continue
        tot += 1
        hit += bool(((mids >= e["start"] - 0.3) & (mids <= e["end"] + 0.3)).any())
    spans = [(e["start"] - 0.5, e["end"] + 0.5) for e in subs if e.get("kind") in ("dialogue", "song")] + list(extra_spans or [])
    inside = sum(any(a <= m <= b for a, b in spans) for m in mids) if len(mids) else 0
    probs = [w["prob"] for w in words]
    return {"segments": len(asr), "words": len(words), "dialogue_lines": tot,
            "line_recall": round(hit / tot, 4) if tot else None,
            "word_in_dlg": round(inside / len(mids), 4) if len(mids) else None,
            "mean_word_prob": round(float(np.mean(probs)), 4) if probs else None,
            "chars": sum(len(s["text"]) for s in asr)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", default="all")
    ap.add_argument("--device", default="auto", help="auto|cuda|cpu")
    ap.add_argument("--compute-type", default=None)
    ap.add_argument("--source", default="vocals", choices=["vocals", "mix"], help="vocals.wav (default) or audio_jpn_16k.wav")
    ap.add_argument("--start", type=float, default=0.0, help="excerpt start (s) - test mode, needs --out")
    ap.add_argument("--dur", type=float, default=None, help="excerpt duration (s) - test mode, needs --out")
    ap.add_argument("--out", default=None, help="write to this file instead of data/processed/anime/asr/epNN.json")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    excerpt = args.start > 0 or args.dur is not None
    if excerpt and not args.out:
        raise SystemExit("--start/--dur is a test mode: pass --out so the real epNN.json is not overwritten with a partial transcript")
    eps = parse_episodes(args.episodes)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = None
    failed = []
    summary = read_json(OUT_DIR / "summary.json", default={}) or {}
    for ep in eps:
        out = Path(args.out) if args.out else OUT_DIR / f"{ep_name(ep)}.json"
        if out.exists() and not args.force:
            continue
        src = ep_dir(ep) / ("vocals.wav" if args.source == "vocals" else "audio_jpn_16k.wav")
        if not src.exists():
            log(f"{ep_name(ep)}: {src.name} missing, skipping (re-run later)")
            failed.append(ep)
            continue
        try:
            if model is None:
                t0 = time.time()
                model = load_model(args.device, args.compute_type, args.threads)
                log(f"faster-whisper loaded in {time.time() - t0:.1f}s")
            t0 = time.time()
            with VramMonitor() as mon:
                asr = transcribe_file(model, src, args.start, args.dur)
            dt = time.time() - t0
            subs = load_subs(ep)
            dur = args.dur or wav_info(src)[0]
            songs = song_spans(subs, pad=2.0) + oped_spans(ep)   # OP/ED/insert songs: lyrics are transcribed too
            for seg in asr:
                mid = (seg["start"] + seg["end"]) / 2
                if any(a <= mid <= b for a, b in songs):
                    seg["song"] = True
            st = sanity(asr, subs, args.start, args.start + dur if excerpt else None, extra_spans=songs)
            st["song_segments"] = sum(1 for seg in asr if seg.get("song"))
            st.update({"runtime_seconds": round(dt, 1), "realtime_factor": round(dur / max(dt, 1e-6), 1), "source": args.source,
                       "compute_type": getattr(model.model, "compute_type", None), "peak_vram_mb": mon.peak_mb or None})
            write_json(out, asr, indent=None)
            if not excerpt and not args.out:
                summary[ep_name(ep)] = st
                write_json(OUT_DIR / "summary.json", summary)
            log(f"{ep_name(ep)}: {st}")
        except Exception as e:
            log(f"{ep_name(ep)}: FAILED {e!r}")
            failed.append(ep)
    if summary and not excerpt and not args.out:
        vals = [v for k, v in summary.items() if k.startswith("ep") and v.get("line_recall") is not None]
        if vals:
            summary["_all"] = {"episodes": len(vals), "line_recall": round(float(np.mean([v["line_recall"] for v in vals])), 4),
                               "word_in_dlg": round(float(np.mean([v["word_in_dlg"] for v in vals if v.get("word_in_dlg") is not None])), 4),
                               "mean_runtime_seconds": round(float(np.mean([v["runtime_seconds"] for v in vals])), 1)}
            write_json(OUT_DIR / "summary.json", summary)
            log(f"ASR summary: {summary['_all']}")
    if failed:
        raise SystemExit(f"ASR incomplete for episodes {failed}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
