"""English-dub audio stack: extraction, vocal separation, ASR, dub lines, per-line speaker embeddings.

The Japanese stack (audio_separate / audio_asr / audio_embed / speakers / name_speakers) stays untouched; this module
re-uses its building blocks on the `eng` audio track and writes next to the Japanese outputs under English names:

    data/interim/anime/epNN/audio_eng.wav        44.1 kHz stereo mix of the dub (TEMPORARY: removed after separation,
                                                 re-extractable from the mkv in ~10 s with `extract`)
    data/interim/anime/epNN/vocals_en.wav        separated dub speech, 44.1 kHz MONO PCM16 (the dub dialogue is centred;
                                                 mono halves the disk use; the instrumental stem is not kept - the
                                                 music is the same as in the Japanese stems)
    data/interim/anime/epNN/vocals_en_16k.wav    16 kHz mono copy (ASR / speaker embeddings): a cache, rebuilt on demand
                                                 (deleted after the first full run to save 1 GB)
    data/interim/anime/epNN/separation_en.json   model, settings, runtime, peak VRAM, residual check
    data/processed/anime/asr_en/epNN.json        faster-whisper large-v3 (language=en), segments with word timestamps
    data/processed/anime/asr_en/summary.json
    data/processed/anime/dub_lines/epNN.json     the dub line table BEFORE naming: [{"idx","start","end","text","words",
                                                 "avg_logprob","word_prob","song"}]  (idx = position in this file)
    data/processed/anime/spk_en/epNN.npz         WeSpeaker ResNet293 embedding per dub line (idx -> dub_lines idx)

Timeline: ffmpeg drops the container start offset of an audio stream when it extracts it, and the season-1 files carry
the dub with a 1 s stream delay. `sync` measures the lag of the dub against the Japanese track (cross-correlation of the
two mix envelopes: music and effects are shared) and shifts vocals_en.wav so that ALL English times are on the same
timeline as the video, the subtitles and lines_named.jsonl (data/interim/anime/sync_en.json holds the measured lags).

Only TRAIN episodes (1-21) are processed unless episodes are given explicitly; held-out episodes are refused for
everything that feeds voice references (see src/anime/speakers_en.py and src/train/voice_bank.py).

    . scripts/env.sh ; PY=$KOMI_WORK/venvs/audio/bin/python
    $PY -m src.anime.audio_en extract  --episodes train
    AUDIO_VRAM_GB=5 scripts/gpu_run.sh 5 $PY -m src.anime.audio_en separate --episodes train   # includes `sync`
    $PY -m src.anime.audio_en sync --episodes train          # (only needed for stems written before `sync` existed)
    AUDIO_ASR_VRAM_GB=6 scripts/gpu_run.sh 6 $PY -m src.anime.audio_en asr --episodes train
    $PY -m src.anime.audio_en lines --episodes train
    scripts/gpu_run.sh 2 $PY -m src.anime.audio_en embed --episodes train
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import time
from pathlib import Path

import numpy as np

from src.anime.audio_common import (ACFG, HELDOUT_EPS, TRAIN_EPS, VramMonitor, cap_torch_vram, ep_dir, ep_name, load_subs,
                                    log, oped_spans, parse_episodes, read_wav, song_spans, to_16k_mono, vram_budget_gb,
                                    wav_info, write_wav_atomic)
from src.common.config import CFG
from src.common.io import read_json, write_json

EN = ACFG.dub_en
PROC = Path(CFG.paths.processed) / "anime"
ASR_DIR = PROC / "asr_en"
LINES_DIR = PROC / "dub_lines"
SPK_DIR = Path(EN.spk_dir)
SR16 = 16000


def raw_file(ep: int) -> Path:
    hits = sorted((Path(CFG.paths.raw) / "anime").glob(f"ep{ep:02d}_*.mkv"))
    if not hits:
        raise FileNotFoundError(f"no source mkv for episode {ep}")
    return hits[0]


def _episodes(spec: str | None, allow_heldout: bool = False) -> list[int]:
    eps = parse_episodes(spec or "train")
    bad = [e for e in eps if e in HELDOUT_EPS]
    if bad and not allow_heldout:
        raise SystemExit(f"held-out episodes {bad} are not processed by the English voice stack (split rule)")
    return eps


# ============================================================================================ extract
def extract(ep: int, force: bool = False) -> Path:
    """`eng` audio track -> audio_eng.wav (44.1 kHz stereo PCM16, same format as audio_jpn.wav)."""
    out = ep_dir(ep) / "audio_eng.wav"
    if out.exists() and out.stat().st_size > 1e6 and not force:
        return out
    tmp = out.with_name("audio_eng.tmp.wav")
    subprocess.run([str(CFG.paths.ffmpeg), "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", str(raw_file(ep)),
                    "-map", f"0:a:m:language:{EN.audio_lang}", "-ac", "2", "-ar", "44100", "-c:a", "pcm_s16le", str(tmp)],
                   check=True, stdin=subprocess.DEVNULL)
    os.replace(tmp, out)
    return out



# ============================================================================================ sync to the video timeline
SYNC_FILE = Path(CFG.paths.interim) / "anime" / "sync_en.json"


def _envelope(path: Path, lang: str, hop: int = 80) -> np.ndarray:
    """100 Hz log-energy envelope of one audio stream of the mkv (decoded exactly like `extract` does)."""
    r = subprocess.run([str(CFG.paths.ffmpeg), "-nostdin", "-loglevel", "error", "-i", str(path), "-map",
                        f"0:a:m:language:{lang}", "-ac", "1", "-ar", "8000", "-f", "s16le", "-"],
                       capture_output=True, stdin=subprocess.DEVNULL, check=True)
    x = np.frombuffer(r.stdout, dtype=np.int16).astype(np.float32) / 32768
    n = len(x) // hop
    return np.log10(np.maximum((x[: n * hop].reshape(n, hop) ** 2).mean(1), 1e-9))


def sync_lag(ep: int, force: bool = False) -> dict:
    """Lag of the extracted dub against the extracted Japanese track: {"lag": t_en - t_jp of the same content (s),
    "windows": [[lag, corr], ...]}. Four 180 s windows; the lag is the correlation-weighted median."""
    from scipy.signal import fftconvolve

    doc = read_json(SYNC_FILE, default={}) or {}
    if ep_name(ep) in doc and not force:
        return doc[ep_name(ep)]
    f = raw_file(ep)
    j, e = _envelope(f, str(CFG.anime.audio_lang)), _envelope(f, str(EN.audio_lang))
    wins = []
    for t0 in (60, 400, 800, 1150):
        a, b = j[t0 * 100:(t0 + 180) * 100], e[(t0 - 10) * 100:(t0 + 190) * 100]
        if len(a) < 18000 or len(b) < 20000:
            continue
        a, b = (a - a.mean()) / (a.std() + 1e-9), (b - b.mean()) / (b.std() + 1e-9)
        c = fftconvolve(b, a[::-1], mode="valid") / len(a)
        wins.append([round((int(c.argmax()) - 1000) / 100, 2), round(float(c.max()), 3)])
    good = [w for w in wins if w[1] >= 0.4] or wins
    lag = float(np.median([w[0] for w in good])) if good else 0.0
    res = {"lag": round(lag, 2), "windows": wins, "agree": bool(good and max(abs(w[0] - lag) for w in good) <= 0.2)}
    doc = read_json(SYNC_FILE, default={}) or {}
    doc[ep_name(ep)] = res
    write_json(SYNC_FILE, doc)
    return res


def apply_sync(ep: int) -> float | None:
    """Shift vocals_en.wav onto the video timeline (idempotent: recorded as `sync_shift_sec` in separation_en.json)."""
    d = ep_dir(ep)
    meta_p, v = d / "separation_en.json", d / "vocals_en.wav"
    meta = read_json(meta_p)
    if meta is None or not v.exists():
        return None
    if "sync_shift_sec" in meta:
        return float(meta["sync_shift_sec"])
    lag = float(sync_lag(ep)["lag"])
    shift = -lag                                   # seconds to ADD to every time of the extracted dub
    if abs(shift) >= 0.02:
        x, sr = read_wav(v, mono=True)
        n = int(round(abs(shift) * sr))
        x = np.concatenate([np.zeros(n, np.float32), x]) if shift > 0 else x[n:]
        write_wav_atomic(v, x, sr)
    else:
        shift = 0.0
    to_16k_mono(v, d / "vocals_en_16k.wav")
    meta.update(sync_shift_sec=round(shift, 2), sync=sync_lag(ep), source_seconds=meta.get("source_seconds", meta["audio_seconds"]),
                audio_seconds=round(wav_info(v)[0], 2))
    write_json(meta_p, meta)
    return shift


# ============================================================================================ separate
def stems_ok(ep: int) -> bool:
    d = ep_dir(ep)
    v, meta = d / "vocals_en.wav", d / "separation_en.json"
    if not (v.exists() and meta.exists()):
        return False
    try:
        return abs(wav_info(v)[0] - float(read_json(meta)["audio_seconds"])) < 0.5
    except Exception:
        return False


def separate(eps: list[int], force: bool = False, keep_mix: bool = False, model: str | None = None) -> None:
    """Same separation model and settings as the Japanese stems (audio_separate.BudgetedSeparator)."""
    from src.anime.audio_separate import SCFG, BudgetedSeparator, chosen_model

    todo = [ep for ep in eps if force or not stems_ok(ep)]
    model = chosen_model(model)
    budget = vram_budget_gb()
    log(f"EN separation: model={model} budget={budget} GB overlap={int(SCFG.overlap)} todo={todo}")
    if not todo:
        return
    tmp_dir = Path(CFG.paths.tmp) / f"separate_en_{os.getpid()}"
    bs = BudgetedSeparator(model, output_dir=tmp_dir, vram_gb=budget, remember=False)   # no shared settings file: workers may run in parallel
    failed = []
    for ep in todo:
        d = ep_dir(ep)
        try:
            src = extract(ep)
            dur = wav_info(src)[0]
            v_st, i_st = tmp_dir / f"{ep_name(ep)}_v.wav", tmp_dir / f"{ep_name(ep)}_i.wav"
            info = bs.separate(src, v_st, i_st, tag=f"{ep_name(ep)}en")
            # reconstruction check on a 2-minute window, then keep only the mono vocal stem
            mix, sr = read_wav(src, 300.0, 120.0)
            v, _ = read_wav(v_st, 300.0, 120.0)
            i, _ = read_wav(i_st, 300.0, 120.0)
            n = min(len(mix), len(v), len(i))
            r = mix[:n] - v[:n] - i[:n]
            res = float(10 * np.log10((np.mean(r ** 2) + 1e-12) / (np.mean(mix[:n] ** 2) + 1e-12)))
            vfull, sr = read_wav(v_st)
            side = float(10 * np.log10((np.mean((vfull[:, 0] - vfull[:, 1]) ** 2) / 4 + 1e-12)
                                       / (np.mean(vfull.mean(axis=1) ** 2) + 1e-12)))   # side/mid energy of the stem
            write_wav_atomic(d / "vocals_en.wav", vfull.mean(axis=1), sr)
            del vfull
            v_st.unlink(missing_ok=True)
            i_st.unlink(missing_ok=True)
            to_16k_mono(d / "vocals_en.wav", d / "vocals_en_16k.wav")
            meta = {"episode": ep, "track": str(EN.audio_lang), "model": model, "library": "audio-separator",
                    "overlap": int(SCFG.overlap), "audio_seconds": round(dur, 2), "runtime_seconds": info["seconds"],
                    "realtime_factor": round(dur / max(info["seconds"], 1e-6), 2), "residual_db": round(res, 2),
                    "vocal_side_to_mid_db": round(side, 2), "stored": "vocals_en.wav mono 44.1 kHz PCM16 (instrumental not kept)",
                    **{k: v for k, v in info.items() if k != "seconds"}, "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
            write_json(d / "separation_en.json", meta)
            shift = apply_sync(ep)
            if not keep_mix:
                src.unlink(missing_ok=True)
            log(f"{ep_name(ep)}: EN stem done in {info['seconds']:.0f}s ({meta['realtime_factor']}x realtime), segment "
                f"{info['segment']}, peak VRAM {info.get('peak_vram_mb')} MiB, residual {res:.1f} dB, side/mid {side:.1f} dB, shifted {shift:+.2f}s onto the video timeline")
        except Exception as e:  # noqa: BLE001 - keep going; the stage is resumable
            log(f"{ep_name(ep)}: FAILED {e!r}")
            failed.append(ep)
            bs.close()
    try:
        tmp_dir.rmdir()
    except OSError:
        pass
    if failed:
        raise SystemExit(f"EN separation incomplete for episodes {failed}")


# ============================================================================================ ASR
_EN_HALLU = ("thanks for watching", "thank you for watching", "subtitles by", "subscribe to", "amara.org", "transcribed by",
             "please subscribe", "see you next time", "translated by")


def transcribe_en(model, audio, offset: float = 0.0, batch_size: int = 0) -> list[dict]:
    """English ASR with word timestamps (same decoding settings as the Japanese pass, language=en, no hotwords:
    character names are passed as initial_prompt so that their spelling is stable).
    batch_size > 0 uses faster-whisper's BatchedInferencePipeline (VAD chunks decoded in parallel: several times
    faster; segments are then whole VAD chunks, which is fine because dub lines are cut from the word timings)."""
    AC = ACFG.asr
    kw = dict(language="en", beam_size=int(AC.beam_size), word_timestamps=True, vad_filter=bool(AC.vad_filter),
              vad_parameters={"min_silence_duration_ms": int(AC.vad_min_silence_ms), "speech_pad_ms": int(AC.vad_speech_pad_ms)},
              no_speech_threshold=float(AC.no_speech_threshold),
              initial_prompt=str(EN.asr_prompt) if EN.get("asr_prompt") else None)
    if batch_size and batch_size > 0:
        from faster_whisper import BatchedInferencePipeline

        segments, _ = BatchedInferencePipeline(model=model).transcribe(audio, batch_size=int(batch_size), **kw)
    else:
        segments, _ = model.transcribe(audio, condition_on_previous_text=False, **kw)
    out = []
    for s in segments:
        text = (s.text or "").strip()
        words = [{"w": w.word, "start": round(float(w.start) + offset, 3), "end": round(float(w.end) + offset, 3),
                  "prob": round(float(w.probability), 4)} for w in (s.words or [])]
        if not text or not words:
            continue
        if any(h in text.lower() for h in _EN_HALLU) and len(text) < 60:
            continue
        out.append({"start": round(float(s.start) + offset, 3), "end": round(float(s.end) + offset, 3), "text": text,
                    "words": words, "avg_logprob": round(float(s.avg_logprob), 4),
                    "no_speech_prob": round(float(s.no_speech_prob), 4)})
    return out


def asr(eps: list[int], device: str = "auto", compute_type: str | None = None, force: bool = False,
        start: float = 0.0, dur: float | None = None, out: str | None = None, source: str = "vocals",
        batch_size: int = 0) -> None:
    from src.anime.audio_asr import load_model

    ASR_DIR.mkdir(parents=True, exist_ok=True)
    summary = read_json(ASR_DIR / "summary.json", default={}) or {}
    model = None
    for ep in eps:
        dst = Path(out) if out else ASR_DIR / f"{ep_name(ep)}.json"
        if dst.exists() and not force:
            continue
        src = ep_dir(ep) / ("vocals_en_16k.wav" if source == "vocals" else "audio_eng.wav")
        if source == "vocals" and not src.exists() and (ep_dir(ep) / "vocals_en.wav").exists():
            to_16k_mono(ep_dir(ep) / "vocals_en.wav", src)      # the 16 kHz copy is a cache: rebuilt on demand
        if not src.exists():
            log(f"{ep_name(ep)}: {src.name} missing (run `separate` first)")
            continue
        if source == "vocals" and apply_sync(ep) is None:
            log(f"{ep_name(ep)}: separation_en.json missing (stem not finished)")
            continue
        if model is None:
            t0 = time.time()
            model = load_model(device, compute_type)
            log(f"faster-whisper loaded in {time.time() - t0:.1f}s")
        from faster_whisper.audio import decode_audio

        audio = decode_audio(str(src), sampling_rate=SR16)
        a = int(start * SR16)
        b = len(audio) if dur is None else min(len(audio), a + int(dur * SR16))
        t0 = time.time()
        with VramMonitor() as mon:
            segs = transcribe_en(model, audio[a:b], offset=start, batch_size=batch_size)
        dt = time.time() - t0
        songs = oped_spans(ep)
        for s in segs:
            mid = (s["start"] + s["end"]) / 2
            if any(x <= mid <= y for x, y in songs):
                s["song"] = True
        write_json(dst, segs, indent=None)
        probs = [w["prob"] for s in segs for w in s["words"]]
        st = {"segments": len(segs), "words": len(probs), "mean_word_prob": round(float(np.mean(probs)), 4) if probs else None,
              "song_segments": sum(1 for s in segs if s.get("song")), "runtime_seconds": round(dt, 1),
              "realtime_factor": round(((b - a) / SR16) / max(dt, 1e-6), 1), "source": src.name,
              "compute_type": getattr(model.model, "compute_type", None), "batch_size": int(batch_size),
              "peak_vram_mb": mon.peak_mb or None}
        if not out:
            summary[ep_name(ep)] = st
            write_json(ASR_DIR / "summary.json", summary)
        log(f"{ep_name(ep)}: EN ASR {st}")


# ============================================================================================ dub lines
_SENT_END = re.compile(r"[.!?…]['\")\]]*$")


def _mk_line(words: list[dict], seg: dict) -> dict:
    text = re.sub(r"\s+", " ", "".join(w["w"] for w in words)).strip()
    return {"start": round(words[0]["start"], 3), "end": round(words[-1]["end"], 3), "text": text,
            "words": [{"w": w["w"].strip(), "start": w["start"], "end": w["end"], "prob": w["prob"]} for w in words],
            "avg_logprob": seg["avg_logprob"], "word_prob": round(float(np.mean([w["prob"] for w in words])), 4)}


def split_segment(seg: dict) -> list[dict]:
    """One Whisper segment -> dub lines: cut at pauses >= lines.split_gap_sec (a speaker change in a dub almost always
    comes with a pause or a sentence end) and at sentence ends once the line is at least lines.min_sentence_sec long."""
    L = EN.lines
    out, cur = [], []
    ws = seg["words"]
    for k, w in enumerate(ws):
        cur.append(w)
        if k == len(ws) - 1:
            break
        gap = ws[k + 1]["start"] - w["end"]
        d = w["end"] - cur[0]["start"]
        sent_end = bool(_SENT_END.search(w["w"].strip()))
        if gap >= float(L.split_gap_sec) or (sent_end and (gap >= float(L.sentence_gap_sec) or d >= float(L.min_sentence_sec))) \
                or d >= float(L.max_sec):
            out.append(_mk_line(cur, seg))
            cur = []
    if cur:
        out.append(_mk_line(cur, seg))
    return out


def build_lines(ep: int, force: bool = False) -> list[dict]:
    """ASR segments -> data/processed/anime/dub_lines/epNN.json (one entry per dub line, `idx` = position)."""
    dst = LINES_DIR / f"{ep_name(ep)}.json"
    if dst.exists() and not force:
        return read_json(dst)
    segs = read_json(ASR_DIR / f"{ep_name(ep)}.json")
    if segs is None:
        raise FileNotFoundError(f"{ASR_DIR / (ep_name(ep) + '.json')} (run `asr` first)")
    L = EN.lines
    # OP / ED / insert songs: by the LINE midpoint (batched ASR segments are whole VAD chunks, too coarse for this)
    songs = list(oped_spans(ep))
    try:
        songs += song_spans(load_subs(ep, generate=False), pad=2.0)
    except FileNotFoundError:
        pass
    lines = []
    for seg in segs:
        for ln in split_segment(seg):
            if ln["word_prob"] < float(L.min_word_prob) or not re.search(r"[A-Za-z0-9]", ln["text"]):
                continue
            mid = (ln["start"] + ln["end"]) / 2
            ln["song"] = any(a <= mid <= b for a, b in songs)
            lines.append(ln)
    lines.sort(key=lambda r: (r["start"], r["end"]))
    for i, ln in enumerate(lines):
        ln["idx"] = i
    write_json(dst, lines, indent=None)
    return lines


# ============================================================================================ embeddings
def embed(eps: list[int], device: str = "auto", force: bool = False) -> None:
    """WeSpeaker ResNet293 embedding per dub line (word-timed span, padded like the Japanese pass)."""
    import soundfile as sf

    from src.anime.audio_embed import MODEL_NAME, SpeakerEmbedder

    SP = ACFG.spk
    SPK_DIR.mkdir(parents=True, exist_ok=True)
    todo = [ep for ep in eps if force or not (SPK_DIR / f"{ep_name(ep)}.npz").exists()]
    if not todo:
        return
    if device != "cpu":
        cap_torch_vram(float(os.environ.get("AUDIO_SPK_VRAM_GB") or SP.vram_gb), reserve_gb=0.5)
    emb = SpeakerEmbedder(device=device)
    log(f"EN speaker embeddings: backend {emb.backend}, todo={todo}")
    for ep in todo:
        lines = build_lines(ep)
        x, sr = sf.read(str(to_16k_mono(ep_dir(ep) / "vocals_en.wav", ep_dir(ep) / "vocals_en_16k.wav")), dtype="float32")
        assert sr == SR16 and x.ndim == 1, (sr, x.shape)
        total = len(x) / SR16
        segs = []
        for n, ln in enumerate(lines):
            s, e = float(ln["start"]), float(ln["end"])
            a, b = s - float(SP.pad), e + float(SP.pad)
            if b - a < float(SP.pad_to):
                ext = (float(SP.pad_to) - (b - a)) / 2
                a, b = a - ext, b + ext
            if b - a > float(SP.max_dur):
                c = (a + b) / 2
                a, b = c - float(SP.max_dur) / 2, c + float(SP.max_dur) / 2
            a, b = max(0.0, a), min(total, b)
            near = lines[max(0, n - 4): n + 5]
            ov = any(o is not ln and min(e, o["end"]) - max(s, o["start"]) > 0.2 for o in near)
            segs.append({"idx": ln["idx"], "start": a, "end": b, "dur": e - s, "overlap": ov})
        wavs = [x[int(s["start"] * SR16): int(s["end"] * SR16)] for s in segs]
        E, sdur = emb.embed(wavs)
        rms = np.array([float(np.sqrt(np.mean(w.astype(np.float64) ** 2))) if len(w) else 0.0 for w in wavs], np.float32)
        keep = (sdur >= 0.25) & (np.linalg.norm(E, axis=1) > 0.5)
        out = SPK_DIR / f"{ep_name(ep)}.npz"
        tmp = out.with_name(out.stem + ".tmp.npz")
        np.savez(tmp, idx=np.array([s["idx"] for s in segs], np.int32)[keep], emb=E[keep].astype(np.float32),
                 dur=np.array([s["dur"] for s in segs], np.float32)[keep], rms=rms[keep],
                 start=np.array([s["start"] for s in segs], np.float32)[keep],
                 end=np.array([s["end"] for s in segs], np.float32)[keep], speech_dur=sdur[keep],
                 overlap=np.array([s["overlap"] for s in segs], bool)[keep],
                 model=np.array(MODEL_NAME), source=np.array("vocals_en"))
        tmp.rename(out)
        log(f"{ep_name(ep)}: {int(keep.sum())}/{len(segs)} dub lines embedded")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["extract", "separate", "sync", "asr", "lines", "embed"])
    ap.add_argument("--episodes", default="train", help="train (default) | 1,2,5 | 1-21")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--compute-type", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--keep-mix", action="store_true", help="separate: keep audio_eng.wav")
    ap.add_argument("--model", default=None)
    ap.add_argument("--start", type=float, default=0.0, help="asr test mode (needs --out)")
    ap.add_argument("--dur", type=float, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--source", default="vocals", choices=["vocals", "mix"])
    ap.add_argument("--batch", type=int, default=0, help="asr: batched decoding (e.g. 8); 0 = sequential")
    ap.add_argument("--allow-heldout", action="store_true", help="only for evaluation data, never for voice references")
    a = ap.parse_args()
    eps = _episodes(a.episodes, a.allow_heldout)
    if a.cmd == "extract":
        for ep in eps:
            p = extract(ep, a.force)
            log(f"{ep_name(ep)}: {p.name} {wav_info(p)[0]:.1f}s")
    elif a.cmd == "separate":
        separate(eps, a.force, a.keep_mix, a.model)
    elif a.cmd == "sync":
        for ep in eps:
            log(f"{ep_name(ep)}: lag {sync_lag(ep, a.force)} -> shift applied {apply_sync(ep)}")
    elif a.cmd == "asr":
        if (a.start > 0 or a.dur is not None) and not a.out:
            raise SystemExit("--start/--dur is a test mode: pass --out")
        asr(eps, a.device, a.compute_type, a.force, a.start, a.dur, a.out, a.source, a.batch)
    elif a.cmd == "lines":
        for ep in eps:
            ls = build_lines(ep, a.force)
            log(f"{ep_name(ep)}: {len(ls)} dub lines, {sum(l['end'] - l['start'] for l in ls):.0f}s")
    elif a.cmd == "embed":
        embed(eps, a.device, a.force)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
