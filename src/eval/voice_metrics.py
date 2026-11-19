"""Voice metrics for generated lines: speaker similarity, intelligibility (CER), duration fit.

  (a) speaker similarity  cosine between ECAPA-TDNN speaker embeddings (SpeechBrain spkrec-ecapa-voxceleb, non-gated)
                          of the generated wav and the character's references (`sim_ref`: the refs used as prompt;
                          `sim_bank`: the voice-bank centroid when --voices is given; `sim_real` / `real_vs_ref`:
                          when a row carries `real_wav`, generated-vs-real and real-vs-refs = the ceiling).
  (b) intelligibility     faster-whisper large-v3 (language=ja) transcript vs the script text. Japanese has no word
                          boundaries, so the metric is CER (character error rate), not WER. Script and transcript
                          are compared in three forms: written (`cer_surface`), kana pronunciation via MeCab
                          (`cer_mecab`) and kana reading via pykakasi (`cer_kakasi`), punctuation removed; `cer` is
                          the lowest of the three, so kanji/kana spelling choices of Whisper do not count as errors.
  (c) duration fit        dur - target_dur (and overshoot of max_dur).
  English rows ("lang": "EN", e.g. from src.generate.tts_qwen): intelligibility is WER (word error rate) with
  language=en ASR; `wer`, `wer_edits`, `wer_ref_len` are added next to the character-level `cer`, the flag is
  `high_wer`, and the aggregate gains `wer` / `wer_micro`. Japanese rows are evaluated exactly as before.
  `prosody_features()` (F0, level, voicing, rate) is the delivery measurement used by src/eval/emotion_probe.py and by
  the English voice bank's emotion-matched reference sets.

CLI (TTS venv):
  $KOMI_WORK/venvs/tts/bin/python -m src.eval.voice_metrics --results <dir>/tts_results.jsonl [--voices voices.json]
      [--device cpu|cuda] [--out <dir>/voice_metrics.json]
The numbers are evidence for the judge (flags), not verdicts.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import statistics
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any

from src.common.config import CFG
from src.common.io import log, read_json, read_jsonl, write_json
from src.generate.tts import T, _lexicon, _tagger, normalize_ja, read_mono, resample, trim_silence

M = T.voice_metrics
_HIRA_RE = re.compile(r"[ぁ-ゖー]")


# ============================================================================================ speaker embeddings
class SpeakerEmbedder:
    """ECAPA-TDNN (VoxCeleb) speaker embeddings, L2-normalised. Small enough to run on CPU."""

    def __init__(self, device: str = "cpu"):
        import torch
        d = Path(M.spk_model_dir)
        if not (d / "embedding_model.ckpt").exists():
            os.environ.setdefault("HF_HOME", str(T.hf_cache))
            import huggingface_hub
            log(f"downloading {M.spk_model} once -> {d}")
            huggingface_hub.snapshot_download(str(M.spk_model), local_dir=str(d))
        from speechbrain.inference.speaker import EncoderClassifier
        self.torch = torch
        self.device = device
        self.model = EncoderClassifier.from_hparams(source=str(d), savedir=str(d), run_opts={"device": device})
        self.model.eval()

    def embed_wave(self, x, sr: int):
        import numpy as np
        x = resample(np.asarray(x, dtype="float32"), sr, 16000)
        if len(x) < 1600:                                   # < 0.1 s: pad so the encoder does not fail
            x = np.pad(x, (0, 1600 - len(x)))
        with self.torch.no_grad():
            e = self.model.encode_batch(self.torch.from_numpy(x).unsqueeze(0).to(self.device))
        e = e.squeeze().detach().cpu().numpy().astype("float32")
        return e / (np.linalg.norm(e) + 1e-9)

    def embed_file(self, path: str | Path, trim: bool = True):
        x, sr = read_mono(path)
        if trim:
            y = trim_silence(x, sr, top_db=40.0, pad_head_ms=20, pad_tail_ms=40)
            x = y if len(y) > 0 else x
        return self.embed_wave(x, sr)


def mean_embedding(embs):
    import numpy as np
    m = np.mean(np.stack(embs), axis=0)
    return m / (np.linalg.norm(m) + 1e-9)


def cos(a, b) -> float:
    import numpy as np
    return float(np.dot(a, b) / ((np.linalg.norm(a) + 1e-9) * (np.linalg.norm(b) + 1e-9)))


# ============================================================================================ ASR
def _preload_cuda_libs() -> None:
    """ctranslate2 needs cuBLAS/cuDNN on the loader path; the torch wheels ship them inside site-packages."""
    import ctypes
    for sp in sys.path:
        for pat in ("nvidia/cublas/lib/libcublasLt.so.*", "nvidia/cublas/lib/libcublas.so.*",
                    "nvidia/cudnn/lib/libcudnn.so.*", "nvidia/cudnn/lib/libcudnn_*.so.*"):
            for f in sorted(glob.glob(os.path.join(sp, pat))):
                try:
                    ctypes.CDLL(f, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass


class Transcriber:
    def __init__(self, device: str = "cpu", threads: int = 8):
        from faster_whisper import WhisperModel
        self.device = device
        if device.startswith("cuda"):
            _preload_cuda_libs()
            self.model = WhisperModel(str(M.asr_model_dir), device="cuda", compute_type=str(M.asr_compute_type))
        else:
            self.model = WhisperModel(str(M.asr_model_dir), device="cpu", compute_type="int8", cpu_threads=threads)

    def transcribe(self, path: str | Path, language: str = "ja") -> str:
        segs, _ = self.model.transcribe(str(path), language=language, beam_size=int(M.asr_beam), temperature=0.0,
                                        condition_on_previous_text=False, vad_filter=False, without_timestamps=True,
                                        initial_prompt=M.get("asr_initial_prompt", None))
        return "".join(s.text for s in segs).strip()


# ============================================================================================ CER
def pronunciation(text: str) -> str:
    """Kana pronunciation (MeCab `pron`: 高校 -> コーコー, は(particle) -> ワ) as hiragana, kana only."""
    tg = _tagger()
    if not tg:
        s = text
    else:
        parts = []
        for w in tg(text):
            p = getattr(w.feature, "pron", None) or getattr(w.feature, "kana", None)
            parts.append(p if (p and p != "*") else w.surface)
        s = "".join(parts)
    s = "".join(chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c for c in s)
    return "".join(_HIRA_RE.findall(s))


_KKS = None


def pronunciation_kakasi(text: str) -> str | None:
    """Second, independent kana reading (pykakasi dictionary). None when pykakasi is not installed."""
    global _KKS
    if _KKS is None:
        try:
            import pykakasi
            _KKS = pykakasi.kakasi()
        except Exception:
            _KKS = False
    if not _KKS:
        return None
    s = "".join(x["hira"] for x in _KKS.convert(text))
    s = "".join(chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c for c in s)
    return "".join(_HIRA_RE.findall(s))


def normalize_for_cer(text: str, lang: str = "JA") -> dict[str, str]:
    """-> comparable forms of a text: {"surface": written form without punctuation, "mecab": pronunciation via
    MeCab/unidic-lite, "kakasi": reading via pykakasi}. Each is one way to cancel spelling differences."""
    if lang.upper() != "JA":
        s = re.sub(r"[^\w]", "", unicodedata.normalize("NFKC", text or "").lower())
        return {"surface": s}
    t = normalize_ja(text or "", _lexicon())          # same front-end as the TTS: names, numbers, ruby
    surface = "".join(c for c in t if unicodedata.category(c)[0] in ("L", "N") or c == "ー")
    surface = "".join(chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c for c in surface)
    out = {"surface": surface, "mecab": pronunciation(t)}
    k = pronunciation_kakasi(t)
    if k is not None:
        out["kakasi"] = k
    return out


def cer_best(ref_text: str, hyp_text: str, lang: str = "JA") -> dict:
    """CER under every normalisation; the reported `cer` is the lowest one (a transcript that matches the script in
    any of the representations is a correct reading: the G2P tools mis-read some words, e.g. unidic-lite reads
    コミュ障 as "こみゅさわり")."""
    ref, hyp = normalize_for_cer(ref_text, lang), normalize_for_cer(hyp_text, lang)
    res: dict[str, Any] = {}
    best = None
    for name in ref:
        c, e, n = cer(ref[name], hyp.get(name, ""))
        res[f"cer_{name}"] = None if c is None else round(c, 4)
        if c is not None and (best is None or c < best[0]):
            best = (c, e, n, name)
    if best is None:
        res.update(cer=None, cer_edits=0, cer_ref_len=0, cer_variant=None)
    else:
        res.update(cer=round(best[0], 4), cer_edits=best[1], cer_ref_len=best[2], cer_variant=best[3])
    res.update(ref_norm=ref.get(res["cer_variant"] or "surface"), hyp_norm=hyp.get(res["cer_variant"] or "surface"))
    return res


# ============================================================================================ WER (English)
_EN_FILLERS = {"uh", "um", "uhm", "hmm", "hm", "mm", "mhm", "ah", "eh", "er", "huh"}
_EN_EQUIV = {"okay": "ok", "alright": "all right", "gonna": "going to", "wanna": "want to", "gotta": "got to",
             "'cause": "because", "cause": "because", "yeah": "yes", "yep": "yes", "nope": "no", "mr": "mister",
             "mrs": "missus", "dr": "doctor", "komi-san": "komi"}


def normalize_en_words(text: str) -> list[str]:
    """Lower-cased word list for WER: punctuation and bracketed tags removed, apostrophes dropped, a few spelling
    variants unified, hesitation fillers ignored (Whisper writes them inconsistently)."""
    t = unicodedata.normalize("NFKC", text or "").lower().replace("’", "'")
    t = re.sub(r"\[[^\]]{0,40}\]", " ", t)
    for k, v in _EN_EQUIV.items():
        t = re.sub(rf"(?<![a-z']){re.escape(k)}(?![a-z'])", v, t)
    t = re.sub(r"[^a-z0-9' ]+", " ", t).replace("'", "")
    return [w for w in t.split() if w not in _EN_FILLERS]


def wer(ref_text: str, hyp_text: str) -> tuple[float | None, int, int]:
    """-> (wer, word edits, reference words). wer is None for an empty reference."""
    ref, hyp = normalize_en_words(ref_text), normalize_en_words(hyp_text)
    if not ref:
        return None, len(hyp), 0
    prev = list(range(len(hyp) + 1))
    for i, a in enumerate(ref, 1):
        cur = [i]
        for j, b in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a != b)))
        prev = cur
    return prev[-1] / len(ref), prev[-1], len(ref)


# ============================================================================================ delivery (prosody)
def prosody_features(x, sr: int, text: str | None = None) -> dict:
    """Delivery numbers of one clip (language independent):
      f0_mean / f0_med   Hz over voiced frames (pYIN, 65-700 Hz)
      f0_range_st        5th-95th percentile span of F0 in semitones (pitch movement); f0_std_st its standard deviation
      voiced_ratio       voiced frames / speech frames (a whisper is unvoiced: this drops towards 0)
      rms_db             RMS level of the speech frames in dBFS, measured on the signal AS GIVEN (TTS outputs are
                         peak-normalised: use the engine's `raw_rms_dbfs` to compare loudness between takes)
      tilt_db            level above 2 kHz minus level below 2 kHz (dB): vocal effort / breathiness, level independent
      speech_sec, wps    seconds of speech frames; words per second of the whole clip when `text` is given
    """
    import librosa
    import numpy as np
    x = resample(np.asarray(x, dtype="float32"), sr, 16000)
    out = {"f0_mean": None, "f0_med": None, "f0_range_st": None, "f0_std_st": None, "voiced_ratio": 0.0, "rms_db": None,
           "tilt_db": None, "speech_sec": 0.0, "clip_sec": round(len(x) / 16000, 3)}
    if len(x) < 3200:
        return out
    hop, win = 256, 1024
    f0, voiced, _ = librosa.pyin(x, fmin=65.0, fmax=700.0, sr=16000, frame_length=win, hop_length=hop)
    rms = librosa.feature.rms(y=x, frame_length=win, hop_length=hop)[0]
    n = min(len(f0), len(rms))
    f0, voiced, rms = f0[:n], voiced[:n], rms[:n]
    speech = rms > max(1e-4, float(rms.max()) * 10 ** (-30 / 20))
    v = voiced & speech & np.isfinite(f0)
    out["speech_sec"] = round(float(speech.sum()) * hop / 16000, 3)
    if speech.any():
        out["rms_db"] = round(float(20 * np.log10(max(1e-6, float(np.sqrt(np.mean(rms[speech] ** 2)))))), 2)
        out["voiced_ratio"] = round(float(v.sum() / speech.sum()), 3)
        S = np.abs(librosa.stft(x, n_fft=win, hop_length=hop))[:, :n] ** 2
        k = int(2000 / (16000 / win))
        lo, hi = float(S[:k, speech].sum()), float(S[k:, speech].sum())
        out["tilt_db"] = round(float(10 * np.log10(max(hi, 1e-12) / max(lo, 1e-12))), 2)
    if v.sum() >= 5:
        st = 12 * np.log2(f0[v] / 100.0)
        out.update(f0_mean=round(float(f0[v].mean()), 1), f0_med=round(float(np.median(f0[v])), 1),
                   f0_range_st=round(float(np.percentile(st, 95) - np.percentile(st, 5)), 2),
                   f0_std_st=round(float(st.std()), 2))
    if text:
        nw = len(normalize_en_words(text)) or len(text.split())
        out["wps"] = round(nw / max(1e-6, len(x) / 16000), 3)
    return out


def edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(ref: str, hyp: str) -> tuple[float | None, int, int]:
    """-> (cer, edits, ref_len). cer is None for an empty reference."""
    if not ref:
        return None, len(hyp), 0
    e = edit_distance(ref, hyp)
    return e / len(ref), e, len(ref)


# ============================================================================================ evaluation
def _agg(vals: list[float]) -> dict | None:
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return {"n": len(vals), "mean": round(statistics.fmean(vals), 4), "median": round(statistics.median(vals), 4),
            "min": round(min(vals), 4), "max": round(max(vals), 4)}


def aggregate(items: list[dict]) -> dict:
    ok = [i for i in items if i.get("ok")]
    edits = sum(i["cer_edits"] for i in ok if i.get("cer") is not None)
    ref_len = sum(i["cer_ref_len"] for i in ok if i.get("cer") is not None)
    timed = [i for i in ok if i.get("target_dur") is not None]
    fits: dict[str, int] = {}
    for i in ok:
        fits[i.get("fit", "none")] = fits.get(i.get("fit", "none"), 0) + 1
    return {
        "n": len(items), "n_ok": len(ok), "n_failed": len(items) - len(ok),
        "sim_ref": _agg([i.get("sim_ref") for i in ok]),
        "sim_ref_reliable": _agg([i.get("sim_ref") for i in ok if not i.get("sim_unreliable")]),
        "sim_bank": _agg([i.get("sim_bank") for i in ok]),
        "sim_real": _agg([i.get("sim_real") for i in ok]),
        "real_vs_ref": _agg([i.get("real_vs_ref") for i in ok]),
        "cer": _agg([i.get("cer") for i in ok]),
        "cer_micro": round(edits / ref_len, 4) if ref_len else None,
        "cer_surface": _agg([i.get("cer_surface") for i in ok]),
        "cer_reliable": _agg([i.get("cer") for i in ok if not i.get("cer_unreliable")]),
        **({"wer": _agg([i.get("wer") for i in ok]),
            "wer_micro": (round(sum(i["wer_edits"] for i in ok if i.get("wer") is not None)
                                / max(1, sum(i["wer_ref_len"] for i in ok if i.get("wer") is not None)), 4)),
            "wer_reliable": _agg([i.get("wer") for i in ok if not i.get("wer_unreliable")])}
           if any(i.get("wer") is not None for i in ok) else {}),
        "dur_abs_err_sec": _agg([abs(i["dur_err"]) for i in timed if i.get("dur_err") is not None]),
        "dur_rel_err": _agg([abs(i["dur_rel_err"]) for i in timed if i.get("dur_rel_err") is not None]),
        "dur_fit_rate": round(sum(1 for i in timed if i.get("fit_ok")) / len(timed), 4) if timed else None,
        "over_max": sum(1 for i in ok if i.get("over_max")),
        "fit_counts": fits,
        "flagged": sum(1 for i in ok if i.get("flags")),
        "compute_sec_per_audio_sec": (round(sum(i.get("synth_sec") or 0 for i in ok if not i.get("cached"))
                                            / max(1e-9, sum(i.get("dur") or 0 for i in ok if not i.get("cached"))), 3)
                                      if any(not i.get("cached") for i in ok) else None),
    }


def evaluate(rows: list[dict], voices: dict | None = None, device: str = "cpu", threads: int = 8,
             do_asr: bool = True, do_sim: bool = True) -> dict:
    """rows = tts_results.jsonl rows (need id, path, text, ref_wavs, speaker, target_dur/max_dur, dur)."""
    import numpy as np
    t0 = time.time()
    emb = SpeakerEmbedder("cpu" if not device.startswith("cuda") else device) if do_sim else None
    asr = Transcriber(device, threads) if do_asr else None
    ref_cache: dict[str, Any] = {}
    bank_cache: dict[str, Any] = {}
    spk_refs: dict[str, set] = {}
    items = []
    for r in rows:
        it: dict[str, Any] = {k: r.get(k) for k in ("id", "speaker", "text", "path", "dur", "raw_dur", "fit", "ok",
                                                     "error", "target_dur", "max_dur", "emotion", "fit_ok", "cached",
                                                     "synth_sec", "duration_factor", "atempo")}
        for k in ("instruct", "mode", "lang"):            # English engine rows (absent for the Japanese engine)
            if r.get(k) is not None:
                it[k] = r[k]
        it["flags"] = []
        items.append(it)
        if not r.get("ok") or not r.get("path") or not Path(r["path"]).is_file():
            it["ok"] = False
            it["error"] = it.get("error") or "wav missing"
            continue
        try:
            import soundfile as sf
            info = sf.info(r["path"])
            dur = info.frames / info.samplerate
            it["dur"] = round(dur, 3)
            # ---- (c) duration
            if r.get("target_dur") is not None:
                tgt = float(r["target_dur"])
                it["dur_err"] = round(dur - tgt, 3)
                it["dur_rel_err"] = round((dur - tgt) / tgt, 4)
                tol = max(float(T.fit.tolerance_rel) * tgt, float(T.fit.tolerance_abs))
                it["fit_ok"] = abs(dur - tgt) <= tol
                if abs(it["dur_rel_err"]) > float(M.flag_dur_rel_above) and abs(it["dur_err"]) > 0.15:
                    it["flags"].append("dur_miss")
            if r.get("max_dur") is not None and dur > float(r["max_dur"]) + 0.02:
                it["over_max"] = round(dur - float(r["max_dur"]), 3)
                it["fit_ok"] = False
                it["flags"].append("over_max")
            # ---- (a) speaker similarity
            if emb is not None:
                g = emb.embed_file(r["path"])
                refs = [p for p in (r.get("ref_wavs") or []) if Path(p).is_file()]
                if refs:
                    key = "|".join(refs)
                    if key not in ref_cache:
                        ref_cache[key] = mean_embedding([emb.embed_file(p) for p in refs])
                    it["sim_ref"] = round(cos(g, ref_cache[key]), 4)
                    spk_refs.setdefault(str(r.get("speaker")), set()).update(refs)
                spk = str(r.get("speaker"))
                if voices and spk in voices.get("speakers", {}):
                    if spk not in bank_cache:
                        c = voices["speakers"][spk].get("centroid")
                        bank_cache[spk] = np.load(c) if c and Path(c).is_file() else None
                    if bank_cache[spk] is not None:
                        it["sim_bank"] = round(cos(g, bank_cache[spk]), 4)
                if r.get("real_wav") and Path(r["real_wav"]).is_file():     # ground-truth clip of the same line
                    rv = emb.embed_file(r["real_wav"])
                    it["sim_real"] = round(cos(g, rv), 4)
                    if refs:
                        it["real_vs_ref"] = round(cos(rv, ref_cache["|".join(refs)]), 4)
                    it["real_dur"] = round(sf.info(r["real_wav"]).duration, 3)
                it["sim_unreliable"] = dur < 1.0          # embeddings of sub-second clips are noisy
                if it.get("sim_ref") is not None and it["sim_ref"] < float(M.flag_sim_below) and not it["sim_unreliable"]:
                    it["flags"].append("low_sim")
            # ---- (b) intelligibility
            if asr is not None:
                lang = str(r.get("lang") or "JA")
                hyp = asr.transcribe(r["path"], language=lang.lower()[:2])
                cr = cer_best(r.get("text") or "", hyp, lang)
                it.update(asr_text=hyp, **cr, cer_unreliable=cr["cer_ref_len"] < 4)
                if lang.upper() == "JA":
                    if cr["cer"] is not None and cr["cer"] > float(M.flag_cer_above) and cr["cer_edits"] >= 2:
                        it["flags"].append("high_cer")
                else:                                   # languages with word boundaries: WER is the metric
                    w, we, wn = wer(r.get("text_norm") or r.get("text") or "", hyp)
                    it.update(lang=lang.upper(), wer=None if w is None else round(w, 4), wer_edits=we, wer_ref_len=wn,
                              wer_unreliable=wn < 3)
                    if w is not None and w > float(M.get("flag_wer_above", 0.25)) and we >= 2:
                        it["flags"].append("high_wer")
        except Exception as ex:  # noqa: BLE001 - keep evaluating the other lines
            it["metric_error"] = f"{type(ex).__name__}: {ex}"[:300]
    # ---- calibration anchors from the real reference clips (what "same speaker" / "different speaker" look like)
    calib: dict[str, Any] = {}
    if emb is not None and spk_refs:
        embs = {s: {p: emb.embed_file(p) for p in sorted(ps)} for s, ps in spk_refs.items()}
        same = [cos(a, b) for s in embs for i, a in enumerate(embs[s].values())
                for j, b in enumerate(embs[s].values()) if j > i]
        cross = [cos(a, b) for s1 in embs for s2 in embs if s1 < s2 for a in embs[s1].values()
                 for b in embs[s2].values()]
        calib = {"real_same_speaker_ref_vs_ref": _agg(same), "real_cross_speaker_ref_vs_ref": _agg(cross),
                 "note": "cosine between REAL reference clips: upper anchor (same character) and lower anchor "
                         "(different characters) for reading sim_ref"}
    per_speaker = {}
    for s in sorted({str(i.get("speaker")) for i in items}):
        per_speaker[s] = aggregate([i for i in items if str(i.get("speaker")) == s])
    return {"aggregate": aggregate(items), "per_speaker": per_speaker, "calibration": calib, "items": items,
            "config": {"spk_model": str(M.spk_model), "asr": Path(str(M.asr_model_dir)).name, "asr_device": device,
                       "wer_definition": "English rows: word edit distance / reference words after normalize_en_words "
                                         "(case, punctuation, fillers, a few spelling variants)",
                       "cer_definition": "edit distance / reference length, punctuation removed, lowest of written "
                                         "form / MeCab pronunciation / pykakasi reading; CER (not WER) because "
                                         "Japanese has no word boundaries",
                       "thresholds": {"sim_below": float(M.flag_sim_below), "cer_above": float(M.flag_cer_above),
                                      "dur_rel_above": float(M.flag_dur_rel_above)}},
            "eval_sec": round(time.time() - t0, 1)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Speaker similarity / CER / duration-fit metrics for a TTS batch")
    ap.add_argument("--results", required=True, help="tts_results.jsonl")
    ap.add_argument("--voices", default=None, help="voices.json from src.train.voice_bank (adds sim_bank)")
    ap.add_argument("--out", default=None, help="metrics JSON (default: voice_metrics.json next to the results)")
    ap.add_argument("--device", default="cpu", help="cpu | cuda (ASR; embeddings follow)")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--no-asr", action="store_true")
    ap.add_argument("--no-sim", action="store_true")
    a = ap.parse_args()
    if a.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("OMP_NUM_THREADS", str(a.threads))
    try:
        import torch
        torch.set_num_threads(a.threads)
    except Exception:
        pass
    rows = read_jsonl(a.results)
    voices = read_json(a.voices) if a.voices else None
    res = evaluate(rows, voices, device=a.device, threads=a.threads, do_asr=not a.no_asr, do_sim=not a.no_sim)
    out = Path(a.out) if a.out else Path(a.results).with_name("voice_metrics.json")
    write_json(out, res)
    log(f"metrics -> {out}")
    import json
    print(json.dumps({"aggregate": res["aggregate"], "calibration": res["calibration"]}, ensure_ascii=False, indent=1))
    for i in res["items"]:
        print(f"  {str(i['id'])[:28]:28s} ok={i['ok']!s:5} fit={i.get('fit')!s:7} dur={i.get('dur')} "
              f"tgt={i.get('target_dur')} sim={i.get('sim_ref')} cer={i.get('cer')}"
              + (f" wer={i.get('wer')}" if i.get("wer") is not None else "") + f" flags={i.get('flags')} "
              f"asr={i.get('asr_text')!r}")


if __name__ == "__main__":
    main()
