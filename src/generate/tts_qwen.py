"""English character TTS: Qwen3-TTS 1.7B voice cloning (+ delivery instructions), same contract as src/generate/tts.py.

Input  JSONL, one utterance per line:
  {"id": "c140_s023_l0", "speaker": "Tadano", "text": "Good morning, Komi.", "lang": "EN",
   "ref_wavs": ["/abs/ref_00.wav", ...], "ref_texts": ["transcript of ref_00", ...],
   "instruct": "nervous, speaking quickly" | null, "target_dur": 2.8 | null, "max_dur": 3.5 | null, "seed": 7 | null}
  optional: "mode": icl | xvec | cv_xvec | cv_icl | custom (default: tts_qwen.default_mode / instruct_mode),
            "preset": "ryan" (mode custom with a preset speaker of the CustomVoice model), "instruct_on_base": true
            (pass the instruction to the Base model in icl / xvec mode), "strip_tags": false, "real_wav", "meta".
Output <out_dir>/<id>.wav (mono, 44.1 kHz, 16-bit, trimmed, peak-normalised) and <out_dir>/tts_results.jsonl:
  {"id","path","dur","raw_dur","fit":"native|stretch|none","ok":bool,"error":str|null, ...extras}

Run (own venv; GPU jobs always through gpu_run.sh):
  . scripts/env.sh
  scripts/gpu_run.sh 8 $KOMI_WORK/venvs/qwentts/bin/python -m src.generate.tts_qwen --in lines.jsonl --out-dir runs/x/tts
From pipeline code in another venv:
  from src.generate.tts_qwen import synthesize, voice_refs_en
  refs, texts, who = voice_refs_en("Tadano", instruct="excited")
  rows = synthesize([{"id": "l0", "speaker": who, "text": "...", "lang": "EN", "ref_wavs": refs, "ref_texts": texts,
                      "instruct": "excited"}], out_dir)

Everything is local. A model is loaded once per batch and only if a line that needs it is not in the cache.
Measured behaviour of every mode (does `instruct` change the delivery, what it costs in similarity): docs/tts_qwen.md.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import unicodedata
import zlib
from pathlib import Path
from typing import Any, Iterable

from omegaconf import OmegaConf

from src.common.config import CFG
from src.common.io import log, read_json, read_jsonl, sha, write_json, write_jsonl
from src.generate.tts import _file_sha, _safe_id, fade, peak_norm, read_mono, resample, rms_dbfs, time_stretch, trim_silence, write_wav

Q = CFG.tts_qwen
MODEL_SR = 24000
RESULTS_NAME = "tts_results.jsonl"
MODES = ("icl", "xvec", "cv_xvec", "cv_icl", "custom")
_TAG = re.compile(r"\[[A-Za-z][A-Za-z ,'-]{0,40}\]|\([A-Za-z][A-Za-z ,'-]{0,40}\)")
_WORD = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z]+)?")


# ============================================================================================ text front-end
def normalize_en(text: str, strip_tags: bool | None = None) -> str:
    """Script text -> text the model reads. Returns "" when nothing speakable is left (e.g. "...", "!?")."""
    t = unicodedata.normalize("NFKC", str(text or ""))
    t = t.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    if bool(Q.strip_tags) if strip_tags is None else strip_tags:
        t = _TAG.sub(" ", t)
    t = re.sub(r"[♪♫☆★♡♥❤※*#＿_|\\/<>=+^`~]", "", t)
    t = re.sub(r"…+|\.{3,}", "...", t)
    t = re.sub(r"\s*[–—―]+\s*|\s+-{2,}\s*|-{2,}", " — ", t)
    t = re.sub(r"!{2,}", "!", t)
    t = re.sub(r"\?{2,}", "?", t)
    t = re.sub(r"(?:\?!|!\?)[!?]*", "?!", t)
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"\s+([,.!?;:])", r"\1", t)
    if not re.search(r"[A-Za-z0-9]", _TAG.sub(" ", t)):
        return ""
    return t


def count_words(text_norm: str) -> int:
    return max(1, len(_WORD.findall(_TAG.sub(" ", text_norm))))


def duration_bounds(n_words: int, text_norm: str) -> tuple[float, float]:
    s = Q.sanity
    pauses = text_norm.count("...") + text_norm.count("—")
    return n_words * float(s.min_sec_per_word), (n_words * float(s.max_sec_per_word) + float(s.max_extra_sec)
                                                 + float(s.pause_sec) * pauses)


# ============================================================================================ references
def voices_en() -> dict:
    return read_json(Path(Q.voice_bank.out_dir) / "voices.json", default={}) or {}


def ref_style(instruct: str | None) -> str | None:
    """"high" / "low" when the instruction names a high- / low-arousal delivery (tts_qwen.ref_style), else None."""
    if not instruct:
        return None
    words = set(re.findall(r"[a-z]+", str(instruct).lower()))
    for style in ("high", "low"):
        if words & {str(w) for w in Q.ref_style[style]}:
            return style
    return None


_MAIN = ("Komi", "Tadano", "Najimi")
_NEVER = ("Narrator", "Classmates", "Other")          # never borrowed: the narrator's voice and many-voice group entries
_CARDS: dict | None = None


def _cards() -> dict:
    """data/processed/cards/cards.json as {id: card} (gender, age, aliases); {} when the file is missing."""
    global _CARDS
    if _CARDS is None:
        rows = read_json(Path(CFG.paths.processed) / "cards" / "cards.json", default=[]) or []
        _CARDS = {str(c.get("id")): c for c in rows if isinstance(c, dict) and c.get("id")}
    return _CARDS


def card_gender(name: str) -> str | None:
    """"Male" / "Female" from the character card, None when there is no card or the card says neither."""
    g = str((_cards().get(name) or {}).get("gender") or "")
    return g if g in ("Male", "Female") else None


def card_age(name: str) -> int | None:
    """First number of the card's age field ("6", "10-12", "14 (first appearance)") or None."""
    m = re.search(r"\d+", str((_cards().get(name) or {}).get("age") or ""))
    return int(m.group(0)) if m else None


def canonical_speaker(speaker: str) -> str:
    """Script name -> bank / card name ("Komi-san" -> "Komi", "Tadano Hitohito" -> "Tadano")."""
    if speaker in voices_en().get("speakers", {}) or speaker in _cards():
        return speaker
    for c in CFG.characters.roster:
        if speaker == c["name"] or speaker in c.get("aliases", []) or speaker == c.get("full"):
            return c["name"]
    for cid, c in _cards().items():
        if speaker == c.get("full_name") or speaker in (c.get("aliases") or []):
            return cid
    return speaker


def bank_voice_gender(name: str) -> str | None:
    """How a BANK voice sounds: by the median pitch of its reference clips (below 150 Hz male, above 190 Hz female);
    in between the character card decides (teenage boys sit there), then 165 Hz. A voice named after a character can
    still be another gender than the card says, which is why the pitch comes first."""
    v = voices_en().get("speakers", {}).get(name)
    if not v:
        return None
    f0 = v.get("voice_f0")                       # median F0 of the reference clips, written by src.train.voice_bank
    if f0 is None:
        return card_gender(name)
    if f0 < 150:
        return "Male"
    if f0 > 190:
        return "Female"
    return card_gender(name) or ("Male" if f0 < 165 else "Female")


def fallback_voice(speaker: str) -> str | None:
    """Bank voice borrowed by a speaker without one: same gender as the character card says, a minor character's
    voice when there is one (a main character's only as a last resort), never the narrator's. Children of 10 or
    younger get one of the highest-pitched minor voices. Deterministic (same speaker -> same voice)."""
    idx = voices_en().get("speakers", {})
    pool = [n for n in sorted(idx) if n not in _NEVER and idx[n].get("default_refs")]
    if not pool:
        return None
    minor = [n for n in pool if n not in _MAIN]
    g = card_gender(speaker)
    age = card_age(speaker)
    if age is not None and age <= 10:
        # small children: a high voice matters more than the label. Same-gender minor voices above 220 Hz if any (the
        # bank's male voices are adult-sounding, so a little boy gets a high minor voice of either gender instead)
        f0 = lambda n: float(idx[n].get("voice_f0") or 0.0)   # noqa: E731
        high = sorted((n for n in minor if f0(n) >= 220 and (g is None or bank_voice_gender(n) == g)), key=f0, reverse=True)[:3] \
            or sorted((n for n in minor if f0(n) >= 220), key=f0, reverse=True)[:3]
        if high:
            return high[sum(map(ord, speaker)) % len(high)]
    for cand in ([n for n in minor if bank_voice_gender(n) == g] if g else minor,
                 [n for n in pool if bank_voice_gender(n) == g] if g else [], minor, pool):
        if cand:
            return cand[sum(map(ord, speaker)) % len(cand)]
    return None


def voice_refs_en(speaker: str, instruct: str | None = None, style: str | None = None) -> tuple[list[str], list[str], str]:
    """(ref_wavs, ref_texts, resolved speaker) from the English voice bank. With an instruction (or style "high" /
    "low") the emotion-matched clips of that speaker come first when the bank has them. A speaker without a bank
    borrows a voice of the same gender (data/processed/cards/cards.json), see `fallback_voice`."""
    idx = voices_en().get("speakers", {})
    who = canonical_speaker(speaker)
    if who not in idx or not idx[who].get("default_refs"):
        who = fallback_voice(who)
        if who is None:
            return [], [], speaker
    v = idx[who]
    style = style or ref_style(instruct)
    sr = (v.get("style_refs") or {}).get(style or "", {})
    if sr.get("refs"):
        return list(sr["refs"]), list(sr["ref_texts"]), who
    return list(v.get("default_refs", [])), list(v.get("default_ref_texts", [])), who


# ============================================================================================ engine
def _seed_all(seed: int) -> None:
    import random

    import numpy as np
    import torch
    seed = int(seed) % (2**31 - 1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def model_key_for(mode: str, speaker: str | None, preset: str | None) -> str:
    """Which checkpoint a line needs: "base" | "custom_voice" | "ft:<speaker>"."""
    if mode in ("icl", "xvec"):
        return "base"
    if mode in ("cv_xvec", "cv_icl"):
        return "custom_voice"
    if preset:
        return "custom_voice"
    if speaker and speaker in (Q.get("finetuned") or {}):
        return f"ft:{speaker}"
    raise ValueError(f"mode custom needs a fine-tuned checkpoint for {speaker!r} (tts_qwen.finetuned) or a preset speaker")


def model_path(key: str) -> Path:
    return Path(Q.finetuned[key[3:]]) if key.startswith("ft:") else Path(Q.models[key])


def model_sig(key: str) -> str:
    p = model_path(key) / "model.safetensors"
    st = p.stat()
    return f"{key}:{st.st_size}:{int(st.st_mtime)}"


class XVector:
    """Speaker embeddings from the Base model's speaker encoder WITHOUT loading the 1.7B talker: only the
    `speaker_encoder.*` tensors are read from the checkpoint. Cached per reference file (content hash)."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.enc = None
        self.dir = Path(Q.cache_dir) / "xvec"

    def _load(self):
        import torch
        from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig
        from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSSpeakerEncoder
        from safetensors import safe_open

        base = model_path("base")
        cfg = Qwen3TTSConfig.from_pretrained(str(base))
        enc = Qwen3TTSSpeakerEncoder(cfg.speaker_encoder_config)
        sd = {}
        with safe_open(str(base / "model.safetensors"), framework="pt") as f:
            for k in f.keys():
                if k.startswith("speaker_encoder."):
                    sd[k[len("speaker_encoder."):]] = f.get_tensor(k).float()
        enc.load_state_dict(sd, strict=True)
        self.sr = int(cfg.speaker_encoder_config.sample_rate)
        self.enc = enc.eval().to(self.device)
        self._torch = torch

    def file(self, path: str):
        """[D] float32 numpy embedding of one reference clip."""
        import numpy as np
        key = sha({"f": _file_sha(path), "m": model_sig("base"), "v": 1})
        p = self.dir / f"{key}.npy"
        if p.exists():
            return np.load(p)
        if self.enc is None:
            self._load()
        from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram
        x, sr = read_mono(path)
        x = resample(x, sr, self.sr)
        with self._torch.no_grad():
            mels = mel_spectrogram(self._torch.from_numpy(x).unsqueeze(0), n_fft=1024, num_mels=128, sampling_rate=self.sr,
                                   hop_size=256, win_size=1024, fmin=0, fmax=12000).transpose(1, 2)
            e = self.enc(mels.to(self.device))[0].float().cpu().numpy()
        self.dir.mkdir(parents=True, exist_ok=True)
        np.save(p, e)
        return e

    def mean(self, paths: list[str]):
        import numpy as np
        return np.mean(np.stack([self.file(p) for p in paths[: int(Q.ref.xvec_max_refs)]]), axis=0).astype("float32")


class QwenEngine:
    """One Qwen3-TTS checkpoint on the device. `generate` takes a list of jobs (a batch) and returns 24 kHz waves."""

    def __init__(self, key: str, device: str | None = None):
        import torch
        from qwen_tts import Qwen3TTSModel

        self.key = key
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        t0 = time.time()
        kw: dict[str, Any] = {"device_map": self.device,
                              "dtype": getattr(torch, str(Q.dtype)) if self.device != "cpu" else torch.float32}
        if Q.get("attn_implementation"):
            kw["attn_implementation"] = str(Q.attn_implementation)
        self.tts = Qwen3TTSModel.from_pretrained(str(model_path(key)), **kw)
        self.model = self.tts.model
        self.type = str(self.model.tts_model_type)
        self.speakers = [str(s) for s in (self.model.get_supported_speakers() or [])]
        self.load_sec = round(time.time() - t0, 2)
        self._torch = torch
        self._codes: dict[str, Any] = {}
        if self.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        log(f"Qwen3-TTS {key} ({self.type}) ready on {self.device} in {self.load_sec}s")

    def ref_codes(self, path: str):
        k = _file_sha(path)
        if k not in self._codes:
            x, sr = read_mono(path)
            n = int(float(Q.ref.icl_max_sec) * sr)
            self._codes[k] = self.model.speech_tokenizer.encode(x[:n], sr=sr).audio_codes[0]
        return self._codes[k]

    def generate(self, jobs: list[dict], seed: int) -> list[dict]:
        """jobs: [{"text","language","mode","instruct","xvec"(np|None),"ref_wav","ref_text","speaker_id","max_tokens"}]
        -> [{"wav": float32 24 kHz, "tokens": n, "hit_cap": bool}] (untrimmed)."""
        import numpy as np
        torch, t = self._torch, self.tts
        input_ids = t._tokenize_texts([t._build_assistant_text(j["text"]) for j in jobs])
        instruct_ids = [t._tokenize_texts([t._build_instruct_text(j["instruct"])])[0] if j.get("instruct") else None
                        for j in jobs]
        modes = {j["mode"] for j in jobs}
        if len(modes) != 1:
            raise ValueError(f"one mode per batch, got {modes}")
        mode = modes.pop()
        kw: dict[str, Any] = {}
        ref_codes = [None] * len(jobs)
        if mode in ("icl", "xvec", "cv_xvec", "cv_icl"):
            icl = mode in ("icl", "cv_icl")
            if icl:
                ref_codes = [self.ref_codes(j["ref_wav"]) for j in jobs]
                kw["ref_ids"] = [t._tokenize_texts([t._build_ref_text(j["ref_text"])])[0] for j in jobs]
            kw["voice_clone_prompt"] = {
                "ref_code": ref_codes if icl else None,
                # cv_xvec_scale: the Base speaker encoder gives vectors of norm ~18, the preset speakers of the CustomVoice
                # model have norm ~15 (measured 2026-11-12); 1.0 = as extracted
                "ref_spk_embedding": [torch.from_numpy(np.asarray(j["xvec"]) * (float(Q.get("cv_xvec_scale") or 1.0) if mode.startswith("cv_") else 1.0))
                                      for j in jobs],
                "x_vector_only_mode": [not icl] * len(jobs), "icl_mode": [icl] * len(jobs)}
            non_streaming = mode.startswith("cv_")          # the wrapper's defaults: clone False, custom voice True
        else:
            kw["speakers"] = [j["speaker_id"] for j in jobs]
            non_streaming = True
        gen = t._merge_generate_kwargs(**OmegaConf.to_container(Q.gen), max_new_tokens=int(max(j["max_tokens"] for j in jobs)))
        _seed_all(seed)
        with torch.no_grad():
            codes_list, _ = self.model.generate(input_ids=input_ids, instruct_ids=instruct_ids,
                                                languages=[j["language"] for j in jobs], non_streaming_mode=non_streaming,
                                                **kw, **gen)
            dec_in = [torch.cat([rc.to(c.device), c], dim=0) if rc is not None else c for rc, c in zip(ref_codes, codes_list)]
            wavs, fs = self.model.speech_tokenizer.decode([{"audio_codes": c} for c in dec_in])
        out = []
        for j, rc, c, full, w in zip(jobs, ref_codes, codes_list, dec_in, wavs):
            w = np.asarray(w, dtype=np.float32).reshape(-1)
            if rc is not None:                               # cut the decoded reference part (as the wrapper does)
                w = w[int(int(rc.shape[0]) / max(int(full.shape[0]), 1) * w.shape[0]):]
            n = int(c.shape[0])
            out.append({"wav": resample(w, int(fs), MODEL_SR), "tokens": n, "hit_cap": n >= int(j["max_tokens"])})
        return out

    def vram_peak_gb(self) -> dict:
        t = self._torch
        if not self.device.startswith("cuda"):
            return {"allocated": 0.0, "reserved": 0.0}
        out = {"allocated": round(t.cuda.max_memory_allocated() / 2**30, 2),
               "reserved": round(t.cuda.max_memory_reserved() / 2**30, 2), "process_mib": None}
        try:
            r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=15)
            for line in r.stdout.splitlines():
                pid, _, mem = line.partition(",")
                if pid.strip() == str(os.getpid()):
                    out["process_mib"] = int(mem.strip())
        except Exception:
            pass
        return out

    def close(self) -> None:
        import gc
        self.tts = self.model = None
        self._codes.clear()
        gc.collect()
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()


# ============================================================================================ one utterance
def _tolerance(target: float) -> float:
    return max(float(Q.fit.tolerance_rel) * target, float(Q.fit.tolerance_abs))


def judge_take(y, n_words: int, text_norm: str, hit_cap: bool) -> tuple[str | None, float]:
    """Sanity check of a trimmed take -> (problem or None, badness)."""
    lo, hi = duration_bounds(n_words, text_norm)
    d = len(y) / MODEL_SR
    if d <= 0.03 or rms_dbfs(y) < float(Q.sanity.min_rms_dbfs):
        return "silent", 1e9
    if hit_cap:
        return "runaway (hit the token cap)", abs(d - hi) + 2.0
    if d > hi:
        return f"too_long ({d:.2f}s for {n_words} words, limit {hi:.2f}s)", d - hi + 1.0
    if d < lo:
        return f"too_short ({d:.2f}s for {n_words} words, limit {lo:.2f}s)", lo - d + 1.0
    return None, 0.0


def finish(y, item: dict, n_words: int, problem: str | None) -> tuple[Any, dict]:
    """Trimmed 24 kHz take -> fitted 44.1 kHz output + the duration part of the result row."""
    f = Q.fit
    sr_out = int(Q.sample_rate)
    raw_dur = len(y) / MODEL_SR
    import numpy as np
    row: dict[str, Any] = {"raw_dur": round(raw_dur, 3), "n_words": n_words, "sec_per_word": round(raw_dur / n_words, 4),
                           "atempo": 1.0, "warnings": [],
                           # level of the take BEFORE peak normalisation (loudness as the model produced it)
                           "raw_rms_dbfs": round(rms_dbfs(y), 2),
                           "raw_peak_dbfs": round(float(20 * np.log10(max(1e-6, float(np.max(np.abs(y)))))), 2)}
    target, max_dur = item.get("target_dur"), item.get("max_dur")
    want = None
    if target is not None:
        want = min(float(target), float(max_dur) * 0.98) if max_dur is not None else float(target)
    elif max_dur is not None and raw_dur > float(max_dur):
        want = float(max_dur) * 0.97
    fit = "none" if (target is None and max_dur is None) else "native"
    stretch = 1.0
    over = max_dur is not None and raw_dur > float(max_dur)
    if want is not None and (over or abs(raw_dur - want) > 0.6 * _tolerance(want)):
        ratio = want / raw_dur
        if ratio > 1 and (not bool(f.slow_down) or ratio > float(f.stretch_max)):
            row["warnings"].append(f"line is {raw_dur:.2f}s for a {want:.2f}s slot: left at natural speed (pad with silence)")
        else:
            stretch = min(float(f.stretch_max), max(float(f.stretch_min), ratio))
            if abs(stretch - 1.0) > 0.01:
                y = time_stretch(y, MODEL_SR, stretch)
                fit = "stretch"
            else:
                stretch = 1.0
    out = peak_norm(fade(resample(y, MODEL_SR, sr_out), sr_out), float(Q.peak_dbfs))
    dur = len(out) / sr_out
    fit_ok = True
    if target is not None:
        fit_ok = abs(dur - float(target)) <= _tolerance(float(target))
        row["dur_err"] = round(dur - float(target), 3)
    if max_dur is not None and dur > float(max_dur) + 0.02:
        fit_ok = False
        row["over_max"] = round(dur - float(max_dur), 3)
    if not fit_ok:
        row["min_feasible_dur"] = round(raw_dur * float(f.stretch_min), 3)
        row["max_feasible_dur"] = round(raw_dur * float(f.stretch_max), 3)
    row.update(dur=round(dur, 3), fit=fit, fit_ok=fit_ok, atempo=round(1.0 / stretch, 4),
               ok=problem is None, error=None if problem is None else f"sanity_failed: {problem}")
    return out, row


# ============================================================================================ batch
def _cache_key(p: dict) -> str:
    return sha({"v": 2, "engine": Q.engine, "pkg": Q.package, "model": model_sig(p["model_key"]),
                "xvec_model": model_sig("base") if p["mode"] != "custom" else None, "mode": p["mode"],
                "text": p["text_norm"], "lang": p["language"], "instruct": p["instruct"], "preset": p.get("preset"),
                "refs": [_file_sha(r) for r in p["refs"]], "ref_text": p["ref_text"] if p["mode"] in ("icl", "cv_icl") else None,
                "seed": p["seed"], "target": p["item"].get("target_dur"), "max": p["item"].get("max_dur"),
                "dtype": str(Q.dtype), "gen": OmegaConf.to_container(Q.gen), "fit": OmegaConf.to_container(Q.fit),
                "trim": OmegaConf.to_container(Q.trim), "sanity": OmegaConf.to_container(Q.sanity),
                "ref_cfg": OmegaConf.to_container(Q.ref), "sr": int(Q.sample_rate), "peak": float(Q.peak_dbfs)}, 24)


def plan_item(item: dict, n: int) -> dict:
    """Validate one input line and resolve mode / model / references (no model needed). Raises on bad input."""
    if not isinstance(item, dict):
        raise ValueError(f"line {n}: not a JSON object ({str(item)[:80]})")
    if item.get("_parse_error"):
        raise ValueError(f"line {n}: invalid JSON ({item['_parse_error']})")
    lang = str(item.get("lang") or "EN").upper()
    if lang not in ("EN", "ENGLISH"):
        raise ValueError(f"tts_qwen is the English engine, got lang={lang!r} (Japanese lines go to src.generate.tts)")
    for k in ("target_dur", "max_dur"):
        v = item.get(k)
        if v is not None:
            v = float(v)
            if not (v > 0 and math.isfinite(v)):
                raise ValueError(f"{k} must be a positive number, got {item.get(k)!r}")
            item[k] = v
    instruct = str(item["instruct"]).strip() if item.get("instruct") else None
    mode = str(item.get("mode") or (Q.instruct_mode if instruct else Q.default_mode))
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r} (one of {MODES})")
    text_norm = normalize_en(item.get("text"), item.get("strip_tags"))
    if not text_norm:
        raise ValueError("no_speakable_text (silent line: nothing to synthesise)")
    preset = str(item["preset"]).lower() if item.get("preset") else None
    mkey = model_key_for(mode, item.get("speaker"), preset)
    refs = [str(p) for p in (item.get("ref_wavs") or [])]
    ref_texts = [str(t) for t in (item.get("ref_texts") or [])]
    warnings = []
    ref_text = None
    if mode != "custom":
        if not refs:
            raise ValueError("no ref_wavs given")
        missing = [p for p in refs if not Path(p).is_file()]
        if missing:
            raise FileNotFoundError(f"reference wav missing: {missing[0]}")
        if mode in ("icl", "cv_icl"):
            ref_text = (ref_texts[0] if ref_texts else "").strip()
            if not ref_text:
                raise ValueError(f"mode {mode} needs ref_texts[0] (transcript of ref_wavs[0])")
        import soundfile as sf
        d0 = sf.info(refs[0]).duration
        if d0 < float(Q.ref.min_sec):
            warnings.append(f"short reference ({d0:.1f}s): clone quality will be poor")
    if instruct and mode in ("icl", "xvec") and not (bool(Q.instruct_on_base) or item.get("instruct_on_base")):
        warnings.append("instruct not passed to the Base model (tts_qwen.instruct_on_base=false): delivery follows the reference")
        eff_instruct = None
    else:
        eff_instruct = instruct
    seed = int(item["seed"]) if item.get("seed") is not None else \
        (int(Q.seed) + zlib.crc32(f"{item.get('speaker')}|{text_norm}|{instruct}".encode())) % (2**31 - 1)
    n_words = count_words(text_norm)
    _, hi = duration_bounds(n_words, text_norm)
    # fine-tuned checkpoints were trained with the language-neutral prefix (upstream recipe) -> same prefix at inference
    language = str(Q.finetuned_language) if mkey.startswith("ft:") else str(Q.language)
    return {"n": n, "item": item, "mode": mode, "model_key": mkey, "preset": preset, "text_norm": text_norm,
            "language": language, "instruct": eff_instruct, "refs": refs, "ref_text": ref_text, "seed": seed,
            "n_words": n_words, "max_tokens": int(max(24, math.ceil(float(Q.codec_hz) * (hi + 1.0)))), "warn0": warnings}


def synthesize_batch(items: Iterable[dict], out_dir: str | Path, device: str | None = None, use_cache: bool = True,
                     fresh: bool = False, batch_size: int | None = None) -> list[dict]:
    """In-process batch synthesis (needs venvs/qwentts). Never raises for a bad line: the row gets ok=false + error."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(Q.cache_dir)
    bs = int(batch_size or Q.batch_size)
    t_start = time.time()
    rows: list[dict] = []
    plans: list[dict] = []
    for n, item in enumerate(items):
        row: dict[str, Any] = {"id": f"line{n:05d}", "path": None, "dur": None, "raw_dur": None, "fit": "none",
                               "ok": False, "error": None}
        rows.append(row)
        try:
            if isinstance(item, dict):
                row["id"] = item.get("id", row["id"])
                for k in ("speaker", "text", "lang", "target_dur", "max_dur", "ref_wavs", "ref_texts", "instruct"):
                    row[k] = item.get(k)
                for k in ("real_wav", "meta", "preset"):
                    if k in item:
                        row[k] = item[k]
            p = plan_item(item, n)
            row.update(lang="EN", mode=p["mode"], model=p["model_key"], text_norm=p["text_norm"], seed=p["seed"],
                       instruct_applied=p["instruct"])
            p["key"] = _cache_key(p)
            row["key"] = p["key"]
            p["wav_path"] = out_dir / f"{_safe_id(row['id'])}.wav"
            p["c_wav"], p["c_json"] = cache_dir / p["key"][:2] / f"{p['key']}.wav", cache_dir / p["key"][:2] / f"{p['key']}.json"
            if use_cache and p["c_wav"].exists() and p["c_json"].exists():
                shutil.copyfile(p["c_wav"], p["wav_path"])
                row.update(json.loads(p["c_json"].read_text()))
                row.update(path=str(p["wav_path"]), cached=True)
                continue
            plans.append(p)
        except Exception as e:  # noqa: BLE001 - one bad line must never stop the batch
            row.update(ok=False, error=f"{type(e).__name__}: {e}"[:500])

    load_sec, vram, engines_used = 0.0, {}, []
    if plans:
        xv = XVector("cpu" if device == "cpu" else ("cuda:0" if _cuda() else "cpu"))
        for p in plans:
            if p["mode"] != "custom":
                try:
                    p["xvec"] = xv.mean(p["refs"])
                except Exception as e:  # noqa: BLE001
                    rows[p["n"]].update(ok=False, error=f"xvector_failed: {type(e).__name__}: {e}"[:500])
                    p["dead"] = True
        xv.enc = None
        plans = [p for p in plans if not p.get("dead")]
        for mkey in sorted({p["model_key"] for p in plans}):
            group = [p for p in plans if p["model_key"] == mkey]
            try:
                eng = QwenEngine(mkey, device)
                load_sec += eng.load_sec
                engines_used.append(mkey)
            except Exception as e:  # noqa: BLE001 - model could not be loaded: its lines fail, the rest goes on
                for p in group:
                    rows[p["n"]].update(ok=False, error=f"model_load_failed: {type(e).__name__}: {e}"[:500])
                log(f"Qwen3-TTS load failed ({mkey}): {type(e).__name__}: {e}")
                continue
            for p in group:
                if p["mode"] == "custom":
                    sid = p["preset"] or str(p["item"].get("speaker"))
                    match = [s for s in eng.speakers if s.lower() == sid.lower()]
                    if not match:
                        rows[p["n"]].update(ok=False, error=f"ValueError: speaker {sid!r} not in checkpoint {mkey} ({eng.speakers})")
                        p["dead"] = True
                    else:
                        p["speaker_id"] = match[0]
            group = [p for p in group if not p.get("dead")]
            # one mode per batch; similar lengths together
            group.sort(key=lambda p: (p["mode"], p["max_tokens"], p["n"]))
            batches, cur = [], []
            for p in group:
                if cur and (len(cur) >= bs or cur[0]["mode"] != p["mode"]):
                    batches.append(cur)
                    cur = []
                cur.append(p)
            if cur:
                batches.append(cur)
            done = 0
            for batch in batches:
                pending = list(batch)
                best: dict[int, tuple] = {}
                stats = {p["n"]: {"synth_sec": 0.0, "takes": 0, "warn": []} for p in batch}
                for k in range(int(Q.sanity.retries) + 1):
                    if not pending:
                        break
                    jobs = [{"text": p["text_norm"], "language": p["language"], "mode": p["mode"], "instruct": p["instruct"],
                             "xvec": p.get("xvec"), "ref_wav": p["refs"][0] if p["refs"] else None, "ref_text": p["ref_text"],
                             "speaker_id": p.get("speaker_id"), "max_tokens": p["max_tokens"]} for p in pending]
                    t0 = time.perf_counter()
                    try:
                        outs = eng.generate(jobs, seed=pending[0]["seed"] + k)
                    except Exception as e:  # noqa: BLE001
                        for p in pending:
                            rows[p["n"]].update(ok=False, error=f"{type(e).__name__}: {e}"[:500])
                        if _cuda():
                            import torch
                            torch.cuda.empty_cache()
                        pending = []
                        break
                    dt = (time.perf_counter() - t0) / len(pending)
                    nxt = []
                    for p, o in zip(pending, outs):
                        st = stats[p["n"]]
                        st["synth_sec"] += dt
                        st["takes"] += 1
                        y = trim_silence(o["wav"], MODEL_SR, float(Q.trim.top_db), float(Q.trim.pad_head_ms), float(Q.trim.pad_tail_ms))
                        problem, bad = judge_take(y, p["n_words"], p["text_norm"], o["hit_cap"])
                        if p["n"] not in best or bad < best[p["n"]][0]:
                            best[p["n"]] = (bad, y, pending[0]["seed"] + k, problem, o["tokens"])
                        if problem is not None:
                            st["warn"].append(f"take seed+{k}: {problem}")
                            nxt.append(p)
                    pending = nxt
                for p in batch:
                    row = rows[p["n"]]
                    if p["n"] not in best:
                        continue
                    _, y, seed_used, problem, tokens = best[p["n"]]
                    st = stats[p["n"]]
                    try:
                        if problem == "silent":
                            row.update(ok=False, error="sanity_failed: silent output", takes=st["takes"],
                                       synth_sec=round(st["synth_sec"], 3), warnings=p["warn0"] + st["warn"])
                            continue
                        wav, res = finish(y, p["item"], p["n_words"], problem)
                        res.update(seed=seed_used, tokens=tokens, takes=st["takes"], synth_sec=round(st["synth_sec"], 3),
                                   warnings=p["warn0"] + st["warn"] + res["warnings"], device=eng.device, batch=len(batch))
                        res["rtf"] = round(res["synth_sec"] / res["dur"], 3) if res.get("dur") else None
                        write_wav(p["wav_path"], wav, int(Q.sample_rate))
                        row.update(res)
                        row.update(path=str(p["wav_path"]), cached=False)
                        if res["ok"] and use_cache:
                            p["c_wav"].parent.mkdir(parents=True, exist_ok=True)
                            shutil.copyfile(p["wav_path"], p["c_wav"])
                            write_json(p["c_json"], {k: v for k, v in res.items() if k != "path"})
                    except Exception as e:  # noqa: BLE001
                        row.update(ok=False, error=f"{type(e).__name__}: {e}"[:500])
                    done += 1
                    log(f"[{mkey} {done}/{len(group)}] {'ok ' if row['ok'] else 'ERR'} {row['id']} mode={p['mode']} raw={row.get('raw_dur')}s "
                        f"dur={row.get('dur')}s takes={row.get('takes')} synth={row.get('synth_sec')}s"
                        + ("" if row["ok"] else f" :: {row['error']}"))
                _write_results(out_dir, rows, fresh)
            vram[mkey] = eng.vram_peak_gb()
            eng.close()

    _write_results(out_dir, rows, fresh)
    done_rows = [r for r in rows if r.get("ok")]
    new = [r for r in done_rows if not r.get("cached")]
    audio_sec = sum(r["dur"] for r in new)
    synth_sec = sum(r.get("synth_sec", 0.0) for r in new)
    summary = {"n": len(rows), "ok": len(done_rows), "failed": len(rows) - len(done_rows),
               "cached": sum(1 for r in done_rows if r.get("cached")), "synthesised": len(new),
               "audio_sec": round(audio_sec, 2), "synth_sec": round(synth_sec, 2),
               "compute_sec_per_audio_sec": round(synth_sec / audio_sec, 3) if audio_sec else None,
               "sec_per_line": round(synth_sec / len(new), 3) if new else None,
               "takes": sum(r.get("takes", 0) for r in new), "model_load_sec": round(load_sec, 2),
               "wall_sec": round(time.time() - t_start, 2), "models": engines_used, "vram_peak_gb": vram,
               "batch_size": bs, "dtype": str(Q.dtype), "engine": str(Q.engine), "package": str(Q.package)}
    write_json(out_dir / "tts_run.json", summary)
    log(f"TTS batch: {summary}")
    return rows


def needs_model(items: Iterable[dict]) -> bool:
    """False when no line of the batch needs synthesis: each one is either in the cache or fails validation."""
    cache_dir = Path(Q.cache_dir)
    for n, item in enumerate(items):
        try:
            p = plan_item(dict(item) if isinstance(item, dict) else item, n)
            key = _cache_key(p)
        except Exception:  # noqa: BLE001 - a bad line needs no model
            continue
        if not ((cache_dir / key[:2] / f"{key}.wav").exists() and (cache_dir / key[:2] / f"{key}.json").exists()):
            return True
    return False


def _cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _write_results(out_dir: Path, rows: list[dict], fresh: bool) -> None:
    path = out_dir / RESULTS_NAME
    ids = {r.get("id") for r in rows}
    old = [] if fresh else [r for r in read_jsonl(path) if r.get("id") not in ids]
    write_jsonl(path, old + rows)


def load_items(path: str | Path) -> list[dict]:
    items = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            items.append(json.loads(line))
        except json.JSONDecodeError as e:
            items.append({"_parse_error": str(e)[:120]})
    return items


def synthesize(items: list[dict], out_dir: str | Path, gpu: bool = True, wait_min: int | None = None,
               extra_args: Iterable[str] = ()) -> list[dict]:
    """Pipeline entry point usable from ANY venv: runs the batch in venvs/qwentts (through gpu_run.sh) and returns the
    result rows of these items (input order)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not list(extra_args) and not needs_model(items):
        # every line is in the cache (or is a bad line): serve it in this process - no venv switch, no GPU queue
        rows = synthesize_batch([dict(it) if isinstance(it, dict) else it for it in items], out_dir)
        return rows
    in_path = out_dir / f"tts_in_{sha(items, 10)}.jsonl"
    write_jsonl(in_path, items)
    cmd = [str(Q.venv_python), "-m", "src.generate.tts_qwen", "--in", str(in_path), "--out-dir", str(out_dir), *extra_args]
    if gpu:
        cmd = [str(Path(CFG.paths.repo) / "scripts" / "gpu_run.sh"), str(int(Q.vram_gb))] + cmd
    else:
        cmd += ["--device", "cpu"]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(CFG.paths.repo)
    env.setdefault("TMPDIR", str(CFG.paths.tmp))
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    if wait_min is not None:
        env["GPU_WAIT_MIN"] = str(wait_min)
    log_path = out_dir / "tts_subprocess.log"
    with open(log_path, "a") as lf:
        r = subprocess.run(cmd, cwd=str(CFG.paths.repo), env=env, stdout=lf, stderr=subprocess.STDOUT,
                           stdin=subprocess.DEVNULL)
    by_id = {r_.get("id"): r_ for r_ in read_jsonl(out_dir / RESULTS_NAME)}
    out = []
    for n, it in enumerate(items):
        rid = it.get("id", f"line{n:05d}")
        out.append(by_id.get(rid) or {"id": rid, "path": None, "dur": None, "raw_dur": None, "fit": "none", "ok": False,
                                      "error": f"tts subprocess failed (exit {r.returncode}), see {log_path}"})
    return out


def _is_json(v: str) -> bool:
    try:
        json.loads(v)
        return True
    except ValueError:
        return False


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch English TTS (Qwen3-TTS 1.7B voice cloning + instructions)")
    ap.add_argument("--in", dest="inp", help="input JSONL (one utterance per line)")
    ap.add_argument("--out-dir", help="output directory (<id>.wav + tts_results.jsonl)")
    ap.add_argument("--device", default=None, help="cuda:0 | cpu (default: auto)")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--fresh", action="store_true", help="overwrite tts_results.jsonl instead of merging by id")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--show-text", metavar="TEXT", help="print the normalised text + word count and exit")
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                    help="config override, repeatable (e.g. --set tts_qwen.instruct_mode=cv_xvec)")
    a = ap.parse_args()
    for ov in a.overrides:
        k, _, v = ov.partition("=")
        OmegaConf.update(CFG, k.strip(), OmegaConf.to_container(OmegaConf.create({"v": json.loads(v) if _is_json(v) else v}))["v"])
    if a.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if a.show_text is not None:
        tn = normalize_en(a.show_text)
        print(json.dumps({"text_norm": tn, "n_words": count_words(tn) if tn else 0,
                          "bounds_sec": duration_bounds(count_words(tn), tn) if tn else None}))
        return
    if not a.inp or not a.out_dir:
        ap.error("--in and --out-dir are required")
    items = load_items(a.inp)
    if a.limit:
        items = items[: a.limit]
    rows = synthesize_batch(items, a.out_dir, device=a.device, use_cache=not a.no_cache, fresh=a.fresh,
                            batch_size=a.batch_size)
    sys.exit(0 if (not rows or any(r.get("ok") for r in rows)) else 2)


if __name__ == "__main__":
    main()
