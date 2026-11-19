"""Japanese character TTS: IndexTTS-2.5 zero-shot voice cloning + emotion vectors + duration fitting.

Input  JSONL, one utterance per line:
  {"id": "c140_s023_l0", "speaker": "Tadano", "text": "日本語のセリフ", "lang": "JA",
   "ref_wavs": ["/abs/ref1.wav", ...], "emotion": "shy" | [8 floats] | null,
   "target_dur": 2.8 | null, "max_dur": 3.5 | null}
  optional: "seed": int (re-roll a take), "emo_ref": "/abs/emotional_clip.wav" (emotion reference audio),
            "real_wav": "/abs/real_line.wav" (ground-truth clip, only used by voice_metrics), "meta": {...}.
Output <out_dir>/<id>.wav (mono, 44.1 kHz, 16-bit, trimmed, peak-normalised) and <out_dir>/tts_results.jsonl:
  {"id","path","dur","raw_dur","fit":"native|speed|stretch|none","ok":bool,"error":str|null, ...extras}

Run (TTS venv; GPU jobs always through gpu_run.sh):
  . scripts/env.sh
  scripts/gpu_run.sh 8 $KOMI_WORK/venvs/tts/bin/python -m src.generate.tts --in lines.jsonl --out-dir runs/x/tts
From pipeline code in another venv:  from src.generate.tts import synthesize ; rows = synthesize(items, out_dir)

Everything is local. The model is loaded once per batch and only if at least one line is not in the cache.
See docs/tts.md for the fitting policy, measured numbers and limits.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unicodedata
import zlib
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable

from omegaconf import OmegaConf

from src.common.config import CFG
from src.common.io import log, read_jsonl, sha, write_json, write_jsonl



def _tts_cfg():
    """configs/tts.yaml is namespaced under its file stem (CFG.tts); tolerate a flat merge into the root as well."""
    t = CFG.get("tts")
    return t if (t is not None and "model_dir" in t) else CFG


T = _tts_cfg()
EMO_DIMS = ["happy", "angry", "sad", "afraid", "disgusted", "melancholic", "surprised", "calm"]
MODEL_SR = 22050
RESULTS_NAME = "tts_results.jsonl"

# ============================================================================================ text front-end
_KANJI = "㐀-䶿一-鿿々〆"
_HIRA = "ぁ-ゖ"
_KATA = "ァ-ヺ"
_KANA = _HIRA + _KATA + "ー"
_RUBY = re.compile(rf"([{_KANJI}]+)\(([{_HIRA}{_KATA}ー]+)\)")
_SPEAKABLE = re.compile(rf"[{_KANJI}{_HIRA}{_KATA}A-Za-z0-9]")
_SMALL = set("ぁぃぅぇぉゃゅょゎァィゥェォャュョヮ")
_PUNCT_END = "。!?…"
_DIGITS = "〇一二三四五六七八九"


def int_to_kanji(n: int) -> str:
    """0..10^16-1 -> kanji numerals (読み上げ用: 123 -> 百二十三, 10000 -> 一万)."""
    if n == 0:
        return "〇"
    out = ""
    for unit, name in ((10**12, "兆"), (10**8, "億"), (10**4, "万"), (1, "")):
        q, n = divmod(n, unit)
        if not q:
            continue
        part = ""
        for u, nm in ((1000, "千"), (100, "百"), (10, "十"), (1, "")):
            d, q = divmod(q, u)
            if not d:
                continue
            bare = d == 1 and u > 1 and not (u == 1000 and unit > 1)     # 千 / 百 / 十, but 一千万
            part += nm if bare else _DIGITS[d] + nm
        out += part + name
    return out


def _num_to_ja(m: re.Match) -> str:
    s = m.group(0).replace(",", "")
    ip, _, fp = s.partition(".")
    if len(ip) > 16 or (len(ip) > 1 and ip.startswith("0")):     # ids, phone numbers: digit by digit
        head = "".join(_DIGITS[int(c)] for c in ip)
    else:
        head = int_to_kanji(int(ip))
    return head + ("点" + "".join(_DIGITS[int(c)] for c in fp) if fp else "")


def _lexicon() -> list[tuple[str, str]]:
    lex = OmegaConf.to_container(T.get("lexicon_ja", {}) or {})
    return sorted(((str(k), str(v)) for k, v in lex.items()), key=lambda kv: -len(kv[0]))


def normalize_ja(text: str, lexicon: list[tuple[str, str]] | None = None) -> str:
    """Script text -> text IndexTTS reads well. Returns "" when nothing speakable is left (e.g. "……")."""
    t = unicodedata.normalize("NFKC", str(text or ""))
    t = _RUBY.sub(r"\2", t)                                   # 漢字(かんじ) -> かんじ
    t = re.sub(r"[「」『』【】《》〈〉\[\]{}\"“”‘’'()]", "", t)
    for k, v in (lexicon if lexicon is not None else _lexicon()):
        t = t.replace(k, v)
    t = re.sub(r"\d[\d,]*(?:\.\d+)?", _num_to_ja, t)
    t = re.sub(rf"(?<=[{_KANA}])[〜~]+", "ー", t)              # え〜 -> えー
    t = re.sub(r"[〜~]+", "", t)
    t = re.sub(r"ー{3,}", "ーー", t)
    t = re.sub(r"\.{2,}|…+|‥+|・{2,}|。{2,}|[―—–]+|-{2,}", "…", t)
    t = re.sub(r"[♪♫☆★♡♥❤※*#＿_|\\/<>=+^`]", "", t)
    t = t.replace("．", "。").replace(",", "、").replace(";", "、").replace(":", "、")
    t = re.sub(r"!{2,}", "!", t)
    t = re.sub(r"\?{2,}", "?", t)
    t = re.sub(r"(?:\?!|!\?)(?:[!?])*", "!?", t)
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(rf"(?<=[{_KANJI}{_KANA}]) (?=[{_KANJI}{_KANA}])", "、", t)   # a space in JA script = a pause
    t = re.sub(r" *([、。!?…]) *", r"\1", t)
    t = re.sub(r"、{2,}", "、", t)
    t = re.sub(r"…[、。]+", "…", t)
    t = re.sub(r"[、。]+…", "…", t)
    t = re.sub(r"…{2,}", "…", t)
    t = re.sub(r"^[、。!?… ]+", "", t)
    t = re.sub(r"、+$", "", t)
    if not _SPEAKABLE.search(t):
        return ""
    if t[-1] not in _PUNCT_END:
        t += "。"
    return t


_TAGGER = None


def _tagger():
    global _TAGGER
    if _TAGGER is None:
        try:
            import fugashi
            _TAGGER = fugashi.Tagger()
        except Exception:
            _TAGGER = False
    return _TAGGER


def kana_reading(text: str) -> str:
    """Hiragana reading via MeCab/unidic-lite when available (else the text itself)."""
    tg = _tagger()
    if not tg:
        return text
    out = []
    for w in tg(text):
        k = getattr(w.feature, "kana", None) or getattr(w.feature, "pron", None)
        out.append(k if (k and k != "*") else w.surface)
    s = "".join(out)
    return "".join(chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c for c in s)


def count_mora(text_norm: str) -> int:
    """Approximate mora count (duration sanity + speaking-rate statistics)."""
    r = kana_reading(text_norm)
    n = 0.0
    for c in r:
        if c in _SMALL:
            continue
        if re.match(rf"[{_HIRA}{_KATA}ー]", c):
            n += 1
        elif re.match(rf"[{_KANJI}]", c):
            n += 2
        elif c.isascii() and c.isalnum():
            n += 0.6
    return max(1, int(round(n)))


# ============================================================================================ emotion
def _label_table() -> tuple[dict, dict]:
    e = T.emotion
    return OmegaConf.to_container(e.labels), OmegaConf.to_container(e.get("aliases", {}) or {})


_INTENSITY = {"slightly": 0.6, "a little": 0.6, "a bit": 0.6, "mildly": 0.6, "somewhat": 0.75, "quietly": 0.7,
              "very": 1.3, "extremely": 1.5, "really": 1.25, "super": 1.4, "so": 1.2}


def _finish_vec(v: list[float], scale: float, bias: bool) -> list[float]:
    v = [max(0.0, float(x)) * scale for x in v]
    if bias:  # same per-dimension bias as IndexTTS2.normalize_emo_vec (webui behaviour)
        v = [x * b for x, b in zip(v, [0.9375, 0.875, 1.0, 1.0, 0.9375, 0.9375, 0.6875, 0.5625])]
    s = sum(v)
    cap = float(T.emotion.max_sum)
    if s > cap:
        v = [x * cap / s for x in v]
    return [round(x, 4) for x in v]


def resolve_emotion(emotion: Any) -> tuple[list[float] | None, str, str | None]:
    """label | [8 floats] | {"happy": .5, ...} | None -> (vector or None, resolved label, warning or None).

    None means "no emotion vector": the model takes prosody from the speaker reference itself.
    Explicit vectors are passed through (only capped to the model's sum limit); labels are scaled by
    tts.emotion.strength and the IndexTTS bias.
    """
    if emotion is None or emotion == "" or emotion is False:
        return None, "none", None
    if isinstance(emotion, dict):
        return _finish_vec([float(emotion.get(k, 0.0)) for k in EMO_DIMS], 1.0, False), "vector", None
    if isinstance(emotion, (list, tuple)):
        if len(emotion) != 8:
            return None, "none", f"emotion vector must have 8 floats, got {len(emotion)} -> neutral"
        try:
            return _finish_vec([float(x) for x in emotion], 1.0, False), "vector", None
        except (TypeError, ValueError):
            return None, "none", "emotion vector is not numeric -> neutral"
    labels, aliases = _label_table()
    raw = str(emotion).strip().lower()
    parts = [p.strip() for p in re.split(r"[,/+&;|]| and | but ", raw) if p.strip()]
    vecs, names, unknown = [], [], []
    for p in parts:
        gain = 1.0
        for word, g in _INTENSITY.items():
            if p.startswith(word + " "):
                gain, p = g, p[len(word) + 1:].strip()
                break
        p = p.replace("_", " ").strip()
        key = None
        for cand in (p, p.replace(" ", "-"), p.replace(" ", ""), p.split(" ")[-1], p.rstrip("ly"), p.split(" ")[0]):
            cand = aliases.get(cand, cand)
            if cand in labels:
                key = cand
                break
        if key is None:
            unknown.append(p)
            continue
        names.append(key)
        if labels[key] is not None:
            vecs.append([float(x) * gain for x in labels[key]])
    warn = f"unknown emotion label(s) {unknown} -> ignored" if unknown else None
    if not vecs:
        return None, "+".join(names) or "none", warn
    mean = [sum(col) / len(vecs) for col in zip(*vecs)]
    vec = _finish_vec(mean, float(T.emotion.strength), bool(T.emotion.apply_bias))
    return (vec if max(vec) > 1e-6 else None), "+".join(names), warn


# ============================================================================================ audio helpers
def _np():
    import numpy as np
    return np


def read_mono(path: str | Path):
    import soundfile as sf
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return x.mean(axis=1), int(sr)


def resample(x, sr_in: int, sr_out: int):
    if sr_in == sr_out:
        return x
    from scipy.signal import resample_poly
    g = math.gcd(sr_in, sr_out)
    return resample_poly(x, sr_out // g, sr_in // g).astype("float32")


def rms_dbfs(x) -> float:
    np = _np()
    if len(x) == 0:
        return -120.0
    return float(20 * np.log10(max(1e-6, float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))))))


def frame_rms(x, sr: int, win_ms: float = 20.0, hop_ms: float = 5.0):
    np = _np()
    win, hop = max(1, int(sr * win_ms / 1000)), max(1, int(sr * hop_ms / 1000))
    if len(x) < win:
        return np.array([float(np.sqrt(np.mean(x**2)))] if len(x) else [0.0]), hop, win
    c = np.concatenate([[0.0], np.cumsum(x.astype(np.float64) ** 2)])
    idx = np.arange(0, len(x) - win + 1, hop)
    return np.sqrt((c[idx + win] - c[idx]) / win), hop, win


def trim_silence(x, sr: int, top_db: float | None = None, pad_head_ms: float | None = None,
                 pad_tail_ms: float | None = None):
    """Cut leading/trailing silence (energy relative to the loudest frame), keep small pads."""
    np = _np()
    top_db = float(T.trim.top_db) if top_db is None else top_db
    pad_head_ms = float(T.trim.pad_head_ms) if pad_head_ms is None else pad_head_ms
    pad_tail_ms = float(T.trim.pad_tail_ms) if pad_tail_ms is None else pad_tail_ms
    r, hop, win = frame_rms(x, sr)
    ref = float(r.max()) if len(r) else 0.0
    if ref <= 1e-5:
        return x[:0]
    on = np.nonzero(r > max(ref * 10 ** (-top_db / 20), 1e-4))[0]
    if len(on) == 0:
        return x[:0]
    a = max(0, int(on[0] * hop - sr * pad_head_ms / 1000))
    b = min(len(x), int(on[-1] * hop + win + sr * pad_tail_ms / 1000))
    return x[a:b]


def fade(x, sr: int, head_ms: float = 4.0, tail_ms: float = 12.0):
    """Raised-cosine fade in/out so trimmed clips never start or end on a step."""
    np = _np()
    x = x.copy()
    nh, nt = min(int(sr * head_ms / 1000), len(x) // 2), min(int(sr * tail_ms / 1000), len(x) // 2)
    if nh >= 2:
        x[:nh] *= (0.5 * (1 - np.cos(np.linspace(0, math.pi, nh)))).astype("float32")
    if nt >= 2:
        x[-nt:] *= (0.5 * (1 + np.cos(np.linspace(0, math.pi, nt)))).astype("float32")
    return x


def peak_norm(x, dbfs: float):
    np = _np()
    p = float(np.max(np.abs(x))) if len(x) else 0.0
    return x if p < 1e-6 else (x * (10 ** (dbfs / 20) / p)).astype("float32")


def time_stretch(x, sr: int, dur_ratio: float):
    """Pitch-preserving stretch with ffmpeg atempo. dur_ratio = new_duration / old_duration."""
    import soundfile as sf
    tempo = 1.0 / dur_ratio
    if abs(tempo - 1.0) < 1e-3:
        return x
    tmp_root = Path(CFG.paths.tmp)
    tmp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=tmp_root, prefix="tts_ts_") as d:
        a, b = Path(d) / "in.wav", Path(d) / "out.wav"
        sf.write(a, x, sr, subtype="FLOAT")
        # -nostdin + DEVNULL: ffmpeg must never touch the terminal (a background process group that does is
        # stopped by SIGTTIN/SIGTTOU, e.g. under `timeout` inside tmux, and the whole batch would hang)
        r = subprocess.run([str(CFG.paths.ffmpeg), "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                            "-i", str(a), "-filter:a", f"atempo={tempo:.5f}", "-c:a", "pcm_f32le", str(b)],
                           capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=120)
        if r.returncode != 0:
            raise RuntimeError(f"ffmpeg atempo failed: {r.stderr[-300:]}")
        y, _ = sf.read(b, dtype="float32")
    return y


def write_wav(path: str | Path, x, sr: int) -> None:
    import soundfile as sf
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.wav")
    sf.write(tmp, x, sr, subtype="PCM_16")
    os.replace(tmp, path)


def _file_sha(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


# ============================================================================================ speaker prompt
def prepare_prompt(ref_wavs: list[str]) -> dict:
    """ref_wavs -> one prompt wav (22.05 kHz mono) cached by content. IndexTTS takes a single clip (<= 15 s).

    mode=concat joins the refs in the given order (each trimmed + level-matched) until ref.max_sec is reached.
    """
    np = _np()
    rc = T.ref
    refs = [str(p) for p in (ref_wavs or [])]
    if not refs:
        raise ValueError("no ref_wavs given")
    missing = [p for p in refs if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError(f"reference wav missing: {missing[0]}")
    if rc.mode == "first":
        refs = refs[:1]
    key = sha({"files": [_file_sha(p) for p in refs], "mode": rc.mode, "max": float(rc.max_sec),
               "gap": int(rc.gap_ms), "v": 2})
    out = Path(T.ref_cache_dir) / f"{key}.wav"
    meta = out.with_suffix(".json")
    if out.exists() and meta.exists():
        return json.loads(meta.read_text())
    max_n = int(float(rc.max_sec) * MODEL_SR)
    gap = np.zeros(int(MODEL_SR * int(rc.gap_ms) / 1000), dtype="float32")
    parts, used, total = [], [], 0
    for p in refs:
        x, sr = read_mono(p)
        x = trim_silence(resample(x, sr, MODEL_SR), MODEL_SR, top_db=35.0, pad_head_ms=30, pad_tail_ms=80)
        if len(x) < int(0.3 * MODEL_SR):
            continue
        room = max_n - total - (len(gap) if parts else 0)
        if room < int(1.0 * MODEL_SR):
            break
        x = fade(peak_norm(x[:room], -3.0), MODEL_SR, 8, 30)
        if parts:
            parts.append(gap)
            total += len(gap)
        parts.append(x)
        total += len(x)
        used.append(p)
    if not parts:
        raise ValueError("reference audio is empty/silent")
    write_wav(out, np.concatenate(parts), MODEL_SR)
    info = {"prompt_wav": str(out), "prompt_key": key, "prompt_sec": round(total / MODEL_SR, 3), "refs_used": used}
    write_json(meta, info)
    return info


# ============================================================================================ engine
def _setup_env(offline: bool | None = None) -> None:
    """All model caches under models/tts/; no network at run time once the auxiliary models exist."""
    hf = Path(T.hf_cache)
    hf.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf)
    os.environ["HF_HUB_CACHE"] = str(hf / "hub")
    os.environ["MODELSCOPE_CACHE"] = str(hf.parent / "modelscope_cache")
    os.environ.setdefault("USE_MODELSCOPE", "false")
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if offline is None:
        offline = bool(T.offline) and aux_models_ready()
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"


def aux_models_ready() -> bool:
    c = Path(T.model_dir) / "hf_cache"
    need = [c / "w2v-bert-2.0" / "model.safetensors", c / "semantic_codec_model.safetensors",
            c / "campplus_cn_common.bin", c / "bigvgan" / "bigvgan_generator.pt", c / "bigvgan" / "config.json"]
    return all(p.is_file() for p in need)


def prepare_models() -> dict:
    """One-time download of the auxiliary models (w2v-bert-2.0, MaskGCT codec, CAMPPlus, BigVGAN)."""
    _setup_env(offline=False)
    from indextts.utils.model_download import ensure_models_available
    return ensure_models_available(str(T.model_dir))


@contextlib.contextmanager
def _quiet(verbose: bool):
    if verbose:
        yield
        return
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        yield


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


class IndexTTSEngine:
    """IndexTTS-2.5 loaded once. `synth` is deterministic for (text, prompt, emotion, seed, duration_factor).

    The GPT (text -> semantic codes) result is memoised per (prompt, emotion, seed, text), so a second call that only
    changes `duration_factor` re-runs just the flow-matching decoder + vocoder (speed fitting costs ~1/5 of a take).
    """

    def __init__(self, device: str | None = None, verbose: bool = False, threads: int | None = None):
        if device == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = ""      # never touch the shared GPU in CPU mode
        if not aux_models_ready():
            log("auxiliary models missing -> downloading once (network)")
            prepare_models()
        _setup_env()
        import torch
        if threads:
            torch.set_num_threads(int(threads))
        self.verbose = verbose
        code_dir = str(T.code_dir)
        if code_dir not in sys.path:
            sys.path.insert(0, code_dir)
        t0 = time.time()
        hub = os.environ["HF_HUB_CACHE"]
        with _quiet(verbose):
            from indextts.infer_v2_5 import IndexTTS2    # (sets HF_HUB_CACHE to a relative path at import)
            os.environ["HF_HUB_CACHE"] = hub
            self.tts = IndexTTS2(cfg_path=str(Path(T.model_dir) / "config.yaml"), model_dir=str(T.model_dir),
                                 use_bf16=bool(T.use_bf16) and device != "cpu", device=device,
                                 use_cuda_kernel=False, use_qwen_emo=False)
        self.device = str(self.tts.device)
        self.precision = "bf16" if self.tts.use_bf16 else "fp32"
        self.load_sec = round(time.time() - t0, 2)
        self._torch = torch
        self._memo: OrderedDict = OrderedDict()
        self._ctx: dict | None = None
        self._orig_inference_speech = self.tts.gpt.inference_speech
        self.tts.gpt.inference_speech = self._inference_speech_memo
        if self.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        log(f"IndexTTS-2.5 ready on {self.device} ({self.precision}) in {self.load_sec}s")

    # -- GPT memo: same seed + same text/prompt/emotion => same codes; decoder noise is re-seeded identically
    def _inference_speech_memo(self, *args, **kwargs):
        ctx = self._ctx
        if ctx is None:
            return self._orig_inference_speech(*args, **kwargs)
        seg = ctx["seg"]
        ctx["seg"] += 1
        toks = tuple(args[1].flatten().tolist())
        k = (ctx["key"], seg, toks, kwargs.get("max_generate_length"))
        hit = self._memo.get(k)
        if hit is None:
            _seed_all(ctx["seed"] + seg)
            t = time.perf_counter()
            codes, latent = self._orig_inference_speech(*args, **kwargs)
            ctx["gpt_sec"] += time.perf_counter() - t
            truncated = bool((codes[:, -1] != self.tts.stop_mel_token).any())   # no stop token = hit the cap
            hit = (codes.clone(), latent, int(codes.shape[-1]), truncated)
            self._memo[k] = hit
            while len(self._memo) > 16:
                self._memo.popitem(last=False)
        else:
            ctx["gpt_cached"] = True
        ctx["mel_tokens"] += hit[2]
        ctx["hit_cap"] = ctx["hit_cap"] or hit[3]
        _seed_all(ctx["seed"] + 7919 * (seg + 1))
        return hit[0].clone(), hit[1]

    def synth(self, text: str, prompt_wav: str, prompt_key: str, lang: str = "JA",
              emo_vector: list[float] | None = None, seed: int = 0, duration_factor: float = 1.0,
              max_mel_tokens: int = 1500, emo_ref: str | None = None):
        """-> (float32 mono waveform at 22.05 kHz, info). Untrimmed model output."""
        np = _np()
        gen = OmegaConf.to_container(T.gen)
        seg_tokens = int(gen.pop("max_text_tokens_per_segment", 120))
        self._ctx = {"key": sha([prompt_key, emo_vector, emo_ref, int(seed), gen, lang]), "seed": int(seed), "seg": 0,
                     "gpt_sec": 0.0, "gpt_cached": False, "mel_tokens": 0, "hit_cap": False}
        ctx = self._ctx
        t0 = time.perf_counter()
        try:
            with _quiet(self.verbose):
                out = self.tts.infer(
                    spk_audio_prompt=str(prompt_wav), text=text, output_path=None, lang=lang.upper(),
                    emo_audio_prompt=emo_ref if emo_vector is None else None,
                    emo_vector=list(emo_vector) if emo_vector is not None else None,
                    verbose=False, max_text_tokens_per_segment=seg_tokens,
                    duration_factor=float(duration_factor), text_normalization=lang.upper() != "JA",
                    max_mel_tokens=int(max_mel_tokens), **gen)
        finally:
            self._ctx = None
        if out is None:
            raise RuntimeError("IndexTTS returned no audio")
        sr, wav = out
        x = np.asarray(wav, dtype=np.float32).reshape(-1) / 32767.0
        info = {"synth_sec": time.perf_counter() - t0, "gpt_sec": ctx["gpt_sec"], "gpt_cached": ctx["gpt_cached"],
                "mel_tokens": ctx["mel_tokens"], "hit_cap": ctx["hit_cap"]}
        return resample(x, int(sr), MODEL_SR), info

    def vram_peak_gb(self) -> dict:
        t = self._torch
        if not self.device.startswith("cuda"):
            return {"allocated": 0.0, "reserved": 0.0}
        out = {"allocated": round(t.cuda.max_memory_allocated() / 2**30, 2),
               "reserved": round(t.cuda.max_memory_reserved() / 2**30, 2), "process_mib": None}
        try:        # what nvidia-smi charges to this process (torch reserve + CUDA context), one query per batch
            r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=15)
            for line in r.stdout.splitlines():
                pid, _, mem = line.partition(",")
                if pid.strip() == str(os.getpid()):
                    out["process_mib"] = int(mem.strip())
        except Exception:
            pass
        return out


# ============================================================================================ one utterance
def _tolerance(target: float) -> float:
    return max(float(T.fit.tolerance_rel) * target, float(T.fit.tolerance_abs))


def duration_bounds(n_mora: int, text_norm: str) -> tuple[float, float]:
    s = T.sanity
    lo = n_mora * float(s.min_sec_per_mora)
    hi = n_mora * float(s.max_sec_per_mora) + float(s.max_extra_sec) + float(s.pause_sec) * text_norm.count("…")
    return lo, hi


def _take(engine: IndexTTSEngine, text_norm: str, prompt: dict, lang: str, emo_vec, seed: int, df: float,
          max_mel: int, emo_ref: str | None, stats: dict):
    x, info = engine.synth(text_norm, prompt["prompt_wav"], prompt["prompt_key"], lang, emo_vec, seed, df, max_mel,
                           emo_ref)
    stats["synth_sec"] += info["synth_sec"]
    stats["gpt_sec"] += info["gpt_sec"]
    stats["takes"] += 1
    y = trim_silence(x, MODEL_SR)
    return y, info


def synthesize_one(engine: IndexTTSEngine, item: dict, text_norm: str, prompt: dict, emo_vec, seed: int) -> tuple[Any, dict]:
    """Native take (+ sanity re-rolls) -> duration fitting. Returns (44.1 kHz waveform, partial result row)."""
    f, s = T.fit, T.sanity
    lang = str(item.get("lang") or "JA")
    emo_ref = item.get("emo_ref")
    n_mora = count_mora(text_norm)
    lo, hi = duration_bounds(n_mora, text_norm)
    max_mel = int(min(1500, max(120, math.ceil(float(T.mel_tokens_per_sec) * (hi + 1.0)))))
    stats = {"synth_sec": 0.0, "gpt_sec": 0.0, "takes": 0}
    warnings: list[str] = []

    # ---- native take with sanity re-rolls
    best = None
    for k in range(int(s.retries) + 1):
        y, info = _take(engine, text_norm, prompt, lang, emo_vec, seed + k, 1.0, max_mel, emo_ref, stats)
        d = len(y) / MODEL_SR
        level = rms_dbfs(y)
        problem = None
        if d <= 0.03 or level < float(s.min_rms_dbfs):
            problem = "silent"
        elif info["hit_cap"]:
            problem = f"runaway (hit the {max_mel}-token cap)"
        elif d > hi:
            problem = f"too_long ({d:.2f}s for {n_mora} mora, limit {hi:.2f}s)"
        elif d < lo:
            problem = f"too_short ({d:.2f}s for {n_mora} mora, limit {lo:.2f}s)"
        badness = 0.0 if problem is None else (1e9 if problem == "silent" else abs(d - min(max(d, lo), hi)) + 1.0)
        if best is None or badness < best[0]:
            best = (badness, y, seed + k, problem)
        if problem is None:
            break
        warnings.append(f"take seed+{k}: {problem}")
    _, y, seed_used, problem = best
    raw_dur = len(y) / MODEL_SR
    row: dict[str, Any] = {"raw_dur": round(raw_dur, 3), "n_mora": n_mora, "seed": seed_used,
                           "sec_per_mora": round(raw_dur / n_mora, 4), "duration_factor": 1.0, "atempo": 1.0}
    if problem == "silent":
        row.update(ok=False, error="sanity_failed: silent output", fit="none", warnings=warnings, **_stats(stats))
        return None, row

    # ---- duration fitting
    target = item.get("target_dur")
    max_dur = item.get("max_dur")
    want = None
    if target is not None:
        want = float(target)
        if max_dur is not None and want > float(max_dur):
            want = float(max_dur) * 0.98
    elif max_dur is not None and raw_dur > float(max_dur):
        want = float(max_dur) * 0.97
    fit = "none" if (target is None and max_dur is None) else "native"
    df, stretch = 1.0, 1.0
    over = max_dur is not None and raw_dur > float(max_dur)         # the cap always triggers a fit
    if want is not None and (over or abs(raw_dur - want) > 0.6 * _tolerance(want)):
        if over:
            want = min(want, float(max_dur) * 0.98)
        ratio = want / raw_dur
        reach_hi = float(f.speed_max) * float(f.stretch_max)
        if ratio > 1 and (not bool(f.slow_down) or ratio > reach_hi):
            warnings.append(f"line is {raw_dur:.2f}s for a {want:.2f}s slot: left at natural speed (pad with silence)")
        else:
            cur = y
            for _ in range(2):                                    # model speed control first (decoder-only re-run)
                new_df = min(float(f.speed_max), max(float(f.speed_min), df * want / (len(cur) / MODEL_SR)))
                if abs(new_df - df) < 0.012:
                    break
                df = new_df
                cur, _ = _take(engine, text_norm, prompt, lang, emo_vec, seed_used, df, max_mel, emo_ref, stats)
                fit = "speed"
                if abs(len(cur) / MODEL_SR - want) <= 0.5 * _tolerance(want):
                    break
            if len(cur) < int(0.03 * MODEL_SR):                    # decoder produced nothing usable: keep native
                cur, df, fit = y, 1.0, "native"
            d = len(cur) / MODEL_SR
            if abs(d - want) > 0.5 * _tolerance(want):             # residual -> small pitch-preserving stretch
                stretch = min(float(f.stretch_max), max(float(f.stretch_min), want / d))
                if abs(stretch - 1.0) > 0.01:
                    cur = time_stretch(cur, MODEL_SR, stretch)
                    fit = "stretch"
                else:
                    stretch = 1.0
            y = cur
    out = peak_norm(fade(resample(y, MODEL_SR, int(T.sample_rate)), int(T.sample_rate)), float(T.peak_dbfs))
    dur = len(out) / int(T.sample_rate)
    fit_ok = True
    if target is not None:
        fit_ok = abs(dur - float(target)) <= _tolerance(float(target))
        row["dur_err"] = round(dur - float(target), 3)
    if max_dur is not None and dur > float(max_dur) + 0.02:
        fit_ok = False
        row["over_max"] = round(dur - float(max_dur), 3)
    if not fit_ok:
        row["min_feasible_dur"] = round(raw_dur * float(f.speed_min) * float(f.stretch_min), 3)
        row["max_feasible_dur"] = round(raw_dur * float(f.speed_max) * float(f.stretch_max), 3)
    row.update(dur=round(dur, 3), fit=fit, fit_ok=fit_ok, duration_factor=round(df, 4),
               atempo=round(1.0 / stretch, 4), warnings=warnings, **_stats(stats))
    if problem is not None:
        row.update(ok=False, error=f"sanity_failed: {problem}")
    else:
        row.update(ok=True, error=None)
    return out, row


def _stats(stats: dict) -> dict:
    return {"synth_sec": round(stats["synth_sec"], 3), "gpt_sec": round(stats["gpt_sec"], 3), "takes": stats["takes"]}


# ============================================================================================ batch
def _safe_id(i: Any) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(i))[:150] or "line"


def _cache_key(text_norm: str, lang: str, prompt_key: str, emo_vec, emo_ref, seed: int, target, max_dur,
               precision: str) -> str:
    model_sig = [f"{n}:{(Path(T.model_dir) / n).stat().st_size}" for n in ("gpt.pth", "s2mel.pth")
                 if (Path(T.model_dir) / n).exists()]
    return sha({"v": 3, "engine": T.engine, "commit": T.code_commit, "model": model_sig, "text": text_norm,
                "lang": lang.upper(), "prompt": prompt_key, "emo": emo_vec,
                "emo_ref": _file_sha(emo_ref) if emo_ref and Path(emo_ref).is_file() else None, "seed": seed,
                "target": target, "max": max_dur, "precision": precision, "gen": OmegaConf.to_container(T.gen),
                "fit": OmegaConf.to_container(T.fit), "trim": OmegaConf.to_container(T.trim),
                "sanity": OmegaConf.to_container(T.sanity), "sr": int(T.sample_rate), "peak": float(T.peak_dbfs)}, 24)


def _precision_for(device: str | None) -> str:
    if device == "cpu":
        return "fp32"
    if device is None:
        try:
            import torch
            if not torch.cuda.is_available():
                return "fp32"
        except Exception:
            return "fp32"
    return "bf16" if bool(T.use_bf16) else "fp32"


def synthesize_batch(items: Iterable[dict], out_dir: str | Path, device: str | None = None, use_cache: bool = True,
                     fresh: bool = False, verbose: bool = False, threads: int | None = None,
                     engine: IndexTTSEngine | None = None) -> list[dict]:
    """In-process batch synthesis (needs the TTS venv). Never raises for a bad line: the row gets ok=false + error."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(T.cache_dir)
    lex = _lexicon()
    precision = engine.precision if engine is not None else _precision_for(device)
    t_start = time.time()
    rows: list[dict | None] = []
    plans: list[dict] = []

    # ---- pass 1: validate, normalise, resolve prompts/emotions, serve cache hits (no model needed)
    for n, item in enumerate(items):
        row: dict[str, Any] = {"id": f"line{n:05d}", "path": None, "dur": None, "raw_dur": None, "fit": "none",
                               "ok": False, "error": None}
        rows.append(row)
        try:
            if not isinstance(item, dict):
                raise ValueError(f"line {n}: not a JSON object ({str(item)[:80]})")
            if item.get("_parse_error"):
                raise ValueError(f"line {n}: invalid JSON ({item['_parse_error']})")
            row["id"] = item.get("id", f"line{n:05d}")
            for k in ("speaker", "text", "lang", "target_dur", "max_dur", "ref_wavs", "emotion"):
                row[k] = item.get(k)
            for k in ("real_wav", "meta"):               # optional pass-through (evaluation / caller bookkeeping)
                if k in item:
                    row[k] = item[k]
            lang = str(item.get("lang") or "JA").upper()
            row["lang"] = lang
            for k in ("target_dur", "max_dur"):
                v = item.get(k)
                if v is not None:
                    v = float(v)
                    if not (v > 0 and math.isfinite(v)):
                        raise ValueError(f"{k} must be a positive number, got {item.get(k)!r}")
                    item[k] = v
            text_norm = normalize_ja(item.get("text"), lex) if lang == "JA" else str(item.get("text") or "").strip()
            row["text_norm"] = text_norm
            if not text_norm:
                raise ValueError("no_speakable_text (silent line: nothing to synthesise)")
            prompt = prepare_prompt(item.get("ref_wavs") or [])
            row.update(prompt_wav=prompt["prompt_wav"], prompt_sec=prompt["prompt_sec"])
            warn0 = []
            if prompt["prompt_sec"] < float(T.ref.min_sec):
                warn0.append(f"short reference ({prompt['prompt_sec']}s): clone quality will be poor")
            emo_vec, emo_label, emo_warn = resolve_emotion(item.get("emotion"))
            if emo_warn:
                warn0.append(emo_warn)
            row.update(emotion_resolved=emo_label, emo_vector=emo_vec)
            seed = int(item["seed"]) if item.get("seed") is not None else \
                (int(T.seed) + zlib.crc32(f"{item.get('speaker')}|{text_norm}".encode())) % (2**31 - 1)
            key = _cache_key(text_norm, lang, prompt["prompt_key"], emo_vec, item.get("emo_ref"), seed,
                             item.get("target_dur"), item.get("max_dur"), precision)
            row["key"] = key
            wav_path = out_dir / f"{_safe_id(row['id'])}.wav"
            c_wav, c_json = cache_dir / key[:2] / f"{key}.wav", cache_dir / key[:2] / f"{key}.json"
            if use_cache and c_wav.exists() and c_json.exists():
                cached = json.loads(c_json.read_text())
                shutil.copyfile(c_wav, wav_path)
                row.update(cached)
                row.update(path=str(wav_path), cached=True)
                continue
            plans.append({"n": n, "item": item, "text_norm": text_norm, "prompt": prompt, "emo_vec": emo_vec,
                          "seed": seed, "key": key, "wav_path": wav_path, "c_wav": c_wav, "c_json": c_json,
                          "warn0": warn0})
        except Exception as e:  # noqa: BLE001 - one bad line must never stop the batch
            row.update(ok=False, error=f"{type(e).__name__}: {e}"[:500])

    # ---- pass 2: synthesise the misses, grouped by speaker prompt (the model caches the last prompt)
    load_sec = 0.0
    if plans:
        if engine is None:
            try:
                engine = IndexTTSEngine(device=device, verbose=verbose, threads=threads)
                load_sec = engine.load_sec
            except Exception as e:  # noqa: BLE001 - model could not be loaded: every pending line fails, cached stay
                for p in plans:
                    rows[p["n"]].update(ok=False, error=f"model_load_failed: {type(e).__name__}: {e}"[:500])
                log(f"TTS model load failed: {type(e).__name__}: {e}")
                plans = []
        plans.sort(key=lambda p: (p["prompt"]["prompt_key"], p["n"]))
        for i, p in enumerate(plans):
            row = rows[p["n"]]
            try:
                wav, res = synthesize_one(engine, p["item"], p["text_norm"], p["prompt"], p["emo_vec"], p["seed"])
                res["warnings"] = p["warn0"] + res.get("warnings", [])
                res["rtf"] = round(res["synth_sec"] / res["dur"], 3) if res.get("dur") else None
                res["device"] = engine.device
                if wav is not None:
                    write_wav(p["wav_path"], wav, int(T.sample_rate))
                    row.update(res)
                    row.update(path=str(p["wav_path"]), cached=False)
                    if res["ok"] and use_cache:
                        p["c_wav"].parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(p["wav_path"], p["c_wav"])
                        write_json(p["c_json"], {k: v for k, v in res.items() if k != "path"})
                else:
                    row.update(res)
            except Exception as e:  # noqa: BLE001
                row.update(ok=False, error=f"{type(e).__name__}: {e}"[:500])
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
            state = "ok " if row["ok"] else "ERR"
            log(f"[{i + 1}/{len(plans)}] {state} {row['id']} fit={row.get('fit')} raw={row.get('raw_dur')}s "
                f"dur={row.get('dur')}s target={row.get('target_dur')} synth={row.get('synth_sec')}s"
                + ("" if row["ok"] else f" :: {row['error']}"))
            _write_results(out_dir, rows, fresh)          # progress survives a crash; reruns hit the cache

    _write_results(out_dir, rows, fresh)
    done = [r for r in rows if r and r.get("ok")]
    new = [r for r in done if not r.get("cached")]
    audio_sec = sum(r["dur"] for r in new)
    synth_sec = sum(r.get("synth_sec", 0.0) for r in new)
    summary = {"n": len(rows), "ok": len(done), "failed": len(rows) - len(done),
               "cached": sum(1 for r in done if r.get("cached")), "synthesised": len(new),
               "audio_sec": round(audio_sec, 2), "synth_sec": round(synth_sec, 2),
               "compute_sec_per_audio_sec": round(synth_sec / audio_sec, 3) if audio_sec else None,
               "gpt_share_of_compute": (round(sum(r.get("gpt_sec", 0.0) for r in new) / synth_sec, 3)
                                        if synth_sec else None),
               "takes": sum(r.get("takes", 0) for r in new),
               "model_load_sec": load_sec, "wall_sec": round(time.time() - t_start, 2),
               "device": engine.device if engine else None, "precision": precision,
               "vram_peak_gb": engine.vram_peak_gb() if engine else None,
               "fit_ok": sum(1 for r in done if r.get("fit_ok")), "engine": T.engine, "code_commit": T.code_commit}
    write_json(out_dir / "tts_run.json", summary)
    log(f"TTS batch: {summary}")
    return [r for r in rows if r is not None]


def _write_results(out_dir: Path, rows: list[dict | None], fresh: bool) -> None:
    """tts_results.jsonl = rows of this batch (input order) + earlier rows of other ids (unless fresh)."""
    path = out_dir / RESULTS_NAME
    cur = [r for r in rows if r is not None]
    ids = {r.get("id") for r in cur}
    old = [] if fresh else [r for r in read_jsonl(path) if r.get("id") not in ids]
    write_jsonl(path, old + cur)


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


def synthesize_jsonl(in_jsonl: str | Path, out_dir: str | Path, **kw) -> list[dict]:
    return synthesize_batch(load_items(in_jsonl), out_dir, **kw)


def synthesize(items: list[dict], out_dir: str | Path, gpu: bool = True, wait_min: int | None = None,
               extra_args: Iterable[str] = ()) -> list[dict]:
    """Pipeline entry point usable from ANY venv: runs the batch in the TTS venv (through gpu_run.sh) and
    returns the result rows of these items (input order)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    in_path = out_dir / f"tts_in_{sha(items, 10)}.jsonl"
    write_jsonl(in_path, items)
    cmd = [str(T.venv_python), "-m", "src.generate.tts", "--in", str(in_path), "--out-dir", str(out_dir), *extra_args]
    if gpu:
        cmd = [str(Path(CFG.paths.repo) / "scripts" / "gpu_run.sh"), str(int(T.vram_gb))] + cmd
    else:
        cmd += ["--device", "cpu"]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(CFG.paths.repo)
    env.setdefault("TMPDIR", str(CFG.paths.tmp))
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
    ap = argparse.ArgumentParser(description="Batch TTS (IndexTTS-2.5 zero-shot cloning + duration fitting)")
    ap.add_argument("--in", dest="inp", help="input JSONL (one utterance per line)")
    ap.add_argument("--out-dir", help="output directory (<id>.wav + tts_results.jsonl)")
    ap.add_argument("--device", default=None, help="cuda:0 | cpu (default: auto)")
    ap.add_argument("--threads", type=int, default=None, help="CPU threads (cpu mode)")
    ap.add_argument("--no-cache", action="store_true", help="ignore and do not write the content-hash cache")
    ap.add_argument("--fresh", action="store_true", help="overwrite tts_results.jsonl instead of merging by id")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--verbose", action="store_true", help="show IndexTTS's own prints")
    ap.add_argument("--prepare", action="store_true", help="only download the auxiliary models, then exit")
    ap.add_argument("--show-text", metavar="TEXT", help="print the normalised text + mora count and exit")
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                    help="config override, repeatable (e.g. --set tts.ref.mode=first --set tts.emotion.strength=0.8)")
    a = ap.parse_args()
    for ov in a.overrides:
        k, _, v = ov.partition("=")
        OmegaConf.update(CFG, k.strip(), OmegaConf.to_container(OmegaConf.create({"v": json.loads(v) if _is_json(v) else v}))["v"])
    if a.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if a.threads:
        os.environ["OMP_NUM_THREADS"] = str(a.threads)
        os.environ["MKL_NUM_THREADS"] = str(a.threads)
    if a.prepare:
        print(json.dumps(prepare_models(), indent=1))
        return
    if a.show_text is not None:
        tn = normalize_ja(a.show_text)
        print(json.dumps({"text_norm": tn, "n_mora": count_mora(tn) if tn else 0, "reading": kana_reading(tn)},
                         ensure_ascii=False))
        return
    if not a.inp or not a.out_dir:
        ap.error("--in and --out-dir are required")
    items = load_items(a.inp)
    if a.limit:
        items = items[: a.limit]
    rows = synthesize_batch(items, a.out_dir, device=a.device, use_cache=not a.no_cache, fresh=a.fresh,
                            verbose=a.verbose, threads=a.threads)
    # exit code: 0 = batch processed (individual failures are in tts_results.jsonl), 2 = nothing could be synthesised
    sys.exit(0 if (not rows or any(r.get("ok") for r in rows)) else 2)


if __name__ == "__main__":
    main()
