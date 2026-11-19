"""Voice bank: per-character reference clips for zero-shot voice cloning (and later fine-tuning data).

in : data/processed/anime/lines_named.jsonl   {"ep","idx","start","end","text","speaker","conf"}  (TRAIN episodes)
     data/interim/anime/epNN/vocals.wav (preferred, separated speech) or audio_jpn.wav (full mix fallback)
out: data/processed/voices/<Speaker>/ref_00.wav ... (mono 44.1 kHz) + meta.json + centroid.npy
     data/processed/voices/voices.json  (index: refs, default_refs, quality per speaker)

A reference must be clean, 3-12 s, one speaker only:
  * candidates = single lines or runs of adjacent lines of the same speaker (gap <= merge_gap_sec), never touching
    another speaker's line (lines_named + every dialogue subtitle of the episode, with guard_sec);
  * scored by speaker consistency (ECAPA cosine to the character centroid, robust to wrong labels), an SNR proxy
    (speech level vs floor; low = music/SFX bleed), duration, speech ratio and label confidence;
  * the best non-overlapping, time-diverse clips are written; `default_refs` are what tts.py concatenates.
Split rule: clips come ONLY from CFG.split.train_episodes (held-out 22-24 are refused).

CLI (TTS venv, CPU is enough):
  $KOMI_WORK/venvs/tts/bin/python -m src.train.voice_bank [--lines X.jsonl] [--out-dir DIR] [--speakers A,B]

English bank (`--lang en`, settings = tts.voice_bank overridden by tts_qwen.voice_bank):
  in : data/processed/anime/lines_en.jsonl (src.anime.speakers_en), data/interim/anime/epNN/vocals_en.wav,
       data/processed/anime/dub_lines/epNN.json (every dub line: the guard against other voices)
  out: data/processed/voices_en/<Speaker>/ref_NN.wav + style_high_NN.wav / style_low_NN.wav + meta.json + centroid.npy,
       data/processed/voices_en/voices.json = the Japanese shape plus "ref_texts", "default_ref_texts" (Qwen3-TTS
       cloning needs the transcript of the reference) and "style_refs": {"high": {"refs", "ref_texts"}, "low": {...}}
       = the speaker's most / least aroused clean clips (pitch + level), for emotion-matched cloning.
  scripts/gpu_run.sh 6 $KOMI_WORK/venvs/tts/bin/python -m src.train.voice_bank --lang en --device cuda --transcribe
"""
from __future__ import annotations

import argparse
import math
import os
import re
import time
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from src.common.config import CFG
from src.common.io import file_sig, log, read_json, read_jsonl, sha, write_json
from src.generate.tts import T, fade, frame_rms, peak_norm, resample, trim_silence, write_wav

V = T.voice_bank
LANG = "ja"
OUT_SR = 44100


def use_lang(lang: str = "ja") -> None:
    """Select the bank: "ja" = tts.voice_bank (Japanese stems + subtitle lines), "en" = the same settings overridden by
    tts_qwen.voice_bank (dub stems, lines_en.jsonl, voices_en/)."""
    global V, LANG
    LANG = str(lang).lower()
    if LANG == "ja":
        V = T.voice_bank
    elif LANG == "en":
        V = OmegaConf.merge(T.voice_bank, CFG.tts_qwen.voice_bank)
    else:
        raise ValueError(f"unknown voice bank language {lang!r}")
_SKIP_NAMES = {"", "none", "null", "unknown", "?", "unk", "other", "crowd", "all", "everyone", "sfx", "sign"}


def _ov(spk: str, key: str, default=None):
    """Per-speaker override of a voice-bank setting (voice_bank.speaker_overrides.<Speaker>.<key>), else the default."""
    o = (V.get("speaker_overrides") or {}).get(spk) or {}
    return o[key] if key in o else (V[key] if default is None and key in V else default)


def squeeze_pauses(x, sr: int, max_pause: float):
    """Shorten every internal silence longer than `max_pause` seconds to `max_pause` (halting speakers: a reference
    clip should be mostly speech). 5 ms cross-fades at the joins."""
    import numpy as np
    r, hop, win = frame_rms(x, sr)
    if len(r) < 3:
        return x
    sil = r < max(float(r.max()) * 10 ** (-35 / 20), 1e-4)
    keep = np.ones(len(x), bool)
    k, n = 0, len(sil)
    while k < n:
        if sil[k]:
            j = k
            while j < n and sil[j]:
                j += 1
            a, b = k * hop + win, j * hop                       # samples that are silent in every frame touching them
            if k > 0 and j < n and (b - a) / sr > max_pause:
                cut = int((b - a) - max_pause * sr)
                m = (a + b) // 2
                keep[m - cut // 2: m - cut // 2 + cut] = False
            k = j
        else:
            k += 1
    return x[keep]


def _safe(name: str) -> str:
    return re.sub(r"[^\w.-]+", "_", name.strip())[:60] or "speaker"


def _stem_for(ep: int, audio_dir: Path) -> tuple[Path | None, str]:
    for name in V.stems:
        p = audio_dir / f"ep{ep:02d}" / str(name)
        if p.is_file():
            return p, ("vocals" if "vocal" in str(name) else "mix")
    return None, "none"


def _read_segment(path: Path, start: float, end: float):
    """Mono float32 segment [start, end] seconds (seeks; never loads the whole episode)."""
    import soundfile as sf
    with sf.SoundFile(str(path)) as f:
        sr = f.samplerate
        a = max(0, int(start * sr))
        b = min(f.frames, int(end * sr))
        if b <= a:
            return None, sr
        f.seek(a)
        x = f.read(b - a, dtype="float32", always_2d=True)
    return x.mean(axis=1), sr


def audio_features(x, sr: int) -> dict:
    """Cheap cleanliness features of a (trimmed) clip."""
    import numpy as np
    r, _, _ = frame_rms(x, sr, 25.0, 10.0)
    r = np.maximum(r, 1e-6)
    db = 20 * np.log10(r)
    hi, lo = float(np.percentile(db, 95)), float(np.percentile(db, 10))
    speech = db > (hi - 30.0)
    return {"snr_db": round(min(60.0, hi - lo), 2), "speech_ratio": round(float(speech.mean()), 3),
            "level_db": round(hi, 2), "clip_frac": round(float((np.abs(x) > 0.985).mean()), 5)}


def _lin(v: float, a: float, b: float) -> float:
    return 0.0 if b == a else min(1.0, max(0.0, (v - a) / (b - a)))


def _dur_score(d: float) -> float:
    lo, hi = float(V.ideal_sec[0]), float(V.ideal_sec[1])
    if lo <= d <= hi:
        return 1.0
    if d < lo:
        return 0.5 + 0.5 * _lin(d, float(V.min_sec), lo)
    return 0.5 + 0.5 * (1 - _lin(d, hi, float(V.max_sec)))


# ============================================================================================ candidates
def build_candidates(lines: list[dict], other_intervals: dict[int, list[tuple[float, float]]]) -> dict[str, list[dict]]:
    """speaker -> candidate windows (single lines and merged runs of adjacent lines)."""
    by_ep: dict[int, list[dict]] = {}
    for ln in lines:
        by_ep.setdefault(int(ln["ep"]), []).append(ln)
    cands: dict[str, list[dict]] = {}
    for ep, ls in by_ep.items():
        ls.sort(key=lambda r: (float(r["start"]), float(r["end"])))
        others = sorted(other_intervals.get(ep, []))
        named = [(float(r["start"]), float(r["end"]), str(r.get("speaker") or "")) for r in ls]

        def clean(a: float, b: float, spk: str, own: set[tuple[float, float]]) -> bool:
            g = float(V.guard_sec)
            for s, e, sp in named:                        # another named speaker inside / too close
                if sp != spk and s < b + g and e > a - g:
                    return False
            for s, e in others:                           # any other dialogue subtitle (unlabelled lines too)
                if (round(s, 2), round(e, 2)) in own:
                    continue
                if s < b + g and e > a - g:
                    return False
            return True

        spk_lines: dict[str, list[dict]] = {}
        for r in ls:
            spk = str(r.get("speaker") or "").strip()
            if spk.lower() in _SKIP_NAMES or float(r.get("conf", 1.0)) < float(V.min_conf):
                continue
            if float(r["end"]) - float(r["start"]) < 0.25:
                continue
            spk_lines.setdefault(spk, []).append(r)
        for spk, rs in spk_lines.items():
            n = len(rs)
            gap_max, min_sec = float(_ov(spk, "merge_gap_sec")), float(_ov(spk, "min_sec"))
            for i in range(n):
                for j in range(i, n):
                    if j > i and float(rs[j]["start"]) - float(rs[j - 1]["end"]) > gap_max:
                        break
                    a, b = float(rs[i]["start"]), float(rs[j]["end"])
                    d = b - a
                    if d > float(V.max_sec):
                        break
                    if d < min_sec:
                        continue
                    own = {(round(float(r["start"]), 2), round(float(r["end"]), 2)) for r in rs[i:j + 1]}
                    if not clean(a, b, spk, own):
                        continue
                    seg = rs[i:j + 1]
                    cands.setdefault(spk, []).append({
                        "ep": ep, "start": round(a, 3), "end": round(b, 3), "dur": round(d, 3),
                        "idx": [r.get("idx") for r in seg], "text": " ".join(str(r.get("text") or "") for r in seg),
                        "conf": round(min(float(r.get("conf", 1.0)) for r in seg), 3), "n_lines": len(seg)})
    return cands


def _load_other_intervals(eps: set[int]) -> dict[int, list[tuple[float, float]]]:
    out: dict[int, list[tuple[float, float]]] = {}
    for ep in eps:
        if V.get("others_dir"):          # English: every dub line of the episode (named or not, songs included)
            rows = read_json(Path(V.others_dir) / f"ep{ep:02d}.json", default=[]) or []
            out[ep] = [(float(s["start"]), float(s["end"])) for s in rows]
            continue
        subs = read_json(Path(CFG.paths.processed) / "anime" / "subs" / f"ep{ep:02d}.json", default=[]) or []
        out[ep] = [(float(s["start"]), float(s["end"])) for s in subs if s.get("kind") == "dialogue"]
    return out


def _style_sets(pool: list[dict], n: int) -> dict[str, list[dict]]:
    """Most / least aroused clean clips of one speaker. arousal = z(log f0 median) + 0.6 z(level) + 0.4 z(f0 range):
    relative to the speaker's own clips, so "high" for a quiet character is still that character."""
    import numpy as np
    from src.eval.voice_metrics import prosody_features
    for f in pool:
        if "f0_med" not in f:
            f.update(prosody_features(f["_x"], f["_sr"], text=f.get("text")))
    ok = [f for f in pool if f.get("f0_med") and f.get("voiced_ratio", 0) >= 0.25]
    if len(ok) < max(8, 3 * n):
        return {}
    z = lambda v: (np.asarray(v) - np.mean(v)) / (np.std(v) + 1e-9)   # noqa: E731
    a = z([math.log2(f["f0_med"]) for f in ok]) + 0.6 * z([f["level_db"] for f in ok]) + 0.4 * z([f["f0_range_st"] for f in ok])
    for f, v in zip(ok, a):
        f["arousal"] = round(float(v), 3)
    out: dict[str, list[dict]] = {}
    for name, order in (("high", sorted(ok, key=lambda f: -f["arousal"])), ("low", sorted(ok, key=lambda f: f["arousal"]))):
        chosen: list[dict] = []
        for f in order:
            if len(chosen) >= n:
                break
            if any(c["ep"] == f["ep"] and f["start"] < c["end"] + 5.0 and f["end"] > c["start"] - 5.0 for c in chosen):
                continue
            chosen.append(f)
        out[name] = chosen
    return out


# ============================================================================================ build
def build_speaker(spk: str, cands: list[dict], embedder, audio_dir: Path, out_dir: Path, n_lines: int,
                  line_sec: float = 0.0, transcriber=None) -> dict:
    import numpy as np
    w = OmegaConf.to_container(V.weights)
    # cheap pre-ranking so we only decode/embed the most promising windows
    cands = sorted(cands, key=lambda c: -(0.6 * _dur_score(c["dur"]) + 0.4 * c["conf"]))[: int(V.max_candidates)]
    feats = []
    for c in cands:
        stem, kind = _stem_for(c["ep"], audio_dir)
        if stem is None:
            continue
        pad = float(V.edge_pad_sec)
        x, sr = _read_segment(stem, c["start"] - pad, c["end"] + pad)
        if x is None or len(x) < sr:
            continue
        x = trim_silence(x, sr, top_db=35.0, pad_head_ms=40, pad_tail_ms=80)
        if _ov(spk, "squeeze_pause_sec", 0):
            x = squeeze_pauses(x, sr, float(_ov(spk, "squeeze_pause_sec", 0)))
        d = len(x) / sr
        if d < float(_ov(spk, "min_sec")) * 0.8:
            continue
        f = audio_features(x, sr)
        min_snr = float(V.min_snr_db) if kind == "vocals" else float(V.min_snr_db_mix)
        if f["speech_ratio"] < float(_ov(spk, "min_speech_ratio")) or f["snr_db"] < min_snr or f["clip_frac"] > 0.002:
            continue
        c = dict(c, **f, stem=kind, stem_path=str(stem), clip_dur=round(d, 3))
        c["_x"], c["_sr"] = x, sr
        c["_emb"] = embedder.embed_wave(x, sr)
        feats.append(c)
    if not feats:
        return {"speaker": spk, "refs": [], "default_refs": [], "skipped": "no clean candidate (3-12 s, single speaker)",
                "n_lines": n_lines, "n_candidates": len(cands)}
    # robust centroid: weight by duration * confidence, drop outliers twice
    keep = list(range(len(feats)))
    centroid = None
    for _ in range(3):
        E = np.stack([feats[i]["_emb"] for i in keep])
        wts = np.array([feats[i]["clip_dur"] * feats[i]["conf"] for i in keep], dtype="float32")
        centroid = (E * wts[:, None]).sum(0)
        centroid /= np.linalg.norm(centroid) + 1e-9
        sims = np.array([float(np.dot(f["_emb"], centroid)) for f in feats])
        thr = max(float(V.min_consistency), float(np.median(sims[keep]) - 0.15))
        new_keep = [i for i in range(len(feats)) if sims[i] >= thr]
        if len(new_keep) < 1 or new_keep == keep:
            break
        keep = new_keep
    for i, f in enumerate(feats):
        f["consistency"] = round(float(sims[i]), 4)
        f["score"] = round(w["consistency"] * _lin(f["consistency"], 0.3, 0.8) + w["snr"] * _lin(f["snr_db"], 12.0, 40.0)
                           + w["duration"] * _dur_score(f["clip_dur"]) + w["speech_ratio"] * f["speech_ratio"]
                           + w["conf"] * f["conf"], 4)
        if LANG == "en" and not re.search(r"[.!?][\"')\]]*$", str(f.get("text") or "").strip()):
            f["score"] = round(f["score"] - 0.08, 4)      # in-context cloning wants a clip that ends on a finished sentence
    ranked = sorted((feats[i] for i in keep), key=lambda f: -f["score"])
    chosen: list[dict] = []
    for f in ranked:
        if len(chosen) >= int(V.n_refs):
            break
        if any(c["ep"] == f["ep"] and f["start"] < c["end"] + float(V.diversity_sec)
               and f["end"] > c["start"] - float(V.diversity_sec) for c in chosen):
            continue
        chosen.append(f)
    if len(chosen) < min(int(V.n_refs), 2):               # tiny speakers: relax diversity, still no overlap
        for f in ranked:
            if len(chosen) >= int(V.n_refs):
                break
            if f in chosen or any(c["ep"] == f["ep"] and f["start"] < c["end"] and f["end"] > c["start"] for c in chosen):
                continue
            chosen.append(f)
    sdir = out_dir / _safe(spk)
    sdir.mkdir(parents=True, exist_ok=True)
    for old in sdir.glob("ref_*.wav"):
        old.unlink()
    refs = []
    for k, f in enumerate(chosen):
        path = sdir / f"ref_{k:02d}.wav"
        y = fade(peak_norm(resample(f["_x"], f["_sr"], OUT_SR), -3.0), OUT_SR, 8, 30)
        write_wav(path, y, OUT_SR)
        ref = {"file": str(path), **{k2: v for k2, v in f.items() if not k2.startswith("_")}}
        _transcribe_ref(ref, path, transcriber)
        refs.append(ref)
    # English bank: emotion-matched sets (most / least aroused clean clips of this speaker)
    style_refs: dict[str, list[dict]] = {}
    if LANG == "en" and int(V.get("n_style", 0)) > 0:
        for old in sdir.glob("style_*.wav"):
            old.unlink()
        pool = [f for f in ranked if 3.0 <= f["clip_dur"] <= float(V.max_sec)]
        for style, clips in _style_sets(pool, int(V.n_style)).items():
            style_refs[style] = []
            for k, f in enumerate(clips):
                path = sdir / f"style_{style}_{k:02d}.wav"
                write_wav(path, fade(peak_norm(resample(f["_x"], f["_sr"], OUT_SR), -3.0), OUT_SR, 8, 30), OUT_SR)
                ref = {"file": str(path), **{k2: v for k2, v in f.items() if not k2.startswith("_")}}
                _transcribe_ref(ref, path, transcriber)
                style_refs[style].append(ref)
    if LANG == "en":                                       # a clip whose last word is cut ("li-") is a bad in-context prompt
        for r in refs:
            r["truncated"] = str(r.get("ref_text") or "").rstrip().endswith("-")
        refs.sort(key=lambda r: r["truncated"])            # stable: complete clips first, quality order kept
    np.save(sdir / "centroid.npy", centroid.astype("float32"))
    assert all(int(r["ep"]) in set(CFG.split.train_episodes) for r in refs), "held-out episode leaked into refs"
    assert all(int(r["ep"]) in set(CFG.split.train_episodes) for rs in style_refs.values() for r in rs), "held-out leak"
    meta = {"speaker": spk, "dir": str(sdir), "refs": refs, "lang": LANG,
            "default_refs": [r["file"] for r in refs[: int(V.n_default)]],
            "centroid": str(sdir / "centroid.npy"), "n_lines": n_lines, "n_candidates": len(cands),
            "n_clean": len(feats), "n_consistent": len(keep),
            "consistency_median": round(float(np.median(sims[keep])), 4),
            "quality": round(float(np.mean([r["score"] for r in refs])), 4) if refs else 0.0,
            "stems": sorted({r["stem"] for r in refs}),
            "line_sec": round(line_sec, 1),            # labelled speech available for this character (all lines)
            "built_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    if LANG == "en":
        meta["voice_f0"] = _voice_f0([r["file"] for r in refs])       # median pitch of the refs: how the voice sounds
        meta["style_refs"] = style_refs
        meta["consistency_mean_refs"] = round(float(np.mean([r["consistency"] for r in refs])), 4) if refs else None
        meta["ref_sec"] = round(float(sum(r["clip_dur"] for r in refs)), 1)
        meta["clean_sec"] = round(float(sum(feats[i]["clip_dur"] for i in keep)), 1)   # overlapping windows count twice
    write_json(sdir / "meta.json", meta)
    return meta


def _voice_f0(files: list[str]) -> float | None:
    """Median F0 (Hz) over the reference clips of a voice (used by the gender-aware fallback of tts_qwen)."""
    from src.eval.voice_metrics import prosody_features
    from src.generate.tts import read_mono
    f0 = []
    for f in files:
        x, sr = read_mono(f)
        v = prosody_features(x, sr).get("f0_med")
        if v:
            f0.append(float(v))
    f0.sort()
    return round(f0[len(f0) // 2], 1) if f0 else None


def backfill_voice_f0(out_dir: str | Path | None = None) -> dict:
    """Add `voice_f0` to an existing English bank (voices.json + meta.json) without rebuilding it."""
    use_lang("en")
    index_path = Path(out_dir or V.out_dir) / "voices.json"
    idx = read_json(index_path)
    for spk, v in idx["speakers"].items():
        if v.get("voice_f0") is None:
            v["voice_f0"] = _voice_f0(v["refs"])
            meta_p = Path(v["dir"]) / "meta.json"
            meta = read_json(meta_p, default={}) or {}
            meta["voice_f0"] = v["voice_f0"]
            write_json(meta_p, meta)
    write_json(index_path, idx)
    return {k: v.get("voice_f0") for k, v in idx["speakers"].items()}


def _transcribe_ref(ref: dict, path: Path, transcriber) -> None:
    """Japanese bank: optional `text_ja`. English bank: `ref_text` = what is actually said in the written clip
    (Whisper on the clip when a transcriber is given, else the text of the dub lines the clip was cut from)."""
    if LANG == "en":
        ref["ref_text"] = re.sub(r"\s+", " ", str(ref.get("text") or "")).strip()
        ref["ref_text_source"] = "lines"
        if transcriber is not None:
            try:
                hyp = transcriber.transcribe(path, "en")
                from src.eval.voice_metrics import wer
                w, _, n = wer(ref["ref_text"], hyp)
                ref["lines_vs_clip_wer"] = None if w is None else round(w, 3)
                if hyp:
                    ref["ref_text"], ref["ref_text_source"] = hyp, "whisper_clip"
            except Exception as e:  # noqa: BLE001
                ref["ref_text_error"] = str(e)[:120]
        return
    if transcriber is not None:
        try:
            ref["text_ja"] = transcriber.transcribe(path, "ja")
        except Exception as e:  # noqa: BLE001
            ref["text_ja_error"] = str(e)[:120]


def build(lines_path: str | Path | None = None, out_dir: str | Path | None = None, speakers: list[str] | None = None,
          device: str = "cpu", transcribe: bool = False, force: bool = False, lang: str = "ja", merge: bool = False) -> dict:
    """merge=True with a `speakers` subset rebuilds only those speakers and keeps every other entry of voices.json."""
    use_lang(lang)
    lines_path = Path(lines_path or V.lines)
    out_dir = Path(out_dir or V.out_dir)
    audio_dir = Path(V.audio_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(lines_path)
    if not rows:
        raise SystemExit(f"no lines in {lines_path}")
    train = {int(e) for e in CFG.split.train_episodes}
    lines = [r for r in rows if int(r["ep"]) in train]
    dropped = len(rows) - len(lines)
    if dropped:
        log(f"split rule: ignored {dropped} lines from non-train episodes")
    eps = {int(r["ep"]) for r in lines}
    stems = {ep: _stem_for(ep, audio_dir) for ep in sorted(eps)}
    sig = sha({"lines": file_sig(lines_path), "cfg": OmegaConf.to_container(V),
               "stems": {ep: (file_sig(p) if p else None) for ep, (p, _) in stems.items()},
               "speakers": speakers, "transcribe": transcribe, "v": 2, **({"lang": LANG} if LANG != "ja" else {})})
    index_path = out_dir / "voices.json"
    old = read_json(index_path, default=None)
    if old and old.get("sig") == sig and not force:
        log(f"voice bank up to date ({index_path})")
        return old
    cands = build_candidates(lines, _load_other_intervals(eps))
    counts: dict[str, int] = {}
    secs: dict[str, float] = {}
    for r in lines:
        s = str(r.get("speaker") or "").strip()
        counts[s] = counts.get(s, 0) + 1
        secs[s] = secs.get(s, 0.0) + max(0.0, float(r["end"]) - float(r["start"]))
    names = [s for s in counts if s.lower() not in _SKIP_NAMES and (not speakers or s in speakers)]
    from src.eval.voice_metrics import SpeakerEmbedder, Transcriber
    embedder = SpeakerEmbedder(device)
    transcriber = Transcriber(device) if transcribe else None
    index: dict[str, Any] = {"sig": sig, "lines": str(lines_path), "episodes": sorted(eps),
                             "stems": {f"ep{ep:02d}": kind for ep, (_, kind) in stems.items()},
                             "speakers": {}, "skipped": {}, "built_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    if merge and speakers and old:
        index["speakers"] = {k: v for k, v in old.get("speakers", {}).items() if k not in speakers}
        index["skipped"] = {k: v for k, v in old.get("skipped", {}).items() if k not in speakers}
        index["sig"] = "merged:" + sig
        index["merged_speakers"] = sorted(set(old.get("merged_speakers", [])) | set(speakers))
    if LANG != "ja":
        index["lang"] = LANG
    for spk in sorted(names, key=lambda s: -counts[s]):
        t0 = time.time()
        meta = build_speaker(spk, cands.get(spk, []), embedder, audio_dir, out_dir, counts[spk], secs[spk],
                             transcriber)
        if not meta["refs"]:
            index["skipped"][spk] = {"reason": meta.get("skipped", "no refs"), "n_lines": counts[spk]}
            log(f"{spk}: skipped ({meta.get('skipped')})")
            continue
        index["speakers"][spk] = {"dir": meta["dir"], "refs": [r["file"] for r in meta["refs"]],
                                  "default_refs": meta["default_refs"], "centroid": meta["centroid"],
                                  "n_refs": len(meta["refs"]), "n_lines": counts[spk], "quality": meta["quality"],
                                  "consistency_median": meta["consistency_median"], "stems": meta["stems"],
                                  "line_sec": meta["line_sec"]}
        if LANG == "en":
            by_file = {r["file"]: r for r in meta["refs"]}
            index["speakers"][spk].update(
                ref_texts=[r.get("ref_text", "") for r in meta["refs"]],
                default_ref_texts=[by_file[f].get("ref_text", "") for f in meta["default_refs"]],
                style_refs={st: {"refs": [r["file"] for r in rs], "ref_texts": [r.get("ref_text", "") for r in rs],
                                 "arousal": [r.get("arousal") for r in rs], "f0_med": [r.get("f0_med") for r in rs]}
                            for st, rs in meta.get("style_refs", {}).items()},
                voice_f0=meta.get("voice_f0"),
                consistency_mean_refs=meta["consistency_mean_refs"], ref_sec=meta["ref_sec"], clean_sec=meta["clean_sec"],
                n_clean=meta["n_clean"], n_consistent=meta["n_consistent"], n_candidates=meta["n_candidates"])
        log(f"{spk}: {len(meta['refs'])} refs (quality {meta['quality']}, consistency {meta['consistency_median']}, "
            f"{meta['n_clean']}/{meta['n_candidates']} clean) in {time.time() - t0:.1f}s")
    write_json(index_path, index)
    log(f"voice bank -> {index_path} ({len(index['speakers'])} speakers, {len(index['skipped'])} skipped)")
    return index


def refs_for(speaker: str, voices_json: str | Path | None = None, lang: str = "ja") -> list[str]:
    """Default ordered reference list of a character ([] if unknown) - what the pipeline puts into `ref_wavs`.
    lang="en" reads the English bank (use src.generate.tts_qwen.voice_refs_en to get the transcripts as well)."""
    out_dir = T.voice_bank.out_dir if lang == "ja" else CFG.tts_qwen.voice_bank.out_dir
    idx = read_json(voices_json or Path(out_dir) / "voices.json", default={}) or {}
    return list((idx.get("speakers", {}).get(speaker) or {}).get("default_refs", []))


def main() -> None:
    ap = argparse.ArgumentParser(description="Build per-character reference sets for voice cloning")
    ap.add_argument("--lines", default=None, help="lines_named.jsonl (default: voice_bank.lines)")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--speakers", default=None, help="comma-separated subset")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--transcribe", action="store_true",
                    help="ja: add a Japanese transcript (text_ja) to every ref; en: re-transcribe every written clip (ref_text)")
    ap.add_argument("--lang", default="ja", choices=["ja", "en"], help="ja = Japanese bank (default), en = English dub bank")
    ap.add_argument("--merge", action="store_true", help="with --speakers: rebuild only those, keep the other entries")
    ap.add_argument("--backfill-f0", action="store_true", help="en: only add voice_f0 to an existing bank")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("OMP_NUM_THREADS", str(a.threads))
    try:
        import torch
        torch.set_num_threads(a.threads)
    except Exception:
        pass
    if a.backfill_f0:
        print(backfill_voice_f0(a.out_dir))
        return
    build(a.lines, a.out_dir, a.speakers.split(",") if a.speakers else None, a.device, a.transcribe, a.force, a.lang,
          a.merge)


if __name__ == "__main__":
    main()
