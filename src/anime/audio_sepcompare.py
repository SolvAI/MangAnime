"""Compare vocal-separation candidates on dialogue-over-music excerpts WITHOUT ground truth (brief §3b).

For every candidate model (CFG.audio.separation.candidates) this script separates
  (a) real excerpts: N seconds of dialogue-over-music picked automatically from the given episodes, and
  (b) one synthetic remix: "dry" dialogue lines (no music around/under them) + music-only spans from season-1 train episodes,
      mixed at a known speech-to-music ratio, so a true SDR can be computed against pseudo-references,
and reports these SDR proxies:
  syn_sdr_voc / syn_sdr_inst   SDR (dB) of the stems on the synthetic remix (higher = better)
  bleed_db                     vocal-stem energy in voice-free frames outside subtitle-dialogue spans (no subtitle line
                               within 0.5 s, no VAD speech) relative to inside dialogue (lower = less bleed)
  gap_supp_db                  vocal-stem energy in those voice-free frames relative to the MIX energy there
                               (lower = more music removed)
  asr_logprob / asr_wordprob   faster-whisper confidence on the vocal stem (higher = cleaner speech)
  cer_others                   mean character error rate of this stem's transcript against the other candidates'
                               transcripts (lower = more "stable" / consensual transcript)
  line_cov                     share of subtitle dialogue lines with at least one ASR word inside
  inst_speech                  characters Whisper still hears in the INSTRUMENTAL stem / characters in the vocal stem
                               (lower = less speech left in the music stem)
  residual_db                  ||mix - (vocals + instrumental)||^2 / ||mix||^2 (sanity check; ~quantisation noise)
  rtf                          realtime factor of the separation
  peak_vram_mb                 peak GPU memory of the process (nvidia-smi) while separating, under the VRAM budget
Selection rule (pre-registered, see docs/audio.md): mean rank over
  [syn_sdr_voc, syn_sdr_inst, bleed_db, asr_logprob, cer_others, inst_speech]; ties -> higher syn_sdr_voc.

Outputs: runs/audio_sep_compare/<name>/{excerpts/, <model>/*.wav, results.json, results.md}; with --write-choice also
data/interim/anime/separation_choice.json (picked up by src.anime.audio_separate).

    scripts/gpu_run.sh 8 $KOMI_WORK/venvs/audio/bin/python -m src.anime.audio_sepcompare \
        --episodes 5,16 --dur 180 --synthetic-episodes 2-11 --synthetic-dur 90 --name full --write-choice
"""
from __future__ import annotations

import argparse
import gc
import re
import time
import unicodedata
from pathlib import Path

import numpy as np

from src.anime.audio_common import (ACFG, TRAIN_EPS, dialogue_events, ep_dir, ep_name, frame_rms_db, load_subs, log,
                                    merge_spans, oped_spans, parse_episodes, read_wav, song_spans, spans_mask, wav_info,
                                    write_wav_atomic)
from src.common.config import CFG
from src.common.io import read_json, sha, write_json

SCFG = ACFG.separation
RATE = 50.0  # analysis frame rate (Hz)
CHOICE_FILE = Path(CFG.paths.interim) / "anime" / "separation_choice.json"


# ----------------------------------------------------------------------------- excerpt selection
_VAD_CACHE: dict[int, list[tuple[float, float]]] = {}


def voiced_spans(ep: int) -> list[tuple[float, float]]:
    """Voice activity of the whole episode mix (Silero VAD, ~3 s per episode on CPU): subtitled dialogue AND unsubtitled
    voices / singing. Used to keep voiced material out of the "music only" frames of the bleed proxy."""
    if ep not in _VAD_CACHE:
        from faster_whisper.vad import VadOptions, get_speech_timestamps

        x, sr = read_wav(ep_dir(ep) / "audio_jpn_16k.wav", mono=True)
        ts = get_speech_timestamps(x, VadOptions(min_silence_duration_ms=300, speech_pad_ms=200))
        _VAD_CACHE[ep] = [(t["start"] / sr, t["end"] / sr) for t in ts]
    return _VAD_CACHE[ep]


def _episode_frames(ep: int):
    x, sr = read_wav(ep_dir(ep) / "audio_jpn_16k.wav", mono=True)
    dbf, rate = frame_rms_db(x, sr, hop_s=1.0 / RATE, win_s=2.0 / RATE)
    subs = load_subs(ep)
    dlg = [(e["start"], e["end"]) for _, e in dialogue_events(subs)]
    songs = song_spans(subs, pad=5.0) + oped_spans(ep)
    return dbf, rate, subs, dlg, songs


def pick_excerpt(ep: int, dur: float) -> dict:
    """Window with 35-75 % dialogue coverage whose voice-free gaps (no subtitle line, no VAD speech) are the loudest,
    i.e. music keeps playing between the lines."""
    dbf, rate, subs, dlg, songs = _episode_frames(ep)
    n = len(dbf)
    m_dlg = spans_mask(dlg, n, rate)
    m_near = spans_mask(dlg, n, rate, pad=0.5) | spans_mask(voiced_spans(ep), n, rate)   # subtitled or voiced
    m_song = spans_mask(songs, n, rate)
    w = int(dur * rate)
    lo, hi = int(180 * rate), n - int(240 * rate) - w
    best = None
    for a in range(lo, max(lo + 1, hi), int(5 * rate)):
        b = a + w
        if m_song[a:b].mean() > 0.0:
            continue
        cov = m_dlg[a:b].mean()
        gap = ~m_near[a:b]
        if not (0.35 <= cov <= 0.75) or gap.sum() < min(20.0, dur * 0.15) * rate:
            continue
        score = float(np.percentile(dbf[a:b][gap], 20))
        if best is None or score > best["gap_p20_db"]:
            best = {"ep": ep, "start": round(a / rate, 2), "dur": dur, "dialogue_cov": round(float(cov), 3), "gap_p20_db": round(score, 2)}
    if best is None:  # fall back to the densest dialogue window
        a = lo
        best = {"ep": ep, "start": round(a / rate, 2), "dur": dur, "dialogue_cov": round(float(m_dlg[a:a + w].mean()), 3), "gap_p20_db": None}
    return best


def _speech_ratio(ep: int, a: float, b: float) -> float:
    """Share of [a, b] that Silero VAD (shipped with faster-whisper) marks as voice, on the 16 kHz mix."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    x, _ = read_wav(ep_dir(ep) / "audio_jpn_16k.wav", start=a, dur=b - a, mono=True)
    if len(x) < 1600:
        return 1.0
    ts = get_speech_timestamps(x, VadOptions(min_silence_duration_ms=200, speech_pad_ms=50))
    return float(sum(t["end"] - t["start"] for t in ts) / len(x))


def build_synthetic(eps: list[int], dur: float, smr_db: float = 5.0, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Pseudo ground truth: dry dialogue lines + music-only spans (train episodes) remixed at a known ratio.

    Returns (speech[T,2], music[T,2], mix[T,2], info). speech/music are the references the stems are scored against.
    """
    rng = np.random.default_rng(seed)
    sr = int(SCFG.sample_rate)
    lines, music, n_voiced = [], [], 0
    for ep in eps:
        dbf, rate, subs, dlg, songs = _episode_frames(ep)
        n = len(dbf)
        m_dlg = spans_mask(dlg, n, rate, pad=0.1)
        m_far = spans_mask(dlg, n, rate, pad=1.0) | spans_mask(songs, n, rate)
        # dry lines: quiet context before/after AND quiet pauses inside the line
        for s, e in dlg:
            if not (1.0 <= e - s <= 8.0) or any(a <= s <= b or a <= e <= b for a, b in songs):
                continue
            a0, a1, b0, b1 = int((s - 1.5) * rate), int((s - 0.2) * rate), int((e + 0.2) * rate), int((e + 1.5) * rate)
            if a0 < 0 or b1 >= n:
                continue
            ctx = np.concatenate([dbf[a0:a1][~m_dlg[a0:a1]], dbf[b0:b1][~m_dlg[b0:b1]]])
            inside = dbf[int(s * rate):int(e * rate)]
            if len(ctx) < 0.5 * rate or len(inside) < 10:
                continue
            ctx_db, line_db, floor_db = np.percentile(ctx, 90), np.percentile(inside, 90), np.percentile(inside, 5)
            if ctx_db < -50.0 and line_db - ctx_db > 28.0 and floor_db < -48.0:
                lines.append((ep, s, e, float(line_db)))
        # music-only spans: far from any dialogue, continuously sounding
        free = ~m_far
        i = 0
        while i < n:
            if not free[i]:
                i += 1
                continue
            j = i
            while j < n and free[j]:
                j += 1
            if (j - i) / rate >= 6.0 and np.percentile(dbf[i:j], 20) > -40.0:
                a_, b_ = i / rate, min(j / rate, i / rate + 15.0)
                if _speech_ratio(ep, a_, b_) < 0.05:  # no unsubtitled voices / singing in the "music" reference
                    music.append((ep, a_, b_))
                else:
                    n_voiced += 1
            i = j
    if not lines or not music:
        raise RuntimeError(f"synthetic set: not enough material (dry lines={len(lines)}, music spans={len(music)})")
    rng.shuffle(lines)
    rng.shuffle(music)
    T = int(dur * sr)
    fade = int(0.02 * sr)
    ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)[:, None]

    def cut(ep, a, b):
        x, _ = read_wav(ep_dir(ep) / "audio_jpn.wav", start=max(0.0, a), dur=b - a)
        if x.shape[1] == 1:
            x = np.repeat(x, 2, axis=1)
        if len(x) > 2 * fade:
            x[:fade] *= ramp
            x[-fade:] *= ramp[::-1]
        return x

    speech = np.zeros((T, 2), np.float32)
    pos, used_l = int(0.3 * sr), []
    for ep, s, e, _ in lines:
        x = cut(ep, s - 0.2, e + 0.2)
        if pos + len(x) > T:
            break
        speech[pos:pos + len(x)] = x
        used_l.append([ep, round(s, 2), round(e, 2)])
        pos += len(x) + int(0.3 * sr)
    T = min(T, pos)  # if the dry lines run out early the remix is shortened instead of padded with music-only time
    speech = speech[:T]
    mus = np.zeros((T, 2), np.float32)
    pos, used_m, k = 0, [], 0
    while pos < T and music:
        ep, a, b = music[k % len(music)]
        k += 1
        x = cut(ep, a, b)
        m = min(len(x), T - pos)
        mus[pos:pos + m] = x[:m]
        used_m.append([ep, round(a, 2), round(b, 2)])
        pos += m
    act = np.abs(speech).max(axis=1) > 1e-4
    ps = float(np.mean(speech[act] ** 2)) if act.any() else 1e-9
    pm = float(np.mean(mus ** 2)) + 1e-12
    g = np.sqrt(ps / pm / (10 ** (smr_db / 10)))
    mus = (mus * g).astype(np.float32)
    mix = speech + mus
    peak = float(np.abs(mix).max())
    if peak > 0.98:
        k_ = 0.98 / peak
        speech, mus, mix = speech * k_, mus * k_, mix * k_
    info = {"episodes": eps, "dur": round(T / sr, 2), "smr_db": smr_db, "n_dry_lines_found": len(lines), "n_music_spans_found": len(music),
            "n_music_spans_rejected_voiced": n_voiced,
            "speech_lines": used_l, "music_spans": used_m, "speech_seconds": round(float(act.sum()) / sr, 1)}
    return speech, mus, mix, info


# ----------------------------------------------------------------------------- metrics
def sdr(ref: np.ndarray, est: np.ndarray) -> float:
    n = min(len(ref), len(est))
    r, e = ref[:n].astype(np.float64), est[:n].astype(np.float64)
    return float(10 * np.log10((np.sum(r ** 2) + 1e-12) / (np.sum((r - e) ** 2) + 1e-12)))


def si_sdr(ref: np.ndarray, est: np.ndarray) -> float:
    n = min(len(ref), len(est))
    r, e = ref[:n].astype(np.float64).ravel(), est[:n].astype(np.float64).ravel()
    a = np.dot(e, r) / (np.dot(r, r) + 1e-12)
    t = a * r
    return float(10 * np.log10((np.sum(t ** 2) + 1e-12) / (np.sum((e - t) ** 2) + 1e-12)))


def bleed_metrics(voc: np.ndarray, mix: np.ndarray, sr: int, subs: list[dict], t0: float,
                  voiced: list[tuple[float, float]] | None = None) -> dict:
    """Energy of the vocal stem inside subtitle-dialogue spans vs. in voice-free frames outside them.

    `voiced` = VAD speech spans of the mix (episode times): frames with unsubtitled voices / singing are not counted as
    "outside", so the number measures music/SFX bleed and not vocalisations the subtitles skip."""
    dlg = [(e["start"] - t0, e["end"] - t0) for _, e in dialogue_events(subs)]
    other = [(e["start"] - t0, e["end"] - t0) for e in subs if e.get("kind") == "song"]
    pv, rate = frame_rms_db(voc.mean(axis=1), sr, hop_s=1.0 / RATE, win_s=2.0 / RATE)
    pm, _ = frame_rms_db(mix.mean(axis=1), sr, hop_s=1.0 / RATE, win_s=2.0 / RATE)
    n = min(len(pv), len(pm))
    pv, pm = 10 ** (pv[:n] / 10), 10 ** (pm[:n] / 10)
    inside = spans_mask(dlg, n, rate)
    outside = ~spans_mask(dlg, n, rate, pad=0.5) & ~spans_mask(other, n, rate, pad=0.5)
    if voiced:
        outside &= ~spans_mask([(a - t0, b - t0) for a, b in voiced], n, rate, pad=0.1)
    if inside.sum() < rate or outside.sum() < rate:
        return {"bleed_db": None, "gap_supp_db": None, "in_ratio_db": None}
    return {
        "bleed_db": round(float(10 * np.log10(pv[outside].mean() / (pv[inside].mean() + 1e-12) + 1e-12)), 2),
        "gap_supp_db": round(float(10 * np.log10(pv[outside].mean() / (pm[outside].mean() + 1e-12) + 1e-12)), 2),
        "in_ratio_db": round(float(10 * np.log10(pv[inside].mean() / (pm[inside].mean() + 1e-12) + 1e-12)), 2),
        "outside_seconds": round(float(outside.sum() / rate), 1), "inside_seconds": round(float(inside.sum() / rate), 1),
    }


_PUNCT = re.compile(r"[\s、。，．,.!！?？…‥・「」『』（）()\[\]【】〜~ー―—\-–:：;；\"'“”‘’♪]+")


def norm_text(t: str) -> str:
    return _PUNCT.sub("", unicodedata.normalize("NFKC", t)).lower()


def cer(ref: str, hyp: str) -> float:
    if not ref:
        return 0.0 if not hyp else 1.0
    try:
        from rapidfuzz.distance import Levenshtein

        d = Levenshtein.distance(ref, hyp)
    except Exception:  # tiny pure-python fallback
        prev = list(range(len(hyp) + 1))
        for i, c in enumerate(ref, 1):
            cur = [i]
            for j, h in enumerate(hyp, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (c != h)))
            prev = cur
        d = prev[-1]
    return d / len(ref)


class Asr:
    """faster-whisper wrapper used for the confidence / stability proxies."""

    def __init__(self, device: str, threads: int = 8):
        from src.anime.audio_asr import load_model, transcribe_file

        self.model = load_model(device=device, cpu_threads=threads)
        self.compute_type = getattr(self.model.model, "compute_type", "?")
        self._tx = transcribe_file

    def run(self, path: Path) -> dict:
        segs = self._tx(self.model, path)
        words = [w for s in segs for w in s["words"]]
        text = "".join(s["text"] for s in segs)
        conf_segs = [s for s in segs if s.get("avg_logprob", 0.0) > -1.0 and s.get("no_speech_prob", 0.0) < 0.6]
        dur = sum(s["end"] - s["start"] for s in segs) or 1e-9
        return {
            "text": text, "norm": norm_text(text), "n_chars": len(norm_text(text)),
            "n_chars_conf": len(norm_text("".join(s["text"] for s in conf_segs))),
            "asr_wordprob": round(float(np.mean([w["prob"] for w in words])), 4) if words else None,
            "asr_logprob": round(float(sum(s.get("avg_logprob", 0.0) * (s["end"] - s["start"]) for s in segs) / dur), 4) if segs else None,
            "words": [(w["start"], w["end"]) for w in words],
        }


def line_coverage(words: list[tuple[float, float]], subs: list[dict], t0: float, dur: float) -> float | None:
    mids = np.array([(a + b) / 2 for a, b in words]) if words else np.zeros(0)
    tot = hit = 0
    for _, e in dialogue_events(subs):
        s, en = e["start"] - t0, e["end"] - t0
        if s < 0 or en > dur or en - s < 0.4:
            continue
        tot += 1
        hit += bool(((mids >= s - 0.3) & (mids <= en + 0.3)).any())
    return round(hit / tot, 3) if tot else None


# ----------------------------------------------------------------------------- main
RANK_KEYS = [("syn_sdr_voc", True), ("syn_sdr_inst", True), ("bleed_db", False), ("asr_logprob", True),
             ("cer_others", False), ("inst_speech", False)]  # (metric, higher_is_better)


def choose(summary: dict[str, dict]) -> tuple[str, dict[str, float]]:
    models = list(summary)
    ranks = {m: [] for m in models}
    for key, hib in RANK_KEYS:
        vals = {m: summary[m].get(key) for m in models}
        if any(v is None for v in vals.values()):
            continue
        order = sorted(models, key=lambda m: -vals[m] if hib else vals[m])
        for r, m in enumerate(order):
            ranks[m].append(r + 1)
    mean_rank = {m: (float(np.mean(r)) if r else 99.0) for m, r in ranks.items()}
    best = sorted(models, key=lambda m: (mean_rank[m], -(summary[m].get("syn_sdr_voc") or -99)))[0]
    return best, mean_rank


def _asr_proxies(items: list[dict], keys: list[str], results: dict, out: Path, device: str) -> dict:
    from src.anime.audio_common import VramMonitor

    with VramMonitor() as mon:
        asr = Asr(device)
        t0 = time.time()
        _asr_proxies_run(asr, items, keys, results, out)
        dt = time.time() - t0
    audio_s = sum(it["dur"] for it in items if it["kind"] == "real") * (1 + 2 * len(keys))
    return {"peak_vram_mb": mon.peak_mb or None, "compute_type": asr.compute_type, "seconds": round(dt, 1),
            "audio_seconds": round(audio_s, 1), "rtf": round(audio_s / max(dt, 1e-6), 1)}


def _asr_proxies_run(asr, items: list[dict], keys: list[str], results: dict, out: Path) -> None:
    tx: dict[str, dict[str, dict]] = {}
    for it in items:
        if it["kind"] != "real":
            continue
        subs = load_subs(it["ep"])
        tx[it["id"]] = {"_mix": asr.run(it["path"])}
        for k in keys:
            a = asr.run(out / k / f"{it['id']}_vocals.wav")
            b = asr.run(out / k / f"{it['id']}_instrumental.wav")
            tx[it["id"]][k] = a
            row = results[k]["items"][it["id"]]
            row.update({"asr_wordprob": a["asr_wordprob"], "asr_logprob": a["asr_logprob"], "asr_chars": a["n_chars"],
                        "line_cov": line_coverage(a["words"], subs, it["start"], it["dur"]),
                        "inst_speech": round(b["n_chars_conf"] / max(a["n_chars"], 1), 4),
                        "cer_vs_mix": round(cer(tx[it["id"]]["_mix"]["norm"], a["norm"]), 4)})
            log(f"ASR {k} {it['id']}: wordprob={a['asr_wordprob']} logprob={a['asr_logprob']} chars={a['n_chars']} inst_speech={row['inst_speech']}")
        for k in keys:
            others = [cer(tx[it["id"]][o]["norm"], tx[it["id"]][k]["norm"]) for o in keys if o != k]
            results[k]["items"][it["id"]]["cer_others"] = round(float(np.mean(others)), 4) if others else None
    write_json(out / "transcripts.json", {i: {k: v["text"] for k, v in d.items()} for i, d in tx.items()})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", default="5,16", help="train episodes to take the real excerpts from")
    ap.add_argument("--dur", type=float, default=180.0, help="seconds per real excerpt")
    ap.add_argument("--synthetic-episodes", default="2-11", help="season-1 train episodes (song spans known) for the remix")
    ap.add_argument("--synthetic-dur", type=float, default=90.0, help="0 disables the synthetic remix")
    ap.add_argument("--models", default=None, help="comma list of candidate keys (default: all in config)")
    ap.add_argument("--overlap", type=int, default=None)
    ap.add_argument("--overlap-check", type=int, default=8,
                    help="re-run the winner on the synthetic remix with this overlap to quantify the overlap setting (0 = skip)")
    ap.add_argument("--segment-check", default="512,256",
                    help="also run the winner on the synthetic remix with these segment sizes (dim_t) to get SDR / peak VRAM "
                         "per setting ('' = skip)")
    ap.add_argument("--name", default="full")
    ap.add_argument("--device", default="auto", help="auto|cuda|cpu (ASR; the separator follows CUDA_VISIBLE_DEVICES)")
    ap.add_argument("--no-asr", action="store_true")
    ap.add_argument("--write-choice", action="store_true", help=f"write the winner to {CHOICE_FILE}")
    args = ap.parse_args()

    import torch

    from src.anime.audio_separate import BudgetedSeparator, residual_db
    from src.anime.audio_common import VramMonitor, vram_budget_gb

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    cands = dict(SCFG.candidates)
    keys = [k.strip() for k in args.models.split(",")] if args.models else list(cands)
    out = Path(CFG.paths.runs) / "audio_sep_compare" / args.name
    (out / "excerpts").mkdir(parents=True, exist_ok=True)
    sr = int(SCFG.sample_rate)

    # ---- inputs
    eps = parse_episodes(args.episodes)
    bad = [e for e in eps + parse_episodes(args.synthetic_episodes) if e not in TRAIN_EPS]
    if bad:
        raise SystemExit(f"comparison excerpts must come from TRAIN episodes, got {bad}")
    items = []  # {"id","path","kind","ep","start","dur"}
    for ep in eps:
        ex = pick_excerpt(ep, args.dur)
        p = out / "excerpts" / f"{ep_name(ep)}_{int(ex['start']):04d}_{int(args.dur)}s.wav"
        if not p.exists():
            x, _ = read_wav(ep_dir(ep) / "audio_jpn.wav", start=ex["start"], dur=args.dur)
            write_wav_atomic(p, x, sr)
        items.append({"id": p.stem, "path": p, "kind": "real", **ex})
        log(f"real excerpt {p.name}: {ex}")
    syn = None
    if args.synthetic_dur > 0:
        sp, mu, mx, info = build_synthetic(parse_episodes(args.synthetic_episodes), args.synthetic_dur)
        sig = sha({"speech": info["speech_lines"], "music": info["music_spans"], "dur": info["dur"], "smr": info["smr_db"]})
        sig_file = out / "excerpts" / "synthetic.sig"
        if not sig_file.exists() or sig_file.read_text() != sig:  # the remix changed: stems of an older remix are stale
            for old in out.glob("*/synthetic_mix_*.wav"):
                old.unlink()
            sig_file.write_text(sig)
        for nm, arr in (("synthetic_speech_ref", sp), ("synthetic_music_ref", mu), ("synthetic_mix", mx)):
            write_wav_atomic(out / "excerpts" / f"{nm}.wav", arr, sr)
        # references re-read from the PCM16 files so quantisation is identical for every model
        syn = {"speech": read_wav(out / "excerpts" / "synthetic_speech_ref.wav")[0], "music": read_wav(out / "excerpts" / "synthetic_music_ref.wav")[0], "info": info}
        items.append({"id": "synthetic_mix", "path": out / "excerpts" / "synthetic_mix.wav", "kind": "synthetic", "dur": info["dur"]})
        log(f"synthetic remix: {info['speech_seconds']}s speech from {len(info['speech_lines'])} dry lines "
            f"(found {info['n_dry_lines_found']}), {len(info['music_spans'])} music spans (found {info['n_music_spans_found']})")

    # ---- separation (every candidate under the SAME VRAM budget as the batch: env AUDIO_VRAM_GB / config)
    budget = vram_budget_gb()
    results: dict[str, dict] = {k: {"model": cands[k], "items": {}} for k in keys}
    for k in keys:
        t0 = time.time()
        bs = BudgetedSeparator(cands[k], output_dir=out / "_tmp", overlap=args.overlap, vram_gb=budget, remember=False)
        peak, segs = 0, set()
        try:
            for it in items:
                v, i = out / k / f"{it['id']}_vocals.wav", out / k / f"{it['id']}_instrumental.wav"
                row: dict = {}
                if not (v.exists() and i.exists()):
                    r = bs.separate(it["path"], v, i, tag=f"{k}_{it['id']}")
                    row.update({"sep_seconds": r["seconds"], "rtf": round(it["dur"] / max(r["seconds"], 1e-6), 2), "segment": r["segment"],
                                "peak_vram_mb": r.get("peak_vram_mb"), "torch_peak_reserved_mb": r.get("torch_peak_reserved_mb")})
                    peak = max(peak, r.get("peak_vram_mb") or 0)
                    segs.add(r["segment"])
                row["residual_db"] = round(residual_db(it["path"], v, i, start=0.0, dur=it["dur"]), 2)
                results[k]["items"][it["id"]] = row
                log(f"{k} {it['id']}: {row}")
        except Exception as e:  # a candidate that cannot be loaded / does not fit must not kill the comparison
            log(f"{k}: FAILED {e!r}")
            results[k]["error"] = repr(e)[:300]
        results[k].update({"seconds_total": round(time.time() - t0, 1), "device": str(getattr(bs.sep, "torch_device", "?")),
                           "peak_vram_mb": peak or None, "segment": sorted(x for x in segs if x) or None, "dim_t_default": bs.dim_t,
                           "oom_fallbacks": bs.oom_events})
        bs.close()
        del bs
        gc.collect()

    keys = [k for k in keys if "error" not in results[k]]
    # ---- signal metrics
    for k in keys:
        for it in items:
            row = results[k]["items"][it["id"]]
            voc, _ = read_wav(out / k / f"{it['id']}_vocals.wav")
            ins, _ = read_wav(out / k / f"{it['id']}_instrumental.wav")
            if it["kind"] == "real":
                mix, _ = read_wav(it["path"])
                row.update(bleed_metrics(voc, mix, sr, load_subs(it["ep"]), it["start"], voiced_spans(it["ep"])))
            else:
                row.update({"syn_sdr_voc": round(sdr(syn["speech"], voc), 2), "syn_sdr_inst": round(sdr(syn["music"], ins), 2),
                            "syn_sisdr_voc": round(si_sdr(syn["speech"], voc), 2), "syn_sisdr_inst": round(si_sdr(syn["music"], ins), 2)})

    # ---- ASR proxies (real excerpts only); a failure here must not lose the signal metrics / the choice
    asr_stats = None
    if not args.no_asr:
        try:
            asr_stats = _asr_proxies(items, keys, results, out, device)
            log(f"ASR proxies: {asr_stats}")
        except Exception as e:
            log(f"ASR proxies FAILED ({e!r}); the choice will use the signal metrics only")

    # ---- summary (mean over items) + choice
    cols = ["syn_sdr_voc", "syn_sdr_inst", "syn_sisdr_voc", "bleed_db", "gap_supp_db", "asr_logprob", "asr_wordprob", "cer_others",
            "cer_vs_mix", "line_cov", "inst_speech", "residual_db", "rtf"]
    summary = {}
    for k in keys:
        s = {}
        for c in cols:
            vals = [r[c] for r in results[k]["items"].values() if r.get(c) is not None]
            s[c] = round(float(np.mean(vals)), 4) if vals else None
        s["peak_vram_mb"] = results[k].get("peak_vram_mb")
        s["segment"] = results[k].get("segment")
        summary[k] = s
    best, mean_rank = choose(summary) if keys else (None, {})
    # ---- what do the speed / memory settings cost? (winner only, synthetic remix only)
    ov_used = int(args.overlap or SCFG.overlap)
    ov_check, seg_check = None, []
    syn_it = next((i for i in items if i["kind"] == "synthetic"), None)
    if best and syn_it is not None and "htdemucs" not in cands[best]:
        def _variant(tag: str, overlap: int, segment: int | None) -> dict | None:
            try:
                b2 = BudgetedSeparator(cands[best], output_dir=out / "_tmp", overlap=overlap, vram_gb=budget, remember=False)
                b2.segment = segment
                v, i_ = out / f"{best}_{tag}" / f"{syn_it['id']}_vocals.wav", out / f"{best}_{tag}" / f"{syn_it['id']}_instrumental.wav"
                r = b2.separate(syn_it["path"], v, i_, tag=f"{best}_{tag}")
                res = {"overlap": overlap, "segment": r["segment"], "syn_sdr_voc": round(sdr(syn["speech"], read_wav(v)[0]), 2),
                       "syn_sdr_inst": round(sdr(syn["music"], read_wav(i_)[0]), 2), "rtf": round(syn_it["dur"] / max(r["seconds"], 1e-6), 2),
                       "peak_vram_mb": r.get("peak_vram_mb"), "torch_peak_reserved_mb": r.get("torch_peak_reserved_mb"),
                       "oom_fallbacks": r.get("oom_fallbacks")}
                b2.close()
                return res
            except Exception as e:
                log(f"variant {tag} failed: {e!r}")
                return None

        used_row = results[best]["items"][syn_it["id"]]
        base = {"overlap": ov_used, "segment": used_row.get("segment"), "syn_sdr_voc": summary[best]["syn_sdr_voc"],
                "syn_sdr_inst": summary[best]["syn_sdr_inst"], "rtf": used_row.get("rtf"), "peak_vram_mb": used_row.get("peak_vram_mb")}
        if args.overlap_check and args.overlap_check != ov_used:
            r = _variant(f"overlap{args.overlap_check}", args.overlap_check, None)
            if r:
                ov_check = {"model": best, **r, "vs_overlap": ov_used, "syn_sdr_voc_at_used": base["syn_sdr_voc"],
                            "syn_sdr_inst_at_used": base["syn_sdr_inst"], "rtf_at_used": base["rtf"]}
                log(f"overlap check: {ov_check}")
        if args.segment_check:
            seg_check.append({"setting": "used in the comparison", **base})
            for seg in [int(x) for x in str(args.segment_check).split(",") if x.strip()]:
                if seg == base["segment"]:
                    continue
                r = _variant(f"segment{seg}", ov_used, seg)
                if r:
                    seg_check.append({"setting": f"segment {seg}", **r})
            log(f"segment / VRAM check: {seg_check}")
    doc = {"name": args.name, "overlap_check": ov_check, "segment_check": seg_check, "asr_stats": asr_stats, "vram_budget_gb": budget, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "device": device, "overlap": int(args.overlap or SCFG.overlap),
           "excerpts": [{k: (str(v) if isinstance(v, Path) else v) for k, v in it.items()} for it in items],
           "synthetic": syn["info"] if syn else None, "results": results, "summary": summary, "mean_rank": mean_rank,
           "rank_keys": [k for k, _ in RANK_KEYS], "choice": best, "choice_model": cands.get(best) if best else None}
    write_json(out / "results.json", doc)
    lines = [f"Separation comparison `{args.name}` ({doc['created']}, device={device}, overlap={doc['overlap']})", "",
             "| model | " + " | ".join(cols) + " | mean_rank |", "|---|" + "---|" * (len(cols) + 1)]
    for k in keys:
        lines.append(f"| {k} | " + " | ".join("—" if summary[k][c] is None else f"{summary[k][c]:.3g}" if abs(summary[k][c]) < 10 else f"{summary[k][c]:.1f}" for c in cols)
                     + f" | {mean_rank[k]:.2f} |")
    lines += ["", f"choice: **{best}** ({cands.get(best)})"]
    if ov_check:
        lines += ["", f"overlap check ({best}, synthetic remix): overlap {ov_check['overlap']} -> syn_sdr_voc {ov_check['syn_sdr_voc']} / syn_sdr_inst "
                      f"{ov_check['syn_sdr_inst']} dB at {ov_check['rtf']}x realtime; overlap {ov_used} (used) -> {ov_check['syn_sdr_voc_at_used']} / "
                      f"{ov_check['syn_sdr_inst_at_used']} dB at {ov_check['rtf_at_used'] or '?'}x realtime"]
    (out / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    lines += ["", f"VRAM budget {budget:g} GB. Per model: " + "; ".join(
        f"{k}: segment {results[k].get('segment')}, peak {results[k].get('peak_vram_mb')} MiB" for k in keys)]
    if seg_check:
        lines += ["", "| winner setting | overlap | segment | syn_sdr_voc | syn_sdr_inst | rtf | peak VRAM MiB |", "|---|---|---|---|---|---|---|"]
        lines += [f"| {r['setting']} | {r.get('overlap')} | {r.get('segment')} | {r.get('syn_sdr_voc')} | {r.get('syn_sdr_inst')} | {r.get('rtf')} | {r.get('peak_vram_mb')} |"
                  for r in seg_check]
    if asr_stats:
        lines += ["", f"ASR proxies: {asr_stats}"]
    (out / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[-8:]), flush=True)
    if args.write_choice and best:
        write_json(CHOICE_FILE, {"model": cands[best], "key": best, "from": str(out / "results.json"), "mean_rank": mean_rank,
                                 "vram_budget_gb": budget, "segment": results[best].get("segment"),
                                 "peak_vram_mb": results[best].get("peak_vram_mb"), "created": doc["created"]})
        log(f"wrote {CHOICE_FILE}")


if __name__ == "__main__":
    main()
