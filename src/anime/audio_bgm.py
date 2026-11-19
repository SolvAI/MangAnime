"""BGM cue library from the instrumental stems of the TRAIN episodes (retrieval only, no music generation).

Stages (all idempotent; held-out episodes are refused):
  1. cut     per episode: continuous music regions of the instrumental stem -> cues of 8-40 s
             (OP/ED/insert songs skipped: `kind == "song"` subtitle spans, data/processed/anime/oped.json, and blocks where
              the vocal stem is active without dialogue subtitles = singing), loudness-normalised wavs
             -> data/processed/bgm/cues/<epNN>_<start>.wav, data/processed/bgm/episodes/epNN.json
  2. embed   CLAP (laion/clap-htsat-unfused, local copy) audio embeddings, cached per cue -> data/processed/bgm/clap_cache/<model>/
  3. index   zero-shot music/non-music filter + mood tags -> data/processed/bgm/cues.jsonl + clap.npy (+ rejected.jsonl)

cues.jsonl row: {"id","ep","start","end","path","dur","rms","emb_file","emb_idx","lufs_in","gain_db","dlg_ratio",
                 "voc_ratio","music_score","music_margin","tags","bpm"}   (path/emb_file relative to the repo root; row i <-> clap.npy[i])

    . scripts/env.sh
    scripts/gpu_run.sh 3 $KOMI_WORK/venvs/audio/bin/python -m src.anime.audio_bgm --episodes train     # AUDIO_BGM_DEVICE=cpu: no GPU
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np

from src.anime.audio_common import (ACFG, HELDOUT_EPS, VramMonitor, cap_torch_vram, dialogue_events, ep_dir, ep_name, frame_rms_db, load_subs, log,
                                    merge_spans, oped_spans, parse_episodes, read_wav, song_spans, spans_mask, wav_info,
                                    write_wav_atomic)
from src.common.config import CFG
from src.common.io import read_json, write_json, write_jsonl

BC = ACFG.bgm
OUT = Path(BC.out_dir)
CLAP_DIR = Path(CFG.paths.models) / "clap" / str(BC.clap_model).split("/")[-1]
RATE = 50.0
CLAP_SR = 48000
MOODS = ["gentle comedy", "playful and silly", "energetic and upbeat", "heartwarming", "romantic", "nostalgic", "sad and melancholic",
         "tense and suspenseful", "awkward and quirky", "mysterious", "dramatic and epic", "calm and peaceful", "triumphant",
         "lonely", "anxious", "dreamy", "funny and mischievous", "emotional piano", "everyday school life", "ominous"]
MUSIC_PROMPTS = ["instrumental background music", "a soundtrack with melody and harmony", "music"]
NONMUSIC_PROMPTS = ["sound effects and foley", "ambient noise, room tone", "people talking", "silence", "footsteps, rustling and impacts",
                    "nature sounds: wind, rain, cicadas", "crowd noise"]


def rel(p: Path) -> str:
    """Path relative to the repo root (data/ is a symlink into the work area)."""
    return str(p).replace(str(CFG.paths.work) + "/", "")


# ----------------------------------------------------------------------------- 1. cut
def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def _close(mask: np.ndarray, gap: int) -> np.ndarray:
    m = mask.copy()
    for a, b in _runs(~mask):
        if b - a <= gap and a > 0 and b < len(mask):
            m[a:b] = True
    return m


def detect_cues(ep: int) -> tuple[list[dict], dict]:
    d = ep_dir(ep)
    inst, sr = read_wav(d / "instrumental.wav", mono=True)
    voc, _ = read_wav(d / "vocals.wav", mono=True)
    idb, rate = frame_rms_db(inst, sr, hop_s=1.0 / RATE, win_s=2.0 / RATE)
    vdb, _ = frame_rms_db(voc, sr, hop_s=1.0 / RATE, win_s=2.0 / RATE)
    n = min(len(idb), len(vdb))
    idb, vdb = idb[:n], vdb[:n]
    subs = load_subs(ep)
    dlg = [(e["start"], e["end"]) for _, e in dialogue_events(subs)]
    m_dlg = spans_mask(dlg, n, rate)
    # --- songs: subtitle `song` spans + lead-provided OP/ED spans + "sung" blocks (vocal stem active, no dialogue subtitle)
    excl = spans_mask(song_spans(subs, pad=float(BC.song_pad)) + oped_spans(ep, pad=2.0), n, rate)
    v_act = vdb > -40.0
    sung = v_act & ~spans_mask(dlg, n, rate, pad=0.5)
    w = int(20 * rate)
    sm = np.convolve(sung.astype(np.float32), np.ones(w, np.float32) / w, mode="same")
    blocks = merge_spans([(a / rate, b / rate) for a, b in _runs(sm > 0.35)], gap=15.0)
    sung_blocks = [(max(0.0, a - 8.0), b + 8.0) for a, b in blocks if b - a >= 40.0]
    excl |= spans_mask(sung_blocks, n, rate)
    # --- music regions on the instrumental stem
    sound = _close(idb > float(BC.music_floor_db), int(float(BC.max_gap) * rate)) & ~excl
    for a, b in _runs(sound):
        if (b - a) / rate < 1.0:
            sound[a:b] = False
    k = int(0.5 * rate)
    e_s = np.convolve(10 ** (idb / 10), np.ones(k) / k, mode="same")
    lo, hi = float(BC.min_cue), float(BC.max_cue)
    pieces: list[list[int]] = []
    for a, b in _runs(sound):
        if (b - a) / rate < lo:
            continue
        first = len(pieces)
        while (b - a) / rate > hi:
            c0, c1 = a + int(0.6 * hi * rate), a + int(hi * rate)
            c = c0 + int(np.argmin(e_s[c0:c1]))
            pieces.append([a, c])
            a = c
        if (b - a) / rate >= lo:
            pieces.append([a, b])
        elif len(pieces) > first and (b - pieces[-1][0]) / rate <= hi:
            pieces[-1][1] = b
    cues = []
    for a, b in pieces:
        dr, vr = float(m_dlg[a:b].mean()), float(v_act[a:b].mean())
        sr_ = float(sung[a:b].mean())
        if dr > float(BC.max_dialogue_ratio) or sr_ > float(BC.max_sung_ratio):
            continue
        cues.append({"ep": ep, "start": round(float(a / rate), 2), "end": round(float(b / rate), 2), "dur": round(float((b - a) / rate), 2),
                     "dlg_ratio": round(dr, 3), "voc_ratio": round(vr, 3), "sung_ratio": round(sr_, 3)})
    info = {"regions": len(_runs(sound)), "pieces": len(pieces), "kept": len(cues), "sung_blocks": [[round(a, 1), round(b, 1)] for a, b in sung_blocks],
            "song_spans": [[round(a, 1), round(b, 1)] for a, b in song_spans(subs, pad=float(BC.song_pad)) + oped_spans(ep)],
            "sounding_ratio": round(float(sound.mean()), 3)}
    return cues, info


def export_cue(ep: int, cue: dict) -> dict | None:
    """Cut the cue from the 44.1 kHz stereo instrumental stem, loudness-normalise, write wav. Returns the finished row."""
    import pyloudnorm as pyln

    x, sr = read_wav(ep_dir(ep) / "instrumental.wav", start=cue["start"], dur=cue["dur"])
    if len(x) < sr:
        return None
    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
    lufs = float(pyln.Meter(sr).integrated_loudness(x.astype(np.float64)))
    if not np.isfinite(lufs) or rms < 1e-4:
        return None
    gain = float(BC.target_lufs) - lufs
    peak = float(np.abs(x).max()) + 1e-9
    gain = min(gain, float(BC.peak_dbfs) - 20 * np.log10(peak), float(BC.get("max_gain_db", 24.0)))  # peak ceiling, gain cap
    y = x * (10 ** (gain / 20))
    f = int(0.03 * sr)
    ramp = np.linspace(0.0, 1.0, f, dtype=np.float32)[:, None]
    y[:f] *= ramp
    y[-f:] *= ramp[::-1]
    cid = f"{ep_name(ep)}_{cue['start']:07.2f}"
    path = OUT / "cues" / f"{cid}.wav"
    write_wav_atomic(path, y.astype(np.float32), sr)
    return {"id": cid, **cue, "path": rel(path), "rms": round(rms, 5), "rms_db": round(float(20 * np.log10(rms)), 2),
            "lufs_in": round(lufs, 2), "gain_db": round(gain, 2)}


def cut_episode(ep: int, force: bool = False) -> list[dict]:
    meta = OUT / "episodes" / f"{ep_name(ep)}.json"
    if meta.exists() and not force:
        return read_json(meta)["cues"]
    t0 = time.time()
    cands, info = detect_cues(ep)
    rows = [r for r in (export_cue(ep, c) for c in cands) if r is not None]
    write_json(meta, {"episode": ep, "info": info, "cues": rows, "seconds": round(time.time() - t0, 1)})
    log(f"{ep_name(ep)}: {len(rows)} cues, {sum(r['dur'] for r in rows):.0f}s of music ({info})")
    return rows


# ----------------------------------------------------------------------------- 2. CLAP
class Clap:
    def __init__(self, device: str = "auto"):
        import torch
        from transformers import ClapModel, ClapProcessor

        self.torch = torch
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        src = str(CLAP_DIR) if (CLAP_DIR / "config.json").exists() else str(BC.clap_model)
        self.model = ClapModel.from_pretrained(src).eval().to(self.device)
        self.proc = ClapProcessor.from_pretrained(src)

    @staticmethod
    def _t(out):
        return out if hasattr(out, "shape") else getattr(out, "pooler_output", out[0])

    def audio(self, wav48: np.ndarray, win_s: float = 10.0, hop_s: float = 5.0) -> np.ndarray:
        """One embedding for a mono 48 kHz clip: mean of L2-normalised 10 s window embeddings (deterministic)."""
        w, h = int(win_s * CLAP_SR), int(hop_s * CLAP_SR)
        if len(wav48) <= w:
            wins = [wav48]
        else:
            starts = list(range(0, len(wav48) - w + 1, h))
            if starts[-1] != len(wav48) - w:
                starts.append(len(wav48) - w)
            wins = [wav48[s:s + w] for s in starts]
        embs = []
        for i in range(0, len(wins), 8):
            inp = self.proc.feature_extractor(wins[i:i + 8], sampling_rate=CLAP_SR, return_tensors="pt")
            with self.torch.no_grad():
                e = self._t(self.model.get_audio_features(**{k: v.to(self.device) for k, v in inp.items()}))
            embs.append(self.torch.nn.functional.normalize(e.float(), dim=-1).cpu().numpy())
        e = np.concatenate(embs).mean(axis=0)
        return (e / max(np.linalg.norm(e), 1e-9)).astype(np.float32)

    def text(self, texts: list[str]) -> np.ndarray:
        inp = self.proc.tokenizer(texts, padding=True, truncation=True, max_length=77, return_tensors="pt")
        with self.torch.no_grad():
            e = self._t(self.model.get_text_features(**{k: v.to(self.device) for k, v in inp.items()}))
        return self.torch.nn.functional.normalize(e.float(), dim=-1).cpu().numpy().astype(np.float32)


def load_48k_mono(path: Path) -> np.ndarray:
    import torch
    import torchaudio.functional as AF

    x, sr = read_wav(path, mono=True)
    return AF.resample(torch.from_numpy(x), sr, CLAP_SR).numpy() if sr != CLAP_SR else x


def embed_cues(rows: list[dict], clap: Clap) -> np.ndarray:
    cache = OUT / "clap_cache" / str(BC.clap_model).split("/")[-1]
    cache.mkdir(parents=True, exist_ok=True)
    out = []
    for i, r in enumerate(rows):
        p = cache / f"{r['id']}.npy"
        if p.exists():
            out.append(np.load(p))
            continue
        e = clap.audio(load_48k_mono(Path(CFG.paths.work) / r["path"]))
        tmp = cache / f"{r['id']}.tmp.npy"
        np.save(tmp, e)
        tmp.rename(p)
        out.append(e)
        if (i + 1) % 100 == 0:
            log(f"CLAP: {i + 1}/{len(rows)} cues")
    return np.stack(out).astype(np.float32) if out else np.zeros((0, 512), np.float32)


def estimate_bpm(path: Path) -> float | None:
    try:
        import librosa

        y, sr = librosa.load(str(path), sr=22050, mono=True, duration=30.0)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        return round(float(np.atleast_1d(tempo)[0]), 1)
    except Exception:
        return None


# ----------------------------------------------------------------------------- 3. index
def build_index(rows: list[dict], E: np.ndarray, clap: Clap, bpm: bool = True) -> tuple[list[dict], np.ndarray, list[dict]]:
    """Zero-shot music filter + mood tags. A cue is rejected only when it is clearly not music:
    music_score (mean cosine to the music prompts) < min_music_score AND it does not beat the best non-music prompt."""
    tm = clap.text(MUSIC_PROMPTS + NONMUSIC_PROMPTS)
    S = E @ tm.T
    music_score = S[:, : len(MUSIC_PROMPTS)].mean(axis=1)
    margin = music_score - S[:, len(MUSIC_PROMPTS):].max(axis=1)
    mood = E @ clap.text([f"{m} music" for m in MOODS]).T
    min_score = float(BC.get("min_music_score", 0.12))
    keep_rows, keep_idx, rejected = [], [], []
    for i, r in enumerate(rows):
        r = dict(r)
        r["music_score"] = round(float(music_score[i]), 3)
        r["music_margin"] = round(float(margin[i]), 3)
        r["tags"] = [MOODS[j] for j in np.argsort(-mood[i])[:3]]
        if music_score[i] < min_score and margin[i] < 0.0:
            rejected.append(r)
            continue
        if bpm:
            r["bpm"] = estimate_bpm(Path(CFG.paths.work) / r["path"])
        r["emb_file"] = rel(OUT / "clap.npy")
        r["emb_idx"] = len(keep_rows)
        keep_rows.append(r)
        keep_idx.append(i)
    return keep_rows, E[keep_idx], rejected


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", default="train", help="TRAIN episodes only (default: train)")
    ap.add_argument("--device", default="auto", help="auto|cuda|cpu (CLAP)")
    ap.add_argument("--force", action="store_true", help="re-cut the given episodes")
    ap.add_argument("--no-bpm", action="store_true")
    ap.add_argument("--skip-heldout", action="store_true", help="silently drop held-out episodes from the list instead of refusing")
    args = ap.parse_args()
    eps = parse_episodes(args.episodes)
    bad = [e for e in eps if e in HELDOUT_EPS]
    if bad and args.skip_heldout:
        log(f"BGM library is TRAIN-only: ignoring held-out episodes {bad}")
        eps = [e for e in eps if e not in HELDOUT_EPS]
    elif bad:
        raise SystemExit(f"the BGM library must be built from TRAIN episodes only; refusing held-out episodes {bad}")
    rows: list[dict] = []
    missing = []
    for ep in eps:
        d = ep_dir(ep)
        if not ((d / "instrumental.wav").exists() and (d / "vocals.wav").exists()):
            missing.append(ep)
            continue
        try:
            rows += cut_episode(ep, args.force)
        except Exception as e:
            log(f"{ep_name(ep)}: cue cutting FAILED {e!r}")
            missing.append(ep)
    # the index always covers every train episode that has been cut so far (not only the ones given now)
    done = {r["id"] for r in rows}
    for f in sorted((OUT / "episodes").glob("ep*.json")):
        if int(f.stem[2:]) in HELDOUT_EPS:
            continue
        rows += [r for r in read_json(f)["cues"] if r["id"] not in done]
    rows.sort(key=lambda r: (r["ep"], r["start"]))
    if not rows:
        if missing:
            raise SystemExit(f"no cues (episodes without stems: {missing})")
        log("BGM library: nothing to do (no train episode in the list and no cues cut so far)")
        return
    t0 = time.time()
    device = os.environ.get("AUDIO_BGM_DEVICE") or args.device
    if device != "cpu":
        cap_torch_vram(float(os.environ.get("AUDIO_BGM_VRAM_GB") or BC.vram_gb), reserve_gb=0.5)
    with VramMonitor() as mon:
        clap = Clap(device)
        E = embed_cues(rows, clap)
        keep, EK, rej = build_index(rows, E, clap, bpm=not args.no_bpm)
    tmp = OUT / "clap.tmp.npy"
    np.save(tmp, EK)
    tmp.rename(OUT / "clap.npy")
    write_jsonl(OUT / "cues.jsonl", keep)
    write_jsonl(OUT / "rejected.jsonl", rej)
    stats = {"cues": len(keep), "rejected_non_music": len(rej), "episodes": sorted({r["ep"] for r in keep}), "total_seconds": round(sum(r["dur"] for r in keep), 1),
             "median_dur": round(float(np.median([r["dur"] for r in keep])), 1) if keep else None,
             "low_dialogue_cues": int(sum(r["dlg_ratio"] <= 0.2 for r in keep)), "clap_model": str(BC.clap_model), "clap_device": clap.device,
             "index_seconds": round(time.time() - t0, 1), **mon.stats(), "episodes_without_stems": missing, "built": time.strftime("%Y-%m-%d %H:%M:%S")}
    write_json(OUT / "stats.json", stats)
    log(f"BGM library: {stats}")
    if missing:
        raise SystemExit(f"BGM library built, but these episodes had no stems yet: {missing}")


if __name__ == "__main__":
    main()
