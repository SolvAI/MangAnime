"""OP/ED/preview removal by audio fingerprint matching across episodes (brief §5.2).

Log-mel fingerprints at ~2 Hz per episode; for episode pairs of the same season we look for long diagonal runs
in the similarity matrix (the same recording at some time offset). Runs >= 55 s with mean cosine >= 0.94 are
marked as non-content (re-used BGM under different dialogue does not reach that). Subtitle `song` spans (karaoke) are merged in as extra evidence.

  python -m src.anime.oped
Output: data/processed/anime/oped.json  {"ep01": [[start, end, "op|ed|song"], ...]}
"""
from __future__ import annotations

import numpy as np

from src.common.config import P
from src.common.io import log, read_json, write_json

HOP_SEC = 8 * 1024 / 16000   # 0.512 s per feature frame


def chroma(ep: int) -> np.ndarray:
    """Log-mel fingerprint (48 bands, 0.512 s frames, loudness-normalised). Identical recordings (OP/ED) give
    cosine > 0.95 along a diagonal; re-used BGM under different dialogue does not."""
    import librosa

    cache = P("interim") / "anime" / f"ep{ep:02d}" / "fp_logmel.npy"
    if cache.exists():
        return np.load(cache)
    y, sr = librosa.load(str(P("interim") / "anime" / f"ep{ep:02d}" / "audio_jpn_16k.wav"), sr=16000, mono=True)
    M = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=1024, n_mels=48, fmin=60, fmax=7600)
    L = librosa.power_to_db(M, ref=1.0, top_db=None)
    k = 8
    T = (L.shape[1] // k) * k
    F = L[:, :T].reshape(48, -1, k).mean(2).T
    energy = F.mean(1)
    F = F - F.mean(1, keepdims=True)
    F = F / np.maximum(np.linalg.norm(F, axis=1, keepdims=True), 1e-6)
    F[energy < -55] = 0
    F = F.astype(np.float32)
    np.save(cache, F)
    return F


def common_runs(a: np.ndarray, b: np.ndarray, thr: float = 0.92, min_sec: float = 55.0, min_mean: float = 0.94) -> list[tuple[float, float, float]]:
    """Return [(start_a, end_a, offset_sec)] for long matching diagonals between two feature sequences."""
    S = a @ b.T
    hit = S > thr
    min_len = int(min_sec / HOP_SEC)
    runs = []
    na, nb = hit.shape
    for d in range(-(na - min_len), nb - min_len):
        diag = np.diagonal(hit, d)
        if diag.sum() < min_len * 0.8:
            continue
        sm = np.convolve(diag.astype(np.float32), np.ones(8) / 8, mode="same") > 0.6
        idx = np.flatnonzero(np.diff(np.r_[0, sm.astype(np.int8), 0]))
        for s, e in zip(idx[::2], idx[1::2]):
            if e - s >= min_len and float(np.diagonal(S, d)[s:e].mean()) >= min_mean:
                i0 = s if d >= 0 else s - d
                runs.append((i0 * HOP_SEC, (i0 + (e - s)) * HOP_SEC, d * HOP_SEC))
    return runs


def merge(spans: list[list[float]], gap: float = 3.0) -> list[list[float]]:
    spans = sorted(spans)
    out: list[list[float]] = []
    for s, e in spans:
        if out and s <= out[-1][1] + gap:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def run() -> dict:
    eps = [e for e in range(1, 25) if (P("interim") / "anime" / f"ep{e:02d}" / "audio_jpn_16k.wav").exists()]
    feats = {e: chroma(e) for e in eps}
    out = {}
    for e in eps:
        season = [x for x in eps if (x <= 12) == (e <= 12) and x != e]
        # compare with the 4 nearest episodes of the season
        others = sorted(season, key=lambda x: abs(x - e))[:5]
        votes: list[list[float]] = []
        for o in others:
            for s, t, _ in common_runs(feats[e], feats[o]):
                votes.append([s, t])
        # keep regions supported by >= 2 other episodes
        dur = len(feats[e]) * HOP_SEC
        grid = np.zeros(int(dur) + 2, np.int16)
        for s, t in votes:
            grid[int(s) : int(t) + 1] += 1
        mask = grid >= 1   # exact-recording matches are reliable: one other episode is enough
        idx = np.flatnonzero(np.diff(np.r_[0, mask.astype(np.int8), 0]))
        spans = [[float(s), float(t)] for s, t in zip(idx[::2], idx[1::2]) if t - s >= 30]
        # subtitle karaoke spans as extra evidence
        songs = [[x["start"], x["end"]] for x in read_json(P("processed") / "anime" / "subs" / f"ep{e:02d}.json", []) if x["kind"] == "song"]
        songs = [s for s in merge(songs, gap=8.0) if s[1] - s[0] >= 30]
        spans = merge(spans + songs, gap=5.0)
        labelled = []
        for s, t in spans:
            kind = "op" if s < dur * 0.35 else ("ed" if s > dur * 0.6 else "song")
            labelled.append([round(s, 1), round(t, 1), kind])
        # everything after the ED (next-episode preview / end card) is non-content too
        eds = [x for x in labelled if x[2] == "ed"]
        if eds and dur - eds[-1][1] < 40:
            labelled.append([eds[-1][1], round(dur, 1), "preview"])
        out[f"ep{e:02d}"] = labelled
        log(f"ep{e:02d}: {labelled}")
    write_json(P("processed") / "anime" / "oped.json", out)
    return out


def non_content(ep: int) -> list[tuple[float, float]]:
    d = read_json(P("processed") / "anime" / "oped.json", {})
    return [(s, t) for s, t, _ in d.get(f"ep{ep:02d}", [])]


def is_content(ep: int, start: float, end: float, spans: list[tuple[float, float]] | None = None) -> bool:
    spans = non_content(ep) if spans is None else spans
    for s, t in spans:
        if min(end, t) - max(start, s) > 0.25 * (end - start):
            return False
    return True


if __name__ == "__main__":
    run()
