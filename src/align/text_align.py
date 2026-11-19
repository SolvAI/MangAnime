"""Dialogue alignment (brief §5.3 steps 1–2): chapter -> episode segment, then monotonic bubble<->subtitle anchors.

Translations differ (scanlation vs official subs), so matching is on meaning: BGE-M3 dense embeddings + a
monotonic DP that allows skips on both sides and 1-to-many / many-to-1 merges.

  python -m src.align.text_align            # all parsed chapters x all episodes
Outputs (data/processed/align/):
  chapter_segments.json   [{chapter, episode, start, end, n_bubbles, n_matched, score}]
  anchors.jsonl           {chapter, panel_order, bubble, episode, line, sim, start, end}
"""
from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.common.config import CFG, P
from src.common.io import log, read_json, read_jsonl, sha, write_json, write_jsonl

_EMB_CACHE = P("cache") / "emb_text"
_model = None


def _get_model():
    global _model
    if _model is None:
        import torch
        from sentence_transformers import SentenceTransformer

        dev = "cpu"
        try:
            if torch.cuda.is_available() and torch.cuda.mem_get_info()[0] > 4 * 2**30:
                dev = "cuda"
        except Exception:
            pass
        _model = SentenceTransformer("BAAI/bge-m3", device=dev)
        _model.max_seq_length = 128
    return _model


def embed(texts: list[str]) -> np.ndarray:
    """L2-normalised BGE-M3 embeddings with a per-text disk cache."""
    _EMB_CACHE.mkdir(parents=True, exist_ok=True)
    keys = [sha("bge-m3:" + t, 24) for t in texts]
    out: list[np.ndarray | None] = [None] * len(texts)
    todo = []
    for i, k in enumerate(keys):
        f = _EMB_CACHE / k[:2] / f"{k}.npy"
        if f.exists():
            out[i] = np.load(f)
        else:
            todo.append(i)
    if todo:
        m = _get_model()
        embs = m.encode([texts[i] for i in todo], batch_size=64, normalize_embeddings=True, show_progress_bar=False)
        for i, e in zip(todo, embs):
            f = _EMB_CACHE / keys[i][:2] / f"{keys[i]}.npy"
            f.parent.mkdir(parents=True, exist_ok=True)
            np.save(f, e.astype(np.float16))
            out[i] = e
    return np.stack([np.asarray(e, dtype=np.float32) for e in out]) if out else np.zeros((0, 1024), np.float32)


_WORD = re.compile(r"[A-Za-z]{2,}")


def norm_text(t: str) -> str:
    t = t.replace("…", "...").strip()
    if t.isupper() or sum(c.isupper() for c in t) > 0.6 * max(1, sum(c.isalpha() for c in t)):
        t = t.capitalize()  # scanlation lettering is ALL CAPS
    return t


def informative(t: str) -> bool:
    return len(_WORD.findall(t)) >= 2 and len(t) >= 8


def load_bubbles(chapters: list[int] | None = None) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = defaultdict(list)
    for r in read_jsonl(P("processed") / "manga" / "panels.jsonl"):
        if chapters and r["chapter"] not in chapters:
            continue
        for bi, d in enumerate(r["dialogue"]):
            if d["kind"] == "sfx":
                continue
            out[r["chapter"]].append({"chapter": r["chapter"], "panel_order": r["order"], "bubble": bi,
                                      "text": norm_text(d["text"]), "kind": d["kind"]})
    return out


def load_lines(ep: int) -> list[dict]:
    ev = read_json(P("processed") / "anime" / "subs" / f"ep{ep:02d}.json", [])
    lines = []
    for i, e in enumerate(ev):
        if e["kind"] in ("dialogue", "notebook") or (e["kind"] == "sign" and informative(e["text"])):
            lines.append({"idx": i, "start": e["start"], "end": e["end"], "text": e["text"], "kind": e["kind"]})
    lines.sort(key=lambda x: (x["start"], x["end"]))
    return lines


def dp_align(S: np.ndarray, tau: float) -> tuple[float, list[tuple[int, int]]]:
    """Monotonic alignment with skips and merges. S[i,j] = sim(bubble i, line j). Returns (score, [(i,j)...])."""
    n, m = S.shape
    M = np.full((n + 1, m + 1), -1e9, dtype=np.float32)  # best score ending with a match at (i,j)
    D = np.zeros((n + 1, m + 1), dtype=np.float32)       # best score over prefix (i,j)
    bm = np.zeros((n + 1, m + 1), dtype=np.int8)         # backpointer for M: 0 diag(D), 1 left(M), 2 up(M)
    bd = np.zeros((n + 1, m + 1), dtype=np.int8)         # backpointer for D: 0 up, 1 left, 2 match
    merge_pen = 0.08
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            g = S[i - 1, j - 1] - tau
            c0 = D[i - 1, j - 1]
            c1 = M[i, j - 1] - merge_pen   # same bubble continues on the next subtitle line
            c2 = M[i - 1, j] - merge_pen   # next bubble merged into the same subtitle line
            k = int(np.argmax([c0, c1, c2]))
            M[i, j] = g + (c0, c1, c2)[k]
            bm[i, j] = k
            d0, d1, d2 = D[i - 1, j], D[i, j - 1], M[i, j]
            k2 = int(np.argmax([d0, d1, d2]))
            D[i, j] = (d0, d1, d2)[k2]
            bd[i, j] = k2
    # backtrack
    pairs = []
    i, j, in_m = n, m, False
    while i > 0 and j > 0:
        if not in_m:
            k = bd[i, j]
            if k == 0:
                i -= 1
            elif k == 1:
                j -= 1
            else:
                in_m = True
        else:
            pairs.append((i - 1, j - 1))
            k = bm[i, j]
            if k == 0:
                i, j, in_m = i - 1, j - 1, False
            elif k == 1:
                j -= 1
            else:
                i -= 1
    pairs.reverse()
    return float(D[n, m]), pairs


def align_chapter_episode(bub: list[dict], Eb: np.ndarray, lines: list[dict], El: np.ndarray,
                          tau_hi: float = 0.62, tau_lo: float = 0.52) -> dict | None:
    """Two passes: strict anchors over the whole episode -> time window -> relaxed DP inside the window."""
    if len(bub) == 0 or len(lines) == 0:
        return None
    S = Eb @ El.T
    info = np.array([informative(b["text"]) for b in bub])
    S_strict = S.copy()
    S_strict[~info] = 0.0   # short interjections ("Eh?!") match everywhere: ignore them for segment finding
    score, pairs = dp_align(S_strict, tau_hi)
    if len(pairs) < 3:
        return None
    t = np.array([lines[j]["start"] for _, j in pairs])
    # densest cluster of anchors (segments are 2–8 min): keep anchors within a window around the median
    med = float(np.median(t))
    keep = np.abs(t - med) < 300
    if keep.sum() < 3:
        return None
    t0, t1 = float(t[keep].min()) - 45, float(t[keep].max()) + 45
    js = [j for j, l in enumerate(lines) if t0 <= l["start"] <= t1]
    if not js:
        return None
    j0, j1 = js[0], js[-1] + 1
    score2, pairs2 = dp_align(S[:, j0:j1], tau_lo)
    pairs2 = [(i, j + j0) for i, j in pairs2]
    # drop weak links inside merge chains and uninformative bubbles ("Eh?!") unless they match almost exactly
    pairs2 = [(i, j) for i, j in pairs2 if S[i, j] >= tau_lo and (info[i] or S[i, j] >= 0.80)]
    if not pairs2:
        return None
    # trim to the core span defined by the strong, informative anchors (stray matches far away are dropped)
    strong_t = sorted(lines[j]["start"] for i, j in pairs2 if S[i, j] >= 0.75 and info[i])
    if len(strong_t) >= 2:
        # densest run of strong anchors (a chapter is one contiguous segment; a gap > 60 s means a stray match)
        runs, cur = [], [strong_t[0]]
        for t in strong_t[1:]:
            if t - cur[-1] <= 60:
                cur.append(t)
            else:
                runs.append(cur)
                cur = [t]
        runs.append(cur)
        best = max(runs, key=len)
        lo, hi = best[0] - 15, best[-1] + 15
        pairs2 = [(i, j) for i, j in pairs2 if lo <= lines[j]["start"] <= hi]
    if not pairs2:
        return None
    sims = [float(S[i, j]) for i, j in pairs2]
    strong = sum(1 for (i, j), s in zip(pairs2, sims) if s >= 0.75 and info[i])
    ts = [lines[j]["start"] for _, j in pairs2]
    te = [lines[j]["end"] for _, j in pairs2]
    return {"score": score2, "pairs": pairs2, "sims": sims, "n_strong": strong, "start": min(ts), "end": max(te),
            "n_matched_bubbles": len({i for i, _ in pairs2}), "n_info": int(info.sum())}


def run(chapters: list[int] | None = None, episodes: list[int] | None = None) -> None:
    bubbles = load_bubbles(chapters)
    eps = episodes or [e for e in range(1, 25) if (P("processed") / "anime" / "subs" / f"ep{e:02d}.json").exists()]
    lines = {e: load_lines(e) for e in eps}
    log(f"embedding {sum(map(len, bubbles.values()))} bubbles, {sum(map(len, lines.values()))} subtitle lines")
    E_lines = {e: embed([l["text"] for l in lines[e]]) for e in eps}
    segs, anchors = [], []
    for ch in sorted(bubbles):
        bub = bubbles[ch]
        Eb = embed([b["text"] for b in bub])
        info = np.array([informative(b["text"]) for b in bub])
        if info.sum() < 3:
            continue
        # cheap episode prefilter: number of informative bubbles with a strong best match
        cand = []
        for e in eps:
            if len(lines[e]) == 0:
                continue
            best = (Eb[info] @ E_lines[e].T).max(1)
            cand.append((float((best >= 0.72).sum()), float(best.mean()), e))
        cand.sort(reverse=True)
        best_res, best_ep = None, None
        for hits, _, e in cand[:3]:
            if hits < 2:
                continue
            res = align_chapter_episode(bub, Eb, lines[e], E_lines[e])
            if res and (best_res is None or res["n_strong"] > best_res["n_strong"] or
                        (res["n_strong"] == best_res["n_strong"] and res["score"] > best_res["score"])):
                best_res, best_ep = res, e
        if best_res is None or best_res["n_strong"] < max(2, 0.12 * best_res["n_info"]):
            segs.append({"chapter": ch, "episode": None, "n_bubbles": len(bub), "n_info": int(info.sum()),
                         "best_hits": cand[0][0] if cand else 0})
            continue
        e = best_ep
        segs.append({"chapter": ch, "episode": e, "start": round(best_res["start"], 2), "end": round(best_res["end"], 2),
                     "n_bubbles": len(bub), "n_info": best_res["n_info"], "n_matched": best_res["n_matched_bubbles"],
                     "n_strong": best_res["n_strong"], "score": round(best_res["score"], 3)})
        for (i, j), s in zip(best_res["pairs"], best_res["sims"]):
            b, l = bub[i], lines[e][j]
            anchors.append({"chapter": ch, "panel_order": b["panel_order"], "bubble": b["bubble"], "episode": e,
                            "line": l["idx"], "sim": round(s, 4), "start": l["start"], "end": l["end"],
                            "manga_text": b["text"], "sub_text": l["text"]})
    # consecutive chapters adapted in the same episode must not overlap in time: cut at the midpoint of the overlap
    ok_segs = sorted([s for s in segs if s.get("episode")], key=lambda s: (s["episode"], s["start"]))
    for a, b in zip(ok_segs[:-1], ok_segs[1:]):
        if a["episode"] == b["episode"] and a["end"] > b["start"] and a["chapter"] < b["chapter"]:
            cut = (a["end"] + b["start"]) / 2
            a["end"], b["start"] = round(cut, 2), round(cut, 2)
    span = {s["chapter"]: (s["episode"], s["start"], s["end"]) for s in ok_segs}
    anchors = [a for a in anchors if a["chapter"] in span and span[a["chapter"]][1] - 1.0 <= a["start"] <= span[a["chapter"]][2] + 1.0]
    out = P("processed") / "align"
    write_json(out / "chapter_segments.json", segs)
    write_jsonl(out / "anchors.jsonl", anchors)
    ok = [s for s in segs if s["episode"]]
    log(f"chapters aligned: {len(ok)}/{len(segs)}; anchors: {len(anchors)}")
    by_ep = defaultdict(list)
    for s in ok:
        by_ep[s["episode"]].append(s["chapter"])
    for e in sorted(by_ep):
        log(f"  ep{e:02d}: chapters {sorted(by_ep[e])}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--chapters", default=None)
    ap.add_argument("--episodes", default=None)
    a = ap.parse_args()
    run([int(x) for x in a.chapters.split(",")] if a.chapters else None,
        [int(x) for x in a.episodes.split(",")] if a.episodes else None)
