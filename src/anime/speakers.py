"""Voice clusters from the per-line speaker embeddings (src/anime/audio_embed.py).

Build (TRAIN episodes only build clusters/centroids; held-out lines are only *assigned*). Re-run `cluster` whenever more
episodes have embeddings: it uses every train epNN.npz present, and cluster ids are inherited from the previous
clusters.json (most similar centroid, cosine >= 0.75), so names given to ids stay valid:
    . scripts/env.sh
    $KOMI_WORK/venvs/audio/bin/python -m src.anime.speakers cluster            # threshold picked automatically
    $KOMI_WORK/venvs/audio/bin/python -m src.anime.speakers cluster --threshold 0.62
    $KOMI_WORK/venvs/audio/bin/python -m src.anime.speakers report

Use (numpy only, works in every venv):
    from src.anime.speakers import load_centroids, assign, get_assignment, cluster_lines
    ids, C = load_centroids()                 # ids [K], C [K, 256] L2-normalised
    cluster_id, cosine = assign(emb)          # nearest centroid for one embedding (or arrays for [N, 256])

Output data/processed/anime/spk/clusters.json:
    {"params": {...},
     "clusters": [{"id", "row", "size", "core_size", "centroid_file", "total_dur", "mean_cos", "episodes": {"ep01": n, ...},
                   "exemplars": [{"ep", "idx", "start", "end", "text", "dur", "cos", "core"}], ...}],
     "assignments": {"epNN:idx": cluster_id},      # every embedded line that has a cluster: TRAIN lines and, assigned to
                                                   # the nearest TRAIN centroid, HELD-OUT lines (ep22-24). Lines without a
                                                   # cluster are not in the map (they are listed in "unassigned").
     "heldout_assignments": {"epNN:idx": {"cluster_id", "nearest", "cosine", "heldout": true}},   # the held-out subset, flagged
     "unassigned": ["epNN:idx", ...],
     "sanity": {...}}
`idx` is the index into data/processed/anime/subs/epNN.json; `ep` is the episode number (int). Centroids, sizes, exemplars
and cluster building use TRAIN episodes only. Keys that other stages add to a cluster (e.g. name / confidence / votes
from src.anime.name_speakers) are carried over when the cluster keeps its id in a re-clustering.
"""
from __future__ import annotations

import argparse
import json
import re
from functools import lru_cache
from pathlib import Path

import numpy as np

from src.anime.audio_common import ACFG
from src.common.config import CFG

SP = ACFG.spk
SPK_DIR = Path(SP.out_dir)
CLUSTERS = SPK_DIR / "clusters.json"
TRAIN_EPS = [int(e) for e in CFG.split.train_episodes]
HELDOUT_EPS = [int(e) for e in CFG.split.heldout_episodes]


def _ep(ep: int) -> str:
    return f"ep{int(ep):02d}"


# ----------------------------------------------------------------------------- public API
@lru_cache(maxsize=4)
def _load(spk_dir: str) -> tuple[np.ndarray, np.ndarray, dict]:
    d = Path(spk_dir)
    doc = json.loads((d / "clusters.json").read_text())
    ids = np.array([c["id"] for c in doc["clusters"]], dtype=np.int64)
    C = np.load(d / "centroids.npy").astype(np.float32)
    assert C.shape[0] == len(ids), (C.shape, len(ids))
    return ids, C, doc


def load_centroids(spk_dir: str | Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(cluster ids [K], centroids [K, D] L2-normalised). Centroids are built from TRAIN episodes only."""
    ids, C, _ = _load(str(spk_dir or SPK_DIR))
    return ids, C


def load_clusters(spk_dir: str | Path | None = None) -> dict:
    return _load(str(spk_dir or SPK_DIR))[2]


def assign(emb: np.ndarray, min_cos: float | None = None, spk_dir: str | Path | None = None):
    """Nearest voice cluster. emb [D] -> (cluster_id, cosine); emb [N, D] -> (ids [N], cosines [N]).

    With `min_cos`, embeddings whose best cosine is below it get cluster_id -1 (the cosine is still returned).
    """
    ids, C = load_centroids(spk_dir)
    e = np.asarray(emb, dtype=np.float32)
    single = e.ndim == 1
    e = np.atleast_2d(e)
    e = e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-9)
    S = e @ C.T
    j = S.argmax(axis=1)
    cos = S[np.arange(len(e)), j]
    cid = ids[j].copy()
    if min_cos is not None:
        cid[cos < min_cos] = -1
    return (int(cid[0]), float(cos[0])) if single else (cid, cos)


def get_assignment(ep: int, idx: int, spk_dir: str | Path | None = None) -> dict | None:
    """{"cluster_id", "heldout", ["cosine", "nearest"]} for subtitle line `idx` of episode `ep`; None if the line was not
    embedded (too short / silent). cluster_id is -1 when the line has no cluster."""
    doc = load_clusters(spk_dir)
    k = f"{_ep(ep)}:{int(idx)}"
    h = doc.get("heldout_assignments", {}).get(k)
    if h is not None:
        return dict(h)
    if k in doc["assignments"]:
        return {"cluster_id": int(doc["assignments"][k]), "heldout": int(ep) in HELDOUT_EPS}
    if k in set(doc.get("unassigned", [])):
        return {"cluster_id": -1, "heldout": int(ep) in HELDOUT_EPS}
    return None


def cluster_lines(cluster_id: int, include_heldout: bool = False, spk_dir: str | Path | None = None) -> list[tuple[int, int]]:
    """[(episode, subtitle idx)] of a cluster. Held-out episodes are excluded unless asked for (never use them as
    voice references or training data)."""
    doc = load_clusters(spk_dir)
    out = [(int(k[2:4]), int(k.split(":")[1])) for k, v in doc["assignments"].items() if int(v) == int(cluster_id)]
    if not include_heldout:
        out = [x for x in out if x[0] not in HELDOUT_EPS]
    return sorted(out)


# ----------------------------------------------------------------------------- building
def _load_npz(ep: int, spk_dir: Path) -> dict | None:
    p = spk_dir / f"{_ep(ep)}.npz"
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=False)
    return {k: z[k] for k in z.files}


def _subs(ep: int) -> list[dict]:
    return json.loads((Path(CFG.paths.processed) / "anime" / "subs" / f"{_ep(ep)}.json").read_text())


_CONT = re.compile(r"(,|\.\.\.|…|—|--|-|;|:|[A-Za-z0-9])$")
_NORM = re.compile(r"[^a-z0-9 ]+")


def _table(eps: list[int], spk_dir: Path) -> dict:
    """Stack the per-episode npz files into one table (with subtitle text / times)."""
    cols: dict[str, list] = {k: [] for k in ("ep", "idx", "emb", "dur", "rms", "speech_dur", "overlap", "multi", "start", "end", "text")}
    for ep in eps:
        z = _load_npz(ep, spk_dir)
        if z is None:
            continue
        subs = _subs(ep)
        n = len(z["idx"])
        cols["ep"].append(np.full(n, ep, np.int32))
        cols["idx"].append(z["idx"].astype(np.int32))
        cols["emb"].append(z["emb"].astype(np.float32))
        for k in ("dur", "rms"):
            cols[k].append(z[k].astype(np.float32))
        cols["speech_dur"].append(z["speech_dur"].astype(np.float32) if "speech_dur" in z else z["dur"].astype(np.float32))
        for k in ("overlap", "multi"):
            cols[k].append(z[k].astype(bool) if k in z else np.zeros(n, bool))
        cols["start"].append(np.array([subs[i]["start"] for i in z["idx"]], np.float32))
        cols["end"].append(np.array([subs[i]["end"] for i in z["idx"]], np.float32))
        cols["text"].append(np.array([subs[i]["text"] for i in z["idx"]], dtype=object))
    if not cols["ep"]:
        return {k: np.zeros((0, 256), np.float32) if k == "emb" else np.zeros(0) for k in cols}
    return {k: np.concatenate(v) for k, v in cols.items()}


def _must_link(T: dict, core: np.ndarray) -> np.ndarray:
    """Pairs (i, j) of consecutive core lines that are very likely the same speaker: the next subtitle starts within
    0.3 s and the first one ends like an unfinished sentence (comma, dash, ellipsis or no terminal punctuation)."""
    pairs = []
    order = np.lexsort((T["start"], T["ep"]))
    for a, b in zip(order[:-1], order[1:]):
        if T["ep"][a] != T["ep"][b] or not (core[a] and core[b]):
            continue
        if T["idx"][b] - T["idx"][a] > 3 or T["start"][b] - T["end"][a] > 0.3 or T["start"][b] < T["end"][a] - 0.05:
            continue
        ta, tb = str(T["text"][a]).strip(), str(T["text"][b]).strip()
        tb = tb.lstrip(".… ")
        if _CONT.search(ta) and (tb[:1].islower() or tb[:2] in ("I ", "I'")):
            pairs.append((a, b))
    return np.array(pairs, dtype=np.int64).reshape(-1, 2)


def _narrator_lines(T: dict, min_eps: int = 5) -> np.ndarray:
    """Indices of lines whose (long) subtitle text recurs verbatim in >= min_eps episodes: in this show that is the
    narrator's fixed introduction ("...they struggle to communicate with others", ...). One voice across episodes."""
    norm = np.array([_NORM.sub("", str(t).lower()).strip() for t in T["text"]], dtype=object)
    seen: dict[str, set] = {}
    for i in range(len(norm)):
        if len(norm[i]) >= 20 and not norm[i].startswith("subtitle translation"):
            seen.setdefault(norm[i], set()).add(int(T["ep"][i]))
    keep = {t for t, e in seen.items() if len(e) >= min_eps}
    return np.array([i for i in range(len(norm)) if norm[i] in keep], dtype=np.int64)


class NotEnoughData(RuntimeError):
    pass


def _stable_ids(C: np.ndarray, spk_dir: Path, fresh: bool = False, min_cos: float = 0.75) -> tuple[np.ndarray, list[dict], dict]:
    """Cluster ids that survive re-clustering: every new centroid inherits the id of the most similar centroid of the
    previous run (greedy one-to-one matching, cosine >= min_cos); unmatched clusters get new ids. Names attached to
    ids therefore stay valid when more episodes are embedded and `cluster` is run again."""
    K = len(C)
    ids = -np.ones(K, dtype=np.int64)
    info = [{"matched_prev": False, "cos": None} for _ in range(K)]
    prev_ids, prev_C = None, None
    prev_by_id: dict[int, dict] = {}
    if not fresh and (spk_dir / "clusters.json").exists() and (spk_dir / "centroids.npy").exists():
        try:
            prev = json.loads((spk_dir / "clusters.json").read_text())
            prev_by_id = {int(c["id"]): c for c in prev["clusters"]}
            prev_ids = np.array([c["id"] for c in prev["clusters"]], dtype=np.int64)
            prev_C = np.load(spk_dir / "centroids.npy").astype(np.float32)
            if len(prev_ids) != len(prev_C) or prev_C.shape[1] != C.shape[1]:
                prev_ids = prev_C = None
        except Exception:
            prev_ids = prev_C = None
        if prev_ids is None:
            prev_by_id = {}
    nxt = 0
    if prev_ids is not None and len(prev_ids):
        S = C @ prev_C.T
        used_new, used_prev = set(), set()
        for flat in np.argsort(-S, axis=None):
            k, j = divmod(int(flat), S.shape[1])
            if S[k, j] < min_cos:
                break
            if k in used_new or j in used_prev:
                continue
            ids[k] = prev_ids[j]
            info[k] = {"matched_prev": True, "cos": round(float(S[k, j]), 4)}
            used_new.add(k)
            used_prev.add(j)
        nxt = int(prev_ids.max()) + 1
    for k in range(K):
        if ids[k] < 0:
            ids[k] = nxt
            nxt += 1
    return ids, info, prev_by_id


def build(spk_dir: Path = SPK_DIR, threshold: float | str = "auto", write: bool = True, fresh: bool = False) -> dict:
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist

    cc = SP.cluster
    T = _table(TRAIN_EPS, spk_dir)
    n = len(T["ep"])
    if n < 50:
        raise NotEnoughData(f"not enough train embeddings in {spk_dir} yet ({n} lines); embed at least one TRAIN episode first")
    eps_present = sorted(set(int(e) for e in T["ep"]))
    core = (T["dur"] >= float(cc.core_min_dur)) & ~T["overlap"] & ~T["multi"] & (T["speech_dur"] >= 0.8)
    ci = np.where(core)[0]
    X = T["emb"][ci]
    Z = linkage(pdist(X, metric="cosine"), method="average")

    # ---- label-free threshold selection
    ml = _must_link(T, core)
    pos = {g: k for k, g in enumerate(ci)}
    ml_c = np.array([(pos[a], pos[b]) for a, b in ml], dtype=np.int64).reshape(-1, 2)
    rng = np.random.default_rng(0)
    rp = rng.integers(0, len(ci), size=(20000, 2))
    rp = rp[rp[:, 0] != rp[:, 1]]
    sweep = []
    for t in np.round(np.arange(0.30, 0.861, 0.02), 2):
        lab = fcluster(Z, t=float(t), criterion="distance")
        sizes = np.bincount(lab)
        big = int((sizes >= int(cc.min_cluster_size)).sum())
        r_ml = float((lab[ml_c[:, 0]] == lab[ml_c[:, 1]]).mean()) if len(ml_c) else float("nan")
        r_rp = float((lab[rp[:, 0]] == lab[rp[:, 1]]).mean())
        cov = float(sizes[sizes >= int(cc.min_cluster_size)].sum() / len(lab))
        sweep.append({"t": float(t), "clusters": big, "must_link_same": round(r_ml, 4), "random_same": round(r_rp, 4),
                      "J": round(r_ml - r_rp, 4), "coverage": round(cov, 4), "largest": int(sizes.max()),
                      "largest_share": round(float(sizes.max() / len(lab)), 4)})
    if threshold == "auto":
        # guard against a degenerate cut: no single voice may hold more than 40 % of the core lines
        ok = [s for s in sweep if s["largest_share"] <= 0.40] or sweep[:1]
        if len(ml_c) >= 30:
            jmax = max(s["J"] for s in ok)
            thr = min(s["t"] for s in ok if s["J"] >= jmax - 0.01)  # most conservative cut within 0.01 of the best J
            how = ("auto: smallest distance threshold whose J = must_link_same - random_same is within 0.01 of the maximum "
                   "(cuts where one cluster holds > 40 % of the core lines are excluded)")
        else:
            thr, how = 0.60, "fallback 0.60 (fewer than 30 must-link pairs)"
    else:
        thr, how = float(threshold), "manual"
    lab = fcluster(Z, t=thr, criterion="distance")

    # ---- clusters (>= min size), centroids from core TRAIN members only
    sizes = np.bincount(lab)
    kept = [l for l in np.argsort(-sizes) if sizes[l] >= int(cc.min_cluster_size) and l > 0]
    cents = []
    for l in kept:
        c = X[lab == l].mean(axis=0)
        cents.append(c / max(np.linalg.norm(c), 1e-9))
    C = np.stack(cents).astype(np.float32)
    # every train line -> nearest centroid; core members of kept clusters keep their agglomerative label
    S = T["emb"] @ C.T
    near = S.argmax(axis=1)
    cos = S[np.arange(n), near]
    cid = np.where(cos >= float(cc.assign_min_cos), near, -1)
    remap = {l: k for k, l in enumerate(kept)}
    for k_local, g in enumerate(ci):
        if lab[k_local] in remap:
            cid[g] = remap[lab[k_local]]
            cos[g] = S[g, cid[g]]
    is_core_member = np.zeros(n, bool)
    is_core_member[ci] = np.isin(lab, kept)

    ids_arr, id_info, prev_by_id = _stable_ids(C, spk_dir, fresh)   # row k of C / cid values -> persistent cluster id
    gid = lambda k: int(ids_arr[k]) if k >= 0 else -1    # noqa: E731

    # ---- exemplars: closest-to-centroid core lines, round-robin over episodes
    clusters = []
    (spk_dir / "centroids").mkdir(parents=True, exist_ok=True)
    if write:
        for old in (spk_dir / "centroids").glob("c*.npy"):
            old.unlink()
    rel = lambda p: str(p).replace(str(CFG.paths.work) + "/", "")  # noqa: E731  (data/processed/... relative to the repo)
    for k in range(len(kept)):
        mem = np.where(cid == k)[0]
        cm = np.where((cid == k) & is_core_member)[0]
        cand = cm[np.argsort(-cos[cm])]
        long_ = [i for i in cand if T["dur"][i] >= 1.5] or list(cand)
        by_ep: dict[int, list[int]] = {}
        for i in long_:
            by_ep.setdefault(int(T["ep"][i]), []).append(int(i))
        ex, r = [], 0
        while len(ex) < int(cc.n_exemplars) and any(len(v) > r for v in by_ep.values()):
            for e in sorted(by_ep, key=lambda e: -cos[by_ep[e][0]]):
                if len(by_ep[e]) > r and len(ex) < int(cc.n_exemplars):
                    ex.append(by_ep[e][r])
            r += 1
        if len(ex) < int(cc.n_exemplars):  # small clusters: top up with the best assigned (non-core) lines
            rest = [int(i) for i in mem[np.argsort(-cos[mem])] if int(i) not in set(ex)]
            ex += rest[: int(cc.n_exemplars) - len(ex)]
        cf = spk_dir / "centroids" / f"c{gid(k):03d}.npy"
        if write:
            np.save(cf, C[k])
        ep_counts = {_ep(e): int((T["ep"][mem] == e).sum()) for e in eps_present if (T["ep"][mem] == e).any()}
        clusters.append({
            "id": gid(k), "row": k, "id_from_previous_run": id_info[k], "size": int(len(mem)), "core_size": int(len(cm)),
            "centroid_file": rel(cf),
            "total_dur": round(float(T["dur"][mem].sum()), 1), "mean_cos": round(float(cos[mem].mean()), 4),
            "n_episodes": len(ep_counts), "episodes": ep_counts,
            "exemplars": [{"ep": int(T["ep"][i]), "idx": int(T["idx"][i]), "start": round(float(T["start"][i]), 3),
                           "end": round(float(T["end"][i]), 3), "text": str(T["text"][i]), "dur": round(float(T["dur"][i]), 2),
                           "cos": round(float(cos[i]), 4), "core": bool(is_core_member[i])} for i in ex]})
        if id_info[k]["matched_prev"]:  # keep what other stages attached to this id (name / confidence / votes ...)
            for key, val in prev_by_id.get(gid(k), {}).items():
                clusters[-1].setdefault(key, val)
    assignments = {f"{_ep(T['ep'][i])}:{int(T['idx'][i])}": gid(int(cid[i])) for i in range(n) if cid[i] >= 0}
    unassigned = [f"{_ep(T['ep'][i])}:{int(T['idx'][i])}" for i in range(n) if cid[i] < 0]

    # ---- held-out lines: nearest centroid only (never used for centroids)
    H = _table(HELDOUT_EPS, spk_dir)
    held = {}
    if len(H["ep"]):
        SH = H["emb"] @ C.T
        hn = SH.argmax(axis=1)
        hc = SH[np.arange(len(hn)), hn]
        for i in range(len(hn)):
            held[f"{_ep(H['ep'][i])}:{int(H['idx'][i])}"] = {"cluster_id": gid(int(hn[i])) if hc[i] >= float(cc.assign_min_cos) else -1,
                                                           "nearest": gid(int(hn[i])), "cosine": round(float(hc[i]), 4), "heldout": True}

    # ---- label-free sanity checks
    same = lambda a, b: bool(cid[a] == cid[b] and cid[a] >= 0)  # noqa: E731
    ml_rate = float(np.mean([same(a, b) for a, b in ml])) if len(ml) else None
    rpg = rng.integers(0, n, size=(20000, 2))
    rp_rate = float(np.mean([same(a, b) for a, b in rpg if a != b]))
    nar = _narrator_lines(T)
    nar_info = None
    if len(nar) >= 5:
        lab_n = cid[nar]
        vals, cnt = np.unique(lab_n[lab_n >= 0], return_counts=True)
        if len(vals):
            top = int(vals[cnt.argmax()])
            nar_info = {"lines": int(len(nar)), "episodes": int(len(set(T["ep"][nar].tolist()))), "modal_cluster": gid(top),
                        "modal_share": round(float(cnt.max() / len(nar)), 4), "unassigned": int((lab_n < 0).sum()),
                        "example_texts": sorted(set(str(t) for t in T["text"][nar]))[:6]}
    assigned = cid >= 0
    sanity = {
        "must_link_pairs": int(len(ml)), "must_link_same_cluster": None if ml_rate is None else round(ml_rate, 4),
        "random_pair_same_cluster": round(rp_rate, 4),
        "narrator_intro": nar_info,
        "train_lines": int(n), "train_assigned": round(float(assigned.mean()), 4),
        "mean_cos_core": round(float(cos[is_core_member].mean()), 4) if is_core_member.any() else None,
        "mean_cos_noncore": round(float(cos[~is_core_member & assigned].mean()), 4) if (~is_core_member & assigned).any() else None,
        "mean_cos_two_speaker_lines": round(float(cos[T["multi"] | T["overlap"]].mean()), 4) if (T["multi"] | T["overlap"]).any() else None,
        "heldout_lines": len(held),
        "heldout_assigned": round(float(np.mean([v["cluster_id"] >= 0 for v in held.values()])), 4) if held else None,
        "heldout_mean_cos": round(float(np.mean([v["cosine"] for v in held.values()])), 4) if held else None,
    }
    src = sorted({str(z["source"]) for z in (_load_npz(e, spk_dir) for e in eps_present) if z is not None and "source" in z})
    doc = {
        "params": {"embedding_model": "Wespeaker/wespeaker-voxceleb-resnet293-LM (256-d, cosine)", "embedding_source": src,
                   "method": "agglomerative, average linkage, cosine distance", "distance_threshold": thr, "threshold_selection": how,
                   "core_min_dur": float(cc.core_min_dur), "min_cluster_size": int(cc.min_cluster_size),
                   "assign_min_cos": float(cc.assign_min_cos), "train_episodes": eps_present,
                   "heldout_episodes": sorted(set(int(e) for e in H["ep"])) if len(H["ep"]) else [],
                   "core_lines": int(len(ci)), "built": __import__("time").strftime("%Y-%m-%d %H:%M:%S"),
                   "id_policy": "ids are stable across re-clustering: a cluster inherits the id of the previous run's most similar "
                                "centroid (cosine >= 0.75, one-to-one); `row` is the row in centroids.npy",
                   "ids_inherited": int(sum(i["matched_prev"] for i in id_info)), "sweep": sweep},
        "clusters": clusters, "assignments": {**assignments, **{k: v["cluster_id"] for k, v in held.items() if v["cluster_id"] >= 0}},
        "heldout_assignments": held, "unassigned": unassigned + [k for k, v in held.items() if v["cluster_id"] < 0], "sanity": sanity,
    }
    if write:
        np.save(spk_dir / "centroids.npy", C)
        tmp = spk_dir / "clusters.json.tmp"
        tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=1))
        tmp.replace(spk_dir / "clusters.json")
        _load.cache_clear()
    return doc


def report(spk_dir: Path = SPK_DIR, n_text: int = 4) -> str:
    doc = json.loads((spk_dir / "clusters.json").read_text())
    p, s = doc["params"], doc["sanity"]
    L = [f"voice clusters: {len(doc['clusters'])} (threshold {p['distance_threshold']}, {p['threshold_selection']})",
         f"train lines {s['train_lines']} (core {p['core_lines']}), assigned {s['train_assigned']:.1%}; held-out lines {s['heldout_lines']}"
         f" assigned {s['heldout_assigned']}", f"sanity: {json.dumps({k: v for k, v in s.items() if k != 'narrator_intro'})}",
         f"narrator intro: {json.dumps(s.get('narrator_intro'), ensure_ascii=False)}", ""]
    for c in doc["clusters"]:
        L.append(f"c{c['id']:03d} size={c['size']:4d} core={c['core_size']:4d} eps={c['n_episodes']:2d} dur={c['total_dur']:7.1f}s cos={c['mean_cos']:.2f} | "
                 + " / ".join(e["text"][:38] for e in c["exemplars"][:n_text]))
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["cluster", "report"])
    ap.add_argument("--threshold", default="auto", help="'auto' or a cosine-distance threshold (e.g. 0.62)")
    ap.add_argument("--spk-dir", default=None)
    ap.add_argument("--fresh", action="store_true", help="do not inherit cluster ids from the previous clusters.json")
    args = ap.parse_args()
    d = Path(args.spk_dir) if args.spk_dir else SPK_DIR
    if args.cmd == "cluster":
        try:
            doc = build(d, args.threshold if args.threshold == "auto" else float(args.threshold), fresh=args.fresh)
        except NotEnoughData as e:  # e.g. only held-out episodes embedded so far: not an error, just nothing to do yet
            print(f"cluster: skipped - {e}")
            return
        print(json.dumps([{k: v for k, v in r.items()} for r in doc["params"]["sweep"]], indent=None))
    print(report(d))


if __name__ == "__main__":
    main()
