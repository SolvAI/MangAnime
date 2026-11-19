"""Speaker names for the English dub lines -> data/processed/anime/lines_en.jsonl.

    {"ep","idx","start","end","text","speaker","conf","source"}     idx -> data/processed/anime/dub_lines/epNN.json

1. transfer   Episodes that have Japanese named lines (lines_named.jsonl, 1-14): the dub is timed to the same picture,
              so a dub line takes the name of the Japanese line(s) it overlaps in time (one speaker must cover
              >= transfer.min_overlap of the dub line and hold >= transfer.min_dominance of the overlapping speech).
2. centroids  Per name, a robust centroid of the English speaker embeddings (WeSpeaker ResNet293, spk_en/epNN.npz)
              of its core SEED lines = lines that got this name by transfer and still carry it.
              Lines named by voice alone never move a centroid, so a centroid cannot drift onto another character.
              Group names (Classmates, Other) get none.
3. refine     A transferred name is kept while its centroid is (nearly) the best one; lines with a wrong or missing
              transfer, and every line of episodes 15-21, take the nearest name centroid when it is clear enough,
              else they stay "Unknown". Repeated until stable. Names whose centroid sits on another name's voice
              (the transfer was wrong for the whole name) or that keep too few seeds are dropped and reported.
4. purity     Label-free / cross-checked numbers written to spk_en/names_en.json (see `purity()`).
5. text check (--llm-check)  every named voice is shown, without its name, to a text-only LLM: who says these lines?
6. discover   (--discover)   voices the transfer could not name (no Japanese seed, or the seed pointed at somebody
              else, as for Komi): the still-unnamed core lines are clustered by voice, each cluster is identified from
              its text by the LLM (and, where it exists, by the Japanese voice cluster of that name overlapping it in
              time); confident roster names become new centroids. Short fragments between two lines of such a voice
              (Komi's halting phone call) are filled in when their own embedding does not point at someone else.

Held-out episodes (22-24) are never read. CPU only (numpy / scipy):

    $KOMI_WORK/venvs/audio/bin/python -m src.anime.speakers_en          # -> lines_en.jsonl + spk_en/names_en.json
"""
from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from src.anime.audio_common import ACFG, HELDOUT_EPS, TRAIN_EPS, ep_name, log
from src.common.config import CFG
from src.common.io import read_json, read_jsonl, write_json, write_jsonl

EN = ACFG.dub_en
PROC = Path(CFG.paths.processed) / "anime"
LINES_DIR = PROC / "dub_lines"
SPK_DIR = Path(EN.spk_dir)
OUT = Path(EN.lines_out)
UNKNOWN = "Unknown"
GROUPS = {str(g) for g in EN.group_names}


def load_table(eps: list[int]) -> dict:
    """All dub lines of the given TRAIN episodes with their embedding (zero row + has_emb False when not embedded)."""
    assert not set(eps) & set(HELDOUT_EPS), "held-out episodes must not be read here"
    rows, embs, has = [], [], []
    for ep in eps:
        lines = read_json(LINES_DIR / f"{ep_name(ep)}.json")
        if lines is None:
            continue
        z = np.load(SPK_DIR / f"{ep_name(ep)}.npz", allow_pickle=False) if (SPK_DIR / f"{ep_name(ep)}.npz").exists() else None
        pos = {int(i): k for k, i in enumerate(z["idx"])} if z is not None else {}
        for ln in lines:
            k = pos.get(int(ln["idx"]))
            rows.append({"ep": ep, "idx": int(ln["idx"]), "start": float(ln["start"]), "end": float(ln["end"]),
                         "text": ln["text"], "song": bool(ln.get("song")), "word_prob": float(ln.get("word_prob", 1.0)),
                         "speech_dur": float(z["speech_dur"][k]) if k is not None else 0.0,
                         "overlap": bool(z["overlap"][k]) if k is not None else False})
            embs.append(z["emb"][k] if k is not None else np.zeros(256, np.float32))
            has.append(k is not None)
    return {"rows": rows, "emb": np.stack(embs).astype(np.float32) if embs else np.zeros((0, 256), np.float32),
            "has": np.array(has, bool)}


# ============================================================================================ 1. transfer
def transfer(rows: list[dict]) -> tuple[list[str | None], list[float], list[bool]]:
    """Name by time overlap with the Japanese named lines. Returns (name or None, confidence, strong) per dub line;
    strong = the Japanese evidence is a named voice cluster (mean conf >= transfer.strong_conf), not the text-only LLM."""
    Tc = EN.transfer
    named = defaultdict(list)
    for r in read_jsonl(PROC / "lines_named.jsonl"):
        if int(r["ep"]) in HELDOUT_EPS or float(r.get("conf", 1.0)) < float(Tc.min_conf):
            continue
        named[int(r["ep"])].append((float(r["start"]), float(r["end"]), str(r["speaker"]), float(r.get("conf", 1.0))))
    for v in named.values():
        v.sort()
    names, confs, strong = [], [], []
    for r in rows:
        best, conf, st = None, 0.0, False
        if not r["song"]:
            ov: dict[str, float] = defaultdict(float)
            cw: dict[str, float] = defaultdict(float)
            for s, e, sp, c in named.get(r["ep"], ()):
                if s >= r["end"]:
                    break
                o = min(e, r["end"]) - max(s, r["start"])
                if o > 0:
                    ov[sp] += o
                    cw[sp] += o * c
            if ov:
                top = max(ov, key=ov.get)
                share = ov[top] / max(1e-6, r["end"] - r["start"])
                dom = ov[top] / sum(ov.values())
                if share >= float(Tc.min_overlap) and dom >= float(Tc.min_dominance):
                    best, conf = top, round(min(1.0, share / 0.8) * dom * (cw[top] / ov[top]), 3)
                    st = cw[top] / ov[top] >= float(Tc.strong_conf)
        names.append(best)
        confs.append(conf)
        strong.append(st)
    return names, confs, strong


# ============================================================================================ 2./3. centroids + refine
def robust_centroid(E: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normalised mean after dropping outliers twice (members below max(0.35, median - 0.15)). Returns (c, kept mask)."""
    keep = np.ones(len(E), bool)
    c = None
    for _ in range(3):
        c = E[keep].mean(axis=0)
        c /= max(np.linalg.norm(c), 1e-9)
        sims = E @ c
        new = sims >= max(0.35, float(np.median(sims[keep])) - 0.15)
        if new.sum() < 3 or (new == keep).all():
            break
        keep = new
    return c, keep


def centroids_for(labels: list[str | None], core: np.ndarray, E: np.ndarray, min_lines: int) -> dict[str, np.ndarray]:
    by: dict[str, list[int]] = defaultdict(list)
    for i, n in enumerate(labels):
        if n and n not in GROUPS and n != UNKNOWN and core[i]:
            by[n].append(i)
    return {n: robust_centroid(E[ix])[0] for n, ix in by.items() if len(ix) >= min_lines}


def _support(rows, t_names, labels, core, C) -> dict:
    """Per name with a centroid: how many transferred seeds stand behind the final voice cluster."""
    out = {}
    for n in sorted(C):
        seed_core = [i for i in range(len(rows)) if t_names[i] == n and core[i]]
        kept = [i for i in seed_core if labels[i] == n]
        by_voice = [i for i in range(len(rows)) if labels[i] == n and core[i]]
        out[n] = {"seed_core_lines": len(seed_core), "seed_kept": len(kept),
                  "seed_keep_rate": round(len(kept) / max(1, len(seed_core)), 3), "core_lines_final": len(by_voice),
                  "seed_share_of_final": round(len(kept) / max(1, len(by_voice)), 3)}
    return out


def refine(T: dict, t_names: list[str | None], t_confs: list[float], t_strong: list[bool] | None = None) -> dict:
    """Name centroids anchored on the transferred seeds + nearest-centroid naming of everything else.
    Returns final labels, confidences, sources and the centroids."""
    N = EN.names
    rows, E, has = T["rows"], T["emb"], T["has"]
    dur = np.array([r["end"] - r["start"] for r in rows])
    sdur = np.array([r["speech_dur"] for r in rows])
    song = np.array([r["song"] for r in rows])
    ovl = np.array([r["overlap"] for r in rows])
    core = has & (dur >= float(N.core_min_dur)) & (sdur >= 0.8) & ~ovl & ~song
    labels: list[str | None] = list(t_names)
    dropped: dict[str, dict] = {}
    C: dict[str, np.ndarray] = {}
    t_strong = t_strong or [False] * len(rows)
    for it in range(int(N.iterations)):
        # seeds = transferred names that are still standing (round 0: all of them)
        seeds = [t_names[i] if (labels[i] == t_names[i] and t_names[i] not in dropped) else None for i in range(len(rows))]
        C_prev = set(C)
        C = centroids_for(seeds, core, E, int(N.min_core_lines))
        for n in C_prev - set(C):                    # too few seeds survived: the name has no voice of its own here
            dropped.setdefault(n, {"reason": "fewer than min_core_lines transferred seeds left on one voice", "iteration": it})
        # two names on one voice: drop the smaller one (its transfer pointed at somebody else's speech)
        sizes = Counter(n for i, n in enumerate(seeds) if n and core[i])
        names = sorted(C, key=lambda n: -sizes[n])
        for a_i, a in enumerate(names):
            for b in names[a_i + 1:]:
                if a in C and b in C and float(C[a] @ C[b]) >= float(N.alias_cos):
                    dropped[b] = {"same_voice_as": a, "centroid_cos": round(float(C[a] @ C[b]), 3), "core_lines": int(sizes[b]),
                                  "iteration": it}
                    del C[b]
        names = sorted(C)
        if not names:
            break
        M = np.stack([C[n] for n in names])
        S = E @ M.T
        order = np.argsort(-S, axis=1)
        new = list(labels)
        for i in range(len(rows)):
            if song[i]:
                new[i] = None
                continue
            seed = t_names[i]
            if seed in dropped:
                seed = None
            if not has[i]:
                new[i] = seed                      # no embedding (silent / too short): transfer only
                continue
            b1 = names[order[i, 0]]
            s1 = float(S[i, order[i, 0]])
            s2 = float(S[i, order[i, 1]]) if len(names) > 1 else -1.0
            if seed in C:
                own = float(S[i, names.index(seed)])
                if own >= float(N.min_own_cos) and own >= s1 - float(N.keep_margin):
                    new[i] = seed
                    continue
            if seed in GROUPS:
                new[i] = b1 if (s1 >= float(N.group_override_cos) and s1 - s2 >= float(N.assign_margin)) else seed
                continue
            new[i] = b1 if (s1 >= float(N.assign_min_cos) and s1 - s2 >= float(N.assign_margin)) else None
        changed = sum(a != b for a, b in zip(labels, new))
        labels = new
        log(f"refine round {it}: {len(C)} name centroids, {changed} labels changed")
        if changed == 0 and it > 0:
            break
    # ---- seed support of every name that still has a centroid; catch-all voices lose their name
    support = _support(rows, t_names, labels, core, C)
    weak = [n for n, v in support.items() if v["seed_share_of_final"] < float(N.min_seed_share)]
    if weak:                                        # few seeds, many other lines: remove and re-assign those lines once
        for n in weak:
            dropped[n] = {"reason": f"catch-all voice: only {support[n]['seed_share_of_final']:.0%} of its core lines were "
                                    f"transferred as {n}", **support[n]}
            del C[n]
        names = sorted(C)
        M = np.stack([C[n] for n in names])
        S = E @ M.T
        order = np.argsort(-S, axis=1)
        for i in range(len(rows)):
            if labels[i] in weak:
                seed = t_names[i] if t_names[i] not in dropped else None
                s1, s2 = float(S[i, order[i, 0]]), (float(S[i, order[i, 1]]) if len(names) > 1 else -1.0)
                ok = has[i] and s1 >= float(N.assign_min_cos) and s1 - s2 >= float(N.assign_margin)
                labels[i] = names[order[i, 0]] if ok else (seed if seed in GROUPS else None)
        log(f"dropped catch-all names {weak}")
        support = _support(rows, t_names, labels, core, C)
    # ---- final confidences / sources
    names = sorted(C)
    M = np.stack([C[n] for n in names]) if names else np.zeros((0, 256), np.float32)
    S = E @ M.T if names else np.zeros((len(rows), 0))
    out_lab, out_conf, out_src, own_cos, best_other = [], [], [], [], []
    for i, r in enumerate(rows):
        lab = labels[i]
        seed = t_names[i] if t_names[i] not in dropped else None
        if lab is None:
            out_lab.append(UNKNOWN)
            out_conf.append(0.0)
            out_src.append("song" if song[i] else "none")
            own_cos.append(None)
            best_other.append(None)
            continue
        if lab in C and has[i]:
            k = names.index(lab)
            own = float(S[i, k])
            oth = float(np.max(np.delete(S[i], k))) if len(names) > 1 else -1.0
            sep = min(1.0, max(0.0, (own - 0.35) / 0.35)) * 0.6 + min(1.0, max(0.0, (own - oth) / 0.25)) * 0.4
            if seed == lab:
                conf, src = 0.6 + 0.39 * sep, "transfer+spk"
            else:
                conf, src = 0.45 + 0.4 * sep, "spk"
            own_cos.append(round(own, 4))
            best_other.append(round(oth, 4))
        else:                                       # group name, or a transfer that could not be checked by voice
            conf, src = (0.5 if lab in GROUPS else 0.4 * max(0.5, t_confs[i] / 0.7)), "transfer"
            own_cos.append(None)
            best_other.append(None)
        out_lab.append(lab)
        out_conf.append(round(float(conf), 3))
        out_src.append(src)
    return {"labels": out_lab, "conf": out_conf, "source": out_src, "own_cos": own_cos, "best_other": best_other,
            "centroids": C, "dropped": dropped, "core": core, "support": support}


# ============================================================================================ 4. purity
_NORM = re.compile(r"[^a-z0-9 ]+")


def purity(T: dict, t_names: list[str | None], R: dict) -> dict:
    """Numbers that say how far the names can be trusted (none of them is a ground-truth accuracy):

    cross_episode   centroids from the RAW transferred names of odd episodes classify the transferred core lines of
                    even episodes and vice versa: share where the nearest centroid is the transferred name
                    (= consistency of the time-overlap transfer with the voices, before any clean-up).
    transfer_fate   what the refinement did with the transferred names (kept / renamed / dropped).
    clusters        label-free agglomerative clusters of the core lines (average linkage, cosine): size-weighted share
                    of the majority FINAL name per cluster (and the same for the raw transfer), at two cuts.
    recurring_text  lines whose text recurs in >= 5 episodes (the narrator's fixed introduction): share with the
                    modal name.
    margin          final named core lines: share whose own centroid beats every other by >= 0.10 cosine.
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist

    rows, E, core = T["rows"], T["emb"], R["core"]
    ep = np.array([r["ep"] for r in rows])
    N = EN.names
    out: dict = {}
    # ---- cross-episode agreement of the raw transfer
    res = []
    for fold in (0, 1):
        tr = [i for i in range(len(rows)) if core[i] and t_names[i] and ep[i] % 2 == fold]
        te = [i for i in range(len(rows)) if core[i] and t_names[i] and ep[i] % 2 != fold]
        lab = [t_names[i] if i in set(tr) else None for i in range(len(rows))]
        C = centroids_for(lab, core, E, int(N.min_core_lines))
        names = sorted(C)
        if not names:
            continue
        M = np.stack([C[n] for n in names])
        for i in te:
            if t_names[i] in C:
                res.append((t_names[i], names[int((E[i] @ M.T).argmax())]))
    per = defaultdict(lambda: [0, 0])
    for a, b in res:
        per[a][0] += a == b
        per[a][1] += 1
    out["cross_episode"] = {"lines": len(res), "agreement": round(float(np.mean([a == b for a, b in res])), 4) if res else None,
                            "per_name": {n: {"lines": v[1], "agreement": round(v[0] / v[1], 3)} for n, v in sorted(per.items(), key=lambda kv: -kv[1][1])}}
    # ---- what happened to the transferred names
    fate = Counter()
    for i in range(len(rows)):
        if not t_names[i] or rows[i]["song"]:
            continue
        f = R["labels"][i]
        fate["kept" if f == t_names[i] else ("dropped" if f == UNKNOWN else "renamed")] += 1
    tot = sum(fate.values())
    out["transfer_fate"] = {**fate, "total": tot, "kept_share": round(fate["kept"] / tot, 4) if tot else None}
    fate_core = Counter()
    for i in np.where(core)[0]:
        if t_names[i] and t_names[i] not in GROUPS:
            f = R["labels"][i]
            fate_core["kept" if f == t_names[i] else ("dropped" if f == UNKNOWN else "renamed")] += 1
    tc = sum(fate_core.values())
    out["transfer_fate_core_named"] = {**fate_core, "total": tc, "kept_share": round(fate_core["kept"] / tc, 4) if tc else None}
    # ---- label-free clusters
    ci = np.where(core)[0]
    out["clusters"] = {}
    if len(ci) >= 50:
        Z = linkage(pdist(E[ci], metric="cosine"), method="average")
        for t in [float(x) for x in EN.purity.thresholds]:
            lab = fcluster(Z, t=t, criterion="distance")
            st = {"clusters": 0, "lines": 0, "final_majority": 0, "transfer_majority": 0, "transfer_lines": 0, "final_named": 0}
            for l in np.unique(lab):
                mem = ci[lab == l]
                if len(mem) < int(EN.purity.min_cluster_size):
                    continue
                fin = Counter(R["labels"][i] for i in mem if R["labels"][i] != UNKNOWN and R["labels"][i] not in GROUPS)
                tra = Counter(t_names[i] for i in mem if t_names[i] and t_names[i] not in GROUPS)
                st["clusters"] += 1
                st["lines"] += len(mem)
                st["final_named"] += sum(fin.values())
                st["final_majority"] += max(fin.values()) if fin else 0
                st["transfer_lines"] += sum(tra.values())
                st["transfer_majority"] += max(tra.values()) if tra else 0
            out["clusters"][f"t={t}"] = {"clusters": st["clusters"], "core_lines_in_clusters": st["lines"],
                                         "final_purity": round(st["final_majority"] / max(1, st["final_named"]), 4),
                                         "transfer_purity": round(st["transfer_majority"] / max(1, st["transfer_lines"]), 4)}
    # ---- recurring text (narrator introduction)
    seen: dict[str, set] = defaultdict(set)
    norm = [_NORM.sub("", r["text"].lower()).strip() for r in rows]
    for i, t in enumerate(norm):
        if len(t) >= 20 and not rows[i]["song"]:
            seen[t].add(rows[i]["ep"])
    rec = [i for i, t in enumerate(norm) if len(seen.get(t, ())) >= 5]
    if rec:
        c = Counter(R["labels"][i] for i in rec)
        top, n = c.most_common(1)[0]
        out["recurring_text"] = {"lines": len(rec), "episodes": len({rows[i]["ep"] for i in rec}), "modal_name": top,
                                 "modal_share": round(n / len(rec), 4), "names": dict(c.most_common(5)),
                                 "examples": sorted({rows[i]["text"] for i in rec})[:5]}
    # ---- margin of the final named core lines
    m = [(R["own_cos"][i] - R["best_other"][i]) for i in np.where(core)[0]
         if R["own_cos"][i] is not None and R["best_other"][i] is not None]
    if m:
        out["margin"] = {"core_named_lines": len(m), "share_margin_ge_0.10": round(float(np.mean(np.array(m) >= 0.10)), 4),
                         "share_margin_ge_0.05": round(float(np.mean(np.array(m) >= 0.05)), 4),
                         "median_margin": round(float(np.median(m)), 4)}
    return out


# ============================================================================================ 5. text check (LLM)
LLM_SCHEMA = {"type": "object", "required": ["speaker", "confidence"], "properties": {
    "speaker": {"type": "string"}, "confidence": {"type": "number"}, "second_guess": {"type": "string"}, "reason": {"type": "string"}}}


def llm_check(T: dict, R: dict, n_lines: int = 28) -> dict:
    """Independent check of every named voice from the TEXT of its lines (DeepSeek, text only, one small call per
    name, cached): the model is shown the lines closest to the name centroid WITHOUT the name and asked which
    character says them. Agreement = evidence that voice and name belong together; a confident disagreement means
    the voice belongs to somebody else."""
    from src.llm.client import LLM

    roster = [(c["name"], c.get("voice", "")) for c in CFG.characters.roster]
    names_txt = "\n".join(f"- {n}: {v}" for n, v in roster) + "\n- Other: a one-off adult / stranger / staff member (not in the list)"
    system = ("You identify who speaks lines of the ENGLISH DUB of the anime \"Komi Can't Communicate\" (the dub drops honorifics: "
              "\"Komi\", \"Tadano\"). You get lines that were all spoken by ONE voice (found by voice clustering; a few "
              "lines may belong to someone else). Decide which character this voice is from what is said, how others are "
              "addressed and the speaking style. Characters:\n" + names_txt +
              "\nReturn ONE JSON object {\"speaker\": <exact name from the list or \"Other\">, \"confidence\": 0..1, "
              "\"second_guess\": <name>, \"reason\": <one short sentence>}.")
    llm = LLM("deepseek", role="en_voice_names")
    rows, E, C = T["rows"], T["emb"], R["centroids"]
    out = {}
    for name in sorted(C):
        ix = [i for i in range(len(rows)) if R["labels"][i] == name and R["core"][i]]
        ix.sort(key=lambda i: -float(E[i] @ C[name]))
        pick, per_ep = [], Counter()
        for i in ix:                                    # the closest lines, at most 8 per episode
            if per_ep[rows[i]["ep"]] < 8:
                pick.append(i)
                per_ep[rows[i]["ep"]] += 1
            if len(pick) >= n_lines:
                break
        if len(pick) < 5:
            out[name] = {"skipped": "fewer than 5 core lines"}
            continue
        pick.sort(key=lambda i: (rows[i]["ep"], rows[i]["start"]))
        user = "Lines of one voice:\n" + "\n".join(f"[ep {rows[i]['ep']}] {rows[i]['text']}" for i in pick)
        try:
            ans = llm.chat([{"role": "system", "content": system}, {"role": "user", "content": user}], schema=LLM_SCHEMA,
                           max_tokens=300, temperature=0.0, tag=f"en_voice:{name}")
        except Exception as e:  # noqa: BLE001 - the check is optional
            out[name] = {"error": f"{type(e).__name__}: {e}"[:200]}
            continue
        guess = str(ans.get("speaker", "")).strip()
        out[name] = {"llm_speaker": guess, "confidence": round(float(ans.get("confidence", 0)), 2),
                     "second_guess": ans.get("second_guess"), "reason": str(ans.get("reason", ""))[:200],
                     "agree": guess.lower() == name.lower(), "agree_top2": name.lower() in (guess.lower(), str(ans.get("second_guess", "")).lower()),
                     "lines_shown": len(pick)}
    return out


def apply_llm_check(R: dict, check: dict, min_conf: float) -> dict:
    """A voice whose text says "somebody else" with confidence >= min_conf loses its name: its voice-named lines become
    that other name when it is a roster name without a centroid of its own, else "Other" (kept out of the voice bank
    as a character). Returns {old name: new name}."""
    roster = {c["name"] for c in CFG.characters.roster}
    renamed = {}
    for name, c in check.items():
        if c.get("agree") is not False or c.get("agree_top2") or float(c.get("confidence", 0)) < min_conf:
            continue
        guess = c["llm_speaker"]
        new = guess if (guess in roster and guess not in R["centroids"] and guess not in GROUPS) else "Other"
        renamed[name] = new
    for i, lab in enumerate(R["labels"]):
        if lab in renamed and R["source"][i] in ("transfer+spk", "spk"):
            R["labels"][i] = renamed[lab]
            R["source"][i] = "spk+llm"
            R["conf"][i] = round(min(R["conf"][i], 0.6), 3) if renamed[lab] != "Other" else 0.5
    for old, new in renamed.items():
        cen = R["centroids"].pop(old)
        if new != "Other":
            R["centroids"][new] = cen
    return renamed


# ============================================================================================ 6. discover unnamed voices
def _jp_cluster_names(rows: list[dict]) -> list[str | None]:
    """Per dub line: the NAME of the Japanese voice cluster (spk/clusters.json, name set by name_speakers) whose
    subtitle lines cover at least half of it - evidence that is independent of the per-line LLM names."""
    doc = read_json(PROC / "spk" / "clusters.json", default=None)
    if not doc:
        return [None] * len(rows)
    named = {int(c["id"]): str(c["name"]) for c in doc["clusters"] if c.get("name")}
    spans: dict[int, list] = defaultdict(list)
    subs_cache: dict[int, list] = {}
    for key, cid in doc["assignments"].items():
        ep, idx = int(key[2:4]), int(key.split(":")[1])
        if ep in HELDOUT_EPS or int(cid) not in named:
            continue
        if ep not in subs_cache:
            subs_cache[ep] = read_json(PROC / "subs" / f"{ep_name(ep)}.json", default=[]) or []
        if idx < len(subs_cache[ep]):
            e = subs_cache[ep][idx]
            spans[ep].append((float(e["start"]), float(e["end"]), named[int(cid)]))
    out = []
    for r in rows:
        ov: dict[str, float] = defaultdict(float)
        for s0, e0, n in spans.get(r["ep"], ()):
            o = min(e0, r["end"]) - max(s0, r["start"])
            if o > 0:
                ov[n] += o
        top = max(ov, key=ov.get) if ov else None
        out.append(top if top and ov[top] >= 0.5 * (r["end"] - r["start"]) else None)
    return out


def discover(T: dict, R: dict) -> dict:
    """Name voices that have no transferred seed. Returns {name: info}; R is updated in place."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist

    from src.llm.client import LLM

    D, N = EN.discover, EN.names
    rows, E, has, core_all = T["rows"], T["emb"], T["has"], R["core"]
    free = np.array([R["labels"][i] in (UNKNOWN, *GROUPS) and not rows[i]["song"] for i in range(len(rows))])
    ci = np.where(core_all & free)[0]
    if len(ci) < 30:
        return {}
    lab = fcluster(linkage(pdist(E[ci], metric="cosine"), method="average"), t=float(D.threshold), criterion="distance")
    jp_names = _jp_cluster_names(rows)
    first_ep = {}
    for c in (read_json(Path(CFG.paths.processed) / "cards" / "cards.json", default=[]) or []):
        try:
            first_ep[str(c["id"])] = int(c.get("first_episode"))
        except (TypeError, ValueError, KeyError):
            pass
    roster = [(c["name"], c.get("voice", "")) for c in CFG.characters.roster]
    valid = {n for n, _ in roster} - GROUPS - {"Narrator"}
    names_txt = "\n".join(f"- {n}: {v}" + (f" (first appears in episode {first_ep[n]})" if n in first_ep else "") for n, v in roster) \
        + "\n- Other: a one-off adult / stranger / staff member / announcer (not in the list)"
    system = ("You identify who speaks lines of the ENGLISH DUB of the anime \"Komi Can't Communicate\" (24 episodes; the dub "
              "drops honorifics). You get lines marked `>> VOICE:` that were all spoken by ONE voice (found by voice clustering; "
              "a few may belong to someone else, long sentences may be cut into fragments), each with the line before and "
              "after it as context (their speakers are given when known, `?` otherwise). Decide which character the marked "
              "voice is from what is said, who is addressed or answers, the episode numbers and the speaking style. "
              "Characters:\n" + names_txt +
              "\nAnswer \"Other\" unless the lines really point at one listed character. Return ONE JSON object "
              "{\"speaker\": <exact name from the list or \"Other\">, \"confidence\": 0..1, \"second_guess\": <name>, "
              "\"reason\": <one short sentence>}.")
    llm = LLM("deepseek", role="en_voice_names")
    by_ep: dict[int, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        by_ep[r["ep"]].append(i)
    pos = {}
    for ep, ix in by_ep.items():
        ix.sort(key=lambda i: rows[i]["start"])
        for k, i in enumerate(ix):
            pos[i] = (ep, k)
    n_words = np.array([len(re.findall(r"[A-Za-z']+", r["text"])) for r in rows])
    dur = np.array([r["end"] - r["start"] for r in rows])

    def ctx(i: int, off: int) -> str:
        ep, k = pos[i]
        ix = by_ep[ep]
        if not (0 <= k + off < len(ix)):
            return ""
        j = ix[k + off]
        if abs(rows[j]["start"] - rows[i]["start"]) > 12 or rows[j]["song"]:
            return ""
        who = R["labels"][j] if R["labels"][j] not in (UNKNOWN,) else "?"
        return f"   {who}: {rows[j]['text']}\n"

    cands = []
    for l, n in sorted(Counter(lab).items(), key=lambda kv: -kv[1]):
        if n < int(D.min_core_lines):
            continue
        mem = ci[lab == l]
        wps = float(n_words[mem].sum() / max(1e-6, dur[mem].sum()))
        if not (float(D.min_wps) <= wps <= float(D.max_wps)):
            continue                                  # mistimed fragments from many voices, not one speaker
        c = E[mem].mean(axis=0)
        c /= max(np.linalg.norm(c), 1e-9)
        order = mem[np.argsort(-(E[mem] @ c))]
        pick, per_ep = [], Counter()
        for i in order:
            if per_ep[rows[i]["ep"]] < 8:
                pick.append(int(i))
                per_ep[rows[i]["ep"]] += 1
            if len(pick) >= 18:
                break
        pick.sort(key=lambda i: (rows[i]["ep"], rows[i]["start"]))
        user = "Lines of one voice, with context:\n" + "\n".join(
            f"[ep {rows[i]['ep']}]\n{ctx(i, -1)}>> VOICE: {rows[i]['text']}\n{ctx(i, 1)}" for i in pick)
        try:
            ans = llm.chat([{"role": "system", "content": system}, {"role": "user", "content": user}], schema=LLM_SCHEMA,
                           max_tokens=300, temperature=0.0, tag="en_voice:discover")
        except Exception as e:  # noqa: BLE001
            log(f"discover: LLM call failed ({type(e).__name__}: {e})")
            continue
        guess, conf = str(ans.get("speaker", "")).strip(), float(ans.get("confidence", 0))
        jp = Counter(jp_names[i] for i in mem if jp_names[i])
        jp_share = (jp[guess] / len(mem)) if guess in jp else 0.0
        cands.append({"name": guess, "confidence": round(conf, 2), "core_lines": int(n), "seconds": round(float(sum(rows[i]["end"] - rows[i]["start"] for i in mem)), 1),
                      "episodes": dict(sorted(Counter(rows[i]["ep"] for i in mem).items())), "jp_cluster_share": round(jp_share, 3),
                      "reason": str(ans.get("reason", ""))[:200], "second_guess": ans.get("second_guess"),
                      "example": [rows[i]["text"] for i in pick[:3]], "_mem": mem, "_c": c})
    # a name claimed by several clusters with DIFFERENT voices is not evidence for any of them (the text of "dry
    # commentary" fits Kometani whoever speaks): such names are refused unless one cluster has Japanese-cluster support
    claims: dict[str, list[dict]] = defaultdict(list)
    for cd in cands:
        if cd["confidence"] >= float(D.min_conf) or cd["jp_cluster_share"] >= float(D.min_jp_share):
            claims[cd["name"]].append(cd)
    ambiguous = set()
    for n, cl in claims.items():
        if len(cl) > 1 and not any(c["jp_cluster_share"] >= float(D.min_jp_share) for c in cl):
            if any(float(a["_c"] @ b["_c"]) < float(D.merge_cos) for k, a in enumerate(cl) for b in cl[k + 1:]):
                ambiguous.add(n)
    found: dict[str, dict] = {}
    for cd in sorted(cands, key=lambda c: -(c["confidence"] + c["jp_cluster_share"]) * c["core_lines"]):
        if cd["name"] in ambiguous:
            cd["accepted"] = False
            cd["rejected"] = f"{len(claims[cd['name']])} different voices fit {cd['name']} by text"
            continue
        too_early = cd["name"] in first_ep and min(cd["episodes"]) < first_ep[cd["name"]] and \
            sum(v for e, v in cd["episodes"].items() if e < first_ep[cd["name"]]) > 0.2 * cd["core_lines"]
        ok = cd["name"] in valid and cd["name"] not in R["centroids"] and not too_early and \
            (cd["confidence"] >= float(D.min_conf) or cd["jp_cluster_share"] >= float(D.min_jp_share))
        if too_early:
            cd["rejected"] = f"lines before episode {first_ep[cd['name']]}, where {cd['name']} first appears"
        cd["accepted"] = bool(ok)
        if not ok:
            continue
        if cd["name"] in found:                        # a second cluster of the same new voice: merge when the voices agree
            f = found[cd["name"]]
            if float(f["_c"] @ cd["_c"]) >= float(D.merge_cos):
                f["_mem"] = np.concatenate([f["_mem"], cd["_mem"]])
                c = E[f["_mem"]].mean(axis=0)
                f["_c"] = c / max(np.linalg.norm(c), 1e-9)
                f["core_lines"] += cd["core_lines"]
                f["seconds"] = round(f["seconds"] + cd["seconds"], 1)
                f["merged_clusters"] = f.get("merged_clusters", 1) + 1
            else:
                cd["accepted"] = False
                cd["rejected"] = f"second cluster named {cd['name']} with a different voice (cos {float(f['_c'] @ cd['_c']):.2f})"
            continue
        found[cd["name"]] = cd
    # ---- new centroids, then assignment of the still-unnamed lines
    for n, f in found.items():
        R["centroids"][n] = f["_c"].astype(np.float32)
    if found:
        names = sorted(R["centroids"])
        M = np.stack([R["centroids"][n] for n in names])
        S = E @ M.T
        order = np.argsort(-S, axis=1)
        member_of = {int(i): n for n, f in found.items() for i in f["_mem"]}
        for i in range(len(rows)):
            if not free[i] or not has[i]:
                continue
            b1, s1 = names[order[i, 0]], float(S[i, order[i, 0]])
            s2 = float(S[i, order[i, 1]]) if len(names) > 1 else -1.0
            # a discovered centroid rests on one cluster: other lines join it only when long enough and clearly close
            new = member_of.get(i) or (b1 if (b1 in found and dur[i] >= 0.8 and s1 >= float(D.assign_min_cos)
                                              and s1 - s2 >= float(D.assign_margin)) else None)
            if new:
                own = float(S[i, names.index(new)])
                R["labels"][i], R["source"][i] = new, "spk+llm"
                R["conf"][i] = round(0.45 + 0.3 * min(1.0, max(0.0, (own - 0.35) / 0.35)), 3)
        # fragments of a halting speaker: an unnamed line between two lines of ONE discovered voice (same episode, gaps
        # <= fill_gap_sec) takes that name unless its own embedding clearly points at another voice
        idx_by_ep: dict[int, list[int]] = defaultdict(list)
        for i, r in enumerate(rows):
            idx_by_ep[r["ep"]].append(i)
        filled = Counter()
        for ep, ix in idx_by_ep.items():
            ix.sort(key=lambda i: rows[i]["start"])
            for _ in range(2):
                for k in range(1, len(ix) - 1):
                    i, a, b = ix[k], ix[k - 1], ix[k + 1]
                    if R["labels"][i] != UNKNOWN or rows[i]["song"]:
                        continue
                    n = R["labels"][a]
                    if n not in found or R["labels"][b] != n:
                        continue
                    if rows[i]["start"] - rows[a]["end"] > float(D.fill_gap_sec) or rows[b]["start"] - rows[i]["end"] > float(D.fill_gap_sec):
                        continue
                    if has[i]:
                        own = float(S[i, names.index(n)])
                        oth = float(np.max(np.delete(S[i], names.index(n))))
                        if oth >= float(D.fill_other_cos) and oth > own:
                            continue                 # e.g. the other party's "Yeah." in the middle of a phone call
                    R["labels"][i], R["source"][i], R["conf"][i] = n, "context", 0.45
                    filled[n] += 1
        for n in found:
            found[n]["lines"] = sum(1 for x in R["labels"] if x == n)
            found[n]["context_filled"] = int(filled[n])
    out = {"accepted": {n: {k: v for k, v in f.items() if not k.startswith("_")} for n, f in found.items()},
           "all_clusters": [{k: v for k, v in c.items() if not k.startswith("_")} for c in cands]}
    return out


def build(eps: list[int] | None = None, llm: bool = False, find_more: bool = False) -> dict:
    eps = [e for e in (eps or TRAIN_EPS) if e not in HELDOUT_EPS]
    T = load_table(eps)
    rows = T["rows"]
    if not rows:
        raise SystemExit("no dub lines yet (run src.anime.audio_en asr / lines / embed first)")
    t_names, t_confs, t_strong = transfer(rows)
    log(f"{len(rows)} dub lines in {len({r['ep'] for r in rows})} episodes; {sum(1 for n in t_names if n)} named by time overlap, "
        f"{int(T['has'].sum())} with a speaker embedding")
    R = refine(T, t_names, t_confs, t_strong)
    check, renamed = {}, {}
    if llm:
        check = llm_check(T, R)
        renamed = apply_llm_check(R, check, float(EN.names.llm_rename_conf))
        log(f"LLM text check: {sum(1 for c in check.values() if c.get('agree'))}/{len(check)} names confirmed; renamed {renamed}")
    disc = {}
    if find_more:
        disc = discover(T, R)
        log(f"discover: accepted {[(n, f['core_lines']) for n, f in disc.get('accepted', {}).items()]} "
            f"of {len(disc.get('all_clusters', []))} unnamed voice clusters")
    out_rows = [{"ep": r["ep"], "idx": r["idx"], "start": round(r["start"], 3), "end": round(r["end"], 3), "text": r["text"],
                 "speaker": R["labels"][i], "conf": R["conf"][i], "source": R["source"][i]} for i, r in enumerate(rows)]
    write_jsonl(OUT, out_rows)
    SPK_DIR.mkdir(parents=True, exist_ok=True)
    names = sorted(R["centroids"])
    np.save(SPK_DIR / "name_centroids.npy", np.stack([R["centroids"][n] for n in names]) if names else np.zeros((0, 256), np.float32))
    cnt, sec, by_src, eps_of = Counter(), Counter(), defaultdict(Counter), defaultdict(set)
    for o in out_rows:
        cnt[o["speaker"]] += 1
        sec[o["speaker"]] += o["end"] - o["start"]
        by_src[o["speaker"]][o["source"]] += 1
        eps_of[o["speaker"]].add(o["ep"])
    stats = {"lines": len(out_rows), "episodes": sorted({o["ep"] for o in out_rows}), "centroid_names": names,
             "dropped_names": R["dropped"], "name_support": R["support"], "llm_text_check": check, "llm_renamed": renamed, "discovered": disc,
             "per_speaker": {n: {"lines": cnt[n], "seconds": round(sec[n], 1), "episodes": len(eps_of[n]), "sources": dict(by_src[n]),
                                 "lines_ep15_21": sum(1 for o in out_rows if o["speaker"] == n and o["ep"] >= 15)}
                             for n, _ in cnt.most_common()},
             "transfer": {"named": sum(1 for n in t_names if n), "by_name": dict(Counter(n for n in t_names if n).most_common())},
             "purity": purity(T, t_names, R), "params": {"transfer": dict(EN.transfer), "names": dict(EN.names)}}
    write_json(SPK_DIR / "names_en.json", stats)
    log(f"-> {OUT} ({len(out_rows)} lines); stats {SPK_DIR / 'names_en.json'}")
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", default="train")
    ap.add_argument("--llm-check", action="store_true",
                    help="check every named voice against the text of its lines with DeepSeek (text only, ~1 small call per name)")
    ap.add_argument("--discover", action="store_true",
                    help="name the remaining unnamed voice clusters from their text (LLM) and Japanese voice-cluster evidence")
    a = ap.parse_args()
    from src.anime.audio_common import parse_episodes

    st = build(parse_episodes(a.episodes), llm=a.llm_check, find_more=a.discover)
    for n, v in st["per_speaker"].items():
        print(f"  {n:16s} lines={v['lines']:5d} sec={v['seconds']:7.1f} eps={v['episodes']:2d} ep15-21={v['lines_ep15_21']:4d} {v['sources']}")
    print("dropped:", st["dropped_names"])
    for n, v in st["name_support"].items():
        print(f"  support {n:16s} {v}")
    for n, v in st["llm_text_check"].items():
        print(f"  llm     {n:16s} {v}")
    print("llm renamed:", st["llm_renamed"])
    for c in st["discovered"].get("all_clusters", []):
        print(f"  cluster {'ACCEPT' if c.get('accepted') else 'reject'} {c['name']:14s} conf={c['confidence']} jp={c['jp_cluster_share']} core={c['core_lines']} "
              f"sec={c['seconds']} eps={c['episodes']} | {c['reason'][:110]}")
    import json

    print(json.dumps(st["purity"], indent=1)[:3000])


if __name__ == "__main__":
    main()
