"""Alignment build (brief §5.3 steps 3–5): panel -> shots, visual gap-filling, silver-set evaluation, adaptation stats,
few-shot bank for the director.

  python -m src.align.build align      # alignment.jsonl (dialogue anchors + visual gap-filling)
  python -m src.align.build silver     # silver set, held-out precision/recall, confidence threshold
  python -m src.align.build stats      # adaptation_stats.json (TRAIN episodes only)
  python -m src.align.build fewshot    # fewshot_bank.json (TRAIN chapters with their real shot lists)
"""
from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from src.align.text_align import informative, norm_text
from src.common.config import CFG, P
from src.common.io import log, read_json, read_jsonl, write_json, write_jsonl
from src.qa.evidence import lineart

ALIGN = P("processed") / "align"


# ----------------------------------------------------------------------------------------------- loading
def load_shots(ep: int) -> list[dict]:
    return read_json(P("interim") / "anime" / f"ep{ep:02d}" / "shots_raw.json") or []


def kf_paths(ep: int, shot: int) -> list[str]:
    d = P("interim") / "anime" / f"ep{ep:02d}" / "keyframes"
    return [str(d / f"s{shot:04d}_{t}.jpg") for t in "amb"]


def shots_for_span(shots: list[dict], start: float, end: float) -> list[int]:
    """Shots that contain the line midpoint, plus shots covering >= 50 % of the line."""
    mid = (start + end) / 2
    out = []
    for s in shots:
        ov = min(end, s["end"]) - max(start, s["start"])
        if s["start"] <= mid < s["end"] or ov >= 0.5 * (end - start):
            out.append(s["shot"])
    return out


def panel_embs(panels: list[dict]) -> np.ndarray:
    from src.align.visual import embed_files

    return embed_files([p["image"] for p in panels], transform=lineart, tag="la")


def shot_embs(ep: int, shot_ids: list[int]) -> np.ndarray:
    """[n_shots, 3, D] line-art embeddings of the a/m/b keyframes."""
    from src.align.visual import embed_files

    paths = [p for s in shot_ids for p in kf_paths(ep, s)]
    e = embed_files(paths, transform=lineart, tag="la")
    return e.reshape(len(shot_ids), 3, -1)


def vis_sim(pe: np.ndarray, se: np.ndarray) -> np.ndarray:
    """[n_panels, n_shots] = max over the 3 keyframes of cosine(panel, keyframe)."""
    return np.einsum("pd,skd->psk", pe, se).max(-1)


# ----------------------------------------------------------------------------------------------- alignment
def gap_fill(sim: np.ndarray, thr: float) -> list[int | None]:
    """Monotonic assignment of consecutive silent panels (rows) to candidate shots (cols); None = no match."""
    n, m = sim.shape
    if n == 0 or m == 0:
        return [None] * n
    D = np.zeros((n + 1, m + 1), np.float32)
    B = np.zeros((n + 1, m + 1), np.int8)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            g = sim[i - 1, j - 1] - thr
            opts = (D[i - 1, j], D[i, j - 1], D[i - 1, j - 1] + g, D[i - 1, j] + g * 0.9)   # skip panel / skip shot / match / share shot
            k = int(np.argmax(opts))
            D[i, j], B[i, j] = opts[k], k
    out: list[int | None] = [None] * n
    i, j = n, m
    while i > 0 and j > 0:
        k = B[i, j]
        if k == 0:
            i -= 1
        elif k == 1:
            j -= 1
        elif k == 2:
            out[i - 1] = j - 1
            i, j = i - 1, j - 1
        else:
            out[i - 1] = j - 1
            i -= 1
    return out


def align_chapter(ch: int, seg: dict, panels: list[dict], anchors: list[dict], *, masked: set[int] | None = None,
                  vis_thr: float = 0.42, use_visual: bool = True) -> list[dict]:
    """Rows of alignment.jsonl for one chapter. `masked` panel orders are treated as silent (silver evaluation)."""
    ep = seg["episode"]
    shots = load_shots(ep)
    if not shots:
        return []
    in_seg = [s for s in shots if s["end"] > seg["start"] - 20 and s["start"] < seg["end"] + 20]
    if not in_seg:
        return []
    by_panel: dict[int, list[dict]] = defaultdict(list)
    for a in anchors:
        if a["chapter"] == ch and not (masked and a["panel_order"] in masked):
            by_panel[a["panel_order"]].append(a)
    sid = [s["shot"] for s in in_seg]
    pos = {s: i for i, s in enumerate(sid)}
    S = None
    if use_visual:
        try:
            S = vis_sim(panel_embs(panels), shot_embs(ep, sid))
        except Exception as e:
            log(f"chapter {ch}: visual embeddings unavailable ({str(e)[:100]})")
    rows: dict[int, dict] = {}
    for pi, p in enumerate(panels):
        an = by_panel.get(p["order"])
        if not an:
            continue
        cand = sorted({s for a in an for s in shots_for_span(shots, a["start"], a["end"]) if s in pos})
        if not cand:
            continue
        conf = max(a["sim"] for a in an)
        vis = float(max(S[pi, pos[s]] for s in cand)) if S is not None else None
        if vis is not None:   # visual agreement nudges the confidence up or down
            conf = float(np.clip(conf + 0.15 * (vis - 0.45), 0, 1))
        rows[p["order"]] = {"chapter": ch, "panel_order": p["order"], "shots": [{"episode": ep, "shot": s} for s in cand],
                            "match": "dialogue", "confidence": round(conf, 4), "text_sim": round(max(a["sim"] for a in an), 4),
                            "visual_sim": None if vis is None else round(vis, 4)}
    if S is not None:  # visual gap-filling between dialogue anchors, with order constraints
        orders = [p["order"] for p in panels]
        anchored = [i for i, o in enumerate(orders) if o in rows]
        bounds = [-1] + anchored + [len(orders)]
        for a, b in zip(bounds[:-1], bounds[1:]):
            gap = list(range(a + 1, b))
            if not gap:
                continue
            lo = pos[rows[orders[a]]["shots"][-1]["shot"]] if a >= 0 else 0
            hi = pos[rows[orders[b]]["shots"][0]["shot"]] if b < len(orders) else len(sid) - 1
            if hi < lo:
                lo, hi = hi, lo
            cols = list(range(lo, hi + 1))
            assign = gap_fill(S[np.ix_(gap, cols)], vis_thr)
            for gi, c in zip(gap, assign):
                if c is None:
                    continue
                v = float(S[gi, cols[c]])
                rows[orders[gi]] = {"chapter": ch, "panel_order": orders[gi], "shots": [{"episode": ep, "shot": sid[cols[c]]}],
                                    "match": "visual", "confidence": round(float(np.clip((v - 0.25) / 0.45, 0, 1)), 4),
                                    "text_sim": None, "visual_sim": round(v, 4)}
    return [rows[o] for o in sorted(rows)]


def build_alignment(use_visual: bool = True) -> list[dict]:
    segs = [s for s in read_json(ALIGN / "chapter_segments.json", []) if s.get("episode")]
    anchors = read_jsonl(ALIGN / "anchors.jsonl")
    panels_all = defaultdict(list)
    for p in read_jsonl(P("processed") / "manga" / "panels.jsonl"):
        panels_all[p["chapter"]].append(p)
    out = []
    for seg in segs:
        panels = sorted(panels_all[seg["chapter"]], key=lambda p: p["order"])
        rows = align_chapter(seg["chapter"], seg, panels, anchors, use_visual=use_visual)
        out += rows
        log(f"chapter {seg['chapter']} -> ep{seg['episode']:02d}: {len(rows)}/{len(panels)} panels aligned "
            f"({sum(1 for r in rows if r['match'] == 'visual')} visual)")
    write_jsonl(ALIGN / "alignment.jsonl", out)
    return out


# ----------------------------------------------------------------------------------------------- silver set
def silver(seed: int = 0, holdout: float = 0.25) -> dict:
    """Silver set = near-exact dialogue anchors with strong visual agreement. Hold part of it out (mask the dialogue
    signal), re-align, and score the visual gap-filling against it. Also sweeps the confidence threshold."""
    rows = read_jsonl(ALIGN / "alignment.jsonl")
    anchors = read_jsonl(ALIGN / "anchors.jsonl")
    segs = {s["chapter"]: s for s in read_json(ALIGN / "chapter_segments.json", []) if s.get("episode")}
    best_anchor = defaultdict(float)
    words = {}
    for a in anchors:
        k = (a["chapter"], a["panel_order"])
        if a["sim"] > best_anchor[k]:
            best_anchor[k] = a["sim"]
            words[k] = len(a["manga_text"].split())
    silver_rows = [r for r in rows if r["match"] == "dialogue" and best_anchor[(r["chapter"], r["panel_order"])] >= 0.85
                   and words.get((r["chapter"], r["panel_order"]), 0) >= 4 and (r["visual_sim"] or 0) >= 0.45 and len(r["shots"]) <= 2]
    write_jsonl(ALIGN / "silver.jsonl", silver_rows)
    rng = random.Random(seed)
    held = [r for r in silver_rows if rng.random() < holdout]
    by_ch = defaultdict(set)
    for r in held:
        by_ch[r["chapter"]].add(r["panel_order"])
    panels_all = defaultdict(list)
    for p in read_jsonl(P("processed") / "manga" / "panels.jsonl"):
        panels_all[p["chapter"]].append(p)
    truth = {(r["chapter"], r["panel_order"]): {s["shot"] for s in r["shots"]} for r in held}
    pred = {}
    for ch, masked in by_ch.items():
        panels = sorted(panels_all[ch], key=lambda p: p["order"])
        for r in align_chapter(ch, segs[ch], panels, anchors, masked=masked):
            if r["panel_order"] in masked:
                pred[(ch, r["panel_order"])] = (r["shots"][0]["shot"], r["confidence"], r["visual_sim"])
    res = {"n_silver": len(silver_rows), "n_heldout": len(held), "thresholds": []}
    for thr in (0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
        sel = {k: v for k, v in pred.items() if v[1] >= thr}
        exact = sum(1 for k, v in sel.items() if v[0] in truth[k])
        near = sum(1 for k, v in sel.items() if any(abs(v[0] - t) <= 1 for t in truth[k]))
        res["thresholds"].append({"conf>=": thr, "predicted": len(sel), "precision_exact": round(exact / max(1, len(sel)), 4),
                                  "precision_pm1": round(near / max(1, len(sel)), 4), "recall_exact": round(exact / max(1, len(held)), 4)})
    ok = [t for t in res["thresholds"] if t["precision_pm1"] >= 0.90 and t["predicted"] >= 10]
    res["chosen_visual_conf_threshold"] = ok[0]["conf>="] if ok else None
    res["note"] = ("Dialogue matches in the silver set are correct by construction; the held-out test measures the VISUAL "
                   "gap-filling, i.e. panels whose dialogue signal was removed. precision_pm1 counts a hit when the predicted "
                   "shot is the silver shot or its direct neighbour.")
    write_json(ALIGN / "silver_eval.json", res)
    log(f"silver: {res['n_silver']} pairs, held-out {res['n_heldout']}; " + str(res["thresholds"][:3]))
    return res


def vlm_spotcheck(n: int = 200, seed: int = 0) -> dict:
    """Local VLM answers 'does this anime frame depict this manga panel?' on random aligned pairs (brief §5.3.4)."""
    from concurrent.futures import ThreadPoolExecutor

    from src.llm.client import LLM

    rows = read_jsonl(ALIGN / "alignment.jsonl")
    panels = {(p["chapter"], p["order"]): p for p in read_jsonl(P("processed") / "manga" / "panels.jsonl")}
    rng = random.Random(seed)
    rng.shuffle(rows)
    rows = rows[:n]
    llm = LLM("local", role="spotcheck")
    schema = {"type": "object", "required": ["same_moment"], "properties": {"same_moment": {"type": "string"}, "reason": {"type": "string"}}}

    def one(r):
        p = panels[(r["chapter"], r["panel_order"])]
        s = r["shots"][0]
        frames = [f for f in kf_paths(s["episode"], s["shot"]) if Path(f).exists()][:2]
        try:
            res = llm.chat([{"role": "user", "content": "Image 1 is a manga panel. The other image(s) are frames of one anime shot. "
                             "Does the anime shot adapt the same story moment as the panel (same characters and action, allowing colour, "
                             "framing and background differences)? Answer JSON {\"same_moment\": \"yes|no|unsure\", \"reason\": \"...\"}."}],
                           schema=schema, images=[p["image"], *frames], max_tokens=120, temperature=0.0, image_max_side=512, tag="spotcheck")
            return {**r, "vlm": res.get("same_moment", "unsure")}
        except Exception as e:
            return {**r, "vlm": "error"}

    with ThreadPoolExecutor(int(CFG.llm.local.parallel)) as ex:
        out = list(ex.map(one, rows))
    summ = {}
    for kind in ("dialogue", "visual"):
        sub = [o for o in out if o["match"] == kind and o["vlm"] in ("yes", "no", "unsure")]
        if sub:
            summ[kind] = {"n": len(sub), "yes": round(sum(o["vlm"] == "yes" for o in sub) / len(sub), 3),
                          "no": round(sum(o["vlm"] == "no" for o in sub) / len(sub), 3)}
    for lo in (0.0, 0.3, 0.5, 0.7):
        sub = [o for o in out if o["confidence"] >= lo and o["vlm"] in ("yes", "no", "unsure")]
        if sub:
            summ[f"conf>={lo}"] = {"n": len(sub), "yes": round(sum(o["vlm"] == "yes" for o in sub) / len(sub), 3)}
    write_json(ALIGN / "vlm_spotcheck.json", {"summary": summ, "pairs": [{k: o[k] for k in ("chapter", "panel_order", "match", "confidence", "vlm")} for o in out]})
    log(f"vlm spot-check: {summ}")
    return summ


# ----------------------------------------------------------------------------------------------- stats + few-shot
def shot_meta(ep: int) -> dict[int, dict]:
    """Per-shot captions if they exist (data/processed/anime/shot_captions/epNN.jsonl)."""
    return {r["shot"]: r for r in read_jsonl(P("processed") / "anime" / "shot_captions" / f"ep{ep:02d}.jsonl")}


def stats() -> dict:
    train = set(CFG.split.train_episodes)
    rows = read_jsonl(ALIGN / "alignment.jsonl")
    segs = [s for s in read_json(ALIGN / "chapter_segments.json", []) if s.get("episode") in train]
    chapters = {c["chapter"]: c for c in read_json(P("processed") / "manga" / "chapters.json", [])}
    n_panels = Counter(p["chapter"] for p in read_jsonl(P("processed") / "manga" / "panels.jsonl"))
    seg_ch = {s["chapter"] for s in segs}
    rows = [r for r in rows if r["chapter"] in seg_ch and r["confidence"] >= 0.3]
    per_panel = [len(r["shots"]) for r in rows]
    aligned = Counter(r["chapter"] for r in rows)
    shot_d, line_d, sizes = [], [], Counter()
    per_ep_ch = Counter(s["episode"] for s in segs)
    sec_page = []
    for s in segs:
        shots = [x for x in load_shots(s["episode"]) if x["start"] >= s["start"] - 1 and x["end"] <= s["end"] + 1]
        shot_d += [x["end"] - x["start"] for x in shots]
        pages = (chapters.get(s["chapter"]) or {}).get("n_pages")
        if pages:
            sec_page.append((s["end"] - s["start"]) / pages)
        meta = shot_meta(s["episode"])
        for x in shots:
            if x["shot"] in meta and meta[x["shot"]].get("shot_size"):
                sizes[meta[x["shot"]]["shot_size"]] += 1
    for ep in train:
        for e in read_json(P("processed") / "anime" / "subs" / f"ep{ep:02d}.json", []):
            if e["kind"] == "dialogue":
                line_d.append(e["end"] - e["start"])
    sd = np.array(shot_d) if shot_d else np.array([3.0])
    tot = sum(sizes.values())
    n_shots_seg = len(shot_d)
    n_pan_seg = sum(n_panels[c] for c in seg_ch) or 1
    out = {
        "source": f"{len(segs)} aligned chapters in {len(per_ep_ch)} training episodes",
        "shots_per_panel": round(n_shots_seg / n_pan_seg, 3),
        "shots_per_aligned_panel": round(float(np.mean(per_panel)), 3) if per_panel else None,
        "panel_coverage": round(sum(aligned.values()) / n_pan_seg, 3),
        "panels_expanded_frac": round(float(np.mean([n >= 2 for n in per_panel])), 3) if per_panel else None,
        "panels_cut_frac": round(1 - sum(aligned.values()) / n_pan_seg, 3),
        "median_shot_sec": round(float(np.median(sd)), 2), "mean_shot_sec": round(float(sd.mean()), 2),
        "p10_shot_sec": round(float(np.percentile(sd, 10)), 2), "p90_shot_sec": round(float(np.percentile(sd, 90)), 2),
        "sec_per_dialogue_line": round(float(np.mean(line_d)), 2) if line_d else None,
        "sec_per_manga_page": round(float(np.median(sec_page)), 1) if sec_page else None,
        "chapters_per_episode": round(float(np.mean(list(per_ep_ch.values()))), 2) if per_ep_ch else None,
        "shot_size_mix": {k: round(v / tot, 3) for k, v in sizes.most_common()} if tot else {},
    }
    write_json(ALIGN / "adaptation_stats.json", out)
    log(f"adaptation stats: {out}")
    return out


def fewshot(max_shots: int = 60) -> list[dict]:
    """Per TRAIN chapter: the manga text and the real anime shot list (duration, size, camera, dialogue, source panels)."""
    from src.generate.director import render_chapter

    train = set(CFG.split.train_episodes)
    segs = [s for s in read_json(ALIGN / "chapter_segments.json", []) if s.get("episode") in train]
    rows = read_jsonl(ALIGN / "alignment.jsonl")
    shot_to_panels: dict[tuple, list[int]] = defaultdict(list)
    for r in rows:
        if r["confidence"] >= 0.3:
            for s in r["shots"]:
                shot_to_panels[(r["chapter"], s["shot"])].append(r["panel_order"])
    panels_all = defaultdict(list)
    for p in read_jsonl(P("processed") / "manga" / "panels.jsonl"):
        panels_all[p["chapter"]].append(p)
    named = defaultdict(dict)
    for l in read_jsonl(P("processed") / "anime" / "lines_named.jsonl"):
        named[l["ep"]][l["idx"]] = l["speaker"]
    anchor_spk: dict[tuple, str] = {}
    for a in read_jsonl(ALIGN / "anchors.jsonl"):
        sp = named.get(a["episode"], {}).get(a["line"])
        if sp and a["sim"] >= 0.6:
            anchor_spk[(a["chapter"], a["panel_order"], a["bubble"])] = sp
    bank = []
    for seg in segs:
        ep, ch = seg["episode"], seg["chapter"]
        panels = sorted(panels_all[ch], key=lambda p: p["order"])
        if not panels or seg.get("n_strong", 0) < 4:
            continue
        for p in panels:      # speakers of aligned bubbles come from the named subtitle line (train chapters only)
            for bi, d in enumerate(p["dialogue"]):
                if not d.get("speaker") and (ch, p["order"], bi) in anchor_spk:
                    d["speaker"] = anchor_spk[(ch, p["order"], bi)]
        subs = read_json(P("processed") / "anime" / "subs" / f"ep{ep:02d}.json", [])
        meta = shot_meta(ep)
        shots = [x for x in load_shots(ep) if x["end"] > seg["start"] and x["start"] < seg["end"]]
        ex_shots = []
        for x in shots[:max_shots]:
            m = meta.get(x["shot"], {})
            dlg = [{"speaker": named[ep].get(i, "?"), "text": e["text"], "kind": "notebook" if e["kind"] == "notebook" else "speech"}
                   for i, e in enumerate(subs) if e["kind"] in ("dialogue", "notebook") and x["start"] <= (e["start"] + e["end"]) / 2 < x["end"]]
            ex_shots.append({"source_panels": sorted(set(shot_to_panels.get((ch, x["shot"]), [])))[:3], "duration": round(x["end"] - x["start"], 1),
                             "shot_size": m.get("shot_size", ""), "camera": m.get("camera", ""), "characters": m.get("characters", []),
                             "action": m.get("action", ""), "dialogue": dlg})
        bank.append({"chapter": ch, "episode": ep, "n_panels": len(panels), "duration": round(seg["end"] - seg["start"], 1),
                     "summary_text": " ".join(d["text"] for p in panels for d in p["dialogue"] if d["kind"] != "sfx")[:4000],
                     "manga_text": render_chapter(panels), "shots": ex_shots, "captioned": bool(meta)})
    write_json(ALIGN / "fewshot_bank.json", bank)
    log(f"few-shot bank: {len(bank)} training chapters ({sum(b['captioned'] for b in bank)} with shot captions)")
    return bank


def fewshot_plan(chapters: list[int], k: int = 3) -> dict:
    """Which TRAIN chapters will be retrieved as few-shot examples for `chapters` (text similarity only), and which
    episode spans / manga chapters therefore need VLM captions first. Prints ready-to-use CLI arguments."""
    from src.align.text_align import embed

    train = set(CFG.split.train_episodes)
    segs = [s for s in read_json(ALIGN / "chapter_segments.json", []) if s.get("episode") in train and s.get("n_strong", 0) >= 4]
    panels_all = defaultdict(list)
    for p in read_jsonl(P("processed") / "manga" / "panels.jsonl"):
        panels_all[p["chapter"]].append(p)
    text = lambda ch: " ".join(d["text"] for p in sorted(panels_all[ch], key=lambda p: p["order"]) for d in p["dialogue"] if d["kind"] != "sfx")[:4000]
    cand = [s for s in segs if s["chapter"] not in chapters and panels_all[s["chapter"]]]
    if not cand:
        return {}
    E = embed([text(s["chapter"]) for s in cand])
    picks: dict[int, dict] = {}
    for ch in chapters:
        q = embed([text(ch)])[0]
        for i in np.argsort(-(E @ q))[:k]:
            picks[cand[int(i)]["chapter"]] = cand[int(i)]
    spans = ",".join(f"{s['episode']}:{max(0, s['start'] - 2):.0f}-{s['end'] + 2:.0f}" for s in picks.values())
    out = {"fewshot_chapters": sorted(picks), "manga_caption_arg": ",".join(str(c) for c in sorted(picks)), "shot_caption_spans": spans}
    write_json(ALIGN / "fewshot_plan.json", out)
    print(out)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["align", "silver", "spotcheck", "stats", "fewshot", "fewshot-plan", "all"])
    ap.add_argument("--no-visual", action="store_true")
    ap.add_argument("--chapters", default="")
    a = ap.parse_args()
    if a.cmd == "fewshot-plan":
        fewshot_plan([int(c) for c in a.chapters.split(",") if c])
        raise SystemExit(0)
    if a.cmd in ("align", "all"):
        build_alignment(use_visual=not a.no_visual)
    if a.cmd in ("silver", "all"):
        silver()
    if a.cmd == "spotcheck":
        vlm_spotcheck()
    if a.cmd in ("stats", "all"):
        stats()
    if a.cmd in ("fewshot", "all"):
        fewshot()
