"""Writing stage against the real episode: how much of the real adaptation's dialogue does the generated script have?

  python -m src.v3.eval_script --runs ep22_v3,ep22_v2 --episode 22
The real episode is used for evaluation only. For every chapter segment of the episode, the official English subtitle lines
(dialogue) are aligned in story order with the lines of the generated shot list (speech, thought, narration, notebook) using
sentence embeddings (BGE-M3, the same matcher that aligned the manga to the anime); both adapt the same chapter, with
different translations, so a pair counts when the meaning matches (cosine >= TAU), not the wording.
  real lines also in the script   - what the script kept of the real adaptation's dialogue
  script lines also in the episode - the rest was written for this version (added, expanded or reworded beyond recognition)
The manga bubbles themselves, aligned the same way when the anime was aligned, give the reference: how much of the MANGA
dialogue the real adaptation kept.
"""
from __future__ import annotations

import argparse

import numpy as np

from src.align.text_align import dp_align, embed, informative, norm_text
from src.common.config import P
from src.common.io import log, read_json, write_json

TAU = 0.62


def real_lines(episode: int, seg: dict) -> list[str]:
    rows = read_json(P("processed") / "anime" / "subs" / f"ep{episode:02d}.json") or []
    out = []
    for r in rows:
        if r.get("kind", "dialogue") != "dialogue" or not (seg["start"] - 0.5 <= r["start"] <= seg["end"] + 0.5):
            continue
        t = norm_text(str(r["text"]).replace("\\N", " "))
        if informative(t):
            out.append(t)
    return out


def script_lines(run: str, chapter: int, source: str = "shots") -> list[str]:
    """Lines of the generated version: from the shot list (what is recorded) or, with source="script", from the movie script."""
    out = []
    if source == "script":
        sc = read_json(P("runs") / run / "script" / f"c{chapter:03d}.json") or {}
        for s in sc.get("scenes", []):
            for b in s.get("beats", []):
                for l in b.get("lines") or []:
                    t = norm_text(str(l.get("text") or ""))
                    if l.get("kind") != "sfx" and informative(t):
                        out.append(t)
        return out
    for s in (read_json(P("runs") / run / "shotlist.json") or {}).get("shots", []):
        if s["chapter"] != chapter:
            continue
        for d in s.get("dialogue", []):
            t = norm_text(str(d.get("text") or ""))
            if d.get("kind") != "sfx" and informative(t):
                out.append(t)
    return out


def evaluate(run: str, episode: int, source: str = "shots") -> dict:
    segs = [s for s in read_json(P("processed") / "align" / "chapter_segments.json") if s.get("episode") == episode]
    rows, tot = [], {"real": 0, "script": 0, "real_matched": 0, "script_matched": 0, "real_words": 0, "script_words": 0}
    for seg in segs:
        R, G = real_lines(episode, seg), script_lines(run, seg["chapter"], source)
        if not R or not G:
            continue
        S = embed(R) @ embed(G).T
        _, pairs = dp_align(S, TAU)
        pairs = [(i, j) for i, j in pairs if S[i, j] >= TAU]
        r = {"chapter": seg["chapter"], "real": len(R), "script": len(G), "real_matched": len({i for i, _ in pairs}), "script_matched": len({j for _, j in pairs}),
             "real_words": sum(len(x.split()) for x in R), "script_words": sum(len(x.split()) for x in G),
             "manga_bubbles_in_real": f"{seg.get('n_matched')}/{seg.get('n_info')}",
             "examples_added": [G[j] for j in range(len(G)) if j not in {b for _, b in pairs}][:6],
             "examples_dropped": [R[i] for i in range(len(R)) if i not in {a for a, _ in pairs}][:6]}
        rows.append(r)
        for k in tot:
            tot[k] += r[k]
    out = {"run": run, "episode": episode, "tau": TAU, "per_chapter": rows, "total": {**tot, "real_kept_share": round(tot["real_matched"] / max(1, tot["real"]), 3),
                                                                                    "script_in_real_share": round(tot["script_matched"] / max(1, tot["script"]), 3)}}
    (P("runs") / run / "eval").mkdir(parents=True, exist_ok=True)
    write_json(P("runs") / run / "eval" / ("script_vs_real.json" if source == "shots" else "moviescript_vs_real.json"), out)
    t = out["total"]
    log(f"script vs real {run}: real lines also in the script {t['real_matched']}/{t['real']} ({t['real_kept_share']:.0%}); script lines also in the episode "
        f"{t['script_matched']}/{t['script']} ({t['script_in_real_share']:.0%}); words {t['script_words']} vs real {t['real_words']}")
    return out


def _kept(B: list[str], G: list[str], merge: int = 4) -> set[int]:
    """Bubbles of B that G keeps. One-to-one alignment in story order first; then a line of G may also stand for the
    bubbles next to its match when the manga broke one sentence over several balloons ("You know," / "it's embarrassing
    to say," / "getting chocolate from you" ...): the run of bubbles, joined, must still mean what the line says."""
    S = embed(B) @ embed(G).T
    _, pairs = dp_align(S, TAU)
    match = {i: j for i, j in pairs if S[i, j] >= TAU}
    kept = set(match)
    for i, j in sorted(match.items()):
        lo = hi = i
        cur = float(S[i, j])
        for _ in range(merge - 1):
            best = None
            for a_, b_ in ((lo - 1, hi), (lo, hi + 1)):
                if a_ < 0 or b_ >= len(B) or any(k in kept and k not in range(lo, hi + 1) for k in range(a_, b_ + 1)):
                    continue
                c = float(embed([" ".join(B[a_ : b_ + 1])])[0] @ embed([G[j]])[0])
                if c > cur + 0.01 and (best is None or c > best[0]):      # the neighbour must make the match BETTER, i.e. be part of the line
                    best = (c, a_, b_)
            if best is None:
                break
            cur, lo, hi = best
            kept.update(range(lo, hi + 1))
    return kept


def bubble_coverage(run: str, source: str = "shots", chapters: list[int] | None = None, episode: int | None = None) -> dict:
    """How much of the MANGA's dialogue a version keeps: informative bubbles of each chapter against the version's lines
    (meaning, not wording; cosine >= TAU; see _kept). run="real" measures the real adaptation the same way (its English
    subtitle lines inside the chapter's segment; evaluation only), so the two numbers can be put side by side."""
    from src.v3.pace import _bubbles

    segs = {s["chapter"]: s for s in read_json(P("processed") / "align" / "chapter_segments.json") if episode is None or s.get("episode") == episode}
    if run == "real":
        chapters = chapters or sorted(segs)
    chapters = chapters or sorted({s["chapter"] for s in (read_json(P("runs") / run / "shotlist.json") or {}).get("shots", [])})
    rows, tot = [], {"bubbles": 0, "kept": 0}
    for ch in chapters:
        B = [norm_text(t) for t in _bubbles().get(ch, [])]
        G = real_lines(segs[ch]["episode"], segs[ch]) if run == "real" else script_lines(run, ch, source)
        if not B or not G:
            continue
        kept = _kept(B, G)
        rows.append({"chapter": ch, "bubbles": len(B), "kept": len(kept), "dropped": [B[i] for i in range(len(B)) if i not in kept]})
        tot["bubbles"] += len(B)
        tot["kept"] += len(kept)
    out = {"run": run, "source": source, "per_chapter": rows, "total": {**tot, "share": round(tot["kept"] / max(1, tot["bubbles"]), 3)}}
    d = P("runs") / ("eval" if run == "real" else run + "/eval")
    d.mkdir(parents=True, exist_ok=True)
    write_json(d / ("bubbles_kept_real.json" if run == "real" else "bubbles_kept.json" if source == "shots" else "bubbles_kept_script.json"), out)
    log(f"manga bubbles kept, {run} ({source}): {tot['kept']}/{tot['bubbles']} ({out['total']['share']:.0%}); " +
        ", ".join(f"ch {r['chapter']} {r['kept']}/{r['bubbles']}" for r in rows))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--episode", type=int, default=22)
    ap.add_argument("--source", default="shots", choices=["shots", "script"])
    a = ap.parse_args()
    bubble_coverage("real", episode=a.episode)
    for run in a.runs.split(","):
        bubble_coverage(run, a.source)
        res = evaluate(run, a.episode, a.source)
        for r in res["per_chapter"]:
            print(f"   ch {r['chapter']}: real {r['real_matched']}/{r['real']} kept; script {r['script_matched']}/{r['script']} in the episode; "
                  f"manga bubbles the anime kept {r['manga_bubbles_in_real']}; words {r['script_words']} vs {r['real_words']}")
