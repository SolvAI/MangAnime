"""How the real anime adapts manga text, measured on TRAINING episodes only (strict split): word and time budgets for the writer.

  python -m src.v3.pace
For every manga chapter aligned to a training episode (data/processed/align/chapter_segments.json): the words printed in the
chapter's informative bubbles, the words of the episode's English subtitle lines inside the chapter's segment, and the
segment's length. The medians are what the writer is held to (src/v3/script.py: tighten).
"""
from __future__ import annotations

import collections
from functools import lru_cache

import numpy as np

from src.align.text_align import informative, norm_text
from src.common.config import CFG, P
from src.common.io import read_json, read_jsonl, write_json


@lru_cache(maxsize=1)
def _bubbles() -> dict[int, list[str]]:
    out = collections.defaultdict(list)
    for p in read_jsonl(P("processed") / "manga" / "panels.jsonl"):
        for d in p["dialogue"]:
            if d["kind"] != "sfx" and informative(norm_text(d["text"])):
                out[p["chapter"]].append(d["text"])
    return out


def manga_words(chapter: int) -> int:
    return sum(len(t.split()) for t in _bubbles().get(chapter, []))


def subtitle_lines(episode: int, start: float, end: float) -> list[str]:
    rows = read_json(P("processed") / "anime" / "subs" / f"ep{episode:02d}.json") or []
    out = [norm_text(str(r["text"]).replace("\\N", " ")) for r in rows if r.get("kind", "dialogue") == "dialogue" and start - 0.5 <= r["start"] <= end + 0.5]
    return [t for t in out if informative(t)]


@lru_cache(maxsize=1)
def adaptation_stats() -> dict:
    rows = []
    for s in read_json(P("processed") / "align" / "chapter_segments.json") or []:
        if s.get("episode") not in CFG.split.train_episodes or s.get("n_info", 0) < 20:
            continue
        mw = manga_words(s["chapter"])
        aw = sum(len(t.split()) for t in subtitle_lines(s["episode"], s["start"], s["end"]))
        if mw < 80 or aw < 80:
            continue
        rows.append({"chapter": s["chapter"], "episode": s["episode"], "manga_words": mw, "anime_words": aw, "seconds": round(s["end"] - s["start"], 1)})
    q = lambda v, p: round(float(np.percentile(v, p)), 3)          # noqa: E731
    wr = [r["anime_words"] / r["manga_words"] for r in rows]
    spw = [r["seconds"] / r["manga_words"] for r in rows]
    wps = [r["anime_words"] / r["seconds"] for r in rows]
    out = {"n_segments": len(rows), "episodes": "training only",
           "anime_words_per_manga_word": {"median": q(wr, 50), "p25": q(wr, 25), "p75": q(wr, 75)},
           "seconds_per_manga_word": {"median": q(spw, 50), "p25": q(spw, 25), "p75": q(spw, 75)},
           "anime_words_per_second": {"median": q(wps, 50), "p25": q(wps, 25), "p75": q(wps, 75)}}
    write_json(P("processed") / "v3" / "adaptation_stats.json", {**out, "segments": rows})
    return out


def budget(chapter: int) -> dict:
    """Word and time budget for one chapter, from the training medians."""
    st, mw = adaptation_stats(), manga_words(chapter)
    return {"manga_words": mw, "dialogue_words": int(round(mw * st["anime_words_per_manga_word"]["median"])),
            "seconds": int(round(mw * st["seconds_per_manga_word"]["median"]))}


if __name__ == "__main__":
    print(adaptation_stats())
    for ch in (114, 116, 117, 118):
        print(ch, budget(ch))
