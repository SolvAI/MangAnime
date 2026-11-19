"""Director evaluation (brief §10): generated shot list vs the real adaptation of the same held-out chapters.

Metrics: shot count, duration distribution, shot-size mix, panel coverage, dialogue coverage.
  python -m src.eval.director_eval --run ep23_base
"""
from __future__ import annotations

import argparse
from collections import Counter

import numpy as np

from src.common.config import P
from src.common.io import log, read_json, read_jsonl, write_json


def _dist(x: list[float]) -> dict:
    a = np.array(x) if x else np.array([0.0])
    return {"n": len(x), "total": round(float(a.sum()), 1), "mean": round(float(a.mean()), 2), "median": round(float(np.median(a)), 2),
            "p10": round(float(np.percentile(a, 10)), 2), "p90": round(float(np.percentile(a, 90)), 2)}


def ks_stat(a: list[float], b: list[float]) -> float:
    a, b = np.sort(a), np.sort(b)
    if len(a) == 0 or len(b) == 0:
        return 1.0
    grid = np.concatenate([a, b])
    return float(np.max(np.abs(np.searchsorted(a, grid, side="right") / len(a) - np.searchsorted(b, grid, side="right") / len(b))))


def wasserstein(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return float("nan")
    q = np.linspace(0, 1, 101)
    return float(np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q))))


def mix(counter: Counter) -> dict:
    t = sum(counter.values()) or 1
    return {k: round(v / t, 3) for k, v in counter.most_common()}


def tv_distance(a: dict, b: dict) -> float:
    keys = set(a) | set(b)
    return round(0.5 * sum(abs(a.get(k, 0) - b.get(k, 0)) for k in keys), 3)


def evaluate(run: str) -> dict:
    rd = P("runs") / run
    gen = (read_json(rd / "shotlist_director.json") or read_json(rd / "shotlist.json") or {}).get("shots", [])
    final = (read_json(rd / "shotlist.json") or {}).get("shots", [])
    chapters = sorted({s["chapter"] for s in gen})
    segs = {s["chapter"]: s for s in read_json(P("processed") / "align" / "chapter_segments.json", []) if s.get("episode")}
    panels = [p for p in read_jsonl(P("processed") / "manga" / "panels.jsonl") if p["chapter"] in chapters]
    align = [r for r in read_jsonl(P("processed") / "align" / "alignment.jsonl") if r["chapter"] in chapters and r["confidence"] >= 0.3]
    anchors = [a for a in read_jsonl(P("processed") / "align" / "anchors.jsonl") if a["chapter"] in chapters]
    out = {"run": run, "chapters": chapters, "per_chapter": []}
    real_d_all, gen_d_all, real_sizes, gen_sizes = [], [], Counter(), Counter()
    for ch in chapters:
        g = [s for s in gen if s["chapter"] == ch and "TITLE_CARD" not in (s.get("notes") or "")]
        gf = [s for s in final if s["chapter"] == ch and "TITLE_CARD" not in (s.get("notes") or "")]
        n_pan = sum(1 for p in panels if p["chapter"] == ch)
        row = {"chapter": ch, "n_panels": n_pan, "gen_shots": len(g), "gen_sec_director": round(sum(s["duration"] for s in g), 1),
               "gen_sec_final": round(sum(s["duration"] for s in gf), 1),
               "gen_panel_coverage": round(len({o for s in g for o in s.get("source_panels", [])}) / max(1, n_pan), 3)}
        gen_d_all += [s["duration"] for s in gf or g]
        for s in g:
            gen_sizes[s.get("shot_size", "")] += 1
        seg = segs.get(ch)
        if seg:
            ep = seg["episode"]
            shots = [x for x in (read_json(P("interim") / "anime" / f"ep{ep:02d}" / "shots_raw.json") or []) if x["end"] > seg["start"] and x["start"] < seg["end"]]
            caps = {r["shot"]: r for r in read_jsonl(P("processed") / "anime" / "shot_captions" / f"ep{ep:02d}.jsonl")}
            real_d = [x["end"] - x["start"] for x in shots]
            real_d_all += real_d
            for x in shots:
                if x["shot"] in caps:
                    real_sizes[caps[x["shot"]].get("shot_size", "")] += 1
            row.update({"episode": ep, "real_shots": len(shots), "real_sec": round(seg["end"] - seg["start"], 1),
                        "real_panel_coverage": round(len({r["panel_order"] for r in align if r["chapter"] == ch}) / max(1, n_pan), 3),
                        "real_dialogue_coverage": round(len({(a["panel_order"], a["bubble"]) for a in anchors if a["chapter"] == ch})
                                                        / max(1, sum(len([d for d in p["dialogue"] if d["kind"] != "sfx"]) for p in panels if p["chapter"] == ch)), 3)})
        n_bub = sum(len([d for d in p["dialogue"] if d["kind"] != "sfx"]) for p in panels if p["chapter"] == ch)
        row["gen_dialogue_lines"] = sum(len(s["dialogue"]) for s in g)
        row["manga_bubbles"] = n_bub
        out["per_chapter"].append(row)
    gm, rm = mix(gen_sizes), mix(real_sizes)
    out["summary"] = {
        "shot_count": {"generated": len(gen_d_all), "real": len(real_d_all)},
        "duration_generated": _dist(gen_d_all), "duration_real": _dist(real_d_all),
        "duration_ks": round(ks_stat(gen_d_all, real_d_all), 3), "duration_wasserstein_sec": round(wasserstein(gen_d_all, real_d_all), 3),
        "shot_size_mix_generated": gm, "shot_size_mix_real": rm, "shot_size_tv_distance": tv_distance(gm, rm) if rm else None,
        "panel_coverage": {"generated": round(float(np.mean([r["gen_panel_coverage"] for r in out["per_chapter"]])), 3),
                           "real": round(float(np.mean([r["real_panel_coverage"] for r in out["per_chapter"] if "real_panel_coverage" in r] or [0])), 3)},
    }
    write_json(rd / "eval" / "director_eval.json", out)
    log(f"director eval: {out['summary']['shot_count']}, KS={out['summary']['duration_ks']}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    evaluate(ap.parse_args().run)
