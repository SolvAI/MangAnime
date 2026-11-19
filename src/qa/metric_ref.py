"""Where REAL TV-anime clips fall on the local QA metrics (reference ranges for the judge's evidence bundle).

  python -m src.qa.metric_ref build [--episode 20] [--n 80]      # CPU is fine (~8 s per clip)
Clips come from a TRAINING episode, cut and encoded exactly like generated clips, and are measured with the same
settings as src.qa.evidence.metrics_evidence. Output: data/processed/qa/real_metric_ref.json (percentiles per metric).
The judge sees, for each metric, the value and the real p5/p50/p95, plus a flag when the value is outside p1..p99.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.common.config import P
from src.common.io import log, read_json, write_json

REF = P("processed") / "qa" / "real_metric_ref.json"
# metric -> which direction is suspicious ("low": below the real range, "high": above, "both")
KEYS = {
    "dover.overall": "low", "dover.technical": "low", "dover.aesthetic": "low",
    "flicker.flicker_index": "high", "flicker.luma_flicker_p95": "high", "flicker.chroma_flicker": "high", "flicker.mean_abs_diff": "high",
    "motion.global_motion_mag": "high", "motion.flow_mag_rel_per_sec": "high", "motion.moving_area_ratio": "high", "motion.n_hard_cuts": "high",
    "warp.warp_error": "high", "warp.warp_error_p95": "high",
}
_CACHE: dict = {}


def _get(m: dict, key: str):
    a, b = key.split(".", 1)
    v = (m.get(a) or {}).get(b) if isinstance(m.get(a), dict) else None
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def build(episode: int = 20, n: int = 80) -> dict:
    from src.eval import video_metrics as vm
    from src.eval.episode_eval import real_clips

    clips = real_clips(episode, P("runs") / "eval_real" / f"ep{episode:02d}", n=150, seed=1)[:n]
    rows = read_json(REF.with_name("real_metric_rows.json")) or {}
    for i, c in enumerate(clips):
        if c.name in rows:
            continue
        m: dict = {}
        for group in (["flicker", "motion", "warp"], ["dover"]):
            try:
                r = vm.analyze_clip(c, metrics=group, max_pairs=12, flow_height=288, allow_static=True)
                m.update({k: v for k, v in r.items() if k in group})
            except Exception as e:
                m[group[0] + "_error"] = str(e)[:120]
        rows[c.name] = {k: _get(m, k) for k in KEYS}
        if i % 10 == 9:
            write_json(REF.with_name("real_metric_rows.json"), rows)
            log(f"metric ref {i + 1}/{len(clips)}")
    write_json(REF.with_name("real_metric_rows.json"), rows)
    ref = {"episode": episode, "n": len(rows), "metrics": {}}
    for k in KEYS:
        v = np.array([r[k] for r in rows.values() if r.get(k) is not None], float)
        if len(v) >= 10:
            ref["metrics"][k] = {f"p{q}": round(float(np.percentile(v, q)), 4) for q in (1, 5, 50, 95, 99)}
    REF.parent.mkdir(parents=True, exist_ok=True)
    write_json(REF, ref)
    log(f"metric reference: {len(rows)} real clips of episode {episode} -> {REF}")
    return ref


def versus_real(metrics: dict | None) -> dict:
    """{metric: {value, real_p5, real_p50, real_p95, flag}} for the judge; {} when no reference exists yet."""
    if "ref" not in _CACHE:
        _CACHE["ref"] = (read_json(REF) or {}).get("metrics", {})
    ref, out = _CACHE["ref"], {}
    if not ref or not isinstance(metrics, dict):
        return out
    for k, direction in KEYS.items():
        v, r = _get(metrics, k), ref.get(k)
        if v is None or not r:
            continue
        flag = "within real range"
        if direction in ("low", "both"):
            if v < r["p1"]:
                flag = "BELOW THE REAL RANGE"
            elif v < r["p5"]:
                flag = "low (below real p5)"
        if direction in ("high", "both"):
            if v > r["p99"]:
                flag = "ABOVE THE REAL RANGE"
            elif v > r["p95"]:
                flag = "high (above real p95)"
        out[k] = {"value": round(v, 4), "real_p5": r["p5"], "real_p50": r["p50"], "real_p95": r["p95"], "flag": flag}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build"])
    ap.add_argument("--episode", type=int, default=20)
    ap.add_argument("--n", type=int, default=80)
    a = ap.parse_args()
    build(a.episode, a.n)
