"""Human results of a blind A/B experiment next to the automatic raters (un-blinds: run after rating).

  python scripts/human_ab_report.py [--experiment ep22_v2b_wan_vs_animatic] [--min-pairs 0]
"""
import argparse
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.config import P  # noqa: E402
from tools.ab_page.stats import fetch_ratings, is_automatic, sign_test_p, wilson_ci  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--experiment", default="ep22_v2b_wan_vs_animatic")
ap.add_argument("--min-pairs", type=int, default=0)
a = ap.parse_args()
rows = fetch_ratings(P("runs") / "ab" / "results.duckdb", a.experiment)
human = [r for r in rows if not is_automatic(r["rater"])]
pairs = {r["pair_id"] for r in human}
print(f"{a.experiment}: {len(pairs)} pairs rated by humans ({sorted({r['rater'] for r in human})})")
if len(pairs) < a.min_pairs:
    sys.exit(3)
for crit in sorted({r["criterion"] for r in human}):
    c = collections.Counter(r["winner_label"] or "tie" for r in human if r["criterion"] == crit)
    labels = [k for k in c if k != "tie"]
    x = "wan" if "wan" in c or not labels else labels[0]
    y = next((k for k in labels if k != x), "animatic")
    wx, wy, t = c.get(x, 0), c.get(y, 0), c.get("tie", 0)
    lo, hi = wilson_ci(wx, wx + wy) if wx + wy else (0, 0)
    print(f"  human   {crit:24s} {x} {wx:3d} : {y} {wy:3d}  tie {t:3d}  share {wx / max(1, wx + wy):.2f} [{lo:.2f}-{hi:.2f}]  p={sign_test_p(max(wx, wy), wx + wy):.3g}")
for rater in sorted({r["rater"] for r in rows if is_automatic(r["rater"])}):
    rs = [r for r in rows if r["rater"] == rater and r["criterion"] == "overall" and r["pair_id"] in pairs]
    c = collections.Counter(r["winner_label"] or "tie" for r in rs)
    hv = {r["pair_id"]: (r["winner_label"] or "tie") for r in human if r["criterion"] == "overall"}
    agree = sum(1 for r in rs if hv.get(r["pair_id"]) == (r["winner_label"] or "tie"))
    print(f"  on the same pairs  {rater:34s} {dict(c)}  agrees with the human on {agree}/{len(rs)}")
