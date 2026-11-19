"""Storage + statistics for blind A/B ratings (shared by the rating page and by automatic judges).

One DuckDB table, one row per (submission, criterion):

    ab_ratings(rating_id, experiment, pair_id, rater, criterion, choice, winner_label, label_a, label_b,
               left_side, ts, duration_ms, note, meta)

    choice        'a' | 'b' | 'tie'   - side of the manifest pair that was preferred
    winner_label  label_a / label_b / 'tie'
    left_side     'a' | 'b' - which manifest side was shown on the left (NULL when there was no layout)
    rater         a human name, or "judge:<name>" / "metric:<name>" for automatic raters

API
---
    record_rating(db_path, experiment, pair_id, rater, choices, label_a, label_b, ...) -> rating_id
    summarize(db_path, experiment) -> dict     (per-label win rates, ties, exact 95 % CI, sign-test p-value)
    binom_ci(k, n), sign_test_p(k, n)          (pure python)

DuckDB allows a single writer process per file. Every call here opens the file, works for a few milliseconds and
closes it again, retrying while another process holds it, so the web app and judge scripts can share one file.
"""
from __future__ import annotations

import contextlib
import datetime as _dt
import json
import math
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Iterable, Mapping

SCHEMA = """
CREATE TABLE IF NOT EXISTS ab_ratings (
    rating_id    VARCHAR NOT NULL,
    experiment   VARCHAR NOT NULL,
    pair_id      VARCHAR NOT NULL,
    rater        VARCHAR NOT NULL,
    criterion    VARCHAR NOT NULL,
    choice       VARCHAR NOT NULL,
    winner_label VARCHAR NOT NULL,
    label_a      VARCHAR NOT NULL,
    label_b      VARCHAR NOT NULL,
    left_side    VARCHAR,
    ts           TIMESTAMP NOT NULL,
    duration_ms  BIGINT,
    note         VARCHAR,
    meta         VARCHAR
);
"""
_LOCK = threading.RLock()
JUDGE_PREFIX = "judge:"
AUTO_PREFIXES = ("judge:", "metric:")       # rater names of automatic raters (LLM judges, metric-based raters)


def is_automatic(rater: str) -> bool:
    """True for automatic raters ("judge:<name>", "metric:<name>"); everything else is a human."""
    return str(rater).lower().startswith(AUTO_PREFIXES)


@contextlib.contextmanager
def connect(db_path: str | Path, retries: int = 60, wait: float = 0.1):
    """Short-lived read-write connection (schema ensured); retries while another process holds the file lock."""
    import duckdb

    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK:
        con = None
        last: Exception | None = None
        for i in range(retries):
            try:
                con = duckdb.connect(str(db_path))
                break
            except duckdb.IOException as e:          # "Could not set lock on file": another process is writing
                last = e
                time.sleep(wait * min(1 + i, 10))
        if con is None:
            raise RuntimeError(f"could not open {db_path}: {last}")
        try:
            con.execute(SCHEMA)
            yield con
        finally:
            con.close()


def record_rating(db_path: str | Path, experiment: str, pair_id: str, rater: str, choices: Mapping[str, str],
                  label_a: str, label_b: str, left_side: str | None = None, duration_ms: int | None = None,
                  note: str | None = None, meta: Any = None, ts: _dt.datetime | None = None) -> str:
    """Store one rating of one pair (one row per criterion). Returns the rating_id.

    choices: {criterion: 'a' | 'b' | 'tie'}; the two labels themselves are accepted as well
             (e.g. {"overall": "wan"} with label_a="wan").
    rater:   human name, or "judge:<name>" / "metric:<name>" for an automatic rater (use the same experiment /
             pair_id / labels as the manifest the humans rate). Automatic raters are reported apart from humans.
    """
    if not choices:
        raise ValueError("choices is empty")
    if label_a == label_b:
        raise ValueError("label_a and label_b must differ")
    if left_side not in (None, "a", "b"):
        raise ValueError("left_side must be 'a', 'b' or None")
    rater = str(rater).strip()
    if not rater:
        raise ValueError("rater is empty")
    rows = []
    rid = uuid.uuid4().hex
    ts = ts or _dt.datetime.now()
    for crit, ch in choices.items():
        ch = str(ch)
        if ch == label_a:
            ch = "a"
        elif ch == label_b:
            ch = "b"
        ch = ch.lower()
        if ch not in ("a", "b", "tie"):
            raise ValueError(f"choice for {crit!r} must be 'a', 'b', 'tie' or one of the labels, got {ch!r}")
        winner = label_a if ch == "a" else label_b if ch == "b" else "tie"
        rows.append((rid, experiment, str(pair_id), rater, str(crit), ch, winner, label_a, label_b, left_side, ts,
                     None if duration_ms is None else int(duration_ms), note,
                     None if meta is None else json.dumps(meta, ensure_ascii=False, default=str)))
    with connect(db_path) as con:
        con.executemany("INSERT INTO ab_ratings VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
    return rid


def delete_rating(db_path: str | Path, rating_id: str) -> None:
    with connect(db_path) as con:
        con.execute("DELETE FROM ab_ratings WHERE rating_id = ?", [rating_id])


def fetch_ratings(db_path: str | Path, experiment: str | None = None, latest_only: bool = True) -> list[dict]:
    """All rows (optionally of one experiment). latest_only keeps the newest row per (experiment, rater, pair, criterion)."""
    if not Path(db_path).exists():
        return []
    q = "SELECT * FROM ab_ratings"
    args: list[Any] = []
    if experiment is not None:
        q += " WHERE experiment = ?"
        args.append(experiment)
    q += " ORDER BY ts, rating_id"
    with connect(db_path) as con:
        cur = con.execute(q, args)
        cols = [d[0] for d in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    if latest_only:
        last: dict[tuple, dict] = {}
        for r in rows:                         # rows are in time order: later ones overwrite earlier ones
            last[(r["experiment"], r["rater"], r["pair_id"], r["criterion"])] = r
        rows = list(last.values())
    return rows


def rated_pairs(db_path: str | Path, experiment: str, rater: str) -> set[str]:
    """pair_ids this rater has already rated in this experiment (for resume)."""
    return {r["pair_id"] for r in fetch_ratings(db_path, experiment) if r["rater"] == rater}


# ----------------------------------------------------------------------------------------------------------------
# statistics (pure python)
# ----------------------------------------------------------------------------------------------------------------
def _binom_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k), X ~ Bin(n, p)."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 0.0
    lp, lq = math.log(p), math.log1p(-p)
    s = 0.0
    for i in range(0, k + 1):
        s += math.exp(math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * lp + (n - i) * lq)
    return min(1.0, s)


def sign_test_p(k: int, n: int) -> float:
    """Two-sided exact sign test: probability of a split at least as uneven as k of n under p = 0.5."""
    if n <= 0:
        return 1.0
    lo = min(k, n - k)
    return min(1.0, 2.0 * _binom_cdf(lo, n, 0.5))


def _bisect(f, target: float, increasing: bool) -> float:
    """p in [0, 1] with f(p) = target, for a monotone f."""
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if (f(mid) < target) == increasing:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def binom_ci(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    """Exact (Clopper-Pearson) confidence interval of a binomial proportion k / n."""
    if n <= 0:
        return (0.0, 1.0)
    alpha = (1.0 - conf) / 2.0
    # lower bound: P(X >= k | p) = alpha (increasing in p); upper bound: P(X <= k | p) = alpha (decreasing in p)
    lower = 0.0 if k <= 0 else _bisect(lambda p: 1.0 - _binom_cdf(k - 1, n, p), alpha, increasing=True)
    upper = 1.0 if k >= n else _bisect(lambda p: _binom_cdf(k, n, p), alpha, increasing=False)
    return (lower, upper)


def wilson_ci(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def _comparison(x: str, y: str, wins_x: int, wins_y: int, ties: int) -> dict:
    n_dec = wins_x + wins_y
    lo, hi = binom_ci(wins_x, n_dec)
    wl, wh = wilson_ci(wins_x, n_dec)
    return {
        "label_x": x, "label_y": y, "wins_x": wins_x, "wins_y": wins_y, "ties": ties, "n": n_dec + ties,
        "win_rate_x": (wins_x / n_dec) if n_dec else None,            # ties excluded
        "ci95": [lo, hi] if n_dec else None,                          # exact binomial (Clopper-Pearson)
        "ci95_wilson": [wl, wh] if n_dec else None,
        "p_sign": sign_test_p(wins_x, n_dec),                         # two-sided exact sign test, ties dropped
        "tie_rate": (ties / (n_dec + ties)) if (n_dec + ties) else None,
        "winner": x if wins_x > wins_y else y if wins_y > wins_x else None,
        "significant": bool(n_dec and sign_test_p(wins_x, n_dec) < 0.05),
    }


def _aggregate(rows: Iterable[dict]) -> dict:
    """{criterion: [comparison, ...]} pooling every row as one observation."""
    rows = list(rows)
    # orientation of each comparison: label_x is the label that is usually given as label_a ("wan" in wan-vs-animatic)
    as_a: dict[tuple[str, str], dict[str, int]] = {}
    for r in rows:
        key = tuple(sorted((r["label_a"], r["label_b"])))
        as_a.setdefault(key, {}).setdefault(r["label_a"], 0)
        as_a[key][r["label_a"]] += 1
    orient = {}
    for key, cnt in as_a.items():
        x = max(key, key=lambda lab: (cnt.get(lab, 0), -key.index(lab)))
        orient[key] = (x, key[1] if x == key[0] else key[0])
    acc: dict[str, dict[tuple[str, str], list[int]]] = {}
    for r in rows:
        x, y = orient[tuple(sorted((r["label_a"], r["label_b"])))]
        c = acc.setdefault(r["criterion"], {}).setdefault((x, y), [0, 0, 0])
        if r["winner_label"] == x:
            c[0] += 1
        elif r["winner_label"] == y:
            c[1] += 1
        else:
            c[2] += 1
    return {crit: [_comparison(x, y, *v) for (x, y), v in sorted(d.items())] for crit, d in acc.items()}


def _majority_rows(rows: list[dict]) -> list[dict]:
    """One pseudo-row per (pair, criterion): the majority winner over raters (tie if no majority)."""
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault((r["pair_id"], r["criterion"], *sorted((r["label_a"], r["label_b"]))), []).append(r)
    out = []
    for (pair, crit, x, y), rs in groups.items():
        wx = sum(r["winner_label"] == x for r in rs)
        wy = sum(r["winner_label"] == y for r in rs)
        out.append({"pair_id": pair, "criterion": crit, "label_a": rs[-1]["label_a"], "label_b": rs[-1]["label_b"],
                    "winner_label": x if wx > wy else y if wy > wx else "tie"})
    return out


def summarize(db_path: str | Path, experiment: str, criteria: Iterable[str] | None = None) -> dict:
    """Win rates of an experiment.

    Returns
    {
      "experiment", "n_rows", "n_ratings" (submissions), "n_pairs_rated",
      "raters": {name: n_pairs},
      "groups": {
         "all" | "human" | "judge" (= automatic raters: names starting with "judge:" or "metric:"): {
            "n_ratings", "n_raters",
            "criteria": {criterion: [ {label_x, label_y, wins_x, wins_y, ties, n, win_rate_x, ci95, ci95_wilson,
                                        p_sign, tie_rate, winner, significant}, ... one per label pair ]},
            "criteria_by_pair_majority": same structure, one vote per pair (majority over raters)
         }},
      "by_rater": {rater: {criterion: [comparison, ...]}},
      "position_bias": {"left", "right", "ties", "left_rate", "p_sign"}        # humans only; should be ~0.5
      "judge_agreement": {judge: {"n", "agree", "rate"}}     # vs the human majority, per (pair, criterion), no ties
    }
    Only the latest rating per (rater, pair, criterion) counts. win_rate_x excludes ties; ci95 is the exact
    binomial interval of that rate; p_sign is the two-sided exact sign test against 50/50.
    """
    rows = fetch_ratings(db_path, experiment)
    if criteria is not None:
        keep = set(criteria)
        rows = [r for r in rows if r["criterion"] in keep]
    is_judge = lambda r: is_automatic(r["rater"])  # noqa: E731
    groups = {"all": rows, "human": [r for r in rows if not is_judge(r)], "judge": [r for r in rows if is_judge(r)]}
    out: dict[str, Any] = {
        "experiment": experiment, "n_rows": len(rows),
        "n_ratings": len({r["rating_id"] for r in rows}),
        "n_pairs_rated": len({r["pair_id"] for r in rows}),
        "raters": {}, "groups": {}, "by_rater": {},
    }
    per_rater: dict[str, list[dict]] = {}
    for r in rows:
        per_rater.setdefault(r["rater"], []).append(r)
    out["raters"] = {k: len({r["pair_id"] for r in v}) for k, v in sorted(per_rater.items())}
    for g, rs in groups.items():
        out["groups"][g] = {
            "n_ratings": len({r["rating_id"] for r in rs}), "n_raters": len({r["rater"] for r in rs}),
            "criteria": _aggregate(rs), "criteria_by_pair_majority": _aggregate(_majority_rows(rs)),
        }
    out["by_rater"] = {k: _aggregate(v) for k, v in sorted(per_rater.items())}

    # position bias (humans): how often was the left video chosen?
    left = right = ties = 0
    for r in groups["human"]:
        if r["left_side"] not in ("a", "b"):
            continue
        if r["choice"] == "tie":
            ties += 1
        elif r["choice"] == r["left_side"]:
            left += 1
        else:
            right += 1
    out["position_bias"] = {"left": left, "right": right, "ties": ties,
                            "left_rate": left / (left + right) if left + right else None,
                            "p_sign": sign_test_p(left, left + right)}

    # judges vs the human majority
    hum = {(r["pair_id"], r["criterion"]): r["winner_label"] for r in _majority_rows(groups["human"])}
    agree: dict[str, dict[str, Any]] = {}
    for r in groups["judge"]:
        h = hum.get((r["pair_id"], r["criterion"]))
        if h is None or h == "tie" or r["winner_label"] == "tie":
            continue
        a = agree.setdefault(r["rater"], {"n": 0, "agree": 0})
        a["n"] += 1
        a["agree"] += int(h == r["winner_label"])
    for a in agree.values():
        a["rate"] = a["agree"] / a["n"] if a["n"] else None
    out["judge_agreement"] = agree
    return out


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Print the A/B summary of an experiment as JSON.")
    ap.add_argument("--db", required=True)
    ap.add_argument("--experiment", required=True)
    a = ap.parse_args(argv)
    print(json.dumps(summarize(a.db, a.experiment), indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
