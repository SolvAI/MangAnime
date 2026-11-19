"""Parsers for musubi-tuner training output. Stdlib only: used locally (smoke summary, monitor.py) and inside the job
(train_job.py imports it as a sibling module).

musubi prints a tqdm bar named `steps` with an `avr_loss` postfix, e.g.

    steps:  12%|#2        | 6/50 [01:02<07:38, 10.42s/it, avr_loss=0.0834]

and the job supervisor re-emits machine-readable lines:

    KOMI_METRIC {"step": 6, "total": 50, "avr_loss": 0.0834, "s_per_step": 10.4, "vram_gb": 61.2, ...}
"""
from __future__ import annotations

import json
import math
import re

TQDM_RE = re.compile(
    r"steps:\s*(?P<pct>\d+)%\|[^|]*\|\s*(?P<step>\d+)/(?P<total>\d+)\s*"
    r"\[(?P<elapsed>[\d:]+)<(?P<eta>[\d:?]+),\s*(?P<rate>[\d.?]+)\s*(?P<unit>s/it|it/s)"
    r"(?:,\s*(?P<postfix>[^\]]*))?\]"
)
LOSS_RE = re.compile(r"avr_loss=(?P<loss>[-+0-9.eE]+|nan|inf|-inf)", re.I)
METRIC_PREFIX = "KOMI_METRIC "
EVENT_PREFIX = "KOMI_EVENT "


def _hms(s: str) -> float | None:
    if "?" in s:
        return None
    parts = [float(x) for x in s.split(":")]
    sec = 0.0
    for p in parts:
        sec = sec * 60 + p
    return sec


def parse_tqdm(fragment: str) -> dict | None:
    """One tqdm refresh -> {step, total, elapsed_s, s_per_step, avr_loss} or None."""
    m = TQDM_RE.search(fragment)
    if not m:
        return None
    rate = m.group("rate")
    s_per_step = None
    if "?" not in rate:
        r = float(rate)
        if r > 0:
            s_per_step = r if m.group("unit") == "s/it" else 1.0 / r
    loss = None
    lm = LOSS_RE.search(m.group("postfix") or "")
    if lm:
        try:
            loss = float(lm.group("loss"))
        except ValueError:
            loss = None
    return {"step": int(m.group("step")), "total": int(m.group("total")), "elapsed_s": _hms(m.group("elapsed")),
            "s_per_step": s_per_step, "avr_loss": loss}


def parse_progress(text: str) -> list[dict]:
    """All tqdm refreshes in a log (split on CR and LF), de-duplicated to the last refresh per step."""
    by_step: dict[int, dict] = {}
    for frag in re.split(r"[\r\n]+", text):
        if "steps:" not in frag:
            continue
        p = parse_tqdm(frag)
        if p and p["step"] > 0:
            by_step[p["step"]] = p
    return [by_step[k] for k in sorted(by_step)]


def parse_metrics(text: str) -> tuple[list[dict], list[dict]]:
    """KOMI_METRIC / KOMI_EVENT lines of a job log -> (metrics, events). Tolerates log prefixes before the marker."""
    metrics, events = [], []
    for line in text.splitlines():
        for prefix, sink in ((METRIC_PREFIX, metrics), (EVENT_PREFIX, events)):
            i = line.find(prefix)
            if i >= 0:
                try:
                    sink.append(json.loads(line[i + len(prefix):].strip()))
                except json.JSONDecodeError:
                    pass
    return metrics, events


def is_bad_loss(x) -> bool:
    return x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))


def divergence(metrics: list[dict], warmup_steps: int = 30, factor: float = 3.0, abs_limit: float = 2.0) -> str | None:
    """Return a reason string if the run looks broken, else None.

    - NaN/inf loss at any step;
    - moving-average loss above `abs_limit` after warm-up (flow-matching MSE of a healthy LoRA run stays well below 1);
    - moving-average loss more than `factor` x the best value seen after warm-up, for 3 consecutive reports.
    """
    best = None
    streak = 0
    for m in metrics:
        loss = m.get("avr_loss")
        if loss is None:
            continue
        if is_bad_loss(float(loss)):
            return f"NaN/inf loss at step {m.get('step')}"
        if m.get("step", 0) < warmup_steps:
            continue
        if loss > abs_limit:
            return f"loss {loss:.3f} > {abs_limit} at step {m.get('step')}"
        best = loss if best is None else min(best, loss)
        if loss > factor * best:
            streak += 1
            if streak >= 3:
                return f"loss {loss:.3f} is > {factor}x the best average ({best:.3f}) at step {m.get('step')}"
        else:
            streak = 0
    return None
