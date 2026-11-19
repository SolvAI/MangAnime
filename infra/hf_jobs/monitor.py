#!/usr/bin/env python3
"""Monitor a Komi POC training job on Hugging Face Jobs (polls `hf jobs inspect` / `hf jobs logs`, hf CLI 2.2.0).

    . scripts/env.sh;  MP=$KOMI_WORK/venvs/musubi/bin/python
    $MP infra/hf_jobs/monitor.py --run wan22-i2v-high-v1                # loop until the job ends
    $MP infra/hf_jobs/monitor.py --run wan22-i2v-high-v1 --once         # one poll (for cron / STATUS.md updates)

Writes, under runs/train/<run>/:
    loss.csv        step,avr_loss,s_per_step,vram_gb,vram_peak_gb,gpu_util,job_min,seg   (one row per reported step)
    status.json     ONE line: stage, step/total, loss, s/step, ETA, VRAM peak, minutes, usd, last pushed step, verdict
    job_<id>.log    raw job log (last poll)
and, when the job reaches a terminal state, appends {ts, job_id, flavor, minutes, usd, run, stage} to runs/gpu_cost.jsonl.

Safety: the job is CANCELLED when the loss is NaN/inf or diverges (see logparse.divergence), when no step is reported
for --stall-min minutes, or when the cost passes --max-usd. Use --no-cancel to only report.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
import logparse  # noqa: E402
from src.train import common as C  # noqa: E402

TERMINAL = {"COMPLETED", "CANCELED", "CANCELLED", "ERROR", "DELETED", "TIMEOUT", "FAILED"}
CSV_COLS = ["step", "avr_loss", "s_per_step", "vram_gb", "vram_peak_gb", "gpu_util", "job_min", "seg"]


def hf_bin() -> str:
    b = os.environ.get("HF_BIN") or C.infra()["jobs"].get("hf_bin") or shutil.which("hf")
    if not b or not Path(b).exists():
        raise SystemExit("hf CLI not found (set HF_BIN)")
    return b


def hf(*args: str, timeout: int = 180) -> tuple[int, str]:
    try:
        out = subprocess.run([hf_bin(), *args], capture_output=True, text=True, timeout=timeout)
        return out.returncode, out.stdout + ("\n" + out.stderr if out.returncode else "")
    except subprocess.TimeoutExpired:
        return 124, ""


def inspect(job_id: str) -> dict | None:
    rc, out = hf("jobs", "inspect", "--json", job_id)
    if rc != 0:
        return None
    try:
        data = json.loads(out[out.index("["):])
        return data[0] if data else None
    except (ValueError, json.JSONDecodeError):
        return None


def fetch_logs(job_id: str) -> str:
    rc, out = hf("jobs", "logs", job_id, timeout=300)
    return out if rc == 0 else ""


def cancel(job_id: str) -> bool:
    rc, _ = hf("jobs", "cancel", job_id)
    return rc == 0


def load_rows(path: Path) -> dict[int, dict]:
    rows: dict[int, dict] = {}
    if path.exists():
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                try:
                    rows[int(r["step"])] = r
                except (KeyError, ValueError):
                    continue
    return rows


def write_rows(path: Path, rows: dict[int, dict]) -> None:
    tmp = path.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS, extrasaction="ignore")
        w.writeheader()
        for k in sorted(rows):
            w.writerow(rows[k])
    os.replace(tmp, path)


def clean(x):
    """NaN/inf are not valid JSON: report them as null (see `loss_is_nan`)."""
    return None if isinstance(x, float) and (math.isnan(x) or math.isinf(x)) else x


def poll(run_dir: Path, rec: dict, a, memo: dict) -> dict:
    job_id = rec["job_id"]
    info = inspect(job_id)
    stage = (info or {}).get("status", {}).get("stage", "UNKNOWN")
    dur = (info or {}).get("durations", {}) or {}
    estimated = False
    if dur.get("running_secs") is not None:
        running_min = float(dur["running_secs"]) / 60
        memo["max_running_min"] = max(memo.get("max_running_min", 0.0), running_min)
    else:
        # CANCELED jobs come back without `durations` / `finished_at`: estimate from started_at up to the moment the
        # terminal state was first seen (upper bound: one poll interval too long at most)
        estimated = True
        if stage in TERMINAL:
            memo.setdefault("terminal_seen", time.time())
        end = memo.get("terminal_seen", time.time())
        running_min = memo.get("max_running_min", 0.0)
        try:
            started = dt.datetime.fromisoformat((info or {})["started_at"]).timestamp()
            running_min = max(running_min, (end - started) / 60)
            if rec.get("timeout_min"):  # a job can never run longer than its timeout
                running_min = min(running_min, float(rec["timeout_min"]))
        except (KeyError, TypeError, ValueError):
            pass
    usd = running_min / 60 * float(rec.get("usd_per_hour", 0))

    text = fetch_logs(job_id)
    if text:
        (run_dir / f"job_{job_id}.log").write_text(text)
    metrics, events = logparse.parse_metrics(text)
    rows = load_rows(run_dir / "loss.csv")
    for m in metrics:
        if m.get("step") is None:
            continue
        row = {"step": m["step"], "avr_loss": "nan" if (m.get("avr_loss") is not None and clean(m.get("avr_loss")) is None) else m.get("avr_loss"), "s_per_step": m.get("tqdm_s_per_step") or m.get("s_per_step"),
               "vram_gb": m.get("vram_gb"), "vram_peak_gb": m.get("vram_peak_gb"), "gpu_util": m.get("gpu_util"),
               "job_min": m.get("job_min"), "seg": m.get("seg")}
        rows[int(m["step"])] = row
    if rows:
        write_rows(run_dir / "loss.csv", rows)

    last = metrics[-1] if metrics else None
    pushes = [e for e in events if e.get("event") == "pushed"]
    names = [e.get("event") for e in events]
    step = last["step"] if last else 0
    total = (last or {}).get("total") or rec.get("max_steps")
    sps = (last or {}).get("tqdm_s_per_step") or (last or {}).get("s_per_step")
    if step != memo.get("step"):
        memo["step"], memo["step_t"] = step, time.time()
    stalled_min = (time.time() - memo.get("step_t", time.time())) / 60

    verdict, reason = "ok", ""
    div = logparse.divergence(metrics, warmup_steps=a.warmup_steps, factor=a.div_factor, abs_limit=a.loss_limit)
    if "nan_loss" in names and not div:
        div = "NaN loss reported by the job"
    if div:
        verdict, reason = "diverged", div
    elif stage == "RUNNING" and stalled_min > (a.stall_min if step else a.startup_min):
        verdict, reason = "stalled", f"no new step for {stalled_min:.0f} min (step {step})"
    elif a.max_usd is not None and usd > a.max_usd:
        verdict, reason = "over_budget", f"${usd:.2f} > --max-usd {a.max_usd}"

    if verdict != "ok" and stage in ("RUNNING", "SCHEDULING") and not a.no_cancel:
        ok = cancel(job_id)
        memo["canceled"] = memo.get("canceled", False) or ok
        memo.setdefault("cancel_reason", f"{verdict}: {reason}")
        C.log(f"CANCEL {job_id}: {verdict} ({reason}) -> {'ok' if ok else 'FAILED, cancel it by hand!'}")
    canceled = memo.get("canceled", False)

    done = "job_done" in names
    status = {
        "run": rec["run"], "job_id": job_id, "stage": stage, "step": step, "total": total,
        "avr_loss": clean((last or {}).get("avr_loss")), "loss_is_nan": clean((last or {}).get("avr_loss")) is None and (last or {}).get("avr_loss") is not None,
        "s_per_step": sps,
        "eta_min": round((total - step) * sps / 60, 1) if (sps and total and step) else None,
        "vram_peak_gb": max([m.get("vram_peak_gb") or 0 for m in metrics], default=None) if metrics else None,
        "running_min": round(running_min, 2), "usd": round(usd, 4), "minutes_estimated": estimated, "flavor": rec.get("flavor"),
        "last_push_step": pushes[-1]["step"] if pushes else None, "n_pushes": len(pushes),
        "resumed": "resumed" in names, "job_done": done, "last_event": names[-1] if names else None,
        "verdict": verdict, "reason": reason, "canceled_by_monitor": canceled, "cancel_reason": memo.get("cancel_reason"),
        "updated": C.now_iso(),
        "url": (info or {}).get("url"),
    }
    tmp = run_dir / "status.json.tmp"
    tmp.write_text(json.dumps(status) + "\n")
    os.replace(tmp, run_dir / "status.json")
    return status


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--job-id", default=None, help="default: runs/train/<run>/job.json")
    p.add_argument("--interval", type=int, default=60)
    p.add_argument("--once", action="store_true")
    p.add_argument("--no-cancel", action="store_true")
    p.add_argument("--stall-min", type=float, default=30, help="cancel if no new step for this long once training runs")
    p.add_argument("--startup-min", type=float, default=30, help="cancel if no step at all this long after the first poll")
    p.add_argument("--warmup-steps", type=int, default=30)
    p.add_argument("--div-factor", type=float, default=3.0)
    p.add_argument("--loss-limit", type=float, default=2.0)
    p.add_argument("--max-usd", type=float, default=None)
    a = p.parse_args()
    C.load_env()
    run_dir = C.RUNS / "train" / a.run
    rec_path = run_dir / "job.json"
    if not rec_path.exists():
        raise SystemExit(f"{rec_path} not found (launch.py writes it)")
    rec = json.loads(rec_path.read_text())
    if a.job_id and a.job_id != rec["job_id"]:
        for line in (run_dir / "jobs.jsonl").read_text().splitlines():
            r = json.loads(line)
            if r["job_id"] == a.job_id:
                rec = r
                break
        else:
            raise SystemExit(f"job {a.job_id} is not in {run_dir / 'jobs.jsonl'}")
    memo: dict = {}
    while True:
        st = poll(run_dir, rec, a, memo)
        C.log(f"{st['stage']:<10} step {st['step']}/{st['total']} loss={st['avr_loss']} {st['s_per_step']} s/step "
              f"eta={st['eta_min']} min vram_peak={st['vram_peak_gb']} GB pushed@{st['last_push_step']} "
              f"{st['running_min']:.1f} min ${st['usd']:.2f} [{st['verdict']}]")
        if st["stage"] in TERMINAL:
            extra = {"run": rec["run"], "stage": st["stage"]}
            if st["minutes_estimated"]:
                extra["estimated"] = True
            C.append_cost(rec["job_id"], rec.get("flavor", "?"), st["running_min"], st["usd"], **extra)
            C.log(f"job ended: {st['stage']}; cost ${st['usd']:.4f} logged to {C.GPU_COST_LOG}"
                  + ("" if st["job_done"] else "  (run NOT complete: relaunch with --resume)" if st["last_push_step"] else ""))
            return 0 if st["stage"] == "COMPLETED" and st["verdict"] == "ok" else 1
        if a.once:
            return 0
        time.sleep(a.interval)


if __name__ == "__main__":
    sys.exit(main())
