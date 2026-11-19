"""Collect the local smoke-test results into docs/training.md (section between the LOCAL_SMOKE markers).

    venvs/musubi/bin/python -m src.train.smoke_report

Reads runs/train/smoke_wan22_i2v_{high,low}/smoke_result.json and runs/train/smoke_kf_qie2511/smoke_result.json.
"""
from __future__ import annotations

import json
import time

from . import common as C

BEGIN, END = "<!-- LOCAL_SMOKE_BEGIN -->", "<!-- LOCAL_SMOKE_END -->"


def main() -> None:
    rows = []
    for name, label in (("smoke_wan22_i2v_high", "Wan 2.2 I2V high-noise"), ("smoke_wan22_i2v_low", "Wan 2.2 I2V low-noise"),
                        ("smoke_kf_qie2511", "Qwen-Image-Edit-2511 keyframe")):
        p = C.RUNS / "train" / name / "smoke_result.json"
        if not p.exists():
            rows.append(f"| {label} | not run yet | | | | | | |")
            continue
        r = json.loads(p.read_text())
        size = r["resolution"] + (f" × {r['frames']} f" if "frames" in r else f", control {r.get('control_resolution')}")
        rows.append(f"| {label} | {'PASS' if r['ok'] else 'FAIL'} | {size} | fp8_scaled, blocks_to_swap {r['blocks_to_swap']} | "
                    f"{r['steps']} | {r['s_per_step']} | {r['vram_peak_gb']} | {r['first_avr_loss']} → {r['last_avr_loss']} |")
    table = "\n".join([
        f"_Updated {time.strftime('%Y-%m-%d %H:%M')} by `src/train/smoke_report.py` (RTX 4090 24 GB, real smoke-set caches)._",
        "",
        "| Run | Result | Size | 24 GB settings | Steps | s/step | Peak VRAM (GB) | avr_loss first → last |",
        "|---|---|---|---|---|---|---|---|",
        *rows,
    ])
    doc = C.REPO / "docs" / "training.md"
    text = doc.read_text(encoding="utf-8")
    if BEGIN not in text or END not in text:
        raise SystemExit(f"markers not found in {doc}")
    head, rest = text.split(BEGIN, 1)
    _, tail = rest.split(END, 1)
    C.write_text(doc, head + BEGIN + "\n" + table + "\n" + END + tail)
    print(table)


if __name__ == "__main__":
    main()
