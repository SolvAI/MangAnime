#!/usr/bin/env python3
"""Stand-in for musubi's *_train_network.py, used to test the job supervisor on CPU (FAKE_TRAINER=1).

Mimics exactly what train_job.py relies on: the `steps:` tqdm line with an `avr_loss` postfix, step checkpoints
`<name>-step%08d.safetensors` + `<name>-step%08d-state/`, the final `<name>.safetensors` + `<name>-state/`, and
`--resume <state_dir>` (the fake state carries a counter so the round trip through the Hub can be verified).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", required=True)
    p.add_argument("--output_name", required=True)
    p.add_argument("--max_train_steps", type=int, required=True)
    p.add_argument("--save_every_n_steps", type=int, default=None)
    p.add_argument("--resume", default=None)
    p.add_argument("--save_state", action="store_true")
    p.add_argument("--dataset_config", default=None)
    p.add_argument("--fake_step_seconds", type=float, default=float(os.environ.get("FAKE_STEP_SECONDS", "0.4")))
    p.add_argument("--fake_nan_at", type=int, default=int(os.environ.get("FAKE_NAN_AT", "0")))
    a, _unknown = p.parse_known_args()

    carried = 0
    if a.resume:
        with open(os.path.join(a.resume, "fake_state.json")) as f:
            carried = json.load(f)["total_steps"]
        print(f"INFO:fake_trainer:resume training from local state: {a.resume} (carried total_steps={carried})", flush=True)
    if a.dataset_config:
        import glob
        import re

        text = open(a.dataset_config).read()
        m = re.search(r'cache_directory = "([^"]+)"', text)
        n = len(glob.glob(os.path.join(m.group(1), "*.safetensors"))) if m else -1
        print(f"INFO:fake_trainer:cache files visible to the trainer: {n}", flush=True)
    os.makedirs(a.output_dir, exist_ok=True)

    def save(name: str, state_name: str, step: int) -> None:
        with open(os.path.join(a.output_dir, name), "wb") as f:
            f.write(b"FAKE-LORA" + os.urandom(4096))
        print(f"\nsaving checkpoint: {os.path.join(a.output_dir, name)}", flush=True)
        if a.save_state:
            d = os.path.join(a.output_dir, state_name)
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, "fake_state.json"), "w") as f:
                json.dump({"total_steps": carried + step}, f)
            with open(os.path.join(d, "optimizer.bin"), "wb") as f:
                f.write(os.urandom(8192))

    t0 = time.time()
    n = a.max_train_steps
    for step in range(1, n + 1):
        time.sleep(a.fake_step_seconds)
        loss = 0.12 * (0.995 ** (carried + step)) + 0.02
        if a.fake_nan_at and carried + step >= a.fake_nan_at:
            loss = float("nan")
        el = time.time() - t0
        rate = el / step
        bar = f"steps: {100 * step // n:3d}%|{'#' * (10 * step // n):<10}| {step}/{n} [{int(el) // 60:02d}:{int(el) % 60:02d}<{int(rate * (n - step)) // 60:02d}:{int(rate * (n - step)) % 60:02d}, {rate:5.2f}s/it, avr_loss={loss:.4f}]"
        sys.stderr.write("\r" + bar)
        sys.stderr.flush()
        if a.save_every_n_steps and step % a.save_every_n_steps == 0:
            save(f"{a.output_name}-step{step:08d}.safetensors", f"{a.output_name}-step{step:08d}-state", step)
    save(f"{a.output_name}.safetensors", f"{a.output_name}-state", n)
    print("\nINFO:fake_trainer:model saved.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
