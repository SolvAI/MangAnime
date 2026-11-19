"""Model memory manager (brief §3b): start/stop the local model servers so only one large model is resident.

Servers run in tmux sessions (survive the calling process):
  qwen  : llama.cpp server, Qwen 3.8 27B Q4 + mmproj, port 8188   (~20 GB VRAM)
  ocr   : llama.cpp server, PaddleOCR-VL 1.6 GGUF, port 8189      (~2.5 GB VRAM)
  comfy : ComfyUI headless (existing install), port 8199          (Wan / Qwen-Image-Edit, up to 24 GB)

CLI: python -m src.common.servers start qwen|ocr|comfy ; stop <name>|all ; status
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

import httpx

from src.common.config import CFG, P
from src.common.io import log

LLAMA = CFG.llm.local.server_bin
WORK = P("work")

SPECS = {
    "qwen": {
        "port": int(CFG.llm.local.port), "health": "/health", "vram_gb": 20,
        "cmd": (
            f"{LLAMA} -m {CFG.llm.local.gguf} --mmproj {CFG.llm.local.mmproj} --alias {CFG.llm.local.model} "
            f"--host 127.0.0.1 --port {CFG.llm.local.port} -ngl 99 -c {int(CFG.llm.local.ctx) * int(CFG.llm.local.parallel)} "
            f"-np {CFG.llm.local.parallel} -fa on --jinja --reasoning-budget 0 --image-max-tokens 1024"
        ),
    },
    "ocr": {
        "port": 8189, "health": "/health", "vram_gb": 3,
        "cmd": (
            f"{LLAMA} -m {P('models')}/ocr/PaddleOCR-VL-1.6-GGUF.gguf --mmproj {P('models')}/ocr/PaddleOCR-VL-1.6-GGUF-mmproj.gguf "
            f"--alias paddleocr-vl --host 127.0.0.1 --port 8189 -ngl 99 -c 16384 -np 8 -fa on --jinja"
        ),
    },
    "comfy": {
        # 21 GB for Wan; KOMI_COMFY_VRAM_GB=12 is enough when only the SDXL keyframe model runs (lets a small job share the card)
        "port": int(CFG.comfy.port), "health": "/system_stats", "vram_gb": int(os.environ.get("KOMI_COMFY_VRAM_GB", 21)),
        "cmd": (
            f"cd '{CFG.paths.comfy_root}' && ./venv/bin/python main.py --listen 127.0.0.1 --port {CFG.comfy.port} "
            f"--output-directory {WORK}/comfy_io/output --input-directory {WORK}/comfy_io/input "
            f"--temp-directory {WORK}/comfy_io/temp --user-directory {WORK}/comfy_io/user "
            f"--extra-model-paths-config {CFG.paths.repo}/configs/comfy_extra_model_paths.yaml --disable-auto-launch"
        ),
    },
}


def _tmux_has(name: str) -> bool:
    return subprocess.run(["tmux", "has-session", "-t", f"srv-{name}"], capture_output=True).returncode == 0


def is_up(name: str) -> bool:
    s = SPECS[name]
    try:
        r = httpx.get(f"http://127.0.0.1:{s['port']}{s['health']}", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def free_vram_gb() -> float:
    r = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], capture_output=True, text=True)
    return float(r.stdout.strip().splitlines()[0]) / 1024


def start(name: str, wait: float = 900.0, gpu_wait_min: int = 240) -> None:
    if is_up(name):
        return
    s = SPECS[name]
    if _tmux_has(name):
        subprocess.run(["tmux", "kill-session", "-t", f"srv-{name}"])
    for sub in ("output", "input", "temp", "user"):
        (WORK / "comfy_io" / sub).mkdir(parents=True, exist_ok=True)
    logf = WORK / "logs" / f"srv_{name}.log"
    import os

    prio = os.environ.get("GPU_PRIO", "1")       # model servers are critical-path jobs by default
    wrapped = f"GPU_PRIO={prio} GPU_WAIT_MIN={gpu_wait_min} GPU_SETTLE_SEC=40 {CFG.paths.repo}/scripts/gpu_run.sh {s['vram_gb']} bash -c \"{s['cmd']}\" > {logf} 2>&1"
    subprocess.run(["tmux", "new-session", "-d", "-s", f"srv-{name}", wrapped], check=True)
    t0 = time.time()
    while time.time() - t0 < wait + gpu_wait_min * 60:
        if is_up(name):
            log(f"server {name} up after {time.time() - t0:.0f}s")
            return
        if not _tmux_has(name):
            raise RuntimeError(f"server {name} exited during startup; see {logf}")
        time.sleep(3)
    raise TimeoutError(f"server {name} not healthy; see {logf}")


def stop(name: str) -> None:
    if _tmux_has(name):
        subprocess.run(["tmux", "send-keys", "-t", f"srv-{name}", "C-c"])
        time.sleep(2)
        subprocess.run(["tmux", "kill-session", "-t", f"srv-{name}"], capture_output=True)
    for _ in range(20):
        if not is_up(name):
            break
        time.sleep(1)
    time.sleep(2)


def only(name: str) -> None:
    """Make `name` the only LARGE server (qwen/comfy are mutually exclusive on 24 GB)."""
    for other in ("qwen", "comfy"):
        if other != name and (is_up(other) or _tmux_has(other)):
            log(f"stopping {other} to free VRAM for {name}")
            stop(other)
    start(name)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "start":
        start(sys.argv[2])
    elif cmd == "only":
        only(sys.argv[2])
    elif cmd == "stop":
        for n in (SPECS if sys.argv[2] == "all" else [sys.argv[2]]):
            stop(n)
    else:
        for n in SPECS:
            print(f"{n:6s} up={is_up(n)} tmux={_tmux_has(n)}")
        print(f"free VRAM: {free_vram_gb():.1f} GB")
