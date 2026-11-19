"""Run a command and record the peak VRAM used by its process tree (the local GPU is shared with other jobs).

    python -m src.train.vram_watch <out.json> -- <cmd...>

Writes {"peak_gb": ..., "samples": N} to <out.json> when the command exits and returns its exit code.
"""
from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from pathlib import Path


def descendants(root: int) -> set[int]:
    """PIDs of `root` and all its descendants (via /proc, no psutil dependency)."""
    children: dict[int, list[int]] = {}
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            stat = (p / "stat").read_text()
            ppid = int(stat.rsplit(")", 1)[1].split()[1])
            children.setdefault(ppid, []).append(int(p.name))
        except (OSError, ValueError, IndexError):
            continue
    out, stack = {root}, [root]
    while stack:
        for c in children.get(stack.pop(), []):
            if c not in out:
                out.add(c)
                stack.append(c)
    return out


def main() -> int:
    if len(sys.argv) < 4 or sys.argv[2] != "--":
        print(__doc__)
        return 2
    out = Path(sys.argv[1])
    proc = subprocess.Popen(sys.argv[3:])
    state = {"peak_mib": 0, "samples": 0}

    def poll() -> None:
        while proc.poll() is None:
            try:
                txt = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=10).stdout
                pids = descendants(proc.pid)
                used = 0
                for line in txt.splitlines():
                    parts = [x.strip() for x in line.split(",")]
                    if len(parts) == 2 and parts[0].isdigit() and int(parts[0]) in pids and parts[1].isdigit():
                        used += int(parts[1])
                state["peak_mib"] = max(state["peak_mib"], used)
                state["samples"] += 1
            except Exception:  # noqa: BLE001
                pass
            time.sleep(2)

    t = threading.Thread(target=poll, daemon=True)
    t.start()
    rc = proc.wait()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"peak_gb": round(state["peak_mib"] / 1024, 2), "samples": state["samples"]}))
    return rc


if __name__ == "__main__":
    sys.exit(main())
