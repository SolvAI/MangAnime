"""Small IO + caching helpers shared by all stages (idempotent, content-hash keyed)."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable

from .config import CFG


def sha(obj: Any, n: int = 16) -> str:
    """Stable content hash of a JSON-able object / bytes / str."""
    if isinstance(obj, bytes):
        b = obj
    elif isinstance(obj, str):
        b = obj.encode()
    else:
        b = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str).encode()
    return hashlib.sha256(b).hexdigest()[:n]


def file_sig(path: str | Path) -> str:
    """Cheap file signature (size + mtime + name). Good enough for derived-file caching."""
    st = os.stat(path)
    return sha(f"{Path(path).name}:{st.st_size}:{int(st.st_mtime)}")


def read_json(path: str | Path, default: Any = None) -> Any:
    p = Path(path)
    if not p.exists():
        return default
    return json.loads(p.read_text())


def write_json(path: str | Path, obj: Any, indent: int | None = 1) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=indent, default=str))
    os.replace(tmp, p)


def read_jsonl(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    os.replace(tmp, p)


def append_jsonl(path: str | Path, row: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def run(cmd: list[str], check: bool = True, quiet: bool = True, **kw) -> subprocess.CompletedProcess:
    """Run a subprocess; raise with stderr tail on failure."""
    kw.setdefault("stdin", subprocess.DEVNULL)      # never let a child (ffmpeg) grab the terminal
    r = subprocess.run([str(c) for c in cmd], capture_output=quiet, text=True, **kw)
    if check and r.returncode != 0:
        tail = (r.stderr or "")[-1500:] if quiet else ""
        raise RuntimeError(f"command failed ({r.returncode}): {' '.join(map(str, cmd))[:400]}\n{tail}")
    return r


def ffmpeg(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return run([CFG.paths.ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "error", "-y", *args], check=check)


def ffprobe_json(path: str | Path, entries: str = "format=duration:stream=index,codec_type,codec_name,width,height,avg_frame_rate,nb_frames") -> dict:
    r = run([CFG.paths.ffprobe, "-v", "error", "-show_entries", entries, "-of", "json", str(path)])
    return json.loads(r.stdout)


def media_duration(path: str | Path) -> float:
    try:
        return float(ffprobe_json(path, "format=duration")["format"]["duration"])
    except Exception:
        return 0.0


class Timer:
    def __init__(self, label: str = ""):
        self.label = label

    def __enter__(self):
        self.t = time.time()
        return self

    def __exit__(self, *a):
        self.dt = time.time() - self.t


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
