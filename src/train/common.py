"""Shared helpers for the LoRA training CLIs (`wan_lora.py`, `kf_lora.py`) and the HF Jobs tools.

Only stdlib + PyYAML at import time; `huggingface_hub` is imported lazily by the upload helpers, so the CLIs work
from either `venvs/main` or `venvs/musubi` (recommended: `venvs/musubi/bin/python -m src.train.wan_lora ...`).

Upload policy (hard rule of the project): the ONLY things that leave this machine are musubi-tuner cache files
(`*.safetensors` VAE latents / text-encoder outputs), job scripts/configs and LoRA checkpoints, and only to PRIVATE
repos. `upload_cache_shards` enforces the file-name allow-list and refuses anything else.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from typing import Iterable, Sequence

import yaml

REPO = Path(os.environ.get("KOMI_REPO", "/home/user/komi-poc"))
WORK = Path(os.environ.get("KOMI_WORK", "/home/user/komi-poc-work"))
TMP = Path(os.environ.get("TMPDIR", str(WORK / "tmp")))
RUNS = WORK / "runs"
GPU_COST_LOG = RUNS / "gpu_cost.jsonl"
GPU_RUN = REPO / "scripts" / "gpu_run.sh"
FFMPEG = WORK / "tools" / "ffenv" / "bin" / "ffmpeg"
FFPROBE = WORK / "tools" / "ffenv" / "bin" / "ffprobe"

# cache files written by musubi-tuner: <item>_<frames>_<WxH>_<arch>.safetensors and <item>_<arch>_te.safetensors
CACHE_FILE_RE = re.compile(r"^[^/\\]+_(wan|qie|qi)(_te)?\.safetensors$")
ARCH_SUFFIX = {"wan": "wan", "kf": "qie"}


def infra() -> dict:
    """configs/train_infra.yaml -> dict (the `train_infra` section)."""
    with open(REPO / "configs" / "train_infra.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["train_infra"]


def load_env() -> None:
    """Load `.env` into os.environ without printing anything (HF_TOKEN, HF_USER, ...)."""
    env = REPO / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def musubi_python() -> str:
    return str(Path(infra()["musubi"]["venv"]) / "bin" / "python")


def musubi_script(name: str) -> str:
    """Absolute path of a musubi-tuner entry script, e.g. `wan_train_network.py`."""
    p = Path(infra()["musubi"]["dir"]) / "src" / "musubi_tuner" / name
    if not p.exists():
        raise FileNotFoundError(p)
    return str(p)


def accelerate_launch(mixed_precision: str) -> list[str]:
    acc = str(Path(infra()["musubi"]["venv"]) / "bin" / "accelerate")
    return [acc, "launch", "--num_cpu_threads_per_process", "1", "--mixed_precision", mixed_precision]


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(cmd: Sequence[str], *, log_file: Path | None = None, env: dict | None = None, cwd: str | None = None,
        check: bool = True) -> int:
    """Run a command, tee its output to `log_file`. Never prints the environment."""
    cmd = [str(c) for c in cmd]
    log("$ " + " ".join(shlex.quote(c) for c in cmd))
    full_env = dict(os.environ)
    if env:
        full_env.update(env)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
    fh = open(log_file, "ab") if log_file is not None else None
    try:
        if fh is not None:
            fh.write(("\n==== " + now_iso() + " $ " + " ".join(shlex.quote(c) for c in cmd) + "\n").encode())
            fh.flush()
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=full_env, cwd=cwd)
        assert proc.stdout is not None
        for chunk in iter(lambda: proc.stdout.read1(8192), b""):
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
            if fh is not None:
                fh.write(chunk)
                fh.flush()
        rc = proc.wait()
    finally:
        if fh is not None:
            fh.close()
    if check and rc != 0:
        raise SystemExit(f"command failed with exit code {rc}: {cmd[0]} ... (log: {log_file})")
    return rc


def gpu_prefix(need_gb: int) -> list[str]:
    """Every local GPU job goes through scripts/gpu_run.sh (waits until `need_gb` of VRAM is free)."""
    return [str(GPU_RUN), str(int(need_gb))]


def cpu_env() -> dict:
    """Environment that makes the GPU invisible (used by the --device cpu caching path)."""
    return {"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": str(max(4, (os.cpu_count() or 8) // 2))}


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def toml_dump(d: dict) -> str:
    """Tiny TOML writer for the dataset config ([general] + [[datasets]])."""

    def val(v):
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, (int, float)):
            return repr(v)
        if isinstance(v, (list, tuple)):
            return "[" + ", ".join(val(x) for x in v) + "]"
        return json.dumps(str(v), ensure_ascii=False)

    out = []
    if d.get("_comment"):
        out += ["# " + line for line in d["_comment"].splitlines()] + [""]
    out.append("[general]")
    out += [f"{k} = {val(v)}" for k, v in d["general"].items()]
    for ds in d["datasets"]:
        out += ["", "[[datasets]]"] + [f"{k} = {val(v)}" for k, v in ds.items()]
    return "\n".join(out) + "\n"


def read_toml(path: Path) -> dict:
    try:
        import tomllib
    except ModuleNotFoundError:  # py<3.11
        import toml as tomllib  # type: ignore
        return tomllib.load(open(path, "r", encoding="utf-8"))
    with open(path, "rb") as f:
        return tomllib.load(f)


def assert_train_split(names: Iterable[str]) -> None:
    """Refuse anything that looks like it comes from the held-out episodes 22-24 (ids carry `epNN`)."""
    lo, hi = infra()["data"]["train_episodes"]
    bad = []
    for n in names:
        m = re.search(r"ep(\d{2})", n)
        if m and not (lo <= int(m.group(1)) <= hi):
            bad.append(n)
    if bad:
        raise SystemExit(f"held-out episodes must never be used for training: {bad[:5]} ({len(bad)} items)")


# ----------------------------------------------------------------------------- Hugging Face helpers

def hf_api():
    load_env()
    from huggingface_hub import HfApi

    if not os.environ.get("HF_TOKEN"):
        raise SystemExit("HF_TOKEN is not set (.env)")
    return HfApi(token=os.environ["HF_TOKEN"])


def ensure_private_repo(repo_id: str, repo_type: str) -> None:
    """Create the repo PRIVATE if missing; abort if it exists and is not private."""
    api = hf_api()
    api.create_repo(repo_id=repo_id, repo_type=repo_type, private=True, exist_ok=True)
    info = api.repo_info(repo_id, repo_type=repo_type)
    if not info.private:
        raise SystemExit(f"refusing to use {repo_type} repo {repo_id}: it is NOT private")


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def list_cache_files(cache_dir: Path, arch: str) -> list[Path]:
    """Cache files of one dataset. Raises if the directory holds anything that is not a musubi cache file."""
    files, alien = [], []
    for p in sorted(cache_dir.iterdir()):
        if p.is_dir():
            alien.append(p.name)
        elif CACHE_FILE_RE.match(p.name) and (p.name.endswith(f"_{arch}.safetensors") or p.name.endswith(f"_{arch}_te.safetensors")):
            files.append(p)
        else:
            alien.append(p.name)
    if alien:
        raise SystemExit(f"{cache_dir} contains non-cache entries (refusing to upload): {alien[:5]}")
    return files


def upload_cache_shards(cache_dir: Path, path_in_repo: str, arch: str, meta: dict, shard_gb: float = 1.0) -> dict:
    """Pack the cache files into uncompressed tar shards and upload them to the PRIVATE dataset repo.

    Layout in the repo: <path_in_repo>/shard-00000.tar ... + <path_in_repo>/manifest.json
    Shards are built one at a time in $TMPDIR and removed after upload; shards that already exist with the same size
    are skipped, so an interrupted upload can simply be re-run.
    """
    repo_id = infra()["hub"]["cache_repo"]
    ensure_private_repo(repo_id, "dataset")
    api = hf_api()
    files = list_cache_files(cache_dir, arch)
    latents = [p for p in files if not p.name.endswith("_te.safetensors")]
    te = [p for p in files if p.name.endswith("_te.safetensors")]
    if not latents or not te:
        raise SystemExit(f"{cache_dir}: found {len(latents)} latent and {len(te)} text-encoder cache files; run `cache` first")
    assert_train_split([p.name for p in files])

    # deterministic sharding by cumulative size
    limit = int(shard_gb * (1 << 30))
    shards: list[list[Path]] = [[]]
    size = 0
    for p in files:
        s = p.stat().st_size
        if shards[-1] and size + s > limit:
            shards.append([])
            size = 0
        shards[-1].append(p)
        size += s

    existing = {}
    try:
        for e in api.list_repo_tree(repo_id, path_in_repo=path_in_repo, repo_type="dataset"):
            existing[e.path] = getattr(e, "size", None)
    except Exception:
        pass  # path does not exist yet

    stage = TMP / "upload_cache" / path_in_repo.replace("/", "__")
    stage.mkdir(parents=True, exist_ok=True)
    manifest = {"created": now_iso(), "arch": arch, "path_in_repo": path_in_repo, "n_files": len(files),
                "n_items": len(latents), "total_bytes": sum(p.stat().st_size for p in files), "meta": meta, "shards": []}
    for i, members in enumerate(shards):
        name = f"shard-{i:05d}.tar"
        tar_path = stage / name
        with tarfile.open(tar_path, "w") as tf:  # uncompressed: safetensors do not compress
            for p in members:
                ti = tarfile.TarInfo(name=p.name)
                ti.size = p.stat().st_size
                ti.mtime = 0
                ti.mode = 0o644
                with open(p, "rb") as fh:
                    tf.addfile(ti, fh)
        tsize = tar_path.stat().st_size
        remote = f"{path_in_repo}/{name}"
        manifest["shards"].append({"name": name, "bytes": tsize, "n_files": len(members),
                                   "files": [{"name": p.name, "bytes": p.stat().st_size} for p in members]})
        if existing.get(remote) == tsize:
            log(f"skip {remote} (already uploaded, {tsize / 1e6:.1f} MB)")
        else:
            log(f"upload {remote} ({tsize / 1e6:.1f} MB, {len(members)} files)")
            api.upload_file(path_or_fileobj=str(tar_path), path_in_repo=remote, repo_id=repo_id, repo_type="dataset",
                            commit_message=f"cache shard {remote}")
        tar_path.unlink()
    # drop stale shards from a previous, larger upload
    stale = [p for p in existing if re.search(r"/shard-\d{5}\.tar$", p) and p.split("/")[-1] not in {s["name"] for s in manifest["shards"]}]
    for p in stale:
        log(f"delete stale {p}")
        api.delete_file(p, repo_id=repo_id, repo_type="dataset")
    api.upload_file(path_or_fileobj=json.dumps(manifest, indent=1).encode(), path_in_repo=f"{path_in_repo}/manifest.json",
                    repo_id=repo_id, repo_type="dataset", commit_message=f"cache manifest {path_in_repo}")
    log(f"uploaded {manifest['n_items']} items / {manifest['n_files']} files / {manifest['total_bytes'] / 1e9:.2f} GB "
        f"in {len(shards)} shard(s) to https://huggingface.co/datasets/{repo_id}/tree/main/{path_in_repo} (private)")
    return manifest


def append_cost(job_id: str, flavor: str, minutes: float, usd: float, **extra) -> None:
    """runs/gpu_cost.jsonl: one line per finished job {ts, job_id, flavor, minutes, usd, ...}.

    Append-only (the file is shared with other stages) and idempotent per job_id.
    """
    GPU_COST_LOG.parent.mkdir(parents=True, exist_ok=True)
    if GPU_COST_LOG.exists():
        for l in GPU_COST_LOG.read_text().splitlines():
            try:
                if l.strip() and json.loads(l).get("job_id") == job_id:
                    return
            except json.JSONDecodeError:
                continue
    row = {"ts": now_iso(), "job_id": job_id, "flavor": flavor, "minutes": round(minutes, 2), "usd": round(usd, 4), **extra}
    with open(GPU_COST_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def total_cost() -> float:
    if not GPU_COST_LOG.exists():
        return 0.0
    total = 0.0
    for l in GPU_COST_LOG.read_text().splitlines():
        try:
            total += float(json.loads(l).get("usd", 0.0)) if l.strip() else 0.0
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
    return total
