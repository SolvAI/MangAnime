#!/usr/bin/env python3
"""Launch a Komi POC training job on Hugging Face Jobs (wraps `hf jobs run`, hf CLI 2.2.0).

    . scripts/env.sh;  MP=$KOMI_WORK/venvs/musubi/bin/python
    $MP infra/hf_jobs/launch.py check                                   # cpu-basic: secrets + private repos (fake trainer)
    $MP infra/hf_jobs/launch.py probe --cache-path wan/wan_smoke/832x480_f81            # <=20 min full-spec probe
    $MP infra/hf_jobs/launch.py wan --expert high --run wan22-i2v-high-v1 --cache-path wan/<set>/832x480_f81 \
        --max-steps 4000 --timeout 24h
    $MP infra/hf_jobs/launch.py wan --expert high --run wan22-i2v-high-v1 --cache-path ... --max-steps 4000 --timeout 24h --resume
    $MP infra/hf_jobs/launch.py kf --run kf-qie2511-v1 --cache-path kf/<set>/<variant> --max-steps 3000 --timeout 8h

What happens: the job scripts (this folder) + configs/train_*.toml are uploaded to `jobs/` of the PRIVATE dataset repo,
the job is started in a stock public PyTorch image with `--secrets HF_TOKEN` and an explicit `--timeout`, and the job id
is written to runs/train/<run>/job.json. Follow it with monitor.py (which also logs the cost when the job ends).
Raw media never leaves the machine: only cache shards, scripts/configs and LoRA checkpoints are exchanged.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
from src.train import common as C  # noqa: E402

JOB_FILES = ["bootstrap.sh", "train_job.py", "logparse.py", "fake_trainer.py"]
ENTRY = (
    "set -e; V=${KOMI_VENV:-/tmp/komi_venv}; python -m venv --system-site-packages $V; "
    "$V/bin/pip install -q --disable-pip-version-check huggingface_hub==1.32.0; "
    "$V/bin/python -c \"import os; from huggingface_hub import snapshot_download as d; "
    "d(os.environ['CACHE_REPO'], repo_type='dataset', revision=os.environ['JOBS_REV'], allow_patterns=['jobs/*'], "
    "local_dir='/tmp/komi_repo', token=os.environ['HF_TOKEN'])\"; "
    "exec bash /tmp/komi_repo/jobs/bootstrap.sh"
)


def hf_bin() -> str:
    b = os.environ.get("HF_BIN") or C.infra()["jobs"].get("hf_bin") or shutil.which("hf")
    if not b or not Path(b).exists():
        raise SystemExit("hf CLI not found (set HF_BIN)")
    return b


def timeout_minutes(s: str) -> float:
    m = re.fullmatch(r"(\d+(?:\.\d+)?)([smhd]?)", s.strip())
    if not m:
        raise SystemExit(f"bad --timeout {s!r} (use e.g. 20m, 24h)")
    return float(m.group(1)) * {"s": 1 / 60, "": 1 / 60, "m": 1, "h": 60, "d": 1440}[m.group(2)]


def price(flavor: str) -> float:
    p = C.infra()["jobs"]["usd_per_hour"].get(flavor)
    if p is None:
        raise SystemExit(f"no price for flavor {flavor} in configs/train_infra.yaml (run `hf jobs hardware`)")
    return float(p)


def gpu_budget() -> float:
    import yaml

    base = yaml.safe_load((REPO / "configs" / "base.yaml").read_text())
    return float(os.environ.get("GPU_BUDGET_USD") or base.get("budget", {}).get("gpu_usd", 200.0))


def unlogged_commitments() -> tuple[float, list[str]]:
    """Worst-case cost (timeout x price) of jobs launched from here whose real cost is not in runs/gpu_cost.jsonl yet:
    still running, or finished without monitor.py having logged them. Counted against the budget until logged."""
    logged = set()
    if C.GPU_COST_LOG.exists():
        for line in C.GPU_COST_LOG.read_text().splitlines():
            try:
                logged.add(json.loads(line).get("job_id"))
            except json.JSONDecodeError:
                continue
    total, runs = 0.0, []
    for f in sorted((C.RUNS / "train").glob("*/jobs.jsonl")):
        for line in f.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("job_id") and r["job_id"] not in logged:
                total += float(r.get("timeout_min", 0)) / 60 * float(r.get("usd_per_hour", 0))
                runs.append(r["run"])
    return total, runs


def sync_job_files() -> str:
    """Upload jobs/ (scripts + hyper-parameter TOMLs) to the PRIVATE dataset repo; return the head revision."""
    inf = C.infra()
    repo = inf["hub"]["cache_repo"]
    C.ensure_private_repo(repo, "dataset")
    C.ensure_private_repo(inf["hub"]["lora_repo"], "model")
    stage = C.TMP / "jobs_stage"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    for f in JOB_FILES:
        shutil.copy2(HERE / f, stage / f)
    for f in sorted((REPO / "configs").glob("train_*.toml")):
        shutil.copy2(f, stage / f.name)
    (stage / "VERSION.json").write_text(json.dumps({"musubi_commit": inf["musubi"]["commit"], "uploaded": C.now_iso()}))
    api = C.hf_api()
    api.upload_folder(folder_path=str(stage), path_in_repo="jobs", repo_id=repo, repo_type="dataset", delete_patterns="*",
                      commit_message="job scripts + train configs")
    return api.dataset_info(repo).sha


SELFTEST_PATH = "wan/_selftest/832x480_f81"


def ensure_selftest_fixture() -> None:
    """`check` must not depend on real data: a 2-file synthetic "cache" (random bytes, not media) exercises the same
    download/unpack code path. Created once under wan/_selftest/ in the PRIVATE dataset repo."""
    api = C.hf_api()
    repo = C.infra()["hub"]["cache_repo"]
    C.ensure_private_repo(repo, "dataset")
    if api.file_exists(repo, f"{SELFTEST_PATH}/manifest.json", repo_type="dataset"):
        return
    d = C.TMP / "selftest_cache"
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    (d / "selftest00_00000-081_0832x0480_wan.safetensors").write_bytes(os.urandom(2048))
    (d / "selftest00_wan_te.safetensors").write_bytes(os.urandom(1024))
    meta = {"kind": "wan", "resolution": [832, 480], "frames": 81, "i2v": True, "dataset": "_selftest", "synthetic": True,
            "dataset_config": {"general": {"resolution": [832, 480], "caption_extension": ".txt", "batch_size": 1,
                                           "enable_bucket": False, "bucket_no_upscale": False},
                               "dataset": {"target_frames": [81], "frame_extraction": "head", "num_repeats": 1}}}
    C.upload_cache_shards(d, SELFTEST_PATH, "wan", meta)
    shutil.rmtree(d)


def check_cache_path(cache_path: str, rev: str) -> dict:
    from huggingface_hub import hf_hub_download

    try:
        p = hf_hub_download(C.infra()["hub"]["cache_repo"], f"{cache_path}/manifest.json", repo_type="dataset", revision=rev,
                            local_dir=str(C.TMP / "launch_meta"), force_download=True, token=os.environ["HF_TOKEN"])
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"cache path {cache_path!r} not found in the dataset repo ({type(e).__name__}); run upload-cache first")
    return json.loads(Path(p).read_text())


def launch(a) -> dict:
    C.load_env()
    inf = C.infra()
    if not os.environ.get("HF_TOKEN"):
        raise SystemExit("HF_TOKEN missing (.env)")
    flavor = a.flavor or inf["jobs"]["flavor"]
    tmin = timeout_minutes(a.timeout)
    worst = tmin / 60 * price(flavor)
    spent, budget = C.total_cost(), gpu_budget()
    committed, open_runs = unlogged_commitments()
    C.log(f"flavor={flavor} (${price(flavor):.2f}/h) timeout={a.timeout} -> worst case ${worst:.2f}; logged spend ${spent:.2f}, "
          f"unlogged/running jobs ${committed:.2f} worst case {sorted(set(open_runs))}; budget ${budget:.0f}")
    if a.max_usd is not None and worst > a.max_usd:
        raise SystemExit(f"worst-case cost ${worst:.2f} exceeds --max-usd {a.max_usd}")
    if spent + committed + worst > 0.8 * budget and not a.force:
        raise SystemExit(f"logged spend + running/unlogged jobs + this job's worst case = ${spent + committed + worst:.2f} would pass 80% "
                         f"of the GPU budget (${budget:.0f}). Log finished jobs with `monitor.py --run <run> --once`, shorten "
                         f"--timeout, or override with --force.")

    if a.cache_path == SELFTEST_PATH:
        ensure_selftest_fixture()
    rev = sync_job_files()
    manifest = check_cache_path(a.cache_path, rev)
    kind = a.kind
    if manifest["meta"]["kind"] != kind:
        raise SystemExit(f"cache {a.cache_path} is for kind={manifest['meta']['kind']}, not {kind}")
    env = {
        "KIND": kind, "RUN_NAME": a.run, "CACHE_REPO": inf["hub"]["cache_repo"], "CACHE_PATH": a.cache_path, "CACHE_REV": rev,
        "JOBS_REV": rev, "LORA_REPO": inf["hub"]["lora_repo"], "MUSUBI_COMMIT": inf["musubi"]["commit"],
        "MAX_STEPS": str(a.max_steps), "SAVE_EVERY": str(a.save_every), "PUSH_EVERY_MIN": str(a.push_every_min),
        "RESUME": "1" if a.resume else "0", "DEADLINE_MIN": f"{max(1.0, tmin - (2 if tmin <= 30 else 5)):.1f}",
        "LOG_EVERY_STEPS": str(a.log_every),
    }
    if kind == "wan":
        hw = inf["hub_weights"][f"wan_dit_{a.expert}"]
        env.update({"EXPERT": a.expert, "TRAIN_CONFIG": f"train_wan22_i2v_{a.expert}.toml"})
    else:
        hw = inf["hub_weights"]["qwen_dit"]
        env.update({"TRAIN_CONFIG": "train_kf_qie2511.toml"})
    env.update({"DIT_REPO": hw["repo"], "DIT_FILE": hw["file"], "DIT_REV": hw["revision"]})
    if a.batch_size:
        env["BATCH_SIZE"] = str(a.batch_size)
    if a.extra_args:
        env["EXTRA_ARGS"] = a.extra_args
    if a.self_resume_at:
        env["SELF_RESUME_AT"] = str(a.self_resume_at)
    if a.fake:
        env["FAKE_TRAINER"] = "1"
    if a.allow_overwrite:
        env["ALLOW_OVERWRITE"] = "1"
    for kv in a.env or []:
        k, v = kv.split("=", 1)
        env[k] = v

    image = a.image or inf["jobs"]["image"]
    cmd = [hf_bin(), "jobs", "run", "--detach", "--flavor", flavor, "--timeout", a.timeout, "--secrets", "HF_TOKEN",
           "--name", a.run, "--label", "project=komi-poc", "--label", f"kind={kind}"]
    for k, v in env.items():
        cmd += ["--env", f"{k}={v}"]
    cmd += [image, "bash", "-c", ENTRY]
    C.log(f"image={image} items={manifest['n_items']} cache={manifest['total_bytes'] / 1e9:.2f} GB "
          f"max_steps={a.max_steps} resume={a.resume}")
    if a.dry_run:
        out = subprocess.run(cmd[:3] + ["--dry-run"] + cmd[3:], capture_output=True, text=True)
        print(out.stdout + out.stderr)
        return {}
    out = subprocess.run(cmd, capture_output=True, text=True)
    text = (out.stdout + "\n" + out.stderr).strip()
    if out.returncode != 0:
        raise SystemExit(f"hf jobs run failed ({out.returncode}):\n{text[-2000:]}")
    m = re.search(r"\b([0-9a-f]{24})\b", text)
    if not m:
        raise SystemExit(f"could not find the job id in:\n{text[-2000:]}")
    job_id = m.group(1)
    run_dir = C.RUNS / "train" / a.run
    run_dir.mkdir(parents=True, exist_ok=True)
    rec = {"job_id": job_id, "run": a.run, "kind": kind, "expert": getattr(a, "expert", None), "flavor": flavor,
           "usd_per_hour": price(flavor), "timeout": a.timeout, "timeout_min": tmin, "image": image, "launched_at": C.now_iso(),
           "launched_ts": time.time(), "resume": bool(a.resume), "max_steps": a.max_steps, "cache_path": a.cache_path,
           "jobs_rev": rev, "env": env}
    (run_dir / "job.json").write_text(json.dumps(rec, indent=1))
    with open(run_dir / "jobs.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    C.log(f"job {job_id} submitted -> {run_dir / 'job.json'}")
    print(f"JOB_ID={job_id}")
    print(f"monitor:  {sys.executable} {C.REPO / 'infra' / 'hf_jobs' / 'monitor.py'} --run {a.run}")
    print(f"cancel:   {hf_bin()} jobs cancel {job_id}")
    return rec


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="kind", required=True)

    def common(s, timeout, steps, save_every, push_every, cache_required=True):
        s.add_argument("--run", required=False, default=None, help="run name = folder runs/<run>/ in the model repo")
        s.add_argument("--cache-path", required=cache_required, default=None if cache_required else SELFTEST_PATH,
                       help="path of the cache in the dataset repo, e.g. wan/full/832x480_f81")
        s.add_argument("--max-steps", type=int, default=steps, help="TOTAL optimisation steps of the run (across resumes)")
        s.add_argument("--timeout", default=timeout, help="explicit job timeout, e.g. 20m, 24h (HF default is only 30 min)")
        s.add_argument("--save-every", type=int, default=save_every, help="steps between local checkpoints (+state)")
        s.add_argument("--push-every-min", type=float, default=push_every, help="minimum minutes between pushes to the Hub")
        s.add_argument("--log-every", type=int, default=5)
        s.add_argument("--resume", action="store_true", help="RESUME=1: continue from the last pushed state of --run")
        s.add_argument("--flavor", default=None)
        s.add_argument("--image", default=None)
        s.add_argument("--batch-size", type=int, default=0)
        s.add_argument("--extra-args", default="",
                       help='extra musubi-tuner arguments; always write it with "=": --extra-args="--learning_rate 5e-5"')
        s.add_argument("--self-resume-at", type=int, default=0, help="probe only: restart from the Hub state at this step")
        s.add_argument("--env", action="append", help="extra KEY=VALUE for the job")
        s.add_argument("--fake", action="store_true", help="FAKE_TRAINER=1 (supervisor test without musubi)")
        s.add_argument("--allow-overwrite", action="store_true")
        s.add_argument("--max-usd", type=float, default=None, help="refuse if timeout x price exceeds this")
        s.add_argument("--force", action="store_true", help="ignore the 80%% budget guard")
        s.add_argument("--dry-run", action="store_true")

    s = sub.add_parser("wan", help="Wan 2.2 A14B I2V LoRA (one expert per job)")
    s.add_argument("--expert", choices=["high", "low"], required=True)
    common(s, "24h", 4000, 100, 30)

    s = sub.add_parser("kf", help="Qwen-Image-Edit-2511 keyframe LoRA")
    common(s, "8h", 3000, 250, 30)

    s = sub.add_parser("probe", help="<=20 min paid probe at full spec (Wan high-noise expert) with push + resume check")
    s.add_argument("--expert", choices=["high", "low"], default="high")
    common(s, "20m", 16, 4, 0)

    s = sub.add_parser("check", help="cpu-basic job: secrets + private repo read/write + resume with the fake trainer")
    s.add_argument("--expert", choices=["high", "low"], default="high")
    common(s, "10m", 60, 10, 0, cache_required=False)

    a = p.parse_args()
    stamp = time.strftime("%m%d-%H%M")
    if a.kind == "probe":
        a.kind = "wan"
        a.run = a.run or f"probe-wan22-i2v-{a.expert}-{stamp}"
        a.self_resume_at = a.self_resume_at or (a.max_steps // 2)
        a.max_usd = a.max_usd if a.max_usd is not None else 1.0
    elif a.kind == "check":
        a.kind = "wan"
        a.run = a.run or f"check-cpu-{stamp}"
        a.fake = True
        a.flavor = a.flavor or "cpu-basic"
        a.image = a.image or "python:3.12"
        a.self_resume_at = a.self_resume_at or (a.max_steps // 2)
        a.max_usd = a.max_usd if a.max_usd is not None else 0.05
    if not a.run:
        raise SystemExit("--run is required")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{2,60}", a.run):
        raise SystemExit("--run: use letters, digits, '.', '_' or '-'")
    launch(a)


if __name__ == "__main__":
    main()
