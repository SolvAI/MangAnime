#!/usr/bin/env python3
"""Job-side supervisor for the Komi POC LoRA trainings (runs inside the HF Job, started by bootstrap.sh).

What it does
  1. downloads the cache shards (+ manifest) from the PRIVATE dataset repo and unpacks them,
  2. downloads the base DiT from the Hub (pinned revision),
  3. writes the musubi dataset TOML (cache-only: no raw media exists in the job),
  4. runs musubi-tuner at full spec (no fp8, no block swap unless EXTRA_ARGS says so), saving LoRA + state every
     SAVE_EVERY steps,
  5. pushes the newest checkpoint + training state to the PRIVATE model repo at most every PUSH_EVERY_MIN minutes
     (and always at the end / before the deadline),
  6. with RESUME=1 pulls the last pushed state and trains the remaining steps,
  7. prints machine-readable lines for the local monitor:
        KOMI_METRIC {"step":..,"total":..,"avr_loss":..,"s_per_step":..,"vram_gb":..,"vram_peak_gb":..}
        KOMI_EVENT  {"event":"pushed","step":..}

Hub layout (model repo):  runs/<RUN_NAME>/<RUN_NAME>-step00000100.safetensors   step checkpoints (cumulative step)
                          runs/<RUN_NAME>/<RUN_NAME>.safetensors                final LoRA (when MAX_STEPS is reached)
                          runs/<RUN_NAME>/state/                                latest accelerate state (+ komi_state.json)
                          runs/<RUN_NAME>/progress.json                         {steps_done, max_steps, done, ...}
                          runs/<RUN_NAME>/logs/                                 trainer logs per segment

musubi restarts its step counter at 0 after `--resume` (trainer_base.py "TODO skip until initial step"), so the
supervisor keeps the cumulative step itself: a resumed segment trains MAX_STEPS - steps_done steps and its checkpoints
are renamed to the cumulative step on upload. Use a constant LR schedule (the configs do) so this is exact.

Environment (set by launch.py): KIND, EXPERT, RUN_NAME, CACHE_REPO, CACHE_PATH, CACHE_REV, LORA_REPO, DIT_REPO,
DIT_FILE, DIT_REV, TRAIN_CONFIG, MAX_STEPS, SAVE_EVERY, PUSH_EVERY_MIN, RESUME, BATCH_SIZE, EXTRA_ARGS, DEADLINE_MIN,
SELF_RESUME_AT, FAKE_TRAINER, LOCAL_DIT, WORK_DIR, MUSUBI_DIR, JOB_ID (set by HF). Secrets: HF_TOKEN (never printed).
"""
from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import logparse  # noqa: E402  (sibling module)

T0 = time.time()
STOPPED_RC = -999  # internal marker: the supervisor stopped the trainer on purpose (deadline / SIGTERM)
CACHE_NAME_RE = re.compile(r"^[^/\\]+_(wan|qie|qi)(_te)?\.safetensors$")
TRAINERS = {
    "wan": {"script": "wan_train_network.py", "mixed_precision": "fp16", "arch": "wan"},
    "kf": {"script": "qwen_image_train_network.py", "mixed_precision": "bf16", "arch": "qie"},
}


def env(name: str, default: str | None = None, required: bool = False) -> str:
    v = os.environ.get(name, default)
    if required and not v:
        raise SystemExit(f"missing environment variable {name}")
    return v or ""


def say(msg: str) -> None:
    print(f"[job +{time.time() - T0:7.0f}s] {msg}", flush=True)


def event(name: str, **kw) -> None:
    print(logparse.EVENT_PREFIX + json.dumps({"event": name, "t": round(time.time() - T0, 1), **kw}), flush=True)


def metric(**kw) -> None:
    print(logparse.METRIC_PREFIX + json.dumps(kw), flush=True)


class Cfg:
    def __init__(self) -> None:
        self.kind = env("KIND", "wan")
        if self.kind not in TRAINERS:
            raise SystemExit(f"KIND must be one of {list(TRAINERS)}")
        self.expert = env("EXPERT", "high")
        self.run = env("RUN_NAME", required=True)
        self.cache_repo = env("CACHE_REPO", required=True)
        self.cache_path = env("CACHE_PATH", required=True).strip("/")
        self.cache_rev = env("CACHE_REV", "main")
        self.lora_repo = env("LORA_REPO", required=True)
        self.max_steps = int(env("MAX_STEPS", required=True))
        self.save_every = int(env("SAVE_EVERY", "100"))
        self.push_every_s = float(env("PUSH_EVERY_MIN", "30")) * 60
        self.resume = env("RESUME", "0") == "1"
        self.batch_size = int(env("BATCH_SIZE", "0"))  # 0 = keep the value stored with the cache
        self.extra_args = shlex.split(env("EXTRA_ARGS", ""))
        self.deadline_s = float(env("DEADLINE_MIN", "0")) * 60  # 0 = none
        self.self_resume_at = int(env("SELF_RESUME_AT", "0"))
        self.fake = env("FAKE_TRAINER", "0") == "1"
        self.work = Path(env("WORK_DIR", "/tmp/komi"))
        self.musubi = Path(env("MUSUBI_DIR", "/opt/musubi-tuner"))
        self.train_config = HERE / env("TRAIN_CONFIG", "")
        self.log_every = int(env("LOG_EVERY_STEPS", "5"))
        self.job_id = env("JOB_ID", "local")
        self.prefix = f"runs/{self.run}"
        self.out = self.work / "out"
        self.cache_dir = self.work / "cache"


# ----------------------------------------------------------------------------- Hub I/O

def api():
    from huggingface_hub import HfApi

    return HfApi(token=os.environ["HF_TOKEN"])


def retry(fn, what: str, tries: int = 4):
    for i in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            if i == tries - 1:
                raise
            say(f"{what} failed ({type(e).__name__}: {str(e)[:200]}); retry {i + 1}/{tries - 1}")
            time.sleep(10 * (i + 1))


def assert_private(cfg: Cfg) -> None:
    a = api()
    for repo, rtype in ((cfg.cache_repo, "dataset"), (cfg.lora_repo, "model")):
        info = a.repo_info(repo, repo_type=rtype)
        if not info.private:
            raise SystemExit(f"{rtype} repo {repo} is not private: refusing to continue")
    say(f"private repos OK: {cfg.cache_repo} (dataset), {cfg.lora_repo} (model)")


def prepare_cache(cfg: Cfg) -> dict:
    from huggingface_hub import snapshot_download

    t = time.time()
    dl = cfg.work / "cache_dl"
    retry(lambda: snapshot_download(cfg.cache_repo, repo_type="dataset", revision=cfg.cache_rev,
                                    allow_patterns=[f"{cfg.cache_path}/*"], local_dir=str(dl),
                                    token=os.environ["HF_TOKEN"]), "cache download")
    src = dl / cfg.cache_path
    manifest = json.loads((src / "manifest.json").read_text())
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for sh in manifest["shards"]:
        tar_path = src / sh["name"]
        if tar_path.stat().st_size != sh["bytes"]:
            raise SystemExit(f"shard {sh['name']}: size mismatch")
        with tarfile.open(tar_path, "r") as tf:
            for m in tf:
                if not m.isfile() or not CACHE_NAME_RE.match(m.name):
                    raise SystemExit(f"unexpected member in {sh['name']}: {m.name!r}")
                fsrc = tf.extractfile(m)
                assert fsrc is not None
                with open(cfg.cache_dir / m.name, "wb") as fdst:
                    shutil.copyfileobj(fsrc, fdst, 1 << 22)
                n += 1
        tar_path.unlink()
    if n != manifest["n_files"]:
        raise SystemExit(f"cache: unpacked {n} files, manifest says {manifest['n_files']}")
    say(f"cache ready: {manifest['n_items']} items, {n} files, {manifest['total_bytes'] / 1e9:.2f} GB in {time.time() - t:.0f}s")
    event("cache_ready", items=manifest["n_items"], files=n, gb=round(manifest["total_bytes"] / 1e9, 3), seconds=round(time.time() - t, 1))
    return manifest


def prepare_dit(cfg: Cfg) -> str:
    if cfg.fake:
        return "fake-dit"
    local = env("LOCAL_DIT")
    if local:
        return local
    from huggingface_hub import hf_hub_download

    t = time.time()
    path = retry(lambda: hf_hub_download(env("DIT_REPO", required=True), env("DIT_FILE", required=True),
                                         revision=env("DIT_REV", "main"), local_dir=str(cfg.work / "weights"),
                                         token=os.environ["HF_TOKEN"]), "DiT download")
    gb = os.path.getsize(path) / 1e9
    say(f"DiT ready: {path} ({gb:.1f} GB in {time.time() - t:.0f}s)")
    event("dit_ready", gb=round(gb, 2), seconds=round(time.time() - t, 1))
    return path


def write_dataset_toml(cfg: Cfg, manifest: dict) -> Path:
    """Cache-only dataset config: the media directory is an empty folder, musubi trains from cache_directory."""
    ds = manifest["meta"]["dataset_config"]
    empty = cfg.work / "empty_media"
    empty.mkdir(parents=True, exist_ok=True)
    general = dict(ds["general"])
    if cfg.batch_size > 0:
        general["batch_size"] = cfg.batch_size
    dataset = dict(ds["dataset"])
    dataset["video_directory" if cfg.kind == "wan" else "image_directory"] = str(empty)
    dataset["cache_directory"] = str(cfg.cache_dir)

    def val(v):
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, (int, float)):
            return repr(v)
        if isinstance(v, list):
            return "[" + ", ".join(val(x) for x in v) + "]"
        return json.dumps(str(v))

    text = "[general]\n" + "".join(f"{k} = {val(v)}\n" for k, v in general.items())
    text += "\n[[datasets]]\n" + "".join(f"{k} = {val(v)}\n" for k, v in dataset.items())
    path = cfg.work / "dataset.toml"
    path.write_text(text)
    say("dataset.toml:\n" + text)
    return path


def read_remote_state(cfg: Cfg) -> dict | None:
    """Latest pushed state of this run, or None. Source of truth = state/komi_state.json (same commit as the state)."""
    from huggingface_hub import hf_hub_download

    try:
        p = hf_hub_download(cfg.lora_repo, f"{cfg.prefix}/state/komi_state.json", repo_type="model",
                            local_dir=str(cfg.work / "remote_meta"), force_download=True, token=os.environ["HF_TOKEN"])
        return json.loads(Path(p).read_text())
    except Exception as e:  # noqa: BLE001
        say(f"no remote state for {cfg.prefix} ({type(e).__name__})")
        return None


def pull_state(cfg: Cfg) -> Path:
    from huggingface_hub import snapshot_download

    dest = cfg.work / "resume_dl"
    if dest.exists():
        shutil.rmtree(dest)
    retry(lambda: snapshot_download(cfg.lora_repo, repo_type="model", allow_patterns=[f"{cfg.prefix}/state/*"],
                                    local_dir=str(dest), token=os.environ["HF_TOKEN"]), "state download")
    state = dest / cfg.prefix / "state"
    (state / "komi_state.json").unlink(missing_ok=True)
    say(f"pulled state: {sorted(p.name for p in state.iterdir())}")
    return state


class Pusher:
    """Uploads checkpoints/state to the PRIVATE model repo; serialised by a lock (trainer keeps running)."""

    def __init__(self, cfg: Cfg) -> None:
        self.cfg = cfg
        self.lock = threading.Lock()
        self.last_push_t = 0.0
        self.last_pushed_step = -1
        self.history: list[dict] = []

    def push(self, cum_step: int, ckpt: Path, state_dir: Path | None, final: bool, log_file: Path | None, seg: int) -> None:
        cfg = self.cfg
        with self.lock:
            if cum_step <= self.last_pushed_step and not final:
                return
            t = time.time()
            a = api()
            name = f"{cfg.run}.safetensors" if final else f"{cfg.run}-step{cum_step:08d}.safetensors"
            retry(lambda: a.upload_file(path_or_fileobj=str(ckpt), path_in_repo=f"{cfg.prefix}/{name}", repo_id=cfg.lora_repo,
                                        repo_type="model", commit_message=f"{cfg.run}: checkpoint step {cum_step}"), "checkpoint upload")
            if state_dir is not None and state_dir.is_dir():
                (state_dir / "komi_state.json").write_text(json.dumps(
                    {"run": cfg.run, "steps_done": cum_step, "max_steps": cfg.max_steps, "checkpoint": name,
                     "job_id": cfg.job_id, "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}))
                retry(lambda: a.upload_folder(folder_path=str(state_dir), path_in_repo=f"{cfg.prefix}/state", repo_id=cfg.lora_repo,
                                              repo_type="model", delete_patterns="*",
                                              commit_message=f"{cfg.run}: state step {cum_step}"), "state upload")
            self.history.append({"step": cum_step, "checkpoint": name, "seg": seg, "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
            progress = {"run": cfg.run, "kind": cfg.kind, "expert": cfg.expert if cfg.kind == "wan" else None,
                        "steps_done": cum_step, "max_steps": cfg.max_steps, "done": bool(final), "checkpoint": name,
                        "has_state": state_dir is not None, "job_id": cfg.job_id, "pushes": self.history[-50:],
                        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            retry(lambda: a.upload_file(path_or_fileobj=json.dumps(progress, indent=1).encode(), path_in_repo=f"{cfg.prefix}/progress.json",
                                        repo_id=cfg.lora_repo, repo_type="model", commit_message=f"{cfg.run}: progress {cum_step}"), "progress upload")
            if log_file is not None and log_file.exists():
                try:
                    a.upload_file(path_or_fileobj=str(log_file), path_in_repo=f"{cfg.prefix}/logs/{log_file.name}", repo_id=cfg.lora_repo,
                                  repo_type="model", commit_message=f"{cfg.run}: log seg {seg}")
                except Exception as e:  # noqa: BLE001
                    say(f"log upload failed: {type(e).__name__}")
            self.last_push_t = time.time()
            self.last_pushed_step = cum_step
            mb = ckpt.stat().st_size / 1e6
            say(f"pushed step {cum_step} ({name}, {mb:.0f} MB{' + state' if state_dir else ''}) in {time.time() - t:.0f}s")
            event("pushed", step=cum_step, checkpoint=name, final=bool(final), state=state_dir is not None,
                  seconds=round(time.time() - t, 1), mb=round(mb, 1))


# ----------------------------------------------------------------------------- GPU telemetry

class Gpu:
    def __init__(self) -> None:
        self.used = 0.0
        self.peak = 0.0
        self.total = 0.0
        self.util = 0
        self.ok = shutil.which("nvidia-smi") is not None
        if self.ok:
            threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self) -> None:
        while True:
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[0]
                used, total, util = [float(x) for x in out.split(",")]
                self.used, self.total, self.util = used / 1024, total / 1024, int(util)
                self.peak = max(self.peak, self.used)
            except Exception:  # noqa: BLE001
                pass
            time.sleep(3)

    def reset_peak(self) -> None:
        self.peak = self.used


# ----------------------------------------------------------------------------- training segments

def trainer_cmd(cfg: Cfg, dit: str, dataset_toml: Path, steps: int, resume_dir: Path | None) -> list[str]:
    tr = TRAINERS[cfg.kind]
    common = ["--dit", dit, "--dataset_config", str(dataset_toml), "--output_dir", str(cfg.out), "--output_name", cfg.run,
              "--logging_dir", str(cfg.work / "tb"), "--max_train_steps", str(steps), "--save_every_n_steps", str(cfg.save_every),
              "--save_state"]
    if resume_dir is not None:
        common += ["--resume", str(resume_dir)]
    if cfg.fake:
        return [sys.executable, str(HERE / "fake_trainer.py"), *common, *cfg.extra_args]
    acc = Path(sys.executable).parent / "accelerate"
    return [str(acc) if acc.exists() else "accelerate", "launch", "--num_cpu_threads_per_process", "1", "--mixed_precision", tr["mixed_precision"],
            str(cfg.musubi / "src" / "musubi_tuner" / tr["script"]), "--config_file", str(cfg.train_config), *common, *cfg.extra_args]


def run_segment(cfg: Cfg, seg: int, dit: str, dataset_toml: Path, offset: int, target: int, resume_dir: Path | None,
                pusher: Pusher, gpu: Gpu, hard_deadline: float | None) -> tuple[int, int]:
    """Train from cumulative step `offset` to `target`. Returns (exit_code, cumulative steps reached and pushed)."""
    steps = target - offset
    if cfg.out.exists():
        shutil.rmtree(cfg.out)
    cfg.out.mkdir(parents=True)
    cmd = trainer_cmd(cfg, dit, dataset_toml, steps, resume_dir)
    log_file = cfg.work / f"train_seg{seg}.log"
    say(f"segment {seg}: steps {offset} -> {target}" + (f" (resume from {resume_dir})" if resume_dir else ""))
    say("$ " + " ".join(shlex.quote(c) for c in cmd))
    event("segment_start", seg=seg, offset=offset, target=target, resumed=resume_dir is not None)
    gpu.reset_peak()
    child_env = dict(os.environ)
    child_env.setdefault("PYTHONUNBUFFERED", "1")
    # own process group: `accelerate launch` spawns the real trainer, a stop must reach both
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=child_env,
                            cwd=str(cfg.musubi) if not cfg.fake else None, start_new_session=True)

    state = {"step": 0, "last_emit": 0, "last_emit_t": 0.0, "loss": None, "s_per_step": None, "t_step1": None,
             "t_last": None, "pushed_local": 0, "stopping": False, "stop_reason": "", "last_bar_t": 0.0}
    final_is_done = target >= cfg.max_steps

    def complete_saves() -> list[tuple[int, Path, Path | None]]:
        out = []
        for p in sorted(cfg.out.glob(f"{cfg.run}-step*.safetensors")):
            m = re.search(r"-step(\d{8})\.safetensors$", p.name)
            if not m:
                continue
            s = int(m.group(1))
            st = cfg.out / f"{cfg.run}-step{s:08d}-state"
            # the state is complete once the trainer has moved on to a later step (or exited)
            if s < state["step"] or proc.poll() is not None:
                out.append((s, p, st if st.is_dir() else None))
        return out

    def maybe_push(force: bool = False) -> None:
        saves = complete_saves()
        if not saves:
            return
        s, ckpt, st = saves[-1]
        if s <= state["pushed_local"]:
            return
        if not force and time.time() - pusher.last_push_t < cfg.push_every_s:
            return
        pusher.push(offset + s, ckpt, st, final=False, log_file=log_file, seg=seg)
        state["pushed_local"] = s
        for old_s, old_ckpt, old_st in saves[:-1]:  # keep local disk small
            if old_st is not None:
                shutil.rmtree(old_st, ignore_errors=True)

    def push_loop() -> None:
        while proc.poll() is None:
            try:
                maybe_push()
            except Exception as e:  # noqa: BLE001
                say(f"push failed: {type(e).__name__}: {str(e)[:300]}")
            time.sleep(5)

    threading.Thread(target=push_loop, daemon=True).start()

    def on_fragment(frag: str) -> None:
        p = logparse.parse_tqdm(frag) if "steps:" in frag else None
        if p is None:
            if not frag.strip():
                return
            if re.search(r"\d+%\|", frag) and "100%" not in frag:  # other progress bars: at most one line / 5 s
                if time.time() - state["last_bar_t"] < 5:
                    return
                state["last_bar_t"] = time.time()
            print(frag, flush=True)
            return
        now = time.time()
        if p["step"] != state["step"]:
            if p["step"] >= 1 and state["t_step1"] is None:
                state["t_step1"] = (now, p["step"])
            state["t_last"] = now
        state["step"] = p["step"]
        if p["avr_loss"] is not None:
            state["loss"] = p["avr_loss"]
        if p["s_per_step"]:
            state["s_per_step"] = p["s_per_step"]
        due = p["step"] > state["last_emit"] and (p["step"] % cfg.log_every == 0 or p["step"] in (1, 2, 3, steps)
                                                    or now - state["last_emit_t"] > 120)
        if due and p["avr_loss"] is not None:
            wall = None
            if state["t_step1"] and p["step"] > state["t_step1"][1]:
                wall = (now - state["t_step1"][0]) / (p["step"] - state["t_step1"][1])
            metric(run=cfg.run, seg=seg, step=offset + p["step"], seg_step=p["step"], total=cfg.max_steps,
                   avr_loss=p["avr_loss"], s_per_step=round(wall, 3) if wall else None,
                   tqdm_s_per_step=round(p["s_per_step"], 3) if p["s_per_step"] else None,
                   vram_gb=round(gpu.used, 2), vram_peak_gb=round(gpu.peak, 2), gpu_util=gpu.util,
                   job_min=round((time.time() - T0) / 60, 2))
            state["last_emit"], state["last_emit_t"] = p["step"], now
        if p["avr_loss"] is not None and p["avr_loss"] != p["avr_loss"]:  # NaN
            event("nan_loss", step=offset + p["step"])

    def stop_child(reason: str) -> None:
        if not state["stopping"] and proc.poll() is None:
            state["stopping"] = True
            say(f"stopping trainer: {reason}")
            event("stopping", reason=reason, step=offset + state["step"])
            state["stop_reason"] = reason

            def _kill(sig):
                try:
                    os.killpg(proc.pid, sig)
                except ProcessLookupError:
                    pass

            _kill(signal.SIGTERM)
            threading.Timer(60, lambda: proc.poll() is None and _kill(signal.SIGKILL)).start()

    def watchdog() -> None:
        while proc.poll() is None:
            if hard_deadline is not None and time.time() > hard_deadline:
                stop_child("deadline reached (job timeout is near)")
                return
            time.sleep(5)

    threading.Thread(target=watchdog, daemon=True).start()
    signal.signal(signal.SIGTERM, lambda *_: stop_child("SIGTERM"))

    assert proc.stdout is not None
    buf = b""
    with open(log_file, "wb") as lf:
        for chunk in iter(lambda: proc.stdout.read1(65536), b""):
            lf.write(chunk)
            lf.flush()
            buf += chunk
            parts = re.split(rb"[\r\n]", buf)
            buf = parts.pop()
            for part in parts:
                on_fragment(part.decode("utf-8", "replace"))
            # tqdm ends a refresh with "]" and no newline: handle a complete bar now instead of one refresh later
            if buf.rstrip().endswith(b"]") and b"steps:" in buf and logparse.parse_tqdm(buf.decode("utf-8", "replace")):
                on_fragment(buf.decode("utf-8", "replace"))
                buf = b""
        if buf:
            on_fragment(buf.decode("utf-8", "replace"))
    rc = proc.wait()
    say(f"segment {seg}: trainer exit code {rc}, last step {state['step']}/{steps}, tqdm {state['s_per_step']} s/step, "
        f"peak VRAM {gpu.peak:.1f} GB")

    reached = offset + state["pushed_local"]
    final_ckpt = cfg.out / f"{cfg.run}.safetensors"
    final_state = cfg.out / f"{cfg.run}-state"
    if rc == 0 and final_ckpt.exists():
        pusher.push(target, final_ckpt, final_state if final_state.is_dir() else None, final=final_is_done, log_file=log_file, seg=seg)
        reached = target
    else:
        try:
            maybe_push(force=True)  # last complete save, if any
            reached = offset + state["pushed_local"]
        except Exception as e:  # noqa: BLE001
            say(f"final push failed: {type(e).__name__}: {str(e)[:300]}")
    event("segment_end", seg=seg, rc=rc, reached=reached, target=target, tqdm_s_per_step=state["s_per_step"],
          vram_peak_gb=round(gpu.peak, 2), stopped=state["stopping"], stop_reason=state["stop_reason"])
    if state["stopping"]:
        rc = STOPPED_RC
    return rc, reached


def main() -> int:
    cfg = Cfg()
    say(f"run={cfg.run} kind={cfg.kind} expert={cfg.expert} max_steps={cfg.max_steps} save_every={cfg.save_every} "
        f"push_every_min={cfg.push_every_s / 60:.0f} resume={cfg.resume} self_resume_at={cfg.self_resume_at} fake={cfg.fake} "
        f"cache={cfg.cache_repo}/{cfg.cache_path}@{cfg.cache_rev[:8]} job_id={cfg.job_id}")
    if not os.environ.get("HF_TOKEN"):
        raise SystemExit("HF_TOKEN secret is missing")
    cfg.work.mkdir(parents=True, exist_ok=True)
    assert_private(cfg)
    hard_deadline = T0 + cfg.deadline_s if cfg.deadline_s > 0 else None

    manifest = prepare_cache(cfg)
    dataset_toml = write_dataset_toml(cfg, manifest)
    dit = prepare_dit(cfg)
    gpu = Gpu()
    pusher = Pusher(cfg)

    offset, resume_dir = 0, None
    if cfg.resume:
        st = read_remote_state(cfg)
        if st is None:
            say("RESUME=1 but no pushed state exists: starting from step 0")
        else:
            offset = int(st["steps_done"])
            if offset >= cfg.max_steps:
                say(f"nothing to do: {offset} >= MAX_STEPS {cfg.max_steps}")
                event("job_done", steps=offset, max_steps=cfg.max_steps, note="already complete")
                return 0
            resume_dir = pull_state(cfg)
            event("resumed", steps_done=offset, checkpoint=st.get("checkpoint"))
    else:
        st = read_remote_state(cfg)
        if st is not None and env("ALLOW_OVERWRITE", "0") != "1":
            raise SystemExit(f"run {cfg.run} already has a pushed state at step {st['steps_done']}: use RESUME=1, "
                             f"another RUN_NAME, or ALLOW_OVERWRITE=1")

    seg, rc = 0, 0
    targets = [cfg.max_steps]
    if cfg.self_resume_at and offset < cfg.self_resume_at < cfg.max_steps:
        targets = [cfg.self_resume_at, cfg.max_steps]  # probe: verify push + pull + resume inside one job
    for target in targets:
        rc, reached = run_segment(cfg, seg, dit, dataset_toml, offset, target, resume_dir, pusher, gpu, hard_deadline)
        if rc != 0 or reached < target:
            stopped = rc == STOPPED_RC
            say(f"segment {seg} ended early ({'stopped by supervisor' if stopped else f'rc={rc}'}, pushed up to step {reached}/{target})")
            event("job_incomplete", steps=reached, max_steps=cfg.max_steps, rc=None if stopped else rc, stopped=stopped,
                  hint="relaunch with --resume")
            return 0 if stopped else (rc or 3)  # a deliberate stop is a clean exit: resume later
        offset = reached
        seg += 1
        if target < cfg.max_steps:
            st = read_remote_state(cfg)  # resume exactly like a fresh job would
            if st is None or int(st["steps_done"]) != offset:
                raise SystemExit(f"self-resume check failed: remote state {st} vs local step {offset}")
            resume_dir = pull_state(cfg)
            event("resumed", steps_done=offset, checkpoint=st.get("checkpoint"), self_test=True)
    event("job_done", steps=offset, max_steps=cfg.max_steps, minutes=round((time.time() - T0) / 60, 2),
          vram_peak_gb=round(gpu.peak, 2))
    say(f"done: {offset}/{cfg.max_steps} steps, final LoRA at {cfg.lora_repo}/{cfg.prefix}/{cfg.run}.safetensors")
    return 0


if __name__ == "__main__":
    sys.exit(main())
