# infra/hf_jobs — LoRA trainings on Hugging Face Jobs

Full-spec trainings (Wan 2.2 A14B I2V high/low experts, Qwen-Image-Edit-2511 keyframe LoRA) run on rented HF GPUs.
Only musubi-tuner **caches**, scripts/configs and LoRA checkpoints leave the machine, and only to **private** repos:

| Repo | Type | Content |
|---|---|---|
| `your-hf-user/komi-poc-train-cache` | dataset (private) | `wan/<set>/<variant>/shard-*.tar + manifest.json`, `kf/<set>/<variant>/...`, `jobs/` (scripts + train configs) |
| `your-hf-user/komi-poc-lora` | model (private) | `runs/<run>/<run>-stepNNNNNNNN.safetensors`, `runs/<run>/<run>.safetensors` (final), `runs/<run>/state/`, `progress.json`, `logs/` |

Files here:

| File | Role |
|---|---|
| `launch.py` | uploads `jobs/`, checks budget + cache, runs `hf jobs run --detach --flavor … --timeout … --secrets HF_TOKEN …`, writes `runs/train/<run>/job.json` |
| `bootstrap.sh` | runs in the job (stock `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime`): installs musubi-tuner @ `f8a1b037` into a venv on top of the image's torch, then starts `train_job.py` |
| `train_job.py` | job supervisor: cache + DiT download, cache-only dataset TOML, training, checkpoint/state push, `RESUME=1`, deadline stop, `KOMI_METRIC`/`KOMI_EVENT` log lines |
| `monitor.py` | polls `hf jobs inspect/logs`, writes `loss.csv` + one-line `status.json`, cancels on NaN/divergence/stall/over-budget, appends the cost to `runs/gpu_cost.jsonl` |
| `logparse.py` | tqdm / metric parsers + divergence rule (shared by the three above and the local smoke test) |
| `fake_trainer.py` | stand-in trainer for CPU tests of the supervisor (`launch.py check`) |
| `Dockerfile` | reference image definition (same pins as `bootstrap.sh`); not pushed anywhere |

All settings (repos, image, pinned commits/revisions, prices, weight paths) are in `configs/train_infra.yaml`;
hyper-parameters are in `configs/train_wan22_i2v_{high,low}.toml` and `configs/train_kf_qie2511.toml`.

## One-line commands

```bash
cd /home/user/komi-poc && . scripts/env.sh && MP=$KOMI_WORK/venvs/musubi/bin/python

# 0. sanity: cpu-basic job, secrets + private repo read/write + push/resume with the fake trainer (~$0.0002)
$MP infra/hf_jobs/launch.py check && $MP infra/hf_jobs/monitor.py --run "$(ls -t $KOMI_WORK/runs/train | grep '^check-cpu' | head -1)"

# 1. cache locally and upload (see docs/training.md for make-config / cache)
$MP -m src.train.wan_lora upload-cache --config $KOMI_WORK/data/processed/train/wan/configs/wan_832x480_f81.toml   # -> wan/full/832x480_f81

# 2. launch the two experts as two separate jobs (explicit timeout ~1.25x the projection: 29.7 s/step, see docs/training.md)
$MP infra/hf_jobs/launch.py wan --expert high --run wan22-i2v-high-v1 --cache-path wan/full/832x480_f81 --max-steps 2000 --timeout 21h
$MP infra/hf_jobs/launch.py wan --expert low  --run wan22-i2v-low-v1  --cache-path wan/full/832x480_f81 --max-steps 3000 --timeout 31h

# 3. monitor (one process per run; cancels on NaN/divergence; logs the cost when the job ends)
tmux new -d -s mon-high "$MP infra/hf_jobs/monitor.py --run wan22-i2v-high-v1 2>&1 | tee -a $KOMI_WORK/logs/monitor_wan22-i2v-high-v1.log"
tmux new -d -s mon-low  "$MP infra/hf_jobs/monitor.py --run wan22-i2v-low-v1  2>&1 | tee -a $KOMI_WORK/logs/monitor_wan22-i2v-low-v1.log"
cat $KOMI_WORK/runs/train/wan22-i2v-high-v1/status.json          # one-line status; loss.csv next to it

# 4. resume after a timeout / crash / cancel: same command + --resume (continues from the last pushed state)
$MP infra/hf_jobs/launch.py wan --expert high --run wan22-i2v-high-v1 --cache-path wan/full/832x480_f81 --max-steps 2000 --timeout 21h --resume

# 5. pull the LoRA back for ComfyUI (-> $KOMI_WORK/models/comfy/loras/<run>.safetensors)
$MP -m src.train.wan_lora pull --run wan22-i2v-high-v1            # final; add --step 2000 for an intermediate checkpoint

# keyframe LoRA (Qwen-Image-Edit-2511)
$MP -m src.train.kf_lora upload-cache --config $KOMI_WORK/data/processed/train/kf/configs/kf_1280x720_c1024x1024.toml
$MP infra/hf_jobs/launch.py kf --run kf-qie2511-v1 --cache-path kf/full/1280x720_c1024x1024 --max-steps 3000 --timeout 8h
$MP -m src.train.kf_lora pull --run kf-qie2511-v1

# emergency
/home/user/.local/bin/hf jobs ps                                   # anything still running?
/home/user/.local/bin/hf jobs cancel <job_id>
```

`launch.py … --dry-run` prints the resolved `hf jobs run` configuration without submitting.

## How a job runs

1. `launch.py` uploads this folder + `configs/train_*.toml` to `jobs/` of the dataset repo and pins the job to that
   revision (`JOBS_REV` = `CACHE_REV`). The job's command is a one-liner that creates a venv, installs
   `huggingface_hub`, downloads `jobs/` and executes `bootstrap.sh`.
2. `bootstrap.sh` installs musubi-tuner at the pinned commit (tarball, no git needed), keeps the image's
   torch 2.8.0+cu128 (pip constraint), swaps `opencv-python` for the headless build and verifies the imports.
3. `train_job.py` downloads and unpacks the cache shards, downloads the fp16/bf16 DiT from the Hub (28.6 GB in ~17 s
   inside HF), writes a dataset TOML whose media directory is empty (musubi trains from `cache_directory` alone) and
   starts `accelerate launch … {wan,qwen_image}_train_network.py --config_file <train toml> …` with
   `--save_every_n_steps SAVE_EVERY --save_state`.
4. Every time a new checkpoint + state is complete and at least `PUSH_EVERY_MIN` minutes have passed since the last
   push, the LoRA is uploaded as `runs/<run>/<run>-step<cumulative>.safetensors` and the state replaces
   `runs/<run>/state/`. The final LoRA is `runs/<run>/<run>.safetensors`.
5. `DEADLINE_MIN` (timeout − 5 min, − 2 min for jobs ≤ 30 min) stops the trainer cleanly before HF kills the job and
   pushes the last complete checkpoint; the job then exits 0 with `KOMI_EVENT {"event": "job_incomplete"}`.

### Resume semantics

musubi restores network/optimizer/scheduler state with `--resume` but restarts its step counter at 0, so the
supervisor does the bookkeeping: `--max-steps` is always the **total** for the run; a resumed job trains
`max_steps − steps_done` steps and renames its checkpoints to cumulative steps. This is exact with a constant LR
(`constant_with_warmup` in the configs); do not switch to cosine/linear decay without changing that logic.
A fresh launch refuses to overwrite a run that already has a pushed state (use `--resume`, a new `--run`, or
`--allow-overwrite`).

### Launch options worth knowing

| Option | Default | Meaning |
|---|---|---|
| `--timeout` | 24h (wan), 8h (kf), 20m (probe) | always explicit; HF's default is 30 min |
| `--save-every` | 100 (wan), 250 (kf) | steps between local checkpoints (+state); bounds the work lost at a hard kill |
| `--push-every-min` | 30 | minimum minutes between pushes |
| `--extra-args` | – | extra musubi arguments; write it with `=` so argparse accepts a value starting with `--`: `--extra-args="--learning_rate 5e-5 --network_dim 64 --network_alpha 32"` |
| `--flavor` | `rtx-pro-6000` | `h200` for the 720p pass; add its price to `configs/train_infra.yaml` if missing |
| `--max-usd` | – | refuse if timeout × price exceeds this |
| `--force` | – | override the budget guard: a launch is refused when logged spend (`runs/gpu_cost.jsonl`) + worst case (timeout × price) of running/unlogged jobs + worst case of the new job > 80 % of the GPU budget (`budget.gpu_usd` in `configs/base.yaml`) |
| `--self-resume-at N` | – | probe only: stop at step N, pull the state back from the Hub and continue in the same job |

### Monitor rules

`status.json` verdicts: `ok`, `diverged` (NaN/inf, loss > 2.0 after 30 steps, or > 3× the best moving average for three
reports), `stalled` (no new step for `--stall-min`, default 30 min), `over_budget` (`--max-usd`). Anything but `ok`
cancels the job unless `--no-cancel` is given. When the job reaches a terminal state the monitor appends
`{ts, job_id, flavor, minutes, usd, run, stage}` to `runs/gpu_cost.jsonl` (`minutes` = `durations.running_secs` from
`hf jobs inspect`; CANCELED jobs come back without durations, so the monitor logs an upper-bound estimate and adds
`"estimated": true`). `usd` = minutes × the list price in `configs/train_infra.yaml`, not an invoice.
A job that ended while no monitor was running is logged by `monitor.py --run <run> --once`.

## Measured on 2026-11-10 (rtx-pro-6000, Wan 2.2 I2V high-noise, 832×480×81, batch 1)

29.7 s/step and 59 GB peak VRAM with the shipped config; scheduling 4 s, bootstrap 38 s, 28.6 GB DiT download 17 s,
push of a 307 MB LoRA + 928 MB state 8–10 s. Details and projections: `docs/training.md` §3.
