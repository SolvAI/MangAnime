# LoRA training — infra, commands, measured numbers

Scope: the Wan 2.2 A14B I2V style LoRA (high-noise + low-noise experts) and the Qwen-Image-Edit-2511 keyframe LoRA
(manga panel → anime frame). Trainer: **musubi-tuner @ `f8a1b037`** (2026-09-27, v0.3.6). Local 24 GB card = caching +
smoke test only; full-spec runs go to Hugging Face Jobs (`rtx-pro-6000`, 96 GB). Job-side details: `infra/hf_jobs/README.md`.

| Piece | Where |
|---|---|
| venv (torch 2.8.0+cu128, musubi editable install) | `/home/user/komi-poc-work/venvs/musubi` |
| musubi-tuner checkout (pinned) | `/home/user/komi-poc-work/third_party/musubi-tuner` |
| CLIs | `src/train/wan_lora.py`, `src/train/kf_lora.py` (+ `common.py`, `musubi_cpu.py`, `synthetic_cache.py`, `vram_watch.py`, `smoke_report.py`) |
| HF Jobs | `infra/hf_jobs/{launch,monitor,train_job,logparse,fake_trainer}.py`, `bootstrap.sh`, `Dockerfile`, `README.md` |
| Settings / hyper-parameters | `configs/train_infra.yaml`, `configs/train_wan22_i2v_{high,low}.toml`, `configs/train_kf_qie2511.toml` |
| Private Hub repos | dataset `your-hf-user/komi-poc-train-cache`, model `your-hf-user/komi-poc-lora` |
| Run outputs | `runs/train/<run>/{job.json,status.json,loss.csv,job_<id>.log}`, costs in `runs/gpu_cost.jsonl` |

## 1. Dataset contract

**Wan I2V** — `data/processed/train/wan/clips/<clip_id>.mp4` (832×480, 16 fps, exactly 81 frames) + `<clip_id>.txt`
(caption starting with `komi_anime_style`). `make-config` refuses: missing captions, captions without the trigger,
clips with fewer than 81 frames or not 16 fps (ffprobe sample, `--probe -1` = all), and any id containing
`ep22`–`ep24` (held-out). Dataset TOML: fixed resolution (no bucketing), `target_frames = [81]`,
`frame_extraction = "head"`.

**Keyframe (Qwen-Image-Edit-2511)** — `data/processed/train/kf/control/<id>.png` (panel), optional
`control/<id>_ref.png` (character reference), `target/<id>.png` (anime frame, 16:9), `target/<id>.txt` (instruction).
musubi's directory mode needs numbered control files, so `make-config` writes a metadata JSONL
(`image_path`, `control_path_0`, `control_path_1`, `caption`) next to the TOML; nothing is renamed. Defaults: target
bucket 1280×720 (bucketing on), `control_resolution = [1024, 1024]` (the official resizing, "strongly recommended" by
the musubi docs). Items with one and with two control images can be mixed (musubi buckets them separately).

Trainer-loadable weights only: musubi cannot read the ComfyUI `fp8_scaled` / `fp8mixed` files used for inference.
Local copies (paths in `configs/train_infra.yaml`): Wan fp16 DiTs on
`/path/to/disk/komi-poc-models/train/wan22/` (2 × 28.6 GB), Qwen bf16 DiT (40.9 GB) + Qwen2.5-VL
(16.6 GB) under `models/train/qwen/`, Wan T5 = the official `models_t5_umt5-xxl-enc-bf16.pth`.

## 2. Commands

```bash
cd /home/user/komi-poc && . scripts/env.sh && MP=$KOMI_WORK/venvs/musubi/bin/python
W=$KOMI_WORK/data/processed/train/wan

# --- Wan: config -> cache -> smoke -> upload
$MP -m src.train.wan_lora make-config --clips $W/clips                      # -> $W/configs/wan_832x480_f81.toml
$MP -m src.train.wan_lora cache --config $W/configs/wan_832x480_f81.toml    # I2V latents + T5, via scripts/gpu_run.sh 14
$MP -m src.train.wan_lora smoke --expert both                               # 60 steps each on the 8-clip smoke set (gpu_run 22)
$MP -m src.train.wan_lora upload-cache --config $W/configs/wan_832x480_f81.toml   # -> wan/full/832x480_f81 (private)

# --- HF Jobs: two experts = two jobs; monitor; resume; pull back
$MP infra/hf_jobs/launch.py wan --expert high --run wan22-i2v-high-v1 --cache-path wan/full/832x480_f81 --max-steps 2000 --timeout 21h
$MP infra/hf_jobs/launch.py wan --expert low  --run wan22-i2v-low-v1  --cache-path wan/full/832x480_f81 --max-steps 3000 --timeout 31h
$MP infra/hf_jobs/monitor.py --run wan22-i2v-high-v1        # loop; status in runs/train/<run>/status.json, loss.csv
$MP infra/hf_jobs/launch.py wan --expert high --run wan22-i2v-high-v1 --cache-path wan/full/832x480_f81 --max-steps 2000 --timeout 21h --resume
$MP -m src.train.wan_lora pull --run wan22-i2v-high-v1      # -> $KOMI_WORK/models/comfy/loras/wan22-i2v-high-v1.safetensors

# --- Keyframe LoRA
K=$KOMI_WORK/data/processed/train/kf
$MP -m src.train.kf_lora make-config --root $K                               # -> $K/configs/kf_1280x720_c1024x1024.{toml,jsonl}
$MP -m src.train.kf_lora cache --config $K/configs/kf_1280x720_c1024x1024.toml
$MP -m src.train.kf_lora smoke                                               # 8 stand-in pairs, 50 steps (gpu_run 22)
$MP -m src.train.kf_lora upload-cache --config $K/configs/kf_1280x720_c1024x1024.toml   # -> kf/full/1280x720_c1024x1024
$MP infra/hf_jobs/launch.py kf --run kf-qie2511-v1 --cache-path kf/full/1280x720_c1024x1024 --max-steps 3000 --timeout 8h
$MP -m src.train.kf_lora pull --run kf-qie2511-v1

# --- helpers
$MP -m src.train.wan_lora check-args --expert both      # CPU only: parses config + CLI, prints the timestep histogram per expert
$MP -m src.train.wan_lora cache --device cpu --config … # CPU caching (GPU hidden); only practical for a few small clips
$MP -m src.train.wan_lora make-synthetic-cache --upload # random, shape-identical cache for speed/VRAM probes (no media)
$MP infra/hf_jobs/launch.py check                       # cpu-basic: secrets + private repos + push/resume (fake trainer)
```

The whole local GPU chain (real smoke caches → Wan smoke high/low → upload of the real smoke cache → keyframe cache +
smoke → results into this file) is queued as one script that waits for the card:
`tmux new -d -s komi-train-smoke "bash $KOMI_WORK/runs/queue/train_local_gpu_smoke.sh"` (log:
`$KOMI_WORK/logs/train_local_gpu_smoke.log`).

**ComfyUI.** `pull` drops the file into `models/comfy/loras/` (already on ComfyUI's search path through
`configs/comfy_extra_model_paths.yaml`). musubi LoRAs use kohya-style keys (`lora_unet_blocks_…`); the musubi docs
state the Wan LoRA "seems to be compatible with ComfyUI (may depend on the nodes used)", so **no conversion is expected**
for the core `LoraLoaderModelOnly` node — load the high-noise LoRA on the high-noise model and the low-noise LoRA on
the low-noise model. If a node rejects the file, `pull --convert` also writes `<name>_diffusers.safetensors` with
musubi's `convert_lora.py --target other` (supported for Wan and Qwen-Image keys). Loading in ComfyUI has **not**
been tested in this workstream.

## 3. Measured numbers

### 3.1 HF Jobs probe — Wan 2.2 I2V high-noise expert at full spec

`rtx-pro-6000` (RTX PRO 6000 Blackwell Server Edition, 97.9 GB), image `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime`,
81 frames, 832×480, batch 1, fp16 weights, **no fp8, no block swap**, gradient checkpointing, SDPA, rank 32,
`configs/train_wan22_i2v_high.toml`. Three short jobs, $0.77 in total. The cache was a **synthetic, shape-identical**
one (8 items of random tensors written with musubi's own cache writers): throughput and VRAM depend only on tensor
shapes, so this measures the real cost, but the loss values (≈ 1.2) mean nothing. Reason: caching real clips at full
spec on the CPU was not feasible while the local GPU was taken (§6).

<!-- HF_PROBE_BEGIN -->
| Job (run) | Settings | Steps | s/step (tqdm average) | Peak VRAM | Running time / cost |
|---|---|---|---|---|---|
| `6af369acab34849891aca5f3` (`probe-wan22-i2v-high-1110a`) | stock Wan 2.2 path | 14 = 8, push, pull state back from the Hub, resume, 6 | 30.8 – 31.4 | 67.9 GB | 9.2 min, $0.42 |
| `6af36c19ab34849891acaab2` (`probe-wan22-i2v-high-compile-1110b`) | + `--compile --compile_mode default --cuda_allow_tf32 --cuda_cudnn_benchmark` | 4 (cancelled once the rate was clear) | 30.6 – 30.7 | 73.0 GB | 3.6 min, $0.17 |
| `6af36d29ab34849891acaca7` (`probe-wan22-i2v-high-v21te-1110c`) | + `--force_v2_1_time_embedding` (**now the default in the configs**) | 5 | **29.6 – 29.7** | **59.2 GB** | 4.1 min, $0.19 |

| Fixed overhead (rtx-pro-6000) | Measured |
|---|---|
| Scheduling (image already cached on HF) | 4 s |
| Bootstrap: venv + musubi install at the pinned commit | 38 s |
| Cache shard + DiT download inside HF (28.6 GB) | 19 s |
| Trainer start + model load until the first step begins | ≈ 14 s |
| Checkpoint + state push (307 MB fp16 LoRA, 928 MB state) | 8 – 10 s, while training continues |

Reading: `torch.compile` buys ~1.5 % for +5 GB, not worth the extra moving part. The 2.1-style time embedding is, by
the code, the same computation for A14B training (every token of a sample shares one timestep); consistent with that,
both runs printed the same five moving-average losses (1.18, 1.18, 1.21, 1.20, 1.22). It saves ~9 GB and ~4 %.
<!-- HF_PROBE_END -->

Also verified on HF Jobs (cpu-basic, $0.0005 in total): `HF_TOKEN` secret, private dataset read, private model write,
push → pull → resume with the fake trainer (`check-cpu-1110-2252`), the musubi installation in the PyTorch image
(`bootcheck-1110-2254`), and the monitor cancelling a job on a NaN loss (`check-nan-1110`, fake trainer).

### 3.2 Local smoke test (RTX 4090, 24 GB)

Settings: 8 clips cut from episode 1, 416×240, 33 frames, fp8_scaled base + `--blocks_to_swap 20`, 60 steps per
expert; keyframe LoRA: 8 stand-in pairs at 832×480 (controls 512×512), fp8_scaled + `--blocks_to_swap 30`, 50 steps.
The local GPU was fully occupied by another job during this session, so nothing below was measured by hand: the chain
`runs/queue/train_local_gpu_smoke.sh` is queued behind `scripts/gpu_run.sh 22` and rewrites the table when it ends
("not run yet" = still waiting or failed, see `logs/train_local_gpu_smoke.log`).

<!-- LOCAL_SMOKE_BEGIN -->
_Updated 2026-11-10 23:20 by `src/train/smoke_report.py` (RTX 4090 24 GB, real smoke-set caches)._

| Run | Result | Size | 24 GB settings | Steps | s/step | Peak VRAM (GB) | avr_loss first → last |
|---|---|---|---|---|---|---|---|
| Wan 2.2 I2V high-noise | not run yet | | | | | | |
| Wan 2.2 I2V low-noise | not run yet | | | | | | |
| Qwen-Image-Edit-2511 keyframe | not run yet | | | | | | |
<!-- LOCAL_SMOKE_END -->

### 3.3 Projection (rtx-pro-6000 at $2.75/h, 29.7 s/step with the shipped configs, batch 1)

One step = one 81-frame clip, so one epoch over N clips = N steps; $0.0227 per step. Fixed overhead per job launch
≈ 1.5 min ($0.07). The low-noise expert runs the same network at the same shapes, so the same s/step is assumed (not
measured separately).

| Steps per expert | Hours | $ per expert | $ both experts |
|---|---|---|---|
| 1,000 | 8.3 | 23 | 45 |
| 1,500 | 12.4 | 34 | 68 |
| 2,000 | 16.5 | 45 | 91 |
| 3,000 | 24.8 | 68 | 136 |
| 4,000 | 33.0 | 91 | 182 |

This is slower than the brief's rough estimate (8–20 GPU-hours per expert). With the $200 GPU budget and the rule to
stop launching at 80 %, the recommended split is **high 2,000 + low 3,000 steps ≈ 41 h ≈ $113**, leaving room for the
keyframe LoRA and a resume. `launch.py` refuses a launch when logged spend + worst case (timeout × price) of running or
unlogged jobs + worst case of the new job passes 80 % of the budget, so keep timeouts near 1.25× the projection
(2,000 steps → `--timeout 21h`, 3,000 → `--timeout 31h`; worst case $143 for the pair). A job that hits its timeout
stops cleanly and is continued with `--resume`, so a tight timeout costs at most one relaunch.
Cache size: 9.4 MB of latents + ~1 MB of text embeddings per clip → ≈ 16 GB for 1,500 clips (16 one-GB shards).

Keyframe LoRA on HF: **not probed**. musubi's table gives 42 GB VRAM at 1024² without fp8 (more with control
images), so it fits the 96 GB card; run a 15-minute probe on the real cache first
(`launch.py kf --run kf-probe --cache-path … --max-steps 20 --save-every 10 --timeout 15m`).

## 4. Recommended hyper-parameters (style/character LoRA, ~1,000–1,500 clips)

Starting points; none of these was tuned by experiment here. The Wan 2.2 specifics follow musubi's `docs/wan.md`:
one expert per run via `--dit`, the recommended min/max timestep table for I2V, and `--preserve_distribution_shape`.

| Setting | High-noise expert | Low-noise expert | Note |
|---|---|---|---|
| `--task` / DiT | `i2v-A14B`, `wan2.2_i2v_high_noise_14B_fp16` | `i2v-A14B`, `wan2.2_i2v_low_noise_14B_fp16` | two separate jobs |
| `min_timestep` – `max_timestep` | 900 – 1000 | 0 – 900 | I2V boundary 0.9 (docs table) |
| `preserve_distribution_shape` | on | on | rejection sampling inside the range |
| `timestep_sampling` / `discrete_flow_shift` | `shift` / 5.0 | `shift` / 3.0 | 5.0 = official I2V inference shift; 3.0 = musubi's 480p I2V default, spreads over the detail range (`check-args` prints both histograms) |
| Rank / alpha | 32 / 16 | 32 / 16 | 64 / 32 if identity of the side cast is weak |
| Optimizer / LR | adamw8bit, 1e-4, `constant_with_warmup` (100 steps) | same | constant LR is required by the resume logic |
| Steps | 2,000 | 3,000 | ≈ 1.5–3 epochs; the low-noise expert carries line/colour style and covers 90 % of the range |
| Precision | `--mixed_precision fp16` | same | the Comfy-Org DiT is fp16; musubi rejects bf16 for it |
| Memory options | gradient checkpointing + `force_v2_1_time_embedding` | same | 59 GB peak (68 GB without the flag); batch 2 does not fit; `torch.compile` not worth it (§3.1) |
| Checkpoints | `--save-every 100`, push every 30 min | same | compare steps 1000 / 1500 / 2000 before spending the rest |

Other levers, in the order I would try them: lower LR to 5e-5 if the monitor reports divergence or the style
over-bakes (`--extra-args="--learning_rate 5e-5"`); LoRA+ (`--extra-args="--network_args loraplus_lr_ratio=4"`) if
2,000 steps underfit. The musubi doc example uses alpha 1 (default) with LR 2e-4, a much
weaker effective update; alpha 16 with 1e-4 sits between that and the alpha = rank, 1e-4 convention of other trainers.

**Keyframe LoRA (Qwen-Image-Edit-2511)** — `--model_version edit-2511`, bf16, rank 32 / alpha 16, adamw8bit, LR 5e-5,
`timestep_sampling shift`, `discrete_flow_shift 2.2`, `weighting_scheme none` (musubi doc example values), 3,000 steps
as a first budget, `control_resolution [1024, 1024]`, target bucket 1280×720.

## 5. Checkpoints, resume, monitoring

- Model repo layout: `runs/<run>/<run>-step00000100.safetensors` …, final `runs/<run>/<run>.safetensors`,
  `runs/<run>/state/` (latest accelerate state), `runs/<run>/progress.json`, `runs/<run>/logs/`.
- `--max-steps` is the total for the run. `--resume` trains the remainder from the last pushed state; checkpoints keep
  cumulative step numbers. musubi itself restarts its counter at 0 after `--resume`, the job supervisor does the
  bookkeeping (exact with a constant LR). Cosmetic side effect: the `ss_steps` metadata inside a checkpoint written
  after a resume counts only the steps of that job; the file name and `progress.json` carry the cumulative step.
- A job stops its trainer 5 min before the timeout and pushes the last complete checkpoint; at most `--save-every`
  steps are lost on a hard kill.
- `monitor.py` cancels on NaN/inf, loss > 2.0 after 30 steps, loss > 3× the best moving average three times in a row,
  no new step for 30 min, or `--max-usd`. The 2.0 limit is a blow-up guard, not a quality signal: unpredictable
  (synthetic) latents sat at ≈ 1.2, real data must stay below that. Thresholds are flags (`--loss-limit`,
  `--div-factor`, `--stall-min`).
- Hand-off for dashboards: `runs/train/<run>/loss.csv` and the one-line `status.json`. Mirroring into MLflow /
  `STATUS.md` is not part of these tools.

## 6. What broke / gotchas

1. **musubi's Wan modules cannot be imported without a visible GPU** (`wan/modules/t5.py` calls
   `torch.cuda.current_device()` in a default argument). `src/train/musubi_cpu.py` hides the GPU, stubs that call
   and switches T5 to float32 for the CPU path.
2. **CPU caching at full spec is not practical**: the I2V cache runs the Wan VAE encoder twice per clip (clip + first
   frame padded to 81 frames); on the loaded machine the first 832×480×81 clip had not finished after 13 minutes, so
   the CPU run was stopped. Consequences: the paid probe used a synthetic cache, and the real smoke caches + local
   smoke test are queued for the GPU.
3. **ComfyUI inference weights are not trainer-loadable** (`fp8_scaled`, `fp8mixed`, KJ fp8): the fp16/bf16 files had
   to be downloaded separately (114 GB in total).
4. **fp16 DiT ⇒ `--mixed_precision fp16`** (assert in `wan_train_network.py`); there is no bf16 Wan 2.2 repack.
5. **musubi `--resume` does not restore the step counter** (`TODO skip until initial step` in `trainer_base.py`), hence
   the cumulative-step logic and the constant-LR requirement.
6. **Progress is only a tqdm bar**; the supervisor turns it into `KOMI_METRIC` lines (loss is musubi's moving average).
7. **Qwen-Image-Edit cache files carry the suffix `_qie`**, not `_qi`; text-encoder caches of every architecture store
   the caption text in their safetensors metadata (`caption1`), so captions travel with the uploaded caches.
8. **Two `hf` CLIs**: the venvs ship 1.32.0, the Jobs syntax used here is from `/home/user/.local/bin/hf` 2.2.0 (path
   pinned in `configs/train_infra.yaml`).
9. **Throughput**: ≈ 30 s/step at full spec, i.e. 16.5 h per 2,000 steps; the brief's 8–20 GPU-hours per expert only
   buys 1,000–2,400 steps (§3.3).
10. **`--extra-args` needs `=`** (`--extra-args="--flag …"`): argparse rejects a separate value that starts with `--`.
11. **`hf jobs inspect` returns no `durations` for CANCELED jobs**; the monitor then logs an upper-bound estimate
    (`"estimated": true` in `runs/gpu_cost.jsonl`).

## 7. Verified / not verified

| Item | State |
|---|---|
| venv, musubi import, config parsing for all three trainers, timestep ranges per expert (`check-args`) | verified (CPU) |
| Cache-only dataset loading (empty media directory + cache files) | verified (CPU and in the HF job) |
| HF Jobs: secrets, private repos, bootstrap, full-spec training step, VRAM, checkpoint push, state pull + resume | verified (paid probe + cpu-basic checks) |
| Monitor: loss.csv, status.json, cost logging, cancel on NaN | verified on real jobs (NaN cancel with the fake trainer on cpu-basic); the divergence-ratio, stall and `--max-usd` rules are implemented but were never triggered |
| Real cache → trainer (latents made by `wan_cache_latents.py` from real clips) | **not verified**: the HF probes used the synthetic cache. The queued local smoke test covers it locally; on HF it is first exercised by the first real launch, or by a short probe on `wan/wan_smoke/832x480_f81` once the queue has uploaded it (`launch.py probe --cache-path wan/wan_smoke/832x480_f81 --max-steps 6 --save-every 3 --timeout 12m`, ≈ $0.30) |
| Local smoke training, both Wan experts and the keyframe LoRA | **not verified** (GPU busy); queued |
| Low-noise expert and keyframe LoRA on HF Jobs | **not run** (same code path; different DiT / trainer script) |
| `pull` and `pull --convert` | verified on the probe checkpoint: 400 LoRA modules, keys `lora_unet_blocks_0_cross_attn_k.lora_down.weight` (musubi) and `diffusion_model.blocks.0.cross_attn.k.lora_A.weight` (converted copy) |
| LoRA loading in ComfyUI | **not verified** |
| Deadline stop before the job timeout, then `RESUME=1` in a new process | verified locally with the fake trainer (stop, push of the last complete checkpoint, exit 0, resume to the end); not exercised by a real HF timeout |
| Real `--resume` relaunch as a second HF job | not run as a separate job; the probe pulled the state back from the Hub and resumed inside the same job (same code path, minus the bootstrap) |
