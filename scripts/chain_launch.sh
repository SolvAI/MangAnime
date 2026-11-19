#!/usr/bin/env bash
# Training launch chain (after the clip captions exist): T5 cache -> upload caches -> launch both Wan experts on HF Jobs.
# The local smoke test (reduced size, fp8 + block swap) runs while the upload is in progress.
. /home/user/komi-poc/scripts/env.sh; cd /home/user/komi-poc
export GPU_PRIO=1 GPU_WAIT_MIN=900
MP=$KOMI_WORK/venvs/musubi/bin/python
W=$KOMI_WORK/data/processed/train/wan
CFG=$W/configs/wan_832x480_f81.toml
STEPS="${STEPS:-1500}"; TIMEOUT="${TIMEOUT:-14h}"; LR="${LR:-2e-4}"
step() { echo; echo "=== $(date +%T) $*"; }
until [ -f $KOMI_WORK/runs/queue/done_wan_captions ]; do sleep 5; done
step "stop qwen (the text encoder needs ~11 GB)"; python -m src.common.servers stop qwen
step "make-config (final clip list)"; $MP -m src.train.wan_lora make-config --clips $W/clips | tail -1
step "T5 text-encoder cache"; $MP -m src.train.wan_lora cache --config $CFG --only te --need-gb 12
touch $KOMI_WORK/runs/queue/done_te_cache
step "verify cache completeness (file counts, no GPU)"
python - <<'PY'
import glob, os, sys
W = "/home/user/komi-poc-work/data/processed/train/wan"
clips = sorted(os.path.basename(f)[:-4] for f in glob.glob(W + "/clips/*.mp4"))
cache = W + "/cache/wan_832x480_f81"
lat = {os.path.basename(f).split("_00000-")[0] for f in glob.glob(cache + "/*_wan.safetensors")}
te = {os.path.basename(f)[: -len("_wan_te.safetensors")] for f in glob.glob(cache + "/*_wan_te.safetensors")}
missing_lat = [c for c in clips if c not in lat]; missing_te = [c for c in clips if c not in te]
stale = [f for f in glob.glob(cache + "/*.safetensors") if not any(os.path.basename(f).startswith(c + "_") for c in clips)]
for f in stale: os.remove(f)          # caches of purged / uncaptioned clips must not be uploaded
print(f"clips={len(clips)} latents={len(lat & set(clips))} te={len(te & set(clips))} stale_removed={len(stale)} missing_lat={missing_lat[:3]} missing_te={missing_te[:3]}")
sys.exit(1 if missing_lat or missing_te else 0)
PY
[ $? -eq 0 ] || { echo "CACHE INCOMPLETE - not launching"; exit 1; }
step "upload caches to the private dataset repo"; $MP -m src.train.wan_lora upload-cache --config $CFG 2>&1 | tail -4
step "launch high-noise expert"; $MP infra/hf_jobs/launch.py wan --expert high --run wan22-i2v-high-v1 --cache-path wan/full/832x480_f81 \
     --max-steps $STEPS --timeout $TIMEOUT --save-every 100 --extra-args="--learning_rate $LR" 2>&1 | tail -6
step "launch low-noise expert"; $MP infra/hf_jobs/launch.py wan --expert low --run wan22-i2v-low-v1 --cache-path wan/full/832x480_f81 \
     --max-steps $STEPS --timeout $TIMEOUT --save-every 100 --extra-args="--learning_rate $LR" 2>&1 | tail -6
touch $KOMI_WORK/runs/queue/done_launch
step "monitors"; tmux new-session -d -s mon_high "cd /home/user/komi-poc && . scripts/env.sh && $MP infra/hf_jobs/monitor.py --run wan22-i2v-high-v1 2>&1 | tee $KOMI_WORK/logs/mon_high.log"
tmux new-session -d -s mon_low "cd /home/user/komi-poc && . scripts/env.sh && $MP infra/hf_jobs/monitor.py --run wan22-i2v-low-v1 2>&1 | tee $KOMI_WORK/logs/mon_low.log"
step "CHAIN_LAUNCH_DONE"
