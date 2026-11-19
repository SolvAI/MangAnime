#!/usr/bin/env bash
# GPU hand-over chain S: TransNetV2 shot boundaries + keyframes for a list of episodes (2 GB).
. /home/user/komi-poc/scripts/env.sh; cd /home/user/komi-poc
export GPU_PRIO=1 GPU_WAIT_MIN=900
scripts/gpu_run.sh 2 python -m src.anime.shots --episodes "$1" 2>&1 | grep -v "Warn\|warn"
echo "=== $(date +%T) CHAIN_SHOTS_DONE $1"
