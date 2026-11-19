#!/usr/bin/env bash
# Qwen burst 1 (critical path for the HF training launch): captions for the Wan LoRA clips.
. /home/user/komi-poc/scripts/env.sh; cd /home/user/komi-poc
export GPU_PRIO=1 GPU_WAIT_MIN=900
echo "=== $(date +%T) waiting for the latent cache to finish"
until [ -f $KOMI_WORK/runs/queue/done_cache_latents ]; do sleep 5; done
echo "=== $(date +%T) purge clips with an internal TransNetV2 cut"; python -m src.train.build_wan_dataset purge
echo "=== $(date +%T) start qwen"; python -m src.common.servers start qwen
echo "=== $(date +%T) wan captions"; python -m src.train.build_wan_dataset caption
echo "=== $(date +%T) finalize"; python -m src.train.build_wan_dataset finalize
touch $KOMI_WORK/runs/queue/done_wan_captions
echo "=== $(date +%T) CHAIN_QWEN1_DONE"
