#!/usr/bin/env bash
# Qwen burst 3 (evaluation-side, run whenever the LLM server is up anyway): held-out shot captions, VLM spot-check,
# LoRA-grid captions, judge calibration evidence.
. /home/user/komi-poc/scripts/env.sh; cd /home/user/komi-poc
export GPU_PRIO=1 GPU_WAIT_MIN=900 PYTHONUNBUFFERED=1
EP="${1:-22}"
step() { echo; echo "=== $(date +%T) $*"; }
step "shot captions of the held-out episode $EP (evaluation only)"; python -m src.anime.build_shots caption --episodes "$EP"
step "VLM spot-check of 200 aligned pairs"; python -m src.align.build spotcheck
step "LoRA grid captions"; python -m src.eval.lora_grid caption
step "shots.jsonl"; python -m src.anime.build_shots merge
step "CHAIN_QWEN3_DONE"
