#!/usr/bin/env bash
# Restart the v2 pipeline at the QA stage so it loads the current judge (src/qa/judge.py).
. /home/user/komi-poc/scripts/env.sh; cd /home/user/komi-poc
RUN="ep22_v2"; LOG="$KOMI_WORK/logs/$RUN.log"
until grep -q "=== stage: qa" "$LOG"; do sleep 5; done
tmux kill-session -t "=ep22v2"; sleep 5
echo "[$(date +%T)] handover: v2 pipeline restarted at the qa stage" | tee -a "$LOG"
PYTHONUNBUFFERED=1 python -m src.generate.pipeline --run "$RUN" --stages qa,mix,assemble,report --kf-backend illustrious --qa-rounds 1 \
  --repair-scope motion --mode threes --wan-lora-high wan2.2_i2v_animestyle_v2_high.safetensors:1.0 \
  --wan-lora-low wan2.2_i2v_animestyle_v2_low.safetensors:1.0 --wan-trigger 'An1meStyl3, AnimeStyle' 2>&1 | tee -a "$LOG"
