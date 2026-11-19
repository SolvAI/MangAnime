#!/usr/bin/env bash
# Restart the pipeline at the QA stage so it loads the re-calibrated judge (src/qa/judge.py, metric reference).
. /home/user/komi-poc/scripts/env.sh; cd /home/user/komi-poc
RUN="${1:-ep22}"; LOG="$KOMI_WORK/logs/$RUN.log"
until grep -q "=== stage: qa" "$LOG"; do sleep 5; done
tmux kill-session -t "=$RUN"; sleep 5
echo "[$(date +%T)] handover: pipeline restarted at the qa stage" | tee -a "$LOG"
PYTHONUNBUFFERED=1 python -m src.generate.pipeline --run "$RUN" --stages qa,mix,assemble,report --qa-rounds ${QA_ROUNDS:-1} --mode ${MODE:-threes} 2>&1 | tee -a "$LOG"
