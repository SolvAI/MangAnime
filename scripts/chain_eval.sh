#!/usr/bin/env bash
# Evaluation chain for a finished run (brief §10). usage: chain_eval.sh <run> <held-out episode> [wait]
# Order: metric-heavy A/B first (GPU free), then the Qwen burst (held-out captions, spot-check, calibration), then the
# episode evaluation that reads all of it.
. /home/user/komi-poc/scripts/env.sh; cd /home/user/komi-poc
export PYTHONUNBUFFERED=1 GPU_PRIO=1 GPU_WAIT_MIN=900
RUN="${1:-ep22}"; EP="${2:-22}"
step() { echo; echo "=== $(date +%T) $*"; }
if [ "$3" = "wait" ]; then
  step "waiting for runs/$RUN/report.json"
  until [ -f "$KOMI_WORK/runs/$RUN/report.json" ] && [ -f "$KOMI_WORK/runs/$RUN/episode.mp4" ]; do sleep 20; done
fi
step "free the GPU"; python -m src.common.servers stop comfy; python -m src.common.servers stop qwen
step "real counterparts"; python -m src.eval.real_pairs --run "$RUN" --episode "$EP" 2>&1 | tail -2
step "A/B wan vs animatic (DeepSeek judge + local metrics)"; python -m src.eval.ab_judge --run "$RUN" 2>&1 | grep -v "Warn\|warn" | tail -4
step "A/B wan vs real episode"; python -m src.eval.ab_judge --run "$RUN" --versus real 2>&1 | grep -v "Warn\|warn" | tail -4
step "A/B keyframes vs real frames"; python -m src.eval.ab_judge --run "$RUN" --frames 2>&1 | grep -v "Warn\|warn" | tail -4
step "segment manifest for the human page"; python -m src.eval.ab_judge --run "$RUN" --segments 2>&1 | tail -1
if [ ! -f "$KOMI_WORK/runs/queue/done_qwen3_$EP" ]; then
  step "Qwen burst (held-out captions, spot-check, grid captions)"; python -m src.common.servers only qwen
  bash scripts/chain_qwen3.sh "$EP" 2>&1 | grep -v "Warn\|warn" | tail -25 && touch "$KOMI_WORK/runs/queue/done_qwen3_$EP"
  step "second, independent judge (local Qwen) on the same blind pairs"
  KOMI_FORCE_LOCAL=1 python -m src.eval.ab_judge --run "$RUN" 2>&1 | grep -v "Warn\|warn" | tail -2
  KOMI_FORCE_LOCAL=1 python -m src.eval.ab_judge --run "$RUN" --versus real 2>&1 | grep -v "Warn\|warn" | tail -2
  KOMI_FORCE_LOCAL=1 python -m src.eval.ab_judge --run "$RUN" --frames 2>&1 | grep -v "Warn\|warn" | tail -2
  step "validation calibration set, if not judged yet"
  [ -f "$KOMI_WORK/runs/judge_calibration_val/report.json" ] || CALIB_SET=val python -m src.qa.calibrate run --phase judge --keep-servers 2>&1 | grep -v "Warn\|warn" | tail -3
  python -m src.common.servers stop qwen
fi
step "episode evaluation"; python -m src.eval.episode_eval --run "$RUN" --episode "$EP" 2>&1 | grep -v "Warn\|warn" | tail -6
step "CHAIN_EVAL_DONE"
