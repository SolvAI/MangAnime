#!/usr/bin/env bash
# Evaluation of v2b (Illustrious keyframes + base Wan, D-041): vs its animatic, vs the LoRA cut (ep22_v2), stage metrics.
. /home/user/komi-poc/scripts/env.sh; cd /home/user/komi-poc
export PYTHONUNBUFFERED=1 GPU_PRIO=1 GPU_WAIT_MIN=900 KOMI_KF_BACKEND=illustrious
RUN="ep22_v2b"; EP=22; R="$KOMI_WORK/runs"
Q() { grep --line-buffered -v "Warn\|warn"; }
step() { echo; echo "=== $(date +%T) $*"; }
step "waiting for $RUN"
until [ -f "$R/$RUN/report.json" ] && [ -f "$R/$RUN/episode.mp4" ] && [ -f "$R/$RUN/animatic/episode.mp4" ]; do sleep 15; done
mkdir -p "$R/best/$RUN"; cp "$R/$RUN/episode.mp4" "$R/best/$RUN/episode_${RUN}_illustrious_wan_base.mp4"
cp "$R/$RUN/animatic/episode.mp4" "$R/best/$RUN/episode_${RUN}_animatic.mp4"
python -m src.common.servers stop comfy; python -m src.common.servers stop qwen
step "HEADLINE v2b: blind A/B wan vs animatic (judge + metrics), frames-only control"
python -m src.eval.real_pairs --run "$RUN" --episode "$EP" 2>&1 | tail -1
python -m src.eval.ab_judge --run "$RUN" 2>&1 | Q | grep "judged"
python -m src.eval.ab_judge --run "$RUN" --no-metrics --tag frames-only 2>&1 | Q | grep "judged"
step "v2b vs the LoRA cut (ep22_v2), same shots"
python -m src.eval.ab_judge --run "$RUN" --versus ep22_v2 2>&1 | Q | grep "judged"
python -m src.eval.ab_judge --run "$RUN" --versus ep22_v2 --no-metrics --tag frames-only 2>&1 | Q | grep "judged"
python -m src.eval.ab_judge --run "$RUN" --segments 2>&1 | tail -1
step "second, local judge: v2b headline, then the v1 headline that was skipped earlier"
python -m src.common.servers only qwen
KOMI_FORCE_LOCAL=1 python -m src.eval.ab_judge --run "$RUN" 2>&1 | Q | grep "judged"
KOMI_FORCE_LOCAL=1 KOMI_KF_BACKEND=qwen_edit python -m src.eval.ab_judge --run ep22 2>&1 | Q | grep "judged"
python -m src.common.servers stop qwen
step "stage-by-stage evaluation vs the real episode"
EVAL_MAX_N=90 python -m src.eval.episode_eval --run "$RUN" --episode "$EP" 2>&1 | Q | tail -2
python scripts/make_results.py --run "$RUN" --episode "$EP" > "$R/$RUN/eval/results_block.md"
step "CHAIN_FINAL_V2B_DONE"
