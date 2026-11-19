#!/usr/bin/env bash
# Evaluation of the v2 episode (Illustrious keyframes + Wan anime LoRA, D-038) and the paused remainder of v1.
# usage: chain_final_v2.sh [run=ep22_v2] [v1=ep22] [episode=22]
. /home/user/komi-poc/scripts/env.sh; cd /home/user/komi-poc
export PYTHONUNBUFFERED=1 GPU_PRIO=1 GPU_WAIT_MIN=900 KOMI_KF_BACKEND=illustrious
RUN="${1:-ep22_v2}"; V1="${2:-ep22}"; EP="${3:-22}"; R="$KOMI_WORK/runs"
Q() { grep --line-buffered -v "Warn\|warn"; }
step() { echo; echo "=== $(date +%T) $*"; }
step "waiting for $RUN"
until [ -f "$R/$RUN/report.json" ] && [ -f "$R/$RUN/episode.mp4" ] && [ -f "$R/$RUN/animatic/episode.mp4" ]; do sleep 15; done
mkdir -p "$R/best/$RUN"; cp "$R/$RUN/episode.mp4" "$R/best/$RUN/episode_${RUN}_illustrious_wan_animelora.mp4"
cp "$R/$RUN/animatic/episode.mp4" "$R/best/$RUN/episode_${RUN}_animatic.mp4"
step "A  free the GPU; real counterparts; v1 clips next to the v2 clips"
python -m src.common.servers stop comfy; python -m src.common.servers stop qwen
python -m src.eval.real_pairs --run "$RUN" --episode "$EP" 2>&1 | tail -1
for d in "$R/$RUN/shots"/*/; do sid=$(basename "$d"); [ -f "$R/$V1/shots/$sid/clip.mp4" ] && cp "$R/$V1/shots/$sid/clip.mp4" "$d/$V1.mp4"; done
step "A  HEADLINE v2: blind A/B wan vs animatic (DeepSeek judge + metrics), then frames-only control"
python -m src.eval.ab_judge --run "$RUN" 2>&1 | Q | grep "judged"
python -m src.eval.ab_judge --run "$RUN" --no-metrics --tag frames-only 2>&1 | Q | grep "judged"
step "A  v2 vs v1, same shots (clips, then keyframes)"
python -m src.eval.ab_judge --run "$RUN" --versus "$V1" 2>&1 | Q | grep "judged"
python -m src.eval.ab_judge --run "$RUN" --versus "$V1" --no-metrics --tag frames-only 2>&1 | Q | grep "judged"
python -m src.eval.ab_judge --run "$RUN" --frames --other-run "$V1" 2>&1 | Q | grep "judged"
step "A  v2 vs the real episode (clips, keyframes); segment manifests for the human page"
python -m src.eval.ab_judge --run "$RUN" --versus real --no-metrics 2>&1 | Q | grep "judged"
python -m src.eval.ab_judge --run "$RUN" --frames 2>&1 | Q | grep "judged"
python -m src.eval.ab_judge --run "$RUN" --segments 2>&1 | tail -1
python -m src.eval.ab_judge --run "$RUN" --segments --other-run "$V1" 2>&1 | tail -1
python scripts/make_results.py --run "$RUN" --episode "$EP" > "$R/$RUN/eval/results_block.md" 2>/dev/null
step "C  Qwen burst (held-out captions, spot-check) + second, local judge"
python -m src.common.servers only qwen
bash scripts/chain_qwen3.sh "$EP" 2>&1 | Q | tail -10
KOMI_FORCE_LOCAL=1 python -m src.eval.ab_judge --run "$RUN" 2>&1 | Q | grep "judged"
KOMI_FORCE_LOCAL=1 python -m src.eval.ab_judge --run "$RUN" --versus "$V1" 2>&1 | Q | grep "judged"
python -m src.common.servers stop qwen
step "C  paused v1 items: A/B of the 16 fps / own-LoRA re-renders, LoRA grid report"
if [ -d "$R/${V1}_lora/shots" ]; then
  KOMI_KF_BACKEND=qwen_edit python -m src.eval.ab_judge --run "${V1}_lora" --versus smooth 2>&1 | Q | grep "judged"
  KOMI_KF_BACKEND=qwen_edit python -m src.eval.ab_judge --run "${V1}_lora" --versus base 2>&1 | Q | grep "judged"
  KOMI_KF_BACKEND=qwen_edit python -m src.eval.ab_judge --run "${V1}_smooth" --versus base 2>&1 | Q | grep "judged"
fi
python -m src.eval.lora_grid report 2>&1 | Q | tail -4
step "C  stage-by-stage evaluation vs the real episode: v2, then v1"
EVAL_MAX_N=90 python -m src.eval.episode_eval --run "$RUN" --episode "$EP" 2>&1 | Q | tail -2
EVAL_MAX_N=90 KOMI_KF_BACKEND=qwen_edit python -m src.eval.episode_eval --run "$V1" --episode "$EP" 2>&1 | Q | tail -2
python scripts/make_results.py --run "$RUN" --episode "$EP" > "$R/$RUN/eval/results_block.md"
python scripts/make_results.py --run "$V1" --episode "$EP" > "$R/$V1/eval/results_block.md"
step "CHAIN_FINAL_V2_DONE"
