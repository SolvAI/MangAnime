#!/usr/bin/env bash
# Final chain for the held-out episode: headline A/B first, then the LoRA comparison on the GPU while API-only
# evaluations run, then the Qwen burst and the stage-by-stage evaluation. usage: chain_final.sh <run> <episode>
. /home/user/komi-poc/scripts/env.sh; cd /home/user/komi-poc
export PYTHONUNBUFFERED=1 GPU_PRIO=1 GPU_WAIT_MIN=900
RUN="${1:-ep22}"; EP="${2:-22}"; MP="$KOMI_WORK/venvs/musubi/bin/python"
Q() { grep --line-buffered -v "Warn\|warn"; }
step() { echo; echo "=== $(date +%T) $*"; }
step "waiting for runs/$RUN/report.json"
until [ -f "$KOMI_WORK/runs/$RUN/report.json" ] && [ -f "$KOMI_WORK/runs/$RUN/episode.mp4" ] && [ -f "$KOMI_WORK/runs/$RUN/animatic/episode.mp4" ]; do sleep 15; done
mkdir -p "$KOMI_WORK/runs/best/$RUN"; cp "$KOMI_WORK/runs/$RUN/episode.mp4" "$KOMI_WORK/runs/best/$RUN/episode_${RUN}_wan_base.mp4"
cp "$KOMI_WORK/runs/$RUN/animatic/episode.mp4" "$KOMI_WORK/runs/best/$RUN/episode_${RUN}_animatic.mp4"
step "A  free the GPU; real counterparts"; python -m src.common.servers stop comfy; python -m src.common.servers stop qwen
python -m src.eval.real_pairs --run "$RUN" --episode "$EP" 2>&1 | tail -1
step "A  HEADLINE: blind A/B wan vs animatic (DeepSeek judge + metrics)"; python -m src.eval.ab_judge --run "$RUN" 2>&1 | Q | tail -3
python scripts/make_results.py --run "$RUN" --episode "$EP" > "$KOMI_WORK/runs/$RUN/eval/results_block.md" 2>/dev/null
# ---------------- B: LoRA on the GPU, API-only evaluations in parallel
step "B  pull the newest LoRA checkpoints"
STEP=$($MP - <<'PY'
import json, re, subprocess, sys
best = 0
for r in ("high", "low"):
    try:
        st = json.load(open(f"/home/user/komi-poc-work/runs/train/wan22-i2v-{r}-v1/status.json"))
        s = int(st.get("step") or 0) // 100 * 100
        best = s if best == 0 else min(best, s)
    except Exception:
        pass
print(best)
PY
)
for S in $STEP $((STEP-100)); do
  ok=1
  for r in high low; do $MP -m src.train.wan_lora pull --run wan22-i2v-$r-v1 --step $S 2>&1 | Q | tail -1 | grep -q "LoRA ->" || ok=0; done
  [ $ok = 1 ] && { STEP=$S; break; }
done
H="wan22-i2v-high-v1-step$(printf %08d $STEP).safetensors"; L="wan22-i2v-low-v1-step$(printf %08d $STEP).safetensors"
echo "LoRA step $STEP: $H / $L"; echo "$STEP" > "$KOMI_WORK/runs/lora_step.txt"
( step "B' API-only evaluations (parallel)"; python -m src.eval.ab_judge --run "$RUN" --versus real --no-metrics 2>&1 | Q | tail -2
  python -m src.eval.ab_judge --run "$RUN" --frames 2>&1 | Q | tail -2
  python -m src.eval.ab_judge --run "$RUN" --segments 2>&1 | tail -1 ) > "$KOMI_WORK/logs/chain_final_api.log" 2>&1 &
if [ -f "$KOMI_WORK/models/comfy/loras/$H" ] && [ -f "$KOMI_WORK/models/comfy/loras/$L" ]; then
  step "B  captions for the LoRA grid (Qwen, 16 clips)"; python -m src.common.servers only qwen
  python -m src.eval.lora_grid caption 2>&1 | Q | tail -1; python -m src.common.servers stop qwen
  step "B  LoRA grid on real held-out first frames: base vs LoRA (16 fps, 81 frames)"
  python -m src.eval.lora_grid run --variant base 2>&1 | Q | tail -1
  python -m src.eval.lora_grid run --variant "lora$STEP" --high "$H" --low "$L" 2>&1 | Q | tail -1
  step "B  re-render every 12th shot of $RUN at Wan's native 16 fps: base model, then base + LoRA"
  python scripts/make_variant_run.py --src "$RUN" --dst "${RUN}_smooth" --every 12 --offset 4 --only-wan
  python scripts/make_variant_run.py --src "$RUN" --dst "${RUN}_lora" --every 12 --offset 4 --only-wan
  python -m src.generate.pipeline --run "${RUN}_lora" --stages video --mode smooth --wan-lora-high "$H" --wan-lora-low "$L" 2>&1 | Q | tail -3
  python -m src.generate.pipeline --run "${RUN}_smooth" --stages video --mode smooth 2>&1 | Q | tail -3
  for d in "$KOMI_WORK/runs/${RUN}_lora/shots"/*/; do sid=$(basename "$d"); f="$KOMI_WORK/runs/${RUN}_smooth/shots/$sid/clip.mp4"; [ -f "$f" ] && cp "$f" "$d/smooth.mp4"; done
  python -m src.common.servers stop comfy
fi
wait
# ---------------- C: Qwen burst, stage metrics, LoRA A/B
step "C  Qwen burst (held-out captions, spot-check, grid captions)"; python -m src.common.servers only qwen
bash scripts/chain_qwen3.sh "$EP" 2>&1 | Q | tail -12
step "C  second, independent judge (local Qwen) on the headline pairs"
KOMI_FORCE_LOCAL=1 python -m src.eval.ab_judge --run "$RUN" 2>&1 | Q | tail -2
python -m src.common.servers stop qwen
step "C  A/B on the re-rendered shots + grid report"
if [ -d "$KOMI_WORK/runs/${RUN}_lora/shots" ]; then
  # in these experiments label "wan" = the variant run's clip; "base" = the clip of the main cut (base model, on threes)
  python -m src.eval.ab_judge --run "${RUN}_lora" --versus smooth 2>&1 | Q | tail -2      # LoRA 16 fps vs base 16 fps (isolates the LoRA)
  python -m src.eval.ab_judge --run "${RUN}_lora" --versus base 2>&1 | Q | tail -2        # LoRA 16 fps vs main cut
  python -m src.eval.ab_judge --run "${RUN}_smooth" --versus base 2>&1 | Q | tail -2      # base 16 fps vs main cut (isolates the cadence)
  python -m src.eval.ab_judge --run "${RUN}_lora" --versus animatic 2>&1 | Q | tail -2
fi
python -m src.eval.lora_grid report 2>&1 | Q | tail -6
step "C  stage-by-stage evaluation vs the real episode"; EVAL_MAX_N=90 python -m src.eval.episode_eval --run "$RUN" --episode "$EP" 2>&1 | Q | tail -4
python scripts/make_results.py --run "$RUN" --episode "$EP" > "$KOMI_WORK/runs/$RUN/eval/results_block.md"
step "CHAIN_FINAL_DONE"
