#!/usr/bin/env bash
# Qwen burst 2b (after the M1 slice): few-shot captions, alignment build, held-out shot captions, spot-check, grid captions.
# usage: chain_qwen2b.sh "<held-out chapters>" <held-out episode>
. /home/user/komi-poc/scripts/env.sh; cd /home/user/komi-poc
export GPU_PRIO=1 GPU_WAIT_MIN=900 PYTHONUNBUFFERED=1
CH="${1:-114,116,117,118}"; EP="${2:-22}"
step() { echo; echo "=== $(date +%T) $*"; }
step "start qwen"; python -m src.common.servers only qwen
step "few-shot plan"; python -m src.align.build fewshot-plan --chapters "$CH"
FS_CH=$(python -c "import json;print(json.load(open('$KOMI_WORK/data/processed/align/fewshot_plan.json'))['manga_caption_arg'])" 2>/dev/null)
FS_SPANS=$(python -c "import json;print(json.load(open('$KOMI_WORK/data/processed/align/fewshot_plan.json'))['shot_caption_spans'])" 2>/dev/null)
if [ -n "$FS_SPANS" ]; then
  step "shot captions for few-shot spans $FS_SPANS (first 48 shots each)"; python -m src.anime.build_shots caption --spans "$FS_SPANS" --max-per-span 48
fi
step "character bank (CCIP, CPU) in the background"; (python -m src.anime.characters build > $KOMI_WORK/logs/plan/characters.log 2>&1 &)
step "alignment build (visual gap-fill) + silver + stats + few-shot bank"; python -m src.align.build all
touch $KOMI_WORK/runs/queue/done_fewshot
step "shots.jsonl"; python -m src.anime.build_shots merge
touch $KOMI_WORK/runs/queue/done_qwen2
step "CHAIN_QWEN2B_DONE"
