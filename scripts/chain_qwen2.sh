#!/usr/bin/env bash
# Qwen burst 2: everything else that needs the local VLM before generation starts.
#   manga captions (M1 chapter + held-out chapters) -> speaker naming -> few-shot captions -> held-out shot captions
#   -> alignment build / stats / few-shot bank -> VLM spot-check -> LoRA-grid captions
# usage: chain_qwen2.sh "<held-out chapters, e.g. 114,116,117,118>" <held-out episode>
. /home/user/komi-poc/scripts/env.sh; cd /home/user/komi-poc
export GPU_PRIO=1 GPU_WAIT_MIN=900
CH="${1:-114,116,117,118}"; EP="${2:-22}"
step() { echo; echo "=== $(date +%T) $*"; }
step "start qwen"; python -m src.common.servers start qwen
step "manga captions: chapter 1 + $CH"; python -m src.manga.caption --chapters "1,$CH"
touch $KOMI_WORK/runs/queue/done_manga_captions
step "character bank (CCIP, CPU) in the background"; (python -m src.anime.characters build > $KOMI_WORK/logs/plan/characters.log 2>&1 &)
step "speaker naming (text attribution + cluster votes)"; python -m src.anime.name_speakers
touch $KOMI_WORK/runs/queue/done_speakers
step "few-shot plan"; python -m src.align.build fewshot-plan --chapters "$CH" | tee $KOMI_WORK/logs/plan/fewshot_plan.txt
FS_CH=$(python -c "import json;print(json.load(open('$KOMI_WORK/data/processed/align/fewshot_plan.json'))['manga_caption_arg'])" 2>/dev/null)
FS_SPANS=$(python -c "import json;print(json.load(open('$KOMI_WORK/data/processed/align/fewshot_plan.json'))['shot_caption_spans'])" 2>/dev/null)
if [ -n "$FS_CH" ]; then
  step "manga captions for few-shot chapters $FS_CH"; python -m src.manga.caption --chapters "$FS_CH"
  step "shot captions for few-shot spans $FS_SPANS"; python -m src.anime.build_shots caption --spans "$FS_SPANS"
fi
step "alignment build (visual gap-fill) + silver + stats + few-shot bank"; python -m src.align.build all
touch $KOMI_WORK/runs/queue/done_fewshot
step "shot captions of the held-out episode $EP (evaluation only)"; python -m src.anime.build_shots caption --episodes "$EP"
step "VLM spot-check of 200 aligned pairs"; python -m src.align.build spotcheck
step "LoRA grid captions"; python -m src.eval.lora_grid caption
step "shots.jsonl"; python -m src.anime.build_shots merge
touch $KOMI_WORK/runs/queue/done_qwen2
step "CHAIN_QWEN2_DONE"
