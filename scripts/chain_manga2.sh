#!/usr/bin/env bash
# Manga chain, part 2: OCR/build for the volumes already parsed by Magi (5-8 plus 9,10,1), alignment; then the rest
# of the training volumes with Magi on CPU (the GPU is reserved for the training-cache / captioning critical path).
. /home/user/komi-poc/scripts/env.sh; cd /home/user/komi-poc
echo "=== $(date +%T) build 8,7,6,5 (OCR on the GPU server)"; python -m src.manga.build --volumes 8,7,6,5
echo "=== $(date +%T) text alignment"; python -m src.align.text_align 2>&1 | grep -v "Warn\|warn"
touch $KOMI_WORK/runs/queue/done_manga_s2
echo "=== $(date +%T) Magi on CPU for volumes 5,4,3,2"
cd $KOMI_WORK
for v in 5 4 3 2; do
  CUDA_VISIBLE_DEVICES="" venvs/manga/bin/python /home/user/komi-poc/src/manga/magi_detect.py --pages-jsonl data/interim/manga/pages.jsonl \
    --out-dir data/interim/manga/magi --cpu --threads 14 --batch 2 --volumes $v 2>&1 | grep -v "Warning\|warn\|GenerationMixin\|owner of the model"
done
cd /home/user/komi-poc
echo "=== $(date +%T) build 1-10"; python -m src.manga.build --volumes 1,2,3,4,5,6,7,8,9,10
echo "=== $(date +%T) text alignment (all)"; python -m src.align.text_align 2>&1 | grep -v "Warn\|warn"
touch $KOMI_WORK/runs/queue/done_manga_all
echo "=== $(date +%T) CHAIN_MANGA2_DONE"
