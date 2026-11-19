#!/usr/bin/env bash
# GPU hand-over chain M: manga parsing on GPU (held-out volumes first), OCR on GPU, panel build, dialogue alignment.
# Start it while the card is still busy: every GPU step waits in gpu_run.sh with priority.
. /home/user/komi-poc/scripts/env.sh; cd /home/user/komi-poc
export GPU_PRIO=1 GPU_WAIT_MIN=900
L=$KOMI_WORK/logs/plan; mkdir -p $L
magi() { scripts/gpu_run.sh 5 bash -c "tmux kill-session -t magi 2>/dev/null; cd $KOMI_WORK && venvs/manga/bin/python /home/user/komi-poc/src/manga/magi_detect.py --pages-jsonl data/interim/manga/pages.jsonl --out-dir data/interim/manga/magi --volumes $1 --batch 2" 2>&1 | grep -v "Warning\|warn\|GenerationMixin\|owner of the model"; }
echo "=== $(date +%T) magi 9,10"; magi 9,10
echo "=== $(date +%T) OCR server on GPU"
tmux kill-session -t build1 2>/dev/null; tmux kill-session -t srv-ocr-cpu 2>/dev/null; sleep 2
python -m src.common.servers start ocr
echo "=== $(date +%T) build 9,10 (+ whatever of volume 1 is parsed)"; python -m src.manga.build --volumes 9,10,1
echo "=== $(date +%T) text alignment (first pass: held-out chapters visible)"; python -m src.align.text_align 2>&1 | grep -v "Warn\|warn"
touch $KOMI_WORK/runs/queue/done_manga_heldout
echo "=== $(date +%T) magi 1-8"; magi 1,2,3,4,5,6,7,8
echo "=== $(date +%T) build 1-10"; python -m src.manga.build --volumes 1,2,3,4,5,6,7,8,9,10
echo "=== $(date +%T) text alignment (all)"; python -m src.align.text_align 2>&1 | grep -v "Warn\|warn"
touch $KOMI_WORK/runs/queue/done_manga_all
echo "=== $(date +%T) CHAIN_MANGA_DONE"
