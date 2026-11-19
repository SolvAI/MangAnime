#!/usr/bin/env bash
# Mirror datasets and outputs to the NAS share (brief "extra paths": smb://nas.local/datasets/).
# Local NVMe stays the working copy (D-001); this is a one-way rsync, safe to re-run. Originals are never touched:
# the target is a SEPARATE folder (Komi_Cant_Communicate_poc), not the assets folder.
set -u
SRC=/home/user/komi-poc-work
DST=/mnt/nas/datasets/Komi_Cant_Communicate_poc
if ! mountpoint -q /mnt/nas/datasets; then echo "NAS not mounted at /mnt/nas/datasets - skipping"; exit 0; fi
mkdir -p "$DST"/{processed,runs,docs}
rsync -a --no-perms --no-owner --no-group --exclude 'train/*/cache*' --exclude 'panel_img' "$SRC/data/processed/" "$DST/processed/" 2>&1 | tail -2
# runs: reports + best outputs + final episodes (not the per-attempt raw clips)
rsync -a --no-perms --no-owner --no-group --prune-empty-dirs \
  --include '*/' --include 'best/***' --include '*.json' --include '*.jsonl' --include '*.md' --include 'episode.mp4' --include 'episode.ass' \
  --include 'animatic/episode.mp4' --exclude '*' "$SRC/runs/" "$DST/runs/" 2>&1 | tail -2
rsync -a --no-perms --no-owner --no-group /home/user/komi-poc/{STATUS.md,RESULTS.md,DECISIONS.md,QUESTIONS.md,README.md,IMPLEMENTATION_REPORT.md} "$DST/docs/" 2>/dev/null
echo "synced to $DST at $(date '+%F %T')"
