#!/usr/bin/env bash
# Source this file: `. scripts/env.sh`  (sets paths; never prints secrets)
export KOMI_REPO="/home/user/komi-poc"                 # symlink to the repo (path without spaces)
export KOMI_WORK="/home/user/komi-poc-work"            # ext4 work area: data/ models/ runs/ venvs/
export UV_PROJECT_ENVIRONMENT="$KOMI_WORK/venvs/main"
export PATH="$KOMI_WORK/venvs/main/bin:$KOMI_WORK/tools/ffenv/bin:$PATH"
export TMPDIR="$KOMI_WORK/tmp"                        # "/" has <10 GB free: never use /tmp for big files
export PYTHONPATH="$KOMI_REPO${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
if [ -f "$KOMI_REPO/.env" ]; then set -a; . "$KOMI_REPO/.env"; set +a; fi
