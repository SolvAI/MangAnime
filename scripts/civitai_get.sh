#!/usr/bin/env bash
# Download one Civitai model version into the project's ComfyUI model folder.
# usage: civitai_get.sh <modelVersionId> <checkpoints|loras> <file name>
# The token is read from .env (CIVITAI_API_KEY), sent only to civitai.com in an Authorization header (curl drops the
# header when the download redirects to the storage host) and never printed.
set -u
set -a; . /home/user/komi-poc/.env; set +a
VID="$1"; KIND="$2"; NAME="$3"
DST="/home/user/komi-poc-work/models/comfy/$KIND"; mkdir -p "$DST"
if [ -s "$DST/$NAME" ]; then echo "exists: $KIND/$NAME"; exit 0; fi
[ -n "${CIVITAI_API_KEY:-}" ] || { echo "CIVITAI_API_KEY missing in .env"; exit 1; }
code=$(curl -sS -fL --retry 3 --retry-delay 5 -A "comfy-model-fetch/1.0" -H "Authorization: Bearer $CIVITAI_API_KEY" \
  -o "$DST/$NAME.part" -w "%{http_code}" "https://civitai.com/api/download/models/$VID") || { echo "FAILED $KIND/$NAME (http $code)"; rm -f "$DST/$NAME.part"; exit 1; }
mv "$DST/$NAME.part" "$DST/$NAME"
echo "ok: $KIND/$NAME $(du -h "$DST/$NAME" | cut -f1)"
