#!/usr/bin/env bash
# Blind A/B rating page for a finished run. usage: scripts/ab_page.sh [run] [animatic|segments|real|keyframes] [port]
#   animatic  = shot-level Wan clip vs pan/zoom animatic (silent)          <- the headline comparison
#   segments  = ~20 s segments of both full cuts, with audio
#   real      = generated shot vs the real shot that adapted the same panel
# Ratings go to runs/ab/results.duckdb next to the automatic raters; open /results only after rating (it un-blinds).
. /home/user/komi-poc/scripts/env.sh; cd /home/user/komi-poc
RUN="${1:-ep22}"; WHAT="${2:-animatic}"; PORT="${3:-8300}"
M="$KOMI_WORK/runs/$RUN/eval/ab_manifest_$WHAT.json"
[ -f "$M" ] || { echo "no manifest $M (run scripts/chain_eval.sh $RUN first)"; exit 1; }
echo "open http://127.0.0.1:$PORT/   (manifest: $M)"
exec python -m tools.ab_page.app --manifest "$M" --db "$KOMI_WORK/runs/ab/results.duckdb" --port "$PORT"
