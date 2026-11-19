#!/usr/bin/env bash
# Evaluation of the v3 episode (script-first pipeline, D-043): animatic of the same cut, blind A/B with the Gemini judge,
# stage-by-stage metrics against the real episode. Waits for the render session to finish first.
# usage: chain_v3_eval.sh [run=ep22_v3] [episode=22] [render tmux session=v3_render]
. /home/user/komi-poc/scripts/env.sh; cd /home/user/komi-poc
export PYTHONUNBUFFERED=1 GPU_PRIO=1 GPU_WAIT_MIN=900 KOMI_KF_BACKEND=illustrious KOMI_AB_JUDGE=gemini
RUN="${1:-ep22_v3}"; EP="${2:-22}"; SESSION="${3:-v3_render}"; R="$KOMI_WORK/runs"
Q() { grep --line-buffered -v "Warn\|warn"; }
step() { echo; echo "=== $(date +%T) $*"; }
step "waiting for $RUN (render session $SESSION)"
while tmux has-session -t "=$SESSION" 2>/dev/null && ! { [ -f "$R/$RUN/report.json" ] && [ -f "$R/$RUN/episode.mp4" ]; }; do sleep 20; done
if ! { [ -f "$R/$RUN/report.json" ] && [ -f "$R/$RUN/episode.mp4" ]; }; then echo "render ended without an episode: stopping"; echo "CHAIN_V3_EVAL_FAILED"; exit 1; fi
step "0  all-ages screen of every clip (D-054); flagged clips become held frames and the episode is re-assembled"
python -m src.v3.screen --run "$RUN" --apply 2>&1 | Q | tail -8
if python scripts/screen_needs_reassembly.py "$RUN"; then
  python -m src.v3.pipeline --run "$RUN" --stages assemble,report 2>&1 | Q | tail -3
fi
mkdir -p "$R/best/$RUN"
cp "$R/$RUN/episode.mp4" "$R/best/$RUN/episode_${RUN}_script_first_customvoice.mp4"
[ -f "$R/$RUN/episode_clonevoice.mp4" ] && cp "$R/$RUN/episode_clonevoice.mp4" "$R/best/$RUN/episode_${RUN}_script_first_clonevoice.mp4"
if [ -f "$R/$RUN/DEFER_EVAL" ]; then step "evaluation deferred (DEFER_EVAL present): screened and copied only"; echo "CHAIN_V3_EVAL_DEFERRED"; exit 0; fi
step "A  animatic of the same cut (same first frames, same audio)"
python -m src.common.servers stop comfy
python - "$RUN" <<'EOF' 2>&1 | Q | tail -2
import sys
from src.common.config import P
from src.common.io import read_json
from src.generate.assemble import assemble
rd = P("runs") / sys.argv[1]
assemble(rd, read_json(rd / "shotlist.json")["shots"], variant="animatic")
EOF
cp "$R/$RUN/animatic/episode.mp4" "$R/best/$RUN/episode_${RUN}_animatic.mp4"
step "A  frame sheets (panel | v2 frame | v3 prompt | v3 frame)"
python scripts/make_v3_mosaic.py --run "$RUN" 2>&1 | tail -1
step "B  blind A/B wan vs animatic (Gemini judge + local metrics), then frames-only control"
python -m src.eval.ab_judge --run "$RUN" 2>&1 | Q | grep "judged"
python -m src.eval.ab_judge --run "$RUN" --no-metrics --tag frames-only 2>&1 | Q | grep "judged"
step "B  v3 vs the real episode (all aligned clips and first frames); segment manifest for the human page"
python -m src.eval.real_pairs --run "$RUN" --episode "$EP" 2>&1 | tail -1
python -m src.eval.ab_judge --run "$RUN" --versus real --no-metrics 2>&1 | Q | grep "judged"
python -m src.eval.ab_judge --run "$RUN" --frames 2>&1 | Q | grep "judged"
python -m src.eval.ab_judge --run "$RUN" --segments 2>&1 | tail -1
step "B  continuity between shots (finished clips)"
python -m src.eval.continuity --run "$RUN" 2>&1 | tail -1
step "C  voice metrics of the lines in the cut (speaker similarity, word error rate), both takes"
python -m src.v3.voice metrics --run "$RUN" 2>&1 | Q | tail -2 || true
step "C  stage-by-stage evaluation vs the real episode"
EVAL_MAX_N=90 python -m src.eval.episode_eval --run "$RUN" --episode "$EP" 2>&1 | Q | tail -2
python scripts/make_results.py --run "$RUN" --episode "$EP" > "$R/$RUN/eval/results_block.md" 2>"$R/$RUN/eval/results_block.err" || echo "results block: see results_block.err"
step "CHAIN_V3_EVAL_DONE"
