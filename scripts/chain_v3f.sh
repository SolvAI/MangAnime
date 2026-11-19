#!/usr/bin/env bash
# ep22_v3f (D-062, D-063): ep22_v3e's script, voices and edited shot list; wardrobe redone with the uniforms as worn in the
# show, camera pass, first frames redrawn with set references, then continuity repair, reframing, clips (16 fps, profile P),
# mix, assembly, all-ages screen, evaluation, blind pair sheets. The shot list must already carry the wardrobe and the
# camera pass (python -m src.v3.wardrobe --run ep22_v3f; python -m src.v3.staging direct --run ep22_v3f).
#   tmux new-session -d -s v3f_chain "bash scripts/chain_v3f.sh > $KOMI_WORK/logs/v3f_chain.log 2>&1"
. /home/user/komi-poc/scripts/env.sh; cd /home/user/komi-poc
export PYTHONUNBUFFERED=1 KOMI_STYLE_REFS=2 KOMI_SET_REFS=2 KOMI_WAN_PROFILE=P
RUN="${1:-ep22_v3f}"; R=/home/user/komi-poc-work/runs
step() { echo; echo "=== $(date +%T) $*"; }
Q() { grep --line-buffered -v "Warn\|warn"; }
step "first frames (set references, camera position), continuity repair, reframing"
python -m src.v3.pipeline --run "$RUN" --stages frames,continuity,reframe --frames gemini --mode smooth 2>&1 | Q
python -m src.v3.staging run --run "$RUN" 2>&1 | Q | tail -3
step "clips (profile P, 16 fps), mix, assembly, report"
python -m src.v3.pipeline --run "$RUN" --stages video,mix,assemble,report --frames gemini --mode smooth 2>&1 | Q
if [ ! -f "$R/$RUN/episode.mp4" ]; then echo "$RUN has no episode: stopping"; echo "V3F_CHAIN_FAILED"; exit 1; fi
step "screen + evaluation chain"
bash scripts/chain_v3_eval.sh "$RUN" 22 none 2>&1
step "writing, frame style, blind pair sheets, contact sheet"
python -m src.v3.eval_script --runs "$RUN" 2>&1 | Q | tail -6
HF_HUB_OFFLINE=1 python -m src.eval.frame_style --runs ep22_v3,ep22_v3b,ep22_v3e,"$RUN" --episode 22 2>&1 | Q | grep "frame style\|look,"
python -m src.eval.blind_set build --run "$RUN" --n 40 2>&1 | Q | tail -1
python -m src.eval.blind_set animatic --run "$RUN" --n 40 2>&1 | Q | tail -1
python scripts/contact_sheet.py --video "$R/best/$RUN/episode_${RUN}_script_first_customvoice.mp4" --out "$R/$RUN/eval/episode_contact.jpg" 2>&1 | tail -1
step "V3F_EVALUATED"
