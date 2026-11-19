#!/usr/bin/env bash
# Redraw the chapter title cards of a finished v3 run (a long title used to run off both edges of the card), make their
# clips again and re-assemble the episode. Nothing else is rendered: the video stage returns once every clip exists.
#   scripts/fix_title_cards.sh ep22_v3e
set -uo pipefail
RUN="$1"
cd "$(dirname "$0")/.." && . scripts/env.sh >/dev/null 2>&1
R=/home/user/komi-poc-work/runs
Q() { grep -v "^$" | cut -c1-240; }
python - "$RUN" <<'E'
import sys
from src.common.config import P
from src.common.io import read_json
from src.generate.keyframes import title_card
rd = P("runs") / sys.argv[1]
n = 0
for s in read_json(rd / "shotlist.json")["shots"]:
    if s["start"] == "title":
        sd = rd / "shots" / s["shot_id"]
        title_card(s["notes"].split(":", 1)[-1].strip() or "…", sd / "kf.png")
        (sd / "clip.mp4").unlink(missing_ok=True)
        n += 1
print(f"{sys.argv[1]}: {n} title cards redrawn")
E
python -m src.v3.pipeline --run "$RUN" --stages video,assemble,report --frames gemini --mode smooth 2>&1 | Q | tail -4
mkdir -p "$R/best/$RUN"
cp "$R/$RUN/episode.mp4" "$R/best/$RUN/episode_${RUN}_script_first_customvoice.mp4"
[ -f "$R/$RUN/episode_clonevoice.mp4" ] && cp "$R/$RUN/episode_clonevoice.mp4" "$R/best/$RUN/episode_${RUN}_script_first_clonevoice.mp4"
echo "=== $(date +%H:%M:%S) TITLE_CARDS_DONE $RUN"
