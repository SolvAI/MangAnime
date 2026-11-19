#!/usr/bin/env bash
# GPU admission control for the single 24 GB card (brief: "load one large model at a time").
# usage: [GPU_PRIO=1] scripts/gpu_run.sh <need_gb> <cmd...>
#   Waits (up to GPU_WAIT_MIN minutes, default 240) until at least <need_gb> GB of VRAM is free, then runs <cmd>.
#   A short flock serialises the "check + start" step so two jobs do not grab the same free memory.
#   GPU_PRIO=1 marks a critical-path job: while it waits it holds a reservation, and ordinary waiters only start if
#   enough memory remains AFTER all older reservations are served (so small jobs cannot starve a large one).
#   Never kills anything. Exit code 75 = timed out waiting.
set -u
NEED_GB="$1"; shift
WORK="${KOMI_WORK:-/home/user/komi-poc-work}"
LOCK="$WORK/gpu.lock"
RES_DIR="$WORK/gpu_reservations"
WAIT_MIN="${GPU_WAIT_MIN:-240}"
SETTLE="${GPU_SETTLE_SEC:-25}"
PRIO="${GPU_PRIO:-0}"
need_mib=$(( NEED_GB * 1024 ))
deadline=$(( $(date +%s) + WAIT_MIN * 60 ))
mkdir -p "$RES_DIR"
MYRES="$RES_DIR/$$"
cleanup() { rm -f "$MYRES"; }
trap cleanup EXIT
if [ "$PRIO" = "1" ]; then echo "$need_mib" > "$MYRES"; fi

reserved_by_others() {   # MiB reserved by live priority waiters that queued before us (all of them if we are ordinary)
  local total=0 f pid
  for f in "$RES_DIR"/*; do
    [ -e "$f" ] || continue
    pid=$(basename "$f")
    [ "$pid" = "$$" ] && continue
    if ! kill -0 "$pid" 2>/dev/null; then rm -f "$f"; continue; fi
    if [ "$PRIO" = "1" ] && [ "$f" -nt "$MYRES" ]; then continue; fi   # younger reservation: we go first
    total=$(( total + $(cat "$f" 2>/dev/null || echo 0) ))
  done
  echo "$total"
}

exec 9>"$LOCK"
while true; do
  flock -x 9
  free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  held=$(reserved_by_others)
  if [ $(( ${free_mib:-0} - held )) -ge "$need_mib" ]; then
    rm -f "$MYRES"
    "$@" &
    pid=$!
    sleep "$SETTLE"          # let the job allocate before the next one checks
    flock -u 9
    wait "$pid"
    exit $?
  fi
  flock -u 9
  if [ "$(date +%s)" -ge "$deadline" ]; then
    echo "gpu_run: timed out waiting for ${NEED_GB} GB free (free=${free_mib} MiB, reserved=${held} MiB)" >&2
    exit 75
  fi
  sleep 20
done
