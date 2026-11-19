#!/usr/bin/env bash
# Standing GPU reservation for a multi-stage run (the card is free for minutes between stages).
# usage: gpu_guard.sh [mib=22000] [max_minutes=420]     stop: touch $KOMI_WORK/gpu_guard.stop (or kill this script)
# It writes a priority reservation under its own PID in gpu_reservations/ (see gpu_run.sh) and keeps touching it:
# ordinary waiters always count it; this project's own priority jobs skip it because it is always "younger" than theirs.
WORK="${KOMI_WORK:-/home/user/komi-poc-work}"; RES="$WORK/gpu_reservations"; mkdir -p "$RES"
MIB="${1:-22000}"; MAXMIN="${2:-420}"; F="$RES/$$"; STOP="$WORK/gpu_guard.stop"; rm -f "$STOP"
trap 'rm -f "$F"' EXIT
end=$(( $(date +%s) + MAXMIN * 60 ))
echo "$MIB" > "$F"; echo "[$(date +%T)] gpu guard: $MIB MiB reserved (pid $$) for up to $MAXMIN min"
while [ ! -e "$STOP" ] && [ "$(date +%s)" -lt "$end" ]; do sleep 4; echo "$MIB" > "$F"; done
echo "[$(date +%T)] gpu guard released"
