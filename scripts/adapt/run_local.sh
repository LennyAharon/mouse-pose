#!/bin/bash
# Run an adaptation grid on this machine: a queue over the job file, at most MAX_GPU GPU processes at once (all jobs on
# the GPU count, also other queues'), each cell via scripts/adapt/run_cell.py. Idempotent and restart-safe: cells with
# `.done` are skipped by run_cell; launched cells are marked in <queue dir>/started_<id>; re-running this script after a
# crash resumes. Durable launch (survives session restarts):
#
#   cd mouse-pose && python scripts/adapt/plan.py --config <cfg> --jobs <results>/adaptation/<name>/_queue/jobs.txt
#   setsid nohup bash scripts/adapt/run_local.sh <cfg> <jobs.txt> [MAX_GPU=2] > <queue dir>/run_local.out 2>&1 < /dev/null & disown
#
# Hold new launches: touch <queue dir>/STOP (running cells finish). Log: <queue dir>/queue.log; per cell
# <queue dir>/<cell id>.out (run_cell output) next to the training log written beside the cell directory.
set -u
CFG="$1"; JOBS="$2"; MAX_GPU="${3:-2}"
cd "$(dirname "$0")/../.." || exit 1
Q=$(dirname "$JOBS")
log() { echo "[$(date -u '+%F %T')] $*" >> "$Q/queue.log"; }
gpu_jobs() { nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .; }
if [ -f "$Q/run_local.pid" ]; then
  old=$(cat "$Q/run_local.pid")
  if kill -0 "$old" 2>/dev/null && grep -a -q run_local.sh /proc/$old/cmdline 2>/dev/null; then
    echo "run_local.sh already running for $Q (pid $old)" >&2; exit 1
  fi
fi
echo $$ > "$Q/run_local.pid"
log "=== start (pid $$; config $CFG; $(wc -l < "$JOBS") cells; max $MAX_GPU GPU jobs; mouse-pose $(git rev-parse --short HEAD))"
mine=0
while read -r CELL; do
  [ -z "$CELL" ] && continue
  [ -f "$Q/started_$CELL" ] && continue
  while true; do
    [ -f "$Q/STOP" ] && { log "STOP present: holding"; while [ -f "$Q/STOP" ]; do sleep 60; done; log "STOP removed: resuming"; }
    # our launches take ~1 min to reach the GPU: count them too until they show up there
    mine=$(jobs -rp | wc -l)
    n=$(gpu_jobs); [ "$mine" -gt "$n" ] && n=$mine
    [ "$n" -lt "$MAX_GPU" ] && break
    sleep 30
  done
  touch "$Q/started_$CELL"; log "start $CELL"
  ( python scripts/adapt/run_cell.py --config "$CFG" --cell "$CELL" > "$Q/$CELL.out" 2>&1
    rc=$?; log "end $CELL (exit $rc): $(tail -1 "$Q/$CELL.out" | cut -c1-160)" ) &
  sleep 90
done < "$JOBS"
wait
log "=== all done"
