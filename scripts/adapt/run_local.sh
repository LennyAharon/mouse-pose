#!/bin/bash
# Run an adaptation grid on this machine: at most MAX_GPU GPU processes at once (every process on the
# GPU counts, also other queues'), each cell via scripts/adapt/run_cell.py. Before every launch the
# queue re-plans (scripts/adapt/plan.py) and takes the first runnable, not yet started cell in grid
# order, so cells blocked on a trunk join in their place as soon as that trunk is COMPLETED. With
# ONLY (a file of cell ids, e.g. a canary) it runs just those. Restart-safe: launched cells are marked
# <queue dir>/started_<id> and never relaunched (a failed cell is retried by deleting its marker);
# done cells have `.done`. Ends when nothing is left, waiting (10-min polls) while cells are blocked.
#
#   cd mouse-pose && setsid nohup bash scripts/adapt/run_local.sh <cfg> [MAX_GPU=2] [ONLY=<ids file>] \
#       > /dev/null 2>&1 < /dev/null & disown
#
# Queue dir: <results_dir>/<out_subdir>/<name>/_queue. Hold new launches: touch <queue dir>/STOP
# (running cells finish). Change the limit while it runs: echo 3 > <queue dir>/max_gpu (read before
# every launch; running cells are never stopped). Log: <queue dir>/queue.log; per cell <queue dir>/<cell id>.out (run_cell
# output) next to the training log written beside the cell directory.
set -u
CFG="$1"; MAX_GPU="${2:-2}"; ONLY="${3:-}"
cd "$(dirname "$0")/../.." || exit 1
Q=$(python - "$CFG" <<'EOF'
import sys
from pathlib import Path
from mighty_mouse.adaptation import load_grid
from mighty_mouse.paths import load_paths
g = load_grid(sys.argv[1])
print(Path(load_paths()["results_dir"]) / g.out_subdir / g.name / "_queue")
EOF
) || exit 1
mkdir -p "$Q"
log() { echo "[$(date -u '+%F %T')] $*" >> "$Q/queue.log"; }
gpu_jobs() { nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .; }
if [ -f "$Q/run_local.pid" ]; then
  old=$(cat "$Q/run_local.pid")
  if kill -0 "$old" 2>/dev/null && grep -a -q run_local.sh /proc/$old/cmdline 2>/dev/null; then
    echo "run_local.sh already running for $Q (pid $old)" >&2; exit 1
  fi
fi
echo $$ > "$Q/run_local.pid"
log "=== start (pid $$; config $CFG; ${ONLY:+only $(wc -l < "$ONLY") cells from $ONLY; }max $MAX_GPU GPU jobs; mouse-pose $(git rev-parse --short HEAD))"
# first runnable, not started cell in grid order (restricted to $ONLY); plan summary in $Q/plan.out
next_cell() {
  python scripts/adapt/plan.py --config "$CFG" --jobs "$Q/pending.txt" > "$Q/plan.out" 2>&1 || return 1
  local c
  while read -r c; do
    [ -n "$ONLY" ] && ! grep -qx "$c" "$ONLY" && continue
    [ -f "$Q/started_$c" ] && continue
    echo "$c"; return 0
  done < "$Q/pending.txt"
  return 0
}
waiting=""
while true; do
  [ -f "$Q/STOP" ] && { log "STOP present: holding"; while [ -f "$Q/STOP" ]; do sleep 60; done; log "STOP removed: resuming"; }
  # our launches take ~1 min to reach the GPU: count them too until they show up there
  mine=$(jobs -rp | wc -l)
  n=$(gpu_jobs); [ "$mine" -gt "$n" ] && n=$mine
  max=$MAX_GPU; [ -s "$Q/max_gpu" ] && max=$(tr -dc 0-9 < "$Q/max_gpu"); max=${max:-$MAX_GPU}
  [ "$max" != "${last_max:-}" ] && { log "limit: $max GPU jobs"; last_max=$max; }
  if [ "$n" -ge "$max" ]; then sleep 30; continue; fi
  if ! CELL=$(next_cell); then log "plan.py failed: $(tail -1 "$Q/plan.out")"; sleep 300; continue; fi
  if [ -z "$CELL" ]; then
    [ "$(jobs -rp | wc -l)" -gt 0 ] && { sleep 60; continue; }
    blocked=$(grep -o "blocked [0-9]*" "$Q/plan.out" | awk '{print $2}')
    if [ -z "$ONLY" ] && [ "${blocked:-0}" -gt 0 ]; then
      [ -z "$waiting" ] && log "nothing runnable; $blocked cells wait for a trunk ($(grep "trunks not ready" "$Q/plan.out" | sed 's/^ *//'))"
      waiting=1; sleep 600; continue
    fi
    break
  fi
  waiting=""
  touch "$Q/started_$CELL"; log "start $CELL"
  ( python scripts/adapt/run_cell.py --config "$CFG" --cell "$CELL" > "$Q/$CELL.out" 2>&1
    rc=$?; log "end $CELL (exit $rc): $(tail -1 "$Q/$CELL.out" | cut -c1-160)" ) &
  sleep 90
done
wait
log "=== nothing left to launch${ONLY:+ from $ONLY}"
