#!/bin/bash
# General GPU job queue for this machine: runs the commands of <queue dir>/jobs.tsv (one job per
# line: <id><TAB><command>, run with bash from the mouse-pose root) with up to PER_GPU jobs on each
# GPU of $GPUS (default: every GPU; every process on a GPU counts, also other queues'), placing each
# job on the least-loaded GPU via CUDA_VISIBLE_DEVICES (scripts/gpu_slots.sh) and one thread per CPU
# pool (OMP/MKL/OpenBLAS/OpenCV; data loading is CPU-bound and pools of concurrent jobs fight).
# Restart-safe: a launched job is marked <queue dir>/started_<id> and never relaunched (delete the
# marker to retry); exit 0 writes done_<id>. Ends when every job has been launched and has finished.
#
#   cd mouse-pose && [GPUS="2 3"] setsid nohup bash scripts/queue_runs.sh <queue dir> [PER_GPU=3] \
#       > /dev/null 2>&1 < /dev/null & disown
#
# Hold new launches: touch <queue dir>/STOP. Per-GPU limit while it runs: echo N > <queue dir>/max_gpu.
# Log: <queue dir>/queue.log; per job <queue dir>/<id>.out.
set -u
Q="$(realpath "$1")"; PER_GPU="${2:-3}"
cd "$(dirname "$0")/.." || exit 1
source scripts/gpu_slots.sh
GPUS="${GPUS:-$(all_gpus)}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}" \
       OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}" OPENCV_FOR_THREADS_NUM="${OPENCV_FOR_THREADS_NUM:-1}"
log() { echo "[$(date -u '+%F %T')] $*" >> "$Q/queue.log"; }
[ -f "$Q/jobs.tsv" ] || { echo "no $Q/jobs.tsv" >&2; exit 1; }
if [ -f "$Q/queue.pid" ]; then
  old=$(cat "$Q/queue.pid")
  if kill -0 "$old" 2>/dev/null && grep -a -q queue_runs.sh /proc/$old/cmdline 2>/dev/null; then
    echo "queue_runs.sh already running for $Q (pid $old)" >&2; exit 1
  fi
fi
echo $$ > "$Q/queue.pid"
log "=== start (pid $$; $(grep -c . "$Q/jobs.tsv") jobs; GPUs [$GPUS], max $PER_GPU per GPU; mouse-pose $(git rev-parse --short HEAD), lightning-pose $(git -C ../lightning-pose rev-parse --short HEAD 2>/dev/null))"
while IFS=$'\t' read -r ID CMD <&3; do
  [ -z "$ID" ] && continue
  [ -f "$Q/started_$ID" ] && continue
  while true; do
    [ -f "$Q/STOP" ] && { log "STOP present: holding"; while [ -f "$Q/STOP" ]; do sleep 60; done; log "STOP removed: resuming"; }
    max=$PER_GPU; [ -s "$Q/max_gpu" ] && max=$(tr -dc 0-9 < "$Q/max_gpu"); max=${max:-$PER_GPU}
    g=$(pick_gpu "$GPUS" "$max")
    [ -n "$g" ] && break
    sleep 30
  done
  touch "$Q/started_$ID"; log "start $ID (gpu $g)"
  ( CUDA_VISIBLE_DEVICES=$g bash -c "$CMD" > "$Q/$ID.out" 2>&1 < /dev/null
    rc=$?; [ $rc = 0 ] && touch "$Q/done_$ID"; log "end $ID (exit $rc)" ) &
  claim_gpu "$g" $!
  sleep 20
done 3< "$Q/jobs.tsv"
wait
log "=== all jobs launched and finished"
