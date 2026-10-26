#!/bin/bash
# Keep Delta's interactive GPU partitions busy with recipe-of-record runs (skills/delta-gpu-queue).
# Interactive partitions have priority factor 10000 (regular 100) but allow 1 job per user per
# partition and 1 h. The limit is per JOB, not per GPU: every minute, each interactive partition
# without a job of mine gets ONE job holding up to N GPUs with the next N unfinished runs side by side
# (scripts/delta/pack_job.sh, -t 00:59:00, 16 CPUs per run). A run takes ~45 min on one A100/A40
# (CPU-bound, measured 2026-10-22). Runs may also be queued on the regular/preempt partitions:
# scripts/delta/train_plan_job.sbatch lets the first job to start a run claim it, duplicates exit.
#
#   setsid nohup scripts/delta/interactive_feeder.sh all loo dedicated > <log> 2>&1 &
#   scripts/delta/interactive_feeder.sh --once all loo     # one pass, then exit
#   scripts/delta/interactive_feeder.sh --head 256 all loo dedicated   # nonlinear head (train_plan.sh --head)
#   scripts/delta/interactive_feeder.sh --specs configs/experiments/v10-schedule.yaml dedicated
#       also feed an experiment definition's unstarted runs (scripts/delta/exp_specs.py), after the stages
#
# Stages as in train_plan.sh (all, loo, dedicated, loo:<ds>, dedicated:<ds>); arch S, seed 0.
# GPUs per job: chosen each time from the scheduler's own estimates (sbatch --test-only for 1, 2, 4 GPUs,
# up to MAX_GPUS of the partition): the size that finishes the most runs per hour, G / (wait + RUN_MIN).
# Rebalancing: when a partition's slot is free but every unstarted run already sits in a PENDING job of
# mine elsewhere whose estimated start is > REBALANCE_MIN later than this partition's, that pending job
# is cancelled (never a running one) so its runs move here on the next pass.
# A run line's optional 7th field (exp_specs.py, from a definition's `partitions:`) restricts it to those
# partitions, e.g. A100 only for runs too long for the A40 1-hour slots.
# Stops when no run is missing and none of my interactive jobs is left; kill it to stop early.
set -u
cd "$(dirname "$0")/../.."
export PATH=/work/nvme/beez/laharon/.conda/envs/mighty-mouse/bin:$PATH
# H200 interactive is a fourth slot (user, 2026-10-23: use it when it fits; its nodes train our runs
# 2-3x faster, so long runs such as 384 px fit in one hour there). 96 CPUs per 8 GPUs: 4 x 16 fits.
declare -A MAX_GPUS=([gpuA40x4-interactive]=4 [gpuA100x4-interactive]=4 [gpuA100x8-interactive]=4
                     [gpuH200x8-interactive]=4)
# FEEDER_MAX_GPUS (env) caps GPUs per job, e.g. "gpuH200x8-interactive=2" when 4-GPU jobs there wait a day
for kv in ${FEEDER_MAX_GPUS:-}; do MAX_GPUS[${kv%%=*}]=${kv#*=}; done
RUN_MIN=50   # one run incl. evaluation, minutes (measured ~45-50, 2026-10-22)
REBALANCE_MIN=30
est_start() {  # partition, gpus -> estimated start (epoch s) of a new job, or empty
  local e
  e=$(sbatch --test-only -A beez-delta-gpu -p "$1" --gpus-per-node="$2" --cpus-per-task=$((16 * $2)) \
      --mem=$((60 * $2))G -t 00:59:00 --wrap true 2>&1 | grep -oE "start at [0-9T:-]+" | cut -d' ' -f3)
  [ -n "$e" ] && date -d "$e" +%s
}
best_gpus() {  # partition, runs available -> GPU count with the best runs/hour by the scheduler's estimate
  local p=$1 avail=$2 best=1 best_score=-1 g est wait score
  for g in 1 2 4 8; do   # 8 only where MAX_GPUS allows it (A100x8: 8 GPUs, 128 CPUs per node)
    [ "$g" -gt "${MAX_GPUS[$p]}" ] || [ "$g" -gt "$avail" ] && continue
    est=$(sbatch --test-only -A beez-delta-gpu -p "$p" --gpus-per-node="$g" --cpus-per-task=$((16 * g)) \
          --mem=$((60 * g))G -t 00:59:00 --wrap true 2>&1 | grep -oE "start at [0-9T:-]+" | cut -d' ' -f3)
    [ -z "$est" ] && continue
    wait=$(( ($(date -d "$est" +%s) - $(date +%s)) / 60 )); [ "$wait" -lt 0 ] && wait=0
    score=$(( 1000 * g / (wait + RUN_MIN) ))
    [ "$score" -gt "$best_score" ] && { best=$g; best_score=$score; }
  done
  echo "$best"
}
# FEEDER_PARTS (env) limits the partitions it fills, e.g. to leave one slot to another job
PARTS="${FEEDER_PARTS:-gpuA40x4-interactive gpuA100x4-interactive gpuA100x8-interactive gpuH200x8-interactive}"
allowed_on() {  # run line, partition -> true if the line's 7th field is empty or lists the partition
  local parts
  parts=$(cut -d'|' -f7 <<< "$1")
  [ -z "$parts" ] || [[ ",$parts," == *",$2,"* ]]
}
ONCE=0; HEADOPT=""; SPECDEFS=()
while [ $# -gt 0 ]; do case "$1" in
  --once) ONCE=1; shift;; --head) HEADOPT="--head $2"; shift 2;; --specs) SPECDEFS+=("$2"); shift 2;;
  *) break;; esac; done
STAGES="${*:-all loo dedicated}"
RESULTS=$(python -c "from mighty_mouse.paths import load_paths; print(load_paths()['results_dir'])")
SPECS="$RESULTS/_logs/packs"; mkdir -p "$SPECS" "$RESULTS/_logs/slurm"

while true; do
  # runs not started anywhere (train_plan.sh args lists only `missing` ones), in stage order
  mapfile -t MISSING < <(bash scripts/train_plan.sh args $STAGES --arch S $HEADOPT 2>/dev/null
                         for d in "${SPECDEFS[@]}"; do python scripts/delta/exp_specs.py "$d"; done)
  # runs already waiting in an interactive queue (pack jobs carry their run names in --comment)
  targeted=$(squeue -u "$USER" -h -p "$(echo $PARTS | tr ' ' ,)" -o "%k" 2>/dev/null | tr ',' '\n')
  mine=$(squeue -u "$USER" -h -p "$(echo $PARTS | tr ' ' ,)" 2>/dev/null | wc -l)
  [ "${#MISSING[@]}" -eq 0 ] && [ "$mine" -eq 0 ] && { echo "[$(date +%T)] nothing left to start"; exit 0; }
  for p in $PARTS; do
    squeue -u "$USER" -h -p "$p" 2>/dev/null | grep -q . && continue   # 1 job per user per partition
    avail=0
    for line in "${MISSING[@]}"; do
      allowed_on "$line" "$p" || continue
      grep -qx "${line%%|*}" <<< "$targeted" || avail=$((avail + 1))
    done
    if [ "$avail" -eq 0 ]; then
      # nothing unassigned: take over a pending job elsewhere that would start much later. Only jobs
      # with a real estimate that have waited >= 10 min (a fresh job's estimate is N/A, which once
      # made this cancel brand-new jobs every pass), and the freed runs go to THIS partition in the
      # same pass so they cannot drift back.
      here=$(est_start "$p" 1); [ -z "$here" ] && continue
      # runs this partition may take (a definition's `partitions:` can exclude it)
      allowed_here=$(for line in "${MISSING[@]}"; do allowed_on "$line" "$p" && echo "${line%%|*}"; done)
      stolen=""
      while read -r jid jpart jstart jsubmit jcomment; do
        [ "$jpart" = "$p" ] || [ "$jstart" = N/A ] && continue
        [ $(( $(date +%s) - $(date -d "$jsubmit" +%s) )) -lt 600 ] && continue
        # only a job whose runs may all move here
        tr ',' '\n' <<< "$jcomment" | grep -vxF -f <(printf '%s\n' "$allowed_here") | grep -q . && continue
        if [ "$(date -d "$jstart" +%s)" -gt $((here + REBALANCE_MIN * 60)) ]; then
          scancel "$jid" && { stolen=$jcomment
            echo "[$(date +%T)] rebalance: took over pending $jid ($jpart, est $jstart) -> $p (~$(date -d @"$here" +%H:%M))"; }
          break
        fi
      done < <(squeue -u "$USER" -h -t PD -p "$(echo $PARTS | tr ' ' ,)" -o "%i %P %S %V %k" 2>/dev/null | sort -k3 -r)
      [ -z "$stolen" ] && continue
      targeted=$(grep -vxF -f <(tr ',' '\n' <<< "$stolen") <<< "$targeted")
      avail=$(tr ',' '\n' <<< "$stolen" | grep -c .)
    fi
    G=$(best_gpus "$p" "$avail")
    pick=()
    for line in "${MISSING[@]}"; do
      name=${line%%|*}
      allowed_on "$line" "$p" || continue
      grep -qx "$name" <<< "$targeted" && continue
      pick+=("$line"); targeted+=$'\n'"$name"
      [ "${#pick[@]}" -ge "$G" ] && break
    done
    [ "${#pick[@]}" -eq 0 ] && continue
    n=${#pick[@]}; spec="$SPECS/$(date +%Y%m%d-%H%M%S)_${p}.txt"
    printf '%s\n' "${pick[@]}" > "$spec"
    names=$(printf '%s\n' "${pick[@]}" | cut -d'|' -f1 | paste -sd,)
    job=$(sbatch --parsable -A beez-delta-gpu -p "$p" -t 00:59:00 -J "mm-pack${n}" --comment "$names" \
          --nodes=1 --ntasks=1 --gpus-per-node="$n" --cpus-per-task=$((16 * n)) --mem=$((60 * n))G \
          -o "$RESULTS/_logs/slurm/%x_%j.out" scripts/delta/pack_job.sh "$spec" 2>&1)
    echo "[$(date +%T)] $p <- $n GPU job $job: $names"
  done
  echo "[$(date +%T)] runs not yet started: ${#MISSING[@]}"
  [ $ONCE = 1 ] && exit 0
  sleep 60
done
