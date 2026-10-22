#!/bin/bash
# GPU wait report for NCSA Delta (skills/delta-gpu-queue): before submitting, which partition gives a
# GPU soonest for THIS request; after submitting, when queued jobs are expected to start and why.
#
#   scripts/delta/gpu_wait.sh                                  # default request: 1 GPU, 16 CPU, 60G, 3h
#   scripts/delta/gpu_wait.sh --time 00:55:00 --gpus 1 --cpus 16 --mem 60G
#   scripts/delta/gpu_wait.sh --jobs                           # only the status of my queued/running jobs
#
# Sections: 1 cluster health (node states, newest GPU job start = is anything starting at all?),
# 2 per-partition estimate for the request (sbatch --test-only; nothing is submitted), priority factor,
# max time, queue depth, my per-user limit, 3 account (fair-share, GPU hours left), 4 my jobs (state,
# estimated start, reason, priority), 5 how long my recent jobs actually took (sacct, last 14 days).
set -u
ACCOUNT=beez-delta-gpu; GPUS=1; CPUS=16; MEM=60G; TIME=03:00:00; ONLY_JOBS=0
PARTS="gpuA100x4-interactive gpuA40x4-interactive gpuA100x8-interactive gpuH200x8-interactive
gpuA100x4 gpuA100x8 gpuH200x8 gpuA40x4 gpuA100x4-preempt gpuA40x4-preempt"
while [ $# -gt 0 ]; do case "$1" in
  --gpus) GPUS=$2; shift 2;; --cpus) CPUS=$2; shift 2;; --mem) MEM=$2; shift 2;;
  --time) TIME=$2; shift 2;; --account) ACCOUNT=$2; shift 2;; --partitions) PARTS=$2; shift 2;;
  --jobs) ONLY_JOBS=1; shift;; *) echo "unknown option $1"; exit 2;; esac; done

secs() { local t=$1 d=0; [[ $t == *-* ]] && { d=${t%%-*}; t=${t#*-}; }; IFS=: read -r h m s <<< "$t"
         [ -z "${s:-}" ] && { s=$m; m=$h; h=0; }; echo $(( d*86400 + 10#$h*3600 + 10#$m*60 + 10#$s )); }
human() { local s=$1; [ "$s" -lt 0 ] && s=0; printf "%dh%02dm" $((s/3600)) $((s%3600/60)); }
now=$(date +%s)
echo "=== Delta GPU wait report  $(date '+%F %T')  request: ${GPUS} GPU, ${CPUS} CPU, ${MEM}, ${TIME} ==="

if [ $ONLY_JOBS = 0 ]; then
  echo; echo "1. Cluster health (GPU nodes by state; drain*/down* = out of service)"
  sinfo -h -p gpuA100x4,gpuA100x8,gpuA40x4,gpuH200x8 -o "%P %t %D" \
    | awk '{a[$1]=a[$1]" "$2"="$3} END {for (p in a) printf "   %-11s%s\n", p":", a[p]}' | sort
  newest=$(timeout 60 squeue -h -t R -o "%S" -p "$(echo $PARTS | tr ' ' ,)" 2>/dev/null | sort | tail -1)
  if [ -n "$newest" ]; then
    age=$(( now - $(date -d "$newest" +%s) ))
    echo "   newest GPU job start (any user): $newest ($(human $age) ago)$([ $age -gt 7200 ] && echo '  <-- NOTHING HAS STARTED FOR >2h: cluster problem')"
  fi

  echo; echo "2. Where would this request start? (sbatch --test-only; nothing submitted)"
  printf "   %-24s %-10s %-9s %-8s %-8s %-20s %s\n" partition prio-fact max-time pending running "estimated start" "wait"
  for p in $PARTS; do
    info=$(scontrol show partition "$p" 2>/dev/null) || continue
    fact=$(grep -oE "PriorityJobFactor=[0-9]+" <<< "$info" | cut -d= -f2)
    maxt=$(grep -oE "MaxTime=[^ ]+" <<< "$info" | cut -d= -f2)
    t=$TIME; [ "$maxt" != UNLIMITED ] && [ "$(secs "$TIME")" -gt "$(secs "$maxt")" ] && t=$maxt
    out=$(sbatch --test-only -A "$ACCOUNT" -p "$p" --gpus-per-node="$GPUS" --cpus-per-task="$CPUS" \
          --mem="$MEM" -t "$t" --wrap true 2>&1)
    if est=$(grep -oE "start at [0-9T:-]+" <<< "$out" | cut -d' ' -f3); [ -n "$est" ]; then
      wait=$(human $(( $(date -d "$est" +%s) - now )))
    elif grep -q QOSMaxSubmitJobPerUser <<< "$out"; then est="(your 1 job slot is in use)"; wait=-
    else est="(no estimate)"; wait=-; fi
    note=""; [ "$t" != "$TIME" ] && note=" [time capped to $t]"
    printf "   %-24s %-10s %-9s %-8s %-8s %-20s %s%s\n" "$p" "$fact" "$maxt" \
      "$(squeue -h -p "$p" -t PD 2>/dev/null | wc -l)" "$(squeue -h -p "$p" -t R 2>/dev/null | wc -l)" "$est" "$wait" "$note"
  done
  echo "   limits: *-interactive = 1 job per user per partition (4 partitions -> up to 4 at once), max 1h;"
  echo "           *-preempt = can be killed and requeued (only with a requeue-safe script)."

  echo; echo "3. Account"
  accounts 2>/dev/null | grep -E "^$ACCOUNT" | awk '{printf "   %s: %s GPU-hours left of %s\n", $1, $2, $3}'
  sshare -h -A "$ACCOUNT" -u "$USER" -o Account,User,FairShare 2>/dev/null | awk 'NF==3 {printf "   fair-share %s (1 = full priority, 0 = none; low = recent heavy use)\n", $3}'
fi

echo; echo "4. My jobs"
if squeue -u "$USER" -h | grep -q .; then
  printf "   %-10s %-34s %-3s %-9s %-20s %-10s %s\n" jobid name st time "start (est/actual)" priority "reason / partition"
  squeue -u "$USER" -h -o "%i|%j|%t|%M|%S|%Q|%r|%P" | sort -t'|' -k5 | while IFS='|' read -r id n st tm s q r pt; do
    [ "$st" = R ] && r=$pt
    printf "   %-10s %-34.34s %-3s %-9s %-20s %-10s %s\n" "$id" "$n" "$st" "$tm" "$s" "$q" "${r:0:40}"
  done
  echo "   running $(squeue -u "$USER" -h -t R | wc -l), pending $(squeue -u "$USER" -h -t PD | wc -l)"
else echo "   none"; fi

if [ $ONLY_JOBS = 0 ]; then
  echo; echo "5. How long my recent GPU jobs took (completed, last 14 days; queue wait / run time)"
  sacct -u "$USER" -X -n -S "$(date -d '-14 days' +%F)" -P -o JobName,Partition,Submit,Start,Elapsed,State 2>/dev/null \
    | awk -F'|' '$2 ~ /gpu/ && $6 ~ /^(COMPLETED|TIMEOUT|FAILED|CANCELLED|PREEMPTED)/ && $4 != "Unknown" && $4 != "None"' \
    | tail -8 | while IFS='|' read -r n p sub st el state; do
      w=$(( $(date -d "$st" +%s) - $(date -d "$sub" +%s) ))
      printf "   %-30.30s %-22.22s waited %-7s ran %-9s %s\n" "$n" "$p" "$(human $w)" "$el" "$state"
    done
fi
