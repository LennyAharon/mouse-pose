#!/bin/bash
# Progress of my running Delta training jobs (skills/delta-gpu-queue): one line per run with steps
# done, speed, projected end of training, and the job's hard time limit, plus the feeder's status.
#
#   scripts/delta/train_progress.sh              # once
#   watch -n 60 scripts/delta/train_progress.sh  # refresh every minute (Ctrl+C to stop)
set -u
cd "$(dirname "$0")/../.."
RESULTS=$(grep -E "^results_dir:" paths.yaml | awk '{print $2}')
LOGS="$RESULTS/_logs/slurm"; MAX_STEPS=12000
now=$(date +%s)
STATE="${XDG_CACHE_HOME:-$HOME/.cache}/train_progress.state"; mkdir -p "$(dirname "$STATE")"
# keep the previous snapshot unless it is under 30 s old (so a quick re-run still has a baseline)
[ -f "$STATE" ] && [ $(( now - $(stat -c %Y "$STATE") )) -ge 30 ] && mv "$STATE" "$STATE.old"
: > "$STATE"
echo "=== training progress $(date '+%F %T') ==="
printf "%-30s %-11s %-13s %-9s %-11s %-11s %s\n" run job "steps" "steps/s" "train ends" "job limit" "status"
squeue -u "$USER" -h -t R -o "%i|%j|%e|%P" | while IFS='|' read -r job name end part; do
  for f in "$LOGS"/*_"$job".out; do
    [ -e "$f" ] || continue
    run=$(basename "$f" "_$job.out"); [[ $run == mm-S-pack* ]] && continue
    last=$(tr '\r' '\n' < "$f" | grep -E "^Epoch [0-9]+:" | tail -1)
    if [ -z "$last" ]; then
      st=$(grep -qE "claimed by|already done" "$f" && echo "skipped (duplicate)" || echo "starting / evaluating")
      printf "%-30s %-11s %-13s %-9s %-11s %-11s %s\n" "$run" "$job" - - - "${end#*T}" "$st"; continue
    fi
    ep=$(grep -oE "^Epoch [0-9]+" <<< "$last" | grep -oE "[0-9]+$")
    st=$(grep -oE "[0-9]+/[0-9]+ \[" <<< "$last" | head -1 | grep -oE "^[0-9]+/[0-9]+")
    steps=$(( ep * ${st#*/} + ${st%/*} )); [ "$steps" -gt $MAX_STEPS ] && steps=$MAX_STEPS
    # speed = steps gained since this script's previous call (30 s - 30 min ago, kept in $STATE), which
    # includes validation pauses; on a first call, Lightning's own running it/s from the progress bar
    prev=$(awk -v r="$run" -v j="$job" '$1==r && $2==j {print $3, $4}' "$STATE.old" 2>/dev/null)
    rate=""
    if [ -n "$prev" ]; then
      dt=$(( now - ${prev#* } )); [ "$dt" -ge 30 ] && [ "$dt" -le 1800 ] && [ "$steps" -gt "${prev% *}" ] &&
        rate=$(awk -v a="${prev% *}" -v b="$steps" -v t="$dt" 'BEGIN {printf "%.2f", (b - a) / t}')
    fi
    [ -z "$rate" ] && rate=$(grep -oE "[0-9.]+it/s" <<< "$last" | head -1 | grep -oE "^[0-9.]+")
    rate=${rate:-0}
    echo "$run $job $steps $now" >> "$STATE"
    eta=$(awk -v s="$steps" -v r="$rate" -v n="$now" -v m=$MAX_STEPS 'BEGIN {printf "%d", (r > 0 ? n + (m - s) / r : n)}')
    flag=""; [ "$eta" -gt "$(( $(date -d "$end" +%s) - 300 ))" ] && flag="  <-- may hit the time limit"
    err=$(grep -E "Traceback|Error" "$f" | grep -vi warn | head -1 | cut -c1-60)
    printf "%-30s %-11s %5d/%-7d %-9s %-11s %-11s %s\n" "$run" "$job" "$steps" $MAX_STEPS "$rate" \
      "$(date -d @"$eta" +%H:%M)" "${end#*T}" "training${flag}${err:+  ERROR: $err}"
  done
done
echo
echo "queued: $(squeue -u "$USER" -h -t PD | wc -l) jobs   finished runs: $(ls -d "$RESULTS"/{trunks,dedicated}*/*/supervised/*/*/*/seed*/eval 2>/dev/null | wc -l) / 23"
feeder=$(ls -t "$RESULTS"/_logs/interactive_feeder_*.log 2>/dev/null | head -1)
[ -n "$feeder" ] && echo "feeder ($(pgrep -u "$USER" -f interactive_feeder.sh >/dev/null && echo running || echo STOPPED)): $(tail -1 "$feeder")"
