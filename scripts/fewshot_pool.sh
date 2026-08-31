#!/bin/bash
# Worker: claim unclaimed cells from the queue (atomic mkdir) and run them in order.
# Usage: fewshot_pool.sh <worker_id>
S="$(cd "$(dirname "$0")" && pwd)"   # scripts dir; queue + claims live in $S/fewshot_queue/
W="$1"; CLAIMS="$S/fewshot_queue/claims"; mkdir -p "$CLAIMS"
while read -r ARM DS N DRAW; do
  [ -z "$ARM" ] && continue
  mkdir "$CLAIMS/$ARM-$DS-tf$N-draw$DRAW" 2>/dev/null || continue
  # never more than 3 few-shot train/eval processes on the GPU (in-flight cells count)
  while [ "$(pgrep -fc "^bash [^ ]*fewshot_cell.sh")" -ge 3 ]; do sleep 30; done
  bash "$S/fewshot_cell.sh" "$ARM" "$DS" "$N" "$DRAW"
done < "$S/fewshot_queue/queue.txt"
echo "=== [$(date -u +%H:%M)] worker $W: queue exhausted ==="
