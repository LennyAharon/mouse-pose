# GPU placement for the local queues (sourced by scripts/adapt/run_local.sh and scripts/queue_runs.sh).
#
#   source scripts/gpu_slots.sh
#   g=$(pick_gpu "$GPUS" "$PER_GPU")      # least-loaded GPU in the list with load < PER_GPU, or empty
#   ( CUDA_VISIBLE_DEVICES=$g <job> ) &
#   claim_gpu "$g" $!                      # count the job on that GPU until it shows in nvidia-smi
#
# Load of a GPU = its compute processes (nvidia-smi) + launch claims younger than 180 s whose pid is alive.
# Claims live in $GPU_CLAIMS (default /tmp/mm_gpu_claims: per machine, shared by every queue), so two
# queues placing jobs on the same GPUs at the same moment cannot overfill one.
GPU_CLAIMS="${GPU_CLAIMS:-/tmp/mm_gpu_claims}"
mkdir -p "$GPU_CLAIMS"

all_gpus() { nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr -d ' ' | tr '\n' ' '; }

gpu_load() {  # <gpu index> -> number of jobs on it
  local g="$1" uuid procs=0 claims=0 f pid
  uuid=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader 2>/dev/null | tr -d ' ' | awk -F, -v g="$g" '$1 == g {print $2}')
  [ -n "$uuid" ] && procs=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null | tr -d ' ' | grep -c -x "$uuid")
  for f in "$GPU_CLAIMS"/"$g".*; do
    [ -e "$f" ] || continue
    pid="${f##*.}"
    if kill -0 "$pid" 2>/dev/null && [ $(( $(date +%s) - $(stat -c %Y "$f") )) -lt 180 ]; then
      claims=$((claims + 1))
    else
      rm -f "$f"
    fi
  done
  echo $((procs + claims))
}

pick_gpu() {  # "<gpu list>" <per-GPU limit> -> least-loaded GPU with load < limit, or empty
  local best="" best_load=999 g load
  for g in $1; do
    load=$(gpu_load "$g")
    if [ "$load" -lt "$2" ] && [ "$load" -lt "$best_load" ]; then best=$g; best_load=$load; fi
  done
  echo "$best"
}

claim_gpu() { touch "$GPU_CLAIMS/$1.$2"; }  # <gpu index> <pid>
