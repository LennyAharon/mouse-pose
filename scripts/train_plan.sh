#!/bin/bash
# Training plan for the current corpus version: which models exist, which are missing, run the
# missing ones in sequence with the recipe of record (shared head, T=2, per-dataset zoom-in/out,
# configs/model_zoominout.yaml, 12k steps, ViT-S DINOv3), one at a time on the single GPU.
#
#   scripts/train_plan.sh plan                      # table: stage / tag / status per seed
#   scripts/train_plan.sh run all dedicated loo     # train the missing ones of these stages (detached)
#   scripts/train_plan.sh run loo --seeds "0;1;2"   # more seeds
#   scripts/train_plan.sh run all dedicated:hantman-mv  # one dataset's dedicated model
#   scripts/train_plan.sh dry all                   # print the train_sweep commands only
#   scripts/train_plan.sh status                    # tail the running plan's log
# Logs: <results_dir>/_logs/train_plan_<stamp>.log, one per `run`; kept as the record of what was
# launched when (small text files), never needed by any script; delete freely once the run is done.
#
# Stages: all = the all-data trunk; dedicated = one model per dataset (its own frames only);
# loo = one leave-one-out trunk per dataset. Datasets and their order come from
# configs/dataset_registry.yaml; tag short names from the TAG map below (keep in sync with
# docs/build_dataset.md). Output layout under results_dir: trunks/<tag>_train/... and
# dedicated/<ds>_train/... (train_sweep.py's own path inside), so --skip_existing works.
set -u
cd "$(dirname "$0")/.."
RESULTS=$(python -c "from mouse_pose.paths import load_paths; print(load_paths()['results_dir'])")
DATA=$(python -c "from mouse_pose.paths import load_paths; print(load_paths()['data_dir'])")
mapfile -t DATASETS < <(python -c "from mouse_pose.registry import load_registry; print('\n'.join(load_registry()))")
declare -A TAG=([facemap]=face [ibl]=ibl [cheese-2d]=cheese [cazettes-side]=caz [kondo]=kondo [hantman-mv]=hmv)
CONFIG=configs/model_zoominout.yaml; BACKBONE=vits_dinov3; SEEDS="0"; STEP_ROOT=""
cmd="${1:-plan}"; shift || true
while [ $# -gt 0 ]; do case "$1" in --seeds) SEEDS="$2"; shift 2;; *) STAGES="${STAGES:-} $1"; shift;; esac; done
STAGES="${STAGES:-all dedicated loo}"

tag_of() { local t=""; for d in "$@"; do t="${t:+$t+}${TAG[$d]:?no short name for $d, add it to TAG}"; done; echo "$t"; }
all_tag=$(tag_of "${DATASETS[@]}")
loo_tag() { local out=$1 keep=(); for d in "${DATASETS[@]}"; do [ "$d" != "$out" ] && keep+=("$d"); done; tag_of "${keep[@]}"; }
out_dir() { echo "$RESULTS/$1/${2}_train/supervised/sampling-T2/tf1/$BACKBONE/seed$3"; }   # train_sweep layout
status_of() { local d=$1; if ls "$d"/eval/*/predictions.csv >/dev/null 2>&1; then echo done; elif [ -d "$d" ]; then echo PARTIAL; else echo missing; fi; }

# rows: stage area tag
rows() {
  for s in $STAGES; do case "$s" in
    all)       echo "all trunks $all_tag";;
    dedicated) for d in "${DATASETS[@]}"; do echo "dedicated dedicated $d"; done;;
    dedicated:*) echo "dedicated dedicated ${s#dedicated:}";;          # a single dataset's model
    loo)       for d in "${DATASETS[@]}"; do echo "loo trunks $(loo_tag "$d")"; done;;
    *) echo "unknown stage $s" >&2; exit 2;; esac; done
}

case "$cmd" in
  plan)
    echo "corpus: $(basename "$DATA")  results: $RESULTS  seeds: $SEEDS"
    printf "%-10s %-38s" stage tag; for s in ${SEEDS//;/ }; do printf " seed%s   " "$s"; done; echo
    rows | while read -r stage area tag; do
      [ -f "$DATA/CollectedData_${tag}_train.csv" ] || { printf "%-10s %-38s  NO CSV (build the tag first)\n" "$stage" "$tag"; continue; }
      printf "%-10s %-38s" "$stage" "$tag"; for s in ${SEEDS//;/ }; do printf " %-8s" "$(status_of "$(out_dir "$area" "$tag" "$s")")"; done; echo
    done ;;
  dry|run)
    # Lightning Pose asserts <data_dir>/videos exists even for labeled-frame training
    [ -e "$DATA/videos" ] || { mkdir -p "$DATA/videos"; echo "created empty $DATA/videos"; }
    mkdir -p "$RESULTS/_logs"; LOG="$RESULTS/_logs/train_plan_$(date -u +%Y%m%d-%H%M).log"
    script=$(mktemp); echo "#!/bin/bash" > "$script"; echo "cd $(pwd)" >> "$script"
    rows | while read -r stage area tag; do
      echo "python scripts/train_sweep.py --config_file $CONFIG --csv_files CollectedData_${tag}_train.csv --train_frames 1 --seeds \"$SEEDS\" --backbones $BACKBONE --sampling_temperatures 2 --head_modes shared --keep_checkpoints --skip_existing --output_root $RESULTS/$area $([ "$cmd" = dry ] && echo --dry_run)" >> "$script"
    done
    echo 'echo "=== TRAIN PLAN COMPLETE ==="' >> "$script"
    if [ "$cmd" = dry ]; then bash "$script"; else
      nvidia-smi --query-gpu=memory.used --format=csv,noheader | grep -q "^0 MiB" || echo "WARNING: GPU is not idle"
      setsid nohup bash "$script" > "$LOG" 2>&1 < /dev/null & echo "$!" > "$RESULTS/_train_plan.pid"
      echo "launched pid $(cat "$RESULTS/_train_plan.pid"); log $LOG"; fi ;;
  status) tail -n 15 "$(ls -t "$RESULTS"/_logs/train_plan_*.log 2>/dev/null | head -1)" 2>/dev/null || echo "no plan log";;
  *) echo "usage: $0 plan|dry|run|status [stages...] [--seeds \"0;1;2\"]"; exit 2;;
esac
