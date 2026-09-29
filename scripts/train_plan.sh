#!/bin/bash
# Training plan for the current corpus version: which models exist, which are missing, run the
# missing ones with the recipe of record (shared head, T=2, per-dataset zoom-in/out) in two sizes:
#   S = ViT-S DINOv3, 12k steps (configs/model_zoominout.yaml)               -> trunks/, dedicated/
#   B = ViT-B DINOv3, 24k steps (configs/ablations/model_zoominout_24k.yaml) -> trunks_24k/, dedicated_24k/
# A plan runs its own jobs ONE AT A TIME. Before each job it waits until fewer than 2 training runs are
# on the GPU, counted globally (every scripts/train_sweep.py process, whoever started it) -- so a plan
# launched while one other run is training starts right away and runs next to it (user, 2026-09-29). After each leave-one-out run, zero-shot predictions on the
# left-out dataset's TRAIN frames go to <run>/zeroshot/<ds>_train_predictions.csv (test frames: eval/).
#
#   scripts/train_plan.sh plan                      # table: stage / tag / status per arch x seed
#   scripts/train_plan.sh run all loo               # train the missing ones, S and B (detached)
#   scripts/train_plan.sh run all --arch S          # ViT-S only ("S", "B" or "SB")
#   scripts/train_plan.sh run loo --seeds "0;1;2"   # more seeds
#   scripts/train_plan.sh run loo:kaufman           # one dataset's leave-one-out trunk
#   scripts/train_plan.sh run all dedicated:hantman-mv  # one dataset's dedicated model
#   scripts/train_plan.sh dry all                   # print the train_sweep commands only
#   scripts/train_plan.sh show loo                  # print the script `run` would launch, launch nothing
#   scripts/train_plan.sh status                    # tail the running plan's log
# Logs: <results_dir>/_logs/train_plan_<stamp>_<stages>.log, one per `run`; kept as the record of what was
# launched when (small text files), never needed by any script; delete freely once the run is done.
#
# Stages: all = the all-data trunk; dedicated = one model per dataset (its own frames only);
# loo = one leave-one-out trunk per dataset. Datasets and their order come from
# configs/dataset_registry.yaml; tag short names from the TAG map below (keep in sync with
# docs/build_dataset.md). Output layout under results_dir: trunks[_24k]/<tag>_train/... and
# dedicated[_24k]/<ds>_train/... (train_sweep.py's own path inside), so --skip_existing works.
set -u
cd "$(dirname "$0")/.."
RESULTS=$(python -c "from mighty_mouse.paths import load_paths; print(load_paths()['results_dir'])")
DATA=$(python -c "from mighty_mouse.paths import load_paths; print(load_paths()['data_dir'])")
mapfile -t DATASETS < <(python -c "from mighty_mouse.registry import load_registry; print('\n'.join(load_registry()))")
declare -A TAG=([facemap]=face [ibl]=ibl [cheese-2d]=cheese [cazettes-side]=caz [kondo]=kondo [hantman-mv]=hmv [cheese-3d]=c3d [kaufman]=kauf)
SEEDS="0"; ARCHS="S B"
cmd="${1:-plan}"; shift || true
while [ $# -gt 0 ]; do case "$1" in
  --seeds) SEEDS="$2"; shift 2;;
  --arch)  ARCHS="$(echo "$2" | sed 's/./& /g')"; shift 2;;
  *) STAGES="${STAGES:-} $1"; shift;; esac; done
STAGES="${STAGES:-all dedicated loo}"

cfg_of()      { [ "$1" = S ] && echo configs/model_zoominout.yaml || echo configs/ablations/model_zoominout_24k.yaml; }
backbone_of() { [ "$1" = S ] && echo vits_dinov3 || echo vitb_dinov3; }
suffix_of()   { [ "$1" = S ] && echo "" || echo "_24k"; }
tag_of() { local t=""; for d in "$@"; do t="${t:+$t+}${TAG[$d]:?no short name for $d, add it to TAG}"; done; echo "$t"; }
all_tag=$(tag_of "${DATASETS[@]}")
loo_tag() { local out=$1 keep=(); for d in "${DATASETS[@]}"; do [ "$d" != "$out" ] && keep+=("$d"); done; tag_of "${keep[@]}"; }
# train_sweep layout; args: area tag seed arch
out_dir() { echo "$RESULTS/$1$(suffix_of "$4")/${2}_train/supervised/sampling-T2/tf1/$(backbone_of "$4")/seed$3"; }
status_of() { local d=$1; if ls "$d"/eval/*/predictions.csv >/dev/null 2>&1; then echo done; elif [ -d "$d" ]; then echo PARTIAL; else echo missing; fi; }

# rows: stage area tag left_out ("-" unless loo)
rows() {
  for s in $STAGES; do case "$s" in
    all)         echo "all trunks $all_tag -";;
    dedicated)   for d in "${DATASETS[@]}"; do echo "dedicated dedicated $d -"; done;;
    dedicated:*) echo "dedicated dedicated ${s#dedicated:} -";;            # a single dataset's model
    loo)         for d in "${DATASETS[@]}"; do echo "loo trunks $(loo_tag "$d") $d"; done;;
    loo:*)       echo "loo trunks $(loo_tag "${s#loo:}") ${s#loo:}";;       # a single left-out dataset
    *) echo "unknown stage $s" >&2; exit 2;; esac; done
}

case "$cmd" in
  plan)
    echo "corpus: $(basename "$DATA")  results: $RESULTS  seeds: $SEEDS  archs: $ARCHS(S = ViT-S 12k, B = ViT-B 24k)"
    printf "%-10s %-38s" stage tag; for a in $ARCHS; do for s in ${SEEDS//;/ }; do printf " %s-seed%s " "$a" "$s"; done; done; echo
    rows | while read -r stage area tag out; do
      [ -f "$DATA/CollectedData_${tag}_train.csv" ] || { printf "%-10s %-38s  NO CSV (build the tag first)\n" "$stage" "$tag"; continue; }
      printf "%-10s %-38s" "$stage" "$tag"
      for a in $ARCHS; do for s in ${SEEDS//;/ }; do printf " %-9s" "$(status_of "$(out_dir "$area" "$tag" "$s" "$a")")"; done; done; echo
    done ;;
  dry|run|show)
    # Lightning Pose asserts <data_dir>/videos exists even for labeled-frame training
    [ -e "$DATA/videos" ] || { mkdir -p "$DATA/videos"; echo "created empty $DATA/videos"; }
    mkdir -p "$RESULTS/_logs"; LOG="$RESULTS/_logs/train_plan_$(date -u +%Y%m%d-%H%M%S)_$(echo $STAGES | tr -s " :" "-").log"
    script=$(mktemp)
    {
      echo "#!/bin/bash"
      echo "cd $(pwd)"
      echo 'running() { ps aux | grep -c "[s]cripts/train_sweep.py"; }'
      echo "zeroshot() {  # run_dir left_out: predict the left-out dataset's train frames"
      echo "  [ -f \"\$1/zeroshot/\${2}_train_predictions.csv\" ] && return; mkdir -p \"\$1/zeroshot\""
      echo "  python scripts/zeroshot_predict.py --run \"\$1\" --dataset \"\$2\" --data_dir \"$DATA\""
      echo "}"
    } > "$script"
    while read -r stage area tag out; do for a in $ARCHS; do for s in ${SEEDS//;/ }; do
      root="$RESULTS/$area$(suffix_of "$a")"; dir=$(out_dir "$area" "$tag" "$s" "$a")
      sweep="python scripts/train_sweep.py --config_file $(cfg_of "$a") --csv_files CollectedData_${tag}_train.csv --train_frames 1 --seeds $s --backbones $(backbone_of "$a") --sampling_temperatures 2 --head_modes shared --keep_checkpoints --skip_existing --output_root $root"
      if [ "$cmd" = dry ]; then
        echo "$sweep --dry_run" >> "$script"
      else
        echo 'while [ "$(running)" -ge 2 ]; do sleep 60; done' >> "$script"
        post=""; [ "$out" != - ] && post="; zeroshot $dir $out"
        echo "echo \"[\$(date -u +%H:%M)] start $a $stage $tag seed$s\"; $sweep$post; echo \"[\$(date -u +%H:%M)] end $a $stage $tag seed$s\"" >> "$script"
      fi
    done; done; done < <(rows)
    echo 'echo "=== TRAIN PLAN COMPLETE ==="' >> "$script"
    if [ "$cmd" = dry ]; then bash "$script"; elif [ "$cmd" = show ]; then cat "$script"; else
      n=$(ps aux | grep -c "[s]cripts/train_sweep.py"); [ "$n" -ge 2 ] && echo "NOTE: 2 runs already training; the plan waits for a free slot"; [ "$n" -eq 1 ] && echo "NOTE: 1 run already training; the plan starts now and runs next to it"
      setsid nohup bash "$script" > "$LOG" 2>&1 < /dev/null & echo "$!" > "$RESULTS/_train_plan.pid"
      echo "launched pid $(cat "$RESULTS/_train_plan.pid"); log $LOG"; fi ;;
  status) tail -n 15 "$(ls -t "$RESULTS"/_logs/train_plan_*.log 2>/dev/null | head -1)" 2>/dev/null || echo "no plan log";;
  *) echo "usage: $0 plan|dry|show|run|status [stages...] [--seeds \"0;1;2\"] [--arch S|B|SB]"; exit 2;;
esac
