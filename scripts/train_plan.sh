#!/bin/bash
# Training plan for the current corpus version: which models exist, which are missing, run the
# missing ones with the recipe of record (shared NONLINEAR head, T=2, per-dataset zoom-in/out) in two sizes:
#   S = ViT-S DINOv3, 12k steps (configs/model_zoominout.yaml)               -> trunks/, dedicated/
#   B = ViT-B DINOv3, 24k steps (configs/ablations/model_zoominout_24k.yaml) -> trunks_24k/, dedicated_24k/
# Keeps up to 2 training runs on the GPU: before each job it waits until fewer than 2 are running,
# counted globally (every scripts/train_sweep.py process, whoever started it), then launches it in the
# background -- a queue with work left always fills a free second slot (user, 2026-09-29). After each leave-one-out run, zero-shot predictions on the
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
#   scripts/train_plan.sh submit all loo --arch S   # NCSA Delta: one SLURM job per run (scripts/delta/train_plan_job.sbatch)
#   scripts/train_plan.sh submit loo --arch S --sbatch "-p gpuA100x4 -t 02:00:00"   # extra sbatch options
#
# HEAD: the shared head is NONLINEAR by default (user, 2026-10-22: the go-to recipe): +model.head_hidden_channels=256
# (ConvT -> 256 -> LayerNorm -> ReLU -> Conv1x1), as the adaptation-curves
# trunks; results in trunks-headmlp256/, dedicated-headmlp256/ (trunks_24k-headmlp256/ ... for B). `--head <N>` sets
# another width; `--head linear` is the stock linear head (old layout: trunks/, dedicated/).
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
declare -A TAG=([facemap]=face [ibl]=ibl [cheese-2d]=cheese [cazettes-side]=caz [kondo]=kondo [hantman-mv]=hmv [cheese-3d]=c3d [kaufman]=kauf
                [aind-vbn-body]=vbnb [aind-vbn-face]=vbnf [cavelli]=cav)
SEEDS="0"; ARCHS="S B"
cmd="${1:-plan}"; shift || true
while [ $# -gt 0 ]; do case "$1" in
  --seeds) SEEDS="$2"; shift 2;;
  --arch)  ARCHS="$(echo "$2" | sed 's/./& /g')"; shift 2;;
  --sbatch) SBATCH_OPTS="$2"; shift 2;;
  --head)  HEAD="$2"; shift 2;;
  *) STAGES="${STAGES:-} $1"; shift;; esac; done
STAGES="${STAGES:-all dedicated loo}"

HEAD="${HEAD:-256}"; [ "$HEAD" = linear ] && HEAD=""; HEADSUF="${HEAD:+-headmlp$HEAD}"
# only the best checkpoint is kept (no periodic ones: ~260 MB each; user, 2026-10-22)
HEAD_OV="${HEAD:++model.head_hidden_channels=$HEAD}"
# every run keeps a rolling checkpoint (lightning-pose training.resume_every_n_steps, 2026-10-23): a run
# cut off by a job's time limit or by preemption continues in the next job instead of starting over
HEAD_OV="${HEAD_OV:+$HEAD_OV;}+training.resume_every_n_steps=500"
cfg_of()      { [ "$1" = S ] && echo configs/model_zoominout.yaml || echo configs/ablations/model_zoominout_24k.yaml; }
backbone_of() { [ "$1" = S ] && echo vits_dinov3 || echo vitb_dinov3; }
suffix_of()   { [ "$1" = S ] && echo "" || echo "_24k"; }
tag_of() { local t=""; for d in "$@"; do t="${t:+$t+}${TAG[$d]:?no short name for $d, add it to TAG}"; done; echo "$t"; }
all_tag=$(tag_of "${DATASETS[@]}")
loo_tag() { local out=$1 keep=(); for d in "${DATASETS[@]}"; do [ "$d" != "$out" ] && keep+=("$d"); done; tag_of "${keep[@]}"; }
# train_sweep layout; args: area tag seed arch
out_dir() { echo "$RESULTS/$1$(suffix_of "$4")$HEADSUF/${2}_train/supervised/sampling-T2/tf1/$(backbone_of "$4")/seed$3"; }
# claim_alive: a live SLURM job holds the run's claim (an unknown answer counts as alive: never double-run)
claim_alive() { local j; j=$(cat "$1.claim/job" 2>/dev/null) || return 1
  case "$(sacct -j "$j" -X -n -o State 2>/dev/null | awk 'NF{print $1; exit}')" in
    COMPLETED|FAILED|CANCELLED*|TIMEOUT|PREEMPTED|NODE_FAIL|OUT_OF_MEMORY|BOOT_FAIL|DEADLINE) return 1;; esac; }
# trained only = training finished (best checkpoint) but no evaluation (a job cut off while evaluating)
trained_only_dir() { grep -q '"status": "COMPLETED"' "$1/train_status.json" 2>/dev/null &&
  ls "$1"/tb_logs/*/version_*/checkpoints/*-best.ckpt >/dev/null 2>&1; }
# resumable = a rolling checkpoint (training.resume_every_n_steps) or trained only, and no live job:
# a new job continues it (or only evaluates it)
status_of() { local d=$1; if ls "$d"/eval/*/predictions.csv >/dev/null 2>&1; then echo done
  elif { [ -f "$d/resume/resume.ckpt" ] || trained_only_dir "$d"; } && ! claim_alive "$d"; then echo resumable
  elif [ -d "$d" ]; then echo PARTIAL; else echo missing; fi; }

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
      # count real python processes only: a /commands/python wrapper shows up as a second train_sweep line
      echo 'running() { ps -eo args | grep -c "^[^ ]*python[0-9.]* scripts/train_sweep.py"; }'
      echo "zeroshot() {  # run_dir left_out: predict the left-out dataset's train frames"
      echo "  [ -f \"\$1/zeroshot/\${2}_train_predictions.csv\" ] && return; mkdir -p \"\$1/zeroshot\""
      echo "  python scripts/zeroshot_predict.py --run \"\$1\" --dataset \"\$2\" --data_dir \"$DATA\""
      echo "}"
    } > "$script"
    while read -r stage area tag out; do for a in $ARCHS; do for s in ${SEEDS//;/ }; do
      root="$RESULTS/$area$(suffix_of "$a")$HEADSUF"; dir=$(out_dir "$area" "$tag" "$s" "$a")
      sweep="python scripts/train_sweep.py --config_file $(cfg_of "$a") --csv_files CollectedData_${tag}_train.csv --train_frames 1 --seeds $s --backbones $(backbone_of "$a") --sampling_temperatures 2 --head_modes shared --keep_checkpoints --skip_existing --output_root $root${HEAD_OV:+ --extra_overrides '$HEAD_OV'}"
      if [ "$cmd" = dry ]; then
        echo "$sweep --dry_run" >> "$script"
      else
        echo 'while [ "$(running)" -ge 2 ]; do sleep 60; done' >> "$script"
        post=""; [ "$out" != - ] && post="; zeroshot $dir $out"
        echo "( echo \"[\$(date -u +%H:%M)] start $a $stage $tag seed$s\"; $sweep$post; echo \"[\$(date -u +%H:%M)] end $a $stage $tag seed$s\" ) & sleep 120" >> "$script"
      fi
    done; done; done < <(rows)
    [ "$cmd" != dry ] && echo "wait" >> "$script"
    echo 'echo "=== TRAIN PLAN COMPLETE ==="' >> "$script"
    if [ "$cmd" = dry ]; then bash "$script"; elif [ "$cmd" = show ]; then cat "$script"; else
      n=$(ps -eo args | grep -c "^[^ ]*python[0-9.]* scripts/train_sweep.py"); [ "$n" -ge 2 ] && echo "NOTE: 2 runs already training; the plan waits for a free slot"; [ "$n" -eq 1 ] && echo "NOTE: 1 run already training; the plan fills the second slot now"
      setsid nohup bash "$script" > "$LOG" 2>&1 < /dev/null & echo "$!" > "$RESULTS/_train_plan.pid"
      echo "launched pid $(cat "$RESULTS/_train_plan.pid"); log $LOG"; fi ;;
  args)
    # one line per missing run: <job name>|<run dir>|<left out>|<train_sweep args>  (used by
    # scripts/delta/interactive_feeder.sh to pack several runs into one multi-GPU job)
    while read -r stage area tag out; do for a in $ARCHS; do for s in ${SEEDS//;/ }; do
      [ -f "$DATA/CollectedData_${tag}_train.csv" ] || continue
      root="$RESULTS/$area$(suffix_of "$a")$HEADSUF"; dir=$(out_dir "$area" "$tag" "$s" "$a")
      case "$(status_of "$dir")" in missing|resumable) ;; *) continue;; esac
      label=$out; [ "$out" = - ] && { [ "$stage" = dedicated ] && label=$tag || label=all; }
      echo "mm-$a$HEAD-$stage-$label-s$s|$dir|$out|--config_file $(cfg_of "$a") --csv_files CollectedData_${tag}_train.csv --train_frames 1 --seeds $s --backbones $(backbone_of "$a") --sampling_temperatures 2 --head_modes shared --keep_checkpoints --skip_existing --output_root $root${HEAD_OV:+ --extra_overrides $HEAD_OV}"
    done; done; done < <(rows) ;;
  submit)
    # NCSA Delta: one SLURM job per missing run; done/PARTIAL runs are skipped (delete a PARTIAL dir to rerun)
    [ -e "$DATA/videos" ] || { mkdir -p "$DATA/videos"; echo "created empty $DATA/videos"; }
    mkdir -p "$RESULTS/_logs/slurm"
    while read -r stage area tag out; do for a in $ARCHS; do for s in ${SEEDS//;/ }; do
      [ -f "$DATA/CollectedData_${tag}_train.csv" ] || { echo "skip $stage $tag: NO CSV (build the tag first)"; continue; }
      root="$RESULTS/$area$(suffix_of "$a")$HEADSUF"; dir=$(out_dir "$area" "$tag" "$s" "$a"); st=$(status_of "$dir")
      case "$st" in missing|resumable) ;; *) echo "skip $a $stage $tag seed$s: $st"; continue;; esac
      label=$out; [ "$out" = - ] && { [ "$stage" = dedicated ] && label=$tag || label=all; }
      name="mm-$a$HEAD-$stage-$label-s$s"
      sbatch --parsable --job-name="$name" -o "$RESULTS/_logs/slurm/%x_%j.out" ${SBATCH_OPTS:-} \
        scripts/delta/train_plan_job.sbatch "$dir" "$out" \
        --config_file "$(cfg_of "$a")" --csv_files "CollectedData_${tag}_train.csv" --train_frames 1 --seeds "$s" \
        --backbones "$(backbone_of "$a")" --sampling_temperatures 2 --head_modes shared --keep_checkpoints \
        --skip_existing --output_root "$root" ${HEAD_OV:+--extra_overrides "$HEAD_OV"} | sed "s|^|submitted $name: job |"
    done; done; done < <(rows) ;;
  status) tail -n 15 "$(ls -t "$RESULTS"/_logs/train_plan_*.log 2>/dev/null | head -1)" 2>/dev/null || echo "no plan log";;
  *) echo "usage: $0 plan|dry|show|run|submit|args|status [stages...] [--seeds \"0;1;2\"] [--arch S|B|SB]"; exit 2;;
esac
