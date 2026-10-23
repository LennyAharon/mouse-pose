#!/bin/bash
# One recipe-of-record run on NCSA Delta, as submitted by `scripts/train_plan.sh submit`:
# train_sweep.py (train + evaluate on every dataset's test set), then, for a leave-one-out
# trunk, zero-shot predictions on the left-out dataset's TRAIN frames
# (<run>/zeroshot/<ds>_train_predictions.csv), exactly as `train_plan.sh run` does locally.
#
#   sbatch [-p ...] [-t ...] scripts/delta/train_plan_job.sbatch <run_dir> <left_out|-> <train_sweep.py args...>
#
# <left_out> may list several datasets (comma-separated, e.g. cheese-2d,cheese-3d for a trunk that leaves
# out both): zero-shot train-frame predictions are then written for each.
#
# Reads paths.yaml (data_dir, results_dir) when the job STARTS: don't repoint it while jobs are queued.
# The same run may be queued on several routes at once (regular/preempt AND interactive, see
# scripts/delta/interactive_feeder.sh): the first job to start claims it (atomic mkdir <run_dir>.claim
# holding its job id); a duplicate whose run is done, or claimed by a live job, exits at once. A claim
# left by a dead job (time limit, crash, preemption) is taken over and the unfinished run dir deleted
# (train_sweep --skip_existing would otherwise skip it), so preemptible partitions are safe too.
#SBATCH --job-name=mm-train
#SBATCH --account=beez-delta-gpu
#SBATCH --partition=gpuA100x4,gpuA100x8,gpuA40x4,gpuA100x4-preempt,gpuA40x4-preempt
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=03:00:00
set -eo pipefail

RUN_DIR="$1"; LEFT_OUT="$2"; shift 2

ENV=/work/nvme/beez/laharon/.conda/envs/mighty-mouse
export PATH="$ENV/bin:$PATH"
export HF_HOME=/work/nvme/beez/laharon/.cache/huggingface
cd /work/nvme/beez/laharon/MM_project/mouse-pose

echo "host=$(hostname) partition=$SLURM_JOB_PARTITION gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "mouse-pose $(git rev-parse --abbrev-ref HEAD)@$(git rev-parse --short HEAD)" \
     "lightning-pose $(git -C ../lightning-pose rev-parse --abbrev-ref HEAD)@$(git -C ../lightning-pose rev-parse --short HEAD)"
done_run() {  # evaluated, and for a leave-out trunk the zero-shot train predictions of each left-out set exist
    ls "$RUN_DIR"/eval/*/predictions.csv >/dev/null 2>&1 || return 1
    [ "$LEFT_OUT" = - ] && return 0
    local ds; for ds in ${LEFT_OUT//,/ }; do [ -f "$RUN_DIR/zeroshot/${ds}_train_predictions.csv" ] || return 1; done
}
done_run && { echo "already done: $RUN_DIR"; exit 0; }
CLAIM="$RUN_DIR.claim"; mkdir -p "$(dirname "$RUN_DIR")"
if mkdir "$CLAIM" 2>/dev/null; then
    echo "$SLURM_JOB_ID" > "$CLAIM/job"
else
    owner=$(cat "$CLAIM/job" 2>/dev/null)
    if [ "$owner" != "$SLURM_JOB_ID" ]; then
        state=$(sacct -j "$owner" -X -n -o State 2>/dev/null | awk 'NF{print $1; exit}')
        case "$state" in RUNNING|PENDING|REQUEUED|CONFIGURING|COMPLETING|"")
            echo "claimed by job $owner ($state): exiting"; exit 0;; esac
        echo "taking over from job $owner ($state)"; echo "$SLURM_JOB_ID" > "$CLAIM/job"
    fi
    # an earlier attempt (ours after a requeue, or a dead owner's) left an unfinished run dir
    # (a run with a rolling checkpoint, training.resume_every_n_steps, is kept and continues instead)
    if [ -d "$RUN_DIR" ] && ! ls "$RUN_DIR"/eval/*/predictions.csv >/dev/null 2>&1; then
        if [ -f "$RUN_DIR/resume/resume.ckpt" ]; then
            echo "resuming $RUN_DIR from its rolling checkpoint"
        else
            echo "removing unfinished $RUN_DIR"; rm -rf "$RUN_DIR"
        fi
    fi
fi
echo "[$(date -u +%H:%M)] start $RUN_DIR"

python scripts/train_sweep.py "$@"

if [ "$LEFT_OUT" != - ]; then
    mkdir -p "$RUN_DIR/zeroshot"
    for ds in ${LEFT_OUT//,/ }; do
        [ -f "$RUN_DIR/zeroshot/${ds}_train_predictions.csv" ] ||
            python scripts/zeroshot_predict.py --run "$RUN_DIR" --dataset "$ds"
    done
fi
echo "[$(date -u +%H:%M)] end $RUN_DIR"
