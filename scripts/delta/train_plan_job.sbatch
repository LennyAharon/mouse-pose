#!/bin/bash
# One recipe-of-record run on NCSA Delta, as submitted by `scripts/train_plan.sh submit`:
# train_sweep.py (train + evaluate on every dataset's test set), then, for a leave-one-out
# trunk, zero-shot predictions on the left-out dataset's TRAIN frames
# (<run>/zeroshot/<ds>_train_predictions.csv), exactly as `train_plan.sh run` does locally.
#
#   sbatch [-p ...] [-t ...] scripts/delta/train_plan_job.sbatch <run_dir> <left_out|-> <train_sweep.py args...>
#
# Reads paths.yaml (data_dir, results_dir) when the job STARTS: don't repoint it while jobs are queued.
#SBATCH --job-name=mm-train
#SBATCH --account=beez-delta-gpu
#SBATCH --partition=gpuA100x4,gpuA100x8,gpuH200x8,gpuA40x4
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
echo "[$(date -u +%H:%M)] start $RUN_DIR"

python scripts/train_sweep.py "$@"

if [ "$LEFT_OUT" != - ] && [ ! -f "$RUN_DIR/zeroshot/${LEFT_OUT}_train_predictions.csv" ]; then
    mkdir -p "$RUN_DIR/zeroshot"
    python scripts/zeroshot_predict.py --run "$RUN_DIR" --dataset "$LEFT_OUT"
fi
echo "[$(date -u +%H:%M)] end $RUN_DIR"
