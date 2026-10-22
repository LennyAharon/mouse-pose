#!/bin/bash
# Zero-shot along training on NCSA Delta: for every leave-out trunk under <results_dir>/<area>/ that still
# has periodic checkpoints, predict each left-out dataset's test AND train frames with every
# *-periodic.ckpt (scripts/zeroshot_predict.py --every_ckpt ->
# <run>/zeroshot/steps/step<k>_<ds>_{test,train}_predictions.csv). Resumable: finished outputs are
# skipped, so a job that hits its time limit is simply resubmitted. Score with scripts/zeroshot_steps.py.
#
#   sbatch -p gpuH200x8-interactive -t 00:59:00 scripts/delta/zeroshot_steps.sbatch [area]
#SBATCH --job-name=mm-zs-steps
#SBATCH --account=beez-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
set -u
AREA="${1:-trunks-headmlp256}"
ENV=/work/nvme/beez/laharon/.conda/envs/mighty-mouse
export PATH="$ENV/bin:$PATH"
export HF_HOME=/work/nvme/beez/laharon/.cache/huggingface
cd /work/nvme/beez/laharon/MM_project/mouse-pose
RESULTS=$(python -c "from mighty_mouse.paths import load_paths; print(load_paths()['results_dir'])")
echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) area=$AREA"
for run in "$RESULTS/$AREA"/*_train/supervised/sampling-T2/tf1/vits_dinov3/seed*; do
    ls "$run"/tb_logs/*/version_*/checkpoints/*-periodic.ckpt >/dev/null 2>&1 || continue
    for zs in "$run"/zeroshot/*_train_predictions.csv; do
        [ -e "$zs" ] || continue
        ds=$(basename "$zs" _train_predictions.csv)
        echo "[$(date +%T)] $ds <- $(basename "$(dirname "$(dirname "$(dirname "$(dirname "$(dirname "$run")")")")")")"
        python scripts/zeroshot_predict.py --run "$run" --dataset "$ds" --every_ckpt 2>&1 | grep -E "^checkpoint|^wrote|Error|Traceback"
    done
done
echo "[$(date +%T)] done"
