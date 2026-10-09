#!/bin/bash
# Run an adaptation grid on a SLURM cluster (e.g. ACCESS): one array task per line of the job file.
#
#   python scripts/adapt/plan.py --config configs/adaptation/cosyne-2026.yaml --jobs jobs.txt
#   sbatch --array=0-$(( $(wc -l < jobs.txt) - 1 ))%16 scripts/adapt/slurm_array.sbatch configs/adaptation/cosyne-2026.yaml jobs.txt
#
# Each task trains + evaluates one cell (~10-40 min on an L4, faster on A100; all-frames cells ~1-1.5 h) and is
# idempotent (`.done` cells exit at once), so the same command can be resubmitted after failures. Requirements on the
# cluster: this repo + paths.yaml pointing at a copy of the data dir and results tree (trunks with checkpoints), and
# an environment with mighty_mouse and Lightning Pose post_sub_lp installed (see skills/adaptation-curves/SKILL.md).
# Adjust the #SBATCH lines to the cluster's partition / account names.
#SBATCH --job-name=mm-adapt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%A_%a.out
set -eu
CFG="$1"; JOBS="$2"
cd "${SLURM_SUBMIT_DIR:-.}"
# activate the environment here if the cluster needs it, e.g.: module load anaconda && conda activate mighty-mouse
python scripts/adapt/run_cell.py --config "$CFG" --jobs "$JOBS" --index "${SLURM_ARRAY_TASK_ID}"
