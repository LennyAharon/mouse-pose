#!/bin/bash
#SBATCH --job-name=mightymouse
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G          # ViT-S; raise to 64G+ and request more GPU memory for ViT-B / higher resolution
#SBATCH --time=04:00:00
# One array task = one model. orchestrate.py submits this file with --array=0-N and exports
# JOBS_FILE (tab-separated: config, csv_file, seed, output_root; one cell per line) and REPO.
# Resource lines above are the per-model profile measured on an L4 (24 GB): ~6 GB GPU memory,
# CPU-bound augmentation (8 cores keep one GPU busy), ~6 GB RAM, ~2.5 h for 12k steps on L4.
# On a B200 the GPU time per model should drop well below an hour; the CPU side will then be
# the bottleneck unless --cpus-per-task is raised or DALI/GPU augmentation is enabled.
set -euo pipefail

# ── environment: adjust to the cluster (module load / conda activate / container) ──
# module load cuda/12.8
# source ~/miniconda3/etc/profile.d/conda.sh && conda activate cloudspace
cd "${REPO:?REPO not set}"

line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${JOBS_FILE:?JOBS_FILE not set}")
IFS=$'\t' read -r CONFIG CSV_FILE SEED OUTPUT_ROOT <<< "$line"
echo "task $SLURM_ARRAY_TASK_ID on $(hostname): $CONFIG $CSV_FILE seed $SEED -> $OUTPUT_ROOT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python scripts/cluster/fit_one.py --config "$CONFIG" --csv_file "$CSV_FILE" --seed "$SEED" \
    --output_root "$OUTPUT_ROOT" --keep_checkpoints
