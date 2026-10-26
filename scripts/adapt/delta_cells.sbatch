#!/bin/bash
# Run several adaptation cells side by side in ONE SLURM job on NCSA Delta, one cell per GPU, so a
# single interactive slot (1 job per user per partition) trains them at once.
#
#   sbatch -p gpuA100x4-interactive --gpus-per-node=3 --cpus-per-task=48 --mem=180G -t 01:00:00 \
#       -o <results_dir>/adaptation/<name>/_queue/%x_%j.out \
#       scripts/adapt/delta_cells.sbatch configs/adaptation/<name>.yaml <cell id> [<cell id> ...]
#
# Each cell runs scripts/adapt/run_cell.py (output <queue dir>/<cell id>.out, training log next to the cell
# dir). Done cells are skipped; a cell whose arm keeps a rolling resume checkpoint
# (+training.resume_every_n_steps) and is cut off by the time limit continues when the same command is
# submitted again. Lightning Pose is imported from the branch checked out in MM_project/lightning-pose.
#SBATCH --job-name=mm-adapt
#SBATCH --account=beez-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=01:00:00
set -u

CFG="$1"; shift
cd /work/nvme/beez/laharon/MM_project/mouse-pose
ENV=/work/nvme/beez/laharon/.conda/envs/mighty-mouse
export PATH="$ENV/bin:$PATH"
export HF_HOME=/work/nvme/beez/laharon/.cache/huggingface
Q=$(python - "$CFG" <<'EOF'
import sys
from mighty_mouse.adaptation import load_grid
from mighty_mouse.paths import load_paths
g = load_grid(sys.argv[1])
print(f"{load_paths()['results_dir']}/{g.out_subdir}/{g.name}/_queue")
EOF
)
mkdir -p "$Q"
echo "host=$(hostname) partition=${SLURM_JOB_PARTITION:-?} gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) cells=$#"
echo "mouse-pose @ $(git rev-parse --short HEAD); lightning-pose: $(git -C ../lightning-pose rev-parse --abbrev-ref HEAD)" \
     "@ $(git -C ../lightning-pose rev-parse --short HEAD)"

read -ra gpus <<< "${CUDA_VISIBLE_DEVICES//,/ }"
[ "${#gpus[@]}" -gt 0 ] || gpus=(0)
cells=("$@"); pids=()
for i in "${!cells[@]}"; do
    CUDA_VISIBLE_DEVICES=${gpus[$((i % ${#gpus[@]}))]} python scripts/adapt/run_cell.py --config "$CFG" \
        --cell "${cells[$i]}" >> "$Q/${cells[$i]}.out" 2>&1 &
    pids+=($!)
    echo "${cells[$i]}: GPU ${gpus[$((i % ${#gpus[@]}))]}, pid $! -> $Q/${cells[$i]}.out"
done
status=0
for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then echo "${cells[$i]}: done"
    else echo "${cells[$i]}: FAILED (see $Q/${cells[$i]}.out)"; status=1; fi
done
exit $status
