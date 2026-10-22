#!/bin/bash
# Train Lightning Pose models on NCSA Delta: one SLURM array task per seed.
#
# Used for the 3-model ensembles of skills/active-pseudo-labeling (seeds 0/1/2, only the seed
# and output dir differ) and for any standalone config (skills/train-lightning-pose-model).
#
#   TAG=$(date +%Y-%m-%d_%H-%M-%S)_round0
#   sbatch --array=0-2 -o $PI/results/ibl/slurm/%x_%A_%a.out \
#     scripts/delta/train_lp.sbatch $PI/configs/ibl.yaml $PI/results/ibl/$TAG
#
# where PI=/work/nvme/beez/laharon/MM_project/poseinterface. Task i writes <prefix>_seed-i/ and
# trains with training.rng_seed_data_pt=i. Any arguments after the prefix are passed on as
# extra Hydra overrides, e.g. training.min_epochs=2 training.max_epochs=2 for a smoke test.
# The -o directory must exist before submitting (SLURM does not create it).
#
# Delta's GPU queues are long, so small models are better run as ONE job: SEEDS="0 1 2" trains
# every listed seed at the same time on the job's single GPU (each logs to <prefix>_seed-i.log),
# so only one job waits in the queue. Submit without --array:
#
#   sbatch --export=ALL,SEEDS="0 1 2" -o ... scripts/delta/train_lp.sbatch <config> <prefix>
#
# -p overrides the partition list below; listing several lets SLURM use whichever frees up first.
#
# Lightning Pose is imported from whatever branch is checked out in MM_project/lightning-pose
# (editable install), so don't switch branches there while jobs are queued or running.
#SBATCH --job-name=lp-train
#SBATCH --account=beez-delta-gpu
#SBATCH --partition=gpuA40x4,gpuA100x4,gpuA100x8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=04:00:00
set -eo pipefail

CONFIG="$1"; PREFIX="$2"; shift 2
SEEDS="${SEEDS:-${SLURM_ARRAY_TASK_ID:-0}}"

ENV=/work/nvme/beez/laharon/.conda/envs/mighty-mouse
export PATH="$ENV/bin:$PATH"
export HF_HOME=/work/nvme/beez/laharon/.cache/huggingface

echo "host=$(hostname) partition=$SLURM_JOB_PARTITION gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) seeds=$SEEDS"
echo "lightning-pose: $(git -C /work/nvme/beez/laharon/MM_project/lightning-pose rev-parse --abbrev-ref HEAD)" \
     "@ $(git -C /work/nvme/beez/laharon/MM_project/lightning-pose rev-parse --short HEAD)"
echo "config=$CONFIG prefix=$PREFIX extra=$*"

read -ra seeds <<< "$SEEDS"
if [ "${#seeds[@]}" -eq 1 ]; then
    litpose train "$CONFIG" --output_dir "${PREFIX}_seed-${seeds[0]}" \
        --overrides "training.rng_seed_data_pt=${seeds[0]}" "$@"
    exit
fi

pids=()
for s in "${seeds[@]}"; do
    litpose train "$CONFIG" --output_dir "${PREFIX}_seed-$s" \
        --overrides "training.rng_seed_data_pt=$s" "$@" > "${PREFIX}_seed-$s.log" 2>&1 &
    pids+=($!)
    echo "seed $s: pid $! -> ${PREFIX}_seed-$s.log"
done
status=0
for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then echo "seed ${seeds[$i]}: done"
    else echo "seed ${seeds[$i]}: FAILED (see ${PREFIX}_seed-${seeds[$i]}.log)"; status=1; fi
done
exit $status
