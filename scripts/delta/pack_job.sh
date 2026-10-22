#!/bin/bash
# Run several recipe-of-record runs side by side in ONE multi-GPU SLURM job (one run per GPU), so an
# interactive partition's single job slot trains N models at once. Submitted by
# scripts/delta/interactive_feeder.sh; each run goes through scripts/delta/train_plan_job.sbatch
# (claim, train_sweep, zero-shot), with its own GPU (CUDA_VISIBLE_DEVICES) and its own share of the
# job's CPUs (SLURM_CPUS_PER_TASK, which Lightning Pose uses as its dataloader worker count).
#
#   sbatch -p <part> --gpus-per-node=N --cpus-per-task=$((16*N)) ... scripts/delta/pack_job.sh <spec file>
#
# Spec file: one run per line, `<name>|<run dir>|<left out>|<train_sweep args>` (train_plan.sh args).
set -u
set -f   # no globbing: train_sweep args can hold Hydra lists like milestone_steps=[8000,10000,11000]
SPEC="$1"
cd /work/nvme/beez/laharon/MM_project/mouse-pose
LOGDIR=$(dirname "$SPEC")/..
mapfile -t LINES < "$SPEC"
n_gpu=$(nvidia-smi -L | wc -l)
per=$(( ${SLURM_CPUS_PER_TASK:-16} / ${#LINES[@]} ))
echo "host=$(hostname) partition=${SLURM_JOB_PARTITION:-?} gpus=$n_gpu runs=${#LINES[@]} cpus/run=$per"
pids=(); names=()
for i in "${!LINES[@]}"; do
    IFS='|' read -r name dir out args <<< "${LINES[$i]}"
    log="$LOGDIR/slurm/${name}_${SLURM_JOB_ID:-local}.out"
    # shellcheck disable=SC2086  # args is a list of train_sweep options
    CUDA_VISIBLE_DEVICES=$(( i % n_gpu )) SLURM_CPUS_PER_TASK=$per \
        bash scripts/delta/train_plan_job.sbatch "$dir" "$out" $args > "$log" 2>&1 &
    pids+=($!); names+=("$name")
    echo "$name -> GPU $(( i % n_gpu )), log $log"
done
status=0
for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then echo "${names[$i]}: done"; else echo "${names[$i]}: FAILED"; status=1; fi
done
exit $status
