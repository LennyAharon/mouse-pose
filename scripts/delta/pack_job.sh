#!/bin/bash
# Run several recipe-of-record runs side by side in ONE multi-GPU SLURM job (one run per GPU), so an
# interactive partition's single job slot trains N models at once. Submitted by
# scripts/delta/interactive_feeder.sh; each run goes through scripts/delta/train_plan_job.sbatch
# (claim, train_sweep, zero-shot), with its own GPU (CUDA_VISIBLE_DEVICES) and its own share of the
# job's CPUs (SLURM_CPUS_PER_TASK, which Lightning Pose uses as its dataloader worker count).
#
#   sbatch -p <part> --gpus-per-node=N --cpus-per-task=$((16*N)) ... scripts/delta/pack_job.sh <spec file>
#
# Spec file: one run per line,
# `<name>|<run dir>|<left out>|<train_sweep args>[|<mouse-pose dir>:<lightning-pose dir>[|<cpus>]]`
# (train_plan.sh args / exp_specs.py). With the optional 5th field the run uses that code: the job script of
# that mouse-pose checkout, run from it, with both checkouts first on PYTHONPATH (experiment worktrees).
# The optional 6th field gives that run its own CPU count (its dataloader workers; e.g. 32 for a dataset
# with large frames, whose augmentation is the bottleneck); the other runs share the rest equally. Leave
# the 5th field empty for baseline code: `...|<train_sweep args>||32`. An optional 7th field
# (allowed partitions, from exp_specs.py) is for the feeder and ignored here.
set -u
set -f   # no globbing: train_sweep args can hold Hydra lists like milestone_steps=[8000,10000,11000]
SPEC="$1"
cd /work/nvme/beez/laharon/MM_project/mouse-pose
LOGDIR=$(dirname "$SPEC")/..
mapfile -t LINES < "$SPEC"
n_gpu=$(nvidia-smi -L | wc -l)
fixed=0; n_free=0   # CPUs of runs with their own count; runs sharing the rest
for line in "${LINES[@]}"; do
    IFS='|' read -r _ _ _ _ _ cpus _ <<< "$line"
    if [ -n "${cpus:-}" ]; then fixed=$((fixed + cpus)); else n_free=$((n_free + 1)); fi
done
per=$(( n_free > 0 ? (${SLURM_CPUS_PER_TASK:-16} - fixed) / n_free : 0 ))
echo "host=$(hostname) partition=${SLURM_JOB_PARTITION:-?} gpus=$n_gpu runs=${#LINES[@]} cpus/run=$per (own counts: $fixed)"
pids=(); names=()
for i in "${!LINES[@]}"; do
    IFS='|' read -r name dir out args code cpus _ <<< "${LINES[$i]}"
    ncpu=${cpus:-$per}
    log="$LOGDIR/slurm/${name}_${SLURM_JOB_ID:-local}.out"
    job=scripts/delta/train_plan_job.sbatch; extra_env=()
    if [ -n "${code:-}" ]; then   # experiment worktrees: their job script, their code first on the path
        job="${code%%:*}/scripts/delta/train_plan_job.sbatch"
        extra_env=(MM_REPO="${code%%:*}" PYTHONPATH="${code#*:}:${code%%:*}")
    fi
    # shellcheck disable=SC2086  # args is a list of train_sweep options
    env "${extra_env[@]}" CUDA_VISIBLE_DEVICES=$(( i % n_gpu )) SLURM_CPUS_PER_TASK=$ncpu \
        bash "$job" "$dir" "$out" $args > "$log" 2>&1 &
    pids+=($!); names+=("$name")
    echo "$name -> GPU $(( i % n_gpu )), $ncpu CPUs, log $log"
done
status=0
for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then echo "${names[$i]}: done"; else echo "${names[$i]}: FAILED"; status=1; fi
done
exit $status
