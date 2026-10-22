#!/bin/bash
# Submit runs given as spec lines (scripts/delta/exp_specs.py, `train_plan.sh args`) as one SLURM job
# each, e.g. as regular / preempt backups of runs the interactive feeder also offers: every job goes
# through scripts/delta/train_plan_job.sbatch, whose per-run claim lets only the first one train.
# sbatch options (partitions, time) are passed through. A 5th field `<mouse-pose dir>:<lightning-pose
# dir>` runs that checkout's job script with both checkouts first on PYTHONPATH (as pack_job.sh).
#
#   python scripts/delta/exp_specs.py <definition> | scripts/delta/submit_specs.sh \
#       -p gpuA40x4-preempt,gpuA100x4-preempt -t 01:30:00
set -u
set -f   # no globbing: train_sweep args can hold Hydra lists like milestone_steps=[8000,10000,11000]
cd "$(dirname "$0")/../.."
export PATH=/work/nvme/beez/laharon/.conda/envs/mighty-mouse/bin:$PATH
RESULTS=$(python -c "from mighty_mouse.paths import load_paths; print(load_paths()['results_dir'])")
mkdir -p "$RESULTS/_logs/slurm"
while IFS='|' read -r name dir out args code; do
    [ -n "$name" ] || continue
    job=scripts/delta/train_plan_job.sbatch; export_env=ALL
    if [ -n "${code:-}" ]; then   # experiment worktrees: their job script, their code first on the path
        job="${code%%:*}/scripts/delta/train_plan_job.sbatch"
        export_env="ALL,MM_REPO=${code%%:*},PYTHONPATH=${code#*:}:${code%%:*}"
    fi
    # shellcheck disable=SC2086  # args is a list of train_sweep options
    id=$(sbatch --parsable -J "$name" -o "$RESULTS/_logs/slurm/%x_%j.out" --export="$export_env" "$@" \
         "$job" "$dir" "$out" $args 2>&1)
    echo "submitted $name: job $id"
done
