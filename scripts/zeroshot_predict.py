"""Forwarder for SLURM jobs queued before the 2026-11-06 file reorganization (their copy of
scripts/delta/train_plan_job.sbatch calls this old path): runs scripts/evaluation/zeroshot_predict.py with the same
arguments. Delete once no such job is queued (squeue -u $USER)."""
import runpy
import sys
from pathlib import Path

target = Path(__file__).resolve().parent / "evaluation" / "zeroshot_predict.py"
sys.argv[0] = str(target)
runpy.run_path(str(target), run_name="__main__")
