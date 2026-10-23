#!/usr/bin/env python
"""
Print the runs of an experiment definition (configs/experiments/<name>.yaml) that have not started yet,
one per line as `<job name>|<run dir>|<left out>|<train_sweep.py args>` (the format of
`train_plan.sh args`), for scripts/delta/interactive_feeder.sh --specs and scripts/delta/pack_job.sh.
A definition's `code:` adds a 5th field (experiment checkouts) and `partitions:` a 7th (allowed partitions).

Each run is the baseline train_sweep command (recipe S, shared head, T=2, the definition's config and
head_overrides) plus the experiment's overrides, in <results_dir>/experiments/<id>/ (or with the baseline
in trunks-headmlp256/ for `area: baseline`), for every target of the definition or the experiment's own
`targets:` subset. A run whose directory exists (started or done) is skipped.

    python scripts/delta/exp_specs.py configs/experiments/v10-schedule.yaml > <spec file>
    python scripts/delta/exp_specs.py configs/experiments/v10-schedule.yaml --all    # include started runs
"""

import argparse
import re
import subprocess
from pathlib import Path

import yaml

from mighty_mouse.paths import load_paths
from mighty_mouse.registry import load_registry

REPO = Path(__file__).resolve().parents[2]
LEAF = "supervised/sampling-T2/tf1"


def resumable(run_dir: Path) -> bool:
    """A started run that can continue: rolling checkpoint, no evaluation, no live job holding it."""
    if not (run_dir / "resume" / "resume.ckpt").exists() or any(run_dir.glob("eval/*/predictions.csv")):
        return False
    claim = run_dir.parent / f"{run_dir.name}.claim" / "job"
    if not claim.exists():
        return True
    out = subprocess.run(["sacct", "-j", claim.read_text().strip(), "-X", "-n", "-o", "State"],
                         capture_output=True, text=True).stdout.split()
    ended = {"COMPLETED", "FAILED", "TIMEOUT", "PREEMPTED", "NODE_FAIL", "OUT_OF_MEMORY",
             "BOOT_FAIL", "DEADLINE"}
    return bool(out) and (out[0] in ended or out[0].startswith("CANCELLED"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1], epilog=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("definition", type=Path)
    ap.add_argument("--all", action="store_true", help="also list runs whose directory exists")
    args = ap.parse_args()

    spec = yaml.safe_load(args.definition.read_text())
    results = Path(load_paths()["results_dir"])
    datasets = load_registry()
    short = dict(re.findall(r"\[([\w-]+)\]=(\w+)", (REPO / "scripts" / "train_plan.sh").read_text()))
    head_ov = list(spec.get("head_overrides", []))
    # optional experiment code (worktrees): appended as a 5th field that scripts/delta/pack_job.sh honors
    code = spec.get("code")
    code_field = f"|{code['mouse_pose']}:{code['lightning_pose']}" if code else ""
    # optional partitions the runs may use (e.g. A100 only for runs too long for A40 1-hour slots):
    # a 7th field after the code and an empty cpus field, honored by interactive_feeder.sh
    parts = spec.get("partitions")
    if parts:
        code_field = f"{code_field or '|'}||{','.join(parts)}"
    for exp_id, exp in spec["experiments"].items():
        root = (results / "trunks-headmlp256" if exp.get("area") == "baseline"
                else results / "experiments" / exp_id)
        targets = exp.get("targets", list(spec["targets"]))   # an experiment may use a subset
        unknown = set(targets) - set(spec["targets"])
        if unknown:
            raise SystemExit(f"{exp_id}: targets {sorted(unknown)} are not in the definition's targets")
        for target in targets:
            left_out = spec["targets"][target]
            keep = [d for d in datasets if d not in left_out]
            tag = "+".join(short[d] for d in keep)
            if not (Path(load_paths()["data_dir"]) / f"CollectedData_{tag}_train.csv").exists():
                raise SystemExit(f"no training csv for tag {tag} (target {target}); build it first")
            for seed in exp.get("seeds", [0]):
                run_dir = root / f"{tag}_train" / LEAF / spec["backbone"] / f"seed{seed}"
                if run_dir.exists() and not args.all and not resumable(run_dir):
                    continue
                overrides = ";".join(head_ov + list(exp.get("overrides", [])))
                sweep = (f"--config_file {spec['config']} --csv_files CollectedData_{tag}_train.csv "
                         f"--train_frames 1 --seeds {seed} --backbones {spec['backbone']} "
                         f"--sampling_temperatures 2 --head_modes shared --keep_checkpoints --skip_existing "
                         f"--output_root {root} --extra_overrides {overrides}")
                print(f"mm-{exp_id}-{target}-s{seed}|{run_dir}|{','.join(left_out)}|{sweep}{code_field}")


if __name__ == "__main__":
    main()
