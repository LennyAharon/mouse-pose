#!/usr/bin/env python
"""
Print the runs of an experiment definition (configs/experiments/<name>.yaml) that have not started yet,
one per line as `<job name>|<run dir>|<left out>|<train_sweep.py args>` (the format of
`train_plan.sh args`), for scripts/delta/interactive_feeder.sh --specs and scripts/delta/pack_job.sh.

Each run is the baseline train_sweep command (recipe S, shared head, T=2, the definition's config and
head_overrides) plus the experiment's overrides, in <results_dir>/experiments/<id>/ (or with the baseline
in trunks-headmlp256/ for `area: baseline`). A run whose directory exists (started or done) is skipped.

    python scripts/delta/exp_specs.py configs/experiments/v10-schedule.yaml > <spec file>
    python scripts/delta/exp_specs.py configs/experiments/v10-schedule.yaml --all    # include started runs
"""

import argparse
import re
from pathlib import Path

import yaml

from mighty_mouse.paths import load_paths
from mighty_mouse.registry import load_registry

REPO = Path(__file__).resolve().parents[2]
LEAF = "supervised/sampling-T2/tf1"


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
    for exp_id, exp in spec["experiments"].items():
        root = (results / "trunks-headmlp256" if exp.get("area") == "baseline"
                else results / "experiments" / exp_id)
        for target, left_out in spec["targets"].items():
            keep = [d for d in datasets if d not in left_out]
            tag = "+".join(short[d] for d in keep)
            if not (Path(load_paths()["data_dir"]) / f"CollectedData_{tag}_train.csv").exists():
                raise SystemExit(f"no training csv for tag {tag} (target {target}); build it first")
            for seed in exp.get("seeds", [0]):
                run_dir = root / f"{tag}_train" / LEAF / spec["backbone"] / f"seed{seed}"
                if run_dir.exists() and not args.all:
                    continue
                overrides = ";".join(head_ov + list(exp.get("overrides", [])))
                sweep = (f"--config_file {spec['config']} --csv_files CollectedData_{tag}_train.csv "
                         f"--train_frames 1 --seeds {seed} --backbones {spec['backbone']} "
                         f"--sampling_temperatures 2 --head_modes shared --keep_checkpoints --skip_existing "
                         f"--output_root {root} --extra_overrides {overrides}")
                print(f"mm-{exp_id}-{target}-s{seed}|{run_dir}|{','.join(left_out)}|{sweep}")


if __name__ == "__main__":
    main()
