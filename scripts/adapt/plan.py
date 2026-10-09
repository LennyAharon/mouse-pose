#!/usr/bin/env python
"""
Expand an adaptation grid config into cells, check prerequisites, and write the job list.

    python scripts/adapt/plan.py --config configs/adaptation/cosyne-2026.yaml   # summary
    python scripts/adapt/plan.py --config ... --jobs <file>    # pending runnable ids, one per line
    python scripts/adapt/plan.py --config ... --status         # done / pending per dataset x arm
    python scripts/adapt/plan.py --config ... --bundle <file>  # result files another machine needs

A cell is done when its directory has `.done`; it is runnable when its trunk finished training
(arms that start from DINOv3 need nothing). The job file feeds scripts/adapt/run_local.sh (this
machine) or scripts/adapt/slurm_array.sbatch (a SLURM cluster such as ACCESS): one task per line.
The bundle file lists, relative to results_dir, every file of the results tree the grid reads
(per trunk: config.yaml, train_status.json, its training csv, best checkpoint, test predictions;
per dedicated model: test predictions), for `rsync -a --files-from=<file> <results_dir>/ ...`.
"""

import argparse
import collections
import json
from pathlib import Path

import yaml

from mighty_mouse.adaptation import (
    INIT_TRUNK,
    best_checkpoint,
    cell_dir,
    expand,
    load_grid,
    trained_keypoints,
    trunk_datasets,
    trunk_problem,
)
from mighty_mouse.paths import load_paths


def bundle_files(results: Path, grid) -> list[str]:
    """Files of the results tree the grid reads, relative to ``results`` (existing ones only)."""
    files = []
    for ds, spec in grid.datasets.items():
        t = results / spec["trunk"]
        if trunk_problem(t):
            continue
        csv = yaml.safe_load((t / "config.yaml").read_text())["data"]["csv_file"]
        files += [t / "config.yaml", t / "train_status.json", t / csv, best_checkpoint(t),
                  t / "eval" / ds / "predictions.csv"]
    for pattern in grid.dedicated.values():
        files += [results / pattern.format(ds=ds) / "eval" / ds / "predictions.csv"
                  for ds in grid.datasets]
    return sorted({str(f.relative_to(results)) for f in files if f.exists()})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1], epilog=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--jobs",   default=None, type=Path, help="write pending runnable ids here")
    ap.add_argument("--status", action="store_true", help="done / pending per dataset x arm")
    ap.add_argument("--bundle", default=None, type=Path, help="write the result files needed here")
    args = ap.parse_args()

    grid    = load_grid(args.config)
    paths   = load_paths()
    results = Path(paths["results_dir"])
    inv     = json.loads((Path(paths["data_dir"]) / "dataset_inventory.json").read_text())
    inv     = inv["datasets"]
    cells   = expand(grid)

    # ── prerequisites ──────────────────────────────────────────────────────────
    trunk_why = {ds: trunk_problem(results / s["trunk"]) for ds, s in grid.datasets.items()}
    trunk_ok  = {ds: why is None for ds, why in trunk_why.items()}
    missing_trunks = [f"{ds} ({why})" for ds, why in sorted(trunk_why.items()) if why]
    missing_ded = []
    for head, pattern in grid.dedicated.items():
        for ds in grid.datasets:
            if not (results / pattern.format(ds=ds) / "eval" / ds / "predictions.csv").exists():
                missing_ded.append(f"{ds} ({head})")

    # a masked setting must hide keypoints the target labels and its trunk trains
    bad_masks = []
    for ds, settings in (grid.masked.get("settings") or {}).items():
        trained = None
        if trunk_ok[ds]:
            t = results / grid.datasets[ds]["trunk"]
            trained = trained_keypoints(trunk_datasets(t, yaml.safe_load(
                (t / "config.yaml").read_text())), inv)
        for kps in settings:
            unlabelled = [k for k in kps if k not in inv[ds]["direct"]]
            untrained  = [k for k in kps if trained is not None and k not in trained]
            if unlabelled:
                bad_masks.append(f"{ds} {kps}: {unlabelled} not labelled by {ds}")
            if untrained:
                bad_masks.append(f"{ds} {kps}: {untrained} not trained by its trunk")

    done, runnable, blocked = [], [], []
    for c in cells:
        if (cell_dir(results, grid, c) / ".done").exists():
            done.append(c)
        elif grid.arms[c.arm]["init"] == INIT_TRUNK and not trunk_ok[c.dataset]:
            blocked.append(c)
        else:
            runnable.append(c)

    n_curve = sum(c.kind == "curve" for c in cells)
    print(f"grid {grid.name}: {len(cells)} cells ({n_curve} curve, {len(cells) - n_curve} "
          f"masked); done {len(done)}, runnable {len(runnable)}, blocked {len(blocked)}")
    print(f"  trunks not ready: {missing_trunks or 'none'}")
    print(f"  dedicated models not evaluated yet (reference lines): {missing_ded or 'none'}")
    for b in bad_masks:
        print(f"  MASKED SETTING INVALID: {b}")

    if args.status:
        tab = collections.defaultdict(lambda: [0, 0])
        for c in cells:
            key = (c.dataset, c.arm + (" (masked)" if c.mask else ""))
            tab[key][0 if c in done else 1] += 1
        print(f"\n{'dataset':14s} {'arm':30s} done / pending")
        for (ds, arm), (d, p) in sorted(tab.items()):
            print(f"{ds:14s} {arm:30s} {d:4d} / {p}")

    if args.jobs:
        args.jobs.parent.mkdir(parents=True, exist_ok=True)
        args.jobs.write_text("".join(c.id + "\n" for c in runnable))
        print(f"\nwrote {len(runnable)} runnable cell ids -> {args.jobs}")

    if args.bundle:
        files = bundle_files(results, grid)
        args.bundle.write_text("".join(f + "\n" for f in files))
        print(f"wrote {len(files)} result files (relative to {results}) -> {args.bundle}")


if __name__ == "__main__":
    main()
