#!/usr/bin/env python
"""
Zero-shot vs number of training datasets: write the experiment definition of random dataset subsets
and build their training CSVs.

For every target dataset and draw d, the other registry datasets (minus the target's sibling: same
rig and mice, so no subset with it is zero-shot) are shuffled once with seed (d, target index);
the subsets are the first k of that order for each --sizes k, so they are nested (3 of 6 of 9).
The largest point of the curve is the existing leave-one-out trunk. Each subset becomes a target
`<dataset>-k<k>-d<d>: {keep: [...], zeroshot: [<dataset>]}` of the definition (scripts/delta/
exp_specs.py); a subset drawn for several targets is trained once (one entry, zero-shot on all of
them). Every trunk is evaluated on every test set anyway, so the other left-out datasets give
extra zero-shot points. Training CSVs: `build_dataset.py --tag <tag> --datasets ... --n_frames -1`
(all frames, as for the leave-one-out tags).

    python scripts/scaling_subsets.py --out configs/experiments/v11-scaling.yaml --build
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

from mighty_mouse.paths import load_paths
from mighty_mouse.registry import load_registry

REPO     = Path(__file__).resolve().parents[1]
PAIRS    = [("cheese-2d", "cheese-3d"), ("kaufman", "kaufman-water"),
            ("aind-vbn-body", "aind-vbn-face")]
SIBLINGS = {**dict(PAIRS), **{b: a for a, b in PAIRS}}   # same rig and mice
HEADER   = """\
# Zero-shot vs number of training datasets (user, 2026-10-26), written by
# scripts/scaling_subsets.py (do not edit by hand). Per target dataset and draw d: the other
# datasets minus the target's sibling, shuffled once (seed d, target index); the first 3 / 6 / 9
# are nested random subsets; 3 draws. The 12-dataset point is the existing leave-one-out trunk
# (experiments/R384-E5). Same recipe as those trunks: E5 at 384 px, nonlinear head, seed 0.
# Runs: <results_dir>/experiments/SCALE/<tag>_train/;
# zero-shot on the target's train frames, and every trunk is evaluated on all 13 test sets.
# Order: draw 0 for every target first, then draw 1, then draw 2.
"""


def subsets(datasets: list[str], sizes: list[int], draws: int) -> dict[str, dict]:
    """Definition targets: name -> {keep, zeroshot}, draw-major, nested per (target, draw)."""
    out, by_keep = {}, {}
    for d in range(draws):
        for k in sizes:
            for i, t in enumerate(datasets):
                pool = [x for x in datasets if x not in (t, SIBLINGS.get(t))]
                order = [str(x) for x in np.random.default_rng([d, i]).permutation(pool)]
                keep = tuple(sorted(order[:k], key=datasets.index))
                if keep in by_keep:                       # same subset drawn for another target
                    out[by_keep[keep]]["zeroshot"].append(t)
                    continue
                name = f"{t}-k{k}-d{d}"
                out[name] = {"keep": list(keep), "zeroshot": [t]}
                by_keep[keep] = name
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1], epilog=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out",     type=Path, required=True, help="experiment definition to write")
    ap.add_argument("--sizes",   default="3,6,9")
    ap.add_argument("--draws",   type=int, default=3)
    ap.add_argument("--build",   action="store_true", help="build the missing training CSVs")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    datasets = load_registry()
    short = dict(kv.split("=") for kv in subprocess.run(
        ["bash", "-c", "grep -oE '\\[[a-z0-9-]+\\]=[a-z0-9]+' scripts/train_plan.sh | tr -d '[]'"],
        cwd=REPO, capture_output=True, text=True).stdout.split())
    targets = subsets(datasets, [int(k) for k in args.sizes.split(",")], args.draws)
    spec = {
        "config": "configs/model_zoominout.yaml", "backbone": "vits_dinov3",
        "head_overrides": ["+model.head_hidden_channels=256", "+training.resume_every_n_steps=500",
                           "training.unfreezing_step=3000", "training.min_steps=14000",
                           "training.max_steps=14000",
                           "training.lr_scheduler_params.multisteplr.milestone_steps=[6000,8000,10000]",
                           "data.image_resize_dims.height=384",
                           "data.image_resize_dims.width=384"],
        "targets": targets,
        "experiments": {"SCALE": {"seeds": [0], "overrides": []}},
    }
    data_dir = Path(load_paths()["data_dir"])
    tags = {"+".join(short[d] for d in e["keep"]): e["keep"] for e in targets.values()}
    missing = {t: ks for t, ks in tags.items()
               if not (data_dir / f"CollectedData_{t}_train.csv").exists()}
    points = sum(len(e["zeroshot"]) for e in targets.values())
    print(f"{len(targets)} trunks ({points} target points), {len(missing)} training csvs to build")
    if args.dry_run:
        for n, e in list(targets.items())[:5]:
            print(f"  {n}: {e}")
        return
    args.out.write_text(HEADER + "\n" + yaml.safe_dump(spec, sort_keys=False, width=200))
    print(f"wrote {args.out}")
    if args.build:
        for tag, keep in missing.items():
            r = subprocess.run([sys.executable, "scripts/build_dataset.py", "--tag", tag,
                                "--datasets", *keep, "--n_frames", "-1"], cwd=REPO,
                               capture_output=True, text=True)
            print(f"  {'built' if r.returncode == 0 else 'FAILED'} {tag}")
            if r.returncode:
                sys.exit(r.stderr[-2000:])


if __name__ == "__main__":
    main()
