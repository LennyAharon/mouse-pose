#!/usr/bin/env python
"""
Zero-shot vs number of training datasets: write the experiment definition of random dataset subsets
and build their training CSVs.

For each --targets dataset, the other registry datasets minus the target's sibling (same rig and
mice, so a subset with it would not be zero-shot) are shuffled once per draw; the subsets are the
first k of that order for each --sizes k, so they are nested (3 of 6 of 9). One trunk per (target,
k, draw), zero-shot on its target (test frames, and its train frames: `zeroshot`). The largest
point of each curve is the existing leave-one-out trunk (both-out for a target with a sibling).
Draw seeds: the first --draws seeds (from 0) whose 3-dataset subsets differ for that target; they
are written into the definition. Training CSVs: `build_dataset.py --tag <tag> --datasets ...
--n_frames -1` (all frames, as for the leave-one-out tags). The definition sets
`drop_checkpoints: true`: scripts/delta/queue_optimizer.py deletes a finished run's checkpoints
once its evaluation and zero-shot predictions exist.

    python scripts/scaling_subsets.py --targets ibl cheese-3d kaufman \\
        --out configs/experiments/v11-scaling.yaml --build
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
# scripts/scaling_subsets.py (do not edit by hand). Per target: the other datasets minus the
# target's sibling, shuffled once per draw; the first 3 / 6 / 9 are nested random subsets; 3 draws.
# The top point of each curve is the existing leave-one-out trunk (experiments/R384-E5; both-out
# for cheese-3d and kaufman). Same recipe as those trunks: E5 at 384 px, nonlinear head, seed 0.
# Runs: <results_dir>/experiments/SCALE/<tag>_train/; checkpoints deleted after evaluation.
"""


def nested(pool: list[str], order_ref: list[str], sizes: list[int], seed: list[int]) -> dict:
    """Nested subsets of each size: the first k datasets of one shuffle of the pool."""
    order = [str(x) for x in np.random.default_rng(seed).permutation(pool)]
    return {k: sorted(order[:k], key=order_ref.index) for k in sizes}


def subsets(datasets: list[str], targets: list[str], sizes: list[int], draws: int) -> tuple:
    """Definition targets ``<target>-k<k>-d<d>`` -> {keep, zeroshot}, and the draw seeds."""
    out, seeds = {}, {}
    for t in targets:
        ti   = datasets.index(t)
        pool = [x for x in datasets if x not in (t, SIBLINGS.get(t))]
        got, smallest = [], set()
        for s in range(1000):
            sub = nested(pool, datasets, sizes, [s, ti])
            if tuple(sub[min(sizes)]) not in smallest:
                got.append((s, sub))
                smallest.add(tuple(sub[min(sizes)]))
            if len(got) == draws:
                break
        seeds[t] = [s for s, _ in got]
        for d, (_, sub) in enumerate(got):
            for k, keep in sub.items():
                same = [e for e in out.values() if e["keep"] == keep]
                if same:                          # drawn for another target too: train once
                    same[0]["zeroshot"].append(t)
                else:
                    out[f"{t}-k{k}-d{d}"] = {"keep": keep, "zeroshot": [t]}
    return out, seeds


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1], epilog=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--targets", nargs="+", required=True, help="zero-shot target datasets")
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
    sizes = [int(k) for k in args.sizes.split(",")]
    targets, seeds = subsets(datasets, args.targets, sizes, args.draws)
    spec = {
        "config": "configs/model_zoominout.yaml", "backbone": "vits_dinov3",
        "drop_checkpoints": True,
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
    print(f"{len(targets)} trunks ({len(tags)} distinct subsets), {len(missing)} csvs to build; "
          f"draw seeds {seeds}")
    if args.dry_run:
        for n, e in targets.items():
            print(f"  {n:22s} {', '.join(e['keep'])}")
        return
    head = HEADER + f"# Draw seeds per target ([seed, target index]): {seeds}.\n"
    args.out.write_text(head + "\n" + yaml.safe_dump(spec, sort_keys=False, width=200))
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
