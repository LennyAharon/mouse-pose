#!/usr/bin/env python
"""
Zero-shot vs number of training datasets: write the experiment definition of random dataset subsets
and build their training CSVs.

For each draw, the registry datasets are shuffled once and the subsets are the first k
of that order for each --sizes k, so they are nested (3 of 6 of 9). One trunk per (k, d): 3 sizes
x 3 draws = 9 trunks. Every trunk is evaluated zero-shot on every dataset it did not train on (test
frames for all of them, train frames too: `zeroshot`), so the 9 trunks give points for all targets.
A held-out dataset whose sibling (same rig and mice) is in the subset is not a clean zero-shot
point: the scoring drops it. The largest point of the curve is the existing leave-one-out trunks.
Training CSVs: `build_dataset.py --tag <tag> --datasets ... --n_frames -1` (all frames, as for the
leave-one-out tags). The draws' shuffle seeds are picked (greedily, among 0-199) to give as many
datasets as possible a clean zero-shot point at every size; they are written into the definition.

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
HEADER   = """\
# Zero-shot vs number of training datasets (user, 2026-10-26), written by
# scripts/scaling_subsets.py (do not edit by hand). Per draw: the 13 datasets shuffled once;
# the first 3 / 6 / 9 are nested random subsets; 3 draws = 9 trunks. Each trunk is
# evaluated zero-shot on every dataset it did not train on. The 12-dataset points are the existing
# leave-one-out trunks (experiments/R384-E5). Same recipe as those trunks: E5 at 384 px, nonlinear
# head, seed 0. Runs: <results_dir>/experiments/SCALE/<tag>_train/.
"""


PAIRS    = [("cheese-2d", "cheese-3d"), ("kaufman", "kaufman-water"),
            ("aind-vbn-body", "aind-vbn-face")]
SIBLINGS = {**dict(PAIRS), **{b: a for a, b in PAIRS}}   # same rig and mice


def nested(datasets: list[str], sizes: list[int], seed: int) -> dict[int, list[str]]:
    """Nested subsets of each size: the first k datasets of one shuffle (seed)."""
    order = [str(x) for x in np.random.default_rng(seed).permutation(datasets)]
    return {k: sorted(order[:k], key=datasets.index) for k in sizes}


def coverage(draws: list[dict[int, list[str]]], datasets: list[str]) -> tuple:
    """How well a set of draws covers the targets: per size, the datasets with at least one clean
    zero-shot point (held out, sibling not trained), then the fewest clean points any dataset gets
    over all sizes; draws with a repeated subset score lowest."""
    if len({tuple(s) for d in draws for s in d.values()}) < sum(len(d) for d in draws):
        return (-1,)
    sizes = list(draws[0])
    clean = {x: [sum(x not in d[k] and SIBLINGS.get(x) not in d[k] for d in draws) for k in sizes]
             for x in datasets}
    per_size = sum(sum(c[i] > 0 for c in clean.values()) for i in range(len(sizes)))
    return (per_size, min(sum(c) for c in clean.values()))


def subsets(datasets: list[str], sizes: list[int], draws: int, search: int = 200) -> tuple:
    """Definition targets ``k<k>-d<d>`` -> {keep, zeroshot}, draw-major, nested per draw, and the
    shuffle seeds used: the `draws` seeds among range(search) (greedy, one at a time) whose subsets
    give the most datasets clean zero-shot points at every size."""
    seeds: list[int] = []
    for _ in range(draws):
        best = max((s for s in range(search) if s not in seeds),
                   key=lambda s: coverage([nested(datasets, sizes, x) for x in [*seeds, s]],
                                          datasets))
        seeds.append(best)
    out = {}
    for d, seed in enumerate(seeds):
        for k, keep in nested(datasets, sizes, seed).items():
            out[f"k{k}-d{d}"] = {"keep": keep, "zeroshot": [x for x in datasets if x not in keep]}
    return out, seeds


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
    targets, seeds = subsets(datasets, [int(k) for k in args.sizes.split(",")], args.draws)
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
    print(f"{len(targets)} trunks ({points} zero-shot points), {len(missing)} csvs to build")
    if args.dry_run:
        for n, e in list(targets.items())[:5]:
            print(f"  {n}: {e}")
        return
    head = HEADER + f"# Shuffle seeds of the 3 draws (picked for coverage): {seeds}.\n"
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
