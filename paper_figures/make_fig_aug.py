#!/usr/bin/env python
"""
Appendix figure: per-dataset zoom augmentation and cross-rig transfer.

Pixel error against ensemble standard deviation, in the notebook's house style. Every model
plotted is a *leave-one-out* model that has never seen the evaluation dataset, so this is the
zero-shot question the augmentation was designed for: does training each source across the
corpus-wide range of apparent scale improve transfer to an unseen rig? Three augmentation
generations are compared, with that dataset's dedicated model as the reference a laboratory
could reach with its own full annotation budget.

Only supported keypoints are scored, meaning those at least one training dataset annotates,
so the model has a trained channel and transfer is a meaningful question. Keypoints only the
held-out dataset annotates are excluded: no leave-one-out model has a channel for them at
all, so every variant emits a near-constant prediction and the comparison is degenerate. The
dedicated model is likewise excluded, being in-domain and fully supervised, which compresses
the axis without bearing on the augmentation.

    python paper_figures/make_fig_aug.py --out ../paper/figures/fig_aug.pdf
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42  # TrueType, not Type 3; NeurIPS PDF requirement
import matplotlib.pyplot as plt
import numpy as np

from mouse_pose.paths import load_paths
from mouse_pose.plots.ensemble import (Ensemble, build_ensemble, compute_percentiles)

DSETS = ["ibl", "cazettes-side", "cheese-2d", "kondo", "facemap"]
LOO = {"facemap": "ibl+cheese+caz+kondo", "ibl": "face+cheese+caz+kondo",
       "cheese-2d": "face+ibl+caz+kondo", "cazettes-side": "face+ibl+cheese+kondo",
       "kondo": "face+ibl+cheese+caz"}
# colours shared with the other figures: grey = the earlier shared model, orange = the
# intermediate generation, purple = the recipe of record, green = the dedicated model.
SERIES = [("leave-one-out, DLC aug", "{loo}_train/supervised/sampling-T2/tf1/vits_dinov3/seed0",
           "#808080"),
          ("leave-one-out, zoom out only", "zoom-aug-exp/{loo}-T2-zoomaug/seed0", "#F58518"),
          ("leave-one-out, zoom in/out", "zoom-aug-exp/{loo}-T2-zoominout/seed0", "#7B3FB8"),
]
SEEDS = [0, 1, 2]


def eval_files(run: Path, dset: str):
    d = run / "eval" / dset
    p, e = d / "predictions.csv", d / "pixel_error.csv"
    return (str(p), str(e)) if p.is_file() and e.is_file() else None


def classes_of(inv, dset):
    others = set().union(*[set(v["direct"]) for k, v in inv.items() if k != dset])
    ev = set(inv[dset]["eval"]) - {"pupil_center_right"}
    return sorted(ev & others), sorted(ev - others)


def build(results, dset, kps):
    ens, err = Ensemble(), {}
    ens.data_to_plot = {}
    for name, tmpl, _ in SERIES:
        runs = ([results / tmpl.format(dset=dset, seed=s) for s in SEEDS]
                if "{seed}" in tmpl else [results / tmpl.format(loo=LOO[dset])])
        for i, r in enumerate(runs):
            f = eval_files(r, dset)
            if f is None:
                raise FileNotFoundError(f"{dset} {name}.{i}: {r}")
            ens.data_to_plot[f"{name}.{i}"], err[f"{name}.{i}"] = f
    build_ensemble(ens, error_csv_dict=err, keypoints=kps, fast=True)
    return ens


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1], epilog=__doc__)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    paths = load_paths()
    results = Path(paths["results_dir"])
    inv = json.loads((Path(paths["data_dir"]) / "dataset_inventory.json").read_text())["datasets"]

    fig, axes = plt.subplots(1, len(DSETS), figsize=(3.05 * len(DSETS), 3.1))
    for col, dset in enumerate(DSETS):
        sup, _ = classes_of(inv, dset)
        for kps, cname in ((sup, "supported"),):
            ax = axes[col]
            ens = build(results, dset, kps)
            d = ens.df_line2.copy()
            d["model2"] = d["model"].apply(lambda s: s.rsplit(".", 1)[0])
            for name, _, colr in SERIES:
                g = d[d["model2"] == name].groupby("ens-std")["pixel_error"].mean().sort_index()
                ax.plot(g.index.to_numpy(), g.to_numpy(), color=colr, lw=1.9, label=name)
            key = next(k for k in ens.n_points_dict if k.startswith(SERIES[0][0]))
            vals, prc = compute_percentiles(np.asarray(ens.n_points_dict[key], dtype=float),
                                            np.asarray(ens.std_vals, dtype=float), [95, 50, 5])
            for p, v in zip(prc, vals):
                ax.axvline(v, ls="--", lw=0.9, color="black", alpha=0.45)
                ax.text(v, ax.get_ylim()[1], f"{p:g}%", ha="left", va="top", fontsize=6,
                        rotation=90)
            ax.set_title(f"{dset} · {cname} ({len(kps)} kp)", fontsize=8.5)
            ax.set_ylim(bottom=0); ax.grid(alpha=0.3); ax.tick_params(labelsize=7.5)
            ax.set_xlabel("ensemble std dev", fontsize=8.5)
    axes[0].set_ylabel("pixel error\n(supported keypoints)", fontsize=8.5)
    axes[0].legend(fontsize=7, loc="upper left", framealpha=0.9)
    fig.tight_layout()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=300)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
