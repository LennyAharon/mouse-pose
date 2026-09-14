#!/usr/bin/env python
"""
Appendix figure: adaptation decomposed by keypoint class.

Top row scores only *supported* keypoints (the target annotates them and its leave-one-out
base model already knows them); bottom row only *new* keypoints (the target annotates them
and no other dataset does, so the base model has never represented them). Same axes as the
main results figure: x = annotated frames, 0 being the frozen base model, y = pooled pixel
error at the 50th-percentile ensemble-std operating point. One ensemble is built per
(dataset, class), so every series in a panel is scored on identical points.

The separation is the point: the two classes have entirely different adaptation curves, and
the three adaptation procedures are nearly indistinguishable within each. That is what makes
the masked-label comparison in the main figure necessary, since no split of the *annotated*
keypoints separates the methods.

    python paper_figures/make_fig_classes.py --out ../paper/figures/fig_classes.pdf
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42  # TrueType, not Type 3; NeurIPS PDF requirement
import matplotlib.pyplot as plt

from mouse_pose.paths import load_paths
from mouse_pose.plots.ensemble import Ensemble, build_ensemble, error_at_percentile

DSETS = ["ibl", "cazettes-side", "cheese-2d", "kondo", "facemap"]
# Paper display names; keys are the on-disk dataset names, which the paper does not use.
DISPLAY = {"ibl": "IBL", "cazettes-side": "Cazettes", "cheese-2d": "Cheese-3D",
           "kondo": "Kondo", "facemap": "Facemap"}
LOO = {"facemap": "ibl+cheese+caz+kondo", "ibl": "face+cheese+caz+kondo",
       "cheese-2d": "face+ibl+caz+kondo", "cazettes-side": "face+ibl+cheese+kondo",
       "kondo": "face+ibl+cheese+caz"}
BUDGETS, DRAWS, SEEDS = [10, 25, 50], [0, 1, 2], [0, 1, 2]
ARMS = [("anchored LoRA", "fewshot-exp-anchor-lora-conf1-zio", "#7B3FB8", "o-"),
        ("full fine-tuning", "fewshot-exp-lr5-zio", "#E45756", "^-"),
        ("LoRA", "fewshot-exp-lora-r16-lr5e-5-head5e-4-zio", "#F58518", "v-"),
        ("DINO, from scratch", "fewshot-exp-dino", "#4C78A8", "s--")]
C_SINGLE, C_ZS = "#54A24B", "#808080"


def eval_files(run: Path, dset: str):
    d = run / "eval" / dset
    p, e = d / "predictions.csv", d / "pixel_error.csv"
    return (str(p), str(e)) if p.is_file() and e.is_file() else None


def classes_of(inv, dset):
    others = set().union(*[set(v["direct"]) for k, v in inv.items() if k != dset])
    ev = set(inv[dset]["eval"]) - {"pupil_center_right"}
    return sorted(ev & others), sorted(ev - others)


def values(results, dset, kps):
    runs = {"zero-shot": [results / "zoom-aug-exp" / f"{LOO[dset]}-T2-zoominout" / "seed0"]}
    for _, root, _, _ in ARMS:
        for n in BUDGETS:
            runs[f"{root}|{n}"] = [results / root / dset / f"tf{n}-draw{d}" for d in DRAWS]
    runs["dedicated"] = [results / f"{dset}_train" / "supervised" / "tf1" / "vits_dinov3" /
                         f"seed{s}" for s in SEEDS]
    ens, err = Ensemble(), {}
    ens.data_to_plot = {}
    for label, paths in runs.items():
        for i, r in enumerate(paths):
            f = eval_files(r, dset)
            if f is None:
                raise FileNotFoundError(f"{dset} {label}.{i}: {r}")
            ens.data_to_plot[f"{label}.{i}"], err[f"{label}.{i}"] = f
    build_ensemble(ens, error_csv_dict=err, keypoints=kps, fast=True)
    return {k: error_at_percentile(ens, k, 50.0) for k in runs}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1], epilog=__doc__)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    paths = load_paths()
    results = Path(paths["results_dir"])
    inv = json.loads((Path(paths["data_dir"]) / "dataset_inventory.json").read_text())["datasets"]

    fig, axes = plt.subplots(2, len(DSETS), figsize=(3.05 * len(DSETS), 5.6))
    for col, dset in enumerate(DSETS):
        sup, new = classes_of(inv, dset)
        for row, (kps, cname) in enumerate(((sup, "supported"), (new, "new"))):
            ax = axes[row, col]
            if not kps:
                ax.text(0.5, 0.5, f"{dset} annotates\nno new keypoints", ha="center",
                        va="center", fontsize=8.5, color="#666666", transform=ax.transAxes)
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_color("#cccccc")
                continue
            v = values(results, dset, kps)
            for name, root, col_, style in ARMS:
                ys = [v[f"{root}|{n}"] for n in BUDGETS]
                xs = BUDGETS
                if root != "fewshot-exp-dino":      # DINO has no zero-shot point
                    xs, ys = [0] + BUDGETS, [v["zero-shot"]] + ys
                ax.plot(xs, ys, style, color=col_, lw=1.7, ms=4.5, label=name)
            ax.axhline(v["dedicated"], color=C_SINGLE, ls=":", lw=1.6, label="dedicated")
            ax.plot([0], [v["zero-shot"]], "o", color=C_ZS, ms=7, zorder=5)
            ax.set_xticks([0] + BUDGETS); ax.tick_params(labelsize=7.5)
            ax.grid(alpha=0.3); ax.set_ylim(bottom=0)
            ax.set_title(f"{DISPLAY.get(dset, dset)} - {cname} ({len(kps)} kp)", fontsize=8.5)
            if row == 1:
                ax.set_xlabel("annotated frames", fontsize=8.5)
    for row, lbl in enumerate(("supported keypoints", "new keypoints")):
        axes[row, 0].set_ylabel(f"{lbl}\npixel error at 50% ens. std", fontsize=8.5)
    axes[0, 0].legend(fontsize=6.8, loc="upper right", framealpha=0.9)
    fig.tight_layout()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=300)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
