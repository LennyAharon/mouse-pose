#!/usr/bin/env python
"""
Appendix figure: head architecture and dataset sampling temperature.

Pixel error against ensemble standard deviation for the two output-head designs, a single
shared 36-keypoint head and one head per source dataset, each trained at sampling temperature
T in {1, 2, inf}. T=1 draws frames in proportion to corpus size, T=inf equalises supervision
across datasets, T=2 is intermediate. All six models are trained on all five datasets, so
every evaluation here is in domain; this fixes the recipe rather than testing transfer.

Two grids exist. ``--grid zio`` is the recipe of record, trained with the zoom-in/out
augmentation the paper adopts; ``--grid dlc`` is the earlier grid under the stock DeepLabCut
augmentation, kept because it shows the ordering does not depend on the augmentation.

A per-dataset head must be told which dataset a frame came from. ``--blind`` scores those
three series from ``eval_blind`` instead of ``eval``, that is, with the head chosen by the
model rather than by an oracle. The shared head is unaffected, having nothing to route.

    python paper_figures/make_fig_arch.py --grid zio --out ../paper/figures/fig_arch.pdf
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
from mouse_pose.plots.ensemble import Ensemble, build_ensemble, compute_percentiles

DSETS   = ["ibl", "cazettes-side", "cheese-2d", "kondo", "facemap"]
# shared head in purple shades (the recipe of record is T=2), per-dataset head in orange
SHADES  = ["#C7B3E0", "#7B3FB8", "#3E1F5C", "#FBD1A2", "#F58518", "#8C4A0A"]
LABELS  = [f"{h} head, $T{{=}}{t}$" for h in ("shared", "per-dataset")
           for t in ("1", "2", "\\infty")]
_DLC    = "face+ibl+cheese+caz+kondo_train/supervised"
_ZIO    = "zoom-aug-exp/face+ibl+cheese+caz+kondo"
GRIDS = {
    "dlc": [f"{_DLC}/tf1/vits_dinov3/seed0",
            f"{_DLC}/sampling-T2/tf1/vits_dinov3/seed0",
            f"{_DLC}/sampling-Tinf/tf1/vits_dinov3/seed0",
            f"{_DLC}/head-per_dataset/tf1/vits_dinov3/seed0",
            f"{_DLC}/head-per_dataset/sampling-T2/tf1/vits_dinov3/seed0",
            f"{_DLC}/head-per_dataset/sampling-Tinf/tf1/vits_dinov3/seed0"],
    "zio": [f"{_ZIO}-T1-zoominout/seed0",
            f"{_ZIO}-T2-zoominout/seed0",
            f"{_ZIO}-Tinf-zoominout/seed0",
            f"{_ZIO}-T1-headperds-zoominout/seed0",
            f"{_ZIO}-T2-headperds-zoominout/seed0",
            f"{_ZIO}-Tinf-headperds-zoominout/seed0"],
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1], epilog=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--grid", default="zio", choices=sorted(GRIDS))
    ap.add_argument("--blind", action="store_true",
                    help="score per-dataset heads from eval_blind (no oracle dataset identity)")
    args = ap.parse_args()
    # the last three series are the per-dataset heads, the only ones a blind readout changes
    series = [(lab, rel, col, "eval_blind" if (args.blind and i >= 3) else "eval")
              for i, (lab, rel, col) in enumerate(zip(LABELS, GRIDS[args.grid], SHADES))]
    paths = load_paths()
    results = Path(paths["results_dir"])
    inv = json.loads((Path(paths["data_dir"]) / "dataset_inventory.json").read_text())["datasets"]

    fig, axes = plt.subplots(1, len(DSETS), figsize=(3.05 * len(DSETS), 3.2))
    summary = {}
    for ax, dset in zip(axes, DSETS):
        kps = sorted(set(inv[dset]["eval"]) - {"pupil_center_right"})
        ens, err = Ensemble(), {}
        ens.data_to_plot = {}
        for name, rel, _, sub in series:
            d = results / rel / sub / dset
            p, e = d / "predictions.csv", d / "pixel_error.csv"
            if not (p.is_file() and e.is_file()):
                raise FileNotFoundError(f"{dset} {name}: {p}")
            ens.data_to_plot[f"{name}.0"], err[f"{name}.0"] = str(p), str(e)
        build_ensemble(ens, error_csv_dict=err, keypoints=kps, fast=True)
        d = ens.df_line2.copy()
        d["model2"] = d["model"].apply(lambda s: s.rsplit(".", 1)[0])
        for name, _, col, _ in series:
            g = d[d["model2"] == name].groupby("ens-std")["pixel_error"].mean().sort_index()
            ax.plot(g.index.to_numpy(), g.to_numpy(), color=col, lw=1.8, label=name)
            summary.setdefault(name, []).append(float(np.interp(
                compute_percentiles(np.asarray(ens.n_points_dict[f"{series[0][0]}.0"], float),
                                    np.asarray(ens.std_vals, float), [50])[0][0],
                g.index.to_numpy(), g.to_numpy())))
        key = f"{series[0][0]}.0"
        vals, prc = compute_percentiles(np.asarray(ens.n_points_dict[key], dtype=float),
                                        np.asarray(ens.std_vals, dtype=float), [95, 50, 5])
        for p_, v in zip(prc, vals):
            ax.axvline(v, ls="--", lw=0.9, color="black", alpha=0.45)
        ax.set_title(dset, fontsize=9); ax.set_ylim(bottom=0)
        ax.set_xlabel("ensemble std dev", fontsize=8.5)
        ax.grid(alpha=0.3); ax.tick_params(labelsize=7.5)
    axes[0].set_ylabel("pixel error", fontsize=8.5)
    axes[0].legend(fontsize=6.4, loc="upper left", framealpha=0.9)
    fig.tight_layout()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=300)
    print(f"wrote {out}\n")
    print(f"{'series':30s} " + "".join(f"{d[:9]:>11s}" for d in DSETS) + f"{'mean':>9s}")
    for name, _, _, _ in series:
        v = summary[name]
        print(f"{name.replace('$','').replace(chr(92)+'infty','inf').replace('{','').replace('}',''):30s} "
              + "".join(f"{x:11.1f}" for x in v) + f"{np.mean(v):9.1f}")


if __name__ == "__main__":
    main()
