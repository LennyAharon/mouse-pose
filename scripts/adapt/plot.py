#!/usr/bin/env python
"""
Figures of an adaptation grid from collect.py's cells.csv (PDF + SVG; PNG only with --png).

    python scripts/adapt/plot.py --config configs/adaptation/cosyne-2026.yaml \
        [--keypoints all|supported|new] [--png]

1. curves_<keypoints>.pdf: one panel per target dataset; x = training frames (zero-shot, then
   the grid's N values and `all`), y = pooled mean px on the target's test frames (log scale),
   mean +- sd over draws; one line per curve arm (MM arms start at the zero-shot point);
   dedicated models as dashed horizontal lines.
2. masked.pdf: masked-label protocol, one bar group per (dataset, hidden keypoints): zero-shot
   trunk, MM + plain LoRA, MM + anchored LoRA, and the labelled reference (anchored LoRA that saw
   the labels), mean +- sd over draws.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from mighty_mouse.adaptation import load_grid  # noqa: E402
from mighty_mouse.paths import load_paths  # noqa: E402

STYLE = {  # arm -> (label, colour, marker)
    "mm-anchored-lora":    ("MM + anchored LoRA", "#c0392b", "o"),
    "mm-lora":             ("MM + LoRA", "#e67e22", "s"),
    "dino-linear":         ("DINOv3 from scratch (linear head)", "#2471a3", "^"),
    "dino-nonlinear":      ("DINOv3 from scratch (nonlinear head)", "#17a589", "v"),
    "dedicated-linear":    ("dedicated (linear head)", "#2471a3", None),
    "dedicated-nonlinear": ("dedicated (nonlinear head)", "#17a589", None),
    "zero-shot":           ("zero-shot (MM trunk)", "#7f8c8d", "D"),
    "labelled":            ("MM + anchored LoRA, labels seen", "#6c3483", "P"),
}


def save(fig, path: Path, png: bool) -> None:
    for ext in ("pdf", "svg") + (("png",) if png else ()):
        fig.savefig(path.with_suffix("." + ext), bbox_inches="tight", dpi=200)
    print(f"wrote {path.with_suffix('.pdf')} (+ svg{' + png' if png else ''})")


def curves(df: pd.DataFrame, grid, keypoints: str, out: Path, png: bool) -> None:
    xs   = ["0"] + [str(n) for n in grid.n_frames]
    xpos = {x: i for i, x in enumerate(xs)}
    cur  = df[(df.kind != "masked") & (df.keypoints == keypoints)]
    dsets = list(grid.datasets)
    ncol = 4
    nrow = int(np.ceil(len(dsets) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.2 * nrow), squeeze=False)
    for ax, ds in zip(axes.ravel(), dsets, strict=False):
        d = cur[cur.dataset == ds]
        for arm in grid.curve_arms:
            a = d[d.arm == arm]
            if grid.arms[arm]["init"] == "trunk":
                a = pd.concat([d[d.arm == "zero-shot"], a])
            g = a.groupby("n")["mean_px"].agg(["mean", "std"])
            g = g.reindex([x for x in xs if x in set(a.n)])
            if g.empty:
                continue
            lab, col, mk = STYLE.get(arm, (arm, None, "o"))
            ax.errorbar([xpos[x] for x in g.index], g["mean"], yerr=g["std"].fillna(0),
                        label=lab, color=col, marker=mk, ms=4, lw=1.5, capsize=2)
        for head in grid.dedicated:
            r = d[d.arm == f"dedicated-{head}"]["mean_px"]
            if len(r):
                lab, col, _ = STYLE.get(f"dedicated-{head}", (head, None, None))
                ax.axhline(r.iloc[0], ls="--", lw=1, color=col, label=lab)
        ax.set_title(ds)
        ax.set_yscale("log")
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels(["zero-shot" if x == "0" else x for x in xs], fontsize=8)
        ax.grid(alpha=0.3, which="both")
    for ax in axes.ravel()[len(dsets):]:
        ax.axis("off")
    for row in axes:
        row[0].set_ylabel(f"pixel error ({keypoints} keypoints)")
    seen = {}
    for ax in axes.ravel():
        for hh, ll in zip(*ax.get_legend_handles_labels(), strict=True):
            seen.setdefault(ll, hh)
    fig.legend(list(seen.values()), list(seen.keys()), loc="lower center", ncol=3,
               frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(f"{grid.name}: adaptation to a held-out lab (x = training frames of that lab)",
                 y=1.01)
    fig.tight_layout()
    save(fig, out / f"curves_{keypoints}", png)


def masked_bars(df: pd.DataFrame, grid, out: Path, png: bool) -> None:
    m = df[(df.kind == "masked") & (df.keypoints == "hidden")]
    if m.empty:
        return
    order  = ["zero-shot"] + list(grid.masked["arms"])[::-1] + ["labelled"]
    groups = list(m.groupby(["dataset", "mask"]).groups)
    fig, ax = plt.subplots(figsize=(max(8, 0.9 * len(groups)), 3.6))
    w = 0.8 / len(order)
    for j, arm in enumerate(order):
        vals = [m[(m.dataset == ds) & (m["mask"] == mk) & (m.arm == arm)]["mean_px"]
                for ds, mk in groups]
        lab, col, _ = STYLE.get(arm, (arm, None, None))
        ax.bar(np.arange(len(groups)) + (j - (len(order) - 1) / 2) * w,
               [v.mean() for v in vals], w, yerr=[v.std() if len(v) > 1 else 0 for v in vals],
               label=lab, color=col, capsize=2)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([f"{ds}\n{mk.replace('_', ' ')[:28]}" for ds, mk in groups], fontsize=7)
    ax.set_yscale("log")
    ax.set_ylabel("pixel error on the hidden keypoints")
    ax.set_title(f"masked-label protocol (N = {grid.masked['n_frames'][0]}): "
                 "transfer of a keypoint the lab never labelled")
    ax.legend(fontsize=8, ncol=4, frameon=False)
    ax.grid(alpha=0.3, axis="y", which="both")
    fig.tight_layout()
    save(fig, out / "masked", png)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1], epilog=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config",    required=True, type=Path)
    ap.add_argument("--keypoints", default="all", choices=["all", "supported", "new"])
    ap.add_argument("--png",       action="store_true", help="also write PNG (only when asked)")
    args = ap.parse_args()

    grid = load_grid(args.config)
    out  = Path(load_paths()["results_dir"]) / grid.out_subdir / grid.name / "summary"
    df   = pd.read_csv(out / "cells.csv", dtype={"n": str, "mask": str}, keep_default_na=False,
                       na_values={"mean_px": [""], "median_px": [""]})
    curves(df, grid, args.keypoints, out, args.png)
    masked_bars(df, grid, out, args.png)


if __name__ == "__main__":
    main()
