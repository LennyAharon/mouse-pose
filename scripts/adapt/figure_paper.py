#!/usr/bin/env python
"""
Paper-style adaptation curves (as Figure 3a of the workshop paper) from an adaptation grid's saved
test predictions: one panel per target dataset, x = annotated frames (0 = zero-shot), lines for the
chosen arms (mean +- sd over draws), the dedicated model as a dotted line.

    python scripts/adapt/figure_paper.py --config configs/adaptation/cosyne-2026.yaml \
        [--keypoints all|supported] [--arms mm-anchored-lora,dino-nonlinear] \
        [--dedicated nonlinear]

Two metrics, both written to summary/paper_<keypoints>.csv and plotted as two figures:
`pooled` = mean px over every (test frame, keypoint) with a visible == 2 label, and `ens<P>` = the
paper's operating point (appendix "Pixel error against ensemble standard deviation"): the spread
(sqrt of the summed x / y variances) of the predicted location of each (frame, keypoint) across
every run in the panel (zero-shot, each arm x N x draw, dedicated), then each run's mean error over
the points whose spread is in the top P % (P = 50: the harder half). Roughly 1.7x the pooled mean.
Figures: summary/paper_<keypoints>_<metric>.pdf / .svg (PNG only with --png).
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from mighty_mouse.adaptation import (  # noqa: E402
    ALL,
    Cell,
    cell_dir,
    keypoint_sets,
    load_grid,
    pixel_errors,
    trained_keypoints,
    trunk_datasets,
)
from mighty_mouse.paths import load_paths  # noqa: E402

STYLE = {  # arm -> (label, colour, marker, linestyle)
    "mm-anchored-lora": ("Mighty Mouse + anchored LoRA", "#7b3fbf", "o", "-"),
    "mm-lora":          ("Mighty Mouse + LoRA", "#e67e22", "s", "-"),
    "mm-full-ft":       ("Mighty Mouse + full fine-tuning", "#b7950b", "X", "-"),
    "dino-linear":      ("DINO backbone, from scratch (linear head)", "#4a7fb5", "s", "--"),
    "dino-nonlinear":   ("DINO backbone, from scratch", "#4a7fb5", "s", "--"),
}


def coords(pred_csv: Path, frames: pd.Index, kps: list[str]) -> np.ndarray:
    """Predicted (x, y) of ``kps`` on ``frames``: array (frames, keypoints, 2)."""
    pr = pd.read_csv(pred_csv, header=[0, 1, 2], index_col=0).loc[frames]
    sc = pr.columns[0][0]
    return np.stack([pr[[(sc, k, "x") for k in kps]].to_numpy(float),
                     pr[[(sc, k, "y") for k in kps]].to_numpy(float)], axis=-1)


def panel(runs: dict[str, Path], label_csv: Path, kps: list[str], pct: float) -> pd.DataFrame:
    """Pooled and ensemble-spread errors of every run of one panel (one row per run)."""
    errs = {r: pixel_errors(p, label_csv) for r, p in runs.items()}
    frames = errs[next(iter(errs))].index
    for e in errs.values():
        frames = frames.intersection(e.index)
    kps = [k for k in kps if all(k in e.columns for e in errs.values())]
    err = np.stack([errs[r].loc[frames, kps].to_numpy(float) for r in runs])   # runs, F, K
    xy  = np.stack([coords(p, frames, kps) for p in runs.values()])           # runs, F, K, 2
    ok  = np.isfinite(err).all(axis=0)                                         # visible == 2
    spread = np.sqrt(xy.var(axis=0).sum(axis=-1))                              # F, K
    thr  = np.percentile(spread[ok], 100 - pct)
    hard = ok & (spread >= thr)
    return pd.DataFrame({"run": list(runs), "pooled": [e[ok].mean() for e in err],
                         f"ens{pct:g}": [e[hard].mean() for e in err],
                         "n_points": int(ok.sum()), "n_hard": int(hard.sum())})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1], epilog=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config",    required=True, type=Path)
    ap.add_argument("--keypoints", default="supported", choices=["all", "supported", "new"])
    ap.add_argument("--arms",      default="mm-anchored-lora,dino-nonlinear")
    ap.add_argument("--dedicated", default="nonlinear", help="head of the dedicated line")
    ap.add_argument("--pct",       default=50.0, type=float, help="percent of points kept")
    ap.add_argument("--png",       action="store_true", help="also write PNG (only when asked)")
    args = ap.parse_args()

    grid  = load_grid(args.config)
    paths = load_paths()
    data, results = Path(paths["data_dir"]), Path(paths["results_dir"])
    inv   = json.loads((data / "dataset_inventory.json").read_text())["datasets"]
    out   = results / grid.out_subdir / grid.name / "summary"
    out.mkdir(parents=True, exist_ok=True)
    arms  = args.arms.split(",")
    ns    = [n for n in grid.n_frames if n != ALL]
    met   = f"ens{args.pct:g}"

    rows = []
    for ds, spec in grid.datasets.items():
        tdir = results / spec["trunk"]
        tcfg = yaml.safe_load((tdir / "config.yaml").read_text())
        kps  = keypoint_sets(ds, inv, trained_keypoints(trunk_datasets(tdir, tcfg), inv))
        kps  = kps[args.keypoints]
        runs = {"zero-shot|0|0": tdir / "eval" / ds / "predictions.csv"}
        ded  = results / grid.dedicated[args.dedicated].format(ds=ds) / "eval" / ds
        if (ded / "predictions.csv").exists():
            runs[f"dedicated|{ALL}|0"] = ded / "predictions.csv"
        for arm in arms:
            for n in ns:
                for d in grid.draws:
                    c = cell_dir(results, grid, Cell(ds, arm, n, d))
                    if (c / ".done").exists():
                        runs[f"{arm}|{n}|{d}"] = c / "eval" / ds / "predictions.csv"
        if not kps:
            continue
        p = panel(runs, data / f"CollectedData_{ds}_test.csv", kps, args.pct)
        p[["arm", "n", "draw"]] = p["run"].str.split("|", expand=True)
        p["dataset"] = ds
        rows.append(p)
        print(f"{ds:14s} {len(runs):2d} runs, {p.n_points.iloc[0]} points "
              f"({p.n_hard.iloc[0]} in the top {args.pct:g} % spread)")
    df = pd.concat(rows, ignore_index=True)
    df.to_csv(out / f"paper_{args.keypoints}.csv", index=False)

    for metric in ("pooled", met):
        dsets = list(dict.fromkeys(df.dataset))
        ncol  = 4
        nrow  = int(np.ceil(len(dsets) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 2.9 * nrow), squeeze=False)
        for ax, ds in zip(axes.ravel(), dsets, strict=False):
            d  = df[df.dataset == ds]
            zs = d[d.arm == "zero-shot"][metric].iloc[0]
            for arm in arms:
                a = d[d.arm == arm]
                g = a.groupby(a.n.astype(int))[metric].agg(["mean", "std"]).sort_index()
                if g.empty:
                    continue
                lab, col, mk, ls = STYLE.get(arm, (arm, None, "o", "-"))
                x = [0] + list(g.index) if arm.startswith("mm") else list(g.index)
                y = [zs] + list(g["mean"]) if arm.startswith("mm") else list(g["mean"])
                e = [0] + list(g["std"].fillna(0)) if arm.startswith("mm") else list(
                    g["std"].fillna(0))
                ax.errorbar(x, y, yerr=e, color=col, marker=mk, ls=ls, lw=2, ms=6, capsize=2,
                            label=lab)
            ax.plot([0], [zs], "o", color="#888888", ms=9, zorder=5)
            dd = d[d.arm == "dedicated"][metric]
            if len(dd):
                ax.axhline(dd.iloc[0], ls=":", lw=2.5, color="#6aa84f",
                           label="dedicated model (all frames)")
            ax.set_title(ds, fontweight="bold")
            ax.set_xticks([0] + ns)
            ax.set_xlabel("annotated frames")
            ax.set_ylim(bottom=0)
            ax.spines[["top", "right"]].set_visible(False)
        for ax in axes.ravel()[len(dsets):]:
            ax.axis("off")
        ylab = ("pixel error (pooled mean)" if metric == "pooled"
                else f"pixel error at {args.pct:g}%\nensemble std")
        for row in axes:
            row[0].set_ylabel(ylab)
        h, lab = axes[0, 0].get_legend_handles_labels()
        fig.legend(h, lab, loc="lower center", ncol=3, frameon=True,
                   bbox_to_anchor=(0.5, -0.07))
        fig.suptitle(f"{grid.name}: {args.keypoints} keypoints, {metric}", y=1.01)
        fig.tight_layout()
        stem = out / f"paper_{args.keypoints}_{metric}"
        for ext in ("pdf", "svg") + (("png",) if args.png else ()):
            fig.savefig(stem.with_suffix("." + ext), bbox_inches="tight", dpi=200)
        print(f"wrote {stem}.pdf (+ svg{' + png' if args.png else ''})")


if __name__ == "__main__":
    main()
