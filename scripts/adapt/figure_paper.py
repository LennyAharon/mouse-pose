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
Figures: summary/paper_<keypoints>_<metric>.pdf / .svg (PNG only with --png). With
--per_keypoint, also summary/keypoints/<dataset>_curves_<keypoints>_<metric>.pdf: the same panel
for every keypoint (each keypoint its own panel: spread and top P % within that keypoint).
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


def load_panel(runs: dict[str, Path], label_csv: Path,
               kps: list[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Errors (runs, frames, keypoints) and predicted (x, y) (runs, frames, keypoints, 2)."""
    errs = {r: pixel_errors(p, label_csv) for r, p in runs.items()}
    frames = errs[next(iter(errs))].index
    for e in errs.values():
        frames = frames.intersection(e.index)
    kps = [k for k in kps if all(k in e.columns for e in errs.values())]
    err = np.stack([errs[r].loc[frames, kps].to_numpy(float) for r in runs])
    xy  = np.stack([coords(p, frames, kps) for p in runs.values()])
    return err, xy, kps


def score(runs: list[str], err: np.ndarray, xy: np.ndarray, pct: float) -> pd.DataFrame | None:
    """Pooled and ensemble-spread errors of every run of one panel (one row per run): the spread
    of each (frame, keypoint) is taken across all the panel's runs; ens<pct> = each run's mean
    error over the points in the top pct % of spread."""
    ok = np.isfinite(err).all(axis=0)                                         # visible == 2
    if not ok.any():
        return None
    spread = np.sqrt(xy.var(axis=0).sum(axis=-1))
    hard = ok & (spread >= np.percentile(spread[ok], 100 - pct))
    return pd.DataFrame({"run": runs, "pooled": [e[ok].mean() for e in err],
                         f"ens{pct:g}": [e[hard].mean() for e in err],
                         "n_points": int(ok.sum()), "n_hard": int(hard.sum())})


def draw_panel(ax, d: pd.DataFrame, arms: list[str], ns: list, metric: str, title: str,
               small: bool = False) -> None:
    """One panel: zero-shot dot, each arm vs annotated frames (MM arms start at zero-shot),
    dedicated as a dotted line."""
    zs = d[d.arm == "zero-shot"][metric].iloc[0]
    for arm in arms:
        a = d[d.arm == arm]
        g = a.groupby(a.n.astype(int))[metric].agg(["mean", "std"]).sort_index()
        if g.empty:
            continue
        lab, col, mk, ls = STYLE.get(arm, (arm, None, "o", "-"))
        start = arm.startswith("mm")
        x = ([0] if start else []) + list(g.index)
        y = ([zs] if start else []) + list(g["mean"])
        e = ([0] if start else []) + list(g["std"].fillna(0))
        ax.errorbar(x, y, yerr=e, color=col, marker=mk, ls=ls, lw=1.5 if small else 2,
                    ms=4 if small else 6, capsize=2, label=lab)
    ax.plot([0], [zs], "o", color="#888888", ms=6 if small else 9, zorder=5)
    dd = d[d.arm == "dedicated"][metric]
    if len(dd):
        ax.axhline(dd.iloc[0], ls=":", lw=2 if small else 2.5, color="#6aa84f",
                   label="dedicated model (all frames)")
    ax.set_title(title, fontweight="bold", fontsize=9 if small else 12)
    ax.set_xticks([0] + ns)
    ax.tick_params(labelsize=7 if small else 10)
    ax.set_ylim(bottom=0)
    ax.spines[["top", "right"]].set_visible(False)


def save(fig, stem: Path, png: bool) -> None:
    for ext in ("pdf", "svg") + (("png",) if png else ()):
        fig.savefig(stem.with_suffix("." + ext), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"wrote {stem}.pdf (+ svg{' + png' if png else ''})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1], epilog=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config",    required=True, type=Path)
    ap.add_argument("--keypoints", default="supported", choices=["all", "supported", "new"])
    ap.add_argument("--arms",      default="mm-anchored-lora,dino-nonlinear")
    ap.add_argument("--dedicated", default="nonlinear", help="head of the dedicated line")
    ap.add_argument("--pct",       default=50.0, type=float, help="percent of points kept")
    ap.add_argument("--per_keypoint", action="store_true",
                    help="also one figure per dataset with a panel per keypoint")
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

    rows, kp_rows = [], []
    for ds, spec in grid.datasets.items():
        tdir = results / spec["trunk"]
        tcfg = yaml.safe_load((tdir / "config.yaml").read_text())
        sets = keypoint_sets(ds, inv, trained_keypoints(trunk_datasets(tdir, tcfg), inv))
        kps  = sets[args.keypoints]
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
        err, xy, kps = load_panel(runs, data / f"CollectedData_{ds}_test.csv", kps)
        p = score(list(runs), err, xy, args.pct)
        p["dataset"] = ds
        rows.append(p)
        print(f"{ds:14s} {len(runs):2d} runs, {p.n_points.iloc[0]} points "
              f"({p.n_hard.iloc[0]} in the top {args.pct:g} % spread)")
        if args.per_keypoint:   # each keypoint its own panel: spread and top pct % within it
            for j, k in enumerate(kps):
                q = score(list(runs), err[:, :, [j]], xy[:, :, [j]], args.pct)
                if q is not None:
                    q["dataset"], q["keypoint"] = ds, k
                    q["supported"] = k in sets["supported"]
                    kp_rows.append(q)
    df = pd.concat(rows, ignore_index=True)
    df[["arm", "n", "draw"]] = df["run"].str.split("|", expand=True)
    df.to_csv(out / f"paper_{args.keypoints}.csv", index=False)
    ylabels = {"pooled": "pixel error (pooled mean)",
               met: f"pixel error at {args.pct:g}%\nensemble std"}

    for metric in ("pooled", met):
        dsets = list(dict.fromkeys(df.dataset))
        ncol  = 4
        nrow  = int(np.ceil(len(dsets) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 2.9 * nrow), squeeze=False)
        for ax, ds in zip(axes.ravel(), dsets, strict=False):
            draw_panel(ax, df[df.dataset == ds], arms, ns, metric, ds)
            ax.set_xlabel("annotated frames")
        for ax in axes.ravel()[len(dsets):]:
            ax.axis("off")
        for row in axes:
            row[0].set_ylabel(ylabels[metric])
        h, lab = axes[0, 0].get_legend_handles_labels()
        fig.legend(h, lab, loc="lower center", ncol=3, frameon=True,
                   bbox_to_anchor=(0.5, -0.07))
        fig.suptitle(f"{grid.name}: {args.keypoints} keypoints, {metric}", y=1.01)
        fig.tight_layout()
        save(fig, out / f"paper_{args.keypoints}_{metric}", args.png)

    if not kp_rows:
        return
    kdf = pd.concat(kp_rows, ignore_index=True)
    kdf[["arm", "n", "draw"]] = kdf["run"].str.split("|", expand=True)
    kout = out / "keypoints"
    kout.mkdir(exist_ok=True)
    kdf.to_csv(kout / f"paper_{args.keypoints}_per_keypoint.csv", index=False)
    for ds, d in kdf.groupby("dataset", sort=False):
        kps  = list(dict.fromkeys(d.keypoint))
        ncol = min(5, len(kps))
        nrow = int(np.ceil(len(kps) / ncol))
        for metric in ("pooled", met):
            fig, axes = plt.subplots(nrow, ncol, figsize=(2.6 * ncol, 2.2 * nrow + 0.4),
                                     squeeze=False)
            for ax, k in zip(axes.ravel(), kps, strict=False):
                dk = d[d.keypoint == k]
                draw_panel(ax, dk, arms, ns, metric,
                           k + ("" if dk.supported.iloc[0] else " (new)"), small=True)
            for ax in axes.ravel()[len(kps):]:
                ax.axis("off")
            for row in axes:
                row[0].set_ylabel(ylabels[metric], fontsize=8)
            for ax in axes[-1]:
                ax.set_xlabel("annotated frames", fontsize=8)
            h, lab = axes[0, 0].get_legend_handles_labels()
            fig.legend(h, lab, loc="lower center", ncol=3, frameon=True, fontsize=8,
                       bbox_to_anchor=(0.5, -0.04 - 0.1 / nrow))
            fig.suptitle(f"{ds}: per keypoint ({args.keypoints} keypoints), {metric}",
                         fontweight="bold", y=1.01)
            fig.tight_layout()
            save(fig, kout / f"{ds}_curves_{args.keypoints}_{metric}", args.png)

if __name__ == "__main__":
    main()
