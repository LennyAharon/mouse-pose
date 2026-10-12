#!/usr/bin/env python
"""
Per-keypoint views of an adaptation grid (supported keypoints: those another lab labels, so the
leave-that-dataset-out trunk trained them), from saved test predictions.

    python scripts/adapt/figure_keypoints.py --config configs/adaptation/cosyne-2026.yaml \
        [--arm mm-anchored-lora] [--scratch dino-nonlinear] [--dedicated nonlinear]

1. summary/keypoints/<dataset>.pdf: one row per supported keypoint (worst zero-shot first, by
   zero-shot / dedicated error): zero-shot, dedicated (all frames), the MM arm and the
   from-scratch arm at the smallest and largest N (mean over draws). Log x axis.
2. summary/keypoints/gain_scatter.pdf: every supported keypoint of every target; x = zero-shot /
   dedicated error (how good the inherited prediction is), y = from-scratch / MM error at the
   smallest N (> 1: Mighty Mouse is the better start), with the Spearman correlation.
Numbers: summary/keypoints/keypoints.csv; readable tables per dataset (one row per keypoint,
grouped by body part, the better start in bold): summary/keypoints/keypoints.md. Error = mean
px over visible == 2 test labels. PDF + SVG (PNG only with --png).
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
    load_grid,
    pixel_errors,
    trained_keypoints,
    trunk_datasets,
)
from mighty_mouse.paths import load_paths  # noqa: E402

PURPLE, BLUE, GREY, GREEN = "#7b3fbf", "#4a7fb5", "#888888", "#6aa84f"


def save(fig, stem: Path, png: bool) -> None:
    for ext in ("pdf", "svg") + (("png",) if png else ()):
        fig.savefig(stem.with_suffix("." + ext), bbox_inches="tight", dpi=200)
    plt.close(fig)


def mean_err(errors: list[pd.DataFrame], k: str) -> float:
    vals = [e[k].dropna().mean() for e in errors if k in e.columns and e[k].notna().any()]
    return float(np.mean(vals)) if vals else np.nan


def body_part(k: str) -> int:
    """Sort key: eye, snout, ear, paw."""
    if k.startswith(("eye", "pupil")):
        return 0
    if k.startswith("ear"):
        return 2
    if k.startswith("wrist") or (k[:1] == "d" and k[1:2].isdigit()):
        return 3
    return 1


def write_markdown(df: pd.DataFrame, path: Path, arm: str, scratch: str, ns: list) -> None:
    """keypoints.md: per dataset, one row per supported keypoint; the better start in bold."""
    parts = ["eye", "snout", "ear", "paw"]
    lines = ["# Supported keypoints: zero-shot vs adaptation, per dataset", "",
             "Mean px on the target's test frames (visible == 2), mean over draws. "
             "MM = Mighty Mouse + anchored LoRA, DINO = DINOv3 from scratch (nonlinear head); "
             "the better of the two at each N in **bold**. zero-shot verdict: zero-shot / "
             "dedicated error, good <= 2.5x, poor <= 7.5x, failed above. dedicated = all of "
             "the lab's frames.", ""]
    head = ("| part | keypoint | zero-shot | verdict | "
            + " | ".join(f"MM N={n} | DINO N={n}" for n in ns) + " | dedicated |")
    for ds, d in df.groupby("dataset", sort=False):
        d = d.assign(part=d.keypoint.map(body_part)).sort_values(["part", "keypoint"])
        wins = {n: int((d[f"{arm}_N{n}"] < d[f"{scratch}_N{n}"]).sum()) for n in ns}
        lines += [f"## {ds}", "",
                  "MM better than DINO on "
                  + ", ".join(f"{wins[n]}/{len(d)} at N={n}" for n in ns) + " keypoints.", "",
                  head, "|" + "---|" * (4 + 2 * len(ns) + 1)]
        for _, r in d.iterrows():
            ratio = r.zs_ratio
            verdict = ("good" if ratio <= 2.5 else "poor" if ratio <= 7.5 else "failed")
            cells = []
            for n in ns:
                a, b = r[f"{arm}_N{n}"], r[f"{scratch}_N{n}"]
                fa, fb = f"{a:.1f}", f"{b:.1f}"
                if np.isfinite(a) and np.isfinite(b):
                    fa, fb = (f"**{fa}**", fb) if a < b else (fa, f"**{fb}**")
                cells += [fa, fb]
            lines.append(f"| {parts[int(r.part)]} | {r.keypoint} | {r.zero_shot:.1f} | "
                         f"{verdict} ({ratio:.1f}x) | " + " | ".join(cells)
                         + f" | {r.dedicated:.1f} |")
        lines.append("")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1], epilog=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config",    required=True, type=Path)
    ap.add_argument("--arm",       default="mm-anchored-lora")
    ap.add_argument("--scratch",   default="dino-nonlinear")
    ap.add_argument("--dedicated", default="nonlinear")
    ap.add_argument("--png",       action="store_true", help="also write PNG (only when asked)")
    args = ap.parse_args()

    grid  = load_grid(args.config)
    paths = load_paths()
    data, results = Path(paths["data_dir"]), Path(paths["results_dir"])
    inv   = json.loads((data / "dataset_inventory.json").read_text())["datasets"]
    out   = results / grid.out_subdir / grid.name / "summary" / "keypoints"
    out.mkdir(parents=True, exist_ok=True)
    ns    = [n for n in grid.n_frames if n != ALL]
    lo, hi = ns[0], ns[-1]

    rows = []
    for ds, spec in grid.datasets.items():
        test = data / f"CollectedData_{ds}_test.csv"
        tdir = results / spec["trunk"]
        tcfg = yaml.safe_load((tdir / "config.yaml").read_text())
        trained = set(trained_keypoints(trunk_datasets(tdir, tcfg), inv))
        zs  = pixel_errors(tdir / "eval" / ds / "predictions.csv", test)
        ded = pixel_errors(results / grid.dedicated[args.dedicated].format(ds=ds) / "eval" / ds
                           / "predictions.csv", test)
        runs = {}
        for arm in (args.arm, args.scratch):
            for n in ns:
                cells = [cell_dir(results, grid, Cell(ds, arm, n, d)) for d in grid.draws]
                runs[(arm, n)] = [pixel_errors(c / "eval" / ds / "predictions.csv", test)
                                  for c in cells if (c / ".done").exists()]
        for k in sorted(set(inv[ds]["direct"]) & trained):
            r = {"dataset": ds, "keypoint": k, "zero_shot": mean_err([zs], k),
                 "dedicated": mean_err([ded], k)}
            for (arm, n), errs in runs.items():
                r[f"{arm}_N{n}"] = mean_err(errs, k)
            if np.isfinite(r["zero_shot"]):
                rows.append(r)
    df = pd.DataFrame(rows)
    df["zs_ratio"] = df.zero_shot / df.dedicated
    df["mm_gain"]  = df[f"{args.scratch}_N{lo}"] / df[f"{args.arm}_N{lo}"]
    df.to_csv(out / "keypoints.csv", index=False)
    write_markdown(df, out / "keypoints.md", args.arm, args.scratch, ns)

    # ── 1. per dataset: one row per supported keypoint ─────────────────────────
    for ds, d in df.groupby("dataset", sort=False):
        d = d.sort_values("zs_ratio")
        y = np.arange(len(d))
        fig, ax = plt.subplots(figsize=(7.5, 0.32 * len(d) + 1.6))
        ax.scatter(d.zero_shot, y, color=GREY, s=55, label="zero-shot (MM trunk)", zorder=3)
        ax.scatter(d.dedicated, y, color=GREEN, marker="|", s=220, lw=2.5,
                   label="dedicated (all frames)", zorder=3)
        for arm, col, mk, name in ((args.arm, PURPLE, "o", "MM + anchored LoRA"),
                                   (args.scratch, BLUE, "s", "DINO from scratch")):
            ax.scatter(d[f"{arm}_N{lo}"], y + 0.15, facecolors="none", edgecolors=col, marker=mk,
                       s=40, lw=1.5, label=f"{name}, N={lo}", zorder=4)
            ax.scatter(d[f"{arm}_N{hi}"], y + 0.15, color=col, marker=mk, s=40,
                       label=f"{name}, N={hi}", zorder=4)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{k}  ({r:.1f}x)" for k, r in zip(d.keypoint, d.zs_ratio,
                                                                 strict=True)], fontsize=8)
        ax.set_xscale("log")
        ax.set_xlabel("pixel error (mean, test frames)")
        ax.set_title(f"{ds}: supported keypoints (in brackets: zero-shot / dedicated)")
        ax.grid(alpha=0.3, axis="x", which="both")
        ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.12 - 2 / (len(d) + 6)),
                  ncol=3, frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        save(fig, out / ds, args.png)

    # ── 2. does the head start come from keypoints zero-shot already knows? ─────
    ok = df[np.isfinite(df.zs_ratio) & np.isfinite(df.mm_gain) & (df.dedicated > 0)]
    rho = ok[["zs_ratio", "mm_gain"]].rank().corr().iloc[0, 1]
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    cmap = plt.get_cmap("tab10")
    for i, (ds, d) in enumerate(ok.groupby("dataset", sort=False)):
        ax.scatter(d.zs_ratio, d.mm_gain, s=30, color=cmap(i), label=ds, alpha=0.85)
    ax.axhline(1, color="k", lw=0.8)
    ax.axvline(2.5, color="k", lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("zero-shot / dedicated error  (left = inherited prediction already good)")
    ax.set_ylabel(f"DINO from scratch / MM + anchored LoRA\nerror at N={lo}  (> 1: MM better)")
    ax.set_title(f"supported keypoints of all targets (n = {len(ok)}), Spearman rho = {rho:.2f}")
    ax.legend(fontsize=7, frameon=False, ncol=2)
    ax.grid(alpha=0.3, which="both")
    save(fig, out / "gain_scatter", args.png)
    print(f"{len(df)} supported keypoints -> {out}/ (per-dataset PDFs, gain_scatter, csv); "
          f"Spearman rho = {rho:.2f}")


if __name__ == "__main__":
    main()
