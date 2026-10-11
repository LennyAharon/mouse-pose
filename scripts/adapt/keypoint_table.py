#!/usr/bin/env python
"""
Per-keypoint zero-shot table of an adaptation grid: for every target dataset and every keypoint it
labels, how well the leave-that-dataset-out trunk does with no adaptation, next to the dedicated
model (trained on all of the lab's frames) and anchored LoRA at the grid's N values.

    python scripts/adapt/keypoint_table.py --config configs/adaptation/cosyne-2026.yaml \
        [--dedicated nonlinear] [--arm mm-anchored-lora] [--good 2.5]

Errors are on the target's test frames, visible == 2 labels only. `class`: supported = some other
lab labels the keypoint (the trunk trained it), new = only this lab does (zero-shot has no trained
channel). `ratio` = zero-shot mean px / dedicated mean px on that keypoint; `zero-shot` verdict:
good (ratio <= --good, the masked-bar rule), poor (<= 3x that), failed (above), new. Writes
summary/zero_shot_keypoints.csv and .md (one section per dataset, worst keypoints first).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from mighty_mouse.adaptation import (
    ALL,
    Cell,
    cell_dir,
    load_grid,
    pixel_errors,
    trained_keypoints,
    trunk_datasets,
)
from mighty_mouse.paths import load_paths


def stats(errors: pd.DataFrame, k: str) -> tuple[float, float, int]:
    v = errors[k].dropna() if k in errors.columns else pd.Series(dtype=float)
    return (float(v.median()), float(v.mean()), len(v)) if len(v) else (np.nan, np.nan, 0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1], epilog=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config",    required=True, type=Path)
    ap.add_argument("--dedicated", default="nonlinear", help="head of the dedicated reference")
    ap.add_argument("--arm",       default="mm-anchored-lora", help="adaptation arm shown")
    ap.add_argument("--good",      default=2.5, type=float, help="max zero-shot/dedicated ratio")
    args = ap.parse_args()

    grid  = load_grid(args.config)
    paths = load_paths()
    data, results = Path(paths["data_dir"]), Path(paths["results_dir"])
    inv   = json.loads((data / "dataset_inventory.json").read_text())["datasets"]
    out   = results / grid.out_subdir / grid.name / "summary"
    ns    = [n for n in grid.n_frames if n != ALL]

    rows = []
    for ds, spec in grid.datasets.items():
        test = data / f"CollectedData_{ds}_test.csv"
        tdir = results / spec["trunk"]
        tcfg = yaml.safe_load((tdir / "config.yaml").read_text())
        trained = set(trained_keypoints(trunk_datasets(tdir, tcfg), inv))
        zs  = pixel_errors(tdir / "eval" / ds / "predictions.csv", test)
        ded = pixel_errors(results / grid.dedicated[args.dedicated].format(ds=ds) / "eval" / ds
                           / "predictions.csv", test)
        adapted = {}
        for n in ns:
            runs = [cell_dir(results, grid, Cell(ds, args.arm, n, d)) for d in grid.draws]
            runs = [r for r in runs if (r / ".done").exists()]
            adapted[n] = [pixel_errors(r / "eval" / ds / "predictions.csv", test) for r in runs]
        for k in sorted(inv[ds]["direct"]):
            zmed, zmean, npts = stats(zs, k)
            if npts == 0:
                continue
            _, dmean, _ = stats(ded, k)
            row = {"dataset": ds, "keypoint": k,
                   "class": "supported" if k in trained else "new", "n_points": npts,
                   "zero_shot_median_px": zmed, "zero_shot_mean_px": zmean,
                   "dedicated_mean_px": dmean, "ratio": zmean / dmean}
            for n in ns:
                vals = [stats(e, k)[1] for e in adapted[n]]
                row[f"{args.arm}_N{n}_mean_px"] = float(np.mean(vals)) if vals else np.nan
            r = row["ratio"]
            row["zero_shot"] = ("new" if k not in trained else "good" if r <= args.good
                                else "poor" if r <= 3 * args.good else "failed")
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out / "zero_shot_keypoints.csv", index=False)

    lines = [f"# {grid.name}: zero-shot per keypoint (test frames, visible == 2)", "",
             f"zero-shot = leave-that-dataset-out trunk, no adaptation; dedicated = "
             f"{args.dedicated} head, all of the lab's frames; ratio = zero-shot / dedicated mean "
             f"px; good <= {args.good:g}x, poor <= {3 * args.good:g}x, failed above; new = no "
             f"other lab labels it. Last columns: {args.arm} mean px (mean over draws).", ""]
    for ds, d in df.groupby("dataset", sort=False):
        d = d.sort_values(["zero_shot", "ratio"], key=lambda c: c.map(
            {"new": 0, "failed": 1, "poor": 2, "good": 3}) if c.name == "zero_shot" else -c)
        counts = d.zero_shot.value_counts().to_dict()
        lines += [f"## {ds}: " + ", ".join(f"{counts.get(v, 0)} {v}"
                                           for v in ("good", "poor", "failed", "new")), "",
                  "| keypoint | zero-shot | ratio | zero-shot median / mean px | dedicated px | "
                  + " | ".join(f"N={n}" for n in ns) + " | test pts |",
                  "|---|---|---|---|---|" + "---|" * len(ns) + "---|"]
        for _, r in d.iterrows():
            adapt = [r.get(f"{args.arm}_N{n}_mean_px", np.nan) for n in ns]
            lines.append(f"| {r.keypoint} | {r.zero_shot} | {r.ratio:.1f} | "
                         f"{r.zero_shot_median_px:.1f} / {r.zero_shot_mean_px:.1f} | "
                         f"{r.dedicated_mean_px:.1f} | "
                         + " | ".join("" if np.isnan(a) else f"{a:.1f}" for a in adapt)
                         + f" | {r.n_points} |")
        lines.append("")
    (out / "zero_shot_keypoints.md").write_text("\n".join(lines) + "\n")
    print(f"{len(df)} keypoints -> {out}/zero_shot_keypoints.csv, .md")


if __name__ == "__main__":
    main()
