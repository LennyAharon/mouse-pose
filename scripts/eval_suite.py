#!/usr/bin/env python
"""
Standard evaluation tables for the recipe-of-record models of a corpus version.

Discovers the finished runs under <results_dir>/trunks and <results_dir>/dedicated (or takes
explicit --run name=path pairs) and, from the pixel_error.csv that evaluate_model writes for every
dataset's test set, produces:

  summary.csv        one row per (run, dataset): pooled median / mean pixel error over labeled
                     keypoints (visible == 2, pupil_center_right excluded), frames, keypoints
  per_keypoint.csv   one row per (run, dataset, keypoint): median / mean px error, n labels
  README.md          the summary as a table (dedicated vs all-data vs leave-one-out per dataset)

    python scripts/eval_suite.py                                  # current results_dir, auto-discover
    python scripts/eval_suite.py --results_dir <path> --out <dir>
    python scripts/eval_suite.py --run all=<run_dir> --run ded-ibl=<run_dir>
"""

import argparse
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from mouse_pose.paths import load_paths
from mouse_pose.registry import load_registry

EXCLUDE = {"pupil_center_right"}
LEAF = "supervised/sampling-T2/tf1/vits_dinov3"


def discover(results_dir: Path) -> dict[str, Path]:
    runs = {}
    for area, prefix in (("trunks", ""), ("dedicated", "dedicated:")):
        for tag_dir in sorted((results_dir / area).glob("*_train")):
            for seed_dir in sorted(tag_dir.glob(f"{LEAF}/seed*")):
                if (seed_dir / "eval").is_dir():
                    name = prefix + tag_dir.name[: -len("_train")] + (f"@{seed_dir.name}" if seed_dir.name != "seed0" else "")
                    runs[name] = seed_dir
    return runs


def md_table(piv: pd.DataFrame) -> str:
    cols = list(piv.columns)
    rows = ["| dataset | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for ds, r in piv.iterrows():
        rows.append(f"| {ds} | " + " | ".join("" if pd.isna(v) else f"{v:.2f}" for v in r) + " |")
    return "\n".join(rows)


def errors(run: Path, dataset: str) -> pd.DataFrame | None:
    f = run / "eval" / dataset / "pixel_error.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f, index_col=0)
    keep = [c for c in df.columns if c not in EXCLUDE and c != "set" and not c.startswith("Unnamed")]
    return df[keep].apply(pd.to_numeric, errors="coerce")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1], epilog=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_dir", type=Path, default=Path(load_paths()["results_dir"]))
    ap.add_argument("--run", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--out", type=Path, help="output folder (default: <results_dir>/qualitative/eval-suite-<MM-DD>)")
    args = ap.parse_args()

    runs = discover(args.results_dir)
    for spec in args.run:
        name, path = spec.split("=", 1); runs[name] = Path(path)
    if not runs:
        raise SystemExit(f"no evaluated runs under {args.results_dir}/trunks or /dedicated")
    datasets = load_registry()
    out = args.out or args.results_dir / "qualitative" / f"eval-suite-{date.today():%m-%d}"
    out.mkdir(parents=True, exist_ok=True)

    rows, kp_rows = [], []
    for name, run in runs.items():
        for ds in datasets:
            e = errors(run, ds)
            if e is None:
                continue
            vals = e.to_numpy().ravel(); vals = vals[np.isfinite(vals)]
            rows.append(dict(run=name, dataset=ds, frames=len(e), keypoints=int(e.notna().any().sum()),
                             labels=int(vals.size), median_px=float(np.median(vals)) if vals.size else np.nan,
                             mean_px=float(np.mean(vals)) if vals.size else np.nan))
            for k in e.columns:
                v = e[k].dropna()
                if len(v):
                    kp_rows.append(dict(run=name, dataset=ds, keypoint=k, n=len(v), median_px=float(v.median()), mean_px=float(v.mean())))
    S = pd.DataFrame(rows); K = pd.DataFrame(kp_rows)
    S.round(3).to_csv(out / "summary.csv", index=False); K.round(3).to_csv(out / "per_keypoint.csv", index=False)

    piv = S.pivot(index="dataset", columns="run", values="median_px").reindex(datasets)
    lines = [f"---", f"title: Evaluation suite {date.today()}", f"date: {date.today()}",
             f"data_version: {Path(load_paths()['data_dir']).name}",
             "question: How do the recipe-of-record models score on every dataset's test set?",
             f"models: {', '.join(runs)}", "outputs: summary.csv, per_keypoint.csv",
             "finding: (fill in after reading the table)", "---", "",
             "# Evaluation suite", "", "Median pixel error over labeled keypoints (visible == 2, `pupil_center_right` excluded),",
             "each run scored on every dataset's test set. A dedicated model scored on another dataset only",
             "measures whatever keypoints the two share.", "", md_table(piv.round(2)), "",
             "Per-keypoint numbers: `per_keypoint.csv`. Runs:", ""]
    lines += [f"- `{n}`: `{p.relative_to(args.results_dir) if p.is_relative_to(args.results_dir) else p}`" for n, p in runs.items()]
    (out / "README.md").write_text("\n".join(lines) + "\n")
    pd.set_option("display.width", 200); print(piv.round(2).to_string()); print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
