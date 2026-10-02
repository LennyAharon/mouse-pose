#!/usr/bin/env python
"""
Per-keypoint table of many models on ONE target dataset's held-out test frames.

The fine-tune convention (eval-suite skill): when adapting to dataset X, report only X, per
keypoint, never the other datasets. Cells are visible == 2 labels (mighty_mouse.comparison
.visible_errors); each cell is "mean / median" px, plus a pooled column over the scored keypoints.
A keypoint the model never trained (not in `trainable` of the datasets of its saved training csv,
e.g. tongue for a leave-ibl-out trunk) is shown as "untrained" and left out of its pooled number.

Models come from a YAML spec (grouped, with optional frames / steps columns) or from --runs:

    title: ibl — every model on the held-out test frames
    dataset: ibl
    keypoints: [nose_tip, pupil_center_left, ...]      # optional; default: every labelled one
    groups:
      - name: Zero-shot (never saw ibl)
        runs:
          - {label: leave-ibl-out ViT-S, run: <run dir>, frames: 0}
          - {label: ..., run: ..., frames: 2000 (d0), steps: 4k}

When a run has run_info.json (scripts/anchor_ft.sh), frames and steps default to its values.

    python scripts/score_target.py --spec table.yaml --out <dir>/table.md
    python scripts/score_target.py --dataset ibl --runs "trunk=<run>" "aLoRA=<run>"
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from mighty_mouse.comparison import target_scores, trained_keypoints, visible_errors
from mighty_mouse.paths import load_paths


def read_dlc(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, header=[0, 1, 2], index_col=0)


def score(run: Path, dataset: str, labels: pd.DataFrame, kps: list[str], inv: dict) -> dict:
    """target_scores of a run's eval predictions; trained = datasets of its saved train csv."""
    pred   = read_dlc(run / "eval" / dataset / "predictions.csv")
    common = labels.index.intersection(pred.index)
    cols   = [c for c in labels.columns if c[1] in kps]
    err    = visible_errors(labels.loc[common, cols], pred.loc[common])
    csvs   = sorted(run.glob("CollectedData_*_train.csv"))
    trained = None
    if csvs:
        frames   = pd.read_csv(csvs[0], header=[0, 1, 2], index_col=0).index
        datasets = set(frames.str.split("/").str[1])
        trained  = trained_keypoints(datasets, inv)
    return target_scores(err, kps, trained)


def cell(v) -> str:
    if v is None:
        return "-"
    return v if isinstance(v, str) else f"{v[0]:.2f} / {v[1]:.2f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1], epilog=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec",     default=None, help="YAML spec (see above)")
    ap.add_argument("--dataset",  default=None, help="target (with --runs; overrides spec)")
    ap.add_argument("--runs",     nargs="+", default=[], help="label=run_dir (one table)")
    ap.add_argument("--data_dir", default=None, help="corpus data dir (default: paths.yaml)")
    ap.add_argument("--out",      default=None, help="write the markdown here (also printed)")
    args = ap.parse_args()

    spec = yaml.safe_load(open(args.spec)) if args.spec else {}
    if args.runs:
        spec.setdefault("groups", []).append(
            {"name": "", "runs": [dict(zip(("label", "run"), r.rsplit("=", 1), strict=True))
                                  for r in args.runs]})
    dataset = args.dataset or spec.get("dataset")
    assert dataset and spec.get("groups"), "need a dataset and at least one run (--spec or --runs)"
    D      = Path(args.data_dir or load_paths()["data_dir"])
    inv    = json.load(open(D / "dataset_inventory.json"))["datasets"]
    labels = read_dlc(D / f"CollectedData_{dataset}_test.csv")
    vis    = labels.xs("visible", axis=1, level=2).apply(pd.to_numeric, errors="coerce")
    kps    = spec.get("keypoints") or [k for k in vis.columns.get_level_values(-1).unique()
                                       if (vis.xs(k, axis=1, level=-1) == 2).any().any()]

    lines = [f"# {spec.get('title', f'{dataset} — held-out test frames')}", "",
             "mean / median px on visible == 2 cells; `untrained` = the model never trained that "
             "keypoint (left out of its pooled number).", ""]
    hdr = f"| model | {dataset} frames | steps | " + " | ".join(kps) + " | pooled |"
    for g in spec["groups"]:
        if g.get("name"):
            lines += [f"## {g['name']}", ""]
        lines += [hdr, "|---|---|---|" + "---|" * (len(kps) + 1)]
        for r in g["runs"]:
            run  = Path(r["run"])
            ri   = run / "run_info.json"
            info = json.load(open(ri)) if ri.exists() else {}
            frames = r.get("frames", info.get("train_frames", "-"))
            steps  = r.get("steps", info.get("steps", "-"))
            s = score(run, dataset, labels, kps, inv)
            cells = " | ".join(cell(s[k]) for k in kps)
            pooled = cell(s["pooled"])
            lines.append(f"| {r['label']} | {frames} | {steps} | {cells} | **{pooled}** |")
        lines.append("")
    text = "\n".join(lines)
    if args.out:
        Path(args.out).write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
