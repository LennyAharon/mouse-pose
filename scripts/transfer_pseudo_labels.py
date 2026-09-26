#!/usr/bin/env python
"""Transfer pseudo-labels from a trained Lightning Pose model onto another dataset's
existing labeled CSVs.

Runs the source model's inference (via the lightning_pose Python API — the `litpose
predict` CLI can't override `data_dir` for CSV inputs, which is needed here since the
CSV belongs to a different dataset than the one the model trained on) against every
image already present in the target dataset's label CSVs, then fills in (x, y) for the
requested keypoints wherever the target cell is currently empty AND the prediction's
likelihood is >= --confidence_threshold.

What this script deliberately does NOT do:
- add new rows/images (only fills cells for images already in the target CSVs)
- touch any cell that already has a label
- touch any keypoint not named in --keypoints
- modify the target dataset's schema (the keypoint names passed in --keypoints must
  already be columns in the target CSV — this never adds new keypoint columns)
- call bump_version.py — bumping the target dataset's version is a separate,
  user-requested step, after they've verified the result

Must be run in the `pose` conda env:
    conda run -n pose python scripts/transfer_pseudo_labels.py ...

See ../skills/transfer-pseudo-labels/README.md for the full workflow and the checklist
of decisions to confirm with the user before running this (keypoint-name overlap
between source model and target dataset, confidence threshold, scope, per-view
masking).

Usage example (dry run first):
    conda run -n pose python scripts/transfer_pseudo_labels.py \\
        --model_dir /media/mattw/poseinterface/results/cheese-3d/2026-09-26_15-45-24 \\
        --target_dataset cheese-2d \\
        --keypoints "pad(top)(left)" "pad(side)(left)" "pad(top)(right)" "pad(side)(right)" \\
        --confidence_threshold 0.7 \\
        --dry_run
"""
import argparse
import re
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import pandas as pd

from mouse_pose.paths import load_paths

CSV_NAMES_DEFAULT = ["CollectedData.csv", "CollectedData_test.csv"]


def guess_group(image_path: str) -> str:
    """Best-effort grouping key for the per-group inventory (e.g. camera view).

    Takes the last underscore-separated token of the image's parent directory name,
    dropping a trailing HH-MM-SS timestamp token if present. This is a generic
    heuristic, not dataset-specific: it happens to recover the camera view for the
    cheese-2d/cheese-3d naming convention (session..._<VIEW>[_HH-MM-SS]), but for a
    dataset with a different directory-naming scheme it may just recover something
    else (or nothing useful) — it's informational only, never used to decide which
    cells get filled.
    """
    parts = image_path.split("/")
    if len(parts) < 2:
        return "UNKNOWN"
    tokens = parts[-2].split("_")
    if re.fullmatch(r"\d{2}-\d{2}-\d{2}", tokens[-1]):
        tokens = tokens[:-1]
    return tokens[-1] if tokens else "UNKNOWN"


def fill_missing(orig: pd.DataFrame, preds: pd.DataFrame, keypoints: list[str], threshold: float):
    """Fill (x, y) for `keypoints` in `orig` from `preds` wherever orig is NaN and
    the prediction's likelihood >= threshold.

    Returns (filled_df, counts, cells_filled_per_row):
      - counts: {keypoint: n_cells_filled}
      - cells_filled_per_row: {image_path: n_keypoints_filled_for_this_row}, omitting
        rows with 0 fills — used only for the per-group inventory, not to decide fills.

    Raises KeyError if a keypoint is missing from either dataframe's columns.
    """
    orig = orig.copy()
    scorer = orig.columns.get_level_values(0)[0]
    pred_scorer = preds.columns.get_level_values(0)[0]

    counts: dict[str, int] = {}
    cells_filled_per_row: dict[str, int] = defaultdict(int)

    for kp in keypoints:
        x_col, y_col = (scorer, kp, "x"), (scorer, kp, "y")
        px_col, py_col, pl_col = (pred_scorer, kp, "x"), (pred_scorer, kp, "y"), (pred_scorer, kp, "likelihood")

        if x_col not in orig.columns or y_col not in orig.columns:
            raise KeyError(f"keypoint {kp!r} not found in target CSV columns")
        if px_col not in preds.columns or pl_col not in preds.columns:
            raise KeyError(f"keypoint {kp!r} not found in model predictions (model wasn't trained on it?)")

        missing_mask = orig[x_col].isna()
        pred_conf = preds.loc[orig.index, pl_col]
        fill_mask = missing_mask & (pred_conf >= threshold)

        n = int(fill_mask.sum())
        if n:
            orig.loc[fill_mask, x_col] = preds.loc[orig.index[fill_mask], px_col].values
            orig.loc[fill_mask, y_col] = preds.loc[orig.index[fill_mask], py_col].values
            for path in orig.index[fill_mask]:
                cells_filled_per_row[path] += 1
        counts[kp] = n

    return orig, counts, dict(cells_filled_per_row)


def verify_untouched(orig: pd.DataFrame, filled: pd.DataFrame) -> None:
    """Sanity check: columns/index unchanged, and every cell that was already
    labeled in `orig` is byte-for-byte unchanged in `filled`. Raises AssertionError
    on any violation — this is the safety net for an irreversible-ish edit to a
    dataset's ground-truth label file.
    """
    assert list(orig.columns) == list(filled.columns), "columns changed"
    assert list(orig.index) == list(filled.index), "row order/index changed"
    for col in orig.columns:
        was_present = orig[col].notna()
        if was_present.any():
            o = orig.loc[was_present, col].astype(float)
            f = filled.loc[was_present, col].astype(float)
            bad = (o - f).abs() > 1e-6
            if bad.any():
                raise AssertionError(f"{bad.sum()} previously-labeled cell(s) changed in column {col}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_dir", required=True, type=Path, help="path to a trained Lightning Pose model dir")
    parser.add_argument(
        "--target_dataset", required=True,
        help="name of the target dataset (must match a directory under raw_dir, see paths.yaml)",
    )
    parser.add_argument(
        "--keypoints", required=True, nargs="+",
        help="keypoint names to transfer (must already be columns in the target CSV(s), "
        "and must be keypoints the source model was trained on)",
    )
    parser.add_argument(
        "--confidence_threshold", type=float, default=0.7,
        help="only fill cells whose prediction likelihood is >= this (default: 0.7)",
    )
    parser.add_argument(
        "--csvs", nargs="+", default=CSV_NAMES_DEFAULT,
        help=f"which of the target dataset's label CSVs to touch (default: {CSV_NAMES_DEFAULT})",
    )
    parser.add_argument("--dry_run", action="store_true", help="print the inventory without writing anything")
    args = parser.parse_args()

    if not args.model_dir.is_dir():
        sys.exit(f"No such model dir: {args.model_dir}")

    raw_dir = Path(load_paths()["raw_dir"]) / args.target_dataset
    if not raw_dir.is_dir():
        sys.exit(f"No such raw dataset directory: {raw_dir}")

    for name in args.csvs:
        if not (raw_dir / name).exists():
            sys.exit(f"Missing {name} in {raw_dir}")

    # Delayed: slow import, and only available in the `pose` conda env.
    from lightning_pose.api import Model

    model = Model.from_dir2(args.model_dir)

    grand_counts: dict[str, dict[str, int]] = {}
    grand_group_counts: dict[str, int] = defaultdict(int)
    filled_frames: dict[str, pd.DataFrame] = {}

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for name in args.csvs:
            orig_path = raw_dir / name
            orig = pd.read_csv(orig_path, header=[0, 1, 2], index_col=0)

            # Copy under a name distinct from every csv this model dir has ever predicted
            # on before (predict_on_label_csv keys its output dir purely off the csv
            # basename — reusing "CollectedData.csv" would silently overwrite the model's
            # own self-eval predictions from training).
            temp_csv = tmp / f"{args.target_dataset}__{name}"
            temp_csv.write_bytes(orig_path.read_bytes())

            result = model.predict_on_label_csv(
                csv_file=temp_csv,
                data_dir=raw_dir,
                compute_metrics=False,
                add_train_val_test_set=False,
            )
            preds = result.predictions

            filled, counts, cells_filled_per_row = fill_missing(orig, preds, args.keypoints, args.confidence_threshold)
            verify_untouched(orig, filled)

            for path, n in cells_filled_per_row.items():
                grand_group_counts[guess_group(path)] += n

            grand_counts[name] = counts
            filled_frames[name] = filled

            print(f"predictions written to: {model.image_preds_dir() / temp_csv.name / 'predictions.csv'}")

    print("\n=== fill inventory (cells filled, previously empty + confidence >= "
          f"{args.confidence_threshold}) ===")
    keypoint_totals: dict[str, int] = defaultdict(int)
    for name, counts in grand_counts.items():
        total = sum(counts.values())
        print(f"\n{name}: {total} cells")
        for kp, n in counts.items():
            if n:
                print(f"  {kp}: +{n}")
            keypoint_totals[kp] += n
    print(f"\ntotal cells filled across all CSVs: {sum(keypoint_totals.values())}")
    print("per-keypoint total:")
    for kp, n in keypoint_totals.items():
        print(f"  {kp}: {n}")
    print("\nper-group (best-effort, see guess_group docstring):")
    for group, n in sorted(grand_group_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {group}: {n}")

    if args.dry_run:
        print("\n(dry run — nothing written)")
        return

    for name, filled in filled_frames.items():
        filled.to_csv(raw_dir / name)
    print(f"\nWrote updated CSVs: {[str(raw_dir / n) for n in args.csvs]}")
    print("Note: version NOT bumped — run bump_version.py yourself once you've verified the result.")


if __name__ == "__main__":
    main()
