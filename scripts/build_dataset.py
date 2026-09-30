#!/usr/bin/env python3
"""
Build the combined training dataset by subsampling from pre-converted per-dataset CSVs.

Reads CollectedData_<dataset>_train.csv and CollectedData_<dataset>_test.csv
from data_dir (produced by convert_dataset.py), subsamples train frames, and
merges across datasets. Keypoints absent from a given dataset get visible=0.

Produces (in data_dir):
  CollectedData_<tag>_train.csv  merged train labels
  CollectedData_<tag>_test.csv   merged test labels (all frames, no subsampling)

--n_frames caps how many frames each dataset contributes (same cap for every
dataset in --datasets, so merges stay balanced — see docs/build_dataset.md).
Pass --n_frames -1 to instead take every available frame from every dataset,
unbalanced. By convention tags built this way drop the frame count from their
name (e.g. "face+cheese"), while balanced tags include it (e.g. "face+cheese-600").

Usage:
  python scripts/build_dataset.py --tag face+ibl-600 --datasets facemap ibl --n_frames 600
  python scripts/build_dataset.py --tag face+ibl     --datasets facemap ibl --n_frames -1
  python scripts/build_dataset.py --tag face+ibl200+cheese --datasets facemap ibl cheese-2d --n_frames -1 --cap ibl=200
"""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from mighty_mouse.datasets import ALL_DATASETS
from mighty_mouse.labels import read_labels_csv
from mighty_mouse.paths import load_paths

_paths   = load_paths()
DATA_DIR = Path(_paths["data_dir"])


def build_merged(per_dataset_dfs: list[tuple[str, pd.DataFrame]]) -> pd.DataFrame:
    """Concatenate per-dataset DataFrames. All CSVs share identical columns so a plain concat suffices."""
    return pd.concat([df for _, df in per_dataset_dfs])


def _dataset_rng(seed: int, name: str) -> np.random.Generator:
    """Independent RNG per dataset — same frames regardless of which other datasets are included."""
    h = int(hashlib.sha256(name.encode()).hexdigest(), 16) % (2 ** 32)
    return np.random.default_rng([seed, h])


def main(datasets: list[str], n_frames: int, seed: int, tag: str, caps: dict[str, int] | None = None,
         masks: dict[str, str] | None = None) -> None:
    caps  = caps or {}
    masks = masks or {}
    train_dfs: list[tuple[str, pd.DataFrame]] = []
    test_dfs:  list[tuple[str, pd.DataFrame]] = []

    for name in datasets:
        train_csv = DATA_DIR / f"CollectedData_{name}_train.csv"
        test_csv  = DATA_DIR / f"CollectedData_{name}_test.csv"

        if not train_csv.exists():
            print(f"WARNING: {train_csv.name} not found — skipping {name}")
            continue

        print(f"\n── {name} ──────────────────────────────────────")
        train_df = read_labels_csv(train_csv)
        print(f"  Train: {len(train_df)} frames available")

        if n_frames < 0:
            n = len(train_df)
        else:
            n = min(n_frames, len(train_df))
            if n < n_frames:
                print(f"  WARNING: only {n} frames available (requested {n_frames})")
        rng    = _dataset_rng(seed, name)
        if name in caps:
            # per-dataset cap: first N of one fixed shuffle, so smaller caps are subsets of larger ones
            n   = min(caps[name], len(train_df))
            idx = rng.permutation(len(train_df))[:n]
            print(f"  cap {caps[name]} (nested across caps)")
        else:
            idx = rng.choice(len(train_df), size=n, replace=False)
        sample = train_df.iloc[sorted(idx)]
        print(f"  Sampled {len(sample)} frames")
        if name in masks:
            # hide these keypoints' labels in the TRAIN frames only (x, y -> NaN, visible -> 0 = unlabeled,
            # no loss); test labels stay, so the hidden keypoints can still be scored on test frames
            import re
            sample = sample.copy()
            kps = [k for k in dict.fromkeys(sample.columns.get_level_values(1)) if re.search(masks[name], k)]
            for col in sample.columns:
                if col[1] in kps:
                    sample[col] = 0 if col[2] == "visible" else np.nan
            print(f"  masked (train only, visible -> 0): {kps}")
        train_dfs.append((name, sample))

        if test_csv.exists():
            test_df = read_labels_csv(test_csv)
            print(f"  Test:  {len(test_df)} frames")
            test_dfs.append((name, test_df))
        else:
            print(f"  WARNING: {test_csv.name} not found — skipping test split for {name}")

    if not train_dfs:
        print("ERROR: no datasets loaded.")
        return

    print("\n── merged train ────────────────────────────────────")
    merged_train = build_merged(train_dfs)
    merged_train_path = DATA_DIR / f"CollectedData_{tag}_train.csv"
    merged_train.to_csv(merged_train_path)
    print(f"  {len(merged_train)} rows → {merged_train_path.name}")

    if test_dfs:
        print("\n── merged test ─────────────────────────────────────")
        merged_test = build_merged(test_dfs)
        merged_test_path = DATA_DIR / f"CollectedData_{tag}_test.csv"
        merged_test.to_csv(merged_test_path)
        print(f"  {len(merged_test)} rows → {merged_test_path.name}")

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build combined dataset from pre-converted per-dataset CSVs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--tag", required=True,
        help='label for output CSVs, e.g. "all" → CollectedData_all_{train,test}.csv',
    )
    parser.add_argument(
        "--datasets", nargs="+", default=ALL_DATASETS,
        help=f"Datasets to include (default: {ALL_DATASETS})",
    )
    parser.add_argument(
        "--n_frames", type=int, default=600,
        help="frames to sample per dataset (default: 600); -1 = use every available frame, no subsampling",
    )
    parser.add_argument("--seed",     type=int, default=42,  help="random seed (default: 42)")
    parser.add_argument(
        "--cap", nargs="*", default=[], metavar="DATASET=N",
        help="per-dataset frame cap overriding --n_frames for that dataset, e.g. --cap ibl=200; frames are the "
             "first N of one fixed shuffle, so ibl=200 is a subset of ibl=1000",
    )
    parser.add_argument(
        "--mask", nargs="*", default=[], metavar="DATASET=REGEX",
        help="hide keypoints matching REGEX in that dataset's TRAIN frames (visible -> 0, unlabeled); test labels "
             "are kept, e.g. --mask kaufman=^ear_",
    )
    args = parser.parse_args()
    caps  = {k: int(v) for k, v in (c.split("=") for c in args.cap)}
    masks = dict(m.split("=", 1) for m in args.mask)
    main(args.datasets, args.n_frames, args.seed, args.tag, caps, masks)
