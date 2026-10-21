"""
Subsample and merge converted per-dataset CSVs into a combined training set.

The CLI is scripts/build_dataset.py; see docs/build_dataset.md for the design (balanced
vs. full tags, why test sets are never subsampled).
"""

import hashlib
import re

import numpy as np
import pandas as pd


def dataset_rng(seed: int, name: str) -> np.random.Generator:
    """Independent RNG per dataset.

    Seeded from both ``seed`` and the dataset name, so a dataset samples the same frames
    for a given seed regardless of which other datasets are included in a build.
    """
    h = int(hashlib.sha256(name.encode()).hexdigest(), 16) % (2 ** 32)
    return np.random.default_rng([seed, h])


def subsample(df: pd.DataFrame, n_frames: int, rng: np.random.Generator) -> pd.DataFrame:
    """Sample up to ``n_frames`` rows without replacement, keeping the original row order.

    Args:
        df: one dataset's converted train split
        n_frames: rows to sample; capped at ``len(df)``. Negative means every row, in
            which case ``rng`` is not used
        rng: see dataset_rng()

    Returns:
        the sampled rows
    """
    if n_frames < 0:
        return df
    idx = rng.choice(len(df), size=min(n_frames, len(df)), replace=False)
    return df.iloc[sorted(idx)]


def subsample_nested(df: pd.DataFrame, n_frames: int, rng: np.random.Generator) -> pd.DataFrame:
    """Take the first ``n_frames`` rows of one fixed shuffle, keeping the original row order.

    Unlike subsample(), smaller caps are subsets of larger ones for the same ``rng`` seed
    (e.g. ibl=200 is a subset of ibl=1000), so a cap sweep varies only the frame count.

    Args:
        df: one dataset's converted train split
        n_frames: rows to keep; capped at ``len(df)``
        rng: see dataset_rng()

    Returns:
        the sampled rows
    """
    idx = rng.permutation(len(df))[:min(n_frames, len(df))]
    return df.iloc[sorted(idx)]


def mask_keypoints(df: pd.DataFrame, pattern: str) -> tuple[pd.DataFrame, list[str]]:
    """Hide the labels of keypoints whose name matches ``pattern``.

    x, y -> NaN and visible -> 0 (unlabeled, no loss). Meant for TRAIN frames only, so the
    hidden keypoints can still be scored on the test split.

    Args:
        df: converted labels with (scorer, bodypart, coord) columns
        pattern: regex searched against each keypoint name

    Returns:
        a masked copy of ``df`` and the masked keypoint names
    """
    df  = df.copy()
    kps = [k for k in dict.fromkeys(df.columns.get_level_values(1)) if re.search(pattern, k)]
    for col in df.columns:
        if col[1] in kps:
            df[col] = 0 if col[2] == "visible" else np.nan
    return df, kps


def build_merged(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate per-dataset DataFrames. Converted CSVs share identical columns (every
    canonical keypoint, see convert.reindex_to_canonical), so a plain concat suffices."""
    return pd.concat(dfs)
