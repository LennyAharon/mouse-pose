"""Consistent visible-keypoint method comparisons, separate from training labels."""

import numpy as np
import pandas as pd

LIP_KEYPOINTS = frozenset({"upperlip_left", "upperlip_right", "lowerlip"})
ALWAYS_EXCLUDED = frozenset({"pupil_center_right"})


def visible_errors(labels: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    """Align frames/channels and mask unknown, occluded and augmentation-only targets."""
    if not labels.index.is_unique or not predictions.index.is_unique:
        raise ValueError("Frame indices must be unique")
    if set(labels.index) != set(predictions.index):
        raise ValueError("Prediction and label frames differ")
    predictions = predictions.loc[labels.index]
    keys = labels.columns.get_level_values(1).unique()
    result = {}
    for key in keys:
        gt = labels.xs(key, axis=1, level=1).droplevel(0, axis=1)
        pred = predictions.xs(key, axis=1, level=1).droplevel(0, axis=1)
        xy = gt[["x", "y"]].to_numpy(dtype=float)
        actual = pred[["x", "y"]].to_numpy(dtype=float)
        visible = gt["visible"].to_numpy() == 2 if "visible" in gt else np.isfinite(xy).all(1)
        if key in ALWAYS_EXCLUDED:
            visible[:] = False
        if not np.isfinite(xy[visible]).all() or not np.isfinite(actual[visible]).all():
            raise ValueError(f"Nonfinite visible coordinates for {key}")
        error = np.linalg.norm(actual - xy, axis=1)
        error[~visible] = np.nan
        result[key] = error
    return pd.DataFrame(result, index=labels.index)


def comparison_summary(errors: pd.DataFrame, diagonals: np.ndarray) -> dict:
    """Return original and lip-excluded scores; retain mouth and all raw columns."""
    diagonals = np.asarray(diagonals, dtype=float)
    if diagonals.shape != (len(errors),) or not np.isfinite(diagonals).all() or (diagonals <= 0).any():
        raise ValueError("One finite positive image diagonal is required per frame")
    result = {}
    for view, excluded in [("all", ALWAYS_EXCLUDED), ("no_lips", ALWAYS_EXCLUDED | LIP_KEYPOINTS)]:
        values = errors.drop(columns=list(excluded), errors="ignore").to_numpy(dtype=float)
        norm = values / diagonals[:, None]
        count = int(np.isfinite(values).sum())
        populated = np.isfinite(values).any(axis=0)
        result[view] = dict(
            points=count, mean_px=float(np.nanmean(values)) if count else None,
            normalized_mean=float(np.nanmean(norm)) if count else None,
            equal_keypoint_normalized_mean=float(np.nanmean(np.nanmean(norm[:, populated], axis=0))) if count else None,
            excluded=sorted(excluded),
        )
    return result
