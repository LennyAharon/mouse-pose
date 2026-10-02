"""Consistent visible-keypoint method comparisons, separate from training labels."""

import numpy as np
import pandas as pd

LIP_KEYPOINTS = frozenset({"upperlip_left", "upperlip_right", "lowerlip"})
# pupil_center_right used to be excluded here because no dataset labeled it (hflip-only channel).
# Since corpus v5 (2026-09-23) facemap and cheese-3d label it directly; scoring only visible == 2
# cells already skips it wherever it is unlabeled, so nothing is excluded by name any more.
ALWAYS_EXCLUDED: frozenset[str] = frozenset()


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


def trained_keypoints(datasets: set[str], inventory: dict) -> set[str]:
    """Keypoints a model trained on these datasets supervises: union of their `trainable` sets.

    Args:
        datasets: dataset names of the model's training csv (unknown names are ignored).
        inventory: the `datasets` dict of dataset_inventory.json.
    """
    return set().union(*[set(inventory[d]["trainable"]) for d in datasets if d in inventory])


def target_scores(errors: pd.DataFrame, keypoints: list[str],
                  trained: set[str] | None = None) -> dict:
    """Per-keypoint (mean, median) px on one target dataset, plus their pooled (mean, median).

    Args:
        errors: output of `visible_errors` (NaN where not visible == 2).
        keypoints: keypoints to report, in order.
        trained: keypoints the model supervised; others are reported as "untrained" and left
            out of the pooled number. None = treat every keypoint as trained.

    Returns:
        {keypoint: (mean, median) | "untrained" | None (no labelled cell),
         "pooled": (mean, median)}
    """
    out, pooled = {}, []
    for k in keypoints:
        e = errors[k].dropna().to_numpy(dtype=float) if k in errors else np.array([])
        if trained is not None and k not in trained:
            out[k] = "untrained"
        elif len(e) == 0:
            out[k] = None
        else:
            out[k] = (float(e.mean()), float(np.median(e)))
            pooled.append(e)
    allerr = np.concatenate(pooled) if pooled else np.array([np.nan])
    out["pooled"] = (float(np.mean(allerr)), float(np.median(allerr)))
    return out
