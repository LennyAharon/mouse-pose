"""
Pixel-error scoring for evaluation runs and the comparison tables.

Only labeled cells are scored: visible == 2 in the converted labels (1 = occluded and
0 = unlabeled are never scored). The tongue and lip keypoints are left out of every pixel-error
comparison by default (ALWAYS_EXCLUDED): they are still trained, predicted and drawn in videos.
"""

import numpy as np
import pandas as pd

LIP_KEYPOINTS    = frozenset({"upperlip_left", "upperlip_right", "lowerlip"})
TONGUE_KEYPOINTS = frozenset({"tongue_tip", "tongue_center", "tongue_end_left",
                              "tongue_end_right"})
ALWAYS_EXCLUDED: frozenset[str] = TONGUE_KEYPOINTS | LIP_KEYPOINTS


def visible_errors(
    labels: pd.DataFrame, predictions: pd.DataFrame, exclude: frozenset[str] = ALWAYS_EXCLUDED,
) -> pd.DataFrame:
    """Per-frame, per-keypoint pixel error over the labeled cells.

    Args:
        labels: converted labels with (scorer, keypoint, x|y|visible) columns; without a
            visible column a finite (x, y) counts as labeled
        predictions: (scorer, keypoint, x|y|likelihood) columns for the same frames
        exclude: keypoints never scored (their column is all NaN)

    Returns:
        DataFrame indexed like labels, one column per keypoint, NaN where not scored

    Raises:
        ValueError: if frame indices repeat or differ between labels and predictions, or a
            scored cell has a non-finite coordinate
    """
    if not labels.index.is_unique or not predictions.index.is_unique:
        raise ValueError("Frame indices must be unique")
    if set(labels.index) != set(predictions.index):
        raise ValueError("Prediction and label frames differ")
    predictions = predictions.loc[labels.index]
    result = {}
    for key in labels.columns.get_level_values(1).unique():
        gt      = labels.xs(key, axis=1, level=1).droplevel(0, axis=1)
        pred    = predictions.xs(key, axis=1, level=1).droplevel(0, axis=1)
        xy      = gt[["x", "y"]].to_numpy(dtype=float)
        actual  = pred[["x", "y"]].to_numpy(dtype=float)
        visible = gt["visible"].to_numpy() == 2 if "visible" in gt else np.isfinite(xy).all(1)
        if key in exclude:
            visible[:] = False
        if not np.isfinite(xy[visible]).all() or not np.isfinite(actual[visible]).all():
            raise ValueError(f"Nonfinite visible coordinates for {key}")
        error = np.linalg.norm(actual - xy, axis=1)
        error[~visible] = np.nan
        result[key] = error
    return pd.DataFrame(result, index=labels.index)


def comparison_summary(errors: pd.DataFrame, diagonals: np.ndarray) -> dict:
    """Pooled error of one evaluation, in pixels and relative to each frame's image diagonal.

    Args:
        errors: output of visible_errors (NaN where not scored)
        diagonals: image diagonal in pixels, one per row of errors

    Returns:
        {"points": scored cells, "mean_px", "median_px", "normalized_mean" (mean of
        error / diagonal), "equal_keypoint_normalized_mean" (the same, averaged per keypoint
        first)}; the means are None when no cell is scored

    Raises:
        ValueError: unless there is one finite, positive diagonal per frame
    """
    diagonals = np.asarray(diagonals, dtype=float)
    if (diagonals.shape != (len(errors),) or not np.isfinite(diagonals).all()
            or (diagonals <= 0).any()):
        raise ValueError("One finite positive image diagonal is required per frame")
    values    = errors.to_numpy(dtype=float)
    norm      = values / diagonals[:, None]
    count     = int(np.isfinite(values).sum())
    populated = np.isfinite(values).any(axis=0)
    if not count:
        return dict(points=0, mean_px=None, median_px=None, normalized_mean=None,
                    equal_keypoint_normalized_mean=None)
    return dict(
        points=count,
        mean_px=float(np.nanmean(values)),
        median_px=float(np.nanmedian(values)),
        normalized_mean=float(np.nanmean(norm)),
        equal_keypoint_normalized_mean=float(np.nanmean(np.nanmean(norm[:, populated], axis=0))),
    )


def trained_keypoints(datasets: set[str], inventory: dict) -> set[str]:
    """Keypoints a model trained on these datasets supervises: the union of their `trainable` sets.

    Args:
        datasets: names of the datasets in the model's training CSV (unknown names are ignored)
        inventory: the `datasets` dict of <data_dir>/dataset_inventory.json
            (python -m mighty_mouse.preprocess.inventory)

    Returns:
        the supervised keypoint names
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


def visible_residuals(
    labels: pd.DataFrame, predictions: pd.DataFrame, exclude: frozenset[str] = ALWAYS_EXCLUDED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Signed per-frame, per-keypoint residuals (prediction - label) over the labeled cells.

    Args:
        labels: converted labels, as for visible_errors
        predictions: (scorer, keypoint, x|y|likelihood) columns for the same frames
        exclude: keypoints never scored (their columns are all NaN)

    Returns:
        (dx, dy): two DataFrames indexed like labels, one column per keypoint, NaN where
        visible_errors is NaN
    """
    err = visible_errors(labels, predictions, exclude)   # validates frames and coordinates
    predictions = predictions.loc[labels.index]
    dx, dy = {}, {}
    for key in err.columns:
        gt   = labels.xs(key, axis=1, level=1).droplevel(0, axis=1)
        pred = predictions.xs(key, axis=1, level=1).droplevel(0, axis=1)
        mask = err[key].notna().to_numpy()
        dx[key] = np.where(mask, pred["x"].to_numpy(float) - gt["x"].to_numpy(float), np.nan)
        dy[key] = np.where(mask, pred["y"].to_numpy(float) - gt["y"].to_numpy(float), np.nan)
    return pd.DataFrame(dx, index=labels.index), pd.DataFrame(dy, index=labels.index)


def offset_corrected_errors(
    labels: pd.DataFrame, predictions: pd.DataFrame, groups: pd.Series | None = None,
    seed: int = 0, min_frames: int = 5, exclude: frozenset[str] = ALWAYS_EXCLUDED,
) -> pd.DataFrame:
    """Pixel errors after removing a constant labeling-convention shift, cross-validated.

    The frames are split at random into two halves. For each keypoint (and group, e.g. camera
    view) the shift is the median residual (prediction - label) on one half; it is subtracted
    from the other half's residuals, and the halves swap. What remains is the model's
    localization error once a lab's constant way of clicking the point is removed. It uses the
    target's labels, so it is a diagnostic next to the raw zero-shot error, not a zero-shot
    number itself.

    Args:
        labels: converted labels, as for visible_errors
        predictions: predictions for the same frames
        groups: per-frame group (same index as labels) with its own shift per keypoint, e.g. the
            camera view; None = one group
        seed: seed of the random split
        min_frames: a (group, keypoint) needs at least this many labeled frames in the fitting
            half for a shift; otherwise its errors are left uncorrected
        exclude: keypoints never scored

    Returns:
        DataFrame like visible_errors (NaN where not scored)
    """
    dx, dy = visible_residuals(labels, predictions, exclude)
    g = (pd.Series("all", index=labels.index) if groups is None
         else groups.reindex(labels.index).astype(str))
    rng = np.random.default_rng(seed)
    fold = pd.Series(rng.permutation(len(labels)) % 2, index=labels.index)
    out = np.hypot(dx, dy)
    for f in (0, 1):
        fit, apply = fold != f, fold == f
        for grp in g.unique():
            fg, ag = fit & (g == grp), apply & (g == grp)
            for key in dx.columns:
                fx, fy = dx.loc[fg, key].dropna(), dy.loc[fg, key].dropna()
                if len(fx) < min_frames:
                    continue
                ox, oy = fx.median(), fy.median()
                out.loc[ag, key] = np.hypot(dx.loc[ag, key] - ox, dy.loc[ag, key] - oy)
    return out
