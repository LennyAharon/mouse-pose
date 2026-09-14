"""
Ensemble pixel-error machinery, extracted verbatim from
``scripts/plot_head_fixed_updated.ipynb`` so figure scripts and the notebook share one
implementation. The ensemble standard deviation is computed across seeds (for few-shot
runs, across label draws) within a condition, then conditions are compared on the same
axes.

``error_at_percentile`` is the scalar the paper reports: the pooled pixel error read off
the error-vs-ensemble-std curve at the std value retaining a given percentile of points.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


import numpy as np
import os
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr


def standardize_scorer_level(df, new_scorer='standard_scorer'):
    df.columns = pd.MultiIndex.from_tuples(
        [(new_scorer, bodypart, coord) for _, bodypart, coord in df.columns],
        names=df.columns.names,
    )
    return df


def compute_ensemble_stddev(df_preds, scorer_name='standard_scorer', keypoints=None):
    """Compute per-frame, per-keypoint ensemble std dev across models.
    
    keypoints: optional list of bodypart names to include (filters by name).
        Used to restrict mixed-model predictions to the eval dataset's keypoints.
    """
    preds = []
    for i, df in enumerate(df_preds):
        df = standardize_scorer_level(df, scorer_name)

        cols_to_keep = [
            col for col in df.columns
            if not col[2].endswith('likelihood') and 'Unnamed' not in col[2]
        ]
        if keypoints is not None:
            # ordered selection so the stddev column order matches the keypoints list
            cols_to_keep = [
                col for kp in keypoints for col in cols_to_keep if col[1] == kp
            ]
        df = df[cols_to_keep]

        if df.isna().any().any():
            nan_indices = df[df.isna().any(axis=1)].index
            nan_columns = df.columns[df.isna().any()]
            print(f'Warning: NaN values in DataFrame {i} — indices: {nan_indices}, columns: {nan_columns}')

        arr_np = df.to_numpy()
        try:
            arr = arr_np.reshape(df.shape[0], -1, 2)
        except ValueError as e:
            print(f'Reshape error: {e}  shape={arr_np.shape}')
            raise
        preds.append(arr)

    preds = np.stack(preds, axis=-1)  # (n_frames, n_keypoints, 2, n_models)

    if np.isnan(preds).any():
        print(f'Warning: NaN values in stacked preds array')
    else:
        print('No NaN values in preds array.')

    return np.std(preds, axis=-1).mean(axis=-1)  # (n_frames, n_keypoints)


def compute_percentiles(arr, std_vals, percentiles):
    num_pts = arr[0]
    vals, prctiles = [], []
    for p in percentiles:
        idx = np.argmin(np.abs(arr - num_pts * p / 100))
        achieved = arr[idx] / num_pts * 100 if idx == len(arr) - 1 else p
        vals.append(std_vals[idx])
        prctiles.append(np.round(achieved, 2))
    return vals, prctiles


class Ensemble:
    data_to_plot: dict[str, str]  # model label -> path to predictions CSV
    pred_csv_list: list[str]
    error_csv_list: list[str]
    df_pred_list: list[pd.DataFrame]
    df_error_list: list[pd.DataFrame]
    n_points_dict: dict[str, dict[str, int]]
    std_vals: list[float]
    df_w_vars: pd.DataFrame
    df_line2: pd.DataFrame


def build_ensemble(ens, std_val_max=8.0, error_csv_dict=None, keypoints=None,
                   fast=False):
    """Load predictions and pixel errors, compute ensemble statistics.
    
    error_csv_dict: optional dict mapping each key in ens.data_to_plot to an
        explicit pixel-error CSV path. If None, derives path from pred CSV stem
        (original behaviour: predictions_test.csv -> predictions_pixel_error_test.csv).
    keypoints: optional list of bodypart names passed to compute_ensemble_stddev.
    fast: skip the two 'ens-std-prctile*' columns, which are O(n_frames * n_points) per
        (model, keypoint) and are used only for exploratory plots. The error-vs-std curve
        and every percentile readout are unaffected.
    """
    model_names_list = list(ens.data_to_plot.keys())
    pred_csv_list = [Path(v) for v in ens.data_to_plot.values()]

    if error_csv_dict is not None:
        error_csv_list = [Path(error_csv_dict[k]) for k in model_names_list]
    else:
        error_csv_list = [
            p.with_stem(
                p.stem
                .replace('_new', '_pixel_error_new')
                .replace('_test', '_pixel_error_test')
            )
            for p in pred_csv_list
        ]

    df_pred_list, df_error_list = [], []
    for pred_csv, error_csv in zip(pred_csv_list, error_csv_list):
        df_pred_list.append(pd.read_csv(pred_csv, header=[0, 1, 2], index_col=0).sort_index())
        df = pd.read_csv(error_csv, header=[0], index_col=0).sort_index()
        if 'set' in df.columns:
            df = df.drop(columns=['set'])
        if keypoints is not None:
            # keep error columns in the same order as the keypoints list, so the
            # enumerate() below indexes the stddev array consistently
            df = df[[k for k in keypoints if k in df.columns]]
        df_error_list.append(df)

    ens_stddev = compute_ensemble_stddev(df_pred_list, keypoints=keypoints)

    df_w_vars = []
    for df_error, df_pred, model_name in zip(df_error_list, df_pred_list, model_names_list):
        assert (df_error.index == df_pred_list[0].index).all()
        assert (df_pred.index == df_pred_list[0].index).all()

        print(f'Total pixel error for model {model_name}: {df_error.sum().sum()}')

        for i, kp in enumerate(df_error.columns):
            index = [f'{frame}_{model_name}_{kp}' for frame in df_error.index]
            df_pred_std = standardize_scorer_level(df_pred.copy())
            df_w_vars.append(pd.DataFrame(
                {
                    'pixel_error': df_error[kp].values,
                    'likelihood': df_pred_std.loc[:, ('standard_scorer', kp, 'likelihood')].values,
                    'ens-std': ens_stddev[:, i],
                    **({} if fast else {
                        'ens-std-prctile': [
                            np.sum(ens_stddev < p) / ens_stddev.size for p in ens_stddev[:, i]
                        ],
                        'ens-std-prctile-kp': [
                            np.sum(ens_stddev[:, i] < p) / ens_stddev[:, i].size
                            for p in ens_stddev[:, i]
                        ],
                    }),
                    'keypoint': kp,
                    'model': model_name,
                },
                index=index,
            ))

    df_w_vars = pd.concat(df_w_vars)
    std_vals = np.arange(0, std_val_max, 0.25)
    n_points_dict = {m: np.nan * np.zeros_like(std_vals) for m in model_names_list}

    df_line2 = []
    for s, std in enumerate(std_vals):
        df_tmp_ = df_w_vars[df_w_vars['ens-std'] > std]
        for model_name in model_names_list:
            d = df_tmp_[df_tmp_.model == model_name]
            n_points_dict[model_name][s] = np.sum(~d['pixel_error'].isna())
            df_line2.append(pd.DataFrame(
                {
                    'ens-std': std,
                    'model': model_name,
                    'pixel_error': d.pixel_error.to_numpy(),
                    'keypoint': d['keypoint'].to_numpy(),
                },
                index=[f'{row}_{s}' for row in d.index],
            ))

    ens.pred_csv_list = pred_csv_list
    ens.error_csv_list = error_csv_list
    ens.df_pred_list = df_pred_list
    ens.df_error_list = df_error_list
    ens.n_points_dict = n_points_dict
    ens.std_vals = std_vals
    ens.df_w_vars = df_w_vars
    ens.df_line2 = pd.concat(df_line2)
    return ens

# ── scalar readout used by the paper figures ─────────────────────────────────

def error_at_percentile(ens, model: str, percentile: float = 50.0) -> float:
    """Pooled pixel error for series ``model`` at the ensemble-std threshold retaining
    ``percentile`` per cent of (frame, keypoint) points.

    ``model`` is a series label (without the ``.seed`` suffix). The threshold is read from
    the shared std grid, so every series in an ensemble is evaluated at the same operating
    point; this is the dashed vertical line the notebook draws.
    """
    key = next(k for k in ens.n_points_dict if k.rsplit(".", 1)[0] == model)
    arr = np.asarray(ens.n_points_dict[key], dtype=float)
    std_vals = np.asarray(ens.std_vals, dtype=float)
    target = compute_percentiles(arr=arr, std_vals=std_vals, percentiles=[percentile])[0][0]

    d = ens.df_line2.copy()
    d["model2"] = d["model"].apply(lambda s: s.rsplit(".", 1)[0])
    g = d[d["model2"] == model].groupby("ens-std")["pixel_error"].mean().sort_index()
    return float(np.interp(target, g.index.to_numpy(), g.to_numpy()))
