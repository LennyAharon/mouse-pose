"""Item 4 step 2: cross-lab calibration of zero-shot confidence (no training of the pose model).

Input: per (frame, keypoint) signals on a HELD-OUT lab's test frames, from scripts/confidence_readouts.py
(readouts__<case>__<ds>.csv) and scripts/zoom_consistency.py (zoom__<case>__<ds>.csv). Only keypoints the trunk trained
are used (union of `trainable` of its datasets).

Target "trustworthy": 1 if the keypoint is labelled visible AND the prediction is within 5 % of the frame width,
0 if labelled visible but further, or labelled absent (visible = 1). Unlabelled cells are dropped (no answer).

A logistic calibrator on the signals is FIT on one held-out lab and TESTED on the other (e.g. fit on kaufman zero-shot,
test on ibl zero-shot) — the calibrator never sees the lab it is scored on. Compared with today's confidence:
  AUROC (ranking trustworthy vs not), ECE (10 bins; is "0.8" right 80 % of the time), Brier score, and at a 0.7 cut-off:
  % of absent cells called present and % of visible-but-wrong cells kept.

    python scripts/calibrate_crosslab.py --dir <eval dir> --pairs "fit_case:fit_ds>test_case:test_ds" ...
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

FRAME_W = {"kaufman": 800, "ibl": 320, "facemap": 400, "cazettes-side": 750, "cheese-3d": 640, "cheese-2d": 640,
           "kondo": 600, "hantman-mv": 640}
HELD_OUT_TRAINED = None


def load(d: Path, case: str, ds: str, data_dir: Path) -> pd.DataFrame:
    r = pd.read_csv(d / "csv" / f"readouts__{case}__{ds}.csv")
    z = pd.read_csv(d / "csv" / f"zoom__{case}__{ds}.csv")
    m = r.merge(z, on=["frame", "keypoint"], how="left")
    inv = json.load(open(data_dir / "dataset_inventory.json"))["datasets"]
    trained = set().union(*[set(inv[x]["trainable"]) for x in inv if x != ds])
    m = m[m.keypoint.isin(trained) & m.visible.isin([1, 2])].copy()
    tau = 0.05 * FRAME_W[ds]
    m["y"] = ((m.visible == 2) & (m.err <= tau)).astype(int)
    m["zoom_frac"] = np.log1p(100 * m.zoom_disp / FRAME_W[ds])      # displacement as % of frame width, log scale
    return m


FEATURES = ["conf_lp", "peak", "mass_r1", "mass_r2", "mass_r4", "neg_entropy", "peak_ratio", "zoom_frac", "conf_zoomed"]


def auroc(y, s):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y, s) if 0 < y.mean() < 1 else np.nan


def ece(y, p, bins=10):
    e, edges = 0.0, np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi) if hi < 1 else (p >= lo) & (p <= hi)
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return e


def report(name: str, df: pd.DataFrame, p: np.ndarray) -> dict:
    y = df.y.values
    absent, wrong = (df.visible == 1).values, ((df.visible == 2) & (df.y == 0)).values
    return {"confidence": name, "AUROC": auroc(y, p), "ECE": ece(y, p), "Brier": np.mean((p - y) ** 2),
            "absent called present % (>=0.7)": 100 * (p[absent] >= 0.7).mean() if absent.any() else np.nan,
            "visible-but-wrong kept % (>=0.7)": 100 * (p[wrong] >= 0.7).mean() if wrong.any() else np.nan,
            "trustworthy kept % (>=0.7)": 100 * (p[y == 1] >= 0.7).mean()}


def main() -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    ap = argparse.ArgumentParser(epilog=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir",      required=True, type=Path)
    ap.add_argument("--data_dir", default="/teamspace/studios/this_studio/poseinterface/data/head-fixed-v9", type=Path)
    ap.add_argument("--pairs",    required=True, nargs="+", help="fit_case:fit_ds>test_case:test_ds")
    a = ap.parse_args()
    rows = []
    for pair in a.pairs:
        (fc, fd), (tc, td) = (s.split(":") for s in pair.split(">"))
        fit, test = load(a.dir, fc, fd, a.data_dir), load(a.dir, tc, td, a.data_dir)
        fit, test = fit.dropna(subset=FEATURES), test.dropna(subset=FEATURES)
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(fit[FEATURES].values, fit.y.values)
        tag = f"fit {fc} -> test {tc}"
        rows.append({"pair": tag, **report("today's confidence (raw)", test, test.conf_lp.values)})
        rows.append({"pair": tag, **report("mass within 2 heatmap px (raw)", test, test.mass_r2.values)})
        rows.append({"pair": tag, **report("cross-lab calibrator (all signals)", test, clf.predict_proba(test[FEATURES].values)[:, 1])})
        coefs = dict(zip(FEATURES, np.round(clf[-1].coef_[0], 2)))
        print(f"{tag}: n_fit {len(fit)} ({100*fit.y.mean():.0f} % trustworthy), n_test {len(test)} ({100*test.y.mean():.0f} %); weights {coefs}")
    t = pd.DataFrame(rows)
    out = a.dir / "csv" / "crosslab_calibration.csv"
    t.to_csv(out, index=False)
    print(t.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
