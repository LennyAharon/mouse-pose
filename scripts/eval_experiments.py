"""Shared evaluation for the v9 experiments (plans/v9-experiments-2026-09-30.md): one table per measure, one row per run.

Reads each run's saved test predictions (<run>/eval/<dataset>/predictions.csv); no inference.

  1. ibl ear transfer (ibl has no ear labels): per ear keypoint, % of ibl test frames with confidence >= 0.7, and the
     % of those that are geometrically plausible relative to ibl's LABELLED pupil and nose (behind the pupil along the
     nose->pupil axis, 0.3-2.5 nose-pupil distances from the pupil).
  2. kaufman ear zero-shot (kaufman test frames label the right ears on every frame): per right-ear keypoint, mean /
     median px, detection (conf >= 0.7), confident-and-wrong (conf >= 0.7 and error > 5 % of frame width = 40 px).
     Also kaufman's other labelled keypoints (pooled) — the "supported keypoints" side.
  3. absent cells (visible == 1 on test frames): % predicted with conf >= 0.7 (false positives) and the AUROC of
     confidence separating visible (2) from absent (1) cells, per dataset that has absent cells.
  4. in-domain accuracy: pooled mean / median px per dataset on visible cells.

    python scripts/eval_experiments.py --runs "name=<run_dir>" ... --out <dir> [--measures 1,2,3,4]
Writes <out>/csv/eval_<measure>.csv and prints markdown tables.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

DATASETS = ["facemap", "ibl", "cheese-2d", "cazettes-side", "kondo", "hantman-mv", "cheese-3d", "kaufman"]
CONF = 0.7
WRONG_FRAC = 0.05          # "wrong" = error > 5 % of frame width
FRAME_W = {"kaufman": 800, "ibl": 320, "facemap": 400, "cazettes-side": 750, "cheese-3d": 640, "cheese-2d": 640,
           "kondo": 600, "hantman-mv": 640}


def read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, header=[0, 1, 2], index_col=0)


def keypoints(df: pd.DataFrame, field: str) -> list[str]:
    s = df.columns[0][0]
    return [k for k in dict.fromkeys(df.columns.get_level_values(1)) if (s, k, field) in df.columns]


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(score of a random positive > score of a random negative), ties count 1/2 (Mann-Whitney U)."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    scores = np.concatenate([pos, neg])
    ranks = pd.Series(scores).rank().values
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def errors(pr: pd.DataFrame, gt: pd.DataFrame, k: str) -> tuple[np.ndarray, np.ndarray]:
    """Pixel errors and confidences on visible (== 2) cells of keypoint k."""
    sp, sg = pr.columns[0][0], gt.columns[0][0]
    vis = pd.to_numeric(gt[(sg, k, "visible")], errors="coerce") == 2
    f = vis[vis].index.intersection(pr.index)
    e = np.hypot(pr.loc[f, (sp, k, "x")].astype(float).values - gt.loc[f, (sg, k, "x")].astype(float).values,
                 pr.loc[f, (sp, k, "y")].astype(float).values - gt.loc[f, (sg, k, "y")].astype(float).values)
    return e, pr.loc[f, (sp, k, "likelihood")].astype(float).values


def ibl_ears(pr: pd.DataFrame, gt: pd.DataFrame) -> dict:
    sp, sg = pr.columns[0][0], gt.columns[0][0]
    pr = pr.loc[gt.index]
    vis = lambda k: (pd.to_numeric(gt[(sg, k, "visible")], errors="coerce") == 2).values
    ok = vis("pupil_center_left") & vis("nose_tip")
    P = gt.loc[:, [(sg, "pupil_center_left", "x"), (sg, "pupil_center_left", "y")]].astype(float).values
    N = gt.loc[:, [(sg, "nose_tip", "x"), (sg, "nose_tip", "y")]].astype(float).values
    axis = P - N; L = np.linalg.norm(axis, axis=1); u = axis / L[:, None]
    row, any_c, any_p = {}, np.zeros(len(gt), bool), np.zeros(len(gt), bool)
    for k in [k for k in keypoints(pr, "likelihood") if k.startswith("ear_")]:
        c = pr[(sp, k, "likelihood")].values >= CONF
        E = pr.loc[:, [(sp, k, "x"), (sp, k, "y")]].astype(float).values
        rel = E - P; along = (rel * u).sum(1); dist = np.linalg.norm(rel, axis=1) / L
        p = c & ok & (along > 0) & (dist > 0.3) & (dist < 2.5)
        if c.any():
            row[f"{k} conf%"] = 100 * c.mean(); row[f"{k} plausible%"] = 100 * p.mean()
        any_c |= c; any_p |= p
    return {"any ear conf%": 100 * any_c.mean(), "any ear plausible%": 100 * any_p.mean(), **row}


def kaufman_ears(pr: pd.DataFrame, gt: pd.DataFrame) -> dict:
    sg = gt.columns[0][0]
    tau = WRONG_FRAC * FRAME_W["kaufman"]
    row, allE, allC = {}, [], []
    ears = [k for k in keypoints(gt, "visible") if k.startswith("ear_") and (pd.to_numeric(gt[(sg, k, "visible")], errors="coerce") == 2).any()]
    for k in ears:
        e, c = errors(pr, gt, k)
        row[f"{k} mean/median px"] = f"{e.mean():.1f} / {np.median(e):.1f}"
        row[f"{k} det%"] = 100 * (c >= CONF).mean()
        allE.append(e); allC.append(c)
    e, c = np.concatenate(allE), np.concatenate(allC)
    other = [k for k in keypoints(gt, "visible") if not k.startswith("ear_") and (pd.to_numeric(gt[(sg, k, "visible")], errors="coerce") == 2).any()]
    oe = np.concatenate([errors(pr, gt, k)[0] for k in other])
    return {"ears mean px": e.mean(), "ears median px": np.median(e), "ears det%": 100 * (c >= CONF).mean(),
            f"ears conf&wrong(>{tau:.0f}px)%": 100 * ((c >= CONF) & (e > tau)).mean(),
            "other kaufman kps mean px": oe.mean(), "other kaufman kps median px": np.median(oe), **row}


def absent_cells(pr: pd.DataFrame, gt: pd.DataFrame) -> dict | None:
    sp, sg = pr.columns[0][0], gt.columns[0][0]
    pr = pr.loc[gt.index]
    pos, neg = [], []
    for k in keypoints(gt, "visible"):
        if (sp, k, "likelihood") not in pr.columns:
            continue
        v = pd.to_numeric(gt[(sg, k, "visible")], errors="coerce").values
        c = pr[(sp, k, "likelihood")].values
        pos.append(c[v == 2]); neg.append(c[v == 1])
    pos, neg = np.concatenate(pos), np.concatenate(neg)
    if len(neg) == 0:
        return None
    return {"absent cells": len(neg), "absent conf>=0.7 %": 100 * (neg >= CONF).mean(),
            "visible conf>=0.7 %": 100 * (pos >= CONF).mean(), "AUROC visible vs absent": auroc(pos, neg)}


def indomain(pr: pd.DataFrame, gt: pd.DataFrame) -> dict:
    e = np.concatenate([errors(pr, gt, k)[0] for k in keypoints(gt, "visible") if (pr.columns[0][0], k, "x") in pr.columns])
    return {"mean px": e.mean(), "median px": np.median(e)}


def main() -> None:
    ap = argparse.ArgumentParser(epilog=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs",     required=True, nargs="+", help="name=run_dir (run_dir has eval/<dataset>/predictions.csv)")
    ap.add_argument("--data_dir", default="/teamspace/studios/this_studio/poseinterface/data/head-fixed-v9", type=Path)
    ap.add_argument("--out",      required=True, type=Path)
    ap.add_argument("--measures", default="1,2,3,4")
    a = ap.parse_args()
    runs = [(r.split("=", 1)[0], Path(r.split("=", 1)[1])) for r in a.runs]
    gts = {d: read(a.data_dir / f"CollectedData_{d}_test.csv") for d in DATASETS}
    pred = lambda run, d: read(run / "eval" / d / "predictions.csv")
    (a.out / "csv").mkdir(parents=True, exist_ok=True)
    meas = a.measures.split(",")
    tables = {}
    if "1" in meas:
        tables["1_ibl_ear_transfer"] = pd.DataFrame({n: ibl_ears(pred(r, "ibl"), gts["ibl"]) for n, r in runs}).T
    if "2" in meas:
        tables["2_kaufman_ear_zeroshot"] = pd.DataFrame({n: kaufman_ears(pred(r, "kaufman"), gts["kaufman"]) for n, r in runs}).T
    if "3" in meas:
        rows = []
        for n, r in runs:
            for d in DATASETS:
                res = absent_cells(pred(r, d), gts[d])
                if res:
                    rows.append({"run": n, "dataset": d, **res})
        tables["3_absent_cells"] = pd.DataFrame(rows).set_index(["dataset", "run"]).sort_index(level=0, sort_remaining=False)
    if "4" in meas:
        tables["4_indomain_px"] = pd.DataFrame({n: {d: "{:.2f} / {:.2f}".format(*indomain(pred(r, d), gts[d]).values()) for d in DATASETS}
                                                 for n, r in runs}).T
    for name, t in tables.items():
        t.to_csv(a.out / "csv" / f"eval_{name}.csv")
        print(f"\n## {name}\n")
        print(t.round(2).to_string())


if __name__ == "__main__":
    main()
