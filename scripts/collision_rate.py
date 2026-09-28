#!/usr/bin/env python3
"""
Collision rate: how often a model puts a transfer keypoint on top of another body part.

A transfer keypoint is one the dataset never labels (visible == 0 on every test frame), so it has
no pixel error. What can be checked is whether its confident prediction (likelihood >= --conf)
lands within --radius (fraction of the longest image side) of a keypoint the frame DOES label
(visible == 2) from a different body-part group: a paw on the ear, a nose on the digits. Groups
come from model.exclusion.groups in the transfer config (keypoints not listed are singletons).

Reads <run_dir>/eval/<dataset>/predictions.csv against the current data_dir test CSVs; writes
<run_dir>/eval/collision_rate.csv (one row per dataset x transfer keypoint) and prints a
per-dataset summary. Lower is better; it measures confident mistakes, not accuracy.

    python scripts/collision_rate.py --run_dir <results>/.../vits_dinov3/seed0
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

from mighty_mouse.paths import load_paths
from mighty_mouse.registry import load_registry

_paths        = load_paths()
DATA_DIR      = Path(_paths["data_dir"])
GROUPS_CONFIG = Path(__file__).resolve().parent.parent / "configs/ablations/model_zoominout_xfer.yaml"


def column(df: pd.DataFrame, kp: str, coord: str) -> np.ndarray:
    return df.xs((kp, coord), axis=1, level=[1, 2]).iloc[:, 0].to_numpy(dtype=float)


def main():
    parser = argparse.ArgumentParser(
        description="Confident transfer predictions landing on another labeled body part.",
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--run_dir", required=True, type=Path)
    parser.add_argument("--conf",    type=float, default=0.7,  help="likelihood floor (qualitative-video floor)")
    parser.add_argument("--radius",  type=float, default=0.03, help="collision radius, fraction of the longest image side")
    parser.add_argument("--groups_config", type=Path, default=GROUPS_CONFIG)
    args = parser.parse_args()

    groups_cfg = yaml.safe_load(args.groups_config.read_text())["model"]["exclusion"]["groups"]
    group_of   = {kp: gi for gi, members in enumerate(groups_cfg) for kp in members}

    rows = []
    for ds in load_registry():
        pred_csv = args.run_dir / "eval" / ds / "predictions.csv"
        lab_csv  = DATA_DIR / f"CollectedData_{ds}_test.csv"
        if not pred_csv.is_file() or not lab_csv.is_file():
            continue
        lab  = pd.read_csv(lab_csv, header=[0, 1, 2], index_col=0)
        pred = pd.read_csv(pred_csv, header=[0, 1, 2], index_col=0).loc[lab.index]
        kps  = list(dict.fromkeys(c[1] for c in lab.columns))
        vis  = {kp: column(lab, kp, "visible") for kp in kps}
        img  = cv2.imread(str(DATA_DIR / lab.index[0]))
        radius = args.radius * max(img.shape[:2])

        for kp in (k for k in kps if (vis[k] == 0).all()):
            others = [j for j in kps if group_of.get(j, j) != group_of.get(kp, kp) and (vis[j] == 2).any()]
            px, py, pc = (column(pred, kp, c) for c in ("x", "y", "likelihood"))
            confident  = pc >= args.conf
            if others:
                ox   = np.stack([np.where(vis[j] == 2, column(lab, j, "x"), np.nan) for j in others], 1)
                oy   = np.stack([np.where(vis[j] == 2, column(lab, j, "y"), np.nan) for j in others], 1)
                dist = np.hypot(ox - px[:, None], oy - py[:, None])
                dist = np.where(np.isnan(dist), np.inf, dist)
                nearest    = dist.min(axis=1)
                on_part    = np.array(others)[dist.argmin(axis=1)]
                collided   = confident & (nearest < radius)
            else:
                collided   = np.zeros(len(lab), dtype=bool)
                on_part    = np.array([""] * len(lab))
            top = pd.Series(on_part[collided]).value_counts()
            rows.append({
                "dataset":   ds,
                "keypoint":  kp,
                "frames":    len(lab),
                "confident": int(confident.sum()),
                "collided":  int(collided.sum()),
                "top_part":  top.index[0] if len(top) else "",
            })

    out = pd.DataFrame(rows)
    out.to_csv(args.run_dir / "eval" / "collision_rate.csv", index=False)
    summary = out.groupby("dataset", sort=False)[["confident", "collided"]].sum()
    summary["rate_%"] = (100 * summary.collided / summary.confident.clip(lower=1)).round(1)
    print(summary.to_string())
    total = summary.sum()
    print(f"\nall datasets: {int(total.collided)}/{int(total.confident)} confident transfer "
          f"predictions on another body part ({100 * total.collided / max(total.confident, 1):.1f}%)")


if __name__ == "__main__":
    main()
