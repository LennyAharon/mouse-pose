#!/usr/bin/env python3
"""
Phase-1 transfer scores for the xfer_head plan (plans/transfer-plan-2026-09-28.md, section 5.2).

Reads <run>/eval/<dataset>/{predictions,pixel_error}.csv of every run given with --run name=path
(a seed directory) and writes one CSV per table plus a printed summary:

  indomain.csv   per run x dataset: mean / median test px over labelled cells
  ears.csv       per run x dataset: fraction of test frames with any ear keypoint >= --conf;
                 on ibl / kaufman also the median distance of confident ears to the reference
                 model's confident prediction of the same keypoint (--ref, default the
                 cheese+kondo+c3d zero-shot trunk) and the fraction within --near px
  kaufman.csv    per run: face-on-paw = confident face predictions within --paw_px of a labelled
                 digit tip / all confident face predictions; face-found = frames where nose_tip
                 and pad_center are both confident and within --near px of the
                 reference; plus per face keypoint firing and found rates

    python scripts/score_xfer.py --run base=<seed dir> --run A=<seed dir> --out <dir>
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from mighty_mouse.paths import load_paths

EARS = [f"ear_{p}_{s}" for p in ("top", "tip", "bottom", "base") for s in ("left", "right")]
FACE = [
    "nose_tip", "nose_top", "nose_bottom", "pad_center", "mouth", "lowerlip", "upperlip_right",
    "upperlip_left", "eye_front_right", "eye_back_right", "eye_top_right", "eye_bottom_right",
]
FOUND = ["nose_tip", "pad_center"]   # the reference finds eye_front_right on only ~30 % of frames
DIGITS = [f"d{i}_tip_right" for i in range(1, 5)]
REF = ("results_dir", "trunks/cheese+kondo+c3d_train/supervised/sampling-T2/tf1/vits_dinov3/seed0")


def read_dlc(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=[0, 1, 2], index_col=0)
    df.columns = df.columns.droplevel(0)
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run",    action="append", required=True, help="name=path to a seed dir with eval/")
    ap.add_argument("--ref",    default=None,  help="reference seed dir (default cheese+kondo+c3d)")
    ap.add_argument("--out",    required=True, type=Path)
    ap.add_argument("--conf",   type=float, default=0.7)
    ap.add_argument("--near",   type=float, default=20.0, help="px: agreement with the reference")
    ap.add_argument("--paw_px", type=float, default=40.0, help="px: face-on-paw radius")
    args = ap.parse_args()

    paths    = load_paths()
    data_dir = Path(paths["data_dir"])
    ref_dir  = Path(args.ref) if args.ref else Path(paths[REF[0]]) / REF[1]
    runs     = dict(r.split("=", 1) for r in args.run)
    args.out.mkdir(parents=True, exist_ok=True)
    datasets = sorted(p.name for p in (Path(next(iter(runs.values()))) / "eval").iterdir() if p.is_dir())

    # ── in-domain ──
    rows = []
    for name, run in runs.items():
        for ds in datasets:
            err = pd.read_csv(Path(run) / "eval" / ds / "pixel_error.csv", index_col=0).to_numpy().ravel()
            err = err[np.isfinite(err)]
            rows.append(dict(run=name, dataset=ds, mean_px=err.mean(), median_px=np.median(err), n=err.size))
    indomain = pd.DataFrame(rows)
    indomain.to_csv(args.out / "indomain.csv", index=False)

    # ── ears ──
    rows = []
    for name, run in runs.items():
        for ds in datasets:
            pred = read_dlc(Path(run) / "eval" / ds / "predictions.csv")
            lik  = pred.xs("likelihood", axis=1, level=1)[EARS]
            row  = dict(run=name, dataset=ds, frames=len(pred), ear_fire=(lik >= args.conf).any(axis=1).mean())
            if ds in ("ibl", "kaufman"):
                ref = read_dlc(ref_dir / "eval" / ds / "predictions.csv").loc[pred.index]
                d_all = []
                for kp in EARS:
                    both = (pred[kp]["likelihood"] >= args.conf) & (ref[kp]["likelihood"] >= args.conf)
                    d = np.hypot(pred[kp]["x"] - ref[kp]["x"], pred[kp]["y"] - ref[kp]["y"])[both]
                    d_all.extend(d.tolist())
                d_all = np.asarray(d_all)
                row.update(ear_vs_ref_median_px=np.median(d_all) if d_all.size else np.nan,
                           ear_vs_ref_near=(d_all <= args.near).mean() if d_all.size else np.nan,
                           ear_vs_ref_n=d_all.size)
            rows.append(row)
    ears = pd.DataFrame(rows)
    ears.to_csv(args.out / "ears.csv", index=False)

    # ── kaufman face ──
    labels = read_dlc(data_dir / "CollectedData_kaufman_test.csv")
    ref    = read_dlc(ref_dir / "eval" / "kaufman" / "predictions.csv")
    rows   = []
    for name, run in runs.items():
        pred = read_dlc(Path(run) / "eval" / "kaufman" / "predictions.csv")
        lab  = labels.loc[pred.index]
        rf   = ref.loc[pred.index]
        dig  = np.stack([lab[d][["x", "y"]].to_numpy(float) for d in DIGITS], axis=1)   # (N, 4, 2)
        on_paw = conf_n = 0
        row = dict(run=name)
        found_all = np.ones(len(pred), bool)
        for kp in FACE:
            xy   = pred[kp][["x", "y"]].to_numpy(float)
            conf = pred[kp]["likelihood"].to_numpy() >= args.conf
            dpaw = np.nanmin(np.linalg.norm(dig - xy[:, None], axis=-1), axis=1)
            paw  = conf & (dpaw <= args.paw_px)
            on_paw += paw.sum(); conf_n += conf.sum()
            near = conf & (np.hypot(*(xy - rf[kp][["x", "y"]].to_numpy(float)).T) <= args.near)
            row[f"{kp}_fire"]  = conf.mean()
            row[f"{kp}_found"] = near.mean()
            if kp in FOUND:
                found_all &= near
        row["face_on_paw"] = on_paw / conf_n if conf_n else np.nan
        row["face_found"]  = found_all.mean()
        rows.append(row)
    kauf = pd.DataFrame(rows)
    kauf.to_csv(args.out / "kaufman.csv", index=False)

    # ── summary ──
    pd.set_option("display.width", 200)
    print("\nin-domain mean px (median)")
    print(indomain.assign(v=indomain.mean_px.round(2).astype(str) + " (" + indomain.median_px.round(2).astype(str) + ")")
          .pivot(index="dataset", columns="run", values="v")[list(runs)])
    print("\near firing (any ear >= conf), fraction of test frames")
    print(ears.pivot(index="dataset", columns="run", values="ear_fire")[list(runs)].round(3))
    cols = ["run", "ear_vs_ref_median_px", "ear_vs_ref_near", "ear_vs_ref_n"]
    print("\nears vs reference on ibl / kaufman")
    print(ears[ears.dataset.isin(["ibl", "kaufman"])][["dataset"] + cols].round(3).to_string(index=False))
    print("\nkaufman face")
    print(kauf[["run", "face_on_paw", "face_found"] + [f"{k}_found" for k in FOUND + ["eye_front_right"]]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
