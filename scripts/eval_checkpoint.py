#!/usr/bin/env python
"""
Evaluate a NAMED checkpoint of a run on every dataset's test set (the sweep's evaluate_model only
knows the canonical *-best.ckpt). Used to compare checkpoints selected by different validation
monitors (training.ckpt_monitors_extra), e.g. *-best.ckpt vs *-best-val_supervised_loss_T.ckpt.

Writes <out>/<dataset>/{predictions,pixel_error}.csv in the same format as the run's eval/ folder,
plus <out>/summary.csv (per dataset: frames, labels, median / mean px over visible == 2 labels,
visible == 2 only) and prints it next to the canonical eval's numbers when present.

    python scripts/eval_checkpoint.py --run <run_dir> --ckpt "*best-val_supervised_loss_T.ckpt"
    python scripts/eval_checkpoint.py --run <run_dir> --ckpt <file> --out <dir>   # default <run_dir>/eval_<ckpt-stem>
"""

import argparse
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from mouse_pose.paths import load_paths
from mouse_pose.registry import load_registry

EXCLUDE: set[str] = set()   # pupil_center_right is scored where labeled (visible == 2) since corpus v5


def shadow_run(run: Path, ckpt: Path) -> Path:
    """A temp model dir whose only checkpoint is `ckpt`, named as the canonical best, so that
    Model.from_dir picks it without touching the real run directory."""
    tmp = Path(tempfile.mkdtemp(prefix="evalckpt_"))
    os.symlink(run / "config.yaml", tmp / "config.yaml")
    ck_dir = tmp / "tb_logs" / "test" / "version_0" / "checkpoints"
    ck_dir.mkdir(parents=True)
    os.symlink(ckpt, ck_dir / "epoch=0-step=0-best.ckpt")
    return tmp


def pixel_error(pred: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    sp, sl = pred.columns.get_level_values(0)[0], labels.columns.get_level_values(0)[0]
    kps = [k for k in dict.fromkeys(labels.columns.get_level_values(1)) if k not in EXCLUDE and not k.startswith("Unnamed")]
    has_vis = "visible" in labels.columns.get_level_values(2)
    out = {}
    for k in kps:
        if (sp, k, "x") not in pred.columns:
            continue
        lx, ly = pd.to_numeric(labels[(sl, k, "x")], errors="coerce"), pd.to_numeric(labels[(sl, k, "y")], errors="coerce")
        px_, py_ = pd.to_numeric(pred[(sp, k, "x")], errors="coerce").reindex(labels.index), pd.to_numeric(pred[(sp, k, "y")], errors="coerce").reindex(labels.index)
        d = np.hypot(px_ - lx, py_ - ly)
        if has_vis:
            d = d.where(pd.to_numeric(labels[(sl, k, "visible")], errors="coerce") == 2)
        out[k] = d
    return pd.DataFrame(out, index=labels.index)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1], epilog=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run",  type=Path, required=True, help="run directory (holds config.yaml, tb_logs/, eval/)")
    ap.add_argument("--ckpt", required=True, help="checkpoint file, or a glob relative to the run directory")
    ap.add_argument("--out",  type=Path, help="output folder (default <run>/eval_<ckpt-stem>)")
    args = ap.parse_args()

    run = args.run.resolve()
    ckpt = Path(args.ckpt)
    if not ckpt.exists():
        matches = sorted(run.rglob(args.ckpt))
        if len(matches) != 1:
            raise SystemExit(f"{len(matches)} checkpoints match {args.ckpt!r} under {run}: {matches}")
        ckpt = matches[0]
    out = args.out or run / f"eval_{ckpt.stem.split('-')[-1] if '-best-' in ckpt.name else ckpt.stem}"
    out.mkdir(parents=True, exist_ok=True)

    from lightning_pose.api.model import Model
    data_dir = Path(load_paths()["data_dir"])
    model = Model.from_dir(shadow_run(run, ckpt))
    rows = []
    for ds in load_registry():
        test_csv = data_dir / f"CollectedData_{ds}_test.csv"
        if not test_csv.exists():
            continue
        res = model.predict_on_label_csv(csv_file=test_csv, data_dir=data_dir, compute_metrics=False)
        labels = pd.read_csv(test_csv, header=[0, 1, 2], index_col=0)
        if labels.index[0] == labels.index.name:
            labels = labels.iloc[1:]
        pe = pixel_error(res.predictions, labels)
        (out / ds).mkdir(exist_ok=True)
        res.predictions.to_csv(out / ds / "predictions.csv"); pe.to_csv(out / ds / "pixel_error.csv")
        vals = pe.to_numpy().ravel(); vals = vals[np.isfinite(vals)]
        row = dict(dataset=ds, frames=len(pe), labels=int(vals.size),
                   median_px=float(np.median(vals)) if vals.size else np.nan, mean_px=float(np.mean(vals)) if vals.size else np.nan)
        canon = run / "eval" / ds / "pixel_error.csv"
        if canon.exists():
            c = pd.read_csv(canon, index_col=0)
            c = c[[k for k in c.columns if k in pe.columns]].apply(pd.to_numeric, errors="coerce").to_numpy().ravel()
            c = c[np.isfinite(c)]
            row["canonical_median_px"] = float(np.median(c)) if c.size else np.nan
            row["canonical_mean_px"] = float(np.mean(c)) if c.size else np.nan
        rows.append(row)
    df = pd.DataFrame(rows).round(3)
    df.to_csv(out / "summary.csv", index=False)
    print(f"checkpoint: {ckpt.name}\n" + df.to_string(index=False) + f"\nwrote {out}")


if __name__ == "__main__":
    main()
