"""Item 4, step 4.0: alternative confidence readouts from a trained model's own heatmaps (no training).

Predicts a dataset's TEST frames with the run's best checkpoint and, from the same heatmaps that produce the
predictions, computes per (frame, keypoint):
  conf_lp      the confidence the model outputs (the learned branch's sigmoid for model.conf_branch models)
  conf_heatmap Lightning Pose's heatmap confidence (5x5 mass after a temperature-1000 re-softmax), always
  peak         maximum of the model's own (temperature-1) softmax heatmap
  mass_r1/2/4  probability mass of that heatmap within radius 1 / 2 / 4 heatmap pixels of its argmax
               (1 heatmap pixel = 4 image pixels at the 256x256 input)
  neg_entropy  1 - normalised entropy of the heatmap (1 = all mass on one pixel, 0 = flat)
  peak_ratio   1 - (second peak outside radius 4 / main peak) (low = two competing peaks)
Then scores each readout as an uncertainty signal on labelled cells:
  AUROC for "error > 5 % of frame width" (higher = flags errors better), Spearman(readout, -error),
  and the AUROC separating visible (2) from absent (1) test cells.
Writes <out>/csv/readouts__<name>__<dataset>.csv and <out>/csv/readout_scores.csv (appended).

    python scripts/confidence_readouts.py --run <run_dir> --name <id> --dataset ibl --out <dir> [--keypoints a,b]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

FRAME_W = {"kaufman": 800, "ibl": 320, "facemap": 400, "cazettes-side": 750, "cheese-3d": 640, "cheese-2d": 640,
           "kondo": 600, "hantman-mv": 640}
READOUTS = ["conf_lp", "conf_heatmap", "peak", "mass_r1", "mass_r2", "mass_r4", "neg_entropy", "peak_ratio"]


def heatmap_readouts(hm):
    """hm: (B, K, H, W) softmax heatmaps (each map sums to 1) -> dict of (B, K) tensors."""
    import torch

    B, K, H, W = hm.shape
    flat = hm.reshape(B, K, -1)
    peak, idx = flat.max(-1)
    yy, xx = torch.meshgrid(torch.arange(H, device=hm.device), torch.arange(W, device=hm.device), indexing="ij")
    py, px = (idx // W).float(), (idx % W).float()
    d = torch.sqrt((yy[None, None] - py[..., None, None]) ** 2 + (xx[None, None] - px[..., None, None]) ** 2)
    out = {"peak": peak}
    for r in (1, 2, 4):
        out[f"mass_r{r}"] = (hm * (d <= r)).reshape(B, K, -1).sum(-1)
    p = flat.clamp_min(1e-12)
    out["neg_entropy"] = 1 - (-(p * p.log()).sum(-1) / np.log(H * W))
    second = (hm * (d > 4)).reshape(B, K, -1).max(-1).values
    out["peak_ratio"] = 1 - second / peak.clamp_min(1e-12)
    return out


def auroc(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = pd.Series(np.concatenate([pos, neg])).rank().values
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main() -> None:
    ap = argparse.ArgumentParser(epilog=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run",       required=True, type=Path)
    ap.add_argument("--name",      required=True)
    ap.add_argument("--dataset",   required=True)
    ap.add_argument("--data_dir",  default="/teamspace/studios/this_studio/poseinterface/data/head-fixed-v9", type=Path)
    ap.add_argument("--out",       required=True, type=Path)
    ap.add_argument("--keypoints", default=None, help="comma list to score (default: every labelled keypoint)")
    a = ap.parse_args()

    import torch
    from lightning_pose.api import Model
    from omegaconf import open_dict

    model = Model.from_dir(a.run)
    with open_dict(model.config.cfg):
        model.config.cfg.training.sampling_temperature = None
    model._load()
    head = model.model.head
    captured = []
    orig = head.run_subpixelmaxima

    def wrapped(heatmaps):
        out = orig(heatmaps)
        with torch.no_grad():
            r = {k: v.detach().cpu() for k, v in heatmap_readouts(heatmaps.float()).items()}
            r["conf_heatmap"] = out[1].detach().cpu()   # heatmap confidence, even when a learned branch replaces it
            captured.append(r)
        return out

    head.run_subpixelmaxima = wrapped
    csv = a.data_dir / f"CollectedData_{a.dataset}_test.csv"
    res = model.predict_on_label_csv(csv_file=str(csv), data_dir=str(a.data_dir), compute_metrics=False)
    head.run_subpixelmaxima = orig

    pr = res.predictions
    sp = pr.columns[0][0]
    kps = [k for k in dict.fromkeys(pr.columns.get_level_values(1)) if (sp, k, "likelihood") in pr.columns]
    ro = {k: torch.cat([c[k] for c in captured]).numpy() for k in captured[0]}
    assert ro["peak"].shape == (len(pr), len(kps)), (ro["peak"].shape, len(pr), len(kps))
    gt = pd.read_csv(csv, header=[0, 1, 2], index_col=0).loc[pr.index]
    sg = gt.columns[0][0]
    rows = []
    for j, k in enumerate(kps):
        if (sg, k, "visible") not in gt.columns:
            continue
        vis = pd.to_numeric(gt[(sg, k, "visible")], errors="coerce").values
        err = np.hypot(pr[(sp, k, "x")].values - gt[(sg, k, "x")].astype(float).values,
                       pr[(sp, k, "y")].values - gt[(sg, k, "y")].astype(float).values)
        for i in range(len(pr)):
            r = {"frame": pr.index[i], "keypoint": k, "visible": vis[i], "err": err[i] if vis[i] == 2 else np.nan,
                 "conf_lp": pr[(sp, k, "likelihood")].values[i]}
            for m in ro:
                r[m] = float(ro[m][i, j])
            rows.append(r)
    per = pd.DataFrame(rows)
    (a.out / "csv").mkdir(parents=True, exist_ok=True)
    per.to_csv(a.out / "csv" / f"readouts__{a.name}__{a.dataset}.csv", index=False)

    tau = 0.05 * FRAME_W[a.dataset]
    sel = [k for k in (a.keypoints.split(",") if a.keypoints else per.keypoint.unique())]
    vis2 = per[(per.visible == 2) & per.keypoint.isin(sel)]
    absent = per[(per.visible == 1) & per.keypoint.isin(sel)]
    scores = []
    for m in READOUTS:
        wrong, right = vis2[vis2.err > tau][m].values, vis2[vis2.err <= tau][m].values
        scores.append({"run": a.name, "dataset": a.dataset, "keypoints": "+".join(sel) if a.keypoints else "all labelled",
                       "readout": m, "n_cells": len(vis2), "pct_wrong": 100 * (vis2.err > tau).mean(),
                       "AUROC_flags_errors": auroc(-wrong, -right),          # low readout should mean wrong
                       "spearman_with_-error": vis2[m].corr(-vis2.err, method="spearman"),
                       "AUROC_visible_vs_absent": auroc(vis2[m].values, absent[m].values) if len(absent) else np.nan})
    sc = pd.DataFrame(scores)
    f = a.out / "csv" / "readout_scores.csv"
    sc.to_csv(f, mode="a", header=not f.exists(), index=False)
    print(f"{a.name} on {a.dataset} test (τ = {tau:.0f} px, {sc.pct_wrong.iloc[0]:.1f} % of labelled cells wrong):")
    print(sc[["readout", "AUROC_flags_errors", "spearman_with_-error", "AUROC_visible_vs_absent"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
