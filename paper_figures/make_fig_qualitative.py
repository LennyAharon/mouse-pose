#!/usr/bin/env python
"""
Qualitative figure for the masked-label protocol: one held-out test frame under four
adaptations in which the pupil's annotations were withheld from the fine-tuning set.

Panels share the frame and the conventions: grey = the keypoints the target dataset still
annotates, red = the withheld ground truth, coloured marker = each model's estimate of it,
red line = the error. The frame is chosen automatically as the one where the two baselines
fail most while anchoring stays on target, so the figure is representative of the failure
rather than hand-picked; --frame overrides.

    python paper_figures/make_fig_qualitative.py --out ../paper/figures/fig_qualitative.pdf
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42  # TrueType, not Type 3; NeurIPS PDF requirement
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from mouse_pose.paths import load_paths

MASKED, DSET, N, DRAW = "pupil_center_left", "ibl", 50, 0
MODELS = [
    ("frozen base model", "zoom-aug-exp/face+cheese+caz+kondo-T2-zoominout/seed0", "#808080"),
    ("full fine-tuning", f"fewshot-exp-lr5-zio-mask-{MASKED}/{DSET}/tf{N}-draw{DRAW}", "#E45756"),
    ("LoRA", f"fewshot-exp-lora-r16-lr5e-5-head5e-4-zio-mask-{MASKED}/{DSET}/tf{N}-draw{DRAW}", "#F58518"),
    ("anchored LoRA", f"fewshot-exp-anchor-lora-conf1-zio-mask-{MASKED}/{DSET}/tf{N}-draw{DRAW}", "#7B3FB8"),
]


def xy(df, frame, kp):
    s = df.columns.get_level_values(0)[0]
    return (float(df.loc[frame, (s, kp, "x")]), float(df.loc[frame, (s, kp, "y")]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1], epilog=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frame", default=None)
    args = ap.parse_args()

    paths = load_paths()
    data, res = Path(paths["data_dir"]), Path(paths["results_dir"])
    gt = pd.read_csv(data / f"CollectedData_{DSET}_test.csv", header=[0, 1, 2], index_col=0)
    preds = {n: pd.read_csv(res / p / "eval" / DSET / "predictions.csv", header=[0, 1, 2],
                            index_col=0) for n, p, _ in MODELS}
    gs = gt.columns.get_level_values(0)[0]

    vis = pd.to_numeric(gt[(gs, MASKED, "visible")], errors="coerce")
    cand = gt.index[vis == 2]
    err = {}
    for f in cand:
        g = np.array(xy(gt, f, MASKED))
        e = {n: float(np.linalg.norm(np.array(xy(preds[n], f, MASKED)) - g)) for n, _, _ in MODELS}
        err[f] = e
    if args.frame:
        frame = args.frame
    else:  # both baselines wrong, anchoring right, and the base model right to begin with
        frame = max(err, key=lambda f: min(err[f]["full fine-tuning"], err[f]["LoRA"])
                    - max(err[f]["anchored LoRA"], err[f]["frozen base model"]))
    print(f"frame: {frame}")
    for n, _, _ in MODELS:
        print(f"   {n:20s} {err[frame][n]:7.1f} px")

    img = np.asarray(Image.open(data / frame).convert("L"), dtype=float)
    lo, hi = np.percentile(img, [1, 99])
    img = np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)
    others = [k for k in dict.fromkeys(gt.columns.get_level_values(1))
              if k != MASKED and pd.to_numeric(
                  pd.Series([gt.loc[frame, (gs, k, "visible")]]), errors="coerce").iloc[0] == 2]
    gx, gy = xy(gt, frame, MASKED)

    fig, axes = plt.subplots(1, 4, figsize=(11.0, 3.4))
    for ax, (name, _, col) in zip(axes, MODELS):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        for k in others:
            x, y = xy(preds[name], frame, k)
            ax.scatter(x, y, s=22, c="#dddddd", edgecolors="#555555", lw=0.4, zorder=2)
        px, py = xy(preds[name], frame, MASKED)
        ax.plot([gx, px], [gy, py], "-", color="#D62728", lw=1.6, zorder=3)
        ax.scatter(gx, gy, s=95, facecolors="none", edgecolors="#D62728", lw=2.1, zorder=4)
        ax.scatter(px, py, s=80, c=col, edgecolors="white", lw=1.1, zorder=5)
        ax.set_title(f"{name}\n{err[frame][name]:.1f}\\,px".replace("\\,", " "),
                     fontsize=9, color=col if name == "anchored LoRA" else "black",
                     weight="bold" if name == "anchored LoRA" else "normal")
        ax.set_xticks([]); ax.set_yticks([])
    handles = [plt.Line2D([], [], marker="o", ls="", mfc="none", mec="#D62728", mew=1.6,
                          ms=7, label="withheld ground truth"),
               plt.Line2D([], [], marker="o", ls="", mfc="#dddddd", mec="#555555", ms=5,
                          label="keypoints still annotated"),
               plt.Line2D([], [], color="#D62728", lw=1.6, label="error")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8.5, frameon=False,
               bbox_to_anchor=(0.5, 0.005))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=300)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
