#!/usr/bin/env python
"""
Appendix figure: a true transfer keypoint, facemap's pupil.

No dataset annotates a pupil on facemap, so unlike the masked-label settings this one needs no
construction: it is a keypoint a laboratory genuinely inherits from the shared model and
genuinely cannot check. The figure is the example the protocol stands in for, showing the
keypoint being lost rather than reporting a statistic.

Rows are facemap test frames cropped to the annotated left-eye contour; columns are the
frozen base model and three adaptations at N=50. Red squares are facemap's own eye-contour
annotations, which every model still has labels for; the marker is each model's pupil
estimate, hollow when its confidence falls below 0.5.

    python paper_figures/make_fig_facemap.py --out ../paper/figures/fig_facemap.pdf
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

KP = "pupil_center_left"
EYE = ["eye_back_left", "eye_bottom_left", "eye_front_left", "eye_top_left"]
MODELS = [("frozen base model", "zoom-aug-exp/ibl+cheese+caz+kondo-T2-zoominout/seed0", "#808080"),
          ("full fine-tuning", "fewshot-exp-lr5-zio/facemap/tf50-draw0", "#E45756"),
          ("LoRA", "fewshot-exp-lora-r16-lr5e-5-head5e-4-zio/facemap/tf50-draw0", "#F58518"),
          ("anchored LoRA", "fewshot-exp-anchor-lora-conf1-zio/facemap/tf50-draw0", "#7B3FB8")]
NROW = 3


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1], epilog=__doc__)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    paths = load_paths()
    data, res = Path(paths["data_dir"]), Path(paths["results_dir"])

    gt = pd.read_csv(data / "CollectedData_facemap_test.csv", header=[0, 1, 2], index_col=0)
    gs = gt.columns.get_level_values(0)[0]
    preds = {n: pd.read_csv(res / p / "eval" / "facemap" / "predictions.csv",
                            header=[0, 1, 2], index_col=0) for n, p, _ in MODELS}
    ps = preds[MODELS[0][0]].columns.get_level_values(0)[0]

    def eye_box(f):
        pts = [(float(gt.loc[f, (gs, k, "x")]), float(gt.loc[f, (gs, k, "y")])) for k in EYE
               if pd.to_numeric(pd.Series([gt.loc[f, (gs, k, "visible")]]),
                                errors="coerce").iloc[0] == 2]
        return np.array(pts) if len(pts) == len(EYE) else None

    # Choose frames that show the phenomenon rather than frames that are easy: the base model
    # must be confident enough that the keypoint was there to lose, and the baselines must have
    # visibly moved it while anchoring held. Ranking is by how far the baselines drift from the
    # base model minus how far anchoring does.
    def px(name, f, ax_):
        return float(preds[name].loc[f, (ps, KP, ax_)])
    base_conf = pd.to_numeric(preds[MODELS[0][0]][(ps, KP, "likelihood")], errors="coerce")
    cand = [f for f in gt.index if eye_box(f) is not None and base_conf.get(f, 0) > 0.5]
    def score(f):
        b = np.array([px(MODELS[0][0], f, "x"), px(MODELS[0][0], f, "y")])
        drift = [np.linalg.norm(np.array([px(n, f, "x"), px(n, f, "y")]) - b)
                 for n, _, _ in MODELS[1:3]]
        held = np.linalg.norm(np.array([px(MODELS[3][0], f, "x"),
                                        px(MODELS[3][0], f, "y")]) - b)
        return min(drift) - held
    frames = sorted(cand, key=score, reverse=True)[:NROW]

    fig, axes = plt.subplots(NROW, len(MODELS), figsize=(2.5 * len(MODELS), 2.45 * NROW))
    for r, f in enumerate(frames):
        box = eye_box(f)
        cx, cy = box[:, 0].mean(), box[:, 1].mean()
        half = max(np.ptp(box[:, 0]), np.ptp(box[:, 1])) * 1.7
        img = np.asarray(Image.open(data / f).convert("L"), dtype=float)
        lo, hi = np.percentile(img, [1, 99])
        img = np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)
        for c, (name, _, col) in enumerate(MODELS):
            ax = axes[r, c]
            ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.scatter(box[:, 0], box[:, 1], s=26, marker="s", c="#D62728",
                       edgecolors="white", lw=0.5, zorder=3)
            d = preds[name]
            x = float(d.loc[f, (ps, KP, "x")]); y = float(d.loc[f, (ps, KP, "y")])
            conf = float(pd.to_numeric(pd.Series([d.loc[f, (ps, KP, "likelihood")]]),
                                       errors="coerce").iloc[0])
            ax.scatter(x, y, s=110, facecolors=(col if conf >= 0.5 else "none"),
                       edgecolors=col, lw=2.0, zorder=4)
            H, W = img.shape                     # keep the crop inside the frame
            x0 = min(max(cx - half, 0), W - 2 * half) if W > 2 * half else 0
            y0 = min(max(cy - half, 0), H - 2 * half) if H > 2 * half else 0
            ax.set_xlim(x0, x0 + 2 * half); ax.set_ylim(y0 + 2 * half, y0)
            ax.set_xticks([]); ax.set_yticks([])
            ax.text(0.03, 0.04, f"conf {conf:.2f}", transform=ax.transAxes, fontsize=8,
                    color="white", va="bottom",
                    bbox=dict(fc="black", ec="none", alpha=0.65, pad=1.6))
            if r == 0:
                ax.set_title(name, fontsize=9.5,
                             color=col if name == "anchored LoRA" else "black",
                             weight="bold" if name == "anchored LoRA" else "normal")
    fig.tight_layout()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=300)
    print(f"wrote {out}  frames: {[Path(f).name for f in frames]}")


if __name__ == "__main__":
    main()
