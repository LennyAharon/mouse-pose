#!/usr/bin/env python
"""
Appendix figure: what the per-dataset augmentation actually produces.

One row per dataset. The leftmost panel is the frame as the loader first sees it, resized to
the training resolution; the rest are independent draws from that dataset's own augmentation
pipeline, built by calling Lightning Pose's ``get_imgaug_transforms_per_dataset`` on the
training config, so these are the pipelines training uses rather than a reimplementation.
Ground-truth keypoints are carried through each transform, which is the point: the geometry
moves and the labels move with it.

Each dataset's zoom range is printed on its row. Ranges are chosen so every source is seen
across the full span of apparent scale in the corpus, which is why facemap, the most
magnified rig, is zoomed out hardest and ibl, the reference scale, is zoomed in hardest.

    python paper_figures/make_fig_aug_examples.py --out ../paper/figures/fig_aug_examples.pdf
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42  # TrueType, not Type 3; NeurIPS PDF requirement
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imgaug.augmentables.kps import Keypoint, KeypointsOnImage
from omegaconf import OmegaConf
from PIL import Image

from lightning_pose.data.factory import get_imgaug_transforms_per_dataset
from mouse_pose.paths import load_paths, repo_root

DSETS = ["ibl", "cazettes-side", "cheese-2d", "kondo", "facemap"]
# Paper display names; keys are the on-disk dataset names, which the paper does not use.
DISPLAY = {"facemap": "Facemap", "ibl": "IBL", "cheese-2d": "Cheese-3D",
           "cazettes-side": "Cazettes", "kondo": "Kondo"}
NAUG = 7
GROUPS = {
    "#0072B2": ["eye_back_left", "eye_back_right", "eye_bottom_left", "eye_bottom_right",
                "eye_front_left", "eye_front_right", "eye_top_left", "eye_top_right",
                "pupil_center_left", "pupil_center_right"],
    "#D55E00": ["nose_tip", "nose_top", "nose_bottom", "pad_center", "mouth", "lowerlip",
                "upperlip_left", "upperlip_right"],
    "#CC79A7": ["tongue_tip", "tongue_center", "tongue_end_left", "tongue_end_right"],
    "#009E73": ["pad_top_left", "pad_top_right", "pad_side_left", "pad_side_right"],
    "#E69F00": ["ear_top_left", "ear_top_right", "ear_tip_left", "ear_tip_right",
                "ear_bottom_left", "ear_bottom_right", "ear_base_left", "ear_base_right"],
    "#56B4E9": ["wrist_left", "wrist_right"],
}
KP_COLOR = {k: c for c, ks in GROUPS.items() for k in ks}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1], epilog=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    np.random.seed(args.seed)

    root = repo_root()
    cfg = OmegaConf.load(root / "configs" / "model_zoominout.yaml")
    cfg.data.dataset_names = DSETS
    pipes = get_imgaug_transforms_per_dataset(cfg)
    zoom = OmegaConf.to_object(cfg.training.imgaug_per_dataset_zoom)
    size = (int(cfg.data.image_resize_dims.width), int(cfg.data.image_resize_dims.height))
    data = Path(load_paths()["data_dir"])

    fig, axes = plt.subplots(len(DSETS), NAUG + 1,
                             figsize=(1.55 * (NAUG + 1), 1.72 * len(DSETS)))
    for r, dset in enumerate(DSETS):
        df = pd.read_csv(data / f"CollectedData_{dset}_train.csv", header=[0, 1, 2], index_col=0)
        s = df.columns.get_level_values(0)[0]
        kps = list(dict.fromkeys(df.columns.get_level_values(1)))
        vis = pd.DataFrame({k: pd.to_numeric(df[(s, k, "visible")], errors="coerce")
                            for k in kps}, index=df.index)
        frame = (vis == 2).sum(axis=1).idxmax()
        img = Image.open(data / frame).convert("L").resize(size, Image.BILINEAR)
        sx = size[0] / Image.open(data / frame).width
        sy = size[1] / Image.open(data / frame).height
        arr = np.asarray(img)[:, :, None].repeat(3, axis=2)
        present = [k for k in kps if vis.loc[frame, k] == 2]
        koi = KeypointsOnImage(
            [Keypoint(x=float(df.loc[frame, (s, k, "x")]) * sx,
                      y=float(df.loc[frame, (s, k, "y")]) * sy) for k in present],
            shape=arr.shape)
        for c in range(NAUG + 1):
            ax = axes[r, c]
            if c == 0:
                im, kk = arr, koi
            else:
                im, kk = pipes[dset](image=arr, keypoints=koi)
            g = im[:, :, 0].astype(float)
            lo, hi = np.percentile(g, [1, 99])
            ax.imshow(np.clip((g - lo) / max(hi - lo, 1e-6), 0, 1), cmap="gray",
                      vmin=0, vmax=1, interpolation="nearest")
            for k, p in zip(present, kk.keypoints):
                if 0 <= p.x < size[0] and 0 <= p.y < size[1]:
                    ax.scatter(p.x, p.y, s=9, c=KP_COLOR.get(k, "#999999"),
                               edgecolors="white", linewidths=0.3, zorder=3)
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(f"{DISPLAY.get(dset, dset)}\n{zoom[dset]}", fontsize=7.2,
                              linespacing=1.4)
                for sp in ax.spines.values():
                    sp.set_color("#D55E00"); sp.set_linewidth(1.4)
            if r == 0:
                ax.set_title("as loaded" if c == 0 else f"draw {c}", fontsize=7.5)
    fig.tight_layout()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=300)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
