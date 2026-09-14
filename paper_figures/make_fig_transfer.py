#!/usr/bin/env python
"""
Appendix figure: what the all-data model adds to each laboratory.

Same twelve-panel layout as Figure 1a, but instead of a dataset's own ground truth each panel
carries the model's predictions for the keypoints that dataset does *not* annotate. These are
the transfer keypoints: the model was taught them by the other four corpora, and they are what
a laboratory gains by adopting a shared model rather than training its own. Nothing here can
be scored, because scoring would require the annotations whose absence defines the class; the
figure is what that class looks like, and Section 4.2's protocol is how it is measured.

The model is the all-data trunk, so this is the checkpoint a laboratory would actually be
handed, not a leave-one-out model. Frames come from held-out sessions, and within each view the
panel shows the *median* frame ranked by the confidence of its transfer predictions, not the
best, so the figure is not a highlight reel. Marker opacity tracks the model's own confidence.

    python paper_figures/make_fig_transfer.py --out ../paper/figures/fig_transfer.pdf
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42  # TrueType, not Type 3; NeurIPS PDF requirement
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from mouse_pose.paths import load_paths

TRUNK = "zoom-aug-exp/face+ibl+cheese+caz+kondo-T2-zoominout/seed0"
PANELS = [
    ("facemap", "cam0"), ("facemap", "cam1"), ("ibl", "left"), ("ibl", "right_flipped"),
    ("cheese-2d", "BC"), ("cheese-2d", "TC"), ("cheese-2d", "L"), ("cheese-2d", "R"),
    ("cheese-2d", "TL"), ("cheese-2d", "TR"),
    ("cazettes-side", "side"), ("kondo", "right"),
]
GROUPS = {
    "eye":        (["eye_back_left", "eye_back_right", "eye_bottom_left", "eye_bottom_right",
                    "eye_front_left", "eye_front_right", "eye_top_left", "eye_top_right",
                    "pupil_center_left", "pupil_center_right"], "#0072B2"),
    "nose/mouth": (["nose_tip", "nose_top", "nose_bottom", "pad_center", "mouth", "lowerlip",
                    "upperlip_left", "upperlip_right"], "#D55E00"),
    "tongue":     (["tongue_tip", "tongue_center", "tongue_end_left", "tongue_end_right"],
                   "#CC79A7"),
    "whiskers":   (["pad_top_left", "pad_top_right", "pad_side_left", "pad_side_right"],
                   "#009E73"),
    "ears":       (["ear_top_left", "ear_top_right", "ear_tip_left", "ear_tip_right",
                    "ear_bottom_left", "ear_bottom_right", "ear_base_left", "ear_base_right"],
                   "#E69F00"),
    "forelimb":   (["wrist_left", "wrist_right"], "#56B4E9"),
}
KP_COLOR = {kp: c for kps, c in GROUPS.values() for kp in kps}


def view_of(dset: str, session: str) -> str:
    if dset == "cheese-2d":
        m = re.search(r"_(BC|TC|TL|TR|L|R)(?:_|$)", session)
        return m.group(1) if m else "?"
    if dset == "facemap":
        m = re.search(r"cam[_]?(\d)", session)
        return f"cam{m.group(1)}" if m else "?"
    if dset == "ibl":
        return "right_flipped" if ("right_flipped" in session or "rightCamera" in session) else "left"
    if dset == "cazettes-side":
        return "side"
    return "right"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1], epilog=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ms", type=float, default=5.4,
                    help="marker size at full confidence")
    ap.add_argument("--min_conf", type=float, default=0.0,
                    help="drop predictions below this confidence; 0 keeps every channel")
    args = ap.parse_args()

    paths = load_paths()
    data  = Path(paths["data_dir"])
    res   = Path(paths["results_dir"])
    inv   = json.loads((data / "dataset_inventory.json").read_text())["datasets"]

    preds = {}
    for d in {p[0] for p in PANELS}:
        df = pd.read_csv(res / TRUNK / "eval" / d / "predictions.csv", header=[0, 1, 2],
                         index_col=0)
        preds[d] = df

    ncol, nrow = 4, 3
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.6 * ncol, 2.35 * nrow))
    for ax, (dset, view) in zip(axes.ravel(), PANELS):
        df = preds[dset]
        sc = df.columns.get_level_values(0)[0]
        kps = [k for k in dict.fromkeys(df.columns.get_level_values(1)) if k in KP_COLOR]
        # the transfer class: the model emits it, this dataset never annotates it
        transfer = [k for k in kps
                    if k not in set(inv[dset]["direct"]) and k != "pupil_center_right"]

        sessions = pd.Series(df.index, index=df.index).map(lambda p: Path(p).parts[2])
        keep = df[sessions.map(lambda s: view_of(dset, s) == view).to_numpy()]
        if len(keep) == 0 or not transfer:
            ax.axis("off"); continue

        conf = np.stack([pd.to_numeric(keep[(sc, k, "likelihood")], errors="coerce").to_numpy()
                         for k in transfer], axis=1)
        # A frame shows a transfer keypoint only when the behaviour is happening: cheese-2d
        # inherits four tongue channels, and a frame with the tongue in shows none of them.
        # So restrict to the frames carrying the most inherited channels, then take the MEDIAN
        # of that pool rather than its best, which keeps the panel typical of an informative
        # frame instead of being the single most flattering one. One rule for every panel.
        n_conf = (np.nan_to_num(conf) >= max(args.min_conf, 0.5)).sum(axis=1)
        pool   = np.flatnonzero(n_conf >= np.median(n_conf))
        if len(pool) == 0:
            pool = np.arange(len(keep))
        rank = np.nanmedian(conf, axis=1)[pool]
        row  = keep.index[pool[np.argsort(rank)[len(rank) // 2]]]

        img = Image.open(data / row).convert("RGB")
        ax.imshow(np.asarray(img))
        shown = 0
        for k in transfer:
            x = pd.to_numeric(pd.Series([keep.loc[row, (sc, k, "x")]]), errors="coerce").iloc[0]
            y = pd.to_numeric(pd.Series([keep.loc[row, (sc, k, "y")]]), errors="coerce").iloc[0]
            c = pd.to_numeric(pd.Series([keep.loc[row, (sc, k, "likelihood")]]),
                              errors="coerce").iloc[0]
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            if not np.isfinite(c) or c < args.min_conf:
                continue
            # opacity tracks the model's own confidence, so weaker estimates recede rather
            # than being given the same visual weight as confident ones
            cc = float(np.clip(c, 0, 1))
            ax.plot([x], [y], marker="o", ms=args.ms * (0.62 + 0.38 * cc),
                    mfc=KP_COLOR[k], mec="white", mew=0.5, ls="",
                    alpha=0.35 + 0.65 * cc)
            shown += 1
        label = dset if view in ("side", "right") else f"{dset} · {view}"
        cap = f"{shown} of {len(transfer)}" if args.min_conf > 0 else f"+{len(transfer)}"
        ax.set_title(f"{label}   {cap} inherited", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.6)

    handles = [plt.Line2D([], [], marker="o", ls="", ms=5, mfc=c, mec="white", label=n)
               for n, (_, c) in GROUPS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=len(GROUPS), fontsize=8.5,
               frameon=False, bbox_to_anchor=(0.5, 0.005))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.965, bottom=0.055, hspace=0.16,
                        wspace=0.03)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=220)
    print(f"wrote {out}  (min_conf={args.min_conf}, ms={args.ms})\n")
    for d in dict.fromkeys(p[0] for p in PANELS):
        t = [k for k in dict.fromkeys(preds[d].columns.get_level_values(1))
             if k in KP_COLOR and k not in set(inv[d]["direct"]) and k != "pupil_center_right"]
        print(f"{d:15s} annotates {len(inv[d]['direct']):2d}, inherits {len(t):2d}")


if __name__ == "__main__":
    main()
