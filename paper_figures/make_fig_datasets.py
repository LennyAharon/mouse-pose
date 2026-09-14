#!/usr/bin/env python
"""
Figure: the corpus, one annotated frame per (dataset, camera view).

Twelve views exist across the five datasets: cheese-2d records six cameras (BC, TC, L,
R, TL, TR), facemap two (cam0, cam1), ibl two (left and a flipped right), and
cazettes-side and kondo one each. Each panel shows a real frame with that dataset's own
ground-truth annotations, so the figure carries both the appearance heterogeneity across
rigs and the annotation heterogeneity across laboratories.

Keypoints are coloured by anatomical group rather than individually, so the same colour
means the same body region in every panel and the reader can see at a glance which
regions a given laboratory annotates.

    python paper_figures/make_fig_datasets.py --out ../paper/figures/fig_datasets.pdf
"""

import argparse
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

# ── anatomical grouping and a colourblind-safe palette (Okabe-Ito) ───────────
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

# Panel order: one row per logical grouping, four columns.
PANELS = [
    ("facemap", "cam0"), ("facemap", "cam1"), ("ibl", "left"), ("ibl", "right_flipped"),
    ("cheese-2d", "BC"), ("cheese-2d", "TC"), ("cheese-2d", "L"), ("cheese-2d", "R"),
    ("cheese-2d", "TL"), ("cheese-2d", "TR"),
    ("cazettes-side", "side"), ("kondo", "right"),
]


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


def view_stats(df: pd.DataFrame, dset: str, view: str) -> tuple[int, int]:
    """Training frames and distinct sessions contributed by one camera view."""
    sessions = pd.Series(df.index, index=df.index).map(lambda p: Path(p).parts[2])
    keep = sessions[sessions.map(lambda x: view_of(dset, x) == view)]
    return len(keep), keep.nunique()


def pick_frame(df: pd.DataFrame, dset: str, view: str):
    """Best frame of this view: maximise the number of distinct anatomical groups visible,
    then the raw keypoint count. Ranking by count alone never surfaces the tongue, which is
    only labelled while extended, so a whole colour of the legend would go unused."""
    scorer = df.columns.get_level_values(0)[0]
    kps = list(dict.fromkeys(df.columns.get_level_values(1)))
    vis = pd.DataFrame(
        {k: pd.to_numeric(df[(scorer, k, "visible")], errors="coerce") for k in kps},
        index=df.index)
    sessions = pd.Series(df.index, index=df.index).map(lambda p: Path(p).parts[2])
    keep = sessions.map(lambda s: view_of(dset, s) == view)
    sub = vis[keep.to_numpy()]
    if sub.empty:
        return None, {}
    grp_of = {kp: g for g, (kps, _) in GROUPS.items() for kp in kps}
    n_kp = (sub == 2).sum(axis=1)
    n_grp = (sub == 2).apply(lambda r: len({grp_of[k] for k in sub.columns[r.to_numpy()]}), axis=1)
    best = (n_grp * 1000 + n_kp).idxmax()
    row = df.loc[best]
    pts = {}
    for k in kps:
        if pd.to_numeric(pd.Series([row[(scorer, k, "visible")]]), errors="coerce").iloc[0] == 2:
            x = pd.to_numeric(pd.Series([row[(scorer, k, "x")]]), errors="coerce").iloc[0]
            y = pd.to_numeric(pd.Series([row[(scorer, k, "y")]]), errors="coerce").iloc[0]
            if np.isfinite(x) and np.isfinite(y):
                pts[k] = (x, y)
    return best, pts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1], epilog=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ms", type=float, default=13.0, help="marker size")
    args = ap.parse_args()

    data = Path(load_paths()["data_dir"])
    dfs = {d: pd.read_csv(data / f"CollectedData_{d}_train.csv", header=[0, 1, 2], index_col=0)
           for d in {p[0] for p in PANELS}}

    ncol, nrow = 4, 3
    fig = plt.figure(figsize=(2.6 * ncol, 2.35 * nrow + 2.0))
    gs = fig.add_gridspec(nrow + 1, ncol, height_ratios=[1] * nrow + [0.85], hspace=0.30)
    axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(ncol)] for r in range(nrow)])
    ax_bar = fig.add_subplot(gs[nrow, :])
    for ax, (dset, view) in zip(axes.ravel(), PANELS):
        frame, pts = pick_frame(dfs[dset], dset, view)
        if frame is None:
            ax.axis("off"); continue
        img = np.asarray(Image.open(data / frame).convert("L"), dtype=float)
        lo, hi = np.percentile(img, [1, 99])          # several rigs record very dark frames
        img = np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)
        ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        for k, (x, y) in pts.items():
            ax.scatter(x, y, s=args.ms, c=KP_COLOR.get(k, "#999999"),
                       edgecolors="white", linewidths=0.35, zorder=3)
        label = dset if view in ("side", "right") else f"{dset} · {view}"
        ax.set_title(label, fontsize=8.2)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.6)
    # ── corpus composition: one stacked bar per dataset, segmented by camera view ──
    order = ["ibl", "facemap", "cazettes-side", "cheese-2d", "kondo"]
    views_of = {}
    for d, v in PANELS:
        views_of.setdefault(d, []).append(v)
    base = {"ibl": "#4C78A8", "facemap": "#54A24B", "cheese-2d": "#E45756",
            "cazettes-side": "#B279A2", "kondo": "#EECA3B"}
    for i, d in enumerate(order):
        left, tot_f, tot_s = 0, 0, 0
        segs = [(v,) + view_stats(dfs[d], d, v) for v in views_of[d]]
        for j, (v, nf, ns) in enumerate(segs):
            shade = 0.55 + 0.45 * (j / max(len(segs) - 1, 1))
            ax_bar.barh(i, nf, left=left, height=0.7, color=base[d], alpha=shade,
                        edgecolor="white", linewidth=0.8)
            if nf > 260:
                ax_bar.text(left + nf / 2, i, v, ha="center", va="center", fontsize=6.4,
                            color="white", weight="bold")
            left += nf; tot_f += nf; tot_s += ns
        ax_bar.text(left + 130, i, f"{tot_f:,} frames · {tot_s} sessions · "
                                  f"{len(segs)} view{'s' if len(segs) > 1 else ''}",
                    va="center", fontsize=7.5)
    ax_bar.set_yticks(range(len(order))); ax_bar.set_yticklabels(order, fontsize=8)
    ax_bar.set_xlabel("annotated training frames", fontsize=8)
    ax_bar.set_xlim(0, 9600)
    ax_bar.tick_params(labelsize=7.5)
    ax_bar.invert_yaxis()
    for sp in ("top", "right", "left"):
        ax_bar.spines[sp].set_visible(False)
    ax_bar.grid(axis="x", alpha=0.3)

    handles = [plt.Line2D([], [], marker="o", ls="", ms=5, mfc=c, mec="white",
                          label=name) for name, (_, c) in GROUPS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=len(GROUPS), fontsize=8.5,
               frameon=False, bbox_to_anchor=(0.5, 0.004),
               title="keypoint group (markers above)", title_fontsize=8.5)
    fig.subplots_adjust(left=0.055, right=0.985, top=0.965, bottom=0.075)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=300)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
