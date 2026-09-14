#!/usr/bin/env python
"""
A more detailed version of Figure 2b: what anchored LoRA does to each output channel.

Three blocks, left to right.

  (i)  Data flow. One frame from the target laboratory goes to both the frozen base model and
       the LoRA student. Both emit all 36 channels; only the student receives gradient.
  (ii) The loss assignment, per (frame, channel), drawn from real annotation data rather
       than sketched. Every cell is one channel of one frame and takes exactly one of three
       states: supervised by ground truth, held by the frozen teacher, or unconstrained.
       The point of the panel is that the split is per cell, not per channel: a keypoint the
       laboratory annotates in one frame and not the next changes state between columns.
  (iii) What the anchor term does. It supplies no target of its own; it holds the channel
       where the frozen model already puts it. That is the sense in which the method does
       not push the unannotated channels anywhere.

    python paper_figures/make_fig_anchor.py --out ../paper/figures/fig_anchor.pdf
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42  # TrueType, not Type 3; NeurIPS PDF requirement
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from mouse_pose.paths import load_paths

TARGET = "facemap"
OTHERS = ["ibl", "cheese-2d", "cazettes-side", "kondo"]
N_FRAMES = 12

C_GT     = "#D55E00"   # ground-truth supervision
C_ANCHOR = "#7B3FB8"   # frozen-teacher anchor (the paper's colour for anchoring)
C_FREE   = "#D9D9D9"   # neither: no term reaches this channel
C_FROZEN = "#4C78A8"


def box(ax, x, y, w, h, label, fc, ec, fs=7.2, tc="black", lw=1.0, style="round,pad=0.012"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style, fc=fc, ec=ec, lw=lw,
                                mutation_aspect=0.6, zorder=2))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fs, color=tc,
            zorder=3, linespacing=1.35)


def arrow(ax, xy1, xy2, ls="-", lw=1.1, color="#444444"):
    ax.add_patch(FancyArrowPatch(xy1, xy2, arrowstyle="-|>", mutation_scale=9, lw=lw,
                                 color=color, ls=ls, shrinkA=1, shrinkB=1, zorder=4))


def panel_flow(ax) -> None:
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_title("(i)  one frame, two models", fontsize=8.5, loc="left", pad=6)

    box(ax, 0.02, 0.42, 0.20, 0.16, "frame from\nthe target lab", "#F2F2F2", "#888888")
    box(ax, 0.36, 0.66, 0.40, 0.17,
        "frozen base model  $\\theta_0$\n(teacher, no gradient)", "#EAF0F7", C_FROZEN)
    box(ax, 0.36, 0.17, 0.40, 0.17,
        "student  $\\theta_0 + \\Delta W_{\\mathrm{LoRA}}$\n1.36M of 21.6M trainable",
        "#F3EDFA", C_ANCHOR)
    arrow(ax, (0.22, 0.53), (0.36, 0.74))
    arrow(ax, (0.22, 0.47), (0.36, 0.26))
    for y, c in ((0.745, C_FROZEN), (0.255, C_ANCHOR)):
        arrow(ax, (0.76, y), (0.90, y), color=c)
        ax.text(0.955, y, "36\nchannels", ha="center", va="center", fontsize=6.8, color=c)
    ax.text(0.50, 0.50, "same frame, same 36 outputs;\nonly the student moves",
            ha="center", va="center", fontsize=6.9, style="italic", color="#555555")


def panel_matrix(ax, states, groups, counts) -> None:
    """states: (n_channels, n_frames) in {0: GT, 1: anchor, 2: free}."""
    cmap = matplotlib.colors.ListedColormap([C_GT, C_ANCHOR, C_FREE])
    ax.imshow(states, cmap=cmap, vmin=-0.5, vmax=2.5, aspect="auto", interpolation="nearest")
    ax.set_xticks(range(states.shape[1]))
    ax.set_xticklabels([f"{i + 1}" for i in range(states.shape[1])], fontsize=6.2)
    ax.set_xlabel("frames of the target laboratory's annotated set", fontsize=7.6)
    # label channels by anatomical group rather than 36 individual ticks
    ticks, labels, edges = [], [], []
    start = 0
    for name, n in groups:
        ticks.append(start + n / 2 - 0.5); labels.append(f"{name} ({n})")
        start += n; edges.append(start - 0.5)
    ax.set_yticks(ticks); ax.set_yticklabels(labels, fontsize=6.6)
    for e in edges[:-1]:
        ax.axhline(e, color="white", lw=1.4)
    for x in np.arange(0.5, states.shape[1] - 0.5):
        ax.axvline(x, color="white", lw=0.7)
    ax.set_ylabel("the 36 shared output channels", fontsize=7.6)
    ax.set_title("(ii)  the loss is assigned per (frame, channel)", fontsize=8.5, loc="left",
                 pad=14)
    ax.text(0, 1.012, counts["subtitle"], transform=ax.transAxes, fontsize=6.5,
            style="italic", color="#777777", va="bottom")
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)

    lab = [(C_GT, "ground truth: annotated in this frame"),
           (C_ANCHOR, "frozen-teacher anchor: not annotated, teacher knows the channel"),
           (C_FREE, "no term reaches it: neither annotated nor known to the teacher")]
    handles = [plt.Rectangle((0, 0), 1, 1, fc=c, ec="none") for c, _ in lab]
    ax.legend(handles, [t for _, t in lab], loc="upper center", bbox_to_anchor=(0.5, -0.20),
              ncol=1, fontsize=6.9, frameon=False, handlelength=1.1, handleheight=1.0)


def panel_hold(ax) -> None:
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_title("(iii)  the anchor holds, it does not push", fontsize=8.5, loc="left", pad=6)

    ax.add_patch(plt.Rectangle((0.06, 0.34), 0.88, 0.48, fc="#FAFAFA", ec="#CCCCCC", lw=0.9))
    ax.text(0.075, 0.855, "one withheld channel; example: ibl with the pupil withheld",
            ha="left", va="center", fontsize=6.5, style="italic", color="#777777")

    tx, ty = 0.24, 0.52
    ax.plot([tx], [ty], marker="o", ms=14, mfc="none", mec=C_FROZEN, mew=1.6, ls="", zorder=3)
    ax.plot([tx], [ty], marker="x", ms=7, color=C_FROZEN, mew=1.8, ls="", zorder=4)
    ax.text(tx, ty + 0.16, "where the frozen\nteacher puts it (2.0 px)", ha="center",
            va="center", fontsize=6.6, color=C_FROZEN, linespacing=1.3)
    ax.plot([tx], [ty], marker="o", ms=5.5, color=C_ANCHOR, ls="", zorder=5)
    ax.text(tx, ty - 0.135, "anchored: stays\n2.1 px", ha="center", va="center",
            fontsize=6.8, color=C_ANCHOR, linespacing=1.3)

    dx, dy = 0.79, 0.62
    arrow(ax, (tx + 0.055, ty + 0.02), (dx - 0.04, dy - 0.012), ls="--", color="#B0B0B0")
    ax.plot([dx], [dy], marker="o", ms=5.5, color="#999999", ls="", zorder=5)
    ax.text(dx, dy - 0.135, "no anchor: drifts\n94.2 px", ha="center", va="center",
            fontsize=6.8, color="#777777", linespacing=1.3)

    ax.text(0.5, 0.16,
            "The anchor term carries no target of its own. It reproduces the output the\n"
            "laboratory's own checkpoint already gives, so the channel is held where it\n"
            "already was rather than driven anywhere new. Remove it and these channels\n"
            "receive no gradient at all, and drift.",
            ha="center", va="center", fontsize=6.8, color="#333333", linespacing=1.5)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1], epilog=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target", default=TARGET)
    args = ap.parse_args()

    paths = load_paths()
    data  = Path(paths["data_dir"])
    inv   = json.loads((data / "dataset_inventory.json").read_text())["datasets"]
    vocab = sorted(set().union(*[set(v["trainable"]) for v in inv.values()]) |
                   {"pupil_center_right"})

    others  = [d for d in inv if d != args.target]
    teacher = set().union(*[set(inv[d]["direct"]) for d in others])
    ann     = set(inv[args.target]["direct"])

    df = pd.read_csv(data / f"CollectedData_{args.target}_train.csv", header=[0, 1, 2],
                     index_col=0)
    scorer = df.columns.get_level_values(0)[0]
    kps    = list(dict.fromkeys(df.columns.get_level_values(1)))
    vis    = np.stack([pd.to_numeric(df[(scorer, k, "visible")], errors="coerce").to_numpy()
                       for k in kps], axis=1) == 2
    # frames spanning the range of completeness, so per-frame variation is visible
    order  = np.argsort(vis.sum(axis=1))
    picked = order[np.linspace(0, len(order) - 1, N_FRAMES).astype(int)]

    # group the vocabulary anatomically so 36 rows stay readable
    GROUPS = [
        ("eye", ["eye_back_left", "eye_back_right", "eye_bottom_left", "eye_bottom_right",
                 "eye_front_left", "eye_front_right", "eye_top_left", "eye_top_right",
                 "pupil_center_left", "pupil_center_right"]),
        ("nose/mouth", ["nose_tip", "nose_top", "nose_bottom", "pad_center", "mouth",
                        "lowerlip", "upperlip_left", "upperlip_right"]),
        ("tongue", ["tongue_tip", "tongue_center", "tongue_end_left", "tongue_end_right"]),
        ("whiskers", ["pad_top_left", "pad_top_right", "pad_side_left", "pad_side_right"]),
        ("ears", ["ear_top_left", "ear_top_right", "ear_tip_left", "ear_tip_right",
                  "ear_bottom_left", "ear_bottom_right", "ear_base_left", "ear_base_right"]),
        ("forelimb", ["wrist_left", "wrist_right"]),
    ]
    ordered = [k for _, ks in GROUPS for k in ks if k in vocab]
    missing = [k for k in vocab if k not in ordered]
    assert not missing, f"ungrouped vocabulary keypoints: {missing}"

    states = np.full((len(ordered), len(picked)), 2, dtype=int)
    for r, k in enumerate(ordered):
        col = kps.index(k) if k in kps else None
        for c, f in enumerate(picked):
            if col is not None and k in ann and vis[f, col]:
                states[r, c] = 0
            elif k in teacher:
                states[r, c] = 1
    per_frame = (states == 0).sum(axis=0)
    counts = {"gt": len(ann), "anchor": len(teacher - ann), "free": 36 - len(ann | teacher)}
    counts["subtitle"] = (
        f"{args.target} as target: {counts['gt']}/36 annotated "
        f"({per_frame.min()}-{per_frame.max()} per frame), {counts['anchor']} anchored, "
        f"{counts['free']} free")

    fig = plt.figure(figsize=(12.4, 3.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.15, 0.95], wspace=0.30)
    panel_flow(fig.add_subplot(gs[0, 0]))
    panel_matrix(fig.add_subplot(gs[0, 1]), states,
                 [(n, len([k for k in ks if k in vocab])) for n, ks in GROUPS], counts)
    panel_hold(fig.add_subplot(gs[0, 2]))
    fig.subplots_adjust(left=0.015, right=0.99, top=0.90, bottom=0.20)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}\n")
    print(f"target {args.target}: {counts['gt']} annotated, {counts['anchor']} anchored, "
          f"{counts['free']} unconstrained")
    print(f"per-frame ground-truth cells range {states.min(axis=0).size and (states == 0).sum(axis=0).min()}"
          f"-{(states == 0).sum(axis=0).max()} of {counts['gt']} annotated channels")


if __name__ == "__main__":
    main()
