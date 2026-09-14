#!/usr/bin/env python
"""
Figure: SuperMouse training (panel A) and anchored-LoRA adaptation (panel B), stacked.

Replaces the hand-made side-by-side schematic. Panel A makes the training recipe explicit
(per-dataset zoom augmentation, temperature-scaled sampling, one shared 36-channel head,
per-dataset masked supervision); panel B makes the adaptation objective explicit (frozen
teacher, LoRA adapters, and the per-channel split between ground truth and distillation).

    python paper_figures/make_fig_architecture.py --out ../paper/figures/fig_architecture.pdf
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42  # TrueType, not Type 3; NeurIPS PDF requirement
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from mouse_pose.paths import load_paths

C_DATA, C_AUG, C_NET, C_HEAD = "#DCE7F2", "#DDF0E4", "#EAE2F3", "#FBE6D5"
C_GT, C_TEACH, C_LORA = "#D55E00", "#0072B2", "#7B3FB8"
ZOOM = {"facemap": "[-0.15, 2.0]", "cheese-2d": "[-0.25, 1.0]", "kondo": "[-0.33, 0.5]",
        "cazettes-side": "[-0.375, 0.25]", "ibl": "[-0.4, 0.15]"}


def box(ax, x, y, w, h, label, fc, fs=7.5, weight="normal", ec="#555555"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                fc=fc, ec=ec, lw=0.8, zorder=2))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fs,
            weight=weight, zorder=3, linespacing=1.35)


def arrow(ax, x0, y0, x1, y1, color="#555555", lw=1.0, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=9,
                                 lw=lw, color=color, linestyle=ls, zorder=1,
                                 shrinkA=1, shrinkB=1))


def panel_a(ax, inv):
    ax.text(0.005, 0.955, "A", fontsize=12, weight="bold", va="top")
    ax.text(0.028, 0.955, "SuperMouse: one model over five differently annotated corpora",
            fontsize=9.5, weight="bold", va="top")
    # drawn bottom-up, so reverse to put facemap on top and match the paper order
    order = ["kondo", "cazettes-side", "cheese-2d", "ibl", "facemap"]
    y0, dy, h = 0.10, 0.155, 0.115
    for i, d in enumerate(order):
        y = y0 + i * dy
        n, k = inv[d]["train_frames"], len(inv[d]["direct"])
        box(ax, 0.02, y, 0.145, h, f"{d}\n{n:,} frames · {k} kp", C_DATA, fs=6.8)
        box(ax, 0.185, y, 0.135, h, f"zoom aug\n{ZOOM[d]}", C_AUG, fs=6.5)
        arrow(ax, 0.165, y + h / 2, 0.185, y + h / 2)
        arrow(ax, 0.32, y + h / 2, 0.365, 0.47)
    ax.text(0.2525, y0 + 5 * dy - 0.012, "per-dataset scale range", fontsize=6.6,
            style="italic", ha="center", color="#333333")
    box(ax, 0.365, 0.33, 0.115, 0.28, "sampler\n$T\\!=\\!2$", C_AUG, fs=7.5)
    ax.text(0.4225, 0.30, "rebalances toward\nsmall corpora", fontsize=6.4, ha="center",
            va="top", style="italic", color="#333333")
    arrow(ax, 0.48, 0.47, 0.525, 0.47)
    box(ax, 0.525, 0.33, 0.155, 0.28, "DINOv3 ViT-S\nbackbone", C_NET, fs=7.5)
    arrow(ax, 0.68, 0.47, 0.715, 0.47)
    box(ax, 0.715, 0.33, 0.115, 0.28, "shared\nhead\n36 ch", C_HEAD, fs=7.5, weight="bold")
    arrow(ax, 0.83, 0.47, 0.862, 0.47)
    for i in range(9):
        ax.add_patch(Rectangle((0.862 + 0.004 * i, 0.335 + 0.004 * i), 0.10, 0.20,
                               fc="white", ec="#888888", lw=0.5, zorder=2 + i))
    ax.text(0.93, 0.30, "36 heatmaps", fontsize=7, ha="center", va="top")
    ax.text(0.5, 0.045, "each frame supervises only the channels its own dataset annotates; "
                        "12,000 steps at $256\\times256$",
            fontsize=7, ha="center", style="italic", color="#333333")


def heat_stack(ax, x, y, w, h, n=6, fc="white", ec="#888888"):
    """A small stack of offset rectangles standing for a set of heatmap channels."""
    for i in range(n):
        ax.add_patch(Rectangle((x + 0.004 * i, y + 0.010 * i), w, h, fc=fc, ec=ec,
                               lw=0.55, zorder=2 + i))


def panel_b(ax):
    ax.text(0.005, 0.965, "B", fontsize=12, weight="bold", va="top")
    ax.text(0.028, 0.965,
            "Anchored LoRA: adapting to a laboratory that annotates only part of the vocabulary",
            fontsize=9.5, weight="bold", va="top")

    # ── 1. the target laboratory's frames feed both paths ───────────────────
    box(ax, 0.012, 0.40, 0.132, 0.22, "target lab\n10 to 50 frames\nannotating $k$ of 36",
        C_DATA, fs=7)
    arrow(ax, 0.146, 0.58, 0.198, 0.775, color=C_TEACH)
    arrow(ax, 0.146, 0.44, 0.198, 0.245, color=C_LORA)

    # ── 2. frozen teacher above, LoRA student below ────────────────────────
    box(ax, 0.198, 0.695, 0.185, 0.165, "frozen base model\nfrom panel A", C_NET,
        fs=7.5, ec=C_TEACH)
    ax.text(0.2905, 0.668, "teacher, no gradients", fontsize=6.5, ha="center", color=C_TEACH)
    box(ax, 0.198, 0.165, 0.185, 0.165, "same weights, frozen\n$+\\,BA$ (rank 16) $+$ head",
        C_NET, fs=7.5, ec=C_LORA)
    ax.text(0.2905, 0.125, "student, 1.36M of 21.6M trainable", fontsize=6.5, ha="center",
            color=C_LORA)

    # ── 3. both emit all 36 channels on the same frame ─────────────────────
    arrow(ax, 0.383, 0.7775, 0.428, 0.7775, color=C_TEACH)
    arrow(ax, 0.383, 0.2475, 0.428, 0.2475, color=C_LORA)
    heat_stack(ax, 0.428, 0.725, 0.062, 0.095, ec=C_TEACH)
    heat_stack(ax, 0.428, 0.195, 0.062, 0.095, ec=C_LORA)
    ax.text(0.475, 0.865, "$H^{t}$", fontsize=8.5, ha="center", color=C_TEACH)
    ax.text(0.475, 0.155, "$H^{s}$", fontsize=8.5, ha="center", color=C_LORA)

    # ── 4. per-channel routing, drawn channel by channel ───────────────────
    ax.text(0.652, 0.645, "per (frame, channel) routing", fontsize=7.6, ha="center",
            weight="bold")
    annotated = {0, 3, 4, 9, 12, 17, 20, 25, 28, 33}   # illustrative: this lab's k channels
    x0, y0, cw, ch, gap = 0.552, 0.475, 0.0092, 0.052, 0.0019
    for i in range(18):
        for r in range(2):
            on = (r * 18 + i) in annotated
            ax.add_patch(Rectangle((x0 + i * (cw + gap), y0 + r * (ch + 0.007)), cw, ch,
                                   fc=(C_GT if on else C_TEACH), alpha=0.85,
                                   ec="white", lw=0.5, zorder=3))
    # the rationale, not just the routing: why each branch gets the target it does
    ax.text(0.552, 0.448, r"$k$ annotated $\rightarrow$ ground-truth loss",
            fontsize=7.0, color=C_GT, va="top")
    ax.text(0.552, 0.396, "the lab's own labels supervise these",
            fontsize=6.2, color=C_GT, va="top", style="italic")
    ax.text(0.552, 0.336, r"$36-k$ unannotated $\rightarrow$ distil frozen teacher",
            fontsize=7.0, color=C_TEACH, va="top")
    ax.text(0.552, 0.288, "no labels, so no gradient reaches them:",
            fontsize=6.2, color=C_TEACH, va="top", style="italic")
    ax.text(0.552, 0.246, "left free they drift and the keypoint is lost",
            fontsize=6.2, color=C_TEACH, va="top", style="italic")
    arrow(ax, 0.492, 0.775, 0.546, 0.612, color=C_TEACH, ls=(0, (3, 2)))
    arrow(ax, 0.492, 0.245, 0.546, 0.468, color=C_LORA)

    # ── 5. one objective ───────────────────────────────────────────────────
    x0s = x0 + 18 * (cw + gap)
    arrow(ax, x0s + 0.004, 0.560, 0.795, 0.585)
    ax.text(0.895, 0.425, "$w=1$,  $\\gamma=1$", fontsize=6.4, ha="center",
            color="#444444")
    box(ax, 0.795, 0.455, 0.20, 0.29, "", "#F7F7F7", ec="#777777")
    ax.text(0.895, 0.700, "one objective", fontsize=7.6, ha="center", weight="bold")
    ax.text(0.895, 0.620, r"$\|H^{s}_{ik}-\mathrm{GT}\|^{2}$", fontsize=8.4,
            ha="center", color=C_GT)
    ax.text(0.895, 0.520,
            r"$+\, w\, c_{ik}^{\gamma}\, \|H^{s}_{ik}-\mathrm{sg}[H^{t}_{ik}]\|^{2}$",
            fontsize=7.2, ha="center", color=C_TEACH)

    ax.text(0.5, 0.045,
            "no source data, no replay buffer, no task boundary: both terms are evaluated on "
            "the same frames in one forward pass",
            fontsize=7, ha="center", style="italic", color="#333333")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1], epilog=__doc__)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    inv = json.loads((Path(load_paths()["data_dir"]) / "dataset_inventory.json").read_text())["datasets"]

    fig, axes = plt.subplots(2, 1, figsize=(9.0, 6.4),
                             gridspec_kw={"height_ratios": [1.0, 1.12], "hspace": 0.14})
    for ax in axes:
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    panel_a(axes[0], inv)
    panel_b(axes[1])
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=300)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
