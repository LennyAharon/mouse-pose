#!/usr/bin/env python
"""
Standalone replacement for Figure 1c: corpus size measured three ways.

Annotated frames alone misstate what each dataset contributes, because the datasets differ
more in keypoints-per-frame than in frame count: cheese-2d annotates 27 keypoints per frame
against ibl's 6, so one cheese-2d frame carries several ibl frames' worth of supervision.
The right panel therefore counts labelled keypoint observations, the quantity the temperature
sampler actually balances (m_d = n_d * kbar_d in Appendix C). The third counts how much of
the shared 36-keypoint vocabulary each dataset annotates at all, which is what decides how much
of the model any one dataset can supervise: no dataset covers more than 27 of 36, and their
union is 35, the missing channel being pupil_center_right, which no dataset annotates and which
horizontal-flip augmentation alone supervises.

The first two panels are stacked by camera view; the third is not, since views of one dataset
share a keypoint set rather than adding to it. All three share the dataset order and colours of
the rest of Figure 1.

Emitted on its own so it can be dropped into the assembled figure by hand.

    python paper_figures/make_fig_corpus_bars.py --out ../paper/figures/fig_corpus_bars.pdf
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

from mouse_pose.paths import load_paths

# Same twelve (dataset, view) panels, order and colours as make_fig_datasets.py.
PANELS = [
    ("facemap", "cam0"), ("facemap", "cam1"), ("ibl", "left"), ("ibl", "right_flipped"),
    ("cheese-2d", "BC"), ("cheese-2d", "TC"), ("cheese-2d", "L"), ("cheese-2d", "R"),
    ("cheese-2d", "TL"), ("cheese-2d", "TR"),
    ("cazettes-side", "side"), ("kondo", "right"),
]
ORDER = ["ibl", "facemap", "cazettes-side", "cheese-2d", "kondo"]
BASE  = {"ibl": "#4C78A8", "facemap": "#54A24B", "cheese-2d": "#E45756",
         "cazettes-side": "#B279A2", "kondo": "#EECA3B"}


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


def view_counts(df: pd.DataFrame, dset: str, view: str) -> tuple[int, int]:
    """Frames and labelled keypoint observations (visible == 2) from one camera view."""
    scorer = df.columns.get_level_values(0)[0]
    kps    = list(dict.fromkeys(df.columns.get_level_values(1)))
    keep   = pd.Series(df.index, index=df.index).map(
        lambda p: view_of(dset, Path(p).parts[2]) == view)
    sub    = df[keep.to_numpy()]
    if len(sub) == 0:
        return 0, 0
    vis = np.stack([pd.to_numeric(sub[(scorer, k, "visible")], errors="coerce").to_numpy()
                    for k in kps], axis=1)
    return len(sub), int((vis == 2).sum())


def draw(ax, per_view, idx, xlabel, annot) -> None:
    """One stacked horizontal bar per dataset, segmented by camera view."""
    for i, d in enumerate(ORDER):
        left, tot = 0, 0
        segs = per_view[d]
        for j, (v, counts) in enumerate(segs):
            n = counts[idx]
            shade = 0.55 + 0.45 * (j / max(len(segs) - 1, 1))
            ax.barh(i, n, left=left, height=0.7, color=BASE[d], alpha=shade,
                    edgecolor="white", linewidth=0.8)
            # only label a segment wide enough to hold the text
            if n > 0.055 * annot["xmax"]:
                ax.text(left + n / 2, i, v, ha="center", va="center", fontsize=6.4,
                        color="white", weight="bold")
            left += n; tot += n
        ax.text(left + 0.015 * annot["xmax"], i, annot["fmt"](tot, segs),
                va="center", fontsize=7.2)
    ax.set_yticks(range(len(ORDER))); ax.set_yticklabels(ORDER, fontsize=8)
    ax.set_xlabel(xlabel, fontsize=8.5)
    ax.set_xlim(0, annot["xmax"])
    ax.tick_params(labelsize=7.5)
    ax.invert_yaxis()
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="x", alpha=0.3)


def draw_vocab(ax, direct, vocab) -> None:
    """Plain bar per dataset against a track of the full vocabulary; not stacked by view,
    since every view of a dataset annotates the same keypoint set."""
    for i, d in enumerate(ORDER):
        ax.barh(i, vocab, height=0.7, color="#DDDDDD", edgecolor="none", zorder=1)
        ax.barh(i, direct[d], height=0.7, color=BASE[d], edgecolor="none", zorder=2)
        ax.text(direct[d] + 0.5, i, f"{direct[d]} / {vocab}", va="center", fontsize=7.2,
                zorder=3)
    ax.set_yticks([])
    ax.set_xlabel("keypoints annotated, of the shared vocabulary", fontsize=8.5)
    ax.set_xlim(0, vocab * 1.22)
    # make the denominator a tick, so the axis itself shows where the vocabulary ends
    ax.set_xticks([0, vocab // 3, 2 * vocab // 3, vocab])
    ax.tick_params(labelsize=7.5)
    ax.invert_yaxis()
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="x", alpha=0.3)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1], epilog=__doc__)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    data = Path(load_paths()["data_dir"])
    dfs  = {d: pd.read_csv(data / f"CollectedData_{d}_train.csv", header=[0, 1, 2], index_col=0)
            for d in ORDER}

    views_of = {}
    for d, v in PANELS:
        views_of.setdefault(d, []).append(v)
    per_view = {d: [(v, view_counts(dfs[d], d, v)) for v in views_of[d]] for d in ORDER}

    f_max = max(sum(c[0] for _, c in per_view[d]) for d in ORDER)
    k_max = max(sum(c[1] for _, c in per_view[d]) for d in ORDER)

    inv    = json.loads((data / "dataset_inventory.json").read_text())["datasets"]
    direct = {d: len(inv[d]["direct"]) for d in ORDER}
    vocab  = 36

    fig, (ax_f, ax_k, ax_v) = plt.subplots(
        1, 3, figsize=(12.6, 2.5), gridspec_kw={"width_ratios": [1.16, 1.16, 0.78]})
    draw(ax_f, per_view, 0, "annotated training frames",
         {"xmax": f_max * 1.42,
          "fmt": lambda t, s: f"{t:,} frames · {len(s)} view{'s' if len(s) > 1 else ''}"})
    draw(ax_k, per_view, 1, "labelled keypoint observations",
         {"xmax": k_max * 1.22, "fmt": lambda t, s: f"{t:,}"})
    ax_k.set_yticks([])
    draw_vocab(ax_v, direct, vocab)

    # the reordering between panels is the point of showing both
    for ax in (ax_f, ax_k, ax_v):
        ax.set_axisbelow(True)
    fig.tight_layout(w_pad=1.4)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}\n")

    print(f"{'dataset':16s}{'frames':>9s}{'kp obs':>10s}{'kp/frame':>10s}"
          f"{'% frames':>10s}{'% kp obs':>10s}{'vocab':>8s}")
    tf = sum(sum(c[0] for _, c in per_view[d]) for d in ORDER)
    tk = sum(sum(c[1] for _, c in per_view[d]) for d in ORDER)
    for d in ORDER:
        f = sum(c[0] for _, c in per_view[d]); k = sum(c[1] for _, c in per_view[d])
        print(f"{d:16s}{f:9,}{k:10,}{k / f:10.1f}{100 * f / tf:9.1f}%"
              f"{100 * k / tk:9.1f}%{direct[d]:5d}/36")
    print(f"{'TOTAL':16s}{tf:9,}{tk:10,}{tk / tf:10.1f}")


if __name__ == "__main__":
    main()
