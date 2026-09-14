#!/usr/bin/env python
"""
Results figure. Panel A: few-shot adaptation per dataset. Panel B: the masked-label protocol.
Panel C: what adaptation costs on channels the target lab does not annotate.

A  x = annotated frames (0 = frozen leave-one-out base model), y = pooled pixel error read
   at the 50th-percentile ensemble-std operating point, the readout the analysis notebook
   marks with its dashed median line. The ensemble std is computed once across every run in
   a panel, so all series sit on identical (frame, keypoint) points. Restricted to
   *supported* keypoints so the panel is a clean statement about the backbone; mixing in
   keypoints the base model never saw lets one hard channel dominate a dataset's curve.

C  Masked-label settings at N=10. Left: error on the keypoints the lab still
   annotates, which is all it can measure. Right: error on the withheld keypoint, which it
   cannot. Shared log axis, because the point is that the left group is flat while the
   right spans two orders of magnitude.

    python paper_figures/make_fig_results.py --out ../paper/figures/fig_results.pdf
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

from mouse_pose.paths import load_paths
from mouse_pose.plots.ensemble import Ensemble, build_ensemble, error_at_percentile

DSETS = ["ibl", "cazettes-side", "cheese-2d", "kondo", "facemap"]
LOO = {"facemap": "ibl+cheese+caz+kondo", "ibl": "face+cheese+caz+kondo",
       "cheese-2d": "face+ibl+caz+kondo", "cazettes-side": "face+ibl+cheese+kondo",
       "kondo": "face+ibl+cheese+caz"}
BUDGETS, DRAWS, SEEDS = [10, 25, 50], [0, 1, 2], [0, 1, 2]

C_GT, C_NET = "#4C78A8", "#EAE2F3"


def box(ax, x, y, w, h, label, fc, fs=8):
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                               fc=fc, ec="#555555", lw=0.8, zorder=2))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fs, zorder=3)


C_ANCHOR, C_DINO, C_SINGLE, C_ZS, C_FT, C_LORA = (
    "#7B3FB8", "#4C78A8", "#54A24B", "#808080", "#E45756", "#F58518")

# Masked settings, all at N=10 so the settings stay comparable. Three datasets were run at
# three annotation subsets and three more at a single subset; whichever exist are averaged,
# and the caption states the split. Anchoring is near-deterministic across subsets (spread
# 0.1 to 0.7 px), the baselines much less so, which is why the single-subset settings are
# reported qualitatively in the text rather than to the decimal.
MASKED = [("ibl", ["pupil_center_left"], "pupil_center_left", "ibl\npupil"),
          ("ibl", ["wrist_left", "wrist_right"], "wrist_left+wrist_right", "ibl\nwrists"),
          ("cazettes-side", ["pupil_center_left"], "pupil_center_left", "cazettes\npupil"),
          ("cazettes-side", ["wrist_left", "wrist_right"], "wrist_left+wrist_right",
           "cazettes\nwrists"),
          ("kondo", ["wrist_left", "wrist_right"], "wrist_left+wrist_right", "kondo\nwrists"),
          ("cheese-2d", ["eye_back_left", "eye_back_right"], "eye_back_left+eye_back_right",
           "cheese-2d\neye"),
          ("facemap", ["nose_tip"], "nose_tip", "facemap\nnose tip"),
          ("kondo", ["lowerlip"], "lowerlip", "kondo\nlowerlip*"),
          ("cazettes-side", ["tongue_tip"], "tongue_tip", "cazettes\ntongue*")]
ARMS = [("full FT", "fewshot-exp-lr5-zio-mask-{s}", C_FT),
        ("LoRA", "fewshot-exp-lora-r16-lr5e-5-head5e-4-zio-mask-{s}", C_LORA),
        # trained from a DINOv3 backbone with no Mighty Mouse initialisation, so the withheld
        # channel is never supervised by anyone: it has nothing to forget and nothing to offer
        ("DINO", "fewshot-exp-dino-mask-{s}", C_DINO),
        ("anchored", "fewshot-exp-anchor-lora-conf1-zio-mask-{s}", C_ANCHOR)]


def eval_files(run: Path, dset: str):
    d = run / "eval" / dset
    p, e = d / "predictions.csv", d / "pixel_error.csv"
    return (str(p), str(e)) if p.is_file() and e.is_file() else None


def supported_of(inv, dset):
    others = set().union(*[set(v["direct"]) for k, v in inv.items() if k != dset])
    return sorted((set(inv[dset]["eval"]) - {"pupil_center_right"}) & others)


def panel_a_values(results, inv, dset):
    runs = {"zero-shot": [results / "zoom-aug-exp" / f"{LOO[dset]}-T2-zoominout" / "seed0"]}
    for n in BUDGETS:
        runs[f"anchored {n}"] = [results / "fewshot-exp-anchor-lora-conf1-zio" / dset /
                                 f"tf{n}-draw{d}" for d in DRAWS]
        runs[f"dino {n}"] = [results / "fewshot-exp-dino" / dset / f"tf{n}-draw{d}"
                             for d in DRAWS]
    runs["dedicated"] = [results / f"{dset}_train" / "supervised" / "tf1" / "vits_dinov3" /
                         f"seed{s}" for s in SEEDS]
    ens, err = Ensemble(), {}
    ens.data_to_plot = {}
    for label, paths in runs.items():
        for i, r in enumerate(paths):
            f = eval_files(r, dset)
            if f is None:
                raise FileNotFoundError(f"{label}.{i}: {r}")
            ens.data_to_plot[f"{label}.{i}"], err[f"{label}.{i}"] = f
    # every keypoint the laboratory annotates and evaluates. Read at the 50th-percentile
    # ensemble-std operating point, the notebook's house style: the ensemble std is computed
    # once across every run in the panel, so all series are scored on identical points. Panel C
    # reports a ratio, which is a different kind of quantity, so the two do not need to share a
    # readout and each uses the one suited to it.
    kps = sorted(set(inv[dset]["eval"]) - {"pupil_center_right"})
    build_ensemble(ens, error_csv_dict=err, keypoints=kps, fast=True)
    return {k: error_at_percentile(ens, k, 50.0) for k in runs}


def pooled(csv: Path, kps):
    if not csv.exists():
        return None
    df = pd.read_csv(csv, index_col=0)
    cols = [k for k in kps if k in df.columns]
    if not cols:
        return None
    a = df[cols].to_numpy(dtype=float).ravel()
    return float(np.nanmean(a)) if np.isfinite(a).any() else None


def panel_b_values(results, inv, n=10):
    """Pooled pixel error, observable vs withheld, from the masked runs only."""
    out = []
    for dset, masked, suffix, label in MASKED:
        annotated = sorted(set(inv[dset]["eval"]) - {"pupil_center_right"} - set(masked))
        base = results / "zoom-aug-exp" / f"{LOO[dset]}-T2-zoominout" / "seed0" / "eval" / dset
        row = {"label": label,
               "base": (pooled(base / "pixel_error.csv", annotated),
                        pooled(base / "pixel_error.csv", masked))}
        for arm, tmpl, _ in ARMS:
            obs, wit = [], []
            for d in DRAWS:   # whichever subsets were run
                csv = results / tmpl.format(s=suffix) / dset / f"tf{n}-draw{d}" / "eval" / dset / "pixel_error.csv"
                o, w = pooled(csv, annotated), pooled(csv, masked)
                if o is not None:
                    obs.append(o); wit.append(w)
            row[arm] = (float(np.mean(obs)), float(np.mean(wit))) if obs else (None, None)
        out.append(row)
    return out



def panel_protocol(ax):
    """Panel B: how a keypoint is withheld during adaptation and recovered for scoring."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    NK, NF = 7, 5                     # illustrative keypoints and frames
    HID = 4                           # the withheld channel
    cw, ch, gap = 0.021, 0.055, 0.004

    def grid(x0, y0, hidden_blank, title, sub, sub_col="#333333"):
        for i in range(NK):
            for j in range(NF):
                x, y = x0 + i * (cw + gap), y0 + j * (ch + gap)
                if i == HID and hidden_blank:
                    ax.add_patch(Rectangle((x, y), cw, ch, fc="white", ec="#BBBBBB",
                                           lw=0.8, ls=(0, (2, 2)), zorder=3))
                else:
                    ax.add_patch(Rectangle((x, y), cw, ch,
                                           fc=(C_GT if i != HID else "#D62728"),
                                           alpha=0.85, ec="white", lw=0.6, zorder=3))
        w = NK * (cw + gap)
        ax.text(x0 + w / 2, y0 + NF * (ch + gap) + 0.05, title, ha="center", fontsize=8.5,
                weight="bold")
        ax.text(x0 + w / 2, y0 - 0.075, sub, ha="center", fontsize=7.2, color=sub_col,
                linespacing=1.35)
        return x0 + w

    ax.text(0.10, 0.93, "annotations, one column per keypoint", fontsize=7.5,
            style="italic", color="#666666")
    r1 = grid(0.10, 0.30, True, "adaptation set, $N$ frames",
              "one keypoint's labels deleted:\nno gradient ever reaches that channel")
    arrow = FancyArrowPatch((r1 + 0.03, 0.47), (r1 + 0.11, 0.47), arrowstyle="-|>",
                            mutation_scale=11, lw=1.2, color="#555555")
    ax.add_patch(arrow)
    ax.text(r1 + 0.07, 0.51, "adapt", ha="center", fontsize=7.5)
    box(ax, r1 + 0.13, 0.36, 0.15, 0.22, "adapted\nmodel", C_NET, fs=8)
    arrow2 = FancyArrowPatch((r1 + 0.29, 0.47), (r1 + 0.37, 0.47), arrowstyle="-|>",
                             mutation_scale=11, lw=1.2, color="#555555")
    ax.add_patch(arrow2)
    ax.text(r1 + 0.33, 0.51, "predict", ha="center", fontsize=7.5)
    grid(r1 + 0.39, 0.30, False, "held-out test frames",
         "labels never modified: the withheld column\nis restored and scored",
         sub_col="#D62728")
    ax.text(0.5, 0.06, "the adapted model has trained under exactly the conditions of a "
                       "transfer keypoint, and is scored with the rigour of a supported one",
            ha="center", fontsize=7.5, style="italic", color="#333333")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1], epilog=__doc__)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    paths = load_paths()
    results = Path(paths["results_dir"])
    inv = json.loads((Path(paths["data_dir"]) / "dataset_inventory.json").read_text())["datasets"]

    fig = plt.figure(figsize=(14.0, 10.2))
    gs = fig.add_gridspec(3, 6, height_ratios=[1.0, 0.62, 0.95], hspace=0.75, wspace=0.32)

    # ── panel A ─────────────────────────────────────────────────────────────
    axes_a = [fig.add_subplot(gs[0, i]) for i in range(5)]
    for ax, dset in zip(axes_a, DSETS):
        v = panel_a_values(results, inv, dset)
        ax.plot([0] + BUDGETS, [v["zero-shot"]] + [v[f"anchored {n}"] for n in BUDGETS],
                "o-", color=C_ANCHOR, lw=2, ms=5, label="Mighty Mouse + anchored LoRA")
        ax.plot(BUDGETS, [v[f"dino {n}"] for n in BUDGETS], "s--", color=C_DINO, lw=2, ms=5,
                label="DINO backbone, from scratch")
        ax.axhline(v["dedicated"], color=C_SINGLE, ls=":", lw=1.8,
                   label="dedicated model (all frames)")
        ax.plot([0], [v["zero-shot"]], "o", color=C_ZS, ms=7, zorder=5)
        ax.set_title(dset, fontsize=9)
        ax.set_xticks([0] + BUDGETS); ax.set_xlabel("annotated frames", fontsize=8.5)
        ax.tick_params(labelsize=8); ax.grid(alpha=0.3); ax.set_ylim(bottom=0)
    axes_a[0].set_ylabel("pixel error at 50%\nensemble std", fontsize=8.5)
    axes_a[0].legend(fontsize=7, loc="upper right", framealpha=0.9)
    axes_a[0].text(-0.34, 1.16, "A", transform=axes_a[0].transAxes, fontsize=13, weight="bold")
    axes_a[2].text(0.5, 1.22, "few-shot adaptation: the shared backbone beats training the "
                              "same architecture from scratch",
                   transform=axes_a[2].transAxes, ha="center", fontsize=9.5, weight="bold")

    # ── panel C: mean pixel error across the masked settings ────────────────
    # Absolute error averaged over settings, not a ratio: the zero-shot base model then
    # appears as its own bar rather than as an implicit reference at 1. The mean is dominated
    # by the high-error settings, which the caption states; the per-setting grid is in the
    # appendix.
    rows = panel_b_values(results, inv, n=10)
    ax_p = fig.add_subplot(gs[1, :]); panel_protocol(ax_p)
    ax_p.text(-0.045, 1.06, "B", transform=ax_p.transAxes, fontsize=13, weight="bold")
    ax_l = fig.add_subplot(gs[2, 0:3]); ax_r = fig.add_subplot(gs[2, 3:6])
    series = [("base model\n(zero-shot)", "base", C_ZS)] + [(a, a, c) for a, _, c in ARMS]
    # both halves share one y scale: on separate scales the left panel's bars look as tall as
    # the right's and the contrast the figure exists to show disappears.
    all_means = [float(np.mean([r[key][i] for r in rows if r[key][i] is not None]))
                 for i in (0, 1) for _, key, _ in series]
    ymax = max(all_means) * 1.22
    for ax, idx, title, sub in ((ax_l, 0, "what the lab can measure",
                                 "keypoints it still annotates"),
                                (ax_r, 1, "what it cannot", "the withheld keypoint")):
        for j, (name, key, col) in enumerate(series):
            v = float(np.mean([r[key][idx] for r in rows if r[key][idx] is not None]))
            b = ax.bar([j], [v], 0.62, color=col)
            ax.bar_label(b, fmt="%.1f", fontsize=9.5, padding=2, weight="bold")
        ax.set_xticks(range(len(series)))
        ax.set_xticklabels([n for n, _, _ in series], fontsize=8.5)
        ax.set_title(f"{title}\n{sub}", fontsize=9.5, linespacing=1.3)
        ax.grid(axis="y", alpha=0.3); ax.tick_params(labelsize=8)
        ax.set_ylim(0, ymax)
    ax_l.set_ylabel("mean pixel error over the 9 settings", fontsize=8.5)
    ax_r.tick_params(labelleft=False)
    ax_l.text(-0.10, 1.22, "C", transform=ax_l.transAxes, fontsize=13, weight="bold")
    ax_l.text(1.12, 1.22, "masked-label protocol, $N{=}10$, averaged over all nine settings",
              transform=ax_l.transAxes, ha="center", fontsize=9.5, weight="bold")

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=300)
    print(f"wrote {out}")
    for r in rows:
        print(f"  {r['label'].replace(chr(10),' '):18s} " +
              "  ".join(f"{k}=({r[k][0]:.1f},{r[k][1]:.1f})" for k in ("base",) + tuple(a for a,_,_ in ARMS)))


if __name__ == "__main__":
    main()
