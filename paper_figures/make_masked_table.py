#!/usr/bin/env python
"""
Emit the paper's main results table from the masked-label runs ONLY.

Every number in a row comes from the same adapted model: a fine-tuning run in which one
keypoint the target dataset really annotates was deleted from the training CSV. That model
is then scored on three disjoint keypoint sets, all derived from the corpus manifest:

  supported  target annotates it, its leave-one-out base model already knows it
  new        target annotates it, no other dataset does, so the base model never saw it
  masked     the withheld keypoint, scored against the annotations held back from training

Mixing these rows with numbers from the unmasked grid would compare different models, which
is why this script reads the `*-mask-*` roots exclusively.

    python paper_figures/make_masked_table.py --out ../paper/table_masked.tex
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mouse_pose.paths import load_paths

# One budget for every setting: N=10 is the smallest annotation budget a lab would
# realistically use, it is complete for every arm and subset, and holding it fixed keeps the
# settings comparable. Budget dependence is reported separately for the ibl pupil.
# Nine settings over six keypoint types, chosen to span keypoint character, teacher quality
# and label isolation. Depth on three (ibl pupil, ibl wrists, cazettes pupil) at three
# annotation subsets; breadth on the rest at subset 0. Whatever subsets exist are averaged and
# the count is reported, so the table cannot silently mix the two.
SETTINGS = [  # (dataset, masked keypoints, label); each is scored at both budgets
    ("ibl",           ["pupil_center_left"],               "pupil"),
    ("ibl",           ["wrist_left", "wrist_right"],       "wrists"),
    ("cazettes-side", ["pupil_center_left"],               "pupil"),
    ("cazettes-side", ["wrist_left", "wrist_right"],       "wrists"),
    ("kondo",         ["wrist_left", "wrist_right"],       "wrists"),
    ("cheese-2d",     ["eye_back_left", "eye_back_right"], "eye"),
    ("facemap",       ["nose_tip"],                        "nose"),
    ("kondo",         ["lowerlip"],                        "lowerlip"),
    ("cazettes-side", ["tongue_tip"],                      "tongue"),
]
BUDGETS = [10, 50]
SUFFIX = {"pupil": "pupil_center_left", "wrists": "wrist_left+wrist_right",
          "eye": "eye_back_left+eye_back_right", "nose": "nose_tip",
          "lowerlip": "lowerlip", "tongue": "tongue_tip"}
ARMS = [("Full fine-tuning", "fewshot-exp-lr5-zio-mask-{s}"),
        ("LoRA",             "fewshot-exp-lora-r16-lr5e-5-head5e-4-zio-mask-{s}"),
        ("Anchored LoRA",    "fewshot-exp-anchor-lora-conf1-zio-mask-{s}")]
LOO = {"facemap": "ibl+cheese+caz+kondo", "ibl": "face+cheese+caz+kondo",
       "cheese-2d": "face+ibl+caz+kondo", "cazettes-side": "face+ibl+cheese+kondo",
       "kondo": "face+ibl+cheese+caz"}
# Paper display names; keys are the on-disk dataset names, which the paper does not use.
DISPLAY = {"facemap": "Facemap", "ibl": "IBL", "cheese-2d": "Cheese-3D",
           "cazettes-side": "Cazettes", "kondo": "Kondo"}


def pooled(csv: Path, kps: list[str]) -> float | None:
    """Pooled pixel error over frames x keypoints; NaN entries (unlabeled) are dropped."""
    if not csv.exists():
        return None
    df = pd.read_csv(csv, index_col=0)
    cols = [k for k in kps if k in df.columns]
    if not cols:
        return None
    a = df[cols].to_numpy(dtype=float).ravel()
    return float(np.nanmean(a)) if np.isfinite(a).any() else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1], epilog=__doc__)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    paths = load_paths()
    res   = Path(paths["results_dir"])
    inv   = json.loads((Path(paths["data_dir"]) / "dataset_inventory.json").read_text())
    direct = {d: set(v["direct"]) for d, v in inv["datasets"].items()}

    L, A = [], None
    A = L.append
    A(r"\begin{table}[t]")
    # 9 settings x 4 arms fills a page almost exactly; the tightened rows keep it from
    # overflowing by a few points and being deferred another page
    A(r"  \footnotesize \centering \renewcommand{\arraystretch}{0.92} \setlength{\tabcolsep}{4.5pt}")
    A(r"  \caption{\textbf{Every row is one adapted model, scored on three disjoint keypoint "
      r"sets.} Under the masked-label protocol one keypoint the target dataset really annotates "
      r"is deleted from the fine-tuning set; the resulting model is then scored on the keypoints "
      r"it still annotates and the base model knows (\emph{supported}), those only it annotates "
      r"(\emph{new}), and the withheld keypoint against the annotations held back "
      r"(\emph{masked}). Pooled pixel error $\downarrow$, at both annotation budgets, averaged "
      r"over whatever annotation subsets exist for that cell: three for the IBL pupil, IBL wrists "
      r"and Cazettes pupil, one for the rest. "
      r"\emph{n/a} marks a class that is empty for that dataset rather than a missing "
      r"measurement: Kondo annotates no keypoint that no other dataset also annotates, so it has "
      r"no \emph{new} column. "
      r"The zero-shot row is the frozen leave-one-out base model, which is also the distillation "
      r"target. \emph{The supported and new columns are what a lab can measure; the masked "
      r"column is what it cannot.}}")
    A(r"  \label{tab:masked}")
    A(r"  \begin{tabular}{llcccccc}")
    A(r"    \toprule")
    A(r"    & & \multicolumn{3}{c}{$N{=}10$} & \multicolumn{3}{c}{$N{=}50$} \\")
    A(r"    \cmidrule(lr){3-5}\cmidrule(lr){6-8}")
    A(r"    Setting & Method & sup. & new & \textbf{masked} & sup. & new & \textbf{masked} \\")

    summary = []
    for dset, masked, label in SETTINGS:
        base   = res / "zoom-aug-exp" / f"{LOO[dset]}-T2-zoominout" / "seed0" / "eval" / dset
        others = set().union(*[v for k, v in direct.items() if k != dset])
        ann    = direct[dset]
        sup    = sorted((ann & others) - set(masked))
        new    = sorted(ann - others - set(masked))
        groups = (sup, new, masked)
        A(r"    \midrule")
        A(rf"    \multicolumn{{8}}{{l}}{{\emph{{{esc(DISPLAY.get(dset, dset))}, {label} masked}}}} \\")

        # the frozen base model does not depend on the budget, so it repeats across both halves
        # an empty keypoint group is a property of the corpus, not a missing run: kondo
        # annotates nothing that no other dataset annotates, so its "new" column cannot exist
        empty = [len(g) == 0 for g in groups]
        z = [pooled(base / "pixel_error.csv", g) for g in groups]
        A(r"    & zero-shot & "
          + " & ".join("n/a" if empty[i % 3] else (f"{v:.1f}" if v is not None else "n/a")
                       for i, v in enumerate(z * 2)) + r" \\")

        rows = {arm: [] for arm, _ in ARMS}
        best = {}
        for N in BUDGETS:
            for arm, tmpl in ARMS:
                root = res / tmpl.format(s=SUFFIX[label]) / dset
                for grp in groups:
                    per = [pooled(root / f"tf{N}-draw{d}" / "eval" / dset / "pixel_error.csv", grp)
                           for d in (0, 1, 2)]
                    per = [v for v in per if v is not None]
                    rows[arm].append(float(np.mean(per)) if per else None)
            # bold the best masked cell within this budget only
            col = 3 * BUDGETS.index(N) + 2
            cand = [a for a in rows if rows[a][col] is not None]
            best[col] = min(cand, key=lambda a: rows[a][col]) if cand else None

        for arm, _ in ARMS:
            cells = []
            for i, x in enumerate(rows[arm]):
                if empty[i % 3]:
                    cells.append("n/a")
                elif x is None:
                    cells.append(r"\inflight{run}")
                elif best.get(i) == arm:
                    cells.append(rf"\textbf{{{x:.1f}}}")
                else:
                    cells.append(f"{x:.1f}")
            A(f"    & {arm} & " + " & ".join(cells) + r" \\")
        summary.append((dset, label, z, rows))

    A(r"    \bottomrule")
    A(r"  \end{tabular}")
    A(r"  \\[3pt]")
    A(r"  \parbox{0.97\linewidth}{\footnotesize Full fine-tuning updates all 21.6M backbone "
      r"parameters; LoRA and anchored LoRA update 1.36M (1.33M adapters, 31K head). None requires "
      r"access to the source corpora at adaptation time, unlike memory replay, and all return a "
      r"single model, unlike routing, which requires the base model and an adapted model at "
      r"inference.}")
    A(r"\end{table}")

    Path(args.out).write_text("\n".join(L) + "\n")
    print(f"wrote {args.out}\n")
    for dset, label, z, rows in summary:
        print(f"{dset} / {label} masked      " + "".join(f"{c:>8s}" for c in
              ("sup10", "new10", "msk10", "sup50", "new50", "msk50")))
        print(f"   {'zero-shot':18s} " + "".join(f"{v:8.1f}" if v is not None else f"{'n/a':>8s}"
              for v in list(z) * 2))
        for arm in rows:
            print(f"   {arm:18s} " + "".join(f"{v:8.1f}" if v is not None else f"{'--':>8s}"
                  for v in rows[arm]))


def esc(s: str) -> str:
    return s.replace("_", r"\_")


if __name__ == "__main__":
    main()
