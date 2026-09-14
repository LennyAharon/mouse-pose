#!/usr/bin/env python
"""
Emit the appendix table of dataset sampling shares at each temperature.

Reproduces the arithmetic of ``lightning_pose.data.samplers.TemperatureSampler``: the
supervision mass of a dataset is m_d = n_d * kbar_d, the target supervision share is
p_d proportional to m_d^(1/T), and the frame-draw probability is q_d proportional to
p_d / kbar_d. kbar_d is the mean number of visible==2 keypoints per training frame,
computed from the training CSVs.

    python paper_figures/make_table_sampling.py --out ../paper/table_sampling.tex
"""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from mouse_pose.paths import load_paths

DS = ["facemap", "ibl", "cheese-2d", "cazettes-side", "kondo"]
# Paper display names; keys are the on-disk dataset names, which the paper does not use.
DISPLAY = {"facemap": "Facemap", "ibl": "IBL", "cheese-2d": "Cheese-3D",
           "cazettes-side": "Cazettes", "kondo": "Kondo"}
TEMPS = [(1.0, "1"), (2.0, "2"), (math.inf, r"\infty")]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1], epilog=__doc__)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    data = Path(load_paths()["data_dir"])

    rows = []
    for d in DS:
        df = pd.read_csv(data / f"CollectedData_{d}_train.csv", header=[0, 1, 2], index_col=0)
        s = df.columns.get_level_values(0)[0]
        kps = list(dict.fromkeys(df.columns.get_level_values(1)))
        vis = np.stack([pd.to_numeric(df[(s, k, "visible")], errors="coerce").to_numpy()
                        for k in kps], axis=1)
        rows.append((d, len(df), float(np.nanmean((vis == 2).sum(axis=1)))))

    kbar = np.array([r[2] for r in rows])
    m = np.array([r[1] * r[2] for r in rows])
    sup = {}
    for T, _ in TEMPS:
        p = np.ones_like(m) if math.isinf(T) else m ** (1.0 / T)
        p = p / p.sum()
        q = p / kbar
        q = q / q.sum()
        sup[T] = (q, (q * kbar) / (q * kbar).sum())

    L = ["\\begin{table}[t]", "  \\footnotesize \\centering",
         "  \\caption{\\textbf{Dataset sampling at each temperature.} $n_d$ is the number of "
         "training frames and $\\bar{k}_d$ the mean number of annotated keypoints per frame, so "
         "the supervision mass $m_d=n_d\\bar{k}_d$ is the count of labelled keypoint "
         "observations a dataset contributes. The sampler targets a supervision share "
         "$p_d\\propto m_d^{1/T}$ and converts it to the frame-draw probability "
         "$q_d\\propto p_d/\\bar{k}_d$. $T{=}1$ is frame-proportional and equivalent to a plain "
         "shuffled loader; $T{=}\\infty$ equalises supervision exactly. Note that the two "
         "columns differ: Cheese-3D annotates 13.2 keypoints per frame against IBL's 4.0, so it "
         "needs far fewer frames to reach the same supervision.}",
         "  \\label{tab:sampling}", "  \\centering",
         "  \\begin{tabular}{lrrr" + "r" * (2 * len(TEMPS)) + "}", "    \\toprule",
         "    & & & & \\multicolumn{3}{c}{frames drawn $q_d$} & "
         "\\multicolumn{3}{c}{supervision received} \\\\",
         "    \\cmidrule(lr){5-7}\\cmidrule(lr){8-10}",
         "    Dataset & $n_d$ & $\\bar{k}_d$ & $m_d$ & "
         + " & ".join(f"$T{{=}}{lbl}$" for _, lbl in TEMPS) + " & "
         + " & ".join(f"$T{{=}}{lbl}$" for _, lbl in TEMPS) + r" \\", "    \\midrule"]
    for i, (d, n, kb) in enumerate(rows):
        L.append(f"    {DISPLAY.get(d, d).replace('_', chr(92)+'_')} & {n:,} & {kb:.1f} & {m[i]:,.0f} & "
                 + " & ".join(f"{sup[T][0][i]:.1%}".replace('%', r'\%') for T, _ in TEMPS) + " & "
                 + " & ".join(f"{sup[T][1][i]:.1%}".replace('%', r'\%') for T, _ in TEMPS)
                 + r" \\")
    L += ["    \\bottomrule", "  \\end{tabular}", "\\end{table}"]
    Path(args.out).write_text("\n".join(L) + "\n")
    print(f"wrote {args.out}")
    for i, (d, n, kb) in enumerate(rows):
        print(f"  {d:15s} n={n:5d} kbar={kb:5.2f}  q@T2={sup[2.0][0][i]:.1%}  "
              f"sup@T2={sup[2.0][1][i]:.1%}")


if __name__ == "__main__":
    main()
