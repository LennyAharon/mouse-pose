#!/usr/bin/env python
"""
Emit the appendix table of per-dataset zoom-augmentation ranges.

Read straight from ``configs/model_zoominout.yaml`` rather than transcribed, so the table
cannot drift from the recipe that was actually trained. imgaug's ``CropAndPad`` takes a
fraction ``p`` of each side: negative crops (enlarging the animal), positive pads (shrinking
it), and with ``keep_size: true`` the result is resized back, so apparent scale is multiplied
by 1/(1+2p). Each dataset's range is therefore reported as the magnification it reaches at
its lower bound and the reduction it reaches at its upper bound.

    python paper_figures/make_table_zoom.py --out ../paper/table_zoom.tex
"""

import argparse
from pathlib import Path

import yaml

from mouse_pose.paths import load_paths

ORDER = ["facemap", "cheese-2d", "cazettes-side", "kondo", "ibl"]
# Snout span as a fraction of frame width, the measurement behind the ranges (Appendix D).
# ibl is absent because it shares no landmark pair with the four that carry the measurement.
SPAN = {"facemap": "55.8", "cazettes-side": "25.2", "cheese-2d": "21.3", "kondo": "19.2",
        "ibl": "n/a"}
# Paper display names; keys are the on-disk dataset names, which the paper does not use.
DISPLAY = {"facemap": "Facemap", "ibl": "IBL", "cheese-2d": "Cheese-3D",
           "cazettes-side": "Cazettes", "kondo": "Kondo"}


def factor(p: float) -> float:
    """Apparent-scale multiplier for a CropAndPad fraction p, with keep_size."""
    return 1.0 / (1.0 + 2.0 * p)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1], epilog=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=None, help="defaults to configs/model_zoominout.yaml")
    args = ap.parse_args()

    root = Path(load_paths().get("repo_dir", Path(__file__).resolve().parents[1]))
    cfg  = yaml.safe_load(Path(args.config or root / "configs" / "model_zoominout.yaml").read_text())
    per  = cfg["training"]["imgaug_per_dataset_zoom"]
    dflt = cfg["training"]["imgaug"]["CropAndPad"]
    p_default, prob = dflt["kwargs"]["percent"], dflt["p"]

    L = [r"\begin{table}[t]", r"  \footnotesize \centering",
         r"  \caption{\textbf{Per-dataset zoom augmentation.} Each source is augmented over its "
         r"own range of imgaug \texttt{CropAndPad} fractions $p$, applied with probability "
         rf"{prob} and \texttt{{keep\_size: true}}, so a frame is cropped or padded by $p$ per "
         r"side and resized back and the animal's apparent scale is multiplied by $1/(1{+}2p)$. "
         r"Negative $p$ enlarges, positive $p$ shrinks. The ranges are asymmetric by design: "
         r"they are chosen so that every dataset is seen at both the largest and the smallest "
         r"apparent scale the corpus contains, which is why Facemap, already the most magnified "
         r"rig, is padded out to five times smaller while IBL, which sets the reference scale, "
         r"is instead cropped in to five times larger. \emph{Snout span} is the measurement the "
         r"ranges were set from, the \texttt{nose\_tip} to \texttt{lowerlip} distance as a "
         r"percentage of frame width; IBL annotates no such pair and its range was set by "
         rf"inspection. Datasets not listed fall back to $p\in[{p_default[0]}, {p_default[1]}]$. "
         r"Figure~\ref{fig:augex} shows the output of these pipelines and "
         r"Figure~\ref{fig:aug} the accuracy they buy.}",
         r"  \label{tab:zoom}",
         r"  \begin{tabular}{lcccc}", r"    \toprule",
         r"    Dataset & snout span (\%) & $p$ range & max enlargement & max reduction \\",
         r"    \midrule"]
    for d in ORDER:
        lo, hi = per[d]
        L.append(rf"    {DISPLAY.get(d, d).replace('_', chr(92) + '_')} & {SPAN[d]} & $[{lo}, {hi}]$ & "
                 rf"${factor(lo):.1f}\times$ & ${1 / factor(hi):.1f}\times$ smaller \\")
    L += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]

    Path(args.out).write_text("\n".join(L) + "\n")
    print(f"wrote {args.out}\n")
    print(f"{'dataset':16s}{'p range':>18s}{'enlarge':>10s}{'reduce':>10s}{'span %':>9s}")
    for d in ORDER:
        lo, hi = per[d]
        print(f"{d:16s}{f'[{lo}, {hi}]':>18s}{factor(lo):9.1f}x{1 / factor(hi):9.1f}x"
              f"{SPAN[d]:>9s}")


if __name__ == "__main__":
    main()
