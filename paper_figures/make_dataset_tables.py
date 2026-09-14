#!/usr/bin/env python
"""
Emit the paper's appendix dataset tables as LaTeX, from the machine-readable
inventory manifest written by `python -m mouse_pose.inventory`.

Two tables:
  1. corpus composition  -- frames, sessions, views and keypoint counts per dataset
  2. keypoint availability -- the 36-keypoint vocabulary x 5 datasets coverage matrix

Never hand-edit the generated block: rerun this after any change to the corpus.

    python paper_figures/make_dataset_tables.py --out ../paper/tables_dataset.tex
"""

import argparse
import json
from pathlib import Path

from mouse_pose.paths import load_paths

# ── canonical keypoint grouping (mirrors configs/model.yaml section banners) ──
GROUPS = [
    ("Eye",      ["eye_back_left", "eye_back_right", "eye_bottom_left", "eye_bottom_right",
                  "eye_front_left", "eye_front_right", "eye_top_left", "eye_top_right",
                  "pupil_center_left", "pupil_center_right"]),
    ("Nose and mouth", ["nose_tip", "nose_top", "nose_bottom", "pad_center", "mouth", "lowerlip",
                        "upperlip_left", "upperlip_right"]),
    ("Tongue",   ["tongue_tip", "tongue_center", "tongue_end_left", "tongue_end_right"]),
    ("Whiskers", ["pad_top_left", "pad_top_right", "pad_side_left", "pad_side_right"]),
    ("Ears",     ["ear_top_left", "ear_top_right", "ear_tip_left", "ear_tip_right",
                  "ear_bottom_left", "ear_bottom_right", "ear_base_left", "ear_base_right"]),
    ("Forelimb", ["wrist_left", "wrist_right"]),
]

# Paper display names; keys are the on-disk dataset names, which the paper does not use.
SHORT = {"facemap": "Facemap", "ibl": "IBL", "cheese-2d": "Cheese-3D",
         "cazettes-side": "Cazettes", "kondo": "Kondo"}


def esc(s: str) -> str:
    return s.replace("_", r"\_")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1], epilog=__doc__)
    p.add_argument("--out", required=True, help="path of the .tex fragment to write")
    args = p.parse_args()

    manifest = Path(load_paths()["data_dir"]) / "dataset_inventory.json"
    inv      = json.loads(manifest.read_text())
    order    = inv["registry"]
    ds       = inv["datasets"]
    canon    = inv["canonical_keypoints"]

    listed = [k for _, ks in GROUPS for k in ks]
    assert sorted(listed) == sorted(canon), (
        f"grouping is stale: {set(canon) ^ set(listed)}")

    L = []
    A = L.append

    # ── table 1: corpus composition ──────────────────────────────────────────
    A(r"\begin{table}[t]")
    A(r"  \footnotesize \centering")
    A(r"  \caption{\textbf{Corpus composition.} Frame and session counts per dataset, the "
      r"camera view of each session (left or right), and the size of each dataset's keypoint "
      r"set. \emph{Direct} is the number of keypoints the dataset annotates in the training "
      r"split; \emph{trainable} adds the lateral partners that horizontal-flip augmentation "
      r"supervises without direct annotation; \emph{eval} is the number annotated in the "
      r"held-out test split. Train and test sessions are disjoint for every dataset.}")
    A(r"  \label{tab:corpus}")
    A(r"  \begin{tabular}{lrrrrcrrr}")
    A(r"    \toprule")
    A(r"    & \multicolumn{2}{c}{Train} & \multicolumn{2}{c}{Test} & Sessions "
      r"& \multicolumn{3}{c}{Keypoints} \\")
    A(r"    \cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-6}\cmidrule(lr){7-9}")
    A(r"    Dataset & frames & sess. & frames & sess. & (L\,/\,R) & direct & train. & eval \\")
    A(r"    \midrule")
    tot = dict(tr=0, trs=0, te=0, tes=0)
    for name in order:
        d = ds[name]
        views = d.get("views", {})
        nl = sum(1 for v in views.values() if v == "left")
        nr = sum(1 for v in views.values() if v == "right")
        tot["tr"] += d["train_frames"]; tot["trs"] += d["train_sessions"]
        tot["te"] += d["test_frames"];  tot["tes"] += d["test_sessions"]
        A(f"    {esc(SHORT[name])} & {d['train_frames']:,} & {d['train_sessions']} & "
          f"{d['test_frames']:,} & {d['test_sessions']} & {nl}\\,/\\,{nr} & "
          f"{len(d['direct'])} & {len(d['trainable'])} & {len(d['eval'])} \\\\")
    A(r"    \midrule")
    A(f"    \\textbf{{total}} & \\textbf{{{tot['tr']:,}}} & \\textbf{{{tot['trs']}}} & "
      f"\\textbf{{{tot['te']:,}}} & \\textbf{{{tot['tes']}}} & & "
      f"\\multicolumn{{3}}{{c}}{{\\textbf{{36 canonical}}}} \\\\")
    A(r"    \bottomrule")
    A(r"  \end{tabular}")
    A(r"\end{table}")
    A("")

    # ── table 2: per-keypoint class matrix ───────────────────────────────────
    # For dataset d and keypoint k the class is decided by who annotates k:
    #   S supported  d annotates it and so does at least one other dataset
    #                (d's leave-one-out model therefore already knows it)
    #   N new        d is the ONLY annotator, so d's leave-one-out model never saw it
    #   T transfer   d does not annotate it but another dataset does, so d inherits it
    #                from the shared model and can never evaluate it locally
    hflip_only = {n: sorted(set(ds[n]["trainable"]) - set(ds[n]["direct"])) for n in order}
    A(r"\begin{table}[t]")
    A(r"  \footnotesize \centering")
    A(r"  \caption{\textbf{Keypoint class by dataset.} Each cell gives the class that keypoint "
      r"takes for that dataset, which is determined entirely by who annotates it: "
      r"\textsc{s}~(supported) the dataset annotates it and so does at least one other, so its "
      r"leave-one-out model already knows it; \textsc{n}~(new) the dataset is the only annotator, "
      r"so its leave-one-out model has never seen it; \textsc{t}~(transfer) the dataset does not "
      r"annotate it but another does, so it is inherited from the shared model and cannot be "
      r"evaluated locally. Blank means no dataset annotates it. Column $a$ counts annotating "
      r"datasets. Every \textsc{t} in this table is a keypoint some lab obtains only "
      r"through the shared model, and is exactly what standard adaptation destroys "
      r"(Section~\ref{sec:protocol}).}")
    A(r"  \label{tab:keypoint-coverage}")
    A(r"  \begin{tabular}{l" + "c" * len(order) + r"c}")
    A(r"    \toprule")
    A(r"    Keypoint & " + " & ".join(esc(SHORT[n]) for n in order) + r" & $a$ \\")
    tally = {"S": 0, "N": 0, "T": 0}
    for gname, keys in GROUPS:
        A(r"    \midrule")
        A(rf"    \multicolumn{{{len(order) + 2}}}{{l}}{{\emph{{{gname}}}}} \\")
        for k in keys:
            annot = [n for n in order if k in ds[n]["direct"]]
            cells = []
            for n in order:
                if k in ds[n]["direct"]:
                    c = "S" if len(annot) > 1 else "N"
                elif annot:
                    c = "T"
                else:
                    c = ""
                if c:
                    tally[c] += 1
                cells.append(rf"\textsc{{{c.lower()}}}" if c else "")
            A(f"    \\texttt{{{esc(k)}}} & " + " & ".join(cells) + f" & {len(annot)} \\\\")
    A(r"    \bottomrule")
    A(r"  \end{tabular}")
    A(r"  \\[3pt]")
    A(r"  \parbox{0.97\linewidth}{\footnotesize Across the corpus this gives "
      rf"{tally['S']} supported, {tally['N']} new and {tally['T']} transfer "
      r"(dataset, keypoint) pairs: the transfer class is the largest of the three, which is why "
      r"its behaviour under adaptation determines what a lab actually gains. "
      + ("Channels supervised only through the horizontal-flip partner, without direct "
         "annotation, are "
         + "; ".join(rf"{esc(SHORT[n])}: \texttt{{{', '.join(esc(x) for x in v)}}}"
                     for n, v in hflip_only.items() if v)
         + ".") + r"}")
    A(r"\end{table}")
    A("")

    out = Path(args.out)
    out.write_text("\n".join(L) + "\n")
    print(f"wrote {out} ({len(L)} lines)")


if __name__ == "__main__":
    main()
