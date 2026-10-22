#!/usr/bin/env python
"""
Zero-shot along training: score every leave-out trunk's periodic-checkpoint predictions of the
dataset it never saw (<run>/zeroshot/steps/step<k>_<ds>_{test,train}_predictions.csv, from
scripts/delta/zeroshot_steps.sbatch) like scripts/zeroshot_score.py: visible == 2 labels, pooled
over the keypoints the trunk was trained on ("TRAINED"), mean / median px, detection (likelihood
>= --conf) and confident-but-wrong (>= --conf and > --wrong_px). The run's final predictions (its
-best.ckpt: eval/<ds>/predictions.csv, zeroshot/<ds>_train_predictions.csv) are step "best".

Question it answers: does zero-shot transfer peak before the end of training (early stopping /
fewer steps would transfer better), or keep improving to the last step?

    python scripts/zeroshot_steps.py                              # area trunks-headmlp256
    python scripts/zeroshot_steps.py --area trunks-headmlp256 --out <dir>

Writes <results_dir>/qualitative/zeroshot-steps-<MM-DD>/{per_step.csv, README.md}.
"""

import argparse
import json
import re
from datetime import date
from pathlib import Path

import pandas as pd
from zeroshot_score import read_preds, score

from mighty_mouse.comparison import trained_keypoints
from mighty_mouse.paths import load_paths
from mighty_mouse.registry import load_registry

LEAF = "supervised/sampling-T2/tf1/vits_dinov3"
SUFFIX = "_train_predictions.csv"


def best_step(run: Path) -> int | None:
    """Training step of the run's -best.ckpt (from its file name), or None."""
    for ck in run.glob("tb_logs/*/version_*/checkpoints/*-best.ckpt"):
        m = re.search(r"step=(\d+)", ck.name)
        if m:
            return int(m.group(1))
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1], epilog=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    paths = load_paths()
    ap.add_argument("--results_dir", type=Path, default=Path(paths["results_dir"]))
    ap.add_argument("--data_dir",    type=Path, default=Path(paths["data_dir"]))
    ap.add_argument("--area",        default="trunks-headmlp256",
                    help="trunk folder under results_dir")
    ap.add_argument("--conf",        type=float, default=0.7)
    ap.add_argument("--wrong_px",    type=float, default=25.0)
    ap.add_argument("--out",         type=Path, default=None)
    args = ap.parse_args()

    datasets = load_registry()
    inventory = json.loads((args.data_dir / "dataset_inventory.json").read_text())["datasets"]
    rows = []
    for run in sorted((args.results_dir / args.area).glob(f"*_train/{LEAF}/seed*")):
        left = [p.name[: -len(SUFFIX)] for p in sorted((run / "zeroshot").glob(f"*{SUFFIX}"))]
        if not left or not (run / "zeroshot" / "steps").is_dir():
            continue
        trained = trained_keypoints(set(datasets) - set(left), inventory)
        bstep = best_step(run)
        for ds in left:
            labels = {s: pd.read_csv(args.data_dir / f"CollectedData_{ds}_{s}.csv",
                                     header=[0, 1, 2], index_col=0)
                      for s in ("test", "train")}
            preds = {}
            for f in (run / "zeroshot" / "steps").glob(f"step*_{ds}_*_predictions.csv"):
                m = re.fullmatch(rf"step(\d+)_{re.escape(ds)}_(test|train)_predictions\.csv",
                                 f.name)
                if m:
                    preds[(m.group(1).lstrip("0") or "0", m.group(2))] = f
            preds[("best", "test")] = run / "eval" / ds / "predictions.csv"
            preds[("best", "train")] = run / "zeroshot" / f"{ds}{SUFFIX}"
            for (step, split), f in preds.items():
                if not f.exists():
                    continue
                t = score(labels[split], read_preds(f), args.conf, args.wrong_px, trained)
                t = t[t.keypoint == "TRAINED"].drop(columns="keypoint")
                t.insert(0, "split", split)
                t.insert(0, "step", step)
                t.insert(0, "dataset", ds)
                t["best_step"] = bstep
                t["run"] = str(run.relative_to(args.results_dir))
                rows.append(t)
    if not rows:
        raise SystemExit(f"no zeroshot/steps/ predictions under {args.results_dir / args.area}")
    S = pd.concat(rows, ignore_index=True)
    out = args.out or args.results_dir / "qualitative" / f"zeroshot-steps-{date.today():%m-%d}"
    out.mkdir(parents=True, exist_ok=True)
    S.round(4).to_csv(out / "per_step.csv", index=False)

    # ── README: one row per left-out dataset and split, steps as columns ──
    steps = sorted({s for s in S.step if s != "best"}, key=int) + ["best"]
    lines = ["---", "title: Zero-shot along training (leave-out trunks, periodic checkpoints)",
             f"date: {date.today()}", f"data_version: {args.data_dir.name}", f"area: {args.area}",
             "question: Does zero-shot transfer peak before the end of training?",
             "outputs: per_step.csv", "---", "",
             "# Zero-shot along training", "",
             "Left-out dataset, pooled over the keypoints the trunk was trained on (TRAINED).",
             "Cells: mean / median px in the dataset's own image pixels; **bold** = lowest",
             "mean in the row. `best` = the run's -best.ckpt (lowest validation loss on the",
             "TRAINING datasets), whose step is in the `best ckpt step` column.", ""]
    for split in ("test", "train"):
        T = S[S.split == split]
        lines += [f"## {split.upper()} frames", "",
                  "| left-out | "
                  + " | ".join(f"step {s}" if s != "best" else "best" for s in steps)
                  + " | best ckpt step | change last -> min mean |",
                  "|---|" + "---|" * (len(steps) + 2)]
        for ds in [d for d in datasets if d in set(T.dataset)]:
            r = T[T.dataset == ds].set_index("step")
            numeric = r.loc[[s for s in steps if s in r.index and s != "best"]]
            low = numeric.mean_px.idxmin() if len(numeric) else None
            cells = []
            for s in steps:
                if s not in r.index:
                    cells.append("")
                    continue
                c = f"{r.loc[s, 'mean_px']:.1f} / {r.loc[s, 'median_px']:.1f}"
                cells.append(f"**{c}**" if s == low else c)
            last = numeric.mean_px.iloc[-1] if len(numeric) else float("nan")
            delta = (numeric.mean_px.min() - last) / last if len(numeric) else float("nan")
            bstep = r.best_step.dropna().iloc[0] if r.best_step.notna().any() else ""
            lines.append(f"| {ds} | " + " | ".join(cells)
                         + f" | {bstep} | {delta:+.0%} (step {low}) |")
        lines.append("")
    (out / "README.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
