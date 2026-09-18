"""Build a pseudo-label replay CSV (SuperAnimal-style memory replay, Ye et al. 2024).

Faithful to their mechanism: during fine-tuning, keypoints the target does not annotate are
filled with the frozen base model's own zero-shot predictions as hard labels, kept only when
prediction confidence clears a threshold. Filling happens on the target's own frames; the
pretraining corpus is never touched. Only the N frames of the few-shot cell are filled, so
training on this CSV with the same ``train_frames``/``rng_seed_data_pt`` sees exactly the
plain cell's frames, with pseudo-labels added.

Inputs: the (possibly masked) train CSV, and a ``litpose predict`` predictions.csv of the
teacher on the cell's N frames. Channels eligible for filling are those the teacher was
trained for (the union of the other datasets' trainable keypoints), where the CSV has no
label.

Usage:
    python scripts/build_psl_replay_csv.py --dataset ibl --train_csv \
        CollectedData_ibl_train_mask-pupil_center_left.csv --preds <predictions.csv> \
        --threshold 0.6
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mouse_pose.paths import load_paths


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset",   required=True)
    p.add_argument("--train_csv", required=True, help="source train CSV name (in data_dir)")
    p.add_argument("--preds",     required=True, help="teacher predictions.csv from litpose predict")
    p.add_argument("--threshold", type=float, default=0.6, help="confidence gate for pseudo-labels")
    a = p.parse_args()

    data_dir = Path(load_paths()["data_dir"])
    inv      = json.loads((data_dir / "dataset_inventory.json").read_text())["datasets"]
    teacher_kps = sorted(set().union(*[set(v["trainable"]) for d, v in inv.items() if d != a.dataset]))

    df    = pd.read_csv(data_dir / a.train_csv, header=[0, 1, 2], index_col=0)
    preds = pd.read_csv(a.preds, header=[0, 1, 2], index_col=0)
    scorer = df.columns.get_level_values(0)[0]

    n_filled, n_gated = 0, 0
    for frame in preds.index:
        for kp in teacher_kps:
            if (scorer, kp, "x") not in df.columns:
                continue
            if not np.isnan(df.loc[frame, (scorer, kp, "x")]):
                continue  # ground truth present; SuperAnimal keeps it
            like = float(preds.loc[frame].xs(kp, level="bodyparts").xs("likelihood", level="coords").iloc[0])
            if like <= a.threshold:
                n_gated += 1
                continue  # below the gate: no pseudo-label, no protection
            row = preds.loc[frame].xs(kp, level="bodyparts")
            df.loc[frame, (scorer, kp, "x")]       = float(row.xs("x", level="coords").iloc[0])
            df.loc[frame, (scorer, kp, "y")]       = float(row.xs("y", level="coords").iloc[0])
            df.loc[frame, (scorer, kp, "visible")] = 2
            n_filled += 1

    out = data_dir / a.train_csv.replace("_train", "_train_psl").replace(".csv", f"-thr{a.threshold}.csv")
    df.to_csv(out)
    print(f"{out.name}: filled {n_filled} pseudo-labels over {len(preds)} frames "
          f"({n_gated} gated out at threshold {a.threshold})")


if __name__ == "__main__":
    main()
