"""Write a training CSV with selected keypoints hidden (masked-label transfer protocol).

The masked keypoints get x/y = NaN and visible = 0 in every row, so a fine-tune sees no label
for them; the dataset's TEST csv is untouched, so the keypoint can be scored afterwards. This
measures "transfer a keypoint the lab never labeled" on the target domain itself.

    python scripts/build_masked_csv.py --dataset ibl --mask pupil_center_left
    -> <data_dir>/CollectedData_ibl_train_mask-pupil_center_left.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from mouse_pose.paths import load_paths


def slug(keys: list[str]) -> str:
    return "+".join(keys)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True)
    p.add_argument("--mask",    required=True, help="comma-separated keypoint names to hide")
    a = p.parse_args()
    keys = [k for k in a.mask.split(",") if k]
    data_dir = Path(load_paths()["data_dir"])
    src = data_dir / f"CollectedData_{a.dataset}_train.csv"
    dst = data_dir / f"CollectedData_{a.dataset}_train_mask-{slug(keys)}.csv"
    df  = pd.read_csv(src, header=[0, 1, 2], index_col=0)
    names = set(df.columns.get_level_values(1))
    unknown = [k for k in keys if k not in names]
    if unknown:
        raise SystemExit(f"not in {src.name}: {unknown}")
    n_hidden = 0
    for k in keys:
        cols = [c for c in df.columns if c[1] == k]
        n_hidden += int(df[[c for c in cols if c[2] == "x"]].notna().sum().sum())
        for c in cols:
            df[c] = 0 if c[2] == "visible" else np.nan
    df.to_csv(dst)
    print(f"{dst.name}: hid {n_hidden} labels of {keys} over {len(df)} rows")


if __name__ == "__main__":
    main()
