#!/usr/bin/env python3
"""
Build _raw/hantman-mv (single-view) from the multi-view DLC project at
_raw/_dlc/hantman-mv.

The source is already standard DLC format, split across two per-view CSVs
(CollectedData_side.csv, CollectedData_front.csv) that share an identical
keypoint schema -- side and front are just two camera angles on the same
17-keypoint finger/paw/pellet skeleton, with session directories that already
encode the view (e.g. "KPC188_20260110_v038_side" / "..._front"), so there's no
naming collision. "Converting to single-view" here means nothing more than
concatenating the two CSVs and writing a project.yaml with no view_names --
no lateralization/remapping happens here (that's a stage-2, canonical-vocab
concern, not this script's job).

Only the images actually referenced by a CSV row are copied -- each
labeled-data/<session>/ directory on disk holds many more frames than are
labeled (unlabeled context frames), so a wholesale directory copy would pull
in a lot of unused images.

Split is subject-level (first '_'-delimited token of the session name,
uppercased against casing slips), same convention as scripts/preprocessing/
hantman-sleap/convert_hantman_sleap.py.

Usage:
    conda run -n pose python scripts/preprocessing/hantman-mv/convert_hantman_mv.py
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from mouse_pose.paths import load_paths

# Target ~10-15% of frames in test. The greedy split below only ever overshoots
# (it stops as soon as the running total reaches the target), and subject sizes
# here are uneven enough that aiming for the range's upper edge (0.15) regularly
# overshoots past it -- aiming for the midpoint leaves room for that overshoot
# while still landing in range.
TEST_FRACTION = 0.125
VIEW_CSVS = ["CollectedData_side.csv", "CollectedData_front.csv"]


def _subject_of(session: str) -> str:
    return session.split("_")[0].upper()


def _split_subjects(df: pd.DataFrame, seed: int) -> tuple[set, set]:
    """Subject-level train/test split targeting TEST_FRACTION of frames."""
    subject_of_row = pd.Series(
        [_subject_of(Path(p).parts[-2]) for p in df.index], index=df.index
    )
    counts = subject_of_row.value_counts().to_dict()

    subjects = list(counts.keys())
    np.random.default_rng(seed).shuffle(subjects)

    total = len(df)
    target = total * TEST_FRACTION

    test_subjects = set()
    test_count = 0
    for s in subjects:
        if test_count >= target:
            break
        test_subjects.add(s)
        test_count += counts[s]

    train_subjects = set(subjects) - test_subjects
    return train_subjects, test_subjects


def copy_images(index: pd.Index, source_dir: Path, out_dir: Path) -> int:
    copied = 0
    for rel_path in index:
        src = source_dir / rel_path
        dst = out_dir / rel_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy2(src, dst)
            copied += 1
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0, help="subject-split seed (default: 0)")
    args = parser.parse_args()

    paths = load_paths()
    raw_dir = Path(paths["raw_dir"])
    source_dir = raw_dir / "_dlc" / "hantman-mv"
    out_dir = raw_dir / "hantman-mv"
    out_dir.mkdir(parents=True, exist_ok=True)

    dfs = []
    for csv_name in VIEW_CSVS:
        view_df = pd.read_csv(source_dir / csv_name, header=[0, 1, 2], index_col=0)
        print(f"  {csv_name}: {len(view_df)} frames")
        dfs.append(view_df)

    assert dfs[0].columns.equals(dfs[1].columns), "side/front keypoint schemas differ"
    combined = pd.concat(dfs)
    print(f"\ncombined: {len(combined)} frames, {len(combined.columns) // 2} keypoints")

    train_subjects, test_subjects = _split_subjects(combined, args.seed)
    print(f"subjects: {len(train_subjects)} train, {len(test_subjects)} test")

    subject_of_row = pd.Series(
        [_subject_of(Path(p).parts[-2]) for p in combined.index], index=combined.index
    )
    for split_name, split_subjects in (("train", train_subjects), ("test", test_subjects)):
        split_df = combined.loc[subject_of_row.isin(split_subjects)]
        csv_name = "CollectedData.csv" if split_name == "train" else "CollectedData_test.csv"
        split_df.to_csv(out_dir / csv_name)
        print(f"  {split_name}: {len(split_df)} frames -> {out_dir / csv_name}")

    n_copied = copy_images(combined.index, source_dir, out_dir)
    print(f"  images: {n_copied} copied, {len(combined) - n_copied} already present")

    with open(source_dir / "project.yaml") as f:
        project = yaml.safe_load(f)
    old_views = project.get("view_names")
    project["view_names"] = []
    with open(out_dir / "project.yaml", "w") as f:
        yaml.safe_dump(project, f, default_flow_style=False, sort_keys=False)
    print(f"  project.yaml written with view_names: [] (was {old_views})")

    (out_dir / "videos").mkdir(exist_ok=True)

    print("\nDone.")


if __name__ == "__main__":
    main()
