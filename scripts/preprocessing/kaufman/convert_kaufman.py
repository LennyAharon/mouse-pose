#!/usr/bin/env python3
"""
Build _raw/kaufman's top-level CollectedData.csv / CollectedData_test.csv from its
104 per-session CollectedData_Hank.csv files (labeled-data/<timestamp>-cam[12]/).

The source is already standard DLC format (project.yaml, labeled-data/, images all
present) -- each session dir just has its own CSV instead of one combined project-level
CSV, so this script's only job is to concatenate all 104 and split. No image copying is
needed (everything already lives at its final _raw/kaufman/ path).

kaufman is a calibrated multi-view source (shared calibration.toml, cam1/cam2 covering
the same trials) but is being built as a SINGLE-view dataset here per an explicit user
decision -- cam1 and cam2 sessions are kept as independent rows, not merged into one
multi-view sample.

Split is by session timestamp (the "<timestamp>" in "<timestamp>-cam[12]"), not by
subject: video filenames carry subject IDs (e.g. "b8sSM5"), but only 2 of the 52
distinct labeled-data timestamps overlap with the 24 sample videos, so a subject
can't be recovered for most sessions. Grouping by timestamp (rather than by
individual labeled-data dir) keeps a session's cam1 and cam2 views -- the same
trial, viewed twice -- on the same side of the split, since treating them as
unrelated would leak near-duplicate frames across train/test.

Usage:
    conda run -n pose python scripts/preprocessing/kaufman/convert_kaufman.py
"""

import argparse
import re
from pathlib import Path

import pandas as pd

from mouse_pose.paths import load_paths
from mouse_pose.subject_split import subject_split

# Target ~10-15% of frames in test. The greedy split in subject_split() only ever
# overshoots (it stops as soon as the running total reaches the target) -- aiming for
# the range's midpoint leaves room for that overshoot while still landing in range.
TEST_FRACTION = 0.125

SESSION_RE = re.compile(r"^(?P<timestamp>\d{8}-\d{6})-cam[12]$")


def timestamp_of(session_dir: str) -> str:
    m = SESSION_RE.match(session_dir)
    if not m:
        raise ValueError(f"unrecognized session dir name: {session_dir!r}")
    return m.group("timestamp")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0, help="split seed (default: 0)")
    args = parser.parse_args()

    paths = load_paths()
    raw_dir = Path(paths["raw_dir"])
    out_dir = raw_dir / "kaufman"

    session_dirs = sorted(p for p in (out_dir / "labeled-data").iterdir() if p.is_dir())
    print(f"found {len(session_dirs)} session dirs")

    dfs = []
    for session_dir in session_dirs:
        csvs = list(session_dir.glob("CollectedData_*.csv"))
        assert len(csvs) == 1, f"expected exactly 1 CollectedData_*.csv in {session_dir}, found {len(csvs)}"
        df = pd.read_csv(csvs[0], header=[0, 1, 2], index_col=0)
        dfs.append(df)

    combined = pd.concat(dfs)
    for df in dfs[1:]:
        assert df.columns.equals(dfs[0].columns), "keypoint schemas differ across sessions"
    print(f"combined: {len(combined)} frames, {len(combined.columns) // 2} keypoints")

    session_of_row = pd.Series(
        [Path(p).parts[-2] for p in combined.index], index=combined.index
    )
    timestamp_of_row = session_of_row.map(timestamp_of)
    counts = timestamp_of_row.value_counts().to_dict()
    print(f"{len(counts)} distinct session timestamps")

    train_ts, test_ts = subject_split(counts, args.seed, TEST_FRACTION)
    print(f"timestamps: {len(train_ts)} train, {len(test_ts)} test")

    for split_name, split_timestamps in (("train", train_ts), ("test", test_ts)):
        split_df = combined.loc[timestamp_of_row.isin(split_timestamps)]
        csv_name = "CollectedData.csv" if split_name == "train" else "CollectedData_test.csv"
        split_df.to_csv(out_dir / csv_name)
        n_sessions = session_of_row[timestamp_of_row.isin(split_timestamps)].nunique()
        print(f"  {split_name}: {len(split_df)} frames, {n_sessions} session dirs -> {out_dir / csv_name}")

    print("\nDone.")


if __name__ == "__main__":
    main()
