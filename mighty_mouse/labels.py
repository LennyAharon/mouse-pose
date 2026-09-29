"""
Conventions for Lightning Pose label CSVs shared across the repo.

LP label (and prediction) CSVs have a three-level column header (scorer / bodyparts /
coords) and an index of image paths ending in ``.../<session>/<frame>.png``.
"""

from pathlib import Path

import pandas as pd

# scorer name written into every converted / combined CSV
SCORER = "All"

# per-keypoint columns in converted CSVs
COORDS = ("x", "y", "visible")


def read_labels_csv(path: Path | str) -> pd.DataFrame:
    """Read an LP-format label or prediction CSV.

    Args:
        path: path to the CSV file

    Returns:
        DataFrame with a (scorer, bodyparts, coords) column MultiIndex, indexed by image path
    """
    return pd.read_csv(path, header=[0, 1, 2], index_col=0)


def session_name(img_path: str) -> str:
    """Return the session (parent directory name) of a labeled-frame image path."""
    return Path(img_path).parts[-2]


def frame_name(img_path: str) -> str:
    """Return the frame filename of a labeled-frame image path."""
    return Path(img_path).parts[-1]
