"""Shared fixtures: a small synthetic raw dataset (CSV + config) and a configs/ directory.

The raw dataset has three sessions and three raw keypoints:

  session  side   excluded     raw keypoints
  sessA    left   no           nose (plain), paw (lateralized), tail (excluded keypoint)
  sessB    right  no
  sessX    -      yes

and is converted against the canonical vocabulary [nose, paw_left, paw_right, ear], so
ear is a canonical keypoint the dataset never labels.
"""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from mighty_mouse.labels import COORDS, SCORER

RAW_SCORER = "orig"
NAN = np.nan

# image path -> {raw keypoint: (x, y)}
RAW_ROWS = {
    "labeled-data/sessA/img0.png": {"nose": (1.0, 2.0), "paw": (3.0, 4.0), "tail": (5.0, 6.0)},
    "labeled-data/sessA/img1.png": {"nose": (NAN, NAN), "paw": (7.0, 8.0), "tail": (NAN, NAN)},
    "labeled-data/sessB/img0.png": {"nose": (9.0, 10.0), "paw": (NAN, NAN), "tail": (1.0, 1.0)},
    "labeled-data/sessX/img0.png": {"nose": (0.0, 0.0), "paw": (0.0, 0.0), "tail": (0.0, 0.0)},
}


def _labels_df(rows: dict[str, dict[str, tuple[float, float]]], scorer: str) -> pd.DataFrame:
    kps = list(next(iter(rows.values())))
    columns = pd.MultiIndex.from_tuples(
        [(scorer, kp, c) for kp in kps for c in ("x", "y")],
        names=["scorer", "bodyparts", "coords"],
    )
    data = [[v for kp in kps for v in row[kp]] for row in rows.values()]
    return pd.DataFrame(data, index=pd.Index(list(rows)), columns=columns)


@pytest.fixture
def make_raw_df() -> Callable[..., pd.DataFrame]:
    """Factory for raw LP label DataFrames: ``make_raw_df(rows, scorer=RAW_SCORER)``."""
    def _make(
        rows: dict[str, dict[str, tuple[float, float]]],
        scorer: str = RAW_SCORER,
    ) -> pd.DataFrame:
        return _labels_df(rows, scorer)
    return _make


@pytest.fixture
def raw_df() -> pd.DataFrame:
    """The synthetic raw dataset described in the module docstring."""
    return _labels_df(RAW_ROWS, RAW_SCORER)


@pytest.fixture
def canonical_kps() -> list[str]:
    return ["nose", "paw_left", "paw_right", "ear"]


@pytest.fixture
def dataset_config() -> dict:
    """Normalized dataset config for ``raw_df``."""
    return {
        "exclude": {"sessions": ["sessX"], "keypoints": ["tail"]},
        "keypoints": {"nose": "nose", "paw": "paw_{side}"},
        "sessions": {"sessA": "left", "sessB": "right"},
    }


@pytest.fixture
def make_processed_df() -> Callable[[dict[str, list[float]]], pd.DataFrame]:
    """Factory for processed (canonical) DataFrames from per-keypoint visibility values.

    ``make_processed_df({"kp": [2.0, 1.0]})`` gives SCORER-level x/y/visible columns for each
    keypoint; x/y are NaN wherever visible != 2.
    """
    def _make(vis: dict[str, list[float]]) -> pd.DataFrame:
        n_rows = len(next(iter(vis.values())))
        arrays = {}
        for kp, v in vis.items():
            v_arr = np.asarray(v, dtype=float)
            xy = np.where(v_arr == 2.0, 1.0, np.nan)
            for coord, values in zip(COORDS, (xy, xy, v_arr), strict=True):
                arrays[(SCORER, kp, coord)] = values
        idx = pd.Index([f"labeled-data/ds/sess/img{i}.png" for i in range(n_rows)])
        return pd.DataFrame(arrays, index=idx)
    return _make


@pytest.fixture
def model_cfg(canonical_kps: list[str]) -> dict:
    """A model.yaml dict consistent with ``canonical_kps``."""
    return {"data": {"keypoint_names": list(canonical_kps), "num_keypoints": len(canonical_kps)}}


@pytest.fixture
def configs_dir(tmp_path: Path, canonical_kps: list[str], model_cfg: dict) -> Path:
    """A configs/ directory with keypoints.yaml, model.yaml, and datasets/toy.yaml."""
    root = tmp_path / "configs"
    (root / "datasets").mkdir(parents=True)
    (root / "keypoints.yaml").write_text(yaml.safe_dump({"keypoints": canonical_kps}))
    (root / "model.yaml").write_text(yaml.safe_dump(model_cfg))
    (root / "datasets" / "toy.yaml").write_text(
        "keypoints:\n"
        "  nose: nose\n"
        "  paw: paw_{side}\n"
        "sessions:\n"
        "  sessA: left\n"
        "exclude:\n"
        "  sessions:\n"
        "  keypoints:\n"
        "    - tail\n"
    )
    return root
