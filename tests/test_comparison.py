"""Test the scoring helpers of the mighty_mouse.comparison module."""

import numpy as np
import pandas as pd
import pytest
import yaml

from mighty_mouse.comparison import (
    ALWAYS_EXCLUDED,
    target_scores,
    trained_keypoints,
    visible_errors,
)
from mighty_mouse.configs import repo_root


@pytest.fixture
def errors() -> pd.DataFrame:
    """Errors as visible_errors returns them: NaN where the cell is not visible == 2."""
    return pd.DataFrame({"nose": [1.0, 3.0, np.nan], "tongue": [10.0, np.nan, np.nan],
                         "ear": [np.nan, np.nan, np.nan]})


class TestTrainedKeypoints:
    """Test the function trained_keypoints."""

    def test_trained_keypoints_union(self):
        inv = {"a": {"trainable": ["nose", "ear"]}, "b": {"trainable": ["tongue"]}}

        assert trained_keypoints({"a", "b"}, inv) == {"nose", "ear", "tongue"}

    def test_trained_keypoints_ignores_unknown_dataset(self):
        inv = {"a": {"trainable": ["nose"]}}

        assert trained_keypoints({"a", "missing"}, inv) == {"nose"}


class TestTargetScores:
    """Test the function target_scores."""

    def test_target_scores_mean_median_and_pooled(self, errors: pd.DataFrame):
        result = target_scores(errors, ["nose", "tongue"])

        assert result["nose"] == (2.0, 2.0)
        assert result["tongue"] == (10.0, 10.0)
        assert result["pooled"] == pytest.approx((14 / 3, 3.0))

    def test_target_scores_untrained_left_out_of_pooled(self, errors: pd.DataFrame):
        result = target_scores(errors, ["nose", "tongue"], trained={"nose"})

        assert result["tongue"] == "untrained"
        assert result["pooled"] == (2.0, 2.0)

    def test_target_scores_no_labelled_cell(self, errors: pd.DataFrame):
        result = target_scores(errors, ["ear", "missing"])

        assert result["ear"] is None
        assert result["missing"] is None
        assert np.isnan(result["pooled"][0])


class TestVisibleErrors:
    """Test the function visible_errors."""

    @staticmethod
    def frame(values: dict[str, list[float]]) -> pd.DataFrame:
        cols = pd.MultiIndex.from_tuples(
            [("s", k, c) for k in values for c in ("x", "y", "visible")]
        )
        rows = [sum(([v, v, 2.0] for v in row), []) for row in zip(*values.values(), strict=True)]
        return pd.DataFrame(rows, index=["a.png", "b.png"], columns=cols)

    def test_visible_errors_masks_tongue_and_lips(self):
        labels = self.frame({"nose": [0.0, 0.0], "tongue_tip": [0.0, 0.0],
                             "lowerlip": [0.0, 0.0]})
        preds = self.frame({"nose": [3.0, 4.0], "tongue_tip": [30.0, 40.0],
                            "lowerlip": [5.0, 6.0]})
        err = visible_errors(labels, preds)
        np.testing.assert_allclose(err["nose"], [3 * np.sqrt(2), 4 * np.sqrt(2)])
        assert err["tongue_tip"].isna().all()
        assert err["lowerlip"].isna().all()

    def test_visible_errors_excludes_every_tongue_and_lip_keypoint_of_the_vocabulary(self):
        path = repo_root() / "configs" / "keypoints.yaml"
        vocab = yaml.safe_load(path.read_text())["keypoints"]
        for part in ("tongue", "lip"):
            names = {k for k in vocab if part in k}
            assert names, f"no {part} keypoints found in configs/keypoints.yaml"
            assert names <= ALWAYS_EXCLUDED

    def test_visible_errors_rejects_mismatched_frames(self):
        labels = self.frame({"nose": [0.0, 0.0]})
        preds = self.frame({"nose": [0.0, 0.0]}).rename(index={"b.png": "c.png"})
        with pytest.raises(ValueError, match="differ"):
            visible_errors(labels, preds)
