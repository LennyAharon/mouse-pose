"""Test the scoring helpers of the mighty_mouse.comparison module."""

import numpy as np
import pandas as pd
import pytest

from mighty_mouse.comparison import target_scores, trained_keypoints


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
