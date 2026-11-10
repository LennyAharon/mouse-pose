"""Test the mighty_mouse.evaluation.comparison module."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from mighty_mouse.evaluation.comparison import (
    ALWAYS_EXCLUDED,
    comparison_summary,
    offset_corrected_errors,
    target_scores,
    trained_keypoints,
    visible_errors,
    visible_residuals,
)


def frame(values: dict[str, list[float]], visible: float = 2.0) -> pd.DataFrame:
    """Two frames; each keypoint at (v, v) with the given visible flag."""
    cols = pd.MultiIndex.from_tuples([("s", k, c) for k in values for c in ("x", "y", "visible")])
    rows = [sum(([v, v, visible] for v in row), []) for row in zip(*values.values(), strict=True)]
    return pd.DataFrame(rows, index=["a.png", "b.png"], columns=cols)


class TestVisibleErrors:
    """Test the function visible_errors."""

    def test_visible_errors_masks_tongue_and_lips(self):
        labels = frame({"nose": [0.0, 0.0], "tongue_tip": [0.0, 0.0], "lowerlip": [0.0, 0.0]})
        preds  = frame({"nose": [3.0, 4.0], "tongue_tip": [30.0, 40.0], "lowerlip": [5.0, 6.0]})
        err = visible_errors(labels, preds)
        np.testing.assert_allclose(err["nose"], [3 * np.sqrt(2), 4 * np.sqrt(2)])
        assert err["tongue_tip"].isna().all()
        assert err["lowerlip"].isna().all()

    def test_visible_errors_custom_exclude(self):
        labels = frame({"nose": [0.0, 0.0], "tongue_tip": [0.0, 0.0]})
        preds  = frame({"nose": [1.0, 1.0], "tongue_tip": [2.0, 2.0]})
        err = visible_errors(labels, preds, exclude=frozenset({"nose"}))
        assert err["nose"].isna().all()
        np.testing.assert_allclose(err["tongue_tip"], [2 * np.sqrt(2)] * 2)

    def test_visible_errors_skips_occluded(self):
        err = visible_errors(frame({"nose": [0.0, 0.0]}, visible=1.0), frame({"nose": [3.0, 4.0]}))
        assert err["nose"].isna().all()

    def test_visible_errors_excludes_every_tongue_and_lip_keypoint_of_the_vocabulary(self):
        path  = Path(__file__).parents[2] / "configs" / "keypoints.yaml"
        vocab = yaml.safe_load(path.read_text())
        for part in ("tongue", "lip"):
            names = {k for k in vocab["keypoints"] if part in k}
            assert names, f"no {part} keypoints found in configs/keypoints.yaml"
            assert names <= ALWAYS_EXCLUDED

    def test_visible_errors_rejects_mismatched_frames(self):
        labels = frame({"nose": [0.0, 0.0]})
        preds  = frame({"nose": [0.0, 0.0]}).rename(index={"b.png": "c.png"})
        with pytest.raises(ValueError, match="differ"):
            visible_errors(labels, preds)


class TestComparisonSummary:
    """Test the function comparison_summary."""

    def test_comparison_summary_pooled_and_normalized(self):
        errors = pd.DataFrame({"nose": [2.0, 4.0], "ear": [6.0, np.nan]})
        s = comparison_summary(errors, np.array([10.0, 20.0]))
        assert s["points"] == 3
        assert s["mean_px"] == pytest.approx(4.0)
        assert s["median_px"] == pytest.approx(4.0)
        assert s["normalized_mean"] == pytest.approx((0.2 + 0.2 + 0.6) / 3)
        assert s["equal_keypoint_normalized_mean"] == pytest.approx((0.2 + 0.6) / 2)

    def test_comparison_summary_nothing_scored(self):
        s = comparison_summary(pd.DataFrame({"nose": [np.nan]}), np.array([10.0]))
        assert s["points"] == 0
        assert s["mean_px"] is None

    def test_comparison_summary_rejects_bad_diagonals(self):
        with pytest.raises(ValueError, match="diagonal"):
            comparison_summary(pd.DataFrame({"nose": [1.0, 2.0]}), np.array([10.0]))


class TestTrainedKeypoints:
    """Test the function trained_keypoints."""

    def test_trained_keypoints_union(self):
        inv = {"a": {"trainable": ["nose", "ear"]}, "b": {"trainable": ["tongue"]}}
        assert trained_keypoints({"a", "b"}, inv) == {"nose", "ear", "tongue"}

    def test_trained_keypoints_ignores_unknown_dataset(self):
        assert trained_keypoints({"a", "missing"}, {"a": {"trainable": ["nose"]}}) == {"nose"}


@pytest.fixture
def errors() -> pd.DataFrame:
    """Errors as visible_errors returns them: NaN where the cell is not visible == 2."""
    return pd.DataFrame({"nose": [1.0, 3.0, np.nan], "tongue": [10.0, np.nan, np.nan],
                         "ear": [np.nan, np.nan, np.nan]})


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


def shifted(n: int = 40, shift: tuple[float, float] = (5.0, -3.0), noise: float = 0.0, seed: int = 0):
    """n frames of one keypoint at (10, 10); predictions shifted by `shift` plus optional noise."""
    rng = np.random.default_rng(seed)
    idx = [f"labeled-data/ds/s{i % 2}/img{i}.png" for i in range(n)]
    cols = pd.MultiIndex.from_tuples([("s", "nose", c) for c in ("x", "y", "visible")])
    labels = pd.DataFrame([[10.0, 10.0, 2.0]] * n, index=idx, columns=cols)
    e = rng.normal(0, noise, (n, 2)) if noise else np.zeros((n, 2))
    pcols = pd.MultiIndex.from_tuples([("s", "nose", c) for c in ("x", "y", "likelihood")])
    preds = pd.DataFrame([[10 + shift[0] + a, 10 + shift[1] + b, 1.0] for a, b in e], index=idx,
                         columns=pcols)
    return labels, preds


class TestOffsetCorrectedErrors:
    """Test the function offset_corrected_errors."""

    def test_offset_corrected_errors_removes_constant_shift(self):
        labels, preds = shifted()
        raw = visible_errors(labels, preds)
        out = offset_corrected_errors(labels, preds)
        np.testing.assert_allclose(raw["nose"], np.hypot(5, 3))
        np.testing.assert_allclose(out["nose"], 0.0, atol=1e-9)

    def test_offset_corrected_errors_keeps_noise(self):
        labels, preds = shifted(n=400, noise=2.0)
        out = offset_corrected_errors(labels, preds)
        assert 1.0 < out["nose"].median() < 4.0      # the random part stays

    def test_offset_corrected_errors_too_few_frames_left_raw(self):
        labels, preds = shifted(n=6)
        out = offset_corrected_errors(labels, preds, min_frames=5)
        np.testing.assert_allclose(out["nose"], np.hypot(5, 3))

    def test_offset_corrected_errors_separate_groups(self):
        labels, preds = shifted(n=40)
        preds.iloc[::2, 0] += 10                         # session s0 shifted 10 px more in x
        groups = pd.Series([i.split("/")[2] for i in labels.index], index=labels.index)
        out = offset_corrected_errors(labels, preds, groups=groups)
        np.testing.assert_allclose(out["nose"], 0.0, atol=1e-9)


class TestVisibleResiduals:
    """Test the function visible_residuals."""

    def test_visible_residuals_signed(self):
        labels, preds = shifted(n=4)
        dx, dy = visible_residuals(labels, preds)
        np.testing.assert_allclose(dx["nose"], 5.0)
        np.testing.assert_allclose(dy["nose"], -3.0)
