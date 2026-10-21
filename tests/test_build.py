"""Test the mighty_mouse.build module."""

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from mighty_mouse.build import (
    build_merged,
    dataset_rng,
    mask_keypoints,
    subsample,
    subsample_nested,
)


@pytest.fixture
def split_df(make_processed_df: Callable) -> pd.DataFrame:
    """A 10-row converted split."""
    return make_processed_df({"nose": [2.0] * 10, "ear": [0.0] * 10})


class TestDatasetRng:
    """Test the function dataset_rng."""

    def test_dataset_rng_reproducible(self):
        a = dataset_rng(42, "facemap").integers(0, 1_000_000, size=5)
        b = dataset_rng(42, "facemap").integers(0, 1_000_000, size=5)

        np.testing.assert_array_equal(a, b)

    def test_dataset_rng_depends_on_name(self):
        a = dataset_rng(42, "facemap").integers(0, 1_000_000, size=5)
        b = dataset_rng(42, "ibl").integers(0, 1_000_000, size=5)

        assert not np.array_equal(a, b)

    def test_dataset_rng_depends_on_seed(self):
        a = dataset_rng(42, "facemap").integers(0, 1_000_000, size=5)
        b = dataset_rng(7, "facemap").integers(0, 1_000_000, size=5)

        assert not np.array_equal(a, b)

    def test_dataset_rng_known_value(self):
        # guards the seeding scheme itself: changing it silently changes which frames
        # every existing tag was built from
        result = dataset_rng(42, "facemap").choice(100, size=3, replace=False)

        assert result.tolist() == [66, 52, 40]


class TestSubsample:
    """Test the function subsample."""

    def test_subsample_size(self, split_df: pd.DataFrame):
        result = subsample(split_df, 4, dataset_rng(42, "toy"))

        assert len(result) == 4
        assert result.index.is_unique

    def test_subsample_keeps_original_order(self, split_df: pd.DataFrame):
        result = subsample(split_df, 6, dataset_rng(42, "toy"))

        positions = [split_df.index.get_loc(i) for i in result.index]
        assert positions == sorted(positions)

    def test_subsample_reproducible(self, split_df: pd.DataFrame):
        a = subsample(split_df, 5, dataset_rng(42, "toy"))
        b = subsample(split_df, 5, dataset_rng(42, "toy"))

        pd.testing.assert_frame_equal(a, b)

    def test_subsample_capped_at_available(self, split_df: pd.DataFrame):
        result = subsample(split_df, 600, dataset_rng(42, "toy"))

        pd.testing.assert_frame_equal(result, split_df)

    def test_subsample_negative_takes_all(self, split_df: pd.DataFrame):
        result = subsample(split_df, -1, dataset_rng(42, "toy"))

        pd.testing.assert_frame_equal(result, split_df)

    def test_subsample_zero(self, split_df: pd.DataFrame):
        result = subsample(split_df, 0, dataset_rng(42, "toy"))

        assert result.empty
        assert list(result.columns) == list(split_df.columns)


class TestSubsampleNested:
    """Test the function subsample_nested."""

    def test_subsample_nested_size(self, split_df: pd.DataFrame):
        result = subsample_nested(split_df, 4, dataset_rng(42, "toy"))

        assert len(result) == 4
        assert result.index.is_unique

    def test_subsample_nested_keeps_original_order(self, split_df: pd.DataFrame):
        result = subsample_nested(split_df, 6, dataset_rng(42, "toy"))

        positions = [split_df.index.get_loc(i) for i in result.index]
        assert positions == sorted(positions)

    def test_subsample_nested_smaller_cap_is_subset(self, split_df: pd.DataFrame):
        small = subsample_nested(split_df, 3, dataset_rng(42, "toy"))
        large = subsample_nested(split_df, 7, dataset_rng(42, "toy"))

        assert set(small.index) <= set(large.index)

    def test_subsample_nested_capped_at_available(self, split_df: pd.DataFrame):
        result = subsample_nested(split_df, 600, dataset_rng(42, "toy"))

        pd.testing.assert_frame_equal(result, split_df)


class TestMaskKeypoints:
    """Test the function mask_keypoints."""

    def test_mask_keypoints_hides_matching(self, make_processed_df: Callable):
        df = make_processed_df({"ear_l": [2.0, 2.0], "ear_r": [2.0, 2.0], "nose": [2.0, 2.0]})

        result, kps = mask_keypoints(df, "^ear_")

        assert kps == ["ear_l", "ear_r"]
        for kp in kps:
            assert (result.xs((kp, "visible"), axis=1, level=[1, 2]) == 0).all().all()
            assert result.xs((kp, "x"), axis=1, level=[1, 2]).isna().all().all()
        pd.testing.assert_frame_equal(result.xs("nose", axis=1, level=1),
                                      df.xs("nose", axis=1, level=1))

    def test_mask_keypoints_leaves_input_unchanged(self, make_processed_df: Callable):
        df       = make_processed_df({"ear_l": [2.0, 2.0]})
        original = df.copy()

        mask_keypoints(df, "^ear_")

        pd.testing.assert_frame_equal(df, original)

    def test_mask_keypoints_no_match(self, make_processed_df: Callable):
        df = make_processed_df({"nose": [2.0, 2.0]})

        result, kps = mask_keypoints(df, "^ear_")

        assert kps == []
        pd.testing.assert_frame_equal(result, df)


class TestBuildMerged:
    """Test the function build_merged."""

    def test_build_merged_concatenates_in_order(self, make_processed_df: Callable):
        a = make_processed_df({"nose": [2.0, 2.0]})
        b = make_processed_df({"nose": [1.0]})
        b.index = pd.Index(["labeled-data/other/sess/img0.png"])

        result = build_merged([a, b])

        assert list(result.index) == [*a.index, *b.index]
        assert list(result.columns) == list(a.columns)

    def test_build_merged_empty_list(self):
        with pytest.raises(ValueError):
            build_merged([])
