"""Test the mighty_mouse.labels module."""

from pathlib import Path

import pandas as pd
import pytest

from mighty_mouse.labels import frame_name, read_labels_csv, session_name


class TestReadLabelsCsv:
    """Test the function read_labels_csv."""

    def test_read_labels_csv_roundtrip(self, tmp_path: Path, raw_df: pd.DataFrame):
        path = tmp_path / "CollectedData.csv"
        raw_df.to_csv(path)

        result = read_labels_csv(path)

        pd.testing.assert_frame_equal(result, raw_df, check_names=False)
        assert result.columns.nlevels == 3

    def test_read_labels_csv_accepts_str_path(self, tmp_path: Path, raw_df: pd.DataFrame):
        path = tmp_path / "CollectedData.csv"
        raw_df.to_csv(path)

        result = read_labels_csv(str(path))

        assert list(result.index) == list(raw_df.index)

    def test_read_labels_csv_missing_file(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            read_labels_csv(tmp_path / "missing.csv")


class TestSessionName:
    """Test the function session_name."""

    def test_session_name_labeled_data_path(self):
        assert session_name("labeled-data/sessA/img0.png") == "sessA"

    def test_session_name_nested_prefix(self):
        assert session_name("labeled-data/facemap/sessA/img0.png") == "sessA"


class TestFrameName:
    """Test the function frame_name."""

    def test_frame_name_labeled_data_path(self):
        assert frame_name("labeled-data/sessA/img0.png") == "img0.png"

    def test_frame_name_nested_prefix(self):
        assert frame_name("labeled-data/facemap/sessA/img0.png") == "img0.png"
