"""Tests for DICOM directory inspection."""

import pytest

from src.python.dcm2bids_config.dicom_inspect import inspect_fieldmaps


@pytest.fixture
def dicom_dir(tmp_path):
    """Create a fake DICOM session directory."""
    return tmp_path


def _make_series_dirs(base, names):
    """Create Series_##_name directories."""
    for name in names:
        (base / name).mkdir(parents=True)


class TestSeriesDescriptionStrategy:
    def test_encoding_retrieval_suffixes(self, dicom_dir):
        _make_series_dirs(dicom_dir, [
            "Series_5_se_epi_ap_encoding",
            "Series_6_se_epi_pa_encoding",
            "Series_20_se_epi_ap_retrieval",
            "Series_21_se_epi_pa_retrieval",
            "Series_10_some_bold_run1",
        ])
        result = inspect_fieldmaps(dicom_dir)
        assert result.strategy == "series_description"
        assert result.groups["encoding"] == {"ap": 5, "pa": 6}
        assert result.groups["retrieval"] == {"ap": 20, "pa": 21}
        assert result.warnings == []


class TestSeriesNumberStrategy:
    def test_standard_two_pairs(self, dicom_dir):
        _make_series_dirs(dicom_dir, [
            "Series_5_se_epi_ap",
            "Series_6_se_epi_pa",
            "Series_20_se_epi_ap",
            "Series_21_se_epi_pa",
        ])
        result = inspect_fieldmaps(dicom_dir)
        assert result.strategy == "series_number"
        assert result.groups["encoding"] == {"ap": 5, "pa": 6}
        assert result.groups["retrieval"] == {"ap": 20, "pa": 21}

    def test_single_pair(self, dicom_dir):
        _make_series_dirs(dicom_dir, [
            "Series_5_se_epi_ap",
            "Series_6_se_epi_pa",
        ])
        result = inspect_fieldmaps(dicom_dir)
        assert result.strategy == "series_number"
        assert "encoding" in result.groups
        assert "retrieval" not in result.groups

    def test_three_pairs_warns(self, dicom_dir):
        _make_series_dirs(dicom_dir, [
            "Series_5_se_epi_ap",
            "Series_6_se_epi_pa",
            "Series_20_se_epi_ap",
            "Series_21_se_epi_pa",
            "Series_35_se_epi_ap",
            "Series_36_se_epi_pa",
        ])
        result = inspect_fieldmaps(dicom_dir)
        assert len(result.warnings) > 0
        assert "3 AP" in result.warnings[0]
        # Still assigns first two pairs as defaults
        assert "encoding" in result.groups
        assert "retrieval" in result.groups


class TestNoFieldmaps:
    def test_empty_directory(self, dicom_dir):
        result = inspect_fieldmaps(dicom_dir)
        assert result.strategy == "none"
        assert result.groups == {}

    def test_no_fmap_series(self, dicom_dir):
        _make_series_dirs(dicom_dir, [
            "Series_10_some_bold_run1",
            "Series_12_another_bold",
        ])
        result = inspect_fieldmaps(dicom_dir)
        assert result.strategy == "none"

    def test_missing_directory(self, tmp_path):
        result = inspect_fieldmaps(tmp_path / "nonexistent")
        assert result.strategy == "none"
        assert len(result.warnings) > 0
