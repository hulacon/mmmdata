"""Tests for behavioral.io.

Unit tests use temporary directories with mock TSV files.
Integration tests (marked) use real BIDS data paths.
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from behavioral.io import (
    _parse_sub_ses,
    _normalize_columns,
    load_tsv,
)


class TestParseSubSes:
    def test_standard_bids_filename(self):
        p = Path("sub-03_ses-13_task-TB2AFC_run-01_beh.tsv")
        result = _parse_sub_ses(p)
        assert result == {"subject": "03", "session": "13"}

    def test_no_match(self):
        p = Path("random_file.tsv")
        result = _parse_sub_ses(p)
        assert result == {}


class TestNormalizeColumns:
    def test_drops_redundant_raw_columns(self):
        # When both raw and filename-based columns exist, drop raw
        df = pd.DataFrame({
            "subject_id": [3, 4],
            "subject": ["03", "04"],
            "session_num": [10, 11],
            "session": ["13", "14"],
            "run_num": [1, 2],
        })
        result = _normalize_columns(df)
        assert "subject_id" not in result.columns
        assert "session_num" not in result.columns
        assert "run" in result.columns  # run_num renamed
        assert result["subject"].iloc[0] == "03"
        assert result["session"].iloc[0] == "13"

    def test_already_normalized(self):
        df = pd.DataFrame({
            "subject": ["03"],
            "session": ["04"],
        })
        result = _normalize_columns(df)
        assert result["subject"].iloc[0] == "03"

    def test_run_num_rename(self):
        df = pd.DataFrame({"run_num": [1, 2]})
        result = _normalize_columns(df)
        assert "run" in result.columns
        assert result["run"].iloc[0] == "01"


class TestLoadTsv:
    def test_loads_file(self, tmp_path):
        tsv = tmp_path / "sub-03_ses-04_task-test_beh.tsv"
        tsv.write_text("col1\tcol2\n1\tn/a\n3\t4\n")
        df = load_tsv(tsv)
        assert len(df) == 2
        assert pd.isna(df["col2"].iloc[0])
        assert df["col2"].iloc[1] == 4

    def test_injects_subject_session(self, tmp_path):
        tsv = tmp_path / "sub-03_ses-04_task-test_beh.tsv"
        tsv.write_text("value\n1\n")
        df = load_tsv(tsv)
        assert "subject" in df.columns
        assert "session" in df.columns
        assert df["subject"].iloc[0] == "03"
        assert df["session"].iloc[0] == "04"

    def test_filename_overrides_column(self, tmp_path):
        tsv = tmp_path / "sub-03_ses-04_task-test_beh.tsv"
        tsv.write_text("subject\tsession\n99\t99\n")
        df = load_tsv(tsv)
        # Filename-based values should override column values
        assert df["subject"].iloc[0] == "03"
        assert df["session"].iloc[0] == "04"


# ---------------------------------------------------------------------------
# Integration tests (require real data on HPC)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestFindFilesIntegration:
    """These tests only run on the HPC with real data."""

    @pytest.fixture(autouse=True)
    def _check_bids_root(self):
        bids_root = Path("/gpfs/projects/hulacon/shared/mmmdata")
        if not bids_root.exists():
            pytest.skip("BIDS root not available")

    def test_find_tb2afc_count(self):
        from behavioral.io import find_tb2afc_files
        files = find_tb2afc_files()
        assert len(files) == 45  # 3 subjects x 15 sessions

    def test_find_encoding_count(self):
        from behavioral.io import find_encoding_files
        files = find_encoding_files()
        assert len(files) > 100  # ~126 files

    def test_find_retrieval_count(self):
        from behavioral.io import find_retrieval_files
        files = find_retrieval_files()
        assert len(files) > 150  # ~168 files

    def test_find_fin2afc_count(self):
        from behavioral.io import find_fin2afc_files
        files = find_fin2afc_files()
        assert len(files) == 3

    def test_find_fintimeline_count(self):
        from behavioral.io import find_fintimeline_files
        files = find_fintimeline_files()
        assert len(files) == 3

    def test_load_tb2afc(self):
        from behavioral.io import load_tb2afc
        df = load_tb2afc()
        assert len(df) > 1000
        assert "subject" in df.columns
        assert "session" in df.columns
        assert "trial_accuracy" in df.columns
