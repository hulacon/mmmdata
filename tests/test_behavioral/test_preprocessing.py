"""Tests for behavioral.preprocessing."""

import numpy as np
import pandas as pd
import pytest

from behavioral.preprocessing import (
    remap_scanner_resp,
    decompose_2afc_resp,
    filter_rt,
    add_session_order,
    validate_dataframe,
)


class TestRemapScannerResp:
    def test_basic_remap(self, sample_encoding):
        result = remap_scanner_resp(sample_encoding)
        valid = result[result["trial_type"] == "image"]
        assert valid["resp"].dropna().isin([1, 2, 3]).all()

    def test_custom_output_col(self, sample_encoding):
        result = remap_scanner_resp(sample_encoding, output_col="rating")
        assert "rating" in result.columns
        # Original resp should be unchanged
        img_rows = result[result["trial_type"] == "image"]
        assert img_rows["resp"].dropna().isin([6, 7, 8]).all()

    def test_does_not_modify_original(self, sample_encoding):
        original_resp = sample_encoding["resp"].copy()
        remap_scanner_resp(sample_encoding)
        pd.testing.assert_series_equal(sample_encoding["resp"], original_resp)


class TestDecompose2afcResp:
    def test_position_derivation(self, sample_tb2afc):
        result = decompose_2afc_resp(sample_tb2afc)
        # resp 1-2 should give position 1, resp 3-4 should give position 2
        for _, row in result.iterrows():
            if row["resp"] in (1, 2):
                assert row["chose_position"] == 1
            else:
                assert row["chose_position"] == 2

    def test_confidence_derivation(self, sample_tb2afc):
        result = decompose_2afc_resp(sample_tb2afc)
        for _, row in result.iterrows():
            if row["resp"] in (1, 4):
                assert row["confidence"] == "sure"
            else:
                assert row["confidence"] == "maybe"

    def test_accuracy_consistency(self, sample_tb2afc):
        result = decompose_2afc_resp(sample_tb2afc)
        for _, row in result.iterrows():
            expected = row["chose_position"] == row["correct_resp"]
            assert row["is_correct"] == expected


class TestFilterRt:
    def test_min_rt_filter(self):
        df = pd.DataFrame({"resp_RT": [0.1, 0.3, 0.5, 1.0, 2.0]})
        result = filter_rt(df, min_rt=0.2, max_sd=None)
        assert len(result) == 4
        assert result["resp_RT"].min() >= 0.2

    def test_max_rt_filter(self):
        df = pd.DataFrame({"resp_RT": [0.5, 1.0, 2.0, 10.0, 50.0]})
        result = filter_rt(df, min_rt=0.0, max_rt=5.0, max_sd=None)
        assert len(result) == 3

    def test_sd_filter(self):
        # Create data with one outlier
        rts = [1.0, 1.1, 0.9, 1.0, 1.1, 0.9, 1.0, 10.0]
        df = pd.DataFrame({"resp_RT": rts})
        result = filter_rt(df, min_rt=0.0, max_sd=2.0)
        assert len(result) < len(df)
        assert 10.0 not in result["resp_RT"].values

    def test_returns_copy(self, sample_tb2afc):
        n_before = len(sample_tb2afc)
        filter_rt(sample_tb2afc)
        assert len(sample_tb2afc) == n_before


class TestAddSessionOrder:
    def test_mapping(self):
        df = pd.DataFrame({"session": ["04", "05", "18"]})
        result = add_session_order(df)
        assert list(result["session_order"]) == [0, 1, 14]

    def test_unknown_session(self):
        df = pd.DataFrame({"session": ["30"]})
        result = add_session_order(df)
        assert pd.isna(result["session_order"].iloc[0])


class TestValidateDataframe:
    def test_valid_tb2afc(self, sample_tb2afc):
        warnings = validate_dataframe(sample_tb2afc, "tb2afc")
        assert len(warnings) == 0

    def test_missing_columns(self):
        df = pd.DataFrame({"subject": ["03"], "trial_accuracy": [1.0]})
        warnings = validate_dataframe(df, "tb2afc")
        assert any("Missing columns" in w for w in warnings)

    def test_invalid_encon(self, sample_tb2afc):
        df = sample_tb2afc.copy()
        df.loc[0, "enCon"] = 99
        warnings = validate_dataframe(df, "tb2afc")
        assert any("enCon" in w for w in warnings)

    def test_unknown_task(self):
        warnings = validate_dataframe(pd.DataFrame(), "unknown_task")
        assert any("Unknown task" in w for w in warnings)
