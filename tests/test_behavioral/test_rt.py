"""Tests for behavioral.rt."""

import numpy as np
import pandas as pd
import pytest

from behavioral.rt import rt_summary, rt_by_accuracy, rt_sequential


class TestRtSummary:
    def test_basic(self, sample_tb2afc):
        result = rt_summary(sample_tb2afc)
        assert len(result) == 1
        assert "mean_rt" in result.columns
        assert "median_rt" in result.columns
        assert "sd_rt" in result.columns
        assert "n_trials" in result.columns

    def test_grouped(self, sample_tb2afc):
        result = rt_summary(sample_tb2afc, group_cols=["subject"])
        assert len(result) == 2
        assert all(result["mean_rt"] > 0)

    def test_skewness(self, sample_tb2afc):
        result = rt_summary(sample_tb2afc)
        assert "skewness" in result.columns
        assert np.isfinite(result["skewness"].iloc[0])


class TestRtByAccuracy:
    def test_correct_vs_incorrect(self, sample_tb2afc):
        result = rt_by_accuracy(sample_tb2afc)
        assert "accurate" in result.columns
        assert len(result) <= 2  # True and/or False

    def test_grouped(self, sample_tb2afc):
        result = rt_by_accuracy(sample_tb2afc, group_cols=["subject"])
        assert "subject" in result.columns


class TestRtSequential:
    def test_rolling_mean(self, sample_tb2afc):
        result = rt_sequential(
            sample_tb2afc, group_cols=["subject", "session"],
        )
        assert "rt_rolling_mean" in result.columns
        assert not result["rt_rolling_mean"].isna().all()

    def test_window_size(self):
        df = pd.DataFrame({
            "resp_RT": [1.0, 2.0, 3.0, 4.0, 5.0],
            "trial_id": [1, 2, 3, 4, 5],
        })
        result = rt_sequential(df, window=3)
        # First value should just be itself
        assert result["rt_rolling_mean"].iloc[0] == 1.0
        # Third value should be mean of first 3
        assert abs(result["rt_rolling_mean"].iloc[2] - 2.0) < 0.01
