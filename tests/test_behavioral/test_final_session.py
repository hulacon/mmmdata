"""Tests for behavioral.final_session."""

import numpy as np
import pandas as pd
import pytest

from behavioral.final_session import (
    fin_vs_tb_accuracy,
    timeline_analysis,
    timeline_by_condition,
    long_term_retention_curve,
)


class TestFinVsTbAccuracy:
    def test_basic(self, sample_fin2afc, sample_tb2afc):
        result = fin_vs_tb_accuracy(sample_fin2afc, sample_tb2afc)
        assert "source" in result.columns
        assert "accuracy" in result.columns
        # Should have both TB and FIN entries
        sources = result["source"].unique()
        assert any(s.startswith("TB-") for s in sources)
        assert any(s.startswith("FIN-") for s in sources)

    def test_with_condition_grouping(self, sample_fin2afc, sample_tb2afc):
        result = fin_vs_tb_accuracy(
            sample_fin2afc, sample_tb2afc, group_cols=["enCon"],
        )
        assert "enCon" in result.columns


class TestTimelineAnalysis:
    def test_basic(self, sample_fintimeline):
        result = timeline_analysis(sample_fintimeline)
        assert "mean_resp" in result.columns
        assert "sd_resp" in result.columns
        assert "accuracy" in result.columns
        assert result["mean_resp"].iloc[0] >= 0
        assert result["mean_resp"].iloc[0] <= 1

    def test_grouped(self, sample_fintimeline):
        result = timeline_analysis(
            sample_fintimeline, group_cols=["subject"],
        )
        assert "subject" in result.columns
        assert len(result) == 2  # 2 subjects


class TestTimelineByCondition:
    def test_basic(self, sample_fintimeline):
        result = timeline_by_condition(sample_fintimeline)
        assert "enCon" in result.columns
        assert "reCon" in result.columns
        assert "subject" in result.columns


class TestLongTermRetentionCurve:
    def test_basic(self, sample_tb2afc, sample_fin2afc):
        result = long_term_retention_curve(sample_tb2afc, sample_fin2afc)
        assert "initial_session" in result.columns
        assert "initial_accuracy" in result.columns
        assert "final_accuracy" in result.columns
        assert "retention_delta" in result.columns

    def test_retention_delta_range(self, sample_tb2afc, sample_fin2afc):
        result = long_term_retention_curve(sample_tb2afc, sample_fin2afc)
        if not result.empty:
            assert all(result["retention_delta"].between(-1, 1))
