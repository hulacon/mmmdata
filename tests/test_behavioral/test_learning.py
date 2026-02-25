"""Tests for behavioral.learning."""

import numpy as np
import pandas as pd
import pytest

from behavioral.learning import (
    session_learning_curve,
    session_dprime_curve,
    compare_conditions_over_sessions,
)


class TestSessionLearningCurve:
    def test_basic(self, sample_tb2afc):
        result = session_learning_curve(sample_tb2afc, group_cols=["subject"])
        assert "session" in result.columns
        assert "session_order" in result.columns
        assert "mean" in result.columns
        assert "se" in result.columns
        assert "n" in result.columns

    def test_sessions_present(self, sample_tb2afc):
        result = session_learning_curve(sample_tb2afc, group_cols=["subject"])
        assert set(result["session"]) == {"04", "05"}

    def test_accuracy_range(self, sample_tb2afc):
        result = session_learning_curve(sample_tb2afc, group_cols=["subject"])
        assert all(result["mean"].between(0, 1))


class TestSessionDprimeCurve:
    def test_basic(self, sample_tb2afc):
        result = session_dprime_curve(sample_tb2afc, group_cols=["subject"])
        assert "dprime_2afc" in result.columns
        assert "session_order" in result.columns

    def test_dprime_finite(self, sample_tb2afc):
        result = session_dprime_curve(sample_tb2afc, group_cols=["subject"])
        assert all(np.isfinite(result["dprime_2afc"]))


class TestCompareConditionsOverSessions:
    def test_basic(self, sample_tb2afc):
        result = compare_conditions_over_sessions(sample_tb2afc)
        assert "condition" in result.columns
        assert "session_order" in result.columns
        assert "mean" in result.columns

    def test_encon_conditions(self, sample_tb2afc):
        result = compare_conditions_over_sessions(sample_tb2afc)
        # Should have enCon values 1, 2, 3
        assert set(result["condition"]) <= {1, 2, 3}
