"""Tests for behavioral.accuracy."""

import numpy as np
import pandas as pd
import pytest

from behavioral.accuracy import (
    accuracy_by_condition,
    dprime,
    criterion,
    compute_sdt_2afc,
    confidence_accuracy_curve,
)


class TestAccuracyByCondition:
    def test_overall_accuracy(self, sample_tb2afc):
        result = accuracy_by_condition(sample_tb2afc)
        assert len(result) == 1
        assert 0 <= result["accuracy"].iloc[0] <= 1
        assert result["n_trials"].iloc[0] == len(sample_tb2afc)

    def test_grouped_by_subject(self, sample_tb2afc):
        result = accuracy_by_condition(
            sample_tb2afc, group_cols=["subject"],
        )
        assert len(result) == 2  # 2 subjects
        assert set(result["subject"]) == {"03", "04"}
        assert all(result["n_trials"] > 0)

    def test_grouped_by_encon(self, sample_tb2afc):
        result = accuracy_by_condition(
            sample_tb2afc, group_cols=["subject", "enCon"],
        )
        assert "enCon" in result.columns
        assert all(result["accuracy"].between(0, 1))

    def test_se_column(self, sample_tb2afc):
        result = accuracy_by_condition(
            sample_tb2afc, group_cols=["subject"],
        )
        assert "se" in result.columns

    def test_empty_dataframe(self):
        df = pd.DataFrame({"trial_accuracy": []})
        result = accuracy_by_condition(df)
        assert len(result) == 0


class TestDprime:
    def test_perfect_discrimination(self):
        # hr=1.0, far=0.0 with correction -> high but finite d'
        d = dprime(1.0, 0.0, n_signal=100, n_noise=100)
        assert d > 3.0  # Should be very high

    def test_chance_performance(self):
        d = dprime(0.5, 0.5, n_signal=100, n_noise=100)
        assert abs(d) < 0.5  # Near zero

    def test_below_chance(self):
        d = dprime(0.2, 0.8, n_signal=100, n_noise=100)
        assert d < 0  # Negative

    def test_loglinear_correction(self):
        # Should not error with extreme rates
        d = dprime(0.0, 1.0, correction="loglinear",
                   n_signal=50, n_noise=50)
        assert np.isfinite(d)

    def test_clip_correction(self):
        d = dprime(1.0, 0.0, correction="clip",
                   n_signal=50, n_noise=50)
        assert np.isfinite(d)

    def test_no_n_provided(self):
        # Should fall back to simple clip
        d = dprime(0.5, 0.5)
        assert np.isfinite(d)


class TestCriterion:
    def test_liberal_bias(self):
        # High hit rate AND high FA rate -> liberal (negative c)
        c = criterion(0.9, 0.7, n_signal=100, n_noise=100)
        assert c < 0

    def test_conservative_bias(self):
        # Low hit rate AND low FA rate -> conservative (positive c)
        c = criterion(0.3, 0.1, n_signal=100, n_noise=100)
        assert c > 0

    def test_unbiased(self):
        c = criterion(0.5, 0.5, n_signal=100, n_noise=100)
        assert abs(c) < 0.3


class TestComputeSdt2afc:
    def test_basic(self, sample_tb2afc):
        result = compute_sdt_2afc(sample_tb2afc, group_cols=["subject"])
        assert len(result) == 2
        assert "dprime_2afc" in result.columns
        assert "accuracy" in result.columns
        assert all(result["n_trials"] > 0)

    def test_dprime_positive_for_above_chance(self):
        # All correct -> high d'
        df = pd.DataFrame({
            "trial_accuracy": [1.0] * 20,
            "subject": ["03"] * 20,
        })
        result = compute_sdt_2afc(df, group_cols=["subject"])
        assert result["dprime_2afc"].iloc[0] > 0

    def test_hit_rate_column(self, sample_tb2afc):
        result = compute_sdt_2afc(sample_tb2afc, group_cols=["subject"])
        if "hit_rate" in result.columns:
            assert all(result["hit_rate"].between(0, 1))


class TestConfidenceAccuracyCurve:
    def test_two_levels(self, sample_tb2afc):
        result = confidence_accuracy_curve(sample_tb2afc)
        assert "confidence_level" in result.columns
        # Should have 1 (maybe) and 2 (sure)
        assert set(result["confidence_level"]) <= {1, 2}

    def test_grouped(self, sample_tb2afc):
        result = confidence_accuracy_curve(
            sample_tb2afc, group_cols=["subject"],
        )
        assert "subject" in result.columns
        assert len(result) > 2  # Multiple groups
