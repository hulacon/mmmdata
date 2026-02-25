"""Smoke tests for behavioral.plotting.

Verify that each plot function creates a figure without errors.
"""

import pytest
import pandas as pd
import numpy as np

from behavioral.accuracy import accuracy_by_condition, confidence_accuracy_curve
from behavioral.learning import session_learning_curve
from behavioral.final_session import timeline_analysis

# Skip all tests if matplotlib is not available
plt = pytest.importorskip("matplotlib.pyplot")


class TestPlotAccuracyByCondition:
    def test_creates_figure(self, sample_tb2afc):
        from behavioral.plotting import plot_accuracy_by_condition
        acc = accuracy_by_condition(
            sample_tb2afc, group_cols=["subject", "enCon"],
        )
        fig = plot_accuracy_by_condition(acc, x="enCon", hue=None)
        assert fig is not None
        plt.close("all")


class TestPlotLearningCurve:
    def test_creates_figure(self, sample_tb2afc):
        from behavioral.plotting import plot_learning_curve
        lc = session_learning_curve(
            sample_tb2afc, group_cols=["subject"],
        )
        fig = plot_learning_curve(lc)
        assert fig is not None
        plt.close("all")


class TestPlotRtDistribution:
    def test_creates_figure(self, sample_tb2afc):
        from behavioral.plotting import plot_rt_distribution
        fig = plot_rt_distribution(sample_tb2afc, group_col="subject")
        assert fig is not None
        plt.close("all")


class TestPlotConfidenceAccuracy:
    def test_creates_figure(self, sample_tb2afc):
        from behavioral.plotting import plot_confidence_accuracy
        cc = confidence_accuracy_curve(
            sample_tb2afc, group_cols=["subject"],
        )
        fig = plot_confidence_accuracy(cc)
        assert fig is not None
        plt.close("all")


class TestPlotTimelineResponses:
    def test_creates_figure(self, sample_fintimeline):
        from behavioral.plotting import plot_timeline_responses
        fig = plot_timeline_responses(sample_fintimeline)
        assert fig is not None
        plt.close("all")
