"""Tests for behavioral.encoding."""

import numpy as np
import pandas as pd
import pytest

from behavioral.encoding import (
    encoding_rating_distribution,
    subsequent_memory_effect,
    retrieval_vividness_by_condition,
)
from behavioral.preprocessing import remap_scanner_resp


class TestEncodingRatingDistribution:
    def test_basic(self, sample_encoding):
        enc = remap_scanner_resp(sample_encoding)
        enc = enc[enc["trial_type"] == "image"]
        result = encoding_rating_distribution(enc)
        assert "rating" in result.columns
        assert "count" in result.columns
        assert "proportion" in result.columns
        # Proportions should sum to ~1
        assert abs(result["proportion"].sum() - 1.0) < 0.01

    def test_grouped(self, sample_encoding):
        enc = remap_scanner_resp(sample_encoding)
        enc = enc[enc["trial_type"] == "image"]
        result = encoding_rating_distribution(enc, group_cols=["subject"])
        assert "subject" in result.columns
        # Per-subject proportions should each sum to ~1
        for sub, grp in result.groupby("subject"):
            assert abs(grp["proportion"].sum() - 1.0) < 0.01


class TestSubsequentMemoryEffect:
    def test_basic(self, sample_encoding, sample_tb2afc):
        enc = remap_scanner_resp(sample_encoding)
        enc = enc[enc["trial_type"] == "image"]
        result = subsequent_memory_effect(enc, sample_tb2afc)
        assert "encoding_rating" in result.columns
        assert "accuracy" in result.columns
        assert "n_trials" in result.columns

    def test_grouped(self, sample_encoding, sample_tb2afc):
        enc = remap_scanner_resp(sample_encoding)
        enc = enc[enc["trial_type"] == "image"]
        result = subsequent_memory_effect(
            enc, sample_tb2afc, group_cols=["subject"],
        )
        if not result.empty:
            assert "subject" in result.columns

    def test_no_overlap_returns_empty(self):
        enc = pd.DataFrame({
            "subject": ["03"], "pairId": [999],
            "resp": [1],
        })
        rec = pd.DataFrame({
            "subject": ["03"], "pairId": [1],
            "trial_accuracy": [1.0],
        })
        result = subsequent_memory_effect(enc, rec)
        assert len(result) == 0


class TestRetrievalVividnessByCondition:
    def test_basic(self, sample_retrieval):
        ret = remap_scanner_resp(sample_retrieval)
        result = retrieval_vividness_by_condition(ret)
        assert "mean_vividness" in result.columns
        assert result["mean_vividness"].iloc[0] >= 1
        assert result["mean_vividness"].iloc[0] <= 3

    def test_grouped(self, sample_retrieval):
        ret = remap_scanner_resp(sample_retrieval)
        result = retrieval_vividness_by_condition(
            ret, group_cols=["subject", "enCon"],
        )
        assert "subject" in result.columns
        assert "enCon" in result.columns
