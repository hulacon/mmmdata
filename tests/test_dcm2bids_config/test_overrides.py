"""Tests for the overrides module."""

import pytest

from src.python.dcm2bids_config.overrides import OverrideResult, apply_overrides
from src.python.dcm2bids_config.session_defs import SESSION_TYPES, SessionDef, TaskDef


@pytest.fixture()
def tb_middle():
    return SESSION_TYPES["tb_middle"]


class TestOverrideResultReturn:
    """apply_overrides should return OverrideResult."""

    def test_returns_override_result(self, tb_middle):
        result = apply_overrides("ses-06", tb_middle, {})
        assert isinstance(result, OverrideResult)

    def test_no_override_returns_unchanged_session_def(self, tb_middle):
        result = apply_overrides("ses-06", tb_middle, {})
        assert result.session_def is tb_middle
        assert result.fmap_info is None
        assert result.run_protocols is None
        assert result.run_series is None

    def test_fmap_series_still_works(self, tb_middle):
        overrides = {
            "ses-10": {
                "fmap_series": {
                    "encoding": {"ap": 5, "pa": 7},
                    "retrieval": {"ap": 21, "pa": 23},
                },
            },
        }
        result = apply_overrides("ses-10", tb_middle, overrides)
        assert isinstance(result, OverrideResult)
        assert result.fmap_info == {
            "encoding": {"ap": 5, "pa": 7},
            "retrieval": {"ap": 21, "pa": 23},
        }


class TestRunProtocolsParsing:
    """run_protocols TOML keys are strings; they must be converted to int."""

    def test_run_protocols_parsed_with_int_keys(self, tb_middle):
        overrides = {
            "ses-20": {
                "run_protocols": {
                    "FINretrieval": {
                        "1": "free_recall_retrieval_run1_attempt2",
                    },
                },
            },
        }
        result = apply_overrides("ses-20", tb_middle, overrides)
        assert result.run_protocols is not None
        assert result.run_protocols["FINretrieval"][1] == "free_recall_retrieval_run1_attempt2"

    def test_run_protocols_none_when_absent(self, tb_middle):
        overrides = {"ses-06": {"note": "nothing special"}}
        result = apply_overrides("ses-06", tb_middle, overrides)
        assert result.run_protocols is None


class TestRunSeriesParsing:
    """run_series TOML keys are strings; they must be converted to int."""

    def test_run_series_parsed_with_int_keys(self, tb_middle):
        overrides = {
            "ses-30": {
                "run_series": {
                    "FINretrieval": {
                        "2": {"bold": 45, "sbref": 44},
                    },
                },
            },
        }
        result = apply_overrides("ses-30", tb_middle, overrides)
        assert result.run_series is not None
        assert result.run_series["FINretrieval"][2] == {"bold": 45, "sbref": 44}

    def test_run_series_none_when_absent(self, tb_middle):
        overrides = {"ses-06": {"note": "nothing special"}}
        result = apply_overrides("ses-06", tb_middle, overrides)
        assert result.run_series is None

    def test_run_series_and_fmap_series_together(self, tb_middle):
        overrides = {
            "ses-30": {
                "fmap_series": {
                    "encoding": {"ap": 10, "pa": 11},
                },
                "run_series": {
                    "FINretrieval": {
                        "2": {"bold": 45, "sbref": 44},
                    },
                },
            },
        }
        result = apply_overrides("ses-30", tb_middle, overrides)
        assert result.fmap_info == {"encoding": {"ap": 10, "pa": 11}}
        assert result.run_series["FINretrieval"][2] == {"bold": 45, "sbref": 44}
