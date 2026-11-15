"""film_boundaries: segment onsets, accepted cuts and scored spans for bench boundary."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

HERE = Path(__file__).resolve().parents[1] / "scripts" / "benchmarks"


@pytest.fixture(scope="module")
def fb():
    spec = importlib.util.spec_from_file_location("film_boundaries", HERE / "film_boundaries.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _inputs(two_annotators=False):
    seg = pd.DataFrame({"stimulus_id": ["film-a"] * 3 + ["film-b"] * 2, "level": ["B", "C", "C", "C", "C"],
                        "onset": [0.0, 0.0, 5.0, 0.0, 8.0],
                        "annotator": ["x", "x", "y" if two_annotators else "x", "z", "z"]})
    cuts = pd.DataFrame({"corpus": ["mmmdata", "mmmdata", "other"], "id": ["film-a", "film-b", "film-a"],
                         "time": [3.0, 4.0, 1.0], "classifier_accept": [True, False, True]})
    reg = pd.DataFrame({"stimulus_id": ["film-a", "film-b"], "duration_s": [100, 50]})
    return seg, cuts, reg


def test_build_keeps_onsets_after_zero_accepted_film_cuts_and_caps(fb):
    b, c, d = fb.build(*_inputs(), {"film-b": 30.0})
    assert b[["stimulus_id", "onset"]].values.tolist() == [["film-a", 5.0], ["film-b", 8.0]]
    assert c.values.tolist() == [["film-a", 3.0]]
    assert dict(zip(d["stimulus_id"], d["duration"])) == {"film-a": 100.0, "film-b": 30.0}


def test_build_refuses_two_annotators_and_bad_caps(fb):
    with pytest.raises(SystemExit, match="more than one annotator"):
        fb.build(*_inputs(two_annotators=True), {})
    with pytest.raises(SystemExit, match="longer than the film"):
        fb.build(*_inputs(), {"film-b": 99.0})
    with pytest.raises(SystemExit, match="not an annotated film"):
        fb.build(*_inputs(), {"film-z": 1.0})
