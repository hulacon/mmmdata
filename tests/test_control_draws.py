"""control_draws.py: the matched-subspace control pool (pure parts; the build needs psytwill)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parents[1] / "scripts" / "braintwill_encoding"


@pytest.fixture(scope="module")
def cd():
    spec = importlib.util.spec_from_file_location("control_draws", HERE / "control_draws.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_nested_gate_centres_present_rows_and_adds_a_presence_column(cd):
    Z = np.array([[1.0, 2.0], [np.nan, 5.0], [3.0, 4.0]])
    g, present = cd.nested_gate(Z)
    np.testing.assert_allclose(g, [[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]])
    assert present.tolist() == [1.0, 0.0, 1.0]
    full = np.arange(6.0).reshape(3, 2)
    g, present = cd.nested_gate(full)
    assert present is None and g is full
    with pytest.raises(RuntimeError, match="absent from every pool row"):
        cd.nested_gate(np.full((2, 2), np.nan))


def test_set_columns_follow_set_order_and_refuse_a_stranger(cd):
    bands = {"a": slice(0, 2), "b": slice(2, 5), "c": slice(5, 6)}
    assert cd.set_columns(bands, ("c", "a")).tolist() == [5, 0, 1]
    with pytest.raises(ValueError, match="not pool sources"):
        cd.set_columns(bands, ("a", "z"))


@pytest.mark.parametrize("src, parsed", [
    ("word-on", ("word-on", None, None, None)),
    ("cuts", ("cuts", None, None, None)),
    ("store:movies_frames:motion", ("store", "movies_frames", "motion", None)),
    ("store:movies_audio_frames:speech_rate.speech_rate_words",
     ("store", "movies_audio_frames", "speech_rate", "speech_rate_words")),
])
def test_parse_extra_reads_every_kind(cd, src, parsed):
    assert cd.parse_extra(src) == parsed


def test_parse_extra_and_check_pool_refuse_what_they_cannot_read(cd):
    with pytest.raises(ValueError, match="expected word-on"):
        cd.parse_extra("model:faces")
    with pytest.raises(ValueError, match="unknown pool"):
        cd.check_pool("nope")


def test_every_declared_set_names_only_pool_sources(cd):
    for pool, decl in cd.POOLS.items():
        extra = set(decl["extra"])
        for name, members in decl["sets"].items():
            assert len(set(members)) == len(members), (pool, name)
            assert not (set(members) & set(decl["skip"])), (pool, name)
            assert all(m in extra or ":" not in m and m != "word-on" for m in members), (pool, name)


def test_pool_films_must_be_one_set_for_every_subject(cd, monkeypatch):
    monkeypatch.setattr(cd.fm, "SUBJECTS", ("01", "02"))
    monkeypatch.setattr(cd.fm, "N_POOL", 2)
    w = pd.DataFrame({"sub": ["01", "01", "02", "02"], "role": "alignment", "ses": "05",
                      "stimulus_id": ["film-a", "film-b", "film-a", "film-b"]})
    assert cd.pool_films(w) == ["film-a", "film-b"]
    w.loc[3, "stimulus_id"] = "film-c"
    with pytest.raises(RuntimeError, match="differ across subjects"):
        cd.pool_films(w)
