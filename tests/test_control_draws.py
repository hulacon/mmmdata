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


def test_project_draws_is_every_basis_in_float64_stored_float32(cd):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((7, 5)).astype(np.float32)
    b = rng.standard_normal((3, 5, 2)).astype(np.float32)
    d = cd.project_draws(x, b)
    assert d.shape == (3, 7, 2) and d.dtype == np.float32
    for i in range(3):
        np.testing.assert_allclose(d[i], x.astype(np.float64) @ b[i].astype(np.float64), rtol=1e-6)
    with pytest.raises(ValueError, match="do not compose"):
        cd.project_draws(x, b[:, :4])


def test_write_draws_writes_each_set_per_film_from_the_written_pool(cd, tmp_path):
    rng = np.random.default_rng(1)
    pool = tmp_path / "p"
    (pool / "pool").mkdir(parents=True)
    xs = {}
    for s, n in (("film-a", 6), ("film-b", 4)):
        xs[s] = rng.standard_normal((n, 5)).astype(np.float32)
        np.savez(pool / "pool" / f"{s}.npz", stimulus_id=s, times=np.arange(n) * 0.5, x=xs[s])
    bases = {"s": rng.standard_normal((3, 5, 2)).astype(np.float32),
             "t": rng.standard_normal((2, 5, 1)).astype(np.float32)}
    for name, b in bases.items():
        np.savez(pool / f"{name}.npz", bases=b)
    got = cd.write_draws(tmp_path, "p", ["s", "t"], ["film-a", "film-b"])
    assert {k: (v["n"], v["k"]) for k, v in got.items()} == {"s": (3, 2), "t": (2, 1)}
    for name, b in bases.items():
        for s, x in xs.items():
            path = cd.draw_features_path(tmp_path, "p", name, s)
            assert path == pool / "draws" / name / f"{s}.npy"
            np.testing.assert_array_equal(np.load(path), cd.project_draws(x, b))
    assert not list(pool.glob("draws/*/*.tmp.npy"))
