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
        cd.check_pool(pool)
        extra = set(decl["extra"])
        for name, entry in decl["sets"].items():
            spec = cd.set_spec(entry)
            for members in (spec["members"], spec["within"] or ()):
                assert len(set(members)) == len(members), (pool, name)
                assert not (set(members) & set(decl["skip"])), (pool, name)
                assert all(m in extra or ":" not in m and m != "word-on" for m in members), (pool, name)


def test_set_spec_reads_both_forms_and_refuses_stray_keys(cd):
    assert cd.set_spec(("a", "b")) == {"members": ("a", "b"), "within": None, "draws": True}
    assert cd.set_spec({"members": ["a"], "within": ["a", "b"], "draws": False}) == {
        "members": ("a",), "within": ("a", "b"), "draws": False}
    with pytest.raises(ValueError, match="expected members"):
        cd.set_spec({"members": ("a",), "n": 3})


def test_pool_blocks_takes_block_or_blocks_once(cd):
    assert cd.pool_blocks({"block": "V"}) == ("V",)
    assert cd.pool_blocks({"blocks": ("V", "A")}) == ("V", "A")
    for bad in ({"block": "V", "blocks": ("A",)}, {}):
        with pytest.raises(ValueError, match="exactly one"):
            cd.pool_blocks(bad)
    for bad in ({"blocks": ("V", "V")}, {"blocks": ("L",)}, {"blocks": ()}):
        with pytest.raises(ValueError, match="each once"):
            cd.pool_blocks(bad)


def test_check_pool_refuses_a_member_outside_its_draws_pool(cd, monkeypatch):
    pools = {"x": {"block": "V", "skip": {}, "extra": (), "sets": {"s": {"members": ("a", "b"), "within": ("a",)}}}}
    monkeypatch.setattr(cd, "POOLS", pools)
    with pytest.raises(ValueError, match=r"members \['b'\] are not in its draws"):
        cd.check_pool("x")


def test_onto_rows_places_a_later_block_on_the_pool_rows(cd):
    at = cd._row_index(["f|0.0", "f|0.5", "g|0.0"])[2]
    X = np.array([[1.0], [2.0], [np.nan]])
    got = cd._onto_rows("A.m", X, ["g|0.0", "f|0.0", "f|0.5"], at)
    np.testing.assert_array_equal(got, [[2.0], [np.nan], [1.0]])
    with pytest.raises(RuntimeError, match="1 pool rows have no row"):
        cd._onto_rows("A.m", X[:2], ["g|0.0", "f|0.0"], at)
    with pytest.raises(RuntimeError, match="two rows"):
        cd._onto_rows("A.m", X, ["f|0.0", "f|0.0", "g|0.0"], at)


class _Drawn:
    def __init__(self, bases):
        self.bases, self.shares = bases, np.full(len(bases), 0.5)
        self.share_range, self.gamma_bracket = (0.4, 0.6), (0.0, 1.0)


def test_build_sets_draws_within_their_sources_and_skips_source_only_sets(cd, tmp_path):
    rng = np.random.default_rng(2)
    X = rng.standard_normal((20, 6))
    bands = {"a": slice(0, 2), "b": slice(2, 4), "c": slice(4, 6)}
    seen = []

    def matched(Xw, k, share, n, n_pool, tol, random_state):
        seen.append((Xw.shape, k))
        return _Drawn([np.ones((Xw.shape[1], k)) * (i + 1) for i in range(n)])

    decl = {"seed": 0, "n": 3, "n_pool": 10, "tol": 0.01, "sets": {
        "all": ("b",), "inner": {"members": ("c",), "within": ("c", "a")}, "only": {"members": ("a",), "draws": False}}}
    sets = cd.build_sets(X, bands, decl, tmp_path, "p", lambda Xw, Q: float(Q.shape[1]) / Xw.shape[1], matched)
    assert seen == [((20, 6), 2), ((20, 4), 2)]  # `only` drew nothing
    inner = np.load(tmp_path / "inner.npz")
    assert inner["columns"].tolist() == [4, 5] and inner["within"].tolist() == [0, 1, 4, 5]
    assert inner["bases"].shape == (3, 6, 2) and not inner["bases"][:, 2:4].any()  # 0 outside `within`
    assert (inner["bases"][1, [0, 1, 4, 5]] == 2).all()
    assert sets["inner"]["share"] == 0.5 and sets["all"]["share"] == pytest.approx(2 / 6)
    only = np.load(tmp_path / "only.npz")
    assert only["bases"].shape == (0, 6, 2) and sets["only"]["draws"] is False
    assert "draw_share_range" not in sets["only"]


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
    np.savez(pool / "u.npz", bases=np.zeros((0, 5, 2), np.float32))  # a source-only set
    assert "u" not in cd.write_draws(tmp_path, "p", ["s", "u"], ["film-a"])
    assert not (pool / "draws" / "u").exists()
