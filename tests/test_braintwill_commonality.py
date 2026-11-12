"""Commonality arms in the encoding harness and their reader (``commonality.py``)."""

from __future__ import annotations

import importlib.util
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parents[1] / "scripts" / "braintwill_encoding"


def _load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except (ImportError, SyntaxError) as exc:  # braintwill needs Python >= 3.12
        pytest.skip(f"braintwill.encoding not importable here: {exc}")
    return mod


@pytest.fixture(scope="module")
def harness():
    return _load("braintwill_encoding_harness_common", "harness.py")


@pytest.fixture(scope="module")
def cm():
    return _load("braintwill_encoding_commonality", "commonality.py")


# --- the partition --------------------------------------------------------------------------

def test_two_group_terms_are_the_textbook_formulas(cm):
    w = cm.term_weights(["a", "b"])
    assert w == {"a": {"a+b": 1, "b": -1}, "b": {"a+b": 1, "a": -1}, "a&b": {"a": 1, "b": 1, "a+b": -1}}


def test_three_group_unique_terms_drop_one_group_from_the_full_fit(cm):
    w = cm.term_weights(["o", "n", "t"])
    assert w["o"] == {"o+n+t": 1, "n+t": -1}
    assert w["o&n&t"] == {"o": 1, "n": 1, "t": 1, "o+n": -1, "o+t": -1, "n+t": -1, "o+n+t": 1}
    assert len(w) == 7


@pytest.mark.parametrize("k", [2, 3, 4])
def test_terms_sum_to_the_full_fit(cm, k):
    groups = [f"g{i}" for i in range(k)]
    rng = np.random.default_rng(k)
    r2 = {cm.subset_key(c): rng.uniform(0, 0.3, size=5)
          for j in range(1, k + 1) for c in itertools.combinations(groups, j)}
    terms = cm.terms_from_r2(r2, cm.term_weights(groups))
    np.testing.assert_allclose(sum(terms.values()), r2[cm.subset_key(groups)], atol=1e-12)


def test_term_weights_refuse_one_group(cm):
    with pytest.raises(ValueError, match="at least two"):
        cm.term_weights(["a"])


def _heldout_r2(X, Y, groups, n_folds=5):
    """Grouped-CV least-squares R² per column (the test's stand-in for a harness fit)."""
    u = np.unique(groups)
    pred = np.zeros_like(Y)
    for f in range(n_folds):
        test = np.isin(groups, u[f::n_folds])
        A = np.c_[X[~test], np.ones((~test).sum())]
        B, *_ = np.linalg.lstsq(A, Y[~test], rcond=None)
        pred[test] = np.c_[X[test], np.ones(test.sum())] @ B
    ss = ((Y - Y.mean(0)) ** 2).sum(0)
    return 1 - ((Y - pred) ** 2).sum(0) / ss


def test_a_timing_signal_both_groups_carry_is_shared_only_until_timing_is_its_own_group(cm):
    rng = np.random.default_rng(0)
    n_films, per = 20, 100
    groups = np.repeat(np.arange(n_films), per)
    n = groups.size
    t = np.sin(np.arange(n) / 7.0)[:, None]  # a timing envelope
    o, m = rng.standard_normal((n, 2)), rng.standard_normal((n, 2))
    feats = {"opaque": np.c_[t, o], "named": np.c_[t, m], "timing": t}
    W = rng.standard_normal((5, 30))
    Y = np.c_[2 * t, o, m] @ W + 0.5 * rng.standard_normal((n, 30))  # timing weighted to be plain

    def partition(gs):
        r2 = {cm.subset_key(c): _heldout_r2(np.hstack([feats[g] for g in c]), Y, groups).mean()
              for j in range(1, len(gs) + 1) for c in itertools.combinations(gs, j)}
        return cm.terms_from_r2(r2, cm.term_weights(gs))

    two = partition(["opaque", "named"])
    three = partition(["opaque", "named", "timing"])
    assert two["opaque&named"] > 0.1  # without a timing group, timing reads as opaque ∩ named
    assert abs(three["opaque&named"]) < 0.01  # with one, nothing is left there
    assert three["opaque&named&timing"] == pytest.approx(two["opaque&named"], abs=0.01)
    assert three["opaque"] == pytest.approx(two["opaque"], abs=0.01)


def _sums(y, p, n_groups):
    yg, pg = np.split(y, n_groups), np.split(p, n_groups)
    return {"labels": [f"film-{i}" for i in range(n_groups)], "n": np.array([len(a) for a in yg]),
            "sy": np.array([a.sum(0) for a in yg]), "syy": np.array([(a ** 2).sum(0) for a in yg]),
            "res": np.array([((a - b) ** 2).sum(0) for a, b in zip(yg, pg)])}


def test_term_stats_read_the_terms_from_group_sums_with_paired_intervals(cm, harness):
    from braintwill.encoding import resampled_r2

    rng = np.random.default_rng(1)
    n_groups, per, n_cols = 25, 40, 8
    y = rng.standard_normal((n_groups * per, n_cols))
    preds = {"a": 0.5 * y, "b": 0.4 * y + 0.1 * rng.standard_normal(y.shape),
             "a+b": 0.6 * y + 0.05 * rng.standard_normal(y.shape)}
    got = {k: _sums(y, p, n_groups) for k, p in preds.items()}
    w = cm.term_weights(["a", "b"])
    rois = {"left": np.arange(4), "right": np.arange(4, 8)}
    counts = np.random.default_rng(0).multinomial(n_groups, np.full(n_groups, 1 / n_groups), size=300)
    out = cm.term_stats(got, rois, counts, w)
    direct = cm.terms_from_r2({k: resampled_r2(v, np.ones(n_groups)) for k, v in got.items()}, w)
    for term in w:
        for roi, ix in rois.items():
            d = out[term][roi]
            assert d["value"] == pytest.approx(direct[term][ix].mean(), abs=1e-10)
            assert d["lo"] <= d["value"] <= d["hi"] and d["n_cols"] == 4
    total = out["total"]["left"]["value"]
    assert total == pytest.approx(sum(out[t]["left"]["value"] for t in w), abs=1e-10)


def test_term_stats_refuse_a_missing_fit_and_mismatched_groups(cm):
    y = np.random.default_rng(2).standard_normal((40, 3))
    s = _sums(y, 0.5 * y, 4)
    w = cm.term_weights(["a", "b"])
    with pytest.raises(ValueError, match="subset fits missing"):
        cm.term_stats({"a": s, "b": s}, {"r": np.arange(3)}, np.ones((2, 4)), w)
    other = {**s, "labels": list(reversed(s["labels"]))}
    with pytest.raises(RuntimeError, match="different CV groups"):
        cm.term_stats({"a": s, "b": other, "a+b": s}, {"r": np.arange(3)}, np.ones((2, 4)), w)


# --- arms in the harness ----------------------------------------------------------------------

def test_parse_common_names_an_arm_and_a_subset_in_arm_order(harness):
    assert harness.parse_common("common-dev-vgg19.opaque+timing") == ("dev-vgg19", ("opaque", "timing"))
    assert harness.parse_common("common-dev-vgg19.named") == ("dev-vgg19", ("named",))


@pytest.mark.parametrize("space, msg", [
    ("common-dev-vgg19.timing+opaque", "arm order"),
    ("common-dev-vgg19.opaque+opaque", "arm order"),
    ("common-dev-vgg19.opaque+brain", "no group"),
    ("common-nope.opaque", "unknown commonality arm"),
    ("common-dev-vgg19", "expected common-"),
])
def test_parse_common_refuses_malformed_names(harness, space, msg):
    with pytest.raises(ValueError, match=msg):
        harness.parse_common(space)


def test_an_arm_lists_every_subset_once_smallest_first(harness):
    spaces = harness.common_spaces("dev-vgg19")
    assert len(spaces) == 7 and len(set(spaces)) == 7
    assert spaces[:3] == ["common-dev-vgg19.opaque", "common-dev-vgg19.named", "common-dev-vgg19.timing"]
    assert spaces[-1] == "common-dev-vgg19.opaque+named+timing"
    assert all(harness.parse_common(s)[0] == "dev-vgg19" for s in spaces)


def test_common_merge_is_one_kernel_band_per_group(harness):
    m = harness.common_merge("dev-vgg19", ("opaque", "timing"))
    assert m == {"opaque": ("vgg19.conv4_3",), "timing": ("onoff", "onsets", "loudness", "llstat")}


def test_a_group_shares_its_columns_spec_across_subsets(harness):
    """The merge for a group is the same whichever subset holds it, so a group enters every fit alike."""
    for g in harness.COMMON_ARMS["dev-vgg19"]:
        seen = {tuple(harness.common_merge("dev-vgg19", harness.parse_common(s)[1])[g])
                for s in harness.common_spaces("dev-vgg19") if g in harness.parse_common(s)[1]}
        assert len(seen) == 1


@pytest.mark.parametrize("arm, msg", [
    ({"a": ("model:fasttext",), "b": ("word-on",)}, "text models"),
    ({"a": ("cache:vgg19.conv9_9",), "b": ("word-on",)}, "cache:<space>.<band>"),
    ({"a": ("model:nope",), "b": ("word-on",)}, "no read-\\(b\\) model"),
    ({"a": ("model:faces",), "b": ("model:faces",)}, "is in groups"),
    ({"a": ("model:faces",)}, "at least two groups"),
    ({"a": ("model:faces",), "b": ()}, "no sources"),
    ({"a": ("model:faces",), "B": ("word-on",)}, "group name"),
    ({"a": ("faces",), "b": ("word-on",)}, "expected model:"),
])
def test_check_arm_refuses_bad_declarations(harness, monkeypatch, arm, msg):
    monkeypatch.setitem(harness.COMMON_ARMS, "test-arm", arm)
    with pytest.raises(ValueError, match=msg):
        harness.check_arm("test-arm")


def _cache(root: Path, sid: str, n_frames: int, width: int = 3, stamp_offset: float = 0.0):
    d = root / "vgg19"
    d.mkdir(parents=True, exist_ok=True)
    x = np.arange(n_frames * (2 + width), dtype=np.float32).reshape(n_frames, 2 + width)
    np.savez(d / f"{sid}.npz", stimulus_id=sid, times=np.arange(n_frames) * 0.5 + stamp_offset, x=x,
             band_names=np.array(["conv1_2", "conv4_3"]), band_bounds=np.array([[0, 2], [2, 2 + width]]))
    return x


def test_film_cache_features_centre_one_band_over_the_window_on_the_model_grid(harness, tmp_path):
    n, tr = 30, 1.5
    x = _cache(tmp_path, "film-a", n_frames=34)  # four frames past the film's grid: dropped
    rows = pd.DataFrame([{"stimulus_id": "film-a", "onset": 0.0, "start": 7, "n": 2, "repetition_time": tr}])
    feats = harness.film_cache_features(tmp_path, "vgg19", "conv4_3", rows, {"film-a": n})
    f = feats["film-a"]
    on = harness.window_bins(np.array([7, 8]) * tr, tr, n)
    assert f.bands == {"vgg19.conv4_3": slice(0, 3)} and f.x.shape == (n, 3)
    band = x[:n, 2:].astype(float)
    np.testing.assert_allclose(f.x[on], band[on] - band[on].mean(0))
    assert not f.x[~on].any()


def test_film_cache_features_take_centre_stamps_and_refuse_a_short_cache(harness, tmp_path):
    n, tr = 30, 1.5
    rows = pd.DataFrame([{"stimulus_id": "film-b", "onset": 0.0, "start": 7, "n": 2, "repetition_time": tr}])
    _cache(tmp_path, "film-b", n_frames=n, stamp_offset=0.25)
    f = harness.film_cache_features(tmp_path, "vgg19", "conv1_2", rows, {"film-b": n})["film-b"]
    assert f.x.shape == (n, 2)
    _cache(tmp_path, "film-b", n_frames=10)  # ends before the window does
    with pytest.raises(RuntimeError, match="window bins have no cache frame"):
        harness.film_cache_features(tmp_path, "vgg19", "conv1_2", rows, {"film-b": n})


def test_arms_differing_only_in_the_opaque_source_share_the_subsets_without_it(harness, monkeypatch):
    other = dict(harness.COMMON_ARMS["dev-vgg19"], opaque=("cache:vgg19.conv3_3",))
    monkeypatch.setitem(harness.COMMON_ARMS, "dev-other", other)
    for groups in [("named",), ("timing",), ("named", "timing")]:
        a, b = (harness.model_spec(group_sums=True, merge=harness.common_merge(arm, groups))
                for arm in ("dev-vgg19", "dev-other"))
        assert a == b
    a, b = (harness.common_merge(arm, ("opaque", "named")) for arm in ("dev-vgg19", "dev-other"))
    assert a != b
