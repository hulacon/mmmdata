"""Commonality arms in the encoding harness and their reader (``commonality.py``)."""

from __future__ import annotations

import importlib.util
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


# --- the reader (the algebra and the bootstrap are braintwill's; tested there) ---------------

def _sums(y, p, n_groups):
    yg, pg = np.split(y, n_groups), np.split(p, n_groups)
    return {"labels": [f"film-{i}" for i in range(n_groups)], "n": np.array([len(a) for a in yg]),
            "sy": np.array([a.sum(0) for a in yg]), "syy": np.array([(a ** 2).sum(0) for a in yg]),
            "res": np.array([((a - b) ** 2).sum(0) for a, b in zip(yg, pg)])}


def test_term_stats_reads_every_term_and_the_total(cm):
    from braintwill.encoding import combine, commonality_weights, draw_counts, resampled_r2

    rng = np.random.default_rng(1)
    n_groups, per = 25, 40
    y = rng.standard_normal((n_groups * per, 8))
    got = {k: _sums(y, c * y + 0.1 * rng.standard_normal(y.shape), n_groups)
           for k, c in {"a": 0.5, "b": 0.4, "a+b": 0.6}.items()}
    w = commonality_weights(["a", "b"])
    rois = {"left": np.arange(4), "right": np.arange(4, 8)}
    out = cm.term_stats(got, rois, draw_counts(n_groups, 200, 0), w)
    assert set(out) == {"a", "b", "a&b", "total"}
    direct = combine({k: resampled_r2(v, np.ones(n_groups)) for k, v in got.items()}, w)
    for t in w:
        assert out[t]["left"]["value"] == pytest.approx(direct[t][:4].mean(), abs=1e-12)
    assert out["total"]["right"]["value"] == pytest.approx(sum(out[t]["right"]["value"] for t in w), abs=1e-12)
    assert cm.term_stats(got, {}, draw_counts(n_groups, 5, 0), w) == {}


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
