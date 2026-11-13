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
    ({"a": ("model:nope",), "b": ("word-on",)}, "no read-\\(b\\) or film-only model"),
    ({"a": ("model:motion.",), "b": ("word-on",)}, "model:<m>.<feature>"),
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


@pytest.mark.parametrize("src, parsed, band", [
    ("model:motion", ("model", "motion", None), "motion"),
    ("model:faces", ("model", "faces", None), "faces"),
    ("model:speech_rate.speech_rate_words", ("model", "speech_rate", "speech_rate_words"),
     "speech_rate.speech_rate_words"),
    ("cuts", ("cuts", "cuts", None), "cuts"),
    ("word-on", ("word-on", "onoff", None), "onoff"),
    ("cache:vgg19.conv4_3", ("cache", "vgg19", "conv4_3"), "vgg19.conv4_3"),
])
def test_common_source_parses_every_kind_and_names_its_band(harness, src, parsed, band):
    assert harness.common_source(src) == parsed
    assert harness.source_band(src) == band


def test_film_only_models_are_arm_sources_and_never_read_b_spaces(harness):
    for m, table in harness.FILM_ONLY_MODELS.items():
        assert f"{harness.MODEL_PREFIX}{m}" not in harness.MODEL_SPACES
        assert harness.source_model_table(m) == table


def _film_row(sid="film-a", start=7, n=2, tr=1.5):
    return pd.DataFrame([{"stimulus_id": sid, "onset": 0.0, "start": start, "n": n, "repetition_time": tr}])


def test_film_cuts_mark_window_cut_bins_only(harness):
    n, rows = 30, _film_row()
    on = harness.window_bins(np.array([7, 8]) * 1.5, 1.5, n)
    inside, outside = int(np.flatnonzero(on)[1]), int(np.flatnonzero(~on)[-1])
    cuts = pd.DataFrame({"stimulus_id": ["film-a"] * 2, "bin": [inside, outside]})
    f = harness.film_cuts(cuts, rows, {"film-a": n})["film-a"]
    assert f.bands == {"cuts": slice(0, 1)} and f.x.shape == (n, 1)
    assert np.flatnonzero(f.x[:, 0]).tolist() == [inside]


def test_film_cuts_refuse_a_film_the_table_lacks_and_a_cut_past_the_grid(harness):
    rows = _film_row()
    with pytest.raises(RuntimeError, match="no cuts for films \\['film-a'\\]"):
        harness.film_cuts(pd.DataFrame({"stimulus_id": ["film-z"], "bin": [3]}), rows, {"film-a": 30})
    with pytest.raises(RuntimeError, match="outside the film's 30 bins"):
        harness.film_cuts(pd.DataFrame({"stimulus_id": ["film-a"], "bin": [30]}), rows, {"film-a": 30})


def test_load_cuts_keeps_accepted_cuts_of_the_dataset_corpus_and_names_a_missing_table(harness, tmp_path):
    with pytest.raises(FileNotFoundError, match="cut_schema_v1.parquet"):
        harness.load_cuts(tmp_path)
    dest = tmp_path / harness.CUT_TABLE
    dest.parent.mkdir(parents=True)
    pd.DataFrame({"corpus": ["mmmdata", "mmmdata", "friends"], "id": ["film-a", "film-a", "film-a"],
                  "bin": [4, 9, 2], harness.CUT_RULE: [True, False, True]}).to_parquet(dest)
    got = harness.load_cuts(tmp_path)
    assert got.to_dict("list") == {"stimulus_id": ["film-a"], "bin": [4]}


def _store_rows(n=30):
    t = np.arange(n) * 0.5
    return pd.concat([pd.DataFrame({"stimulus_id": "film-a", "time": t, "model": "speech_rate",
                                    "feature": f, "value": v})
                      for f, v in [("speech_rate_words", np.arange(n, dtype=float)),
                                   ("speech_rate_pauses", np.ones(n))]], ignore_index=True)


def test_a_one_feature_model_source_keeps_that_feature_under_its_own_band(harness, monkeypatch):
    monkeypatch.setattr(harness, "load_model_long", lambda stimfeat, table, model, rows: _store_rows())
    monkeypatch.setattr(harness, "declared_undefined", lambda stimfeat: {})
    rows = _film_row()
    feats, names = harness.load_film_source(Path("."), rows, "speech_rate", "speech_rate_words")
    f = feats["film-a"]
    assert names == ["speech_rate_words"] and f.bands == {"speech_rate.speech_rate_words": slice(0, 1)}
    on = harness.window_bins(np.array([7, 8]) * 1.5, 1.5, 30)
    v = np.arange(30, dtype=float)
    np.testing.assert_allclose(f.x[on, 0], v[on] - v[on].mean())
    whole, names = harness.load_film_source(Path("."), rows, "speech_rate")
    assert names == ["speech_rate_pauses", "speech_rate_words"] and whole["film-a"].bands == {"speech_rate": slice(0, 2)}
    with pytest.raises(ValueError, match="no feature 'speech_rate_nope'"):
        harness.load_film_source(Path("."), rows, "speech_rate", "speech_rate_nope")


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


# --- stacked arms (the stacking engine) -------------------------------------------------------

def test_parse_stack_names_a_subset_or_one_source_of_a_group(harness):
    assert harness.parse_stack("stack-dev-vgg19.opaque+timing") == ("dev-vgg19", ("opaque", "timing"), None)
    assert harness.parse_stack("stack-dev-vgg19.timing/onsets") == ("dev-vgg19", ("timing",), "onsets")
    assert harness.stack_sources("dev-vgg19", ("timing",), "onsets") == ["model:onsets"]
    assert harness.stack_sources("dev-vgg19", ("opaque", "named"), None) == \
        ["cache:vgg19.conv4_3", "model:faces", "model:depth"]


@pytest.mark.parametrize("space, msg", [
    ("stack-dev-vgg19.timing+opaque", "arm order"),
    ("stack-dev-vgg19.timing/faces", "no source band"),
    ("stack-dev-vgg19.brain/faces", "no group"),
    ("stack-nope.opaque", "unknown commonality arm"),
])
def test_parse_stack_refuses_malformed_names(harness, space, msg):
    with pytest.raises(ValueError, match=msg):
        harness.parse_stack(space)


def test_an_arm_stacks_every_subset_and_each_source_of_a_several_source_group(harness):
    spaces = harness.stack_spaces("dev-vgg19")
    subsets = [s.replace("common-", "stack-", 1) for s in harness.common_spaces("dev-vgg19")]
    assert spaces[:7] == subsets
    solo = spaces[7:]
    assert len(solo) == 2 + 4 and len(set(spaces)) == len(spaces)
    assert not any(s.startswith("stack-dev-vgg19.opaque/") for s in solo)  # one source: its subset is its solo
    assert harness.stack_spaces("dev-vgg19", solo=False) == subsets
    # every space's sources are distinct lists, so no two spaces are one unit by construction
    srcs = [tuple(harness.stack_sources(*harness.parse_stack(s))) for s in spaces]
    assert len(set(srcs)) == len(srcs)


def test_stack_spec_keys_the_route_and_uses_read_b_first_level(harness):
    from braintwill.encoding import cache_key

    a, b = harness.stack_spec("reuse"), harness.stack_spec("nested")
    assert a.ridge == harness.model_spec().ridge and a.group_sums
    reg = {"name": "films"}
    assert cache_key("f" * 64, reg, a, "sub-01")["spec_hash"] != cache_key("f" * 64, reg, b, "sub-01")["spec_hash"]
    with pytest.raises(ValueError, match="--oof"):
        harness.stack_spec("cheap")


def test_masked_register_keys_the_mask_content(harness):
    m = np.zeros(50, bool)
    m[:20] = True
    a = harness.masked_register({"name": "films"}, m)
    assert a == harness.masked_register({"name": "films"}, m.copy())
    m2 = m.copy()
    m2[30] = True
    assert a["scored_columns"]["sha256"] != harness.masked_register({"name": "films"}, m2)["scored_columns"]["sha256"]
    assert a["scored_columns"]["n"] == 20 and a["name"] == "films"


def test_stack_mask_missing_names_the_verb_that_writes_it(harness, tmp_path):
    with pytest.raises(SystemExit, match="reliability.py films --sub 01"):
        harness.stack_mask(tmp_path, "01")


def test_score_refuses_a_stacked_space_before_loading_anything(harness):
    import argparse

    with pytest.raises(SystemExit, match="harness.py stack"):
        harness.cmd_score(argparse.Namespace(space="stack-dev-vgg19.opaque", register="films", sub="01"))


def test_combine_sources_stacks_bands_in_the_order_named(harness):
    from braintwill.encoding import FrameFeatures

    t = np.arange(4) * 0.5
    per = {s: {"film": FrameFeatures("film", t, np.full((4, w), i, np.float64), 0.5, {s: slice(0, w)})}
           for i, (s, w) in enumerate([("a", 2), ("b", 3)])}
    one = harness.combine_sources(per, ["b"])
    assert one is per["b"]
    two = harness.combine_sources(per, ["b", "a"])["film"]
    assert two.bands == {"b": slice(0, 3), "a": slice(3, 5)}
    np.testing.assert_array_equal(two.x[:, :3], 1.0)


# --- fit_stack: synthetic fits (numpy backend) ------------------------------------------------

def _toy_stack_spec():
    from braintwill.encoding import RidgeSpec, StackSpec

    ridge = RidgeSpec(alphas=tuple(np.logspace(-2, 6, 9)), inner_folds=3, solver="random_search", n_iter=5,
                      random_state=0, dtype="float64")
    return StackSpec(ridge=ridge, outer_folds=3, oof="reuse", qp_jitter=1e-10, group_sums=True)


def _toy_design(seed=0, n_groups=9, rows=50):
    rng = np.random.default_rng(seed)
    n = n_groups * rows
    x = rng.standard_normal((n, 9)).astype(np.float32)
    bands = {"a": slice(0, 3), "b": slice(3, 6), "c": slice(6, 9)}
    y = 1.5 * rng.standard_normal((n, 12))
    y[:, :6] += x[:, :3] @ rng.standard_normal((3, 6))
    y[:, 6:] += x[:, 3:6] @ rng.standard_normal((3, 6))
    groups = np.repeat([f"film-{i}" for i in range(n_groups)], rows)
    return x, ((y - y.mean(0)) / y.std(0)).astype(np.float32), groups, bands


def test_fit_stack_scores_mask_columns_only_and_alike_with_or_without_the_rest(harness):
    pytest.importorskip("himalaya")
    x, y, groups, bands = _toy_design()
    spec = _toy_stack_spec()
    mask = np.zeros(y.shape[1], bool)
    mask[[0, 2, 5, 7, 11]] = True
    unions = {"a+b": ["a", "b"], "a": ["a"]}
    masked = harness.fit_stack(x, y, groups, bands, mask, unions, spec, "numpy")
    full = harness.fit_stack(x, y, groups, bands, np.ones_like(mask), unions, spec, "numpy")
    for u in unions:
        np.testing.assert_array_equal(masked[u].valid, mask)
        assert np.isnan(masked[u].r2[~mask]).all()
        # each column is fitted and weighted on its own: restricting columns moves no score
        np.testing.assert_allclose(masked[u].r2[mask], full[u].r2[mask], rtol=1e-6, atol=1e-7)
    assert masked["a"].bands == {"a": 3} and masked["a+b"].bands == {"a": 3, "b": 3}


def test_a_union_scores_the_same_whatever_else_the_job_fits(harness):
    """So a union cached from one arm's job is the score another arm's job would write."""
    pytest.importorskip("himalaya")
    x, y, groups, bands = _toy_design(seed=3)
    spec = _toy_stack_spec()
    mask = np.ones(y.shape[1], bool)
    alone = harness.fit_stack(x, y, groups, bands, mask, {"a+b": ["a", "b"]}, spec, "numpy")["a+b"]
    among = harness.fit_stack(x, y, groups, bands, mask, {"a+b": ["a", "b"], "c": ["c"], "a+b+c": ["a", "b", "c"]},
                              spec, "numpy")["a+b"]
    np.testing.assert_allclose(among.r2, alone.r2, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(among.band_weights, alone.band_weights, rtol=1e-6, atol=1e-7)
    for k in ("sy", "syy", "res"):
        np.testing.assert_allclose(among.groups[k], alone.groups[k], rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("unions, mask_n, msg", [
    ({"a+z": ["a", "z"]}, 12, "bands the design lacks"),
    ({"a": ["a"]}, 11, "mask has"),
])
def test_fit_stack_refuses_unknown_bands_and_a_misshapen_mask(harness, unions, mask_n, msg):
    x, y, groups, bands = _toy_design()
    with pytest.raises(ValueError, match=msg):
        harness.fit_stack(x, y, groups, bands, np.ones(mask_n, bool), unions, _toy_stack_spec(), "numpy")
