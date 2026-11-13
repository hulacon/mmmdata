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


def test_term_values_match_term_stats_values(cm):
    from braintwill.encoding import commonality_weights, draw_counts

    rng = np.random.default_rng(2)
    n_groups, per = 20, 30
    y = rng.standard_normal((n_groups * per, 6))
    got = {k: _sums(y, c * y + 0.2 * rng.standard_normal(y.shape), n_groups)
           for k, c in {"a": 0.5, "b": 0.3, "a+b": 0.6}.items()}
    w = commonality_weights(["a", "b"])
    rois = {"x": np.array([0, 2, 4]), "y": np.arange(6)}
    stats = cm.term_stats(got, rois, draw_counts(n_groups, 20, 0), w)
    vals = cm.term_values(got, rois, w)
    assert set(vals) == set(w)
    for term in w:
        for roi in rois:
            assert vals[term][roi] == pytest.approx(stats[term][roi]["value"], abs=1e-12)
    assert cm.term_values(got, {}, w) == {}


def test_draw_rank_places_a_value_among_draws(cm):
    d = np.arange(100, dtype=float)  # 0 … 99
    top = cm.draw_rank(150.0, d)
    assert top["percentile"] == 100.0 and top["exceeds_q95"] and top["p_draws"] == pytest.approx(1 / 101)
    assert top["draw_q95"] == pytest.approx(np.quantile(d, 0.95)) and top["n_draws"] == 100
    mid = cm.draw_rank(50.0, d)  # 50 below, one tie
    assert mid["percentile"] == pytest.approx(50.5) and not mid["exceeds_q95"]
    assert mid["p_draws"] == pytest.approx((1 + 50) / 101)
    # at the 95th percentile is not above it
    assert not cm.draw_rank(float(np.quantile(d, 0.95)), d)["exceeds_q95"]
    with pytest.raises(ValueError, match="finite"):
        cm.draw_rank(1.0, np.array([0.0, np.nan]))
    with pytest.raises(ValueError, match="draw"):
        cm.draw_rank(1.0, np.array([]))


def test_draw_read_ranks_draw_dependent_terms_and_reports_the_share(cm):
    real = {"opaque": {"r": 0.06, "z": -0.01}, "nameable": {"r": 0.01, "z": 0.0},
            "opaque&nameable": {"r": 0.04, "z": 0.005}, "total": {"r": 0.11, "z": -0.005}}
    draws = [{"opaque": {"r": 0.1 - s, "z": -0.005 - s}, "nameable": {"r": 0.0, "z": 0.0},
              "opaque&nameable": {"r": s, "z": s}, "total": {"r": 0.1, "z": -0.005}}
             for s in np.linspace(0.0, 0.02, 21)]
    out = cm.draw_read(real, draws, ctrl="nameable", other="opaque")
    assert set(out) == {"nameable", "opaque&nameable", "total"}  # C(opaque) moves only with the draw's split
    sh = out["opaque&nameable"]["r"]
    assert sh["percentile"] == 100.0 and sh["exceeds_q95"]
    assert sh["r2_other"] == pytest.approx(0.10) and sh["share"] == pytest.approx(0.4)
    assert sh["share_draw_median"] == pytest.approx(0.01 / 0.10)
    z = out["opaque&nameable"]["z"]  # R²(opaque) = -0.005: no share, the term is still ranked
    assert z["share"] is None and z["share_draw_q95"] is None and z["n_draws"] == 21


def test_draw_groups_names_the_control_set_group(cm, harness, monkeypatch):
    assert cm.draw_groups("dev-vgg19-ctrl") == ("nameable", "opaque")
    with pytest.raises(ValueError, match="two groups"):
        cm.draw_groups("dev-vgg19")
    monkeypatch.setitem(harness.COMMON_ARMS, "dev-noctrl", {"opaque": ("cache:vgg19.conv4_3",),
                                                            "named": ("model:faces",)})
    monkeypatch.setattr(cm, "h", harness)
    with pytest.raises(ValueError, match="ctrl"):
        cm.draw_groups("dev-noctrl")


def test_read_draws_reuses_the_base_units_and_reads_each_draw(cm, monkeypatch):
    import argparse

    from braintwill.encoding import commonality_weights

    rng = np.random.default_rng(3)
    n_groups, per = 10, 20
    y = rng.standard_normal((n_groups * per, 4))
    fit = lambda c: _sums(y, c * y + 0.3 * rng.standard_normal(y.shape), n_groups)  # noqa: E731
    got = {"opaque": fit(0.5), "nameable": fit(0.4), "opaque+nameable": fit(0.6)}
    valid = np.ones(4, bool)
    keys = {s: {"space": f"stack-dev-vgg19-ctrl.{s}", "features": f"real-{s}", "spec": "x"} for s in got}
    asked = []

    def fake_load(paths, cache, register, spaces, sub, oof, engine):
        asked.append(list(spaces))
        draw = int(spaces[0].split("@d")[1][:3])
        c = 0.05 * draw
        g = {"nameable": fit(c), "opaque+nameable": fit(0.5 + c / 2)}
        k = {s: {"space": sp, "features": f"draw{draw}-{s}", "spec": "x"}
             for s, sp in zip(g, spaces)}
        return g, valid, k, {}, {s: {"qp_max_gap": 1e-12 * (draw + 1)} for s in g}

    monkeypatch.setattr(cm, "load_subsets", fake_load)
    args = argparse.Namespace(arm="dev-vgg19-ctrl", engine="stack", register="films", oof="reuse")
    rois = {"all": np.arange(4)}
    w = commonality_weights(["opaque", "nameable"])
    out = cm.read_draws(None, None, args, "##", range(0, 3), "nameable", "opaque", got, valid, keys, rois, w)
    assert asked == [[f"stack-dev-vgg19-ctrl@d{i:03d}.nameable", f"stack-dev-vgg19-ctrl@d{i:03d}.opaque+nameable"]
                     for i in range(3)]  # the opaque unit is never reloaded
    assert out["read"]["opaque&nameable"]["all"]["n_draws"] == 3
    assert out["qp_max_gap_max"] == pytest.approx(3e-12)

    def stale_load(*a, **k):
        g, v, kk, c, st = fake_load(*a, **k)
        return g, v, {s: {**x, "features": f"real-{s}"} for s, x in kk.items()}, c, st

    monkeypatch.setattr(cm, "load_subsets", stale_load)
    with pytest.raises(RuntimeError, match="base arm's features"):
        cm.read_draws(None, None, args, "##", range(0, 1), "nameable", "opaque", got, valid, keys, rois, w)


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


# --- matched-subspace control sources (control_draws.py writes them; ctrl:<pool>.<set>[.dNNN]) ---

def _ctrl_pool(root: Path, sid: str, n_frames: int, d: int = 5, k: int = 2, n_draws: int = 3, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n_frames, d)).astype(np.float32)
    pool = root / "p" / "pool"
    pool.mkdir(parents=True, exist_ok=True)
    np.savez(pool / f"{sid}.npz", stimulus_id=sid, times=np.arange(n_frames) * 0.5, x=x,
             band_names=np.array(["a", "b"]), band_bounds=np.array([[0, 2], [2, d]]))
    bases = np.stack([np.linalg.qr(rng.standard_normal((d, k)))[0] for _ in range(n_draws)]).astype(np.float32)
    np.savez(root / "p" / "s.npz", members=np.array(["a"]), columns=np.array([0, 1]), k=k, share=0.4,
             bases=bases, shares=np.full(n_draws, 0.4), gamma_bracket=np.array([0.0, 1.0]))
    draws = root / "p" / "draws" / "s"  # control_draws.py's layout, written as its draws verb does
    draws.mkdir(parents=True, exist_ok=True)
    np.save(draws / f"{sid}.npy", np.matmul(x.astype(np.float64)[None], bases.astype(np.float64)).astype(np.float32))
    return x, bases


def test_film_ctrl_features_refuse_missing_or_stale_draw_features(harness, tmp_path):
    n = 30
    _ctrl_pool(tmp_path, "film-a", n_frames=n)
    rows = _film_row()
    f = tmp_path / "p" / "draws" / "s" / "film-a.npy"
    np.save(f, np.zeros((2, n, 2), np.float32))  # two draws written, the bases hold three
    with pytest.raises(RuntimeError, match="rerun `control_draws.py draws`"):
        harness.film_ctrl_features(tmp_path, "p", "s", 0, rows, {"film-a": n})
    f.unlink()
    with pytest.raises(FileNotFoundError, match="control_draws.py draws --pool p"):
        harness.film_ctrl_features(tmp_path, "p", "s", 0, rows, {"film-a": n})
    # the set itself needs no draw features
    assert harness.film_ctrl_features(tmp_path, "p", "s", None, rows, {"film-a": n})["film-a"].x.shape == (n, 2)


@pytest.mark.parametrize("src, parsed, band", [
    ("ctrl:dev-pool.nameable", ("ctrl", "ctrl-dev-pool.nameable", None), "ctrl-dev-pool.nameable"),
    ("ctrl:dev-pool.no-e.d007", ("ctrl", "ctrl-dev-pool.no-e", "d007"), "ctrl-dev-pool.no-e.d007"),
])
def test_ctrl_sources_parse_a_set_and_a_draw(harness, src, parsed, band):
    assert harness.common_source(src) == parsed and harness.source_band(src) == band


@pytest.mark.parametrize("src", ["ctrl:pool", "ctrl:pool.set.d7", "ctrl:pool.set.x001", "ctrl:Pool.set"])
def test_ctrl_sources_refuse_malformed_names(harness, src):
    with pytest.raises(ValueError, match="ctrl:<pool>.<set>"):
        harness.common_source(src)


def test_a_draw_arm_swaps_the_one_control_set_for_its_draw(harness, monkeypatch):
    monkeypatch.setitem(harness.COMMON_ARMS, "t-ctrl", {"opaque": ("cache:vgg19.conv4_3",),
                                                        "named": ("ctrl:p.s",)})
    assert harness.check_arm("t-ctrl@d012") == {"opaque": ("cache:vgg19.conv4_3",), "named": ("ctrl:p.s.d012",)}
    monkeypatch.setitem(harness.COMMON_ARMS, "t-none", {"a": ("model:faces",), "b": ("word-on",)})
    with pytest.raises(ValueError, match="exactly one ctrl"):
        harness.check_arm("t-none@d000")


def test_a_draw_job_stacks_each_draw_once_and_shares_the_opaque_unions(harness, monkeypatch):
    monkeypatch.setitem(harness.COMMON_ARMS, "t-ctrl", {"opaque": ("cache:vgg19.conv4_3",),
                                                        "named": ("ctrl:p.s",)})
    spaces = harness.stack_job_spaces("t-ctrl", "3:5")
    assert spaces[:3] == ["stack-t-ctrl.opaque", "stack-t-ctrl.named", "stack-t-ctrl.opaque+named"]
    assert "stack-t-ctrl@d004.opaque+named" in spaces and len(spaces) == 9
    srcs = {s: harness.stack_sources(*harness.parse_stack(s)) for s in spaces}
    assert srcs["stack-t-ctrl@d003.opaque"] == srcs["stack-t-ctrl.opaque"]
    assert srcs["stack-t-ctrl@d003.named"] == ["ctrl:p.s.d003"]
    for bad in ("4:4", "x:3", "3"):
        with pytest.raises(ValueError, match="--draws"):
            harness.stack_job_spaces("t-ctrl", bad)


def test_film_ctrl_features_read_the_set_columns_and_project_a_draw(harness, tmp_path):
    n = 30
    x, bases = _ctrl_pool(tmp_path, "film-a", n_frames=n)
    rows = _film_row()
    on = harness.window_bins(np.array([7, 8]) * 1.5, 1.5, n)
    s = harness.film_ctrl_features(tmp_path, "p", "s", None, rows, {"film-a": n})["film-a"]
    assert s.bands == {"ctrl-p.s": slice(0, 2)}
    np.testing.assert_allclose(s.x[on], x[on][:, :2] - x[on][:, :2].mean(0), rtol=1e-6)
    d = harness.film_ctrl_features(tmp_path, "p", "s", 2, rows, {"film-a": n})["film-a"]
    written = np.load(tmp_path / "p" / "draws" / "s" / "film-a.npy")[2]  # loaded, not projected
    assert d.bands == {"ctrl-p.s.d002": slice(0, 2)} and not d.x[~on].any()
    np.testing.assert_allclose(d.x[on], written[on] - written[on].mean(0), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(written, x.astype(np.float64) @ bases[2], rtol=1e-5)
    with pytest.raises(ValueError, match="draw 3 outside the 3 written"):
        harness.film_ctrl_features(tmp_path, "p", "s", 3, rows, {"film-a": n})
    with pytest.raises(FileNotFoundError, match="control_draws.py build --pool q"):
        harness.film_ctrl_features(tmp_path, "q", "s", None, rows, {"film-a": n})
