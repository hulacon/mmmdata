"""scripts/braintwill_encoding/harness.py — the films register's row selection and held-out guard.

Synthetic window tables only; the fits themselves are braintwill's and tested there.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

HARNESS = Path(__file__).resolve().parents[1] / "scripts" / "braintwill_encoding" / "harness.py"


@pytest.fixture(scope="module")
def harness():
    spec = importlib.util.spec_from_file_location("braintwill_encoding_harness", HARNESS)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except (ImportError, SyntaxError) as exc:  # braintwill needs Python >= 3.12
        pytest.skip(f"braintwill.encoding not importable here: {exc}")
    return mod


def _windows(harness, n_pool=None):
    n_pool = harness.fm.N_POOL if n_pool is None else n_pool
    rows = [{"sub": "01", "ses": f"{2 + i // 4:02d}", "stimulus_id": f"film-{i:02d}", "role": "alignment"}
            for i in range(n_pool)]
    held = harness.fm.HELDOUT_SESSIONS[0]
    rows += [{"sub": "01", "ses": held, "stimulus_id": "film-held", "role": "heldout"},
             {"sub": "02", "ses": "02", "stimulus_id": "film-00", "role": "alignment"}]
    return pd.DataFrame(rows)


def test_the_pool_is_one_subjects_alignment_rows(harness):
    w = _windows(harness)
    rows = harness.alignment_rows(w, "01")
    assert len(rows) == harness.fm.N_POOL
    assert set(rows["role"]) == {"alignment"} and set(rows["sub"]) == {"01"}


def test_a_subset_keeps_table_order(harness):
    w = _windows(harness)
    rows = harness.alignment_rows(w, "01", ["film-03", "film-01"])
    assert list(rows["stimulus_id"]) == ["film-01", "film-03"]


def test_a_film_outside_the_pool_is_refused(harness):
    with pytest.raises(RuntimeError, match="not in the alignment pool"):
        harness.alignment_rows(_windows(harness), "01", ["film-held"])


def test_a_heldout_session_row_with_an_alignment_role_is_refused(harness):
    w = _windows(harness)
    w.loc[w["stimulus_id"] == "film-held", "role"] = "alignment"
    with pytest.raises(RuntimeError, match="held-out sessions"):
        harness.alignment_rows(w, "01")


def test_a_short_pool_is_refused(harness):
    with pytest.raises(RuntimeError, match="expected the pool"):
        harness.alignment_rows(_windows(harness, n_pool=10), "01")


# --- reliability.py: the films mask ------------------------------------------------------------

RELIABILITY = HARNESS.with_name("reliability.py")


@pytest.fixture(scope="module")
def reliability():
    spec = importlib.util.spec_from_file_location("braintwill_encoding_reliability", RELIABILITY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_mean_pair_corr_equals_the_pairwise_average(reliability):
    import numpy as np

    rng = np.random.default_rng(0)
    z = np.stack([reliability._zscore(s) for s in rng.standard_normal((5, 50, 7))])
    pairs = [np.mean(z[i] * z[j], axis=0) for i in range(5) for j in range(i + 1, 5)]
    np.testing.assert_allclose(reliability.mean_pair_corr(z), np.mean(pairs, axis=0), atol=1e-12)


def test_bh_matches_the_step_up_rule(reliability):
    import numpy as np

    p = np.array([0.01, 0.04, 0.03, 0.2])
    # sorted .01 .03 .04 .2 -> p·m/rank .04 .06 .0533 .2 -> running min from the top
    np.testing.assert_allclose(reliability.bh(p), [0.04, 0.16 / 3, 0.16 / 3, 0.2], rtol=1e-12)


def test_repeat_rows_refuse_a_heldout_session(reliability):
    held = reliability.fm.HELDOUT_SESSIONS[0]
    w = pd.DataFrame([{"sub": "01", "ses": "02", "stimulus_id": "rep-a", "role": "dropped_repeat", "showing": 1},
                      {"sub": "01", "ses": held, "stimulus_id": "rep-a", "role": "dropped_repeat", "showing": 2}])
    with pytest.raises(RuntimeError, match="held-out sessions"):
        reliability.repeat_rows(w, "01")
    ok = reliability.repeat_rows(w[w["ses"] != held], "01")
    assert list(ok) == ["rep-a"] and list(ok["rep-a"]) == ["1"]


def test_item_noise_is_the_item_weighted_within_variance(reliability):
    import numpy as np

    rng = np.random.default_rng(1)
    y = rng.standard_normal((9, 4))
    labels = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2])
    want = np.mean([y[labels == i].var(axis=0, ddof=1) for i in range(3)], axis=0)
    np.testing.assert_allclose(reliability.item_noise(y, labels, 3), want, atol=1e-12)


def test_permutation_stays_within_blocks(reliability):
    import numpy as np

    labels = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    blocks = [np.arange(4), np.arange(4, 8)]
    out = reliability.permute_within(labels, blocks, np.random.default_rng(0))
    assert sorted(out[:4]) == [0, 0, 1, 1] and sorted(out[4:]) == [2, 2, 3, 3]


# --- reliability.py: the time-series ceiling (tb-nat-encoding settle 3) -------------------------

def _tb_events():
    """A triplet (pairIds 1-3) intact, then broken by a rest, then intact in A-C-B order; one pair."""
    slots = [(9.0, 1, 3), (13.5, 2, 3), (18.0, 3, 3), (22.5, None, None),
             (27.0, 1, 3), (31.5, 2, 3), (36.0, None, None), (40.5, 3, 3),
             (45.0, 1, 3), (49.5, 3, 3), (54.0, 2, 3), (58.5, 7, 2)]
    rows = []
    for on, pid, con in slots:
        if pid is None:
            rows.append({"onset": on, "onset_actual": float("nan"), "trial_type": "rest",
                         "pairId": "n/a", "mmmId": "n/a", "enCon": "n/a"})
            continue
        for kind in ("image", "word"):
            rows.append({"onset": on, "onset_actual": on + 0.02, "trial_type": kind,
                         "pairId": pid, "mmmId": 100 + pid, "enCon": con})
    return pd.DataFrame(rows)


def test_triplet_presentations_split_at_rests_and_keep_item_order(reliability):
    p = reliability.triplet_presentations(_tb_events(), 1.5)
    assert list(p["k0"]) == [6, 18, 27, 30]
    assert list(p["n_slots"]) == [3, 2, 1, 3]
    assert set(p["seq"]) == {0}
    assert p["items"].iloc[0] == (101, 102, 103) and p["items"].iloc[3] == (101, 103, 102)


def test_triplet_presentations_refuse_an_offgrid_onset(reliability):
    e = _tb_events()
    e.loc[e["onset"] == 9.0, "onset"] = 9.2
    with pytest.raises(RuntimeError, match="off the volume grid"):
        reliability.triplet_presentations(e, 1.5)


def test_identical_volumes_are_those_whose_whole_fir_window_is_inside(reliability):
    import numpy as np

    tr, delays, k0 = 1.5, (3, 4, 5, 6, 7), 6
    got = reliability.identical_volumes(k0, 3, 4.5, tr, delays)
    lo, hi = k0 * tr, k0 * tr + 13.5
    want = [k for k in range(60) if all(lo <= k * tr - d * tr and k * tr - (d - 1) * tr <= hi for d in delays)]
    assert list(got) == want == list(np.arange(13, 18))


def test_ceiling_pairs_take_intact_same_order_presentations_only(reliability):
    e = _tb_events()
    late = e.assign(onset=e["onset"] + 90.0, onset_actual=e["onset_actual"] + 90.0)  # same design, later
    p = pd.concat([reliability.triplet_presentations(e, 1.5).assign(run="r1"),
                   reliability.triplet_presentations(late, 1.5).assign(run="r2")]).assign(ses="04")
    pairs = reliability.ceiling_pairs(p, 1.5, (3, 4, 5, 6, 7), {"r1": 210, "r2": 210}, 7)
    # intact: (A,B,C)@6 and @66 -> one same-order pair; (A,C,B)@30 and @90 -> another
    assert pairs["pair"].nunique() == 2 and len(pairs) == 10
    first = pairs[pairs["pair"] == 0]
    assert list(first["k_i"]) == [13, 14, 15, 16, 17] and list(first["k_j"]) == [73, 74, 75, 76, 77]
    assert set(pairs["unit"]) == {"04:0"}


def test_pair_products_average_to_mean_pair_corr(reliability):
    import numpy as np

    rng = np.random.default_rng(2)
    z = np.stack([reliability._zscore(s) for s in rng.standard_normal((4, 40, 6))])
    np.testing.assert_allclose(reliability.pair_products(z).mean(axis=0), reliability.mean_pair_corr(z),
                               atol=1e-12)


def test_unit_bootstrap_se(reliability):
    import numpy as np

    rng = np.random.default_rng(3)
    same = np.ones((20, 3))
    assert np.allclose(reliability.unit_bootstrap(same, np.ones(20), rng), 0.0)
    v = rng.standard_normal((400, 1))
    se = reliability.unit_bootstrap(v, np.ones(400), rng, n_boot=2000)
    np.testing.assert_allclose(se, v.std(ddof=1) / np.sqrt(400), rtol=0.1)
    # strata: units never move between strata, so two constant strata give SE 0
    strat = np.r_[np.zeros(10), np.ones(10)].astype(int)
    vals = np.r_[np.zeros(10), np.ones(10)][:, None]
    assert np.allclose(reliability.unit_bootstrap(vals, np.ones(20), rng, strata=strat), 0.0)


def test_ceiling_floor_needs_both_parts(reliability):
    import numpy as np

    nc = np.array([0.05, 0.12, 0.12, 0.30, np.nan])
    se = np.array([0.001, 0.05, 0.03, 0.01, 0.01])
    assert list(reliability.ceiling_floor(nc, se)) == [False, False, True, True, False]


def _tb_trials():
    """Two sessions: a triplet shown twice, a repeated pair, a single, and one super repeat x2 per session."""
    rows = []
    for ses in ("ses-04", "ses-05"):
        tag = ses[-1]
        seq = ([("t1" + tag, 3, 1), ("t2" + tag, 3, 2), ("t3" + tag, 3, 3)] * 2
               + [("r" + tag, 2, 4)] * 2 + [("s" + tag, 1, 5)] + [("super", 2, 6)] * 2)
        rows += [{"session": ses, "stimulus_id": s, "enCon": float(c), "pairId": float(pid)} for s, c, pid in seq]
    return pd.DataFrame(rows)


def test_tb_rows_average_super_repeats_and_group_triplets(harness):
    import numpy as np

    t = _tb_trials()
    betas = np.random.default_rng(2).standard_normal((len(t), 3)).astype(np.float32)
    rows, y = harness.tb_rows(t, betas)
    z = harness.zscore_sessions(t, betas)
    assert len(rows) == len(t) - 2  # one super row per session instead of two
    sup = rows["stimulus_id"] == "super"
    assert list(rows.loc[sup, "n_trials"]) == [2, 2]
    first = np.flatnonzero(sup.to_numpy())[0]
    sel = ((t["stimulus_id"] == "super") & (t["session"] == "ses-04")).to_numpy()
    # the super row is its trials' mean, shifted by the session re-centring like every row
    ses4 = (rows["session"] == "ses-04").to_numpy()
    rows4 = np.concatenate([z[(t["session"] == "ses-04").to_numpy() & ~(t["stimulus_id"] == "super").to_numpy()],
                            z[sel].mean(axis=0, keepdims=True)])
    np.testing.assert_allclose(y[first], z[sel].mean(axis=0) - rows4.mean(axis=0), atol=1e-5)
    np.testing.assert_allclose(y[ses4].mean(axis=0), 0, atol=1e-6)
    trip = rows[rows["stimulus_id"].str.startswith("t")]
    assert trip.groupby("session")["group"].nunique().tolist() == [1, 1]
    assert trip["group"].str.startswith("triplet:").all()
    other = rows[~rows["stimulus_id"].str.startswith("t") & ~sup]
    assert (other["group"] == other["stimulus_id"]).all()
    np.testing.assert_allclose(z[(t["session"] == "ses-04").to_numpy()].mean(axis=0), 0, atol=1e-6)


def test_tb_rows_refuse_a_broken_triplet(harness):
    import numpy as np

    t = _tb_trials()
    t.loc[t["stimulus_id"] == "t34", "pairId"] = 4.0  # third mate now falls in the next triplet slot
    with pytest.raises(RuntimeError, match="triplet groups"):
        harness.tb_rows(t, np.zeros((len(t), 2), np.float32))


# ---------------------------------------------------------------------------
# tb-enc-ts: on/off band, modulators, run selection
# ---------------------------------------------------------------------------

def _long(times, model, values):
    """Composed-table rows: one row per (time, feature); ``values`` is (n_times, n_features)."""
    rows = []
    for t, v in zip(times, values):
        for j, x in enumerate(v):
            rows.append({"time": t, "model": model, "feature": f"{model}_{j:03d}", "value": x})
    return pd.DataFrame(rows)


def test_modulators_centre_over_on_bins_and_zero_the_rest(harness):
    import numpy as np

    on = np.array([False, True, True, True, False])
    f = np.array([[np.nan], [1.0], [2.0], [6.0], [np.nan]])
    out = harness.modulators(f, on)
    np.testing.assert_allclose(out[:, 0], [0, -2, -1, 3, 0])
    f[2, 0] = np.nan  # undefined at an on-bin stays undefined, and leaves the mean
    out = harness.modulators(f, on)
    assert out[0, 0] == 0 and np.isnan(out[2, 0])
    np.testing.assert_allclose(out[[1, 3], 0], [-2.5, 2.5])


def test_modulators_refuse_a_value_where_the_stream_is_off(harness):
    import numpy as np

    with pytest.raises(RuntimeError, match="stream is off"):
        harness.modulators(np.array([[1.0], [2.0]]), np.array([False, True]))


def test_run_features_build_onoff_then_modulators(harness):
    import numpy as np

    n = 8
    vis_t = np.arange(n) * 0.5
    vis = np.full((n, 2), np.nan)
    vis[2:6] = [[1, 0], [3, 0], [1, 4], [3, 4]]
    aud_t = np.arange(n) * 0.5 + 0.25  # the generic audio table stamps bin centres
    aud = np.full((n, 1), np.nan)
    aud[3] = 7.0
    tables = {"movies_frames": (_long(vis_t, "ebind", vis), "start"),
              "movies_audio_frames": (_long(aud_t, "clap", aud), "center")}
    f = harness.run_features("run-x", tables, "ebind", n)
    assert list(f.bands) == ["onoff", "ebind"]
    np.testing.assert_array_equal(f.x[:, 0], [0, 0, 1, 1, 1, 1, 0, 0])  # visual-on
    np.testing.assert_array_equal(f.x[:, 1], [0, 0, 0, 1, 0, 0, 0, 0])  # audio-on, centre shifted to bin 3
    np.testing.assert_allclose(f.x[2:6, 2:], [[-1, -2], [1, -2], [-1, 2], [1, 2]])
    assert not f.x[[0, 1, 6, 7], 2:].any()


def test_onoff_space_is_the_onoff_band_alone(harness):
    import numpy as np

    n = 6
    vis = np.full((n, 1), np.nan)
    vis[1:4] = 2.0
    aud = np.full((n, 1), np.nan)
    aud[2] = 1.0
    tables = {"movies_frames": (_long(np.arange(n) * 0.5, "ebind", vis), "start"),
              "movies_audio_frames": (_long(np.arange(n) * 0.5 + 0.25, "clap", aud), "center")}
    f = harness.run_features("run-x", tables, harness.ONOFF_SPACE, n)
    assert list(f.bands) == ["onoff"] and f.x.shape == (n, 2)
    np.testing.assert_array_equal(f.x[:, 0], [0, 1, 1, 1, 0, 0])
    np.testing.assert_array_equal(f.x[:, 1], [0, 0, 1, 0, 0, 0])
    assert harness.ONOFF_SPACE not in harness.MODEL_SPACES


def test_voice_features_one_column_per_voice_on_the_word_bins(harness):
    import numpy as np

    n = 8
    x = np.zeros((n, 2))
    x[[1, 4, 6], 1] = 1                                   # audio-on bins
    f = harness.FrameFeatures("run-x", np.arange(n) * 0.5, x, 0.5, {"onoff": slice(0, 2)})
    words = pd.DataFrame({"onset_actual": [3.1, 0.52, 2.01], "voice": ["nova", "echo", "nova"]})
    g = harness.voice_features(f, words)
    assert list(g.bands) == ["onoff", "voice"] and g.x.shape == (n, 2 + len(harness.VOICES))
    v = g.x[:, g.bands["voice"]]
    np.testing.assert_allclose(v[[1, 4, 6], 0], np.array([1, 0, 0]) - 1 / 3)   # echo
    np.testing.assert_allclose(v[[1, 4, 6], 1], np.array([0, 1, 1]) - 2 / 3)   # nova
    assert not v[[0, 2, 3, 5, 7]].any() and not v[:, 2:].any()
    with pytest.raises(RuntimeError, match="do not land"):
        harness.voice_features(f, words.assign(onset_actual=[3.6, 0.52, 2.01]))
    with pytest.raises(RuntimeError, match="outside"):
        harness.voice_features(f, words.assign(voice=["nova", "alloy", "nova"]))


def _manifest(rows):
    base = {"sub": "01", "task": "TBencoding", "n_vol": "210", "n_nss": "0", "RepetitionTime": "1.5"}
    return pd.DataFrame([{**base, **r, "path": f"sub-01/ses-{r['ses']}/func/sub-01_ses-{r['ses']}_task-TBencoding"
                                              f"_run-{r['run']}_space-fsaverage6_desc-reference_bold.dtseries.nii"}
                         for r in rows])


def test_tb_ts_runs_sort_and_strip_the_stem(harness):
    m = _manifest([{"ses": "05", "run": "01"}, {"ses": "04", "run": "02"}, {"ses": "04", "run": "01"}])
    r = harness.tb_ts_runs(m, "01")
    assert list(zip(r["ses"], r["run"])) == [("04", "01"), ("04", "02"), ("05", "01")]
    assert r["stem"].iloc[0] == "sub-01_ses-04_task-TBencoding_run-01"


def test_tb_ts_runs_refuse_a_heldout_session(harness):
    held = harness.fm.HELDOUT_SESSIONS[0]
    with pytest.raises(RuntimeError, match="outside the trial-based sessions"):
        harness.tb_ts_runs(_manifest([{"ses": "04", "run": "01"}, {"ses": held, "run": "01"}]), "01")


def test_tb_ts_runs_refuse_too_many_nonsteady_volumes(harness):
    m = _manifest([{"ses": "04", "run": "01", "n_nss": str(harness.TB_TS_DROP)}])
    with pytest.raises(RuntimeError, match="non-steady-state"):
        harness.tb_ts_runs(m, "01")


def test_the_drop_is_the_first_volume_whose_delays_all_land_in_the_run(harness):
    import numpy as np
    from braintwill.encoding import fir_design

    n, tr = 60, 1.5  # 30 s of 0.5 s bins
    f = harness.FrameFeatures("run-x", np.arange(n) * 0.5, np.ones((n, 1)), 0.5)
    fir_design(f, np.arange(harness.TB_TS_DROP, 20) * tr, tr, harness.fs_enc.DELAYS_TR, end=n * 0.5)
    with pytest.raises(Exception, match="no frames"):
        fir_design(f, np.arange(harness.TB_TS_DROP - 1, 20) * tr, tr, harness.fs_enc.DELAYS_TR, end=n * 0.5)


# ---------------------------------------------------------------------------
# model spaces (read (b)): gridding, nested gates, film windows
# ---------------------------------------------------------------------------

def test_model_spaces_are_new_names_on_both_registers(harness):
    assert len(harness.MODEL_SPACES) == 27
    assert set(harness.MODEL_SPACES) <= set(harness.TB_TS_SPACES)
    assert harness.TB_TS_SPACES["ebind"] == {"ebind": ("movies_frames", "ebind")}  # summary space untouched
    spec = harness.model_spec().to_dict()["ridge"]
    assert spec["dtype"] == "float64" and "diagonalize_method" not in spec["solver_params"]
    assert "dtype" not in harness.film_spec().to_dict()["ridge"]  # existing specs keep their hash


def test_group_sums_is_its_own_model_spec(harness):
    plain, kept = harness.model_spec().to_dict(), harness.model_spec(group_sums=True).to_dict()
    assert kept.get("group_sums") is True and "group_sums" not in plain
    assert {k: v for k, v in kept.items() if k != "group_sums"} == plain


def test_grid_wide_collapses_identical_copies_and_refuses_differing_ones(harness):
    import numpy as np

    d = _long([0.0, 0.5], "m", [[1.0], [2.0]])
    w = harness.grid_wide(pd.concat([d, d]), "m", "start", 3)
    np.testing.assert_array_equal(w.to_numpy()[:, 0], [1.0, 2.0, np.nan])
    other = d.assign(value=[1.0, 5.0])
    with pytest.raises(RuntimeError, match="different values"):
        harness.grid_wide(pd.concat([d, other]), "m", "start", 3)


def test_infer_stamp(harness):
    import numpy as np

    assert harness.infer_stamp(np.array([0.0, 0.5, 1.0])) == "start"
    assert harness.infer_stamp(np.array([0.25, 0.75])) == "center"
    with pytest.raises(RuntimeError):
        harness.infer_stamp(np.array([0.0, 0.25]))


def test_window_bins_are_the_bins_the_fir_spans_read(harness):
    import numpy as np

    tr = 1.5
    starts = np.array([10.5, 12.0])  # spans [starts - 7 TR, starts - 3 TR + TR)
    on = harness.window_bins(starts, tr, 40, delays=(3, 4, 5, 6, 7))
    t = np.arange(40) * 0.5
    np.testing.assert_array_equal(t[on], np.arange(0.0, 12.0 - 3 * tr + tr, 0.5))


def test_gated_modulators_nest_the_gate(harness):
    import numpy as np

    on = np.array([False, True, True, True, True, False])
    w = pd.DataFrame({"a": [np.nan, 1, 2, 3, 6, np.nan],                # ungated
                      "g": [np.nan, 2, np.nan, 4, np.nan, np.nan]})     # gated: defined on 2 of 4
    x, names = harness.gated_modulators(w, on, {"g"})
    assert names == ["a", "g", "g:defined"]
    np.testing.assert_allclose(x[:, 0], [0, -2, -1, 0, 3, 0])
    np.testing.assert_allclose(x[:, 1], [0, -1, 0, 1, 0, 0])            # centred over defined, 0 elsewhere
    np.testing.assert_allclose(x[:, 2], [0, .5, -.5, .5, -.5, 0])       # defined − its on-mean
    # a gated feature defined nowhere gives a zero value column (and a zero indicator)
    x, _ = harness.gated_modulators(w.assign(g=np.nan), on, {"g"})
    assert not x[:, 1:].any()
    with pytest.raises(RuntimeError, match="stream is off"):
        harness.gated_modulators(w.assign(g=[9, 2, np.nan, 4, np.nan, np.nan]), on, {"g"})


def test_prune_indicators_drops_zero_and_repeated_gates(harness):
    import numpy as np

    def ff(sid, x):
        return harness.FrameFeatures(sid, np.arange(3) * 0.5, np.asarray(x, float), 0.5,
                                     {"onoff": slice(0, 1), "m": slice(1, 5)})
    names = ["a", "a:defined", "b:defined", "c:defined"]
    feats = {"r1": ff("r1", [[1, 0, .5, .5, 0], [1, 1, -.5, -.5, 0], [0, 0, 0, 0, 0]]),
             "r2": ff("r2", [[1, 2, .5, .5, 0], [1, 3, -.5, -.5, 0], [0, 0, 0, 0, 0]])}
    out, gates = harness.prune_indicators(feats, "m", names)
    assert gates["indicators"] == ["a:defined"]
    assert gates["dropped"] == {"zero": ["c:defined"], "duplicate": {"b:defined": "a:defined"}}
    assert out["r1"].bands == {"onoff": slice(0, 1), "m": slice(1, 3)}
    np.testing.assert_array_equal(out["r2"].x, [[1, 2, .5], [1, 3, -.5], [0, 0, 0]])


def test_film_model_features_centre_over_the_window(harness):
    import numpy as np

    n = 30  # a 15 s film, audio stamped at bin centres
    vals = np.arange(n, dtype=float)[:, None]
    long = _long(np.arange(n) * 0.5 + 0.25, "m", vals).assign(stimulus_id="film-a")
    tr = 1.5
    rows = pd.DataFrame([{"stimulus_id": "film-a", "onset": 0.0, "start": 7, "n": 2, "repetition_time": tr}])
    feats, names = harness.film_model_features(long, rows, "m", set())
    f = feats["film-a"]
    on = harness.window_bins(np.array([7, 8]) * tr, tr, n)
    assert names == ["m_000"] and f.bands == {"m": slice(0, 1)}
    np.testing.assert_allclose(f.x[on, 0], vals[on, 0] - vals[on, 0].mean())
    assert not f.x[~on].any()


# --- rescue spaces (psytwill-space-next settle 2) ---------------------------------------------

RESCUE = HARNESS.with_name("rescue.py")


@pytest.fixture(scope="module")
def rescue():
    spec = importlib.util.spec_from_file_location("braintwill_encoding_rescue", RESCUE)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except (ImportError, SyntaxError) as exc:
        pytest.skip(f"braintwill.encoding not importable here: {exc}")
    return mod


def test_parse_rescue_names_a_block_and_an_optional_member(harness):
    assert harness.parse_rescue("rescue-V") == ("V", None)
    assert harness.parse_rescue("rescue-V+clip") == ("V", "clip")
    assert harness.parse_rescue("rescue-A+loudness") == ("A", "loudness")
    for bad in ("rescue-L", "rescue-V+", "rescue-V+loudness", "rescue-A+clip", "model-clip"):
        with pytest.raises(ValueError):
            harness.parse_rescue(bad)


def _film_row(start=7, n=2, tr=1.5):
    return pd.DataFrame([{"stimulus_id": "film-a", "onset": 0.0, "start": start, "n": n,
                          "repetition_time": tr}])


def test_film_rescue_base_is_the_same_band_with_or_without_a_member(harness):
    import numpy as np

    n = 30
    rng = np.random.default_rng(0)
    base = pd.DataFrame({"stimulus_id": "film-a", "time": np.arange(n) * 0.5,
                         "V_000": rng.standard_normal(n), "V_001": rng.standard_normal(n)})
    vals = np.arange(n, dtype=float)[:, None]
    long = _long(np.arange(n) * 0.5, "m", vals).assign(stimulus_id="film-a")
    rows = _film_row()
    alone, _ = harness.film_rescue_features(base, rows, None, None, set())
    both, names = harness.film_rescue_features(base, rows, long, "m", set())
    a, b = alone["film-a"], both["film-a"]
    assert a.bands == {"base": slice(0, 2)} and b.bands == {"base": slice(0, 2), "m": slice(2, 3)}
    np.testing.assert_array_equal(a.x, b.x[:, :2])
    on = harness.window_bins(np.array([7, 8]) * 1.5, 1.5, n)
    np.testing.assert_allclose(b.x[on, 2], vals[on, 0] - vals[on, 0].mean())
    np.testing.assert_allclose(b.x[on, :2].mean(axis=0), 0, atol=1e-6)  # float32 frames
    assert not b.x[~on].any() and names == ["m_000"]


def test_film_rescue_refuses_a_window_bin_without_a_row(harness):
    import numpy as np

    n = 30
    base = pd.DataFrame({"stimulus_id": "film-a", "time": np.arange(n) * 0.5, "V_000": np.arange(n, dtype=float)})
    on = harness.window_bins(np.array([7, 8]) * 1.5, 1.5, n)
    hole = int(np.flatnonzero(on)[1])
    with pytest.raises(RuntimeError, match="window bins have no row"):
        harness.film_rescue_features(base.drop(index=hole), _film_row(), None, None, set())
    long = _long(np.arange(n) * 0.5, "m", np.ones((n, 1))).assign(stimulus_id="film-a")
    with pytest.raises(RuntimeError, match="window bins have no row"):
        harness.film_rescue_features(base, _film_row(), long[long["time"] != hole * 0.5], "m", set())


def test_image_rescue_joins_raw_values_and_refuses_gaps(harness):
    import numpy as np

    base = pd.DataFrame({"stimulus_id": ["i1", "i2"], "V_000": [1.0, 2.0], "V_001": [3.0, 4.0]})
    long = pd.DataFrame({"stimulus_id": ["i2", "i1"], "model": "m", "feature": "f", "value": [20.0, 10.0]})
    got = harness.image_rescue_features(base, long, "m")
    assert got["i1"].bands == {"base": slice(0, 2), "m": slice(2, 3)}
    np.testing.assert_array_equal(got["i2"].x, [[2.0, 4.0, 20.0]])
    with pytest.raises(RuntimeError, match="differ"):
        harness.image_rescue_features(base, long.iloc[:1], "m")
    with pytest.raises(RuntimeError, match="undefined"):
        harness.image_rescue_features(base, long.assign(value=[np.nan, 1.0]), "m")


def test_holm_steps_down_and_stays_monotone(rescue):
    adj = rescue.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj == pytest.approx({"a": 0.03, "c": 0.06, "b": 0.06})
    assert rescue.holm({"a": 0.6, "b": 0.5}) == pytest.approx({"b": 1.0, "a": 1.0})


def _group_sums(res_scale, n_groups=30, rows=50, n_cols=6, seed=0):
    """Per-group sums of z-scored responses whose residual is ``res_scale`` x SS per column."""
    import numpy as np

    rng = np.random.default_rng(seed)
    y = rng.standard_normal((n_groups, rows, n_cols))
    p = y * (1 - np.sqrt(np.asarray(res_scale))) + 0.05 * rng.standard_normal(y.shape)
    return {"labels": [f"g{i}" for i in range(n_groups)], "n": np.full(n_groups, rows),
            "sy": y.sum(1), "syy": (y ** 2).sum(1), "res": ((y - p) ** 2).sum(1)}


def test_roi_deltas_see_a_gain_only_where_the_member_adds_one(rescue):
    import numpy as np

    base = _group_sums([0.9] * 6)
    member = _group_sums([0.8, 0.8, 0.8, 0.9, 0.9, 0.9])  # gain in columns 0-2 only
    rois = {"gain": np.arange(3), "none": np.arange(3, 6)}
    counts = rescue.draw_counts(30, 500, 0)
    assert (counts.sum(axis=1) == 30).all()
    d = rescue.roi_deltas(base, member, rois, counts)
    assert d["gain"]["delta"] > 0.05 and d["gain"]["p"] < 0.01
    assert abs(d["none"]["delta"]) < 1e-12 and d["none"]["p"] > 0.5
    with pytest.raises(RuntimeError, match="different CV groups"):
        rescue.roi_deltas(base, {**member, "labels": member["labels"][::-1]}, rois, counts)


def test_verdict_needs_two_subjects_and_the_floor_in_one_roi(rescue):
    def r(delta, p):
        return {"delta": delta, "p": p, "p_holm": p, "n_cols": 50}

    two = {"03": {"x": r(0.003, 0.001)}, "04": {"x": r(0.004, 0.01)}, "05": {"x": r(-0.001, 0.4)}}
    assert rescue.verdict(two)["rescued"] and rescue.verdict(two)["rescued_by"] == ["x"]
    small = {s: {"x": r(0.001, 0.001)} for s in ("03", "04", "05")}
    assert not rescue.verdict(small)["rescued"]  # reliable but under the .002 floor
    one = {"03": {"x": r(0.01, 0.001)}, "04": {"x": r(0.01, 0.2)}, "05": {"y": r(0.01, 0.001)}}
    assert not rescue.verdict(one)["rescued"]  # passes in one subject per ROI


def test_roi_columns_keep_masked_valid_columns_and_drop_small_rois(rescue):
    import numpy as np

    labels = np.array(["a"] * 5 + ["b"] * 2 + [""] * 3, dtype=object)
    mask = np.array([1, 1, 1, 0, 1, 1, 1, 1, 1, 1], bool)
    valid = np.array([1, 1, 1, 1, 0, 1, 1, 1, 1, 1], bool)
    got = rescue.roi_columns(labels, mask, valid, min_cols=3)
    assert list(got) == ["a"] and got["a"].tolist() == [0, 1, 2]
