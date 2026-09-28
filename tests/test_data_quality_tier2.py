"""data-quality tier 2: film viewings, windows, envelope binning, lag scan, LOO-ISFC, build + diff."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neuroimaging import data_quality_tier2 as t2

TR = 1.5
N_VOL = 120
SPACE = "MNI152NLin2009cAsym_res-2"
PARCELS = ["17Networks_LH_VisCent_ExStr_1", "17Networks_LH_SomMotB_Aud_1",
           "17Networks_LH_SomMotB_Aud_2", "17Networks_RH_SomMotB_S2_1"]


# ---------------------------------------------------------------------------
# Small pieces
# ---------------------------------------------------------------------------

def test_window_keeps_only_volumes_wholly_inside_the_film():
    # film 15.48 s .. 45.48 s at TR 1.5: volume 10 starts at 15.0 (partly before), 11 at 16.5;
    # volume 29 ends at 45.0, volume 30 ends at 46.5 (partly after)
    assert t2.window(15.48, 30.0, TR, 100) == (11, 19)
    assert t2.window(15.0, 30.0, TR, 100) == (10, 20)       # exact boundaries count as inside
    assert t2.window(15.0, 300.0, TR, 100) == (10, 90)      # clipped at the run's end


def test_envelope_bins_frames_by_film_time_onto_window_volumes():
    frames = pd.DataFrame({"time": np.arange(0.25, 30, 0.5)})
    frames["loudness_rms"] = frames["time"]                 # value = its own centre time
    onset, start, n = 1.2, 1, 4                            # volume 1 spans film time 0.3 .. 1.8
    env = t2.envelope_on_volumes(frames, "loudness_rms", onset, start, n, TR)
    assert env[0] == pytest.approx(np.mean([0.75, 1.25, 1.75]))
    assert env[1] == pytest.approx(np.mean([2.25, 2.75, 3.25]))
    assert len(env) == n


def test_lag_sign_negative_means_audio_leads_bold():
    rng = np.random.default_rng(0)
    env = rng.normal(size=200)
    bold = np.r_[rng.normal(size=3), env[:-3]]            # BOLD = audio 3 volumes later
    curve = t2.lag_curves(bold, env, max_lag=6)[:, 0]
    lags = np.arange(-6, 7)
    assert lags[np.nanargmax(curve)] == -3
    assert curve[lags == -3][0] == pytest.approx(1.0)


def test_lag_curves_are_nan_for_a_parcel_with_a_missing_value():
    rng = np.random.default_rng(1)
    env = rng.normal(size=60)
    bold = rng.normal(size=(60, 2))
    bold[5, 1] = np.nan
    curves = t2.lag_curves(bold, env, max_lag=2)
    assert np.isfinite(curves[:, 0]).all()
    assert np.isnan(curves[:, 1]).all()


def test_loo_isfc_diagonal_is_loo_isc_and_matrix_is_asymmetric():
    rng = np.random.default_rng(2)
    shared = rng.normal(size=(80, 5))
    segs = np.stack([shared + rng.normal(size=(80, 5)) for _ in range(3)])
    m = t2.loo_isfc(segs)
    others = segs[1:].mean(axis=0)
    for p in range(5):
        assert m[0, p, p] == pytest.approx(np.corrcoef(segs[0][:, p], others[:, p])[0, 1])
    assert not np.allclose(m[0], m[0].T)


def test_loo_isfc_nan_parcel_gives_nan_row_and_column_never_a_partial_mean():
    rng = np.random.default_rng(3)
    segs = rng.normal(size=(3, 50, 4))
    segs[1, 10, 2] = np.nan
    m = t2.loo_isfc(segs)
    assert np.isnan(m[1, 2, :]).all()                       # sub 1's own parcel 2
    assert np.isnan(m[0, :, 2]).all() and np.isnan(m[2, :, 2]).all()   # others' mean parcel 2
    assert np.isfinite(m[0, :, :2]).all()


def test_loo_isfc_needs_three_subjects():
    with pytest.raises(ValueError, match="at least 3"):
        t2.loo_isfc(np.zeros((2, 10, 3)))


def test_movie_name_index_takes_every_spelling_casefolded(tmp_path):
    reg = tmp_path / "movies.tsv"
    reg.write_text("stimulus_id\tmovie_name\tmovie_name_variants\n"
                   "dad-to-son\tFrom Dad To Son\tFrom Dad to Son\n"
                   "bench\tThe Bench\t\n")
    idx = t2.movie_name_index(reg)
    assert idx["from dad to son"] == "dad-to-son"
    assert idx["the bench"] == "bench"


def test_movie_name_index_rejects_a_spelling_claimed_by_two_films(tmp_path):
    reg = tmp_path / "movies.tsv"
    reg.write_text("stimulus_id\tmovie_name\tmovie_name_variants\na\tSame\t\nb\tOther\tsame\n")
    with pytest.raises(ValueError, match="maps to both"):
        t2.movie_name_index(reg)


# ---------------------------------------------------------------------------
# Synthetic tree: 3 subjects x 2 sessions; film A shown in both sessions, film B once
# ---------------------------------------------------------------------------

REGIMES = ("none", "base")


def _events(films: list[tuple[str, float, float]]) -> pd.DataFrame:
    rows = []
    for name, onset, dur in films:
        rows.append({"onset": onset - 3, "duration": 3.0, "trial_type": "title", "movie_name": ""})
        rows.append({"onset": onset, "duration": dur, "trial_type": "movie", "movie_name": name})
    return pd.DataFrame(rows)


@pytest.fixture
def synthetic(tmp_path):
    rng = np.random.default_rng(11)
    bids, tree = tmp_path / "bids", tmp_path / "tree"
    reg = tmp_path / "movies.tsv"
    reg.write_text("stimulus_id\tmovie_name\tmovie_name_variants\n"
                   "film-a\tFilm A\tFILM A\nfilm-b\tFilm B\t\n")
    frames = {sid: pd.DataFrame({"time": np.arange(0.25, 80, 0.5)}) for sid in ("film-a", "film-b")}
    for f in frames.values():
        f["loudness_rms"] = np.abs(rng.normal(size=len(f))) + 0.1
        f["loudness_db"] = 20 * np.log10(f["loudness_rms"])
    shared = rng.normal(size=(N_VOL, len(PARCELS)))
    rows = []
    # sub-03's first session shows Film A with its own spelling
    layout = {"19": [("Film A", 16.0, 60.0), ("Film B", 90.0, 60.0)], "20": [("FILM A", 20.0, 60.0)]}
    for sub in ("03", "04", "05"):
        for ses, films in layout.items():
            func = bids / f"sub-{sub}" / f"ses-{ses}" / "func"
            func.mkdir(parents=True)
            _events(films).to_csv(func / f"sub-{sub}_ses-{ses}_task-NATencoding_run-01_events.tsv",
                                  sep="\t", index=False)
            out = tree / f"sub-{sub}" / f"ses-{ses}" / "func"
            out.mkdir(parents=True, exist_ok=True)
            for regime in REGIMES:
                absent = sub == "05" and ses == "19" and regime == "base"
                rows.append({"sub": sub, "ses": ses, "task": "NATencoding", "run": "01", "space": SPACE,
                             "regime": regime, "n_vol": str(N_VOL), "repetition_time": str(TR),
                             "absent": "True" if absent else "False"})
                if absent:
                    continue
                ts = pd.DataFrame(shared + rng.normal(size=shared.shape), columns=PARCELS)
                ts.iloc[0] = np.nan                         # a non-steady-state volume, outside every window
                name = (f"sub-{sub}_ses-{ses}_task-NATencoding_run-01_space-{SPACE}"
                        f"_seg-{t2.SEG}_desc-{regime}_timeseries.tsv")
                ts.to_csv(out / name, sep="\t", index=False, na_rep="n/a")
    pd.DataFrame(rows).to_csv(tree / "tier1_runs.tsv", sep="\t", index=False)

    def events_path(sub, ses, run):
        p = bids / f"sub-{sub}" / f"ses-{ses}" / "func" / f"sub-{sub}_ses-{ses}_task-NATencoding_run-{run}_events.tsv"
        return p if p.exists() else None

    return {"tree": tree, "registry": reg, "frames": frames, "events_path": events_path}


def test_film_viewings_flags_the_first_showing_across_sessions(synthetic):
    runs = t2.load_tier1_runs(synthetic["tree"])
    v = t2.film_viewings(runs, synthetic["events_path"], t2.movie_name_index(synthetic["registry"]))
    a = v[(v["sub"] == "03") & (v["stimulus_id"] == "film-a")]
    assert list(a["ses"]) == ["19", "20"]
    assert list(a["first_viewing"]) == [True, False]
    assert v["first_viewing"].sum() == 6                   # 3 subjects x 2 films


def test_film_viewings_is_loud_about_a_missing_events_file(synthetic):
    runs = t2.load_tier1_runs(synthetic["tree"])
    with pytest.raises(FileNotFoundError, match="No events file"):
        t2.film_viewings(runs, lambda s, e, r: None, t2.movie_name_index(synthetic["registry"]))


def _build(synthetic, dest):
    runs = t2.load_tier1_runs(synthetic["tree"])
    v = t2.film_viewings(runs, synthetic["events_path"], t2.movie_name_index(synthetic["registry"]))
    result = t2.compute(synthetic["tree"], v, synthetic["frames"], list(REGIMES), runs)
    t2.write(result, dest, {"test": True})
    return result


def test_build_skips_absent_cells_and_needs_three_subjects_for_isfc(synthetic, tmp_path):
    result = _build(synthetic, tmp_path / "out")
    # sub-05 ses-19 is absent under `base`: both films drop to 2 subjects there
    assert set(result.isfc_group) == {("film-a", "none"), ("film-b", "none")}
    reasons = {(s["stimulus_id"], s["regime"], s["reason"].split(",")[0]) for s in result.skipped}
    assert ("film-a", "base", "ISFC needs >= 3 subjects") in reasons
    assert ("film-a", "base", "tier-1 cell declared absent") in reasons
    # the envelope scan still runs for the subjects that have the cell
    lag = result.envelope_lag
    assert set(lag.loc[lag["regime"] == "base", "sub"]) == {"03", "04"}
    assert (lag["n_parcels"] == 2).all()                    # the two LH _Aud_ parcels
    assert len(lag) == 6 * 2 * 41 + 4 * 2 * 41             # showings x envelopes x lags, per regime
    par = result.envelope_parcel
    assert par["lag_best"].dropna().between(*t2.PLAUSIBLE_LAGS).all()


def test_rebuild_is_identical_and_diff_catches_a_change(synthetic, tmp_path):
    _build(synthetic, tmp_path / "a")
    _build(synthetic, tmp_path / "b")
    assert t2.diff(tmp_path / "a", tmp_path / "b") == []
    isc = tmp_path / "b" / "loo_isc.tsv"
    isc.write_text(isc.read_text().replace("\t0.", "\t1.", 1))
    assert t2.diff(tmp_path / "a", tmp_path / "b") == ["loo_isc.tsv differs"]


# ---------------------------------------------------------------------------
# Schaefer names: tier-1 columns carry TemplateFlow's pre-rename names
# ---------------------------------------------------------------------------

def _tables(atlases, new_names, colors_new=None):
    old = pd.DataFrame({"index": [1, 2, 3, 4], "name": PARCELS, "color": ["#01", "#02", "#03", "#04"]})
    new = pd.DataFrame({"index": [1, 2, 3, 4], "name": new_names,
                        "color": colors_new or ["#01", "#02", "#03", "#04"]})
    for rel, df in ((t2.SCHAEFER_TIER1_TABLE, old), (t2.SCHAEFER_CURRENT_TABLE, new)):
        path = atlases / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, sep="\t", index=False)


CURRENT = ["17Networks_LH_VisCent_ExStr_1", "17Networks_LH_SomMotB_Aud_1",
           "17Networks_LH_SomMotB_Ins_1", "17Networks_RH_SomMotB_Aud_1"]


def test_schaefer_current_names_maps_by_index(tmp_path):
    _tables(tmp_path, CURRENT)
    m = t2.schaefer_current_names(tmp_path)
    assert m["17Networks_LH_SomMotB_Aud_2"] == "17Networks_LH_SomMotB_Ins_1"
    assert m["17Networks_RH_SomMotB_S2_1"] == "17Networks_RH_SomMotB_Aud_1"


def test_schaefer_current_names_refuses_when_a_colour_disagrees(tmp_path):
    _tables(tmp_path, CURRENT, colors_new=["#01", "#02", "#99", "#04"])
    with pytest.raises(ValueError, match="colour"):
        t2.schaefer_current_names(tmp_path)


def test_build_selects_the_auditory_roi_by_current_names(synthetic, tmp_path):
    _tables(tmp_path, CURRENT)
    rename = t2.schaefer_current_names(tmp_path)
    runs = t2.load_tier1_runs(synthetic["tree"])
    v = t2.film_viewings(runs, synthetic["events_path"], t2.movie_name_index(synthetic["registry"]))
    result = t2.compute(synthetic["tree"], v, synthetic["frames"], ["none"], runs, rename)
    assert set(result.envelope_parcel["parcel"]) == set(CURRENT)
    assert set(result.loo_isc["parcel"]) == set(CURRENT)
    assert (result.envelope_lag["n_parcels"] == 2).all()    # LH Aud_1 + RH Aud_1, not LH Ins_1


def test_load_series_is_loud_about_a_column_the_name_map_lacks(synthetic):
    path = next(synthetic["tree"].rglob("*_timeseries.tsv"))
    with pytest.raises(KeyError, match="not in the Schaefer name map"):
        t2.load_series(path, N_VOL, rename={"only_one": "x"})


def test_covered_prefix_accepts_a_trailing_truncation_only():
    assert t2.covered_prefix(np.array([1.0, 2.0, 3.0])) == 3
    assert t2.covered_prefix(np.array([1.0, 2.0, np.nan, np.nan])) == 2
    with pytest.raises(ValueError, match="not a trailing truncation"):
        t2.covered_prefix(np.array([1.0, np.nan, 3.0]))


def test_truncated_stimulus_scans_the_covered_stretch_and_records_it(synthetic, tmp_path):
    frames = dict(synthetic["frames"])
    frames["film-b"] = frames["film-b"][frames["film-b"]["time"] < 30]   # file stops 30 s into a 60 s showing
    runs = t2.load_tier1_runs(synthetic["tree"])
    v = t2.film_viewings(runs, synthetic["events_path"], t2.movie_name_index(synthetic["registry"]))
    result = t2.compute(synthetic["tree"], v, frames, ["none"], runs)
    b = result.envelope_lag[result.envelope_lag["stimulus_id"] == "film-b"]
    assert (b["n_vol"] < b["n_window"]).all() and (b["n_vol"] == 20).all()   # 90 s onset = volume 60 exactly: 30 s = 20 volumes
    a = result.envelope_lag[result.envelope_lag["stimulus_id"] == "film-a"]
    assert (a["n_vol"] == a["n_window"]).all()
