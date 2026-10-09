"""data-quality voxel maps and alignment (voxel-quality): pure helpers on synthetic data."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

nib = pytest.importorskip("nibabel")

from neuroimaging import data_quality as dq  # noqa: E402
from neuroimaging import data_quality_alignment as dqa  # noqa: E402
from neuroimaging import data_quality_voxelmaps as dqv  # noqa: E402


# ---------------------------------------------------------------------------
# Aggregation and dropout
# ---------------------------------------------------------------------------

def test_nanmedian_stack_ignores_nan_and_keeps_all_nan():
    a = np.array([1.0, np.nan, np.nan])
    b = np.array([3.0, 2.0, np.nan])
    c = np.array([5.0, 4.0, np.nan])
    out = dqv.nanmedian_stack([a, b, c])
    assert out[0] == pytest.approx(3.0)
    assert out[1] == pytest.approx(3.0)
    assert np.isnan(out[2])


def test_relative_is_nan_where_subject_not_positive():
    out = dqv.relative(np.array([1.0, 2.0, 3.0, np.nan]), np.array([2.0, 0.0, -1.0, 1.0]))
    assert out[0] == pytest.approx(0.5)
    assert np.isnan(out[1:]).all()


def test_dropout_summary_separates_dropped_and_uncovered():
    rel = np.array([1.0, 0.7, 0.79, np.nan, 0.95])
    s = dqv.dropout_summary(rel, np.ones(5, bool))
    assert s["n"] == 5
    assert s["frac_dropped"] == pytest.approx(2 / 5)
    assert s["frac_uncovered"] == pytest.approx(1 / 5)
    assert s["frac_lost"] == pytest.approx(3 / 5)
    assert dqv.dropout_summary(rel, np.zeros(5, bool))["n"] == 0


def test_examine_flags_against_subject_median():
    frac = pd.Series([0.01, 0.01, 0.02, 0.05])
    assert dqv.examine_flags(frac).tolist() == [False, False, False, True]
    # A subject median of zero flags any session with a loss at all.
    assert dqv.examine_flags(pd.Series([0.0, 0.0, 0.0, 0.001])).tolist() == [False, False, False, True]


def test_run_scale_uses_domain_and_refuses_empty():
    mean = np.array([10.0, 20.0, 30.0, np.nan])
    assert dqv.run_scale(mean, np.array([True, True, False, True])) == pytest.approx(15.0)
    with pytest.raises(ValueError):
        dqv.run_scale(mean, np.array([False, False, False, True]))


# ---------------------------------------------------------------------------
# Ribbon sampling and grids
# ---------------------------------------------------------------------------

def _grid(shape=(10, 10, 10), zoom=2.0):
    aff = np.diag([zoom, zoom, zoom, 1.0])
    aff[:3, 3] = -zoom * (np.array(shape) - 1) / 2
    return aff


def test_ribbon_points_interpolate_white_to_pial():
    white = np.zeros((2, 3))
    pial = np.array([[1.0, 0, 0], [0, 2.0, 0]])
    pts = dqv.ribbon_points(white, pial)
    assert pts.shape == (len(dqv.DEPTHS), 2, 3)
    assert pts[-1] == pytest.approx(pial)
    assert pts[2, 1, 1] == pytest.approx(0.8)


def test_sample_volume_never_mixes_in_out_of_mask_voxels():
    shape = (10, 10, 10)
    aff = _grid(shape)
    vals = np.full(shape, 100.0)
    mask = np.zeros(shape, bool)
    mask[:5] = True
    vals[5:] = 1e6  # outside the mask: must never leak into a sample
    # A point exactly between voxel 4 (in) and 5 (out) along i.
    ijk = np.array([4.5, 4.0, 4.0, 1.0])
    world = (aff @ ijk)[:3]
    val, w = dqv.sample_volume(vals, mask, aff, world[None])
    assert val[0] == pytest.approx(100.0)
    assert w[0] == pytest.approx(0.5)
    # Fully outside: NaN value, zero weight.
    out = (aff @ np.array([8.0, 4, 4, 1]))[:3]
    val, w = dqv.sample_volume(vals, mask, aff, out[None])
    assert np.isnan(val[0]) and w[0] == 0


def test_ribbon_sample_inmask_fraction():
    shape = (10, 10, 10)
    aff = _grid(shape)
    vals = np.full(shape, 7.0)
    mask = np.zeros(shape, bool)
    mask[:5] = True
    white = (aff @ np.array([2.0, 4, 4, 1]))[:3][None]
    pial = (aff @ np.array([7.0, 4, 4, 1]))[:3][None]  # ribbon runs out of the mask
    v, frac = dqv.ribbon_sample(vals, mask, aff, dqv.ribbon_points(white, pial))
    assert v[0] == pytest.approx(7.0)
    assert 0.3 < frac[0] < 0.7


def test_onto_grid_identity_and_resample():
    shape = (8, 8, 8)
    aff = _grid(shape)
    vals = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    mask = np.ones(shape, bool)
    same, m = dqv.onto_grid(vals, mask, aff, shape, aff)
    assert same is vals and m is mask
    # A grid shifted by one whole voxel reproduces the values shifted by one.
    aff2 = aff.copy()
    aff2[:3, 3] += 2.0
    out, keep = dqv.onto_grid(vals, mask, aff, shape, aff2)
    assert out[0, 0, 0] == pytest.approx(vals[1, 1, 1])
    assert not keep[-1, -1, -1]
    assert np.isnan(out[-1, -1, -1])


def test_spearman_needs_three_pairs():
    r, n = dqv.spearman(np.array([1.0, 2, np.nan]), np.array([1.0, 2, 3]))
    assert np.isnan(r) and n == 2
    r, n = dqv.spearman(np.array([1.0, 2, 3, 4]), np.array([10.0, 20, 30, 50]))
    assert r == pytest.approx(1.0) and n == 4


def test_session_map_paths_are_distinct_and_bids_shaped():
    names = {dqv.session_map_path("/t", "04", "05", k, "T1w").name
             for k in ("mean", "tsnr_drift", "rel_mean", "rel_tsnr_drift", "dropped")}
    assert len(names) == 5
    assert "sub-04_ses-05_space-T1w_desc-drift_tsnr.nii.gz" in names
    assert dqv.session_map_path("/t", "04", None, "surfvol_mean", "fsnative", "L").name == \
        "sub-04_hemi-L_space-fsnative_desc-surfvolmean_ratio.func.gii"
    assert dqv.session_map_path("/t", "04", "05", "mean", dqv.MNI_SPACE).name == \
        f"sub-04_ses-05_space-{dqv.MNI_SPACE}_res-2_desc-norm_mean.nii.gz"


def test_cleaning_grid_collect_skips_voxelmaps_sidecars(tmp_path):
    d = tmp_path / "sub-04" / "ses-05" / "func"
    d.mkdir(parents=True)
    (d / "sub-04_ses-05_task-x_space-T1w_desc-drift_tsnr.json").write_text(
        json.dumps({"cell": dqv.CELL, "sub": "04"}))
    runs, _ = dq.collect(tmp_path)
    assert runs.empty


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------

def test_itk_affine_identity_and_translation():
    M = dqa.itk_affine_to_matrix(np.r_[np.eye(3).ravel(), [1.0, 2.0, 2.0]], np.array([10.0, -5, 3]))
    pts = np.array([[0.0, 0, 0], [50, 50, 50]])
    assert dqa.displacement(M, pts) == pytest.approx([3.0, 3.0])
    assert dqa.rotation_deg(M) == pytest.approx(0.0)


def test_itk_rotation_about_centre_leaves_centre_fixed():
    th = np.radians(2.0)
    R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    c = np.array([10.0, 20.0, 30.0])
    M = dqa.itk_affine_to_matrix(np.r_[R.ravel(), [0, 0, 0]], c)
    assert dqa.displacement(M, c[None])[0] == pytest.approx(0.0, abs=1e-9)
    # A point 50 mm from the axis moves 2 * 50 * sin(1 deg).
    assert dqa.displacement(M, (c + [50.0, 0, 0])[None])[0] == pytest.approx(100 * np.sin(th / 2))
    assert dqa.rotation_deg(M) == pytest.approx(2.0)


def test_ras_to_lps_flips_x_and_y():
    assert dqa.ras_to_lps(np.array([[1.0, 2.0, 3.0]])).tolist() == [[-1.0, -2.0, 3.0]]


def test_dice_and_edge_correlation():
    a = np.zeros((10, 10, 10), bool)
    a[2:8, 2:8, 2:8] = True
    assert dqa.dice(a, a) == pytest.approx(1.0)
    assert np.isnan(dqa.dice(np.zeros_like(a), np.zeros_like(a)))
    rng = np.random.default_rng(0)
    img = rng.normal(size=(20, 20, 20)).cumsum(axis=0)
    mask = np.ones(img.shape, bool)
    assert dqa.edge_correlation(img, img, mask, np.ones(3)) == pytest.approx(1.0)
    shifted = np.roll(img, 2, axis=1)
    assert dqa.edge_correlation(img, shifted, mask, np.ones(3)) < 0.99


def test_loso_template_excludes_the_left_out_session():
    means = {"01": np.full(4, 1.0), "02": np.full(4, 2.0), "03": np.full(4, 100.0)}
    masks = {"01": np.array([1, 1, 0, 0], bool), "02": np.array([1, 0, 1, 0], bool),
             "03": np.array([1, 1, 1, 1], bool)}
    t, m = dqa.loso_template(means, masks, "03")
    assert t == pytest.approx(np.full(4, 1.5))
    assert m.tolist() == [True, True, True, False]
    with pytest.raises(ValueError):
        dqa.loso_template({"01": means["01"]}, {"01": masks["01"]}, "01")


# ---------------------------------------------------------------------------
# Tier 2
# ---------------------------------------------------------------------------

from neuroimaging import data_quality_voxelquality as dqvq  # noqa: E402


def test_parcel_sessions_counts_sessions_mostly_lost():
    parcels = pd.DataFrame({
        "sub": ["04"] * 4, "ses": ["01", "02", "03", None], "space": ["T1w"] * 4, "hemi": [None] * 4,
        "atlas": ["HOSPA"] * 4, "parcel": ["Left Amygdala"] * 4, "frac_lost": [0.1, 0.6, 0.9, np.nan]})
    out = dqvq.parcel_sessions(parcels)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["n_sessions"] == 3 and row["n_sessions_lost"] == 2
    assert row["sessions_lost"] == "02,03"


def test_verdict_reads_adequacy_and_examined_sessions():
    sessions = pd.DataFrame({"sub": ["04"] * 3, "ses": ["01", "02", "01"], "space": ["T1w", "T1w", "fsnative"],
                             "hemi": [None, None, "L"], "frac_lost": [0.01, 0.2, 0.0],
                             "examine": [False, True, False]})
    surfvol = pd.DataFrame({"sub": ["04", "04"], "hemi": ["L", "R"], "surfvol_mean_median": [1.0, 1.02],
                            "rho_parcel_sampled": [0.97, 0.95], "rho_parcel_ribbon": [0.9, 0.9],
                            "frac_ribbon_below_floor": [0.01, 0.02], "adequate": [True, False]})
    v = dqvq.verdict(sessions, surfvol, None).iloc[0]
    assert v["sessions_examine_T1w"] == "02"
    assert v["sessions_examine_fsnative"] == "n/a"
    assert not v["surface_adequate"]
