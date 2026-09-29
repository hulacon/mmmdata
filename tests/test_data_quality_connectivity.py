"""data-quality tier 2 connectivity: FC, within-subject QC-FC, fingerprinting, hippocampal profile, build + diff."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from neuroimaging import data_quality_connectivity as dqc

SPACE = "MNI152NLin2009cAsym_res-2"
PARCELS = [f"17Networks_LH_Net_{i}" for i in range(1, 7)]
HOSPA = ["Left Thalamus", "Left Hippocampus", "Right Hippocampus"]
N_VOL = 80


# ---------------------------------------------------------------------------
# Small pieces
# ---------------------------------------------------------------------------

def test_fc_vector_is_the_upper_triangle_in_row_major_order():
    rng = np.random.default_rng(0)
    a = rng.normal(size=200)
    x = np.c_[a, a + 0.1 * rng.normal(size=200), rng.normal(size=200)]
    z = dqc.fc_vector(x)
    assert z.shape == (3,)                            # (0,1), (0,2), (1,2)
    assert z[0] > 2 and abs(z[1]) < 0.3 and abs(z[2]) < 0.3


def test_fc_vector_is_nan_for_a_parcel_with_a_missing_value():
    x = np.random.default_rng(1).normal(size=(50, 3))
    x[4, 2] = np.nan
    z = dqc.fc_vector(x)
    assert np.isfinite(z[0]) and np.isnan(z[1]) and np.isnan(z[2])


def test_qcfc_is_within_subject():
    rng = np.random.default_rng(2)
    groups = np.repeat(["03", "04", "05"], 20)
    offset = np.repeat([0.0, 1.0, 2.0], 20)            # one subject moves more AND has stronger edge 0
    fd_within = rng.normal(size=60)
    fd = offset + 0.3 * fd_within
    fc = np.c_[offset + 0.3 * rng.normal(size=60),     # edge 0: only the between-subject confound
               0.5 * fd_within + 0.3 * rng.normal(size=60)]   # edge 1: real within-subject motion coupling
    naive = np.corrcoef(fc[:, 0], fd)[0, 1]
    r, p, dof = dqc.qcfc(fc, fd, groups)
    assert naive > 0.8                                 # pooled across runs, edge 0 looks motion-driven
    assert abs(r[0]) < 0.35                            # within subject it is not
    assert r[1] > 0.7 and p[1] < 1e-6
    assert dof == 60 - 3 - 1


def test_qcfc_summary_distance_dependence_sign():
    distance = np.linspace(10, 100, 50)
    r = -distance / 100 + 0.01 * np.random.default_rng(3).normal(size=50)
    out = dqc.qcfc_summary(r, np.full(50, 0.5), distance)
    assert out["distance_dependence"] < -0.9 and out["frac_p05"] == 0


def test_session_fingerprint_finds_own_half_and_falls_to_chance_without_one():
    rng = np.random.default_rng(4)
    run_sig = rng.normal(size=(12, 40))
    ident = dqc.identify_sessions(run_sig + 0.3 * rng.normal(size=(12, 40)),
                                  run_sig + 0.3 * rng.normal(size=(12, 40)))
    assert ident["correct"].all() and (ident["margin"] > 0).all()
    assert len(ident) == 24                            # both directions
    noise = dqc.identify_sessions(rng.normal(size=(12, 40)), rng.normal(size=(12, 40)))
    assert noise["correct"].mean() < 0.5


def test_subject_fingerprint_needs_a_same_subject_run_and_another_subject():
    rng = np.random.default_rng(5)
    base = rng.normal(size=(3, 30))
    subjects = np.repeat(["03", "04", "05"], 4)
    fc = base[np.repeat([0, 1, 2], 4)] + 0.3 * rng.normal(size=(12, 30))
    ident = dqc.identify_subjects(fc, subjects)
    assert len(ident) == 12 and ident["correct"].all()


# ---------------------------------------------------------------------------
# Synthetic tier-1 tree
# ---------------------------------------------------------------------------

def _write_series(tree, sub, ses, regime, schaefer, hospa, weights):
    d = tree / f"sub-{sub}" / f"ses-{ses}" / "func"
    d.mkdir(parents=True, exist_ok=True)
    stem = f"sub-{sub}_ses-{ses}_task-TBresting_space-{SPACE}"
    pd.DataFrame(schaefer, columns=PARCELS).to_csv(
        d / f"{stem}_seg-Schaefer17n400_desc-{regime}_timeseries.tsv", sep="\t", index=False, na_rep="n/a")
    hp = d / f"{stem}_seg-HOSPA_desc-{regime}_timeseries.tsv"
    pd.DataFrame(hospa, columns=HOSPA).to_csv(hp, sep="\t", index=False, na_rep="n/a")
    parcels = {n: {"n_voxels_mask": w} for n, w in zip(HOSPA, weights)}
    hp.with_suffix(".json").write_text(json.dumps({"parcels": parcels}))


@pytest.fixture
def synthetic(tmp_path):
    """3 subjects x 12 sessions (+ a 3-session provisional one), one TBresting run each, two regimes."""
    rng = np.random.default_rng(10)
    tree = tmp_path / "tree"
    mixing = {s: rng.normal(size=(6, 6)) for s in ("03", "04", "05", "06")}
    rows, motion = [], []
    for sub, n_ses in (("03", 12), ("04", 12), ("05", 12), ("06", 3)):
        for k in range(1, n_ses + 1):
            ses = f"{k:02d}"
            n_nss = 2 if (sub, k) == ("03", 1) else 0
            latent = rng.normal(size=(N_VOL, 6)) @ mixing[sub]
            hip = latent[:, :2].mean(axis=1, keepdims=True) + 0.5 * rng.normal(size=(N_VOL, 2))
            for regime, noise in (("none", 1.0), ("gsr", 0.3)):
                x = latent + noise * rng.normal(size=(N_VOL, 6))
                h = np.c_[rng.normal(size=N_VOL), hip]
                x[:n_nss], h[:n_nss] = np.nan, np.nan
                _write_series(tree, sub, ses, regime, x, h, [100, 300, 100])
            rows.append({"sub": sub, "ses": ses, "task": "TBresting", "run": "n/a", "space": SPACE,
                         "n_vol": str(N_VOL), "n_nss": str(n_nss)})
            motion.append({"sub": sub, "ses": ses, "task": "TBresting", "run": "",
                           "fd_mean": rng.uniform(0.05, 0.3), "fdf_mean": rng.uniform(0.03, 0.1)})
    centroids = pd.DataFrame({"index": range(1, 7), "name": PARCELS,
                              "x": np.arange(6) * 10.0, "y": 0.0, "z": 0.0})
    return {"tree": tree, "tier1": pd.DataFrame(rows), "motion": pd.DataFrame(motion), "centroids": centroids}


def _compute(s, regimes=("none", "gsr"), absent=frozenset(), provisional=("06",)):
    runs = dqc.rest_runs(s["tier1"], s["motion"], provisional=provisional)
    return dqc.compute(s["tree"], runs, list(regimes), s["centroids"], set(absent))


def test_rest_runs_is_loud_about_a_run_without_motion(synthetic):
    with pytest.raises(KeyError, match="no tier1_motion.tsv row"):
        dqc.rest_runs(synthetic["tier1"], synthetic["motion"].iloc[1:])


def test_compute_shapes_scopes_and_provisional_marks(synthetic):
    res = _compute(synthetic)
    assert set(res.fc) == {"none", "gsr"} and res.fc["none"].shape == (39, 15)
    scopes = set(res.qcfc["scope"])
    assert scopes == {"sub-03", "sub-04", "sub-05", "pooled", "pooled_confirmed"}   # sub-06 has 3 runs
    pooled = res.qcfc[res.qcfc["scope"] == "pooled"]
    assert pooled["provisional"].all() and len(pooled) == 2 * 2                    # regimes x FD kinds
    assert not res.qcfc.loc[res.qcfc["scope"] == "pooled_confirmed", "provisional"].any()
    assert (pooled["dof"] == 39 - 4 - 1).all()
    fp = res.fingerprint.set_index(["regime", "test", "scope"])
    assert fp.loc[("gsr", "subject", "pooled"), "id_rate"] == 1.0                   # distinct mixing per subject
    assert fp.loc[("gsr", "session", "sub-03"), "chance"] == pytest.approx(1 / 12)


def test_hippocampal_profile_pools_both_sides_by_voxel_weight(synthetic):
    res = _compute(synthetic, regimes=("gsr",))
    prof = res.hipp_profile
    assert set(prof["seed"]) == {"left", "right", "both"}
    assert set(prof["parcel"]) == set(PARCELS)
    assert (prof.groupby(["sub", "seed"]).size() == 6).all()
    assert (prof.loc[prof["sub"] == "03", "n_runs"] == 12).all()


def test_hippocampal_similarity_pairs_regimes(synthetic):
    res = _compute(synthetic)
    sim = res.hipp_similarity
    assert set(zip(sim["regime_a"], sim["regime_b"])) == {("gsr", "none")}
    assert len(sim) == 4 * 3                                                        # subjects x seeds


def test_a_regime_with_an_absent_rest_run_is_skipped_whole(synthetic):
    res = _compute(synthetic, absent={("04", "02", "TBresting", "n/a", "gsr")})
    assert set(res.fc) == {"none"}
    assert res.skipped[0]["regime"] == "gsr"


def test_nss_rows_are_dropped_and_halves_are_equal(synthetic):
    res = _compute(synthetic, regimes=("none",))
    assert np.isfinite(res.fc["none"]).all()       # the 2 NSS rows of sub-03 ses-01 did not leak in


def test_rebuild_is_identical_and_diff_catches_a_change(synthetic, tmp_path):
    for d in ("a", "b"):
        dqc.write(_compute(synthetic), tmp_path / d, {"test": True})
    assert dqc.diff(tmp_path / "a", tmp_path / "b") == []
    np.save(tmp_path / "b" / "fc" / "seg-Schaefer17n400_desc-gsr_fc.npy", np.zeros((39, 15), np.float32))
    assert dqc.diff(tmp_path / "a", tmp_path / "b") == ["fc/seg-Schaefer17n400_desc-gsr_fc.npy differs"]
