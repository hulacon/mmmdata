"""data-quality GLMsingle-native rows: NSD ncsnr (T2.12), fit QC (T2.11), the tier-2 part."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

nib = pytest.importorskip("nibabel")

from neuroimaging import data_quality as dq  # noqa: E402
from neuroimaging import data_quality_glmsingle as dqgs  # noqa: E402

SPACE = "MNI152NLin2009cAsym_res-2"


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------

def _repeats(rng, n_vox, n_cond, n_rep, signal_sd, noise_sd):
    """(voxels, trials) betas: a per-condition signal repeated n_rep times plus iid noise."""
    sig = rng.normal(scale=signal_sd, size=(n_vox, n_cond))
    betas = np.repeat(sig, n_rep, axis=1) + rng.normal(scale=noise_sd, size=(n_vox, n_cond * n_rep))
    groups = np.arange(n_cond * n_rep).reshape(n_cond, n_rep)
    return betas, groups


def test_ncsnr_recovers_signal_to_noise_ratio():
    rng = np.random.default_rng(0)
    betas, groups = _repeats(rng, n_vox=200, n_cond=600, n_rep=3, signal_sd=1.0, noise_sd=2.0)
    z = dqgs.zscore_by_session(betas, np.zeros(betas.shape[1]))
    snr, noise = dqgs.ncsnr(z, groups, chunk=37)
    assert np.median(snr) == pytest.approx(0.5, abs=0.03)
    assert np.median(noise) == pytest.approx(0.8, abs=0.02)  # 4 / (1 + 4)


def test_ncsnr_is_zero_without_signal_and_ignores_session_scale():
    rng = np.random.default_rng(1)
    betas, groups = _repeats(rng, n_vox=100, n_cond=400, n_rep=2, signal_sd=0.0, noise_sd=1.0)
    snr0, _ = dqgs.ncsnr(dqgs.zscore_by_session(betas, np.zeros(betas.shape[1])), groups)
    assert np.median(snr0) < 0.1
    # A session-wise offset and gain on every voxel does not change the per-session z-scores.
    betas, groups = _repeats(rng, n_vox=50, n_cond=300, n_rep=2, signal_sd=1.0, noise_sd=1.0)
    ses = np.repeat(["a", "b"], betas.shape[1] // 2)
    shifted = betas.copy()
    shifted[:, ses == "b"] = 5.0 + 3.0 * shifted[:, ses == "b"]
    a = dqgs.ncsnr(dqgs.zscore_by_session(betas, ses), groups)[0]
    b = dqgs.ncsnr(dqgs.zscore_by_session(shifted, ses), groups)[0]
    np.testing.assert_allclose(a, b, rtol=1e-4)


def test_zscore_by_session_zero_variance_is_nan_and_single_trial_refused():
    z = dqgs.zscore_by_session(np.array([[1.0, 1.0, 2.0, 4.0]]), np.array(["a", "a", "b", "b"]))
    assert np.isnan(z[0, :2]).all() and np.isfinite(z[0, 2:]).all()
    with pytest.raises(ValueError, match="at least 2"):
        dqgs.zscore_by_session(np.ones((1, 3)), np.array(["a", "a", "b"]))


def test_noise_ceiling_values():
    np.testing.assert_allclose(dqgs.noise_ceiling(np.array([1.0, 0.0]), 1), [50.0, 0.0])
    np.testing.assert_allclose(dqgs.noise_ceiling(np.array([1.0]), 3), [75.0])


def test_shuffle_null_sits_at_its_level_without_signal():
    # The NSD estimator's positive floor needs both heavy tails and a session z-score taken over
    # many more trials than the repeats (as in the real arms: 874 singletons beside 120 pairs):
    # the repeats' mean square then has median < 1, because outlier trials mostly fall outside them.
    rng = np.random.default_rng(3)
    betas, groups = _repeats(rng, n_vox=400, n_cond=120, n_rep=2, signal_sd=0.0, noise_sd=1.0)
    betas = np.hstack([betas, rng.normal(size=(400, 900))])
    betas = rng.standard_t(3, size=betas.shape)  # no signal at all, heavy tails
    block = np.repeat(["r1", "r2", "r3"], [120, 120, 900])
    groups = np.stack([np.arange(120), np.arange(120, 240)], axis=1)
    z = dqgs.zscore_by_session(betas, np.zeros(betas.shape[1]))
    matched = dqgs.ncsnr(z, groups)[0]
    null = dqgs.ncsnr_null(z, groups, block, 19, seed=0)
    assert np.median(matched) > 0.02  # the floor the null exists to measure
    exceed = np.mean(matched > null.max(axis=0))
    assert exceed < 0.12  # null level 1/20
    assert dqgs.null_seed("a", "B") == dqgs.null_seed("a", "B") != dqgs.null_seed("a", "C")


def test_shuffle_groups_keeps_runs_and_first_presentation():
    groups = np.arange(12).reshape(6, 2)
    block = np.array(["a"] * 6 + ["b"] * 3 + ["c"] * 3)
    g = dqgs.shuffle_groups(groups, block, np.random.default_rng(0))
    assert (g[:, 0] == groups[:, 0]).all()
    assert (block[g[:, 1]] == block[groups[:, 1]]).all() and sorted(g[:, 1]) == sorted(groups[:, 1])


def test_ncsnr_refuses_single_repeat_groups():
    with pytest.raises(ValueError, match="n >= 2"):
        dqgs.ncsnr(np.zeros((2, 4)), np.arange(4).reshape(4, 1))


def _trial_info(scope="within"):
    """Two anchors x 4 presentations over 2 sessions; 3 items x 2 repeats; 2 singles."""
    rows = []
    for rep in range(4):
        for a in ("A1", "A2"):
            rows.append({"session": f"ses-0{1 + rep // 2}", "run": "run-01", "task": "TBencoding",
                         "condition_id": a, "sharedId": 1})
    for c in ("r1", "r2", "r3"):
        for rep in range(2):
            ses = "ses-01" if scope == "within" else f"ses-0{1 + rep}"
            rows.append({"session": ses, "run": f"run-0{1 + rep}", "task": "TBencoding",
                         "condition_id": c, "sharedId": 0})
    for c in ("s1", "s2"):
        rows.append({"session": "ses-02", "run": "run-01", "task": "TBencoding", "condition_id": c, "sharedId": 0})
    return pd.DataFrame(rows)


def test_condition_sets_split_anchors_and_record_scope():
    sets = dqgs.condition_sets(_trial_info("within"))
    assert sets["anchor"]["n_conditions"] == 2 and sets["anchor"]["n_reps"] == 4
    assert sets["anchor"]["rep_scope"] == "cross-session"
    assert sets["repeat"]["n_conditions"] == 3 and sets["repeat"]["n_reps"] == 2
    assert sets["repeat"]["rep_scope"] == "within-session"
    assert sets["repeat"]["frac_conditions_one_run"] == 0.0
    ti = _trial_info("within")
    for g in sets["repeat"]["groups"]:
        assert ti.loc[g, "condition_id"].nunique() == 1
    assert dqgs.condition_sets(_trial_info("cross"))["repeat"]["rep_scope"] == "cross-session"


def test_condition_sets_refuse_mixed_repeat_counts():
    ti = _trial_info()
    ti = pd.concat([ti, ti[ti["condition_id"] == "r1"].iloc[:1]], ignore_index=True)
    with pytest.raises(ValueError, match="mixed repeat counts"):
        dqgs.condition_sets(ti)


def test_meanvol_floor():
    keep = dqgs.meanvol_floor(np.array([100.0, 100.0, 100.0, 20.0, np.nan]))
    assert keep.tolist() == [True, True, True, False, False]


def test_region_of():
    assert dqgs.region_of("Schaefer17n400", "17Networks_LH_VisCent_ExStr_1") == "LH_VisCent"
    assert dqgs.region_of("HOSPA", "Left Hippocampus ") == "Left Hippocampus"
    with pytest.raises(ValueError):
        dqgs.region_of("Schaefer17n400", "Background")


# ---------------------------------------------------------------------------
# A fake fit end to end: tier-1 cell, collect, tier-2 tables
# ---------------------------------------------------------------------------

SHAPE = (6, 6, 4)


@pytest.fixture
def atlases(tmp_path, monkeypatch):
    root = tmp_path / "atlases"
    root.mkdir()
    lab = np.zeros(SHAPE, dtype=np.int16)
    lab[:3] = 1
    lab[3:] = 2
    nib.Nifti1Image(lab, np.eye(4)).to_filename(str(root / "sch.nii.gz"))
    pd.DataFrame({"index": [1, 2], "name": ["17Networks_LH_VisCent_ExStr_1", "17Networks_RH_DefaultA_IPL_1"]}
                 ).to_csv(root / "sch.tsv", sep="\t", index=False)
    lab2 = np.zeros(SHAPE, dtype=np.int16)
    lab2[:, :2] = 9
    nib.Nifti1Image(lab2, np.eye(4)).to_filename(str(root / "ho.nii.gz"))
    pd.DataFrame({"index": [9], "name": ["Left Hippocampus"]}).to_csv(root / "ho.tsv", sep="\t", index=False)
    monkeypatch.setattr(dq, "PARCELLATIONS", {
        "Schaefer17n400": {"stem": "sch", "exclude_substrings": ()},
        "HOSPA": {"stem": "ho", "exclude_substrings": ()},
    })
    return root


def _fake_fit(root, sub="03", arm="enc", seed=0):
    """A GLMsingle fit dir on SHAPE: 2 anchors x 4, 40 items x 2, 4 singles; TYPEB/C/D pickles."""
    rng = np.random.default_rng(seed)
    rows = []
    for rep in range(4):
        for a in ("A1", "A2"):
            rows.append({"session": f"ses-0{1 + rep // 2}", "run": "run-01", "task": "TBencoding",
                         "condition_id": a, "sharedId": 1})
    for i in range(40):
        for rep in range(2):
            rows.append({"session": "ses-01", "run": f"run-0{1 + rep}", "task": "TBencoding",
                         "condition_id": f"c{i}", "sharedId": 0})
    for i in range(4):
        rows.append({"session": "ses-02", "run": "run-02", "task": "TBencoding", "condition_id": f"s{i}", "sharedId": 0})
    ti = pd.DataFrame(rows)
    fdir = root / f"sub-{sub}" / arm
    (fdir / "glmsingle_outputs").mkdir(parents=True)
    ti.to_csv(fdir / "trial_info.csv", index=False)
    (fdir / "run_metadata.json").write_text(json.dumps({"subject": f"sub-{sub}", "arm": arm}))
    codes = {c: i for i, c in enumerate(sorted(ti["condition_id"].unique()))}
    sig = rng.normal(size=SHAPE + (len(codes),))
    meanvol = np.full(SHAPE, 1000.0)
    meanvol[0, 0, 0] = 10.0  # one sub-floor voxel
    pool = np.zeros(SHAPE, dtype=bool)
    pool[5, 5, :] = True  # four voxels, two of them outside the mask below
    for bt, fname in dqgs.BETA_FILES.items():
        betas = sig[..., ti["condition_id"].map(codes).to_numpy()] + rng.normal(scale=1.0, size=SHAPE + (len(ti),))
        d = {"betasmd": betas.astype(np.float32), "R2": rng.uniform(0, 10, SHAPE).astype(np.float32),
             "meanvol": meanvol, "HRFindex": rng.integers(0, dqgs.N_HRFS, SHAPE)}
        if bt in ("C", "D"):
            d.update(noisepool=pool, pcnum=3)
        if bt == "D":
            d["FRACvalue"] = rng.uniform(0.05, 1, SHAPE).astype(np.float32)
        np.save(fdir / "glmsingle_outputs" / fname, d, allow_pickle=True)
    return fdir


def _mask():
    m = np.ones(SHAPE, dtype=bool)
    m[5, 5, :2] = False
    return m


def test_write_cell_collect_and_tier2(tmp_path, atlases):
    fits_root = tmp_path / "glmsingle_tb"
    tree = tmp_path / "data_quality"
    fdir = _fake_fit(fits_root)
    assert dqgs.find_fits(fits_root) == [("03", "enc")]
    keys = dqgs.fit_keys(fdir)
    atl = "x" * 64
    assert not dqgs.is_current(tree, "03", "enc", SPACE, keys, atl)
    side = dqgs.write_cell(tree, "03", "enc", SPACE, fdir, _mask(), np.eye(4), atlases, keys,
                           {"input_atlases_sha256": atl, "code_version": "t"}, log=lambda m: None)
    assert dqgs.is_current(tree, "03", "enc", SPACE, keys, atl)
    assert not dqgs.is_current(tree, "03", "enc", SPACE, keys, "y" * 64)

    assert side["n_voxels_mask"] == _mask().sum()
    assert side["n_voxels_floor"] == _mask().sum() - 1
    assert side["condition_sets"]["repeat"] == {"n_conditions": 40, "n_reps": 2, "rep_scope": "within-session",
                                                "frac_conditions_one_run": 0.0, "null_block_median": 40.0,
                                                "null_frac_unshufflable": 0.0}
    b = side["beta_types"]["B"]
    # real signal (sd 1 = noise sd): matched beats the shuffles in most voxels; the null sits near 0
    assert b["repeat_frac_exceed"] > 0.5 and b["repeat_null_ncsnr_median"] < 0.3 and b["repeat_excess_median"] > 0.5
    assert side["condition_sets"]["anchor"]["rep_scope"] == "cross-session"
    assert sum(side["hrfindex_histogram"]) == side["n_voxels_floor"]
    c = side["beta_types"]["C"]
    assert c["pcnum"] == 3 and c["noisepool_n_total"] == 4 and c["noisepool_frac_outside_mask"] == 0.5
    assert "frac_median" in side["beta_types"]["D"] and "frac_median" not in side["beta_types"]["B"]
    # signal sd 1, noise sd 1 -> ncsnr ~1 (noisy with 40 conditions)
    assert 0.6 < side["beta_types"]["B"]["repeat_ncsnr_median"] < 1.5

    maps = nib.load(str(dqgs.cell_paths(tree, "03", "enc", SPACE)["maps"]))
    assert maps.shape == SHAPE + (len(side["volumes"]),)
    data = maps.get_fdata()
    assert np.isnan(data[5, 5, 0]).all()  # outside the mask: NaN, never filled
    assert data[0, 0, 0, side["volumes"].index("floor")] == 0.0

    fits, parcels = dqgs.collect(tree)
    assert len(fits) == 3 and sorted(fits["beta_type"]) == ["B", "C", "D"]
    assert set(parcels["atlas"]) == {"Schaefer17n400", "HOSPA"} and len(parcels) == 3 * 3

    fs = dqgs.fit_summary(fits, expected_subjects=["03", "06"], provisional=["06"])
    absent = fs[fs["absent"]]
    # sub-06 has no fit at all; sub-03 has only its enc arm, so its two ret arms are absent too.
    assert sorted(zip(absent["sub"], absent["arm"])) == [("03", "ret-image"), ("03", "ret-word"),
                                                         ("06", "enc"), ("06", "ret-image"), ("06", "ret-word")]
    assert absent.groupby("sub")["provisional"].all().to_dict() == {"03": False, "06": True}
    net = dqgs.networks(parcels)
    assert set(net["region"]) == {"LH_VisCent", "RH_DefaultA", "Left Hippocampus"}

    bench_root = tmp_path / "bench"
    (bench_root / "sub-03").mkdir(parents=True)
    (bench_root / "sub-03" / "sub-03_6cell.tsv").write_text("subject\tdelta\nsub-03\t0.123456\n")
    bench, shas = dqgs.benchmark_6cell(bench_root, ["03"])
    assert bench["delta"].tolist() == ["0.123456"] and len(shas) == 1
    with pytest.raises(FileNotFoundError, match="6-cell"):
        dqgs.benchmark_6cell(bench_root, ["04"])

    tables = {"fit_summary": fs, "networks": net, "benchmark_6cell": bench}
    a, b = tmp_path / "t2a", tmp_path / "t2b"
    dqgs.write(tables, a, {"created": "1"})
    dqgs.write(tables, b, {"created": "2"})
    assert dqgs.diff(a, b) == []
    assert (a / "benchmark_6cell.tsv").read_text() == "subject\tdelta\nsub-03\t0.123456\n"


def test_reduce_refuses_trial_count_mismatch(tmp_path):
    fdir = _fake_fit(tmp_path / "fits")
    ti = pd.read_csv(fdir / "trial_info.csv")
    d = dqgs.load_beta_dict(fdir, "B")
    with pytest.raises(ValueError, match="per-trial layout"):
        dqgs.reduce_beta_type(d, "B", _mask(), ti.iloc[:-1], dqgs.condition_sets(ti.iloc[:-1]))


def test_load_tables_missing_is_loud(tmp_path):
    with pytest.raises(FileNotFoundError, match="tier1.py glmsingle"):
        dqgs.load_tables(tmp_path)
