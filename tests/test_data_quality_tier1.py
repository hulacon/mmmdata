"""data-quality tier 1: the regime registry, the cleaner, its measures and the driver's verbs."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

nib = pytest.importorskip("nibabel")

from neuroimaging import data_quality as dq  # noqa: E402
from neuroimaging.confounds import (  # noqa: E402
    confirmed_regimes,
    describe_regimes,
    get_regime,
    load_regimes,
    polynomial_drift,
    regime_design,
)
from neuroimaging.constants import DEFAULT_SPACE, MOTION_6  # noqa: E402
from neuroimaging.glm.reference import reference_config  # noqa: E402
from neuroimaging.io import FmriprepRun  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SHAPE = (6, 6, 6)
N_VOL = 60
TR = 1.5
AFFINE = np.diag([2.0, 2.0, 2.0, 1.0])


# ---------------------------------------------------------------------------
# Synthetic run + atlases
# ---------------------------------------------------------------------------

def _confounds(n_vol: int, rng: np.random.Generator) -> pd.DataFrame:
    df = pd.DataFrame({c: rng.normal(size=n_vol) * 0.1 for c in MOTION_6})
    for c in MOTION_6:
        d = np.diff(df[c].to_numpy(), prepend=np.nan)
        df[f"{c}_derivative1"] = d
    df["framewise_displacement"] = np.r_[np.nan, np.abs(rng.normal(size=n_vol - 1)) * 0.2]
    df["csf"] = rng.normal(size=n_vol)
    df["white_matter"] = rng.normal(size=n_vol)
    df["global_signal"] = rng.normal(size=n_vol)
    for i in range(20):
        df[f"a_comp_cor_{i:02d}"] = rng.normal(size=n_vol)
    t = np.arange(n_vol)
    df["cosine00"] = np.cos(np.pi * t / n_vol)
    df["cosine01"] = np.cos(2 * np.pi * t / n_vol)
    nss = np.zeros(n_vol)
    nss[0] = 1
    df["non_steady_state_outlier00"] = nss
    return df


@pytest.fixture
def tree(tmp_path):
    """A BIDS root with one fMRIPrep run on a 6x6x6 grid, and the two atlases."""
    rng = np.random.default_rng(7)
    bids = tmp_path / "bids"
    fp = bids / "derivatives" / "fmriprep"
    func = fp / "sub-01" / "ses-01" / "func"
    func.mkdir(parents=True)
    (fp / "dataset_description.json").write_text(json.dumps(
        {"Name": "fMRIPrep", "BIDSVersion": "1.9.0", "DatasetType": "derivative",
         "GeneratedBy": [{"Name": "fMRIPrep", "Version": "25.2.5"}]}))

    conf = _confounds(N_VOL, rng)
    mask = np.ones(SHAPE, dtype=np.uint8)
    mask[0, :, :] = 0  # one plane outside the brain
    n_vox = int(mask.sum())
    # Signal = 1000 baseline + a slope + the global signal + noise, per voxel.
    t = np.linspace(-1, 1, N_VOL)
    gs = conf["global_signal"].to_numpy()
    data = (
        1000.0
        + 5.0 * t[:, None]
        + 3.0 * gs[:, None] * rng.uniform(0.5, 1.5, size=n_vox)[None, :]
        + rng.normal(size=(N_VOL, n_vox)) * 2.0
    )
    data[0] += 400.0  # the non-steady-state volume is hot
    vol = np.zeros(SHAPE + (N_VOL,), dtype=np.float32)
    vol[mask.astype(bool)] = data.T.astype(np.float32)
    bold = nib.Nifti1Image(vol, AFFINE)
    bold.header.set_zooms((2.0, 2.0, 2.0, TR))
    prefix = "sub-01_ses-01_task-rest_run-01"
    space = f"space-{DEFAULT_SPACE.replace('_res-', '_res-')}"
    bold_path = func / f"{prefix}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz"
    mask_path = func / f"{prefix}_space-MNI152NLin2009cAsym_res-2_desc-brain_mask.nii.gz"
    bold.to_filename(str(bold_path))
    nib.Nifti1Image(mask, AFFINE).to_filename(str(mask_path))
    (func / f"{prefix}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.json").write_text(
        json.dumps({"RepetitionTime": TR}))
    conf_path = func / f"{prefix}_desc-confounds_timeseries.tsv"
    conf.to_csv(conf_path, sep="\t", index=False, na_rep="n/a")
    (func / f"{prefix}_desc-confounds_timeseries.json").write_text("{}")

    atlases = bids / "derivatives" / "atlases"
    anat = atlases / "tpl-MNI152NLin2009cAsym" / "anat"
    anat.mkdir(parents=True)
    sch = np.zeros(SHAPE, dtype=np.int16)
    sch[1:3, :, :] = 1
    sch[3:6, :, :] = 2
    sch[0, :, :] = 3  # entirely outside the mask
    stem = dq.PARCELLATIONS["Schaefer17n400"]["stem"]
    nib.Nifti1Image(sch, AFFINE).to_filename(str(atlases / f"{stem}.nii.gz"))
    pd.DataFrame({"index": [1, 2, 3], "name": ["17Networks_LH_A_1", "17Networks_RH_B_1", "17Networks_LH_C_1"],
                  "color": ["#000000"] * 3}).to_csv(atlases / f"{stem}.tsv", sep="\t", index=False)
    hos = np.zeros(SHAPE, dtype=np.int16)
    hos[1:6, 0:2, :] = 1  # tissue class, excluded
    hos[1:6, 2:4, :] = 9  # hippocampus
    stem = dq.PARCELLATIONS["HOSPA"]["stem"]
    nib.Nifti1Image(hos, AFFINE).to_filename(str(atlases / f"{stem}.nii.gz"))
    pd.DataFrame({"index": [1, 9], "name": ["Left Cerebral White Matter", "Left Hippocampus"]}).to_csv(
        atlases / f"{stem}.tsv", sep="\t", index=False)

    run = FmriprepRun(
        subject="01", session="01", task="rest", run="01", variant="fmriprep", space=DEFAULT_SPACE,
        bold=bold_path, mask=mask_path, confounds=conf_path,
        confounds_json=func / f"{prefix}_desc-confounds_timeseries.json",
    )
    return {"bids": bids, "fmriprep": fp, "atlases": atlases, "run": run, "conf": conf, "mask": mask}


def _tier1():
    spec = importlib.util.spec_from_file_location("tier1", REPO / "scripts" / "data_quality" / "tier1.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _argv(tree, *rest):
    return [*rest, "--bids-root", str(tree["bids"]), "--tree-root", str(tree["bids"] / "derivatives" / "data_quality"),
            "--atlases-dir", str(tree["atlases"])]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_registry_has_the_charter_regimes_with_their_statuses():
    regimes = load_regimes()
    assert set(regimes) >= {"reference", "none", "drift", "base", "basecsfwm", "baseacc6", "baseacc20", "gsr"}
    assert regimes["reference"].status == "frozen"
    assert {regimes[n].status for n in ("none", "drift")} == {"defined"}
    assert {regimes[n].status for n in ("base", "basecsfwm", "baseacc6", "baseacc20", "gsr")} == {"provisional"}
    assert set(confirmed_regimes()) == {"reference", "none", "drift"}
    assert len({r.version for r in regimes.values()}) == len(regimes)
    assert set(describe_regimes()["regime"]) == set(regimes)


def test_colleague_regimes_are_verbatim_from_the_slides():
    base = get_regime("base")
    assert len(base.confounds) == 13 and "framewise_displacement" in base.confounds
    assert all(f"{c}_derivative1" in base.confounds for c in MOTION_6)
    assert (base.drift, base.drift_order) == ("polynomial", 2)
    assert get_regime("basecsfwm").confounds == base.confounds + ("csf", "white_matter")
    assert (get_regime("baseacc6").acompcor_n, get_regime("baseacc20").acompcor_n) == (6, 20)
    gsr = get_regime("gsr")
    assert gsr.confounds == ("global_signal",) and gsr.drift == "none"


def test_reference_config_refuses_regimes_glmconfig_cannot_express():
    assert reference_config("reference").acompcor_n == 6
    with pytest.raises(ValueError, match="drift"):
        reference_config("drift")
    with pytest.raises(ValueError, match="drift"):
        reference_config("none")


def test_polynomial_drift_is_orthogonal_and_has_no_constant():
    p = polynomial_drift(50, 2)
    assert p.shape == (50, 2)
    assert abs(p[:, 0].sum()) < 1e-9 and abs(p[:, 0] @ p[:, 1]) < 1e-9
    assert polynomial_drift(50, 0).shape == (50, 0)


# ---------------------------------------------------------------------------
# Designs
# ---------------------------------------------------------------------------

def test_regime_design_columns_and_nss(tree):
    conf = tree["conf"]
    d = regime_design(get_regime("base"), conf)
    assert d.n_regressors == 15 and d.n_drift == 2
    assert d.nss.sum() == 1 and d.nss[0]
    assert d.dof_resid == N_VOL - 1 - 15 - 1
    assert not d.columns.isna().any().any()  # the n/a first samples became 0
    assert regime_design(get_regime("none"), conf).n_regressors == 0
    assert regime_design(get_regime("gsr"), conf).columns.columns.tolist() == ["global_signal"]
    ref = regime_design(get_regime("reference"), conf)
    assert ref.n_regressors == 6 + 6 + 2 and ref.n_drift == 2


def test_reference_without_cosines_is_not_an_error_and_records_zero_drift(tree):
    conf = tree["conf"].drop(columns=["cosine00", "cosine01"])
    d = regime_design(get_regime("reference"), conf)
    assert d.n_drift == 0 and d.n_regressors == 12


def test_missing_columns_are_named(tree):
    conf = tree["conf"].drop(columns=["framewise_displacement"])
    with pytest.raises(KeyError, match="framewise_displacement"):
        regime_design(get_regime("base"), conf)
    with pytest.raises(KeyError, match="a_comp_cor_19"):
        regime_design(get_regime("baseacc20"), tree["conf"].drop(columns=["a_comp_cor_19"]))


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def test_none_regime_tsnr_is_mean_over_sd_and_nss_rows_are_nan(tree):
    inputs = dq.load_run_inputs(tree["run"], tree["atlases"])
    res = dq.clean(inputs.data, regime_design(get_regime("none"), inputs.confounds))
    fit = inputs.data[1:].astype(np.float64)
    np.testing.assert_allclose(res.tsnr, fit.mean(0) / fit.std(0, ddof=1), rtol=1e-6)
    assert np.isnan(res.residuals[0]).all() and not np.isnan(res.residuals[1:]).any()
    np.testing.assert_allclose(res.var_ratio, 1.0, atol=1e-9)


def test_residuals_are_orthogonal_to_the_design_and_regression_removes_variance(tree):
    inputs = dq.load_run_inputs(tree["run"], tree["atlases"])
    d = regime_design(get_regime("gsr"), inputs.confounds)
    res = dq.clean(inputs.data, d)
    X = np.column_stack([np.ones(N_VOL - 1), d.columns.to_numpy()[1:]])
    assert np.abs(X.T @ res.residuals[1:].astype(np.float64)).max() < 1e-3
    # The synthetic signal carries the global signal, so gsr removes real variance.
    assert np.nanmedian(res.var_ratio) < 0.9
    none = dq.clean(inputs.data, regime_design(get_regime("none"), inputs.confounds))
    assert np.nanmedian(res.tsnr) > np.nanmedian(none.tsnr)


def test_chunked_solve_matches_one_shot(tree):
    inputs = dq.load_run_inputs(tree["run"], tree["atlases"])
    d = regime_design(get_regime("reference"), inputs.confounds)
    a = dq.clean(inputs.data, d, chunk=7)
    b = dq.clean(inputs.data, d, chunk=10 ** 6)
    np.testing.assert_allclose(a.residuals[1:], b.residuals[1:], rtol=1e-5, atol=1e-3)
    np.testing.assert_allclose(a.tsnr, b.tsnr, rtol=1e-9)
    np.testing.assert_allclose(a.var_ratio, b.var_ratio, rtol=1e-9)


def test_mismatched_confounds_length_is_refused(tree):
    inputs = dq.load_run_inputs(tree["run"], tree["atlases"])
    d = regime_design(get_regime("none"), inputs.confounds.iloc[:-1])
    with pytest.raises(ValueError, match="volumes"):
        dq.clean(inputs.data, d)


def test_parcellation_coverage_and_exclusions(tree):
    inputs = dq.load_run_inputs(tree["run"], tree["atlases"])
    sch = inputs.parcellations["Schaefer17n400"].table.set_index("name")
    assert sch.loc["17Networks_LH_A_1", "coverage"] == 1.0
    assert sch.loc["17Networks_LH_C_1", "n_voxels_mask"] == 0 and sch.loc["17Networks_LH_C_1", "coverage"] == 0.0
    hos = inputs.parcellations["HOSPA"].table
    assert hos["name"].tolist() == ["Left Hippocampus"]


def test_parcel_timeseries_shape_nans_and_var_removed(tree):
    inputs = dq.load_run_inputs(tree["run"], tree["atlases"])
    res = dq.clean(inputs.data, regime_design(get_regime("gsr"), inputs.confounds))
    ts, table = dq.parcel_timeseries(res, inputs.parcellations["Schaefer17n400"])
    assert ts.shape == (N_VOL, 3)
    assert ts["17Networks_LH_C_1"].isna().all()
    assert ts.iloc[0].isna().all() and not ts.iloc[1:, :2].isna().any().any()
    assert 0.0 < table.set_index("name").loc["17Networks_LH_A_1", "var_removed"] < 1.0


def test_atlas_on_another_grid_is_refused(tree):
    mask = tree["mask"].astype(bool)
    with pytest.raises(ValueError, match="grid"):
        dq.load_parcellation("Schaefer17n400", tree["atlases"], mask, np.diag([3.0, 3.0, 3.0, 1.0]))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def test_run_writes_every_output_with_provenance_and_is_idempotent(tree, capsys):
    tier1 = _tier1()
    root = tree["bids"] / "derivatives" / "data_quality"
    tier1.main(_argv(tree, "run", "--sub", "01", "--ses", "01", "--task", "rest", "--run", "01"))
    assert (root / "dataset_description.json").exists()
    run = tree["run"]
    names = sorted(p.name for p in (root / "sub-01" / "ses-01" / "func").iterdir())
    assert names[0] == "sub-01_ses-01_task-rest_run-01_space-MNI152NLin2009cAsym_res-2_desc-drift_tsnr.json"
    assert not any("__" in n for n in names)
    assert len(names) == 3 * (2 + 2 * len(dq.PARCELLATIONS))
    for regime in confirmed_regimes():
        nii, js = dq.tsnr_paths(root, run, regime)
        assert nii.exists() and js.exists()
        meta = json.loads(js.read_text())
        assert meta["fmriprep_version"] == "25.2.5" and len(meta["input_bold_sha256"]) == 64
        assert meta["regime"] == regime and meta["n_nss"] == 1 and meta["schema_version"] == dq.SCHEMA_VERSION
        img = nib.load(str(nii))
        assert img.get_data_dtype() == np.float32 and img.shape == SHAPE
        assert (np.asarray(img.dataobj)[~tree["mask"].astype(bool)] == 0).all()
        for seg in dq.PARCELLATIONS:
            tsv, sj = dq.timeseries_paths(root, run, regime, seg)
            assert tsv.exists() and sj.exists()
            df = pd.read_csv(tsv, sep="\t", na_values=["n/a"])
            assert len(df) == N_VOL and df.iloc[0].isna().all()
            side = json.loads(sj.read_text())
            assert side["SamplingFrequency"] == pytest.approx(1 / TR) and side["atlas"] == seg
    # No provisional regime was built without the flag.
    assert not dq.tsnr_paths(root, run, "base")[0].exists()
    # The second call skips everything.
    capsys.readouterr()
    tier1.main(_argv(tree, "run", "--sub", "01", "--ses", "01", "--task", "rest", "--run", "01"))
    assert "skipping" in capsys.readouterr().out
    # A changed input is stale.
    js = dq.tsnr_paths(root, run, "none")[1]
    meta = json.loads(js.read_text())
    meta["input_bold_sha256"] = "0" * 64
    js.write_text(json.dumps(meta))
    assert not dq.is_current(root, run, get_regime("none"), dq.file_sha256(run.bold))


def test_provisional_regimes_need_the_flag(tree):
    tier1 = _tier1()
    with pytest.raises(SystemExit, match="provisional"):
        tier1.main(_argv(tree, "run", "--sub", "01", "--ses", "01", "--task", "rest", "--run", "01",
                         "--regimes", "base"))
    tier1.main(_argv(tree, "run", "--sub", "01", "--ses", "01", "--task", "rest", "--run", "01",
                     "--regimes", "base,gsr", "--include-provisional"))
    root = tree["bids"] / "derivatives" / "data_quality"
    assert dq.tsnr_paths(root, tree["run"], "base")[0].exists()
    assert dq.tsnr_paths(root, tree["run"], "gsr")[0].exists()


def test_plan_units_and_collect(tree, tmp_path, capsys):
    tier1 = _tier1()
    units = tmp_path / "units.txt"
    tier1.main(_argv(tree, "plan", "--units", str(units)))
    assert units.read_text().splitlines() == ["01\t01\trest\t01"]
    tier1.main(_argv(tree, "run", "--units", str(units), "--index", "1"))
    tier1.main(_argv(tree, "plan", "--units", str(units), "--check-hashes"))
    assert units.read_text() == ""
    tier1.main(_argv(tree, "collect"))
    root = tree["bids"] / "derivatives" / "data_quality"
    runs = pd.read_csv(root / "tier1_runs.tsv", sep="\t")
    parcels = pd.read_csv(root / "tier1_parcels.tsv", sep="\t", na_values=["n/a"])
    assert len(runs) == 3 and set(runs["regime"]) == {"reference", "none", "drift"}
    assert set(runs.columns) >= {"sub", "ses", "task", "run", "regime", "dof_resid", "tsnr_median_mask", "fmriprep_version"}
    assert len(parcels) == 3 * (3 + 1)
    assert set(parcels["atlas"]) == set(dq.PARCELLATIONS)


def test_missing_pipeline_description_is_loud(tmp_path):
    with pytest.raises(FileNotFoundError, match="dataset_description.json"):
        dq.pipeline_version(tmp_path)
