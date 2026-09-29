"""The frozen reference spec: it resolves every GlmConfig field, refuses edits, and may only grow regimes."""

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from neuroimaging.constants import MOTION_6
from neuroimaging.glm import reference
from neuroimaging.glm.config import GlmConfig
from neuroimaging.glm.reference import (
    CONSUMER_FIELDS,
    REGIME_FIELDS,
    load_reference_spec,
    reference_config,
    spec_digest,
)
from neuroimaging.io import FmriprepRun, check_native_pool, mask_intersection


def test_spec_matches_its_digest():
    spec = load_reference_spec()
    assert spec["sha256"] == spec_digest(spec)


def test_every_glmconfig_field_is_decided_by_the_spec():
    # A new GlmConfig field fails here until the spec says what the reference does with it.
    spec = load_reference_spec()
    fields = {f.name for f in dataclasses.fields(GlmConfig)}
    decided = set(spec["config"]) | set(REGIME_FIELDS) | set(CONSUMER_FIELDS)
    assert fields == decided


def test_reference_regime_is_pinned():
    regime = load_reference_spec()["confound_regimes"]["reference"]
    assert regime["status"] == "frozen"
    assert regime["confounds"] == list(MOTION_6) and regime["acompcor_n"] == 6


def test_reference_config_is_ols_spm_unsmoothed_with_spikes():
    cfg = reference_config()
    assert (cfg.noise_model, cfg.hrf_model, cfg.smoothing_fwhm) == ("ols", "spm", None)
    assert cfg.include_non_steady_state and cfg.include_cosine and cfg.drift_model is None
    assert cfg.confounds == tuple(MOTION_6) and cfg.acompcor_n == 6
    assert reference_config(calibrated=True).noise_model == "ar1"
    assert reference_config(output_tree="elsewhere").output_tree == "elsewhere"


def test_unknown_regime_is_named():
    with pytest.raises(KeyError, match="reference"):
        reference_config("nope")


def _copy(tmp_path, mutate):
    spec = json.loads(reference.SPEC_PATH.read_text())
    mutate(spec)
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))
    return path


def test_edited_spec_is_refused(tmp_path):
    path = _copy(tmp_path, lambda s: s["config"].update(smoothing_fwhm=5.0))
    with pytest.raises(RuntimeError, match="edited"):
        load_reference_spec(path)


def test_new_regime_is_an_extension_not_an_edit(tmp_path):
    path = _copy(tmp_path, lambda s: s["confound_regimes"].update(
        extra={"status": "provisional", "confounds": [], "acompcor_n": 0}))
    assert "extra" in load_reference_spec(path)["confound_regimes"]


def _mask_run(tmp_path, name, arr, affine=np.eye(4)):
    nib = pytest.importorskip("nibabel")
    path = tmp_path / f"{name}_mask.nii.gz"
    nib.Nifti1Image(arr.astype(np.uint8), affine).to_filename(str(path))
    return FmriprepRun(subject="aa", session="01", task="t", run=name, space="x", variant="fmriprep", mask=path)


def test_mask_intersection_keeps_voxels_inside_every_run(tmp_path):
    a = np.ones((3, 3, 3)); b = np.ones((3, 3, 3)); b[0, 0, 0] = 0
    _, inter = mask_intersection([_mask_run(tmp_path, "01", a), _mask_run(tmp_path, "02", b)])
    assert inter.sum() == 26 and not inter[0, 0, 0]


def test_mask_intersection_refuses_mixed_grids(tmp_path):
    a = _mask_run(tmp_path, "01", np.ones((3, 3, 3)))
    b = _mask_run(tmp_path, "02", np.ones((3, 3, 3)), affine=np.diag([2.0, 2.0, 2.0, 1.0]))
    with pytest.raises(ValueError, match="grid differs"):
        mask_intersection([a, b])


def _native_run(tmp_path, ses, run, translation, shape=(10, 10, 10), zoom=2.0):
    """A func-space run with a brain mask and a boldref->T1w ITK coreg transform."""
    nib = pytest.importorskip("nibabel")
    d = tmp_path / f"ses-{ses}"
    d.mkdir(exist_ok=True)
    prefix = f"sub-aa_ses-{ses}_task-t_run-{run}"
    mask = d / f"{prefix}_desc-brain_mask.nii.gz"
    nib.Nifti1Image(np.ones(shape, dtype=np.uint8), np.diag([zoom, zoom, zoom, 1.0])).to_filename(str(mask))
    conf = d / f"{prefix}_desc-confounds_timeseries.tsv"
    conf.write_text("x\n0\n")
    t = " ".join(str(v) for v in translation)
    (d / f"{prefix}_from-boldref_to-T1w_mode-image_desc-coreg_xfm.txt").write_text(
        "#Insight Transform File V1.0\n#Transform 0\nTransform: AffineTransform_float_3_3\n"
        f"Parameters: 1 0 0 0 1 0 0 0 1 {t}\nFixedParameters: 0 0 0\n")
    return FmriprepRun(subject="aa", session=ses, task="t", run=run, space="func", variant="fmriprep",
                       mask=mask, confounds=conf)


def test_native_pool_within_a_session_passes_under_half_a_voxel(tmp_path):
    runs = [_native_run(tmp_path, "02", "01", (0, 0, 0)), _native_run(tmp_path, "02", "02", (0.3, 0, 0))]
    assert check_native_pool(runs) == pytest.approx(0.3)
    _, inter = mask_intersection(runs)
    assert inter.all()


def test_native_pool_refuses_head_movement_between_runs(tmp_path):
    # Same grid, same session, but the second run's anatomy sits 1.5 mm away (> 1 mm = half a 2 mm voxel).
    runs = [_native_run(tmp_path, "02", "01", (0, 0, 0)), _native_run(tmp_path, "02", "02", (0, 1.5, 0))]
    with pytest.raises(ValueError, match="head moved"):
        mask_intersection(runs)


def test_native_pool_refuses_sessions_even_on_an_identical_grid(tmp_path):
    runs = [_native_run(tmp_path, "02", "01", (0, 0, 0)), _native_run(tmp_path, "03", "01", (0, 0, 0))]
    with pytest.raises(ValueError, match="span sessions"):
        mask_intersection(runs)


def test_template_space_pools_across_sessions_without_the_native_check(tmp_path):
    a = _native_run(tmp_path, "02", "01", (0, 0, 0))
    b = _native_run(tmp_path, "03", "01", (0, 9, 0))
    a, b = (dataclasses.replace(r, space="T1w") for r in (a, b))
    _, inter = mask_intersection([a, b])
    assert inter.all()


# --- regime drift through the GLM (data-quality T1.5/T1.6, 2026-09-29) ------------------------

def test_every_regime_resolves_to_its_declared_drift():
    for name, entry in load_reference_spec()["confound_regimes"].items():
        cfg = reference_config(name)
        drift = entry.get("drift", "cosine")
        assert cfg.confounds == tuple(entry["confounds"]) and cfg.acompcor_n == entry["acompcor_n"]
        if drift == "cosine":
            assert cfg.drift_model is None and cfg.include_cosine
        elif drift == "polynomial":
            assert (cfg.drift_model, cfg.drift_order, cfg.include_cosine) == ("polynomial", entry["drift_order"], False)
        else:
            assert drift == "none" and cfg.drift_model is None and not cfg.include_cosine


def test_unknown_drift_is_refused(tmp_path, monkeypatch):
    path = _copy(tmp_path, lambda s: s["confound_regimes"].update(
        odd={"status": "provisional", "confounds": [], "acompcor_n": 0, "drift": "spline"}))
    monkeypatch.setattr(reference, "load_reference_spec", lambda: load_reference_spec(path))
    with pytest.raises(ValueError, match="spline"):
        reference_config("odd")


N_SCANS = 140
_ACC = [f"a_comp_cor_{i:02d}" for i in range(20)]
_DERIV = [f"{c}_derivative1" for c in MOTION_6]


def _confounds(rng, n_nss=0):
    import pandas as pd

    cols = list(MOTION_6) + _DERIV + ["framewise_displacement", "csf", "white_matter", "global_signal"] + _ACC
    conf = pd.DataFrame(rng.normal(size=(N_SCANS, len(cols))), columns=cols)
    conf.loc[0, _DERIV + ["framewise_displacement"]] = np.nan  # fMRIPrep's n/a first row
    t = np.arange(N_SCANS)
    for k in range(3):
        conf[f"cosine{k:02d}"] = np.cos(np.pi * (k + 1) * (t + 0.5) / N_SCANS)
    for v in range(n_nss):
        conf[f"non_steady_state_outlier{v:02d}"] = (t == v).astype(float)
    return conf


def _motor_events():
    import pandas as pd

    rows, t = [], 0.0
    for _ in range(2):
        for c in ["hand", "foot", "mouth", "saccade", "speak", "rest"]:
            rows.append({"onset": t, "duration": 20.0, "trial_type": c})
            t += 20.0
    return pd.DataFrame(rows)


def _residuals(X, Y):
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    return Y - X @ beta, beta


def _glm_design(regime, conf):
    pytest.importorskip("nilearn")
    from neuroimaging.glm.design import build_design_matrix
    from neuroimaging.glm.models import load_model

    model = load_model("motor")
    dm = build_design_matrix(_motor_events(), conf, 1.5, N_SCANS, model, reference_config(regime))
    return dm, list(model.conditions)


def test_polynomial_regime_design_has_nilearn_drift_and_no_cosines():
    dm, _ = _glm_design("base", _confounds(np.random.default_rng(0)))
    assert not any(c.startswith("cosine") for c in dm.columns)
    assert {"drift_1", "drift_2", "constant"} <= set(dm.columns)
    dm, _ = _glm_design("gsr", _confounds(np.random.default_rng(0)))
    assert not any(c.startswith(("cosine", "drift")) for c in dm.columns) and "constant" in dm.columns


def test_glm_confound_residuals_equal_the_cleaners_on_every_confirmed_regime():
    # The GLM's nuisance space (regime columns + nilearn drift + intercept + one spike per lead-in
    # volume) must leave the same residuals, on the steady-state volumes, as the data-quality cleaner
    # (regime_design + intercept, lead-in rows dropped). Different drift bases, same span.
    from neuroimaging.confounds import confirmed_regimes, get_regime, regime_design

    rng = np.random.default_rng(1)
    conf = _confounds(rng, n_nss=2)
    Y = rng.normal(size=(N_SCANS, 5)) + np.linspace(0, 3, N_SCANS)[:, None] ** 2
    for name in confirmed_regimes():
        dm, conds = _glm_design(name, conf)
        nuis = dm.drop(columns=conds).to_numpy()
        r_glm, _ = _residuals(nuis, Y)
        rd = regime_design(get_regime(name), conf)
        keep = ~rd.nss
        Xc = np.column_stack([rd.columns.to_numpy()[keep], np.ones(keep.sum())])
        r_clean, _ = _residuals(Xc, Y[keep])
        assert np.allclose(r_glm[keep], r_clean, atol=1e-8), name
        assert np.allclose(r_glm[~keep], 0.0, atol=1e-8), name  # spikes absorb the lead-in volumes


#: The colleague's regimes as their code builds them (github.com/ntpouba/mmm @ ff11baa,
#: volume/regress_out_confounds_volume.py `CONFOUND_VARIANTS` + `build_covariates`), keyed by
#: our registry names: their base / basecsfwm / baseacc6 / baseacc20 are our base12fd* (renamed
#: 2026-09-28). Transcribed rather than imported: their repo is not a dependency.
_COLLEAGUE = {
    "base12fd": ([], True),
    "base12fdcsfwm": (["csf", "white_matter"], True),
    "base12fdacc6": (_ACC[:6], True),
    "base12fdacc20": (_ACC, True),
    "gsr": (None, False),
}
_THEIR_MOTION = [c for m in MOTION_6 for c in (m, f"{m}_derivative1")] + ["framewise_displacement"]


def _colleague_covariates(conf, name):
    extra, drift = _COLLEAGUE[name]
    cols = ["global_signal"] if extra is None else _THEIR_MOTION + extra
    cov = conf[cols].copy().fillna(0)
    if drift:
        cov["linear"] = np.linspace(0, 1, len(cov))
        cov["quadratic"] = cov["linear"] ** 2
    return cov.to_numpy()


def test_glm_matches_the_colleagues_regressors_on_their_five_regimes():
    # Their runs carry no lead-in columns, and they fit every volume; so does this check.
    # Nuisance residuals match theirs (nilearn.signal.clean projects on the z-scored covariates,
    # and their z-scoring removes the mean: the same as an intercept), and the task betas of a
    # GLM carrying their covariates equal ours: raw t, t^2 vs nilearn's orthogonal polynomials
    # is a change of basis, not of model.
    rng = np.random.default_rng(2)
    conf = _confounds(rng)
    Y = rng.normal(size=(N_SCANS, 5)) + np.linspace(0, 3, N_SCANS)[:, None] ** 2
    for name in _COLLEAGUE:
        dm, conds = _glm_design(name, conf)
        theirs = np.column_stack([_colleague_covariates(conf, name), np.ones(N_SCANS)])
        r_ours, _ = _residuals(dm.drop(columns=conds).to_numpy(), Y)
        r_theirs, _ = _residuals(theirs, Y)
        assert np.allclose(r_ours, r_theirs, atol=1e-8), name
        _, b_ours = _residuals(np.column_stack([dm[conds], dm.drop(columns=conds)]), Y)
        _, b_theirs = _residuals(np.column_stack([dm[conds].to_numpy(), theirs]), Y)
        assert np.allclose(b_ours[: len(conds)], b_theirs[: len(conds)], atol=1e-8), name
