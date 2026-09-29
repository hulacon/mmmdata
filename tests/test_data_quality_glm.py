"""data-quality tier 1 GLM rows: T1.5 task R², T1.6 localizer betas, T1.8 motion–task r."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

nib = pytest.importorskip("nibabel")
pytest.importorskip("nilearn")

from neuroimaging import data_quality as dq  # noqa: E402
from neuroimaging import data_quality_glm as dqg  # noqa: E402
from neuroimaging.confounds import confirmed_regimes, get_regime  # noqa: E402
from neuroimaging.constants import DEFAULT_SPACE, MOTION_6  # noqa: E402
from neuroimaging.glm.design import build_design_matrix  # noqa: E402
from neuroimaging.glm.estimators import NilearnEstimator  # noqa: E402
from neuroimaging.glm.models import load_model  # noqa: E402
from neuroimaging.glm.reference import reference_config  # noqa: E402
from neuroimaging.io import FmriprepRun  # noqa: E402
if str(Path(__file__).resolve().parent) not in sys.path:  # the sibling module below; src/python is conftest's
    sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_data_quality_tier1 import N_VOL, SHAPE, TR, _argv, _confounds, _tier1, tree  # noqa: E402,F401

MOTOR = ["hand", "foot", "mouth", "saccade", "speak", "rest"]


def _motor_events(block=10.0):
    order = MOTOR + MOTOR[:3]  # 9 x 10 s = 90 s = 60 volumes at 1.5 s
    return pd.DataFrame({"onset": np.arange(len(order)) * block, "duration": block, "trial_type": order})


@pytest.fixture
def motor_tree(tree):
    """The tier-1 synthetic run relabelled task-motor, with events and a hand response in parcel 2."""
    fp_func = tree["fmriprep"] / "sub-01" / "ses-01" / "func"
    for p in list(fp_func.iterdir()):
        p.rename(p.with_name(p.name.replace("task-rest", "task-motor")))
    raw = tree["bids"] / "sub-01" / "ses-01" / "func"
    raw.mkdir(parents=True)
    prefix = "sub-01_ses-01_task-motor_run-01"
    events = _motor_events()
    events.to_csv(raw / f"{prefix}_events.tsv", sep="\t", index=False)
    (raw / f"{prefix}_bold.json").write_text(json.dumps({"RepetitionTime": TR}))
    bold_path = fp_func / f"{prefix}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz"
    conf_path = fp_func / f"{prefix}_desc-confounds_timeseries.tsv"
    conf = pd.read_csv(conf_path, sep="\t", na_values=["n/a"])
    dm = build_design_matrix(events, conf, TR, N_VOL, load_model("motor"), reference_config("none"))
    img = nib.load(str(bold_path))
    vol = np.asarray(img.dataobj, dtype=np.float32).copy()
    vol[3:6] += 40.0 * dm["hand"].to_numpy()[None, None, None, :].astype(np.float32)  # Schaefer parcel 2
    out = nib.Nifti1Image(vol, img.affine, img.header)
    out.to_filename(str(bold_path))
    run = FmriprepRun(
        subject="01", session="01", task="motor", run="01", variant="fmriprep", space=DEFAULT_SPACE,
        bold=bold_path, mask=fp_func / f"{prefix}_space-MNI152NLin2009cAsym_res-2_desc-brain_mask.nii.gz",
        confounds=conf_path, events=raw / f"{prefix}_events.tsv",
    )
    return {**tree, "run": run, "events": events, "conf": conf}


# ---------------------------------------------------------------------------
# Which runs, which model
# ---------------------------------------------------------------------------

def test_generic_model_takes_every_non_baseline_type_sorted():
    ev = pd.DataFrame({"onset": [0, 5, 10, 15, 20], "duration": 1.0,
                       "trial_type": ["word", "rest", "image", "blank", "word"]})
    m = dqg.generic_model("TBencoding", ev)
    assert m.conditions == ("image", "word") and m.contrasts == () and m.task == "TBencoding"
    with pytest.raises(ValueError, match="baseline"):
        dqg.generic_model("x", ev[ev.trial_type.isin(["rest", "blank"])])


def test_eligibility_needs_events_and_skips_the_excluded_tasks(tmp_path):
    ev = tmp_path / "e.tsv"
    mk = lambda task, events: FmriprepRun(subject="01", session="01", task=task, run=None,  # noqa: E731
                                          variant="fmriprep", space=DEFAULT_SPACE, events=events)
    assert dqg.eligible(mk("TBmath", ev)) and dqg.eligible(mk("floc", ev))
    assert not dqg.eligible(mk("NATresting", None))
    assert not dqg.eligible(mk("auditory", ev)) and not dqg.eligible(mk("fixation", ev))


def test_localizers_use_their_shipped_models():
    assert set(dqg.LOCALIZER_MODELS) == {"floc", "motor", "tone"}
    for task, name in dqg.LOCALIZER_MODELS.items():
        assert load_model(name).task == task and load_model(name).contrasts


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------

def _synthetic(seed=0, n_vox=30):
    rng = np.random.default_rng(seed)
    conf = _confounds(N_VOL, rng)
    events = _motor_events()
    dm = build_design_matrix(events, conf, TR, N_VOL, load_model("motor"), reference_config("base"))
    data = 500.0 + rng.normal(size=(N_VOL, n_vox)) * 3.0
    data[:, :10] += 25.0 * dm["hand"].to_numpy()[:, None]
    data[:, :] += 4.0 * conf["trans_x"].to_numpy()[:, None]
    return dm, data.astype(np.float32)


def test_fit_matches_nilearn_ols_in_percent_signal_change():
    dm, data = _synthetic()
    model = load_model("motor")
    res = dqg.fit(data, dm, model.conditions)
    weights = dqg.condition_weights(model, list(dm.columns), model.conditions)
    # The same data as a 4D image through the stand-in's own estimator.
    n_vox = data.shape[1]
    vol = data.T.reshape(n_vox, 1, 1, N_VOL)
    img = nib.Nifti1Image(vol, np.eye(4))
    mask = nib.Nifti1Image(np.ones((n_vox, 1, 1), dtype=np.uint8), np.eye(4))
    cfg = reference_config("base")
    from neuroimaging.glm.design import contrast_vectors

    est = NilearnEstimator().fit_run(img, dm, contrast_vectors(model, list(dm.columns)), t_r=TR, mask=mask, cfg=cfg)
    for name, w in weights.items():
        eff, var = dqg.contrast(res, w)
        assert np.allclose(eff, est[name].effect.get_fdata().ravel(), rtol=1e-4, atol=1e-5), name
        assert np.allclose(var, est[name].variance.get_fdata().ravel(), rtol=1e-4), name
    assert res.dof == est["handVsRest"].dof


def test_partial_r2_is_the_task_blocks_share_of_the_nuisance_residual():
    dm, data = _synthetic()
    conds = load_model("motor").conditions
    res = dqg.fit(data, dm, conds)
    X = dm.to_numpy()
    Xn = dm.drop(columns=list(conds)).to_numpy()
    Y = data.astype(np.float64)  # unscaled: R² is invariant to per-voxel scaling
    rss = ((Y - X @ np.linalg.lstsq(X, Y, rcond=None)[0]) ** 2).sum(0)
    rss_n = ((Y - Xn @ np.linalg.lstsq(Xn, Y, rcond=None)[0]) ** 2).sum(0)
    assert np.allclose(res.r2_task, 1 - rss / rss_n, atol=1e-8)
    dof_n = N_VOL - np.linalg.matrix_rank(Xn)
    assert np.allclose(res.r2_task_adj, 1 - (rss / res.dof) / (rss_n / dof_n), atol=1e-8)
    assert res.r2_task[:10].min() > 0.5 and np.median(res.r2_task[10:]) < 0.3


def test_adjusted_r2_is_near_zero_under_the_null_whatever_the_nuisance_count():
    # Pure noise: the raw partial R² sits at ~k / dof_nuisance and climbs as a regime adds
    # columns; the adjusted form stays ~0. This is why the adjusted form is the headline.
    rng = np.random.default_rng(11)
    conf = _confounds(N_VOL, rng)
    conds = load_model("motor").conditions
    Y = rng.normal(size=(N_VOL, 4000)) + 500.0
    raw, adj = {}, {}
    for regime in ("none", "base12fdacc20"):
        dm = build_design_matrix(_motor_events(), conf, TR, N_VOL, load_model("motor"), reference_config(regime))
        res = dqg.fit(Y, dm, conds)
        raw[regime], adj[regime] = np.median(res.r2_task), np.mean(res.r2_task_adj)
    assert raw["base12fdacc20"] > 1.5 * raw["none"]
    assert abs(adj["none"]) < 0.02 and abs(adj["base12fdacc20"]) < 0.05


def test_betas_are_masked_where_percent_signal_change_is_undefined():
    dm, data = _synthetic()
    data = data.copy()
    data[:, 0] -= 500.0 + 30.0  # mean below zero, like a resampled edge voxel
    data[:, 1] += 2.0 - data[:, 1].mean()  # positive, but ~0.4 % of the typical mean
    res = dqg.fit(data, dm, load_model("motor").conditions)
    assert not res.psc_defined[0] and not res.psc_defined[1] and res.psc_defined[2:].all()
    assert np.isnan(res.betas[:, :2]).all() and np.isnan(res.sigma2[:2]).all()
    assert np.isfinite(res.betas[:, 2:]).all()
    assert np.isfinite(res.r2_task[:2]).all()  # scale-free, kept
    assert res.psc_floor == pytest.approx(0.01 * np.median(data.mean(0)[data.mean(0) > 0]))


def test_chunked_fit_matches_one_shot():
    dm, data = _synthetic(n_vox=50)
    conds = load_model("motor").conditions
    a, b = dqg.fit(data, dm, conds), dqg.fit(data, dm, conds, chunk=7)
    assert np.allclose(a.betas, b.betas, equal_nan=True) and np.allclose(a.r2_task, b.r2_task)
    assert np.allclose(a.sigma2, b.sigma2, equal_nan=True)


def test_motion_task_correlation_finds_the_planted_pair_and_skips_lead_in():
    rng = np.random.default_rng(3)
    conf = _confounds(N_VOL, rng)
    dm = build_design_matrix(_motor_events(), conf, TR, N_VOL, load_model("motor"), reference_config("none"))
    conf["rot_y"] = dm["mouth"].to_numpy() + rng.normal(size=N_VOL) * 0.01
    conf.loc[0, "rot_y"] = 1e6  # the lead-in volume must not enter
    r = dqg.motion_task_correlation(dm, load_model("motor").conditions, conf)
    assert r["motion_task_r_motion"] == "rot_y" and r["motion_task_r_condition"] == "mouth"
    assert r["motion_task_r_max"] > 0.99


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _glm(tree, *rest):
    return _tier1().main(_argv(tree, "glm", "--sub", "01", "--ses", "01", "--task", "motor", "--run", "01", *rest))


def test_glm_writes_every_cell_with_betas_and_is_idempotent(motor_tree, capsys):
    root = motor_tree["bids"] / "derivatives" / "data_quality"
    run = motor_tree["run"]
    _glm(motor_tree)
    for regime in confirmed_regimes():
        tsv, js = dqg.glm_paths(root, run, regime)
        betas, s2 = dqg.beta_paths(root, run, regime)
        assert tsv.exists() and js.exists() and betas.exists() and s2.exists(), regime
        meta = json.loads(js.read_text())
        assert meta["model"] == "motor" and meta["beta_volumes"] == MOTOR and meta["localizer"]
        assert np.array(meta["cov_unscaled"]).shape == (6, 6) and meta["n_nss"] == 1
        assert meta["input_events_sha256"] == dq.file_sha256(run.events)
        img = nib.load(str(betas))
        assert img.shape == SHAPE + (6,) and img.get_data_dtype() == np.float32
        parcels = pd.read_csv(tsv, sep="\t", na_values=["n/a"])
        assert set(parcels["atlas"]) == set(dq.PARCELLATIONS)
        sch = parcels[parcels.atlas == "Schaefer17n400"].set_index("index")
        # The hand response was planted in parcel 2 only; parcel 3 is outside the mask.
        assert sch.loc[2, "effect_handVsRest"] > 2 * abs(sch.loc[1, "effect_handVsRest"]), regime
        assert sch.loc[2, "task_r2adj_mean"] > sch.loc[1, "task_r2adj_mean"], regime
        assert pd.isna(sch.loc[3, "task_r2adj_mean"])
    capsys.readouterr()
    _glm(motor_tree)
    assert "GLM regimes current, skipping" in capsys.readouterr().out


def test_glm_plan_units_and_collect(motor_tree, tmp_path):
    tier1 = _tier1()
    units = tmp_path / "glm_units.txt"
    tier1.main(_argv(motor_tree, "glm-plan", "--units", str(units)))
    assert units.read_text().splitlines() == ["01\t01\tmotor\t01"]
    tier1.main(_argv(motor_tree, "glm", "--units", str(units), "--index", "1"))
    tier1.main(_argv(motor_tree, "glm-plan", "--units", str(units), "--check-hashes"))
    assert units.read_text() == ""
    # A changed events file makes every cell stale.
    ev = motor_tree["events"].copy()
    ev.loc[0, "duration"] = 9.0
    ev.to_csv(motor_tree["run"].events, sep="\t", index=False)
    tier1.main(_argv(motor_tree, "glm-plan", "--units", str(units), "--check-hashes"))
    assert units.read_text().splitlines() == ["01\t01\tmotor\t01"]
    tier1.main(_argv(motor_tree, "glm", "--units", str(units), "--index", "1"))
    tier1.main(_argv(motor_tree, "collect"))
    root = motor_tree["bids"] / "derivatives" / "data_quality"
    glm = pd.read_csv(root / "tier1_glm.tsv", sep="\t", na_values=["n/a"])
    parcels = pd.read_csv(root / "tier1_glm_parcels.tsv", sep="\t", na_values=["n/a"])
    confirmed = confirmed_regimes()
    assert len(glm) == len(confirmed) and set(glm["regime"]) == set(confirmed) and not glm["absent"].any()
    assert (glm["n_conditions"] == 6).all() and {"task_r2adj_median", "task_r2adj_p99", "task_r2_median", "dof_resid", "dof_nuisance"} <= set(glm.columns)
    assert len(parcels) == len(confirmed) * (3 + 1)
    assert "effect_speakVsRest" in parcels.columns


def test_a_run_short_of_acompcor_declares_the_glm_cell_absent(motor_tree, capsys):
    root = motor_tree["bids"] / "derivatives" / "data_quality"
    run = motor_tree["run"]
    motor_tree["conf"].drop(columns=["a_comp_cor_19"]).to_csv(run.confounds, sep="\t", index=False, na_rep="n/a")
    _glm(motor_tree)
    assert "base12fdacc20  ABSENT (19 of 20" in capsys.readouterr().out
    tsv, js = dqg.glm_absent_paths(root, run, "base12fdacc20")
    assert tsv.exists() and json.loads(js.read_text())["absent"] is True
    assert not dqg.glm_paths(root, run, "base12fdacc20")[1].exists()
    assert dqg.glm_is_current(root, run, get_regime("base12fdacc20"), dqg.GlmKeys(
        dq.file_sha256(run.bold), dq.file_sha256(run.events), dqg.model_sha256(load_model("motor")), "ignored"))
    _tier1().main(_argv(motor_tree, "collect"))
    glm = pd.read_csv(root / "tier1_glm.tsv", sep="\t", na_values=["n/a"])
    gone = glm[glm.regime == "base12fdacc20"].iloc[0]
    assert bool(gone["absent"]) and pd.isna(gone["task_r2adj_median"])


def test_motion_row_carries_t18_for_task_runs_only(motor_tree):
    tier1 = _tier1()
    run = motor_tree["run"]
    row = tier1.motion_task_row(run, motor_tree["conf"], TR)
    assert row["motion_task_r_motion"] in MOTION_6 and row["motion_task_r_condition"] in MOTOR
    assert 0 <= row["motion_task_r_max"] <= 1
    rest = FmriprepRun(subject="01", session="01", task="rest", run="01", variant="fmriprep", space=DEFAULT_SPACE)
    assert tier1.motion_task_row(rest, motor_tree["conf"], TR)["motion_task_r_max"] is None
