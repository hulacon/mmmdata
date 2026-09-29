"""data-quality tier 2, univariate part: T1.5/T1.8 summaries and the T2.13 split-half."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

nib = pytest.importorskip("nibabel")

from neuroimaging import data_quality_univariate as dqu  # noqa: E402

SHAPE = (6, 6, 4)
CONTRASTS = {"handVsRest": [1.0, -1.0], "footVsRest": [0.0, -1.0]}


def _write_cell(root, sub, ses, run, regime, betas, sigma2, cov):
    func = root / f"sub-{sub}" / f"ses-{ses}" / "func"
    func.mkdir(parents=True, exist_ok=True)
    stem = f"sub-{sub}_ses-{ses}_task-motor_run-{run}_space-MNI152NLin2009cAsym_res-2_desc-{regime}"
    nib.Nifti1Image(betas.astype(np.float32), np.eye(4)).to_filename(str(func / f"{stem}_betas.nii.gz"))
    nib.Nifti1Image(sigma2.astype(np.float32), np.eye(4)).to_filename(str(func / f"{stem}_sigmasquared.nii.gz"))
    (func / f"{stem}_glm.json").write_text(json.dumps({
        "sub": sub, "ses": ses, "task": "motor", "run": run, "regime": regime,
        "contrasts": CONTRASTS, "cov_unscaled": cov.tolist(),
    }))


def _tree(tmp_path, signal=True, seed=0):
    """Four motor runs over two sessions; a fixed hand-vs-rest map plus independent noise per run."""
    rng = np.random.default_rng(seed)
    pattern = rng.normal(size=SHAPE)
    mask = np.ones(SHAPE, dtype=bool)
    mask[0] = False
    rows = []
    for ses, run in (("01", "01"), ("01", "02"), ("02", "01"), ("02", "02")):
        b = np.zeros(SHAPE + (2,))
        b[..., 0] = (3.0 * pattern if signal else 0.0) + rng.normal(size=SHAPE)
        b[..., 1] = rng.normal(size=SHAPE)
        s2 = np.where(mask, 1.0 + rng.uniform(size=SHAPE), 0.0)
        b[~mask] = 0.0
        if (ses, run) == ("02", "02"):
            s2[1, 1, 1] = np.nan  # below that run's PSC floor
            b[1, 1, 1] = np.nan
        _write_cell(tmp_path, "01", ses, run, "reference", b, s2, np.array([[0.5, 0.1], [0.1, 0.5]]))
        rows.append({"sub": "01", "ses": ses, "task": "motor", "run": run, "regime": "reference", "absent": False})
    return pd.DataFrame(rows), mask


def test_half_t_is_nilearn_precision_weighted_fixed_effects(tmp_path):
    from nilearn.glm.contrasts import compute_fixed_effects

    glm, mask = _tree(tmp_path)
    cells = [dqu._load_cell(p) for p in sorted(tmp_path.glob("sub-01/ses-*/func/*_glm.json"))[:3]]
    t = dqu._half_t(cells, "handVsRest", mask)
    w = np.array(CONTRASTS["handVsRest"])
    effs = [nib.Nifti1Image(np.tensordot(c.betas, w, axes=([3], [0])), np.eye(4)) for c in cells]
    vars_ = [nib.Nifti1Image(c.sigma2 * float(w @ np.array(c.meta["cov_unscaled"]) @ w), np.eye(4)) for c in cells]
    res = compute_fixed_effects(effs, vars_, mask=nib.Nifti1Image(mask.astype(np.uint8), np.eye(4)),
                                precision_weighted=True)
    assert np.allclose(t, res[2].get_fdata()[mask], rtol=1e-5)


def test_split_half_finds_a_shared_map_and_not_noise(tmp_path):
    glm, mask = _tree(tmp_path / "a", signal=True)
    sh = dqu.split_half(tmp_path / "a", glm)
    hand = sh[sh.contrast == "handVsRest"].iloc[0]
    assert hand["n_runs"] == 4 and hand["runs_half1"] == "01:01,02:01"  # sorted (ses, run), odd positions
    assert hand["n_mask"] == int(mask.sum()) and hand["n_valid"] == hand["n_mask"] - 1  # the NaN voxel drops
    assert hand["r"] > 0.8
    foot = sh[sh.contrast == "footVsRest"].iloc[0]
    assert abs(foot["r"]) < 0.4
    assert "dice@500" in sh.columns  # motor's pre-registered N sets
    glm, _ = _tree(tmp_path / "b", signal=False)
    null = dqu.split_half(tmp_path / "b", glm)
    assert abs(null.loc[null.contrast == "handVsRest", "r"].iloc[0]) < 0.4


def test_split_half_skips_a_subject_with_one_run_and_non_localizers(tmp_path):
    glm, _ = _tree(tmp_path)
    one = glm.iloc[:1]
    assert dqu.split_half(tmp_path, one).empty
    other = glm.assign(task="TBencoding")
    assert dqu.split_half(tmp_path, other).empty


def _runs():
    rows = []
    for sub, prov in (("03", False), ("06", True)):
        for regime in ("none", "reference"):
            for i, task in enumerate(("floc", "TBmath")):
                absent = regime == "reference" and task == "TBmath" and sub == "03"
                rows.append({"sub": sub, "ses": "01", "task": task, "run": "01", "regime": regime, "absent": absent,
                             "task_frac_p001": np.nan if absent else 0.1 * (i + 1),
                             "task_r2adj_median": np.nan if absent else 0.01 * (i + 1),
                             "task_r2adj_p99": np.nan if absent else 0.2, "task_r2_median": 0.05,
                             "dof_resid": 100, "n_regressors": 10})
    glm = pd.DataFrame(rows)
    motion = glm[dqu.KEYS].drop_duplicates().assign(motion_task_r_max=0.3)
    return glm, motion


def test_task_r2_scopes_absent_and_motion():
    glm, motion = _runs()
    runs = dqu.run_table(glm, motion, provisional=["06"])
    t = dqu.task_r2(runs)
    assert set(t["scope"]) == {"sub-03", "sub-06", "pooled", "pooled_confirmed"}
    row = t[(t.scope == "sub-03") & (t.task == "TBmath") & (t.regime == "reference")].iloc[0]
    assert row["n_runs"] == 1 and row["n_absent"] == 1 and pd.isna(row["r2adj_median"])
    allrow = t[(t.scope == "pooled_confirmed") & (t.task == "all") & (t.regime == "none")].iloc[0]
    assert allrow["n_runs"] == 2 and allrow["r2adj_median"] == pytest.approx(0.015)
    assert allrow["frac_p001_median"] == pytest.approx(0.15)
    assert allrow["motion_task_r_max"] == pytest.approx(0.3) and not allrow["provisional"]


def test_run_table_needs_t18():
    glm, motion = _runs()
    with pytest.raises(KeyError, match="motion_task_r_max"):
        dqu.run_table(glm, motion.drop(columns=["motion_task_r_max"]))
    with pytest.raises(KeyError, match="no T1.8 value"):
        dqu.run_table(glm, motion.iloc[1:])


def test_write_is_byte_stable_and_diff_sees_changes(tmp_path):
    glm, motion = _runs()
    res = dqu.UnivariateResult(task_r2=dqu.task_r2(dqu.run_table(glm, motion)), split_half=pd.DataFrame())
    dqu.write(res, tmp_path / "a", {"created": "x"})
    dqu.write(res, tmp_path / "b", {"created": "y"})
    assert dqu.diff(tmp_path / "a", tmp_path / "b") == []
    (tmp_path / "b" / "task_r2.tsv").write_text("changed\n")
    assert dqu.diff(tmp_path / "a", tmp_path / "b") == ["task_r2.tsv differs"]
