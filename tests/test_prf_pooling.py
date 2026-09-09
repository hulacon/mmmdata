"""How fit_prf.py combines runs into a fit unit.

Only the data-free logic: which runs group together, whether the guard that
protects averaging actually fires, and what the pooled design is made of. The
model itself is exercised by `fit_prf.py --self-test`, which needs the aperture
files and so cannot live here.

The decisions under test are in mmmdata-agents/docs/workbench/prf-retinotopy/
(DECIDED 2026-09-09).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import fit_prf  # noqa: E402


def _block(seed, res=None, n_tr=None):
    res = res or fit_prf.APERTURE_RES
    n_tr = n_tr or fit_prf.N_TR
    rng = np.random.default_rng(seed)
    return rng.random((n_tr, res, res)).astype(np.float32)


def _design(session, run, setnum, block):
    return {"session": session, "run": run, "setnum": setnum, "block": block}


@pytest.fixture
def six_runs():
    """The real acquisition order: ses-02 is 93,94,93 and ses-03 is 94,93,94."""
    bars, wedge = _block(0), _block(1)
    return [
        _design("02", 1, 93, bars), _design("02", 2, 94, wedge),
        _design("02", 3, 93, bars), _design("03", 1, 94, wedge),
        _design("03", 2, 93, bars), _design("03", 3, 94, wedge),
    ]


def test_average_groups_by_setnum_not_by_session(six_runs):
    _, _, groups, setnums = fit_prf.group_designs(six_runs, "average")
    assert setnums == [93, 94]
    assert [len(g) for g in groups] == [3, 3]
    # Both sessions must land in BOTH pseudo-runs; that is what keeps session
    # and stimulus type from being confounded, and it is a property of the
    # 93,94,93 / 94,93,94 order rather than of this code.
    for g in groups:
        assert {six_runs[i]["session"] for i in g} == {"02", "03"}


def test_average_yields_one_block_per_setnum(six_runs):
    S, run_index, groups, _ = fit_prf.group_designs(six_runs, "average")
    assert S.shape == (2 * fit_prf.N_TR, fit_prf.APERTURE_RES, fit_prf.APERTURE_RES)
    assert list(np.unique(run_index)) == [0, 1]
    assert (run_index == 0).sum() == fit_prf.N_TR


def test_concat_keeps_every_run(six_runs):
    S, run_index, groups, setnums = fit_prf.group_designs(six_runs, "concat")
    assert S.shape[0] == 6 * fit_prf.N_TR
    assert [len(g) for g in groups] == [1] * 6
    assert setnums == [93, 94, 93, 94, 93, 94]


def test_averaged_block_is_the_mean_of_its_runs():
    """Timing jitter makes same-setnum blocks differ slightly; the design must
    average them rather than pick one, to match what is done to the data."""
    a, b = _block(0), None
    rng = np.random.default_rng(7)
    b = (a + rng.normal(0, 1e-4, a.shape)).astype(np.float32)   # jitter-sized
    designs = [_design("02", 1, 93, a), _design("03", 2, 93, b)]
    S, _, _, _ = fit_prf.group_designs(designs, "average")
    np.testing.assert_allclose(S[: fit_prf.N_TR], (a + b) / 2, rtol=0, atol=1e-6)


def test_guard_tolerates_timing_jitter_but_refuses_a_wrong_aperture():
    """The floor sits between the two by orders of magnitude: same-setnum runs
    measured r >= 0.9999998, a run paired with the other setnum r = -0.004."""
    bars = _block(0)
    rng = np.random.default_rng(3)
    jittered = (bars + rng.normal(0, 1e-4, bars.shape)).astype(np.float32)
    fit_prf.group_designs(
        [_design("02", 1, 93, bars), _design("03", 2, 93, jittered)], "average")

    mislabelled = _block(99)          # a different stimulus wearing setnum 93
    with pytest.raises(SystemExit) as excinfo:
        fit_prf.group_designs(
            [_design("02", 1, 93, bars), _design("03", 2, 93, mislabelled)], "average")
    assert "wrong" in str(excinfo.value)


def test_guard_does_not_apply_to_concat(six_runs):
    """Concatenation pairs each run with its own aperture, so unequal blocks
    are not an error there."""
    six_runs[2]["block"] = _block(42)
    fit_prf.group_designs(six_runs, "concat")


def test_run_projector_removes_trends_and_confounds():
    n_tr = fit_prf.N_TR
    t = np.linspace(-1, 1, n_tr)
    motion = np.column_stack([np.sin(3 * t), np.cos(5 * t)])
    Q = fit_prf.run_projector(motion)
    # A signal made only of drift and confounds must residualise to nothing.
    y = 4.0 + 2.0 * t + 0.5 * t ** 2 - 3.0 * motion[:, 0] + 1.5 * motion[:, 1]
    assert np.abs(y - Q @ (Q.T @ y)).max() < 1e-8
    # ... and a component orthogonal to both must survive.
    keep = np.polynomial.legendre.legval(t, [0] * 8 + [1])
    keep -= Q @ (Q.T @ keep)
    assert np.linalg.norm(keep) > 1.0


def test_run_projector_without_confounds_is_legendre_only():
    Q = fit_prf.run_projector(None)
    assert Q.shape[1] == fit_prf.POLY_DEGREE + 1


def test_confound_presets_name_real_columns():
    assert fit_prf.CONFOUND_PRESETS["none"] == ()
    assert len(fit_prf.CONFOUND_PRESETS["motion6"]) == 6
    # aCompCor rides on top of the six motion parameters, per the bake-off.
    assert len(fit_prf.CONFOUND_PRESETS["acompcor"]) == 12
    assert set(fit_prf.CONFOUND_PRESETS["motion6"]) < set(fit_prf.CONFOUND_PRESETS["acompcor"])
