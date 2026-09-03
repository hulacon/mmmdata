"""The nilearn estimator recovers a planted effect; fixed effects sharpen it.

Synthetic data only: a tiny 4D volume with a block design and a known
active cluster. This is the estimator *interface* under test, not nilearn.
"""

import numpy as np
import pandas as pd
import pytest

nib = pytest.importorskip("nibabel")
pytest.importorskip("nilearn")

from neuroimaging.constants import MOTION_6  # noqa: E402
from neuroimaging.glm.config import GlmConfig  # noqa: E402
from neuroimaging.glm.design import build_design_matrix, contrast_vectors  # noqa: E402
from neuroimaging.glm.estimators import NilearnEstimator, fixed_effects, get_estimator  # noqa: E402
from neuroimaging.glm.models import load_model  # noqa: E402

TR = 1.5
N_SCANS = 200
SHAPE = (6, 6, 6)
ACTIVE = (slice(0, 2), slice(0, 2), slice(0, 2))


def _events():
    rows, t = [], 0.0
    for _ in range(3):
        for c in ("hand", "foot", "mouth", "saccade", "rest"):
            rows.append({"onset": t, "duration": 20.0, "trial_type": c})
            t += 20.0
    return pd.DataFrame(rows)


def _synthetic_run(seed, effect=2.0):
    """BOLD where the ACTIVE corner follows the convolved `hand` regressor."""
    rng = np.random.default_rng(seed)
    cfg = GlmConfig(smoothing_fwhm=None)
    model = load_model("motor")
    conf = pd.DataFrame(rng.normal(scale=0.1, size=(N_SCANS, 6)), columns=MOTION_6)
    dm = build_design_matrix(_events(), conf, TR, N_SCANS, model, cfg)
    signal = dm["hand"].to_numpy()
    data = rng.normal(loc=100.0, scale=1.0, size=SHAPE + (N_SCANS,))
    data[ACTIVE] += effect * signal[None, None, None, :]
    img = nib.Nifti1Image(data.astype(np.float32), np.eye(4))
    mask = nib.Nifti1Image(np.ones(SHAPE, dtype=np.uint8), np.eye(4))
    return img, mask, dm, contrast_vectors(model, list(dm.columns)), cfg


def test_nilearn_estimator_recovers_the_planted_effect():
    img, mask, dm, vecs, cfg = _synthetic_run(0)
    est = NilearnEstimator().fit_run(img, dm, vecs, t_r=TR, mask=mask, cfg=cfg)
    assert set(est) == {"handVsRest", "footVsRest", "mouthVsRest", "saccadeVsRest"}
    hand = est["handVsRest"]
    eff = hand.effect.get_fdata()
    z = hand.z.get_fdata()
    assert eff[ACTIVE].mean() > 1.0  # planted 2.0, hand - rest
    assert abs(eff[3:, 3:, 3:].mean()) < 0.5  # nothing planted
    assert z[ACTIVE].min() > 3.0
    assert hand.variance.get_fdata().min() > 0
    assert hand.dof is not None and hand.dof > 100


def test_control_contrast_is_null_where_nothing_was_planted():
    img, mask, dm, vecs, cfg = _synthetic_run(1)
    est = NilearnEstimator().fit_run(img, dm, vecs, t_r=TR, mask=mask, cfg=cfg)
    foot_z = est["footVsRest"].z.get_fdata()
    assert abs(foot_z[ACTIVE].mean()) < 2.0


def test_fixed_effects_across_two_runs_sharpens_the_estimate():
    runs = [_synthetic_run(s) for s in (2, 3)]
    ests = []
    for img, mask, dm, vecs, cfg in runs:
        ests.append(NilearnEstimator().fit_run(img, dm, vecs, t_r=TR, mask=mask, cfg=cfg)["handVsRest"])
    fx = fixed_effects(ests, mask=runs[0][1])
    assert fx.n_runs == 2
    single = ests[0].z.get_fdata()[ACTIVE].mean()
    pooled = (fx.z if fx.z is not None else fx.stat).get_fdata()[ACTIVE].mean()
    assert pooled > single
    assert fx.variance.get_fdata()[ACTIVE].mean() < ests[0].variance.get_fdata()[ACTIVE].mean()


def test_fixed_effects_needs_a_run():
    with pytest.raises(ValueError):
        fixed_effects([])


def test_estimator_registry():
    assert get_estimator("nilearn").name == "nilearn"
    with pytest.raises(KeyError, match="available"):
        get_estimator("flame")


def test_ols_control_arm_is_selectable():
    img, mask, dm, vecs, cfg = _synthetic_run(4)
    cfg_ols = GlmConfig(smoothing_fwhm=None, noise_model="ols")
    est = NilearnEstimator().fit_run(img, dm, vecs, t_r=TR, mask=mask, cfg=cfg_ols)
    assert est["handVsRest"].effect.get_fdata()[ACTIVE].mean() > 1.0
