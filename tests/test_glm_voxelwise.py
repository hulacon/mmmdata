"""Per-voxel HRF fitting stitches one design per kernel group back into whole maps."""

import numpy as np
import pandas as pd
import pytest

nib = pytest.importorskip("nibabel")
pytest.importorskip("nilearn")
pytest.importorskip("glmsingle")

from neuroimaging.constants import MOTION_6  # noqa: E402
from neuroimaging.glm.config import GlmConfig  # noqa: E402
from neuroimaging.glm.design import build_design_matrix  # noqa: E402
from neuroimaging.glm.estimators import NilearnEstimator  # noqa: E402
from neuroimaging.glm.models import load_model  # noqa: E402
from neuroimaging.glm.voxelwise_hrf import fit_run_voxelwise  # noqa: E402

TR, N, SHAPE = 1.5, 200, (6, 6, 6)


def _events():
    rows, t = [], 0.0
    for _ in range(3):
        for c in ("hand", "foot", "mouth", "saccade", "rest"):
            rows.append({"onset": t, "duration": 20.0, "trial_type": c})
            t += 20.0
    return pd.DataFrame(rows)


def test_voxelwise_fit_uses_each_groups_kernel_and_covers_the_mask():
    rng = np.random.default_rng(0)
    model = load_model("motor")
    conf = pd.DataFrame(rng.normal(scale=0.1, size=(N, 6)), columns=MOTION_6)
    cfg = GlmConfig(smoothing_fwhm=None, noise_model="ols")
    idx = np.zeros(SHAPE, dtype=int)
    idx[3:, :, :] = 12  # two kernel groups split the volume
    data = rng.normal(loc=100.0, scale=1.0, size=SHAPE + (N,))
    for k, sl in ((0, np.s_[0:3, 0:2, 0:2]), (12, np.s_[3:6, 0:2, 0:2])):
        dm = build_design_matrix(_events(), conf, TR, N, model, GlmConfig(smoothing_fwhm=None, hrf_model=f"glmsingle:{k}"))
        data[sl] += 2.0 * dm["hand"].to_numpy()[None, None, None, :]
    bold = nib.Nifti1Image(data.astype(np.float32), np.eye(4))
    mask = nib.Nifti1Image(np.ones(SHAPE, dtype=np.uint8), np.eye(4))
    hidx = nib.Nifti1Image(idx.astype(np.int16), np.eye(4))

    est = fit_run_voxelwise(NilearnEstimator(), bold, _events(), conf, TR, model, cfg, hidx, mask)
    hand = est["handVsRest"]
    eff = hand.effect.get_fdata()
    z = hand.z.get_fdata()
    assert eff[0:3, 0:2, 0:2].mean() > 1.0 and eff[3:6, 0:2, 0:2].mean() > 1.0
    assert z[0:3, 0:2, 0:2].min() > 3.0 and z[3:6, 0:2, 0:2].min() > 3.0
    assert abs(eff[:, 3:, 3:].mean()) < 0.5
    assert np.isfinite(hand.variance.get_fdata()).all()  # every masked voxel was fitted by some group
    assert hand.dof is not None


def test_grid_mismatch_is_refused():
    bold = nib.Nifti1Image(np.zeros(SHAPE + (N,), dtype=np.float32), np.eye(4))
    mask = nib.Nifti1Image(np.ones(SHAPE, dtype=np.uint8), np.eye(4))
    bad = nib.Nifti1Image(np.zeros((5, 5, 5), dtype=np.int16), np.eye(4))
    with pytest.raises(ValueError, match="grid mismatch"):
        fit_run_voxelwise(NilearnEstimator(), bold, _events(), None, TR, load_model("motor"),
                          GlmConfig(smoothing_fwhm=None), bad, mask)
