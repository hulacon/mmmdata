"""The standalone GLMsingle arms' pure parts: designs, trial-beta contrasts, label alignment."""

import numpy as np
import pandas as pd
import pytest

nib = pytest.importorskip("nibabel")

from neuroimaging.glm.glmsingle_arm import (  # noqa: E402
    block_design,
    contrast_from_trial_betas,
    tb_trial_labels,
    trial_conditions,
    welch_contrast,
)


def test_block_design_marks_onset_trs_and_refuses_collisions():
    ev = pd.DataFrame({"onset": [0.0, 4.0, 8.0, 12.0], "duration": 4.0,
                       "trial_type": ["adult", "car", "baseline", "adult"]})
    d = block_design(ev, ["adult", "car"], 2.0, 10)
    assert d.shape == (10, 2) and d.sum() == 3
    assert d[0, 0] == 1 and d[2, 1] == 1 and d[6, 0] == 1  # baseline ignored
    with pytest.raises(ValueError, match="round to TR"):
        block_design(pd.DataFrame({"onset": [0.0, 0.4], "duration": 1, "trial_type": ["a", "a"]}), ["a"], 2.0, 5)
    assert trial_conditions([d, d]).tolist() == [0, 1, 0, 0, 1, 0]


def test_contrast_from_trial_betas_recovers_planted_difference():
    rng = np.random.default_rng(0)
    cond = np.array([0, 1] * 12)
    betas = rng.normal(size=(3, 3, 3, 24))
    betas[0, 0, 0, cond == 0] += 3.0
    ce = contrast_from_trial_betas(betas, cond, {0: 1.0, 1: -1.0}, np.eye(4), mask=np.ones((3, 3, 3), bool))
    assert ce.effect.get_fdata()[0, 0, 0] > 2.0 and ce.z.get_fdata()[0, 0, 0] > 3.0
    assert abs(ce.effect.get_fdata()[2, 2, 2]) < 1.5 and ce.dof == 22
    with pytest.raises(ValueError, match="at least 2"):
        contrast_from_trial_betas(betas, np.zeros(24, int), {0: 1.0, 1: -1.0}, np.eye(4))


def test_welch_contrast_masks_and_recovers():
    rng = np.random.default_rng(1)
    a = rng.normal(size=(3, 3, 3, 30))
    b = rng.normal(size=(3, 3, 3, 60))
    a[1, 1, 1] += 2.0
    mask = np.ones((3, 3, 3), bool)
    mask[0, 0, 0] = False
    ce = welch_contrast(a, b, np.eye(4), mask)
    assert ce.z.get_fdata()[1, 1, 1] > 3.0 and ce.effect.get_fdata()[0, 0, 0] == 0.0
    assert np.isnan(ce.variance.get_fdata()[0, 0, 0]) and 30 < ce.dof < 90


def test_trial_labels_align_by_session_run_onset_and_refuse_mismatch():
    trial_info = pd.DataFrame({"session": ["ses-04", "ses-04", "ses-05"], "run": ["run-01", "run-02", "run-01"],
                               "onset": [9.0, 13.5, 9.0]})
    ev = [pd.DataFrame({"ses_num": [4], "run_idx": [1], "onset": [9.0], "trial_type": ["first"]}),
          pd.DataFrame({"ses_num": [4], "run_idx": [2], "onset": [13.5], "trial_type": ["later"]}),
          pd.DataFrame({"ses_num": [5], "run_idx": [1], "onset": [9.0], "trial_type": ["once"]})]
    assert tb_trial_labels(trial_info, ev).tolist() == ["first", "later", "once"]
    with pytest.raises(KeyError, match="beta columns and events disagree"):
        tb_trial_labels(trial_info, ev[:2])
