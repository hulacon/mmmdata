"""Design construction: what goes in, what is refused."""

import numpy as np
import pandas as pd
import pytest

from neuroimaging.constants import MOTION_6
from neuroimaging.glm.config import GlmConfig
from neuroimaging.glm.design import (
    DesignError,
    confound_regressors,
    contrast_vector,
    model_events,
)
from neuroimaging.glm.models import load_model


def _motor_events(with_rest=True, n_blocks=2):
    rows = []
    t = 0.0
    conds = ["hand", "foot", "mouth", "saccade"] + (["rest"] if with_rest else [])
    for _ in range(n_blocks):
        for c in conds:
            rows.append({"onset": t, "duration": 20.0, "trial_type": c, "run_idx": 1})
            t += 20.0
    return pd.DataFrame(rows)


def test_model_events_keeps_only_declared_conditions_and_bids_columns():
    m = load_model("floc")
    ev = pd.DataFrame(
        {
            "onset": [0, 4, 8],
            "duration": [4, 4, 4],
            "trial_type": ["baseline", "adult", "car"],
            "domain": ["baseline", "face", "object"],
        }
    )
    out = model_events(ev, m, strict=False)
    assert list(out.columns) == ["onset", "duration", "trial_type"]
    assert list(out["trial_type"]) == ["adult", "car"]  # baseline is implicit


def test_strict_refuses_a_run_missing_a_declared_condition():
    m = load_model("motor")
    with pytest.raises(DesignError, match="absent from this run's events.*rest"):
        model_events(_motor_events(with_rest=False), m)


def test_events_missing_the_factor_column_are_refused():
    m = load_model("motor")
    with pytest.raises(DesignError, match="trial_type"):
        model_events(pd.DataFrame({"onset": [0.0], "duration": [1.0]}), m)


def test_confound_regressors_take_motion_plus_cosines_and_fill_nan():
    cols = MOTION_6 + ["cosine00", "cosine01", "framewise_displacement"]
    df = pd.DataFrame(np.ones((5, len(cols))), columns=cols)
    df.loc[0, "trans_x"] = np.nan
    regs = confound_regressors(df, GlmConfig())
    assert list(regs.columns) == MOTION_6 + ["cosine00", "cosine01"]
    assert regs.loc[0, "trans_x"] == 0.0


def test_missing_confound_column_is_loud():
    df = pd.DataFrame({"trans_x": [0.0, 0.0]})
    with pytest.raises(KeyError, match="trans_y"):
        confound_regressors(df, GlmConfig())


def test_contrast_vector_places_weights_on_condition_columns():
    m = load_model("motor")
    cols = ["hand", "foot", "mouth", "saccade", "rest", "trans_x", "constant"]
    vec = contrast_vector(m.contrast("handVsRest"), cols)
    assert vec.tolist() == [1, 0, 0, 0, -1, 0, 0]


def test_contrast_vector_refuses_a_missing_column():
    m = load_model("motor")
    with pytest.raises(DesignError, match="rest"):
        contrast_vector(m.contrast("handVsRest"), ["hand", "constant"])


def test_build_design_matrix_has_conditions_confounds_and_intercept():
    pytest.importorskip("nilearn")
    from neuroimaging.glm.design import build_design_matrix

    m = load_model("motor")
    ev = _motor_events()
    n_scans = 140
    conf = pd.DataFrame(np.random.default_rng(0).normal(size=(n_scans, 7)), columns=MOTION_6 + ["cosine00"])
    dm = build_design_matrix(ev, conf, t_r=1.5, n_scans=n_scans, model=m, cfg=GlmConfig())
    assert dm.shape == (n_scans, 5 + 7 + 1)
    for c in m.conditions:
        assert c in dm.columns
    assert "constant" in dm.columns
    # A block regressor peaks during its own blocks.
    hand = dm["hand"].to_numpy()
    assert hand[: int(20 / 1.5) + 4].max() > hand[int(40 / 1.5) : int(60 / 1.5)].max()


def test_build_design_matrix_refuses_confound_length_mismatch():
    pytest.importorskip("nilearn")
    from neuroimaging.glm.design import build_design_matrix

    m = load_model("motor")
    conf = pd.DataFrame(np.zeros((10, 6)), columns=MOTION_6)
    with pytest.raises(DesignError, match="rows"):
        build_design_matrix(_motor_events(), conf, 1.5, 140, m, GlmConfig())
