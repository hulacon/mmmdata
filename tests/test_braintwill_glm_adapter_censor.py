"""adapter.py: FD censoring through braintwill's ``retain`` and the confound-column override."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("nilearn")

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "braintwill_glm"
sys.path.insert(0, str(SCRIPTS))
try:
    import adapter  # noqa: E402
except ImportError as exc:  # braintwill not importable: adapter refuses loudly
    pytest.skip(f"adapter not importable: {exc}", allow_module_level=True)


def test_censor_retain_keeps_first_volume_and_drops_above_threshold():
    conf = pd.DataFrame({"framewise_displacement": [np.nan, 0.1, 0.5, 0.51, 2.0, 0.0]})
    keep = adapter.censor_retain(conf, 0.5)
    assert keep.tolist() == [True, True, True, False, False, True]


def test_censor_retain_refuses_without_fd():
    with pytest.raises(KeyError, match="framewise_displacement"):
        adapter.censor_retain(pd.DataFrame({"trans_x": [0.0, 0.1]}), 0.5)


def test_regime_config_override_replaces_named_confounds_only():
    ref = adapter.regime_config("reference")
    bare = adapter.regime_config("reference", [])
    assert ref.confounds and bare.confounds == ()
    assert bare.acompcor_n == ref.acompcor_n
    assert bare.include_cosine == ref.include_cosine
    assert bare.include_non_steady_state == ref.include_non_steady_state
    assert adapter.regime_config("reference", None) == ref


def test_split_design_carries_retain():
    dm = pd.DataFrame({"A": [0.0, 1.0, 1.0, 0.0], "trans_x": [0.0, 0.1, 0.2, 0.1], "constant": 1.0})
    keep = np.array([True, True, False, True])
    d = adapter.split_design(dm, ["A"], keep)
    assert d.retain.tolist() == keep.tolist()
    assert d.nuisance == ("trans_x", "constant")
    assert adapter.split_design(dm, ["A"]).retain is None
