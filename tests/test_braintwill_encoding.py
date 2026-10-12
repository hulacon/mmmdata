"""scripts/braintwill_encoding/harness.py — the films register's row selection and held-out guard.

Synthetic window tables only; the fits themselves are braintwill's and tested there.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

HARNESS = Path(__file__).resolve().parents[1] / "scripts" / "braintwill_encoding" / "harness.py"


@pytest.fixture(scope="module")
def harness():
    spec = importlib.util.spec_from_file_location("braintwill_encoding_harness", HARNESS)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except (ImportError, SyntaxError) as exc:  # braintwill needs Python >= 3.12
        pytest.skip(f"braintwill.encoding not importable here: {exc}")
    return mod


def _windows(harness, n_pool=None):
    n_pool = harness.fm.N_POOL if n_pool is None else n_pool
    rows = [{"sub": "01", "ses": f"{2 + i // 4:02d}", "stimulus_id": f"film-{i:02d}", "role": "alignment"}
            for i in range(n_pool)]
    held = harness.fm.HELDOUT_SESSIONS[0]
    rows += [{"sub": "01", "ses": held, "stimulus_id": "film-held", "role": "heldout"},
             {"sub": "02", "ses": "02", "stimulus_id": "film-00", "role": "alignment"}]
    return pd.DataFrame(rows)


def test_the_pool_is_one_subjects_alignment_rows(harness):
    w = _windows(harness)
    rows = harness.alignment_rows(w, "01")
    assert len(rows) == harness.fm.N_POOL
    assert set(rows["role"]) == {"alignment"} and set(rows["sub"]) == {"01"}


def test_a_subset_keeps_table_order(harness):
    w = _windows(harness)
    rows = harness.alignment_rows(w, "01", ["film-03", "film-01"])
    assert list(rows["stimulus_id"]) == ["film-01", "film-03"]


def test_a_film_outside_the_pool_is_refused(harness):
    with pytest.raises(RuntimeError, match="not in the alignment pool"):
        harness.alignment_rows(_windows(harness), "01", ["film-held"])


def test_a_heldout_session_row_with_an_alignment_role_is_refused(harness):
    w = _windows(harness)
    w.loc[w["stimulus_id"] == "film-held", "role"] = "alignment"
    with pytest.raises(RuntimeError, match="held-out sessions"):
        harness.alignment_rows(w, "01")


def test_a_short_pool_is_refused(harness):
    with pytest.raises(RuntimeError, match="expected the pool"):
        harness.alignment_rows(_windows(harness, n_pool=10), "01")


# --- reliability.py: the films mask ------------------------------------------------------------

RELIABILITY = HARNESS.with_name("reliability.py")


@pytest.fixture(scope="module")
def reliability():
    spec = importlib.util.spec_from_file_location("braintwill_encoding_reliability", RELIABILITY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_mean_pair_corr_equals_the_pairwise_average(reliability):
    import numpy as np

    rng = np.random.default_rng(0)
    z = np.stack([reliability._zscore(s) for s in rng.standard_normal((5, 50, 7))])
    pairs = [np.mean(z[i] * z[j], axis=0) for i in range(5) for j in range(i + 1, 5)]
    np.testing.assert_allclose(reliability.mean_pair_corr(z), np.mean(pairs, axis=0), atol=1e-12)


def test_bh_matches_the_step_up_rule(reliability):
    import numpy as np

    p = np.array([0.01, 0.04, 0.03, 0.2])
    # sorted .01 .03 .04 .2 -> p·m/rank .04 .06 .0533 .2 -> running min from the top
    np.testing.assert_allclose(reliability.bh(p), [0.04, 0.16 / 3, 0.16 / 3, 0.2], rtol=1e-12)


def test_repeat_rows_refuse_a_heldout_session(reliability):
    held = reliability.fm.HELDOUT_SESSIONS[0]
    w = pd.DataFrame([{"sub": "01", "ses": "02", "stimulus_id": "rep-a", "role": "dropped_repeat", "showing": 1},
                      {"sub": "01", "ses": held, "stimulus_id": "rep-a", "role": "dropped_repeat", "showing": 2}])
    with pytest.raises(RuntimeError, match="held-out sessions"):
        reliability.repeat_rows(w, "01")
    ok = reliability.repeat_rows(w[w["ses"] != held], "01")
    assert list(ok) == ["rep-a"] and list(ok["rep-a"]) == ["1"]
