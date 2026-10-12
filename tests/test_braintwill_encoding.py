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


def test_item_noise_is_the_item_weighted_within_variance(reliability):
    import numpy as np

    rng = np.random.default_rng(1)
    y = rng.standard_normal((9, 4))
    labels = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2])
    want = np.mean([y[labels == i].var(axis=0, ddof=1) for i in range(3)], axis=0)
    np.testing.assert_allclose(reliability.item_noise(y, labels, 3), want, atol=1e-12)


def test_permutation_stays_within_blocks(reliability):
    import numpy as np

    labels = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    blocks = [np.arange(4), np.arange(4, 8)]
    out = reliability.permute_within(labels, blocks, np.random.default_rng(0))
    assert sorted(out[:4]) == [0, 0, 1, 1] and sorted(out[4:]) == [2, 2, 3, 3]


def _tb_trials():
    """Two sessions: a triplet shown twice, a repeated pair, a single, and one super repeat x2 per session."""
    rows = []
    for ses in ("ses-04", "ses-05"):
        tag = ses[-1]
        seq = ([("t1" + tag, 3, 1), ("t2" + tag, 3, 2), ("t3" + tag, 3, 3)] * 2
               + [("r" + tag, 2, 4)] * 2 + [("s" + tag, 1, 5)] + [("super", 2, 6)] * 2)
        rows += [{"session": ses, "stimulus_id": s, "enCon": float(c), "pairId": float(pid)} for s, c, pid in seq]
    return pd.DataFrame(rows)


def test_tb_rows_average_super_repeats_and_group_triplets(harness):
    import numpy as np

    t = _tb_trials()
    betas = np.random.default_rng(2).standard_normal((len(t), 3)).astype(np.float32)
    rows, y = harness.tb_rows(t, betas)
    z = harness.zscore_sessions(t, betas)
    assert len(rows) == len(t) - 2  # one super row per session instead of two
    sup = rows["stimulus_id"] == "super"
    assert list(rows.loc[sup, "n_trials"]) == [2, 2]
    first = np.flatnonzero(sup.to_numpy())[0]
    sel = ((t["stimulus_id"] == "super") & (t["session"] == "ses-04")).to_numpy()
    np.testing.assert_allclose(y[first], z[sel].mean(axis=0), rtol=1e-6)
    trip = rows[rows["stimulus_id"].str.startswith("t")]
    assert trip.groupby("session")["group"].nunique().tolist() == [1, 1]
    assert trip["group"].str.startswith("triplet:").all()
    other = rows[~rows["stimulus_id"].str.startswith("t") & ~sup]
    assert (other["group"] == other["stimulus_id"]).all()
    np.testing.assert_allclose(z[(t["session"] == "ses-04").to_numpy()].mean(axis=0), 0, atol=1e-6)


def test_tb_rows_refuse_a_broken_triplet(harness):
    import numpy as np

    t = _tb_trials()
    t.loc[t["stimulus_id"] == "t34", "pairId"] = 4.0  # third mate now falls in the next triplet slot
    with pytest.raises(RuntimeError, match="triplet groups"):
        harness.tb_rows(t, np.zeros((len(t), 2), np.float32))
