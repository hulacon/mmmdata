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
