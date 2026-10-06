"""write_events_tsv writes 'n/a' for every missing cell, never an empty one.

The PsychoPy free-recall converters fill non-applicable columns with "" (a
title row has no movie_name). fillna does not see "", so before 2026-10-06
every NATencoding and NATretrieval events file carried empty cells, which the
bids-validator rejects.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
CONVERTERS = REPO_ROOT / "src" / "python" / "raw2bids_converters"
if str(CONVERTERS) not in sys.path:
    sys.path.insert(0, str(CONVERTERS))

from raw2bids_converters.common import NA, write_events_tsv  # noqa: E402


def _written(tmp_path, df):
    out = tmp_path / "sub-##_ses-##_task-X_run-01_events.tsv"
    write_events_tsv(df, str(out))
    return out.read_text().splitlines()


def test_empty_string_and_nan_both_become_na(tmp_path):
    df = pd.DataFrame({
        "onset": [1.0, 2.0],
        "trial_type": ["title", "movie"],
        "movie_name": ["", "some film"],
        "condition": [np.nan, 2],
    })
    lines = _written(tmp_path, df)
    assert lines[1].split("\t") == ["1.0", "title", NA, NA]
    assert all("" not in line.split("\t") for line in lines)


def test_only_whole_empty_cells_are_replaced(tmp_path):
    df = pd.DataFrame({"onset": [0.0], "word": ["a b"], "note": [" "]})
    lines = _written(tmp_path, df)
    assert lines[1].split("\t") == ["0.0", "a b", " "]
