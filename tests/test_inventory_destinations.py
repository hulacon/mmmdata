"""Two inventory rows must not share a BIDS destination.

run_all converts row by row and each converter overwrites its output, so a
shared destination silently keeps only the last source. Only physio rows were
checked; e.g. retrieval eye-tracking EDFs carry no run entity, so two runs in
one session would map to the same physio file.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CONVERTERS = REPO_ROOT / "src" / "python" / "raw2bids_converters"
if str(CONVERTERS) not in sys.path:
    sys.path.insert(0, str(CONVERTERS))

from raw2bids_converters import generate_inventory as gi  # noqa: E402
from raw2bids_converters import run_all  # noqa: E402


def _row(src, dest, ct):
    return {"source_file": src, "bids_destination": dest, "conversion_type": ct}


def test_shared_destination_raises():
    dest = "sub-##/ses-##/func/sub-##_ses-##_task-NATretrieval_recording-eye_physio.tsv.gz"
    rows = [_row("a.EDF", dest, "edf_to_physio"), _row("b.EDF", dest, "edf_to_physio")]
    with pytest.raises(ValueError, match="b.EDF"):
        gi.check_unique_destinations(rows)


def test_non_converting_rows_may_share():
    rows = [
        _row("a_timing.csv", "n/a (timing input)", "timing_input"),
        _row("b_timing.csv", "n/a (timing input)", "timing_input"),
        _row("a.EDF", "x_physio.tsv.gz", "edf_excluded"),
        _row("b.EDF", "x_physio.tsv.gz", "edf_to_physio"),
    ]
    gi.check_unique_destinations(rows)


def test_classified_retrieval_edfs_of_two_runs_collide():
    """Documents the collision the guard exists for: retrieval EDFs of runs 1
    and 2 in one session classify to the same run-less destination."""
    dests = {
        gi.classify_eyetracking_file(f"s9s9r{run}r_0000.EDF", 9, 9)["bids_destination"]
        for run in (1, 2)
    }
    assert len(dests) == 1


def test_non_converting_types_match_run_all():
    assert gi.NON_CONVERTING_TYPES == run_all.SKIP_TYPES
