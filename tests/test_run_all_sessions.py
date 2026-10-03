"""run_all.py --sessions: write only the named BIDS sessions.

Every converter overwrites its output, so converting a newly acquired session
must not touch the sessions already in the tree. The filter keys on the BIDS
destination, not the source path.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CONVERTERS = REPO_ROOT / "src" / "python" / "raw2bids_converters"
# The converter modules import each other by bare name (``from common import
# ...``), which resolves only when their own directory is on sys.path.
if str(CONVERTERS) not in sys.path:
    sys.path.insert(0, str(CONVERTERS))

from raw2bids_converters import run_all  # noqa: E402


def _row(src, dest, ct="timed_events"):
    return {"source_file": src, "bids_destination": dest, "conversion_type": ct}


ROWS = [
    _row("sub-##/ses-04/behavioral/a.csv", "sub-06/ses-04/func/sub-06_ses-04_task-X_events.tsv"),
    _row("sub-##/ses-10/behavioral/b.csv", "sub-06/ses-10/func/sub-06_ses-10_task-X_events.tsv"),
    _row("sub-##/ses-10/behavioral/b_timing.csv", "n/a (timing input)", "timing_input"),
    _row("sub-##/ses-11/dicom/Series_10_x_PhysioLog", "sub-06/ses-11/func/sub-06_ses-11_task-X", "physio_dcm"),
]


def test_no_session_filter_keeps_everything():
    assert run_all.filter_rows(ROWS) == ROWS


def test_sessions_match_on_destination():
    got = run_all.filter_rows(ROWS, sessions=["ses-10", "ses-11"])
    assert [r["bids_destination"].split("/")[1] for r in got] == ["ses-10", "ses-11"]


def test_bare_and_unpadded_session_labels():
    assert run_all.filter_rows(ROWS, sessions=["10"]) == run_all.filter_rows(ROWS, sessions=["ses-10"])
    assert len(run_all.filter_rows(ROWS, sessions=["4"])) == 1


def test_destination_wins_over_source_path():
    # A source recorded under one session can land in another (an aborted
    # session re-run later); the filter follows where the file is written.
    rows = [_row("sub-##/ses-31/behavioral/c.csv", "sub-07/ses-08/func/sub-07_ses-08_task-X_events.tsv")]
    assert run_all.filter_rows(rows, sessions=["ses-08"]) == rows
    assert run_all.filter_rows(rows, sessions=["ses-31"]) == []


def test_rows_without_bids_destination_drop_out():
    got = run_all.filter_rows(ROWS, sessions=["ses-10"])
    assert all(r["conversion_type"] != "timing_input" for r in got)
