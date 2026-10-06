"""physio_triage.csv is generated data: untracked, and loud when missing.

Third of the three generated conversion tables (after edf_triage.csv and
file_inventory.csv, untracked 2026-08-24). Before this, ``load_physio_triage``
printed a WARNING and continued, so a fresh checkout built an inventory with
every scanner physio row silently gone.
"""

import csv
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CONVERTERS = REPO_ROOT / "src" / "python" / "raw2bids_converters"
# The converter modules import each other by bare name (``from common import
# ...``), which resolves only when their own directory is on sys.path -- as it
# is when they run as scripts from that directory.
if str(CONVERTERS) not in sys.path:
    sys.path.insert(0, str(CONVERTERS))

from raw2bids_converters import generate_inventory as gi  # noqa: E402
from raw2bids_converters import generate_physio_triage as gpt  # noqa: E402


def _write_table(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=gpt.FIELDS)
        w.writeheader()
        w.writerows(rows)


def _row(sub, ses, series, status):
    return {
        "sub": sub, "ses": ses, "series": series, "size_mb": "1.0",
        "status": status, "num_volumes": "10", "tr_ms": "1500.0",
        "expected_dur": "15.0", "rec_dur": "15.0", "ratio": "1.0",
        "sections": "PULS,RESP", "source_path": f"{sub}/{ses}/dicom/{series}",
    }


def _bold(bids_root: Path, sub, ses, task_run, series_number):
    """A converted BOLD's sidecar, which is all the pairing rule reads."""
    func = bids_root / sub / ses / "func"
    func.mkdir(parents=True, exist_ok=True)
    side = func / f"{sub}_{ses}_task-{task_run}_bold.json"
    side.write_text(json.dumps({"SeriesNumber": series_number}))


@pytest.fixture
def table_path(tmp_path, monkeypatch):
    path = tmp_path / "physio_triage.csv"
    monkeypatch.setattr(gi, "PHYSIO_TRIAGE_CSV", str(path))
    return path


# ---------------------------------------------------------------------------
# untracked
# ---------------------------------------------------------------------------

def test_table_is_gitignored():
    ignored = (REPO_ROOT / ".gitignore").read_text().splitlines()
    assert "src/python/raw2bids_converters/physio_triage.csv" in ignored


def test_regeneration_driver_exists_and_names_the_interpreter():
    sbatch = REPO_ROOT / "scripts" / "physio_triage.sbatch"
    assert sbatch.is_file()
    text = sbatch.read_text()
    assert "generate_physio_triage.py" in text
    # The whole point of the driver: the interpreter is written down here.
    assert "PYTHON=" in text and "pydicom" in text


# ---------------------------------------------------------------------------
# loader
# ---------------------------------------------------------------------------

def test_missing_table_raises_and_names_the_fix(table_path):
    with pytest.raises(FileNotFoundError) as exc:
        gi.load_physio_triage()
    msg = str(exc.value)
    assert str(table_path) in msg
    assert "physio_triage.sbatch" in msg
    assert "required=False" in msg


def test_missing_table_is_allowed_only_when_asked(table_path):
    assert gi.load_physio_triage(required=False) == []


def test_present_but_empty_table_is_data_not_absence(table_path):
    _write_table(table_path, [])
    assert gi.load_physio_triage() == []


def test_status_filter_and_subject_scope(table_path, tmp_path):
    bids = tmp_path / "bids"
    _bold(bids, "sub-aa", "ses-04", "TBencoding_run-01", 8)
    _bold(bids, "sub-aa", "ses-04", "TBencoding_run-02", 10)
    _bold(bids, "sub-aa", "ses-04", "TBmath", 12)
    _bold(bids, "sub-bb", "ses-19", "NATresting", 18)
    _write_table(table_path, [
        _row("sub-aa", "ses-04", "Series_10_cued_recall_encoding_run1_PhysioLog", "COMPLETE"),
        _row("sub-aa", "ses-04", "Series_12_cued_recall_encoding_run2_PhysioLog", "TRUNCATED"),
        _row("sub-aa", "ses-04", "Series_14_cued_recall_math_PhysioLog", "INFO_ONLY"),
        _row("sub-bb", "ses-19", "Series_20_free_recall_resting_PhysioLog", "PARTIAL"),
    ])
    rows = gi.load_physio_triage(bids_root=bids)
    assert [r["source_file"] for r in rows] == [
        "sub-aa/ses-04/dicom/Series_10_cued_recall_encoding_run1_PhysioLog",
        "sub-bb/ses-19/dicom/Series_20_free_recall_resting_PhysioLog",
    ]
    assert rows[0]["bids_destination"] == (
        "sub-aa/ses-04/func/sub-aa_ses-04_task-TBencoding_run-01"
    )
    assert rows[1]["bids_destination"] == (
        "sub-bb/ses-19/func/sub-bb_ses-19_task-NATresting"
    )
    assert {r["conversion_type"] for r in rows} == {"physio_dcm"}

    scoped = gi.load_physio_triage(subjects={"sub-aa"}, bids_root=bids)
    assert [r["source_file"].split("/")[0] for r in scoped] == ["sub-aa"]


# ---------------------------------------------------------------------------
# pairing with the converted BOLD
# ---------------------------------------------------------------------------

def test_recording_without_a_converted_bold_is_not_inventoried(table_path, tmp_path):
    """An excluded BOLD's recording would otherwise become an orphan physio file."""
    _write_table(table_path, [
        _row("sub-aa", "ses-02", "Series_44_localizer_tone_PhysioLog", "COMPLETE"),
    ])
    assert gi.load_physio_triage(bids_root=tmp_path / "bids") == []


def test_aborted_attempt_does_not_take_the_real_runs_destination(table_path, tmp_path):
    """Two series map to floc run 1; only the one written after the kept BOLD pairs."""
    bids = tmp_path / "bids"
    _bold(bids, "sub-aa", "ses-02", "floc_run-01", 30)
    _write_table(table_path, [
        _row("sub-aa", "ses-02", "Series_26_localizer_floc_run1_PhysioLog", "COMPLETE"),
        _row("sub-aa", "ses-02", "Series_32_localizer_floc_run1_PhysioLog", "COMPLETE"),
    ])
    rows = gi.load_physio_triage(bids_root=bids)
    assert [r["source_file"].rsplit("/", 1)[1] for r in rows] == [
        "Series_32_localizer_floc_run1_PhysioLog"
    ]


def test_offset_of_one_pairs(table_path, tmp_path):
    bids = tmp_path / "bids"
    _bold(bids, "sub-aa", "ses-11", "TBencoding_run-01", 8)
    _write_table(table_path, [
        _row("sub-aa", "ses-11", "Series_09_cued_recall_encoding_run1_PhysioLog", "COMPLETE"),
    ])
    assert len(gi.load_physio_triage(bids_root=bids)) == 1


def test_run1_series_takes_a_runless_bolds_name(table_path, tmp_path):
    """A single-run task is converted with no run entity; its physio must match."""
    bids = tmp_path / "bids"
    _bold(bids, "sub-aa", "ses-21", "NATretrieval", 30)
    _write_table(table_path, [
        _row("sub-aa", "ses-21", "Series_32_free_recall_retrieval_run1_PhysioLog", "COMPLETE"),
    ])
    (row,) = gi.load_physio_triage(bids_root=bids)
    assert row["bids_destination"] == "sub-aa/ses-21/func/sub-aa_ses-21_task-NATretrieval"


def test_two_series_pairing_with_one_run_raise(table_path, tmp_path):
    bids = tmp_path / "bids"
    _bold(bids, "sub-aa", "ses-04", "TBmath", 12)
    _write_table(table_path, [
        _row("sub-aa", "ses-04", "Series_13_cued_recall_math_PhysioLog", "COMPLETE"),
        _row("sub-aa", "ses-04", "Series_14_cued_recall_math_PhysioLog", "COMPLETE"),
    ])
    with pytest.raises(ValueError, match="two PhysioLog series"):
        gi.load_physio_triage(bids_root=bids)


# ---------------------------------------------------------------------------
# generator interpreter check
# ---------------------------------------------------------------------------

def test_generator_names_the_interpreter_when_pydicom_is_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "pydicom", None)  # makes `import pydicom` raise
    with pytest.raises(SystemExit) as exc:
        gpt._require_pydicom()
    assert "physio_triage.sbatch" in str(exc.value)
    assert "pydicom" in str(exc.value)


def test_generator_checks_interpreter_before_touching_files(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "pydicom", None)
    out = tmp_path / "physio_triage.csv"
    with pytest.raises(SystemExit):
        gpt.main(["--output", str(out)])
    assert not out.exists()


@pytest.mark.parametrize("series, expected", [
    ("Series_56_cued_recall_retrieval_run1_RR_PhysioLog", ("TBretrieval", 1)),
    ("Series_14_cued_recall_encoding_run2_PhysioLog", ("TBencoding", 2)),
    ("Series_34_cued_recall_resting_RR_PhysioLog", ("TBresting", None)),
])
def test_rerecon_physiolog_maps_to_its_task_and_run(series, expected):
    """A scanner re-reconstruction ("_RR") is the same task and run."""
    assert gi._physio_series_to_bids(series) == expected
