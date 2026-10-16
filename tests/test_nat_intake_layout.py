"""NAT intake: per-recording eyetracking dirs, side files, triage gaps, NaN onsets.

From the second cohort on, the operator copies each EDF into a directory of its
own (``eyetracking/<stem>/<stem>.EDF``, AOI files under ``<stem>/aoi/``), and the
NAT PsychoPy script writes extra side files beside the .csv it always wrote: an
incremental ``.tsv`` data log, ``_timings.txt`` movie logs, and a
``trials_recall.csv`` trial export. The walker skipped every directory except
``aoi`` without a word, and the side files were unclassified. These tests pin
the layout with synthetic files only.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CONVERTERS = REPO_ROOT / "src" / "python" / "raw2bids_converters"
SCRIPTS = REPO_ROOT / "scripts"
for p in (CONVERTERS, SCRIPTS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from raw2bids_converters import generate_inventory as gi  # noqa: E402
import psychopy_encoding  # noqa: E402
import psychopy_retrieval  # noqa: E402
import edf_triage  # noqa: E402

SUBJ = 99
TS = "0000_00_00_00_00"           # EDF stamp placeholder
PSY_TS = "0000-00-00_00h00.00.000"  # PsychoPy stamp placeholder


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("")
    return path


def _walk(tmp_path, monkeypatch):
    monkeypatch.setattr(gi, "SOURCE_ROOT", str(tmp_path))
    return {r["source_file"]: r for r in gi.walk_subject(SUBJ)}


# ── eyetracking walk ─────────────────────────────────────────────────────────

def test_nested_recording_dirs_are_classified_like_flat_files(tmp_path, monkeypatch):
    et = tmp_path / "sub-99" / "ses-19" / "eyetracking"
    enc = _touch(et / f"s99s1r1m_{TS}" / f"s99s1r1m_{TS}.EDF")
    ret = _touch(et / f"s99s1r1r_{TS}" / f"s99s1r1r_{TS}.EDF")
    aoi = _touch(et / f"s99s1r1r_{TS}" / "aoi" / "TRIAL_0001_ROUTINE_01.ias")

    rows = _walk(tmp_path, monkeypatch)

    rel = lambda p: str(p.relative_to(tmp_path))  # noqa: E731
    assert rows[rel(enc)]["conversion_type"] == "edf_to_physio"
    assert "task-NATencoding_run-01_recording-eye" in rows[rel(enc)]["bids_destination"]
    assert rows[rel(ret)]["conversion_type"] == "edf_to_physio"
    assert "task-NATretrieval_recording-eye" in rows[rel(ret)]["bids_destination"]
    assert rows[rel(aoi)]["conversion_type"] == "supplementary"


def test_flat_layout_unchanged(tmp_path, monkeypatch):
    et = tmp_path / "sub-99" / "ses-19" / "eyetracking"
    enc = _touch(et / f"s99s1r1m_{TS}.EDF")
    aoi = _touch(et / "aoi" / "TRIAL_0001_ROUTINE_01.ias")

    rows = _walk(tmp_path, monkeypatch)

    assert rows[str(enc.relative_to(tmp_path))]["conversion_type"] == "edf_to_physio"
    assert rows[str(aoi.relative_to(tmp_path))]["conversion_type"] == "supplementary"


def test_unknown_eyetracking_dir_and_file_surface_as_unclassified(tmp_path, monkeypatch):
    et = tmp_path / "sub-99" / "ses-19" / "eyetracking"
    stray = _touch(et / "notes" / "readme.txt")
    odd = _touch(et / f"s99s1r1m_{TS}" / "extra.dat")

    rows = _walk(tmp_path, monkeypatch)

    for p in (stray, odd):
        row = rows[str(p.relative_to(tmp_path))]
        assert row["bids_destination"] == "n/a (unclassified)"
        assert row["conversion_type"] == "no_conversion"


# ── behavioural side files ───────────────────────────────────────────────────

def test_nat_side_files_are_classified_and_csv_still_converts(tmp_path, monkeypatch):
    beh = tmp_path / "sub-99" / "ses-19" / "behavioral"
    enc_csv = _touch(beh / f"99_1_1_mem_search_recall_{PSY_TS}.csv")
    enc_tsv = _touch(beh / f"99_1_1_mem_search_recall_{PSY_TS}.tsv")
    enc_tim = _touch(beh / f"99_1_1_mem_search_recall_{PSY_TS}_timings.txt")
    ret_csv = _touch(beh / f"99_1_free_recall_recall_{PSY_TS}.csv")
    ret_tsv = _touch(beh / f"99_1_free_recall_recall_{PSY_TS}.tsv")
    ret_trl = _touch(beh / f"99_1_free_recall_recall_{PSY_TS}trials_recall.csv")
    ses_tim = _touch(beh / f"99_{PSY_TS}_timings.txt")

    rows = _walk(tmp_path, monkeypatch)
    r = lambda p: rows[str(p.relative_to(tmp_path))]  # noqa: E731

    assert r(enc_csv)["conversion_type"] == "psychopy_encoding"
    assert r(ret_csv)["conversion_type"] == "psychopy_retrieval"
    for side in (enc_tsv, enc_tim, ret_tsv, ret_trl, ses_tim):
        assert r(side)["conversion_type"] == "no_conversion"
        assert r(side)["bids_destination"] != "n/a (unclassified)", side.name
    # The .tsv must not compete with the .csv for the events destination.
    gi.check_unique_destinations(list(rows.values()))


def test_calibration_csv_in_behavioral_is_supplementary(tmp_path, monkeypatch):
    beh = tmp_path / "sub-99" / "ses-18" / "behavioral"
    cal = _touch(beh / "eyetracking_calibration_subj99_sess1_run1_0000_Jan_00_0000_timing.csv")

    rows = _walk(tmp_path, monkeypatch)

    row = rows[str(cal.relative_to(tmp_path))]
    assert row["conversion_type"] == "supplementary"
    assert row["bids_destination"] == "n/a (calibration data)"


# ── EDF triage coverage ──────────────────────────────────────────────────────

def _edf_row(src):
    return {"source_file": src, "description": "d",
            "bids_destination": "x_physio.tsv.gz", "conversion_type": "edf_to_physio"}


def test_untriaged_edf_is_refused():
    triage = {"sub-99/ses-19/eyetracking/a.EDF": {"decision": "include", "channels": "gaze+pupil",
                                                    "gaze_valid_pct": "90", "pupil_valid_pct": "90"}}
    rows = [_edf_row("sub-99/ses-19/eyetracking/a.EDF"),
            _edf_row("sub-99/ses-19/eyetracking/b/b.EDF")]
    with pytest.raises(ValueError, match=r"edf_triage\.sbatch") as exc:
        gi.apply_edf_triage(rows, triage)
    assert "b/b.EDF" in str(exc.value)


def test_untriaged_edf_passes_when_triage_waived():
    rows = [_edf_row("sub-99/ses-19/eyetracking/b/b.EDF")]
    out = gi.apply_edf_triage(rows, {}, required=False)
    assert out[0]["conversion_type"] == "edf_to_physio"


def test_edf_triage_paths_relative_to_configured_root(tmp_path):
    nested = _touch(tmp_path / "sub-99" / "ses-19" / "eyetracking"
                    / f"s99s1r2m_{TS}" / f"s99s1r2m_{TS}.EDF")
    flat = _touch(tmp_path / "sub-99" / "ses-20" / "eyetracking" / f"s99s2r1r_{TS}.EDF")
    _touch(tmp_path / "sub-99" / "ses-20" / "eyetracking" / f"s99s2r2r_{TS}.EDF.tmp")

    found = edf_triage.find_edfs(tmp_path)
    assert found == sorted([str(nested), str(flat)])

    assert edf_triage.parse_edf_path(str(nested), tmp_path) == ("sub-99", "ses-19", 2, "encoding")
    assert edf_triage.parse_edf_path(str(flat), tmp_path) == ("sub-99", "ses-20", 1, "retrieval")
    assert edf_triage.source_key(str(nested), tmp_path) == (
        f"sub-99/ses-19/eyetracking/s99s1r2m_{TS}/s99s1r2m_{TS}.EDF")


# ── converters refuse a NaN onset ────────────────────────────────────────────

def _encoding_csv(tmp_path, drop_trial):
    rows = [{"movie_loop.thisN": None, "wait.started": 10.0}]
    for i in range(2):
        start = 20.0 + 100 * i
        rows.append({
            "movie_loop.thisN": float(i), "wait.started": None,
            "movie_title.started": start, "movie_title.stopped": start + 2,
            "fixation.started": start + 2, "fixation.stopped": start + 3,
            "movie_name": "Film One", "condition": 1, "mov_len": 90.0,
            "style": "s", "free_recall_position": 1,
            "movies.started": None if i == drop_trial else start + 3,
            "movies.stopped": None if i == drop_trial else start + 93,
        })
    path = tmp_path / f"99_1_1_mem_search_recall_{PSY_TS}.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_encoding_nan_movie_onset_raises(tmp_path):
    src = _encoding_csv(tmp_path, drop_trial=1)
    with pytest.raises(ValueError) as exc:
        psychopy_encoding.convert_file(str(src), str(tmp_path / "out_events.tsv"), dry_run=True)
    msg = str(exc.value)
    assert src.name in msg and "trial 2" in msg and "movies.started" in msg


def test_retrieval_nan_prompt_onset_raises(tmp_path):
    rows = [{"trials_recall.thisN": None, "wait.started": 10.0},
            {"trials_recall.thisN": 0.0, "wait.started": None,
             "recall1.started": 20.0, "text_recallany.started": 20.5,
             "text_recallany.stopped": 22.0, "text_plus.started": None,
             "recall1.stopped": 80.0, "movie_name": "Film One", "condition": 1,
             "mov_len": 90.0, "style": "s", "free_recall_position": 1}]
    src = tmp_path / f"99_1_free_recall_recall_{PSY_TS}.csv"
    pd.DataFrame(rows).to_csv(src, index=False)
    with pytest.raises(ValueError) as exc:
        psychopy_retrieval.convert_file(str(src), str(tmp_path / "out_events.tsv"), dry_run=True)
    msg = str(exc.value)
    assert src.name in msg and "trial 1" in msg and "text_plus.started" in msg
