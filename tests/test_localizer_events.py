"""Auditory localizer events take session from the path and run from the BOLD.

``convert_auditory`` used to hardcode the final session and read the run
entity from the CSV filename. Under the regularised protocol auditory runs in
the localizer sessions, and ``runN`` in the filename is a per-subject counter
(``sess3_run2`` is that session's only run), so both were wrong for every
subject after the first cohort. Motor had the same trap and the same fix.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CONVERTERS = REPO_ROOT / "src" / "python" / "raw2bids_converters"
# The converter modules import each other by bare name (``from common import
# ...``), which resolves only when their own directory is on sys.path -- as it
# is when they run as scripts from that directory.
if str(CONVERTERS) not in sys.path:
    sys.path.insert(0, str(CONVERTERS))

from raw2bids_converters import localizer_events as le  # noqa: E402

# Synthetic subject so the fixture cannot be mistaken for a real record.
SUBJ, SES = 98, 3
STIM_ON, STIM_OFF, WINDOW_END = 0.24, 561.91, 612.0


@pytest.fixture
def tree(tmp_path, monkeypatch):
    """A source CSV named like a regularised-protocol run, and a BIDS root."""
    src = tmp_path / "src" / f"sub-{SUBJ}" / f"ses-{SES:02d}" / "behavioral"
    src.mkdir(parents=True)
    csv = src / f"localizer_auditory_subj{SUBJ}_sess{SES}_run2_YYYY_timing.csv"
    pd.DataFrame([{
        "sub_id": f"sub_{SUBJ}", "task_id": "localizer_auditory",
        "sess_id": SES, "run_id": 2, "trial_id": 1,
        "stim_start": STIM_ON, "stim_end": STIM_OFF,
        "stim_fixation_start": STIM_ON, "stim_fixation_end": WINDOW_END,
    }]).to_csv(csv, index=False)

    bids = tmp_path / "bids"
    func = bids / f"sub-{SUBJ}" / f"ses-{SES:02d}" / "func"
    func.mkdir(parents=True)
    monkeypatch.setattr(le, "BIDS_ROOT", str(bids))
    monkeypatch.setattr(le, "run_duration_s", lambda bold: 408 * 1.5)
    return csv, func


def _stem(run=None):
    run_part = "" if run is None else f"_run-{run:02d}"
    return f"sub-{SUBJ}_ses-{SES:02d}_task-auditory{run_part}"


def test_session_from_path_and_no_run_entity_when_bold_has_none(tree):
    csv, func = tree
    (func / f"{_stem()}_bold.nii.gz").touch()

    subj, ses, run_entity = le.localizer_target(str(csv), "auditory")
    assert (subj, ses, run_entity) == (SUBJ, SES, None)

    out = func / f"{_stem()}_events.tsv"
    le.convert_auditory(str(csv), str(out))
    events = pd.read_csv(out, sep="\t")
    assert set(events["ses_num"]) == {SES}      # not the final session
    assert set(events["run_idx"]) == {1}        # not the filename's run2
    assert list(events["trial_type"]) == ["stimulus", "fixation"]
    assert events["onset"].iloc[0] == pytest.approx(STIM_ON)

    sidecar = json.loads(out.with_suffix(".json").read_text())
    assert "PlaybackRate" in sidecar and "Sources" in sidecar
    assert "+0.000 s" in sidecar["TimingVerification"]


def test_run_entity_from_bold_when_bold_carries_one(tree):
    csv, func = tree
    (func / f"{_stem(2)}_bold.nii.gz").touch()
    assert le.localizer_target(str(csv), "auditory") == (SUBJ, SES, 2)


def test_missing_bold_is_an_error_not_a_guess(tree):
    csv, _ = tree
    with pytest.raises(FileNotFoundError, match="No auditory BOLD"):
        le.localizer_target(str(csv), "auditory")


def test_surplus_acquisition_raises(tree, monkeypatch):
    """A whole-TR lead-in would hide in surplus volumes -- the fLoc bug."""
    csv, func = tree
    (func / f"{_stem()}_bold.nii.gz").touch()
    monkeypatch.setattr(le, "run_duration_s", lambda bold: WINDOW_END + 12.0)
    with pytest.raises(ValueError, match="acceptance bar"):
        le.convert_auditory(str(csv), str(func / f"{_stem()}_events.tsv"),
                            dry_run=True)


def test_filename_subject_disagreeing_with_path_raises(tree):
    csv, func = tree
    wrong = csv.with_name(csv.name.replace(f"subj{SUBJ}", "subj97"))
    csv.rename(wrong)
    with pytest.raises(ValueError, match="The path wins"):
        le.convert_auditory(str(wrong), str(func / f"{_stem()}_events.tsv"),
                            dry_run=True)
