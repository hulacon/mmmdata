"""Localizer sessions and the two-designs-under-one-label trap.

Two facts about the dataset drive these tests (record:
mmmdata-agents ``docs/results/task-instructions-provenance.md``):

* Functional localizers were acquired in two places — the dedicated
  localizer sessions (ses-02/03) and the final session (ses-30). A session
  list that names only the first pair silently drops every ses-30 run.
* ``motor`` and ``auditory`` each ran under one task label but two different
  programs, one per session group, and no subject ran both. Selecting on
  ``task=motor`` across sessions therefore pools two designs, perfectly
  confounded with subject. Nothing in BIDS marks this; the code has to.
"""

from pathlib import Path

import pytest

from neuroimaging import io as nio
from neuroimaging.constants import (
    LOCALIZER_DESIGNS,
    LOCALIZER_SESSIONS,
    TASK_STREAM_MAP,
    MixedLocalizerDesignError,
    check_single_design,
    localizer_design,
)


# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

def test_localizer_sessions_include_final_session():
    # The bug: ("02", "03") excluded every ses-30 localizer.
    assert "30" in LOCALIZER_SESSIONS
    assert {"02", "03"} <= set(LOCALIZER_SESSIONS)


def test_localizer_sessions_are_zero_padded_and_unique():
    assert all(len(s) == 2 and s.isdigit() for s in LOCALIZER_SESSIONS)
    assert len(set(LOCALIZER_SESSIONS)) == len(LOCALIZER_SESSIONS)


def test_split_design_tasks_are_known_localizers():
    for task in LOCALIZER_DESIGNS:
        assert task in TASK_STREAM_MAP, f"{task} is not a known task label"


def test_every_design_session_is_a_localizer_session():
    for task, designs in LOCALIZER_DESIGNS.items():
        assert len(designs) >= 2, f"{task} lists a single design; drop it"
        for label, sessions in designs.items():
            assert sessions, f"{task}/{label} has no sessions"
            assert set(sessions) <= set(LOCALIZER_SESSIONS), (task, label)


def test_designs_within_a_task_do_not_share_sessions():
    for task, designs in LOCALIZER_DESIGNS.items():
        seen: dict[str, str] = {}
        for label, sessions in designs.items():
            for s in sessions:
                assert s not in seen, f"{task}: ses-{s} in {seen[s]} and {label}"
                seen[s] = label


def test_motor_and_auditory_are_split_by_session_group():
    for task in ("motor", "auditory"):
        assert localizer_design(task, "02") == localizer_design(task, "03")
        assert localizer_design(task, "30") != localizer_design(task, "02")


def test_localizer_design_is_none_for_single_design_tasks():
    assert localizer_design("floc", "02") is None
    assert localizer_design("TBencoding", "04") is None


def test_localizer_design_rejects_unknown_session_for_split_task():
    with pytest.raises(ValueError, match="ses-04"):
        localizer_design("motor", "04")


def test_check_single_design_passes_within_one_design():
    check_single_design("motor", ["02", "03"])
    check_single_design("motor", ["30", "30"])
    check_single_design("floc", ["02", "30"])  # not a split task
    check_single_design("motor", [])


def test_check_single_design_raises_across_designs():
    with pytest.raises(MixedLocalizerDesignError) as exc:
        check_single_design("motor", ["02", "30"])
    msg = str(exc.value)
    assert "motor" in msg
    assert "ses-02" in msg and "ses-30" in msg
    # The error is a ValueError so existing callers can catch it generically.
    assert isinstance(exc.value, ValueError)


# ---------------------------------------------------------------------------
# find_fmriprep_runs guard
# ---------------------------------------------------------------------------

def _touch_confounds(root: Path, sub: str, ses: str, task: str, run: str | None):
    func = root / "derivatives" / "fmriprep" / f"sub-{sub}" / f"ses-{ses}" / "func"
    func.mkdir(parents=True, exist_ok=True)
    run_part = f"_run-{run}" if run else ""
    (func / f"sub-{sub}_ses-{ses}_task-{task}{run_part}_desc-confounds_timeseries.tsv").touch()


@pytest.fixture
def two_cohort_root(tmp_path):
    """One subject per design group, as in the real dataset (no overlap)."""
    # Later cohort: motor once in a localizer session.
    _touch_confounds(tmp_path, "aa", "02", "motor", None)
    _touch_confounds(tmp_path, "aa", "02", "floc", "01")
    # First cohort: motor twice in the final session.
    _touch_confounds(tmp_path, "bb", "30", "motor", "01")
    _touch_confounds(tmp_path, "bb", "30", "motor", "02")
    _touch_confounds(tmp_path, "bb", "03", "floc", "01")
    return tmp_path


def test_task_motor_across_sessions_raises(two_cohort_root):
    with pytest.raises(MixedLocalizerDesignError) as exc:
        nio.find_fmriprep_runs(task="motor", bids_root=two_cohort_root)
    msg = str(exc.value)
    assert "sub-aa" in msg and "sub-bb" in msg
    assert "allow_mixed_designs" in msg


def test_task_motor_within_one_session_group_is_fine(two_cohort_root):
    runs = nio.find_fmriprep_runs(task="motor", session="30", bids_root=two_cohort_root)
    assert [r.run for r in runs] == ["01", "02"]
    runs = nio.find_fmriprep_runs(task="motor", subject="aa", bids_root=two_cohort_root)
    assert len(runs) == 1


def test_task_motor_opt_in_pooling(two_cohort_root):
    runs = nio.find_fmriprep_runs(
        task="motor", bids_root=two_cohort_root, allow_mixed_designs=True
    )
    assert {r.session for r in runs} == {"02", "30"}


def test_non_split_task_across_sessions_is_fine(two_cohort_root):
    runs = nio.find_fmriprep_runs(task="floc", bids_root=two_cohort_root)
    assert {r.session for r in runs} == {"02", "03"}


def test_unfiltered_sweep_is_not_blocked(two_cohort_root):
    # No task filter = a QC-style sweep, not a GLM selection; must not raise.
    runs = nio.find_fmriprep_runs(bids_root=two_cohort_root)
    assert len(runs) == 5
