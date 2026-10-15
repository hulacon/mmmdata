"""check_events_alignment must recognise the task of a multi-run BOLD.

``_task-(\\w+)`` also matches underscores, so ``task-floc_run-01_bold``
parsed as task ``floc_run``; that is not in the needs-events set, and every
multi-run task's missing events file went unreported.
"""

from dcm2bids_config.validate_bids import check_events_alignment


def _bold(func, name):
    (func / name).write_bytes(b"")


def test_missing_events_reported_for_multi_run_task(tmp_path):
    func = tmp_path / "sub-01" / "ses-01" / "func"
    func.mkdir(parents=True)
    _bold(func, "sub-01_ses-01_task-floc_run-01_bold.nii.gz")
    _bold(func, "sub-01_ses-01_task-tone_bold.nii.gz")
    issues = check_events_alignment(tmp_path, "sub-01", "ses-01")
    flagged = sorted(i.file.rsplit("/", 1)[-1] for i in issues)
    assert flagged == [
        "sub-01_ses-01_task-floc_run-01_bold.nii.gz",
        "sub-01_ses-01_task-tone_bold.nii.gz",
    ]


def test_present_events_not_reported(tmp_path):
    func = tmp_path / "sub-01" / "ses-01" / "func"
    func.mkdir(parents=True)
    _bold(func, "sub-01_ses-01_task-floc_run-01_bold.nii.gz")
    (func / "sub-01_ses-01_task-floc_run-01_events.tsv").write_text("onset\n")
    assert check_events_alignment(tmp_path, "sub-01", "ses-01") == []


def test_physio_gap_within_multi_run_task_reported(tmp_path):
    """Same regex in check_physio_alignment: each run became its own task, so
    a run missing physio beside siblings that have it was never flagged."""
    from dcm2bids_config.validate_bids import check_physio_alignment

    func = tmp_path / "sub-01" / "ses-01" / "func"
    func.mkdir(parents=True)
    for run in (1, 2):
        (func / f"sub-01_ses-01_task-floc_run-0{run}_bold.json").write_text("{}")
    (func / "sub-01_ses-01_task-floc_run-01_recording-cardiac_physio.json").write_text("{}")
    issues = check_physio_alignment(tmp_path, "sub-01", "ses-01")
    assert [i.category for i in issues] == ["physio_incomplete"]
    assert "task-floc:" in issues[0].message
