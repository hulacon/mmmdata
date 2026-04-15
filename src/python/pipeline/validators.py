"""Postcondition validators for pipeline steps.

One function per step. All share the signature
``(subject, session, bids_root=None, **kwargs) -> ValidationResult``.

Validators check that outputs *exist and are structurally valid*. They do
not re-run analyses. The status values they return:

* ``skipped``  — no raw inputs exist for this sub/ses (not applicable)
* ``missing``  — outputs are entirely absent (ready to submit)
* ``partial``  — some outputs exist, some are missing
* ``complete`` — all expected outputs present
* ``error``    — outputs present but malformed

Expected counts are derived from the raw BIDS ``func/`` directory
(authoritative source of "what should exist"). This avoids coupling to
``dataset_expectations.toml`` for basic presence checks while remaining
consistent with the actual acquired data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from neuroimaging.constants import (
    DEFAULT_BIDS_ROOT,
    DERIVATIVES_DIRS,
)
from neuroimaging.io import _resolve_bids_root, find_fmriprep_runs

from .steps import ValidationResult


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _raw_bold_files(subject: str, session: str, bids_root: Path) -> list[Path]:
    """Return sorted list of raw BOLD NIfTIs in sub-XX/ses-YY/func/."""
    func_dir = bids_root / f"sub-{subject}" / f"ses-{session}" / "func"
    if not func_dir.exists():
        return []
    return sorted(func_dir.glob("*_bold.nii.gz"))


def _session_has_raw(subject: str, session: str, bids_root: Path) -> bool:
    """Return True if raw BIDS data exists for this sub/ses."""
    ses_dir = bids_root / f"sub-{subject}" / f"ses-{session}"
    return ses_dir.exists()


def _make_result(
    step: str,
    subject: str,
    session: str,
    expected: int,
    found: int,
    details: Optional[list[str]] = None,
    metrics: Optional[dict] = None,
    override_status: Optional[str] = None,
) -> ValidationResult:
    """Assemble a ValidationResult with status inferred from counts."""
    details = details or []
    metrics = metrics or {}
    if override_status is not None:
        status = override_status
    elif expected == 0:
        status = "skipped"
    elif found == 0:
        status = "missing"
    elif found < expected:
        status = "partial"
    elif found == expected:
        status = "complete"
    else:  # found > expected — unexpected, but don't fail
        status = "complete"
        details.append(f"Found {found} outputs; only {expected} expected.")
    return ValidationResult(
        step=step,
        subject=subject,
        session=session,
        status=status,
        expected=expected,
        found=found,
        details=details,
        metrics=metrics,
    )


# ---------------------------------------------------------------------------
# Layer 0: raw data
# ---------------------------------------------------------------------------

def validate_bidsification(
    subject: str,
    session: str,
    bids_root: Optional[Path] = None,
) -> ValidationResult:
    """Check raw BIDS tree: BOLD + sidecar JSON + events TSV present.

    Considered ``complete`` if every BOLD has a JSON sidecar. Events TSV
    presence is task-dependent (resting-state has none) and only
    contributes to ``details``, not status.
    """
    bids_root = _resolve_bids_root(bids_root)
    if not _session_has_raw(subject, session, bids_root):
        return _make_result("bidsification", subject, session, 0, 0,
                            details=["No raw sub/ses directory."])

    bolds = _raw_bold_files(subject, session, bids_root)
    expected = len(bolds)
    details: list[str] = []

    json_count = 0
    for bold in bolds:
        sidecar = bold.with_name(bold.name.replace("_bold.nii.gz", "_bold.json"))
        if sidecar.exists():
            json_count += 1
        else:
            details.append(f"Missing sidecar: {sidecar.name}")

    return _make_result(
        "bidsification", subject, session, expected, json_count, details
    )


# ---------------------------------------------------------------------------
# Layer 1a: MRIQC
# ---------------------------------------------------------------------------

def validate_mriqc(
    subject: str,
    session: str,
    bids_root: Optional[Path] = None,
) -> ValidationResult:
    """Check MRIQC JSON outputs exist for every BOLD run."""
    bids_root = _resolve_bids_root(bids_root)
    bolds = _raw_bold_files(subject, session, bids_root)
    if not bolds:
        return _make_result("mriqc", subject, session, 0, 0,
                            details=["No raw BOLDs to QC."])

    mriqc_dir = (
        bids_root / DERIVATIVES_DIRS["mriqc"]
        / f"sub-{subject}" / f"ses-{session}" / "func"
    )
    expected = len(bolds)
    found = 0
    missing: list[str] = []
    for bold in bolds:
        # MRIQC writes *_bold.json (IQM) alongside *_timeseries.json
        iqm_name = bold.name.replace("_bold.nii.gz", "_bold.json")
        if (mriqc_dir / iqm_name).exists():
            found += 1
        else:
            missing.append(iqm_name)

    details = [f"Missing: {n}" for n in missing[:3]]
    if len(missing) > 3:
        details.append(f"... and {len(missing) - 3} more")
    return _make_result("mriqc", subject, session, expected, found, details)


# ---------------------------------------------------------------------------
# Layer 1b: NORDIC
# ---------------------------------------------------------------------------

def validate_nordic_denoise(
    subject: str,
    session: str,
    bids_root: Optional[Path] = None,
) -> ValidationResult:
    """Check NORDIC-denoised BOLD exists for every raw BOLD."""
    bids_root = _resolve_bids_root(bids_root)
    bolds = _raw_bold_files(subject, session, bids_root)
    if not bolds:
        return _make_result("nordic_denoise", subject, session, 0, 0,
                            details=["No raw BOLDs to denoise."])

    nordic_dir = (
        bids_root / DERIVATIVES_DIRS["nordic"]
        / f"sub-{subject}" / f"ses-{session}" / "func"
    )
    expected = len(bolds)
    found = 0
    missing: list[str] = []
    for bold in bolds:
        if (nordic_dir / bold.name).exists():
            found += 1
        else:
            missing.append(bold.name)

    details = [f"Missing: {n}" for n in missing[:3]]
    if len(missing) > 3:
        details.append(f"... and {len(missing) - 3} more")
    return _make_result(
        "nordic_denoise", subject, session, expected, found, details
    )


def validate_nordic_bids_input(
    subject: str,
    session: str,
    bids_root: Optional[Path] = None,
) -> ValidationResult:
    """Check the NORDIC BIDS input tree for fMRIPrep: BOLDs + sidecars + fmaps."""
    bids_root = _resolve_bids_root(bids_root)
    bolds = _raw_bold_files(subject, session, bids_root)
    if not bolds:
        return _make_result("nordic_bids_input", subject, session, 0, 0,
                            details=["No raw BOLDs."])

    bids_input = (
        bids_root / DERIVATIVES_DIRS["nordic"] / "bids_input"
        / f"sub-{subject}" / f"ses-{session}"
    )
    func_dir = bids_input / "func"
    fmap_dir = bids_input / "fmap"

    expected = len(bolds)
    found = sum(1 for b in bolds if (func_dir / b.name).exists())

    details: list[str] = []
    if not fmap_dir.exists():
        details.append("fmap/ directory missing")
    elif not any(fmap_dir.iterdir()):
        details.append("fmap/ directory empty")

    return _make_result(
        "nordic_bids_input", subject, session, expected, found, details
    )


# ---------------------------------------------------------------------------
# Layer 1c: fMRIPrep (two variants, same check)
# ---------------------------------------------------------------------------

def _validate_fmriprep_variant(
    step_name: str,
    variant: str,
    subject: str,
    session: str,
    bids_root: Optional[Path] = None,
) -> ValidationResult:
    """Shared logic for fmriprep and fmriprep_nordic validators."""
    bids_root = _resolve_bids_root(bids_root)
    bolds = _raw_bold_files(subject, session, bids_root)
    if not bolds:
        return _make_result(step_name, subject, session, 0, 0,
                            details=[f"No raw BOLDs for {variant}."])

    runs = find_fmriprep_runs(
        subject=subject, session=session, variant=variant, bids_root=bids_root,
    )
    expected = len(bolds)
    found = len(runs)

    details: list[str] = []
    # Per-run completeness spot-check (first few only)
    incomplete = []
    for r in runs:
        missing_fields = [
            f for f in ("bold", "mask", "boldref", "confounds")
            if getattr(r, f) is None
        ]
        if missing_fields:
            incomplete.append(f"{r.entity_prefix}: missing {missing_fields}")
    if incomplete:
        details.extend(incomplete[:3])
        if len(incomplete) > 3:
            details.append(f"... and {len(incomplete) - 3} more incomplete")
        return _make_result(
            step_name, subject, session, expected, found, details,
            override_status="error",
        )

    return _make_result(step_name, subject, session, expected, found, details)


def validate_fmriprep(
    subject: str, session: str, bids_root: Optional[Path] = None,
) -> ValidationResult:
    """Check fMRIPrep (original) outputs for every BOLD run."""
    return _validate_fmriprep_variant(
        "fmriprep", "fmriprep", subject, session, bids_root
    )


def validate_fmriprep_nordic(
    subject: str, session: str, bids_root: Optional[Path] = None,
) -> ValidationResult:
    """Check fMRIPrep (NORDIC) outputs for every BOLD run."""
    return _validate_fmriprep_variant(
        "fmriprep_nordic", "fmriprep_nordic", subject, session, bids_root
    )


# ---------------------------------------------------------------------------
# Layer 1d: Preprocessing QC decisions (human-reviewed gate to Layer 2)
# ---------------------------------------------------------------------------

def validate_preprocessing_qc(
    subject: str,
    session: str,
    bids_root: Optional[Path] = None,
) -> ValidationResult:
    """Check QC decisions TSV exists. Warn if still at default (un-reviewed)."""
    bids_root = _resolve_bids_root(bids_root)
    if not _session_has_raw(subject, session, bids_root):
        return _make_result("preprocessing_qc", subject, session, 0, 0,
                            details=["No raw sub/ses."])

    qc_path = (
        bids_root / DERIVATIVES_DIRS["preprocessing_qc"]
        / f"sub-{subject}"
        / f"sub-{subject}_ses-{session}_qc_decisions.tsv"
    )
    expected = 1
    if not qc_path.exists():
        return _make_result(
            "preprocessing_qc", subject, session, expected, 0,
            details=[f"QC decisions file not found: {qc_path.name}"],
        )

    # File exists. Lightweight sanity check (don't import pandas if we can help it).
    details: list[str] = []
    try:
        import pandas as pd
        df = pd.read_csv(qc_path, sep="\t")
        required_cols = {"task", "run", "exclude", "nordic"}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            return _make_result(
                "preprocessing_qc", subject, session, expected, 1,
                details=[f"QC TSV missing columns: {sorted(missing_cols)}"],
                override_status="error",
            )
    except Exception as exc:
        return _make_result(
            "preprocessing_qc", subject, session, expected, 1,
            details=[f"Failed to parse QC TSV: {exc}"],
            override_status="error",
        )

    return _make_result(
        "preprocessing_qc", subject, session, expected, 1, details
    )


# ---------------------------------------------------------------------------
# Layer 2: analysis streams
# ---------------------------------------------------------------------------

def _validate_stream(
    step_name: str,
    stream: str,
    subject: str,
    session: str,
    bids_root: Optional[Path] = None,
) -> ValidationResult:
    """Shared logic: does the ready/ output directory have files for each run in this stream?"""
    from neuroimaging.constants import TASK_STREAM_MAP

    bids_root = _resolve_bids_root(bids_root)
    bolds = _raw_bold_files(subject, session, bids_root)
    if not bolds:
        return _make_result(step_name, subject, session, 0, 0,
                            details=["No raw BOLDs."])

    # Parse task from filenames to determine which runs belong in this stream
    stream_bolds: list[Path] = []
    for bold in bolds:
        # Filename: sub-XX_ses-YY_task-NAME[_run-NN]_bold.nii.gz
        name = bold.name
        try:
            task = name.split("_task-")[1].split("_")[0]
        except IndexError:
            continue
        if TASK_STREAM_MAP.get(task) == stream:
            stream_bolds.append(bold)

    expected = len(stream_bolds)
    if expected == 0:
        return _make_result(step_name, subject, session, 0, 0,
                            details=[f"No runs route to {stream!r} for this session."])

    stream_dir = (
        bids_root / DERIVATIVES_DIRS["ready"] / stream
        / f"sub-{subject}" / f"ses-{session}" / "func"
    )
    # GLMsingle produces a confounds_ready.tsv; naturalistic/connectivity produce preproc_bold
    if stream == "glmsingle":
        marker_suffix = "_desc-confounds_ready.tsv"
    else:
        marker_suffix = "_desc-preproc_bold.nii.gz"

    found = 0
    if stream_dir.exists():
        for bold in stream_bolds:
            prefix = bold.name.replace("_bold.nii.gz", "")
            # Glob because space/hemi parts vary
            if list(stream_dir.glob(f"{prefix}*{marker_suffix}")):
                found += 1

    return _make_result(step_name, subject, session, expected, found)


def validate_stream_glmsingle(
    subject: str, session: str, bids_root: Optional[Path] = None,
) -> ValidationResult:
    """Check ready/glmsingle/ outputs (confounds_ready + outliers_mask TSVs)."""
    return _validate_stream(
        "stream_glmsingle", "glmsingle", subject, session, bids_root
    )


def validate_stream_naturalistic(
    subject: str, session: str, bids_root: Optional[Path] = None,
) -> ValidationResult:
    """Check ready/naturalistic/ outputs (cleaned BOLD in MNI + fsaverage6)."""
    return _validate_stream(
        "stream_naturalistic", "naturalistic", subject, session, bids_root
    )


def validate_stream_connectivity(
    subject: str, session: str, bids_root: Optional[Path] = None,
) -> ValidationResult:
    """Check ready/connectivity/ outputs (cleaned BOLD in MNI + fsaverage6)."""
    return _validate_stream(
        "stream_connectivity", "connectivity", subject, session, bids_root
    )
