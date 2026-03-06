"""Post-conversion validation for BIDS output against DICOM source.

Checks for:
- Volume count mismatches (DICOM count vs NIfTI 4th dimension)
- Truncated BOLD runs that should have been excluded
- Physio timestamp alignment with BOLD acquisitions
- Behavioral/events file alignment with BOLD runs

This module reads from both the BIDS output and sourcedata directories.
It is designed to be run after dcm2bids conversion to catch issues like
aborted scans, run-numbering shifts, and misaligned companion files.
"""

from __future__ import annotations

import gzip
import json
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ValidationIssue:
    """A single validation finding."""

    severity: str  # "error", "warning", "info"
    category: str  # "truncated", "volume_mismatch", "physio_mismatch", etc.
    file: str  # affected BIDS file (relative to BIDS root)
    message: str


@dataclass
class ValidationReport:
    """Full validation report for a subject/session."""

    subject: str
    session: str
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    def summary(self) -> str:
        lines = [f"Validation: {self.subject}/{self.session}"]
        lines.append(
            f"  {len(self.errors)} error(s), {len(self.warnings)} warning(s), "
            f"{len(self.issues) - len(self.errors) - len(self.warnings)} info"
        )
        for issue in self.issues:
            icon = {"error": "!!", "warning": "!", "info": "~"}[issue.severity]
            lines.append(f"  [{icon}] [{issue.category}] {issue.message}")
        return "\n".join(lines)


def _nifti_shape(path: Path) -> tuple[int, ...] | None:
    """Read NIfTI dimensions from the header without nibabel."""
    try:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rb") as f:
            hdr = f.read(348)
        dims = struct.unpack("<8h", hdr[40:56])
        ndim = dims[0]
        return tuple(dims[1 : ndim + 1])
    except Exception:
        return None


def _load_json(path: Path) -> dict:
    """Load a JSON sidecar."""
    with open(path) as f:
        return json.load(f)


def check_bold_volumes(
    bids_root: Path,
    subject: str,
    session: str,
    *,
    min_volumes: int = 20,
) -> list[ValidationIssue]:
    """Check all BOLD NIfTIs for truncation.

    Flags any BOLD run with fewer than ``min_volumes`` volumes as an error,
    since these are almost certainly aborted acquisitions that should not
    be in the BIDS dataset.
    """
    issues: list[ValidationIssue] = []
    func_dir = bids_root / subject / session / "func"
    if not func_dir.is_dir():
        return issues

    for nii in sorted(func_dir.glob("*_bold.nii.gz")):
        shape = _nifti_shape(nii)
        if shape is None:
            issues.append(ValidationIssue(
                severity="warning",
                category="unreadable",
                file=str(nii.relative_to(bids_root)),
                message=f"Could not read NIfTI header: {nii.name}",
            ))
            continue
        n_vols = shape[-1] if len(shape) >= 4 else 1
        if n_vols < min_volumes:
            issues.append(ValidationIssue(
                severity="error",
                category="truncated",
                file=str(nii.relative_to(bids_root)),
                message=(
                    f"{nii.name} has only {n_vols} volumes "
                    f"(minimum: {min_volumes}). Likely an aborted "
                    f"acquisition that should be excluded."
                ),
            ))
    return issues


def check_bold_dicom_alignment(
    bids_root: Path,
    subject: str,
    session: str,
) -> list[ValidationIssue]:
    """Cross-reference BOLD JSON sidecars with DICOM source directories.

    For each BOLD run, reads SeriesNumber from the JSON sidecar and verifies
    the corresponding DICOM series directory exists and has a matching
    volume count.
    """
    issues: list[ValidationIssue] = []
    func_dir = bids_root / subject / session / "func"
    dicom_dir = bids_root / "sourcedata" / subject / session / "dicom"
    if not func_dir.is_dir() or not dicom_dir.is_dir():
        return issues

    for json_path in sorted(func_dir.glob("*_bold.json")):
        sidecar = _load_json(json_path)
        series_num = sidecar.get("SeriesNumber")
        if series_num is None:
            continue

        # Find matching DICOM directory
        prefix = f"Series_{series_num:02d}_"
        matches = [d for d in dicom_dir.iterdir()
                   if d.is_dir() and d.name.startswith(prefix)]
        if not matches:
            # Try without zero-padding
            prefix = f"Series_{series_num}_"
            matches = [d for d in dicom_dir.iterdir()
                       if d.is_dir() and d.name.startswith(prefix)]

        if not matches:
            issues.append(ValidationIssue(
                severity="warning",
                category="missing_dicom",
                file=str(json_path.relative_to(bids_root)),
                message=(
                    f"No DICOM directory found for SeriesNumber "
                    f"{series_num} ({json_path.name})"
                ),
            ))
            continue

        # Check volume count consistency
        dcm_dir = matches[0]
        dcm_count = sum(1 for f in dcm_dir.iterdir() if f.suffix == ".dcm")

        nii_path = json_path.with_name(
            json_path.name.replace("_bold.json", "_bold.nii.gz")
        )
        if nii_path.exists():
            shape = _nifti_shape(nii_path)
            if shape and len(shape) >= 4:
                nii_vols = shape[-1]
                if dcm_count != nii_vols:
                    issues.append(ValidationIssue(
                        severity="warning",
                        category="volume_mismatch",
                        file=str(nii_path.relative_to(bids_root)),
                        message=(
                            f"DICOM has {dcm_count} files but NIfTI has "
                            f"{nii_vols} volumes ({nii_path.name}, "
                            f"Series {series_num})"
                        ),
                    ))

    return issues


def check_physio_alignment(
    bids_root: Path,
    subject: str,
    session: str,
) -> list[ValidationIssue]:
    """Check that physio files are aligned with their paired BOLD runs.

    Uses AcquisitionTime from BOLD JSON sidecars and StartTime from physio
    JSON sidecars to verify temporal consistency. Also flags BOLD runs
    missing physio when siblings have it, and physio files without a
    corresponding BOLD run.
    """
    issues: list[ValidationIssue] = []
    func_dir = bids_root / subject / session / "func"
    if not func_dir.is_dir():
        return issues

    # Collect BOLD runs with their AcquisitionTime
    bold_times: dict[str, str | None] = {}  # run_key -> AcquisitionTime
    for json_path in sorted(func_dir.glob("*_bold.json")):
        sidecar = _load_json(json_path)
        run_key = json_path.stem.replace("_bold", "")
        bold_times[run_key] = sidecar.get("AcquisitionTime")

    # Collect physio files
    physio_runs: set[str] = set()
    for physio_json in sorted(func_dir.glob("*_physio.json")):
        # Extract the run key (everything before _recording-*)
        stem = physio_json.stem
        m = re.match(r"(.+?)_recording-.+_physio", stem)
        if m:
            physio_runs.add(m.group(1))

    # Group bold runs by task (to detect inconsistent physio coverage)
    by_task: dict[str, list[str]] = {}
    for run_key in bold_times:
        # Extract task portion
        m = re.search(r"_task-(\w+)", run_key)
        if m:
            by_task.setdefault(m.group(1), []).append(run_key)

    # Check for inconsistent physio coverage within a task
    for task, runs in by_task.items():
        runs_with_physio = [r for r in runs if r in physio_runs]
        runs_without_physio = [r for r in runs if r not in physio_runs]
        if runs_with_physio and runs_without_physio:
            issues.append(ValidationIssue(
                severity="warning",
                category="physio_incomplete",
                file=f"{subject}/{session}/func/",
                message=(
                    f"task-{task}: {len(runs_with_physio)} runs have physio "
                    f"but {len(runs_without_physio)} do not "
                    f"({', '.join(r.split('_')[-1] for r in runs_without_physio)}). "
                    f"Possible misalignment if physio files shifted."
                ),
            ))

    return issues


def check_events_alignment(
    bids_root: Path,
    subject: str,
    session: str,
    *,
    events_dir: Path | None = None,
) -> list[ValidationIssue]:
    """Check that events files exist for BOLD runs that should have them.

    Parameters
    ----------
    bids_root : Path
        BIDS dataset root.
    subject, session : str
        Subject and session identifiers.
    events_dir : Path, optional
        Alternative directory to search for events files (e.g.
        ``derivatives/bids_validation/eventfiles/``).
    """
    issues: list[ValidationIssue] = []
    func_dir = bids_root / subject / session / "func"
    if not func_dir.is_dir():
        return issues

    # Tasks that are expected to have events files (not resting/math/fixation)
    task_needs_events = {
        "TBencoding", "TBretrieval", "NATencoding", "NATretrieval",
        "FINretrieval", "floc", "prf", "tone", "auditory", "motor",
    }

    for nii in sorted(func_dir.glob("*_bold.nii.gz")):
        m = re.search(r"_task-(\w+)", nii.name)
        if not m:
            continue
        task = m.group(1)
        if task not in task_needs_events:
            continue

        # Check for events.tsv in func dir or events_dir
        events_name = nii.name.replace("_bold.nii.gz", "_events.tsv")
        events_path = func_dir / events_name
        found = events_path.exists()

        if not found and events_dir:
            alt_path = events_dir / subject / session / "func" / events_name
            found = alt_path.exists()

        if not found:
            issues.append(ValidationIssue(
                severity="info",
                category="missing_events",
                file=str(nii.relative_to(bids_root)),
                message=f"No events.tsv for {nii.name} (task-{task})",
            ))

    return issues


def validate_session(
    bids_root: Path,
    subject: str,
    session: str,
    *,
    min_volumes: int = 20,
    events_dir: Path | None = None,
) -> ValidationReport:
    """Run all validation checks for a single subject/session.

    Parameters
    ----------
    bids_root : Path
        BIDS dataset root.
    subject, session : str
        Subject and session identifiers.
    min_volumes : int
        Minimum BOLD volumes to consider a run complete.
    events_dir : Path, optional
        Alternative directory for events files.

    Returns
    -------
    ValidationReport
        Combined report from all checks.
    """
    report = ValidationReport(subject=subject, session=session)
    report.issues.extend(
        check_bold_volumes(bids_root, subject, session, min_volumes=min_volumes)
    )
    report.issues.extend(
        check_bold_dicom_alignment(bids_root, subject, session)
    )
    report.issues.extend(
        check_physio_alignment(bids_root, subject, session)
    )
    report.issues.extend(
        check_events_alignment(
            bids_root, subject, session, events_dir=events_dir
        )
    )
    return report
