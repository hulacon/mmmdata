"""Project-wide pipeline status reporting.

The two main entry points:

* :func:`pipeline_status` — DataFrame of (subject, session, step, status)
  for all configured sub/ses/step combinations.
* :func:`runnable_sessions` — (subject, session) pairs where all
  prerequisites of a given step are complete but the step itself is not.

Both discover subjects/sessions by scanning the raw BIDS tree (``sub-*/ses-*``).
Callers can restrict to specific subjects/sessions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from neuroimaging.io import _resolve_bids_root

from .steps import STEPS, ValidationResult, all_steps, get_step


# ---------------------------------------------------------------------------
# Sub/ses discovery
# ---------------------------------------------------------------------------

def _discover_sessions(
    subjects: Optional[Sequence[str]],
    sessions: Optional[Sequence[str]],
    bids_root: Path,
) -> list[tuple[str, str]]:
    """Return sorted list of (subject, session) pairs present in raw BIDS."""
    sub_glob = [f"sub-{s}" for s in subjects] if subjects else ["sub-*"]
    pairs: set[tuple[str, str]] = set()
    for pattern in sub_glob:
        for sub_dir in sorted(bids_root.glob(pattern)):
            if not sub_dir.is_dir():
                continue
            subject = sub_dir.name.removeprefix("sub-")
            if subject.startswith(".") or not subject[0].isdigit():
                continue
            for ses_dir in sorted(sub_dir.glob("ses-*")):
                if not ses_dir.is_dir():
                    continue
                session = ses_dir.name.removeprefix("ses-")
                if sessions is not None and session not in sessions:
                    continue
                pairs.add((subject, session))
    return sorted(pairs)


# ---------------------------------------------------------------------------
# Status table
# ---------------------------------------------------------------------------

def pipeline_status(
    subjects: Optional[Sequence[str]] = None,
    sessions: Optional[Sequence[str]] = None,
    steps: Optional[Sequence[str]] = None,
    bids_root: Optional[Path] = None,
) -> pd.DataFrame:
    """Run all validators for a set of sub/ses/step combinations.

    Parameters
    ----------
    subjects, sessions : sequence of str, optional
        Zero-padded IDs without prefixes. If None, all sub/ses found in
        raw BIDS are included.
    steps : sequence of str, optional
        Step names. If None, all registered steps are included (in
        topological order).
    bids_root : Path, optional
        BIDS root. If None, resolved via ``core.config``.

    Returns
    -------
    pd.DataFrame
        Columns: subject, session, step, status, expected, found, details, metrics.
        One row per (sub, ses, step) triple. ``details`` is a semicolon-joined
        string for easy TSV export; ``metrics`` is a dict.
    """
    bids_root = _resolve_bids_root(bids_root)
    pairs = _discover_sessions(subjects, sessions, bids_root)

    all_step_specs = all_steps()
    if steps is not None:
        step_names = set(steps)
        all_step_specs = [s for s in all_step_specs if s.name in step_names]

    rows: list[dict] = []
    for subject, session in pairs:
        for spec in all_step_specs:
            result = spec.validate(subject, session, bids_root=bids_root)
            rows.append(_result_to_row(result))
    return pd.DataFrame(rows)


def _result_to_row(result: ValidationResult) -> dict:
    return {
        "subject": result.subject,
        "session": result.session,
        "step": result.step,
        "status": result.status,
        "expected": result.expected,
        "found": result.found,
        "details": "; ".join(result.details),
        "metrics": result.metrics,
    }


# ---------------------------------------------------------------------------
# Runnable sessions
# ---------------------------------------------------------------------------

def runnable_sessions(
    step_name: str,
    subjects: Optional[Sequence[str]] = None,
    sessions: Optional[Sequence[str]] = None,
    bids_root: Optional[Path] = None,
) -> list[tuple[str, str]]:
    """Return (subject, session) pairs where this step can be submitted.

    A pair is runnable iff:

    1. All prerequisites (``step.depends_on``) are ``complete`` for this pair.
    2. This step is not already ``complete`` for this pair.

    Pairs where the step is ``partial`` are still returned (a re-run can
    fill gaps, and the submission script is expected to be idempotent).
    """
    bids_root = _resolve_bids_root(bids_root)
    spec = get_step(step_name)
    pairs = _discover_sessions(subjects, sessions, bids_root)

    runnable: list[tuple[str, str]] = []
    for subject, session in pairs:
        # Check prerequisites
        prereqs_ok = True
        for dep_name in spec.depends_on:
            dep = get_step(dep_name)
            dep_result = dep.validate(subject, session, bids_root=bids_root)
            if not dep_result.is_complete:
                prereqs_ok = False
                break
        if not prereqs_ok:
            continue

        # Check this step
        result = spec.validate(subject, session, bids_root=bids_root)
        if result.status in ("complete", "skipped"):
            continue
        runnable.append((subject, session))

    return runnable


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def status_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate a ``pipeline_status`` DataFrame into a step-x-status matrix.

    Returns
    -------
    pd.DataFrame
        Rows indexed by step (in DAG order); columns are status values;
        cell values are counts.
    """
    if df.empty:
        return df
    # Preserve topological order of steps
    step_order = [s for s in [spec.name for spec in all_steps()] if s in df["step"].unique()]
    pivot = (
        df.groupby(["step", "status"]).size().unstack(fill_value=0)
    )
    return pivot.reindex(step_order)
