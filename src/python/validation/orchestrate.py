"""Programmatic entry points for MMMData dataset validation.

Catalog-backed since 2026-08-17 (Contract A consumer port): validation
is a query over the catalog's four-state ``resolution`` view
(present / missing / excepted / surplus), not a check engine run
against the frozen manifest.db. The declaration lives in
``<bids_root>/expectations/dataset.toml`` and is ingested by
``scripts/catalog_expectations.py``; judgment is a SQL join inside
catalog.duckdb. All query logic lives in ``core.catalog``; this module
is the thin validation-flavored face of it.

The legacy engine (checks.py + dataset_expectations.toml) is retired.
Its deep-content checks (volume counts, events rows/columns/timing,
JSON sidecars) were declared out of scope of expectations
schema_version 1.0 — they judged against a manifest frozen in 2026-04,
which is worse than not judging at all. They return when the catalog
grows a content tier.

Usage:
    from validation import orchestrate

    db = orchestrate.default_catalog()
    report = orchestrate.run_validation(db, subjects=["sub-03"])
    print(report["summary"])
"""

from pathlib import Path

# validation/ and core/ share the src/python root; consumers put it on
# sys.path (see run.py, or mmmdata-agents' ensure_mmmdata_importable).
from core import catalog

# mmmdata repo root (this file lives at src/python/validation/)
_REPO_ROOT = Path(__file__).resolve().parents[3]


def default_catalog() -> Path:
    """Resolve the catalog path from the repo config's BIDS root."""
    from core.config import load_config

    cfg = load_config(config_dir=_REPO_ROOT / "config")
    return catalog.catalog_path(cfg["paths"]["bids_project_dir"])


def available_statuses() -> list[str]:
    """The four-state resolution vocabulary."""
    return ["present", "missing", "excepted", "surplus"]


def run_validation(
    db_path: Path | str,
    subjects: list[str] | None = None,
    sessions: list[str] | None = None,
    statuses: list[str] | None = None,
    include_all: bool = False,
) -> dict:
    """Report expectation resolution for the dataset.

    Compares declared expectations against observed catalog rows via
    the resolution view. By default returns only issues (missing,
    surplus, excepted-pending); excepted-accepted and present are the
    quiet states.

    Args:
        db_path: Path to catalog.duckdb (see default_catalog).
        subjects: Subject labels, "03" or "sub-03" (default: all).
        sessions: Session labels, "04" or "ses-04" (default: all).
        statuses: Explicit status filter over the four-state
            vocabulary (overrides the default issues-only filter).
        include_all: Return every resolution row, including present.

    Returns a dict with keys: summary (status[-disposition] -> count),
    n_rows, rows.
    """
    invalid = set(statuses or []) - set(available_statuses())
    if invalid:
        raise ValueError(
            f"Unknown statuses: {sorted(invalid)}. "
            f"Available: {available_statuses()}"
        )
    return catalog.resolution_report(
        db_path,
        subjects=subjects,
        sessions=sessions,
        statuses=statuses,
        include_all=include_all,
    )


def lookup_expectations(
    db_path: Path | str,
    task: str | None = None,
    session_type: str | None = None,
) -> dict:
    """Look up declared expectations (see core.catalog.lookup_expectations)."""
    return catalog.lookup_expectations(
        db_path, task=task, session_type=session_type
    )
