"""Programmatic entry points for MMMData manifest validation.

``validation.run`` is the CLI (printing, TSV export, result storage);
this module is the library face of the same engine: resolve the
expectations schema, run a chosen set of checks against the manifest
database, apply known exceptions, and return plain Python data.

Usage:
    from validation import orchestrate

    schema = orchestrate.load_schema(bids_root)
    report = orchestrate.run_validation(db_path, schema,
                                        subjects=["sub-03"])
    print(report["summary"])
"""

import tomllib
from pathlib import Path
from typing import Sequence

from .checks import ALL_CHECKS
from .run import apply_exceptions, load_exceptions

# mmmdata repo root (this file lives at src/python/validation/)
_REPO_ROOT = Path(__file__).resolve().parents[3]

_SCHEMA_NAME = "dataset_expectations.toml"


def find_schema(
    bids_root: Path | str | None = None,
    extra_candidates: Sequence[Path | str] = (),
) -> Path:
    """Locate dataset_expectations.toml, trying several known homes.

    Search order:
      1. The standalone mmmdata-docs clone under the BIDS root
         (``<bids_root>/code/mmmdata-docs/``), if bids_root is given.
      2. Any caller-supplied extra candidate paths (e.g. another
         repo's docs submodule).
      3. The mmmdata repo's own docs submodule
         (``docs/doc/shared/``).

    Returns the first existing path; raises FileNotFoundError listing
    everything that was tried.
    """
    candidates: list[Path] = []
    if bids_root is not None:
        candidates.append(
            Path(bids_root) / "code" / "mmmdata-docs" / _SCHEMA_NAME
        )
    candidates.extend(Path(p) for p in extra_candidates)
    candidates.append(_REPO_ROOT / "docs" / "doc" / "shared" / _SCHEMA_NAME)

    for p in candidates:
        if p.exists():
            return p

    raise FileNotFoundError(
        f"Cannot find {_SCHEMA_NAME}. Tried:\n"
        + "\n".join(f"  {p}" for p in candidates)
    )


def load_schema(
    bids_root: Path | str | None = None,
    extra_candidates: Sequence[Path | str] = (),
) -> dict:
    """Find and parse dataset_expectations.toml (see find_schema)."""
    with open(find_schema(bids_root, extra_candidates), "rb") as f:
        return tomllib.load(f)


def available_checks() -> list[str]:
    """Names of all registered validation checks."""
    return list(ALL_CHECKS.keys())


def run_validation(
    db_path: Path | str,
    schema: dict,
    checks: Sequence[str] | None = None,
    subjects: Sequence[str] | None = None,
    sessions: Sequence[str] | None = None,
) -> dict:
    """Run validation checks and return a structured summary.

    Compares the manifest database (what exists on disk) against the
    expectations schema (what should exist). Known exceptions from the
    schema's [[exceptions]] array are matched and annotated on the
    results (``[KNOWN]`` message prefix).

    Args:
        db_path: Path to manifest.db.
        schema: Parsed dataset_expectations.toml (see load_schema).
        checks: Check names to run (default: all in ALL_CHECKS).
            Unknown names raise ValueError.
        subjects: Subjects to validate (default: the schema's
            active subjects).
        sessions: Sessions to validate (default: all).

    Returns a dict with keys:
        total_checks: number of individual results produced
        summary: status -> count (pass/fail/warn/info)
        issues: the fail/warn result dicts
        checks_run: the check names that were executed
    """
    import sqlite3

    # Resolve subjects/sessions
    subs = list(subjects) if subjects else schema.get("subjects", {}).get("active", [])
    sess = list(sessions) if sessions else []

    # Resolve checks
    check_names = list(checks) if checks else list(ALL_CHECKS.keys())
    invalid = set(check_names) - set(ALL_CHECKS.keys())
    if invalid:
        raise ValueError(f"Unknown checks: {sorted(invalid)}")

    conn = sqlite3.connect(str(db_path))
    try:
        all_results = []
        for name in check_names:
            results = ALL_CHECKS[name](conn, schema, subs, sess)
            all_results.extend(results)
    finally:
        conn.close()

    # Apply exceptions
    exceptions = load_exceptions(schema)
    apply_exceptions(all_results, exceptions)

    # Summarize
    counts = {"pass": 0, "fail": 0, "warn": 0, "info": 0}
    for r in all_results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1

    issues = [r for r in all_results if r["status"] in ("fail", "warn")]

    return {
        "total_checks": len(all_results),
        "summary": counts,
        "issues": issues,
        "checks_run": check_names,
    }


def lookup_expectations(
    schema: dict,
    task: str | None = None,
    session_type: str | None = None,
) -> dict:
    """Look up what the expectations schema says about a task or
    session type.

    Tasks are matched by config key first, then by ``task_label``.
    With neither argument, returns an overview of available tasks,
    session types, subjects, and functional defaults.

    Args:
        schema: Parsed dataset_expectations.toml (see load_schema).
        task: Task name (e.g. "TBencoding", "floc").
        session_type: Session type (e.g. "localizer", "cued_recall").

    Returns a dict mirroring what was requested (task/config,
    session_type/sessions/session_scans, or the overview keys).
    """
    result: dict = {}

    if task:
        tasks = schema.get("tasks", {})
        task_cfg = tasks.get(task)
        if task_cfg:
            result["task"] = task
            result["config"] = task_cfg
        else:
            # Try matching by task_label
            for name, cfg in tasks.items():
                if cfg.get("task_label") == task:
                    result["task"] = name
                    result["config"] = cfg
                    break
            if "config" not in result:
                result["error"] = f"Task {task!r} not found in schema"

    if session_type:
        st = schema.get("session_types", {})
        sessions_list = st.get(session_type, [])
        session_cfg = schema.get("sessions", {}).get(session_type, {})
        result["session_type"] = session_type
        result["sessions"] = sessions_list
        result["session_scans"] = session_cfg

    if not task and not session_type:
        # Return overview
        result["available_tasks"] = list(schema.get("tasks", {}).keys())
        result["available_session_types"] = list(
            schema.get("session_types", {}).keys()
        )
        result["subjects"] = schema.get("subjects", {})
        result["functional_defaults"] = schema.get("functional_defaults", {})

    return result
