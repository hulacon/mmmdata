#!/usr/bin/env python3
"""
Run MMMData manifest validation.

Compares the manifest database (reality) against the expectations schema
(intent), produces structured results, and optionally writes them to the
validation_results table and a TSV report.

Usage:
    python -m validation.run                          # full run
    python -m validation.run --checks file_presence   # specific checks
    python -m validation.run --subjects sub-03        # specific subject
    python -m validation.run --sessions ses-01 ses-02 # specific sessions
    python -m validation.run --tsv report.tsv         # export TSV
"""

import argparse
import csv
import datetime
import fnmatch
import sqlite3
import sys
import tomllib
from pathlib import Path

from .checks import ALL_CHECKS

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent.parent  # mmmdata repo root

sys.path.insert(0, str(_REPO_ROOT / "src" / "python"))
try:
    from core.config import load_config
    _config = load_config(config_dir=_REPO_ROOT / "config")
    BIDS_ROOT = Path(_config["paths"]["bids_project_dir"])
except Exception:
    BIDS_ROOT = Path("/gpfs/projects/hulacon/shared/mmmdata")

DEFAULT_DB = BIDS_ROOT / "inventory" / "manifest.db"
DEFAULT_SCHEMA = (
    BIDS_ROOT / "code" / "mmmdata-docs" / "dataset_expectations.toml"
)

# Fallback: check the submodule paths used by mmmdata and mmmdata-agents
_ALT_SCHEMA_PATHS = [
    _REPO_ROOT / "docs" / "doc" / "shared" / "dataset_expectations.toml",
    BIDS_ROOT / "code" / "mmmdata" / "docs" / "doc" / "shared" / "dataset_expectations.toml",
]


def find_schema_path() -> Path:
    if DEFAULT_SCHEMA.exists():
        return DEFAULT_SCHEMA
    for p in _ALT_SCHEMA_PATHS:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Cannot find dataset_expectations.toml. Tried:\n"
        f"  {DEFAULT_SCHEMA}\n" +
        "\n".join(f"  {p}" for p in _ALT_SCHEMA_PATHS)
    )


# ---------------------------------------------------------------------------
# Exception matching
# ---------------------------------------------------------------------------

def load_exceptions(schema: dict) -> list[dict]:
    """Extract the [[exceptions]] array from the schema."""
    return schema.get("exceptions", [])


def match_exception(result: dict, exceptions: list[dict]) -> dict | None:
    """
    Check if a validation result matches a known exception.

    Matching rules:
    - subject: exact match or "*" (wildcard)
    - session: exact match or "*" (wildcard)
    - task: exact match (if present in exception) or not checked
    - category: matched against check_name prefix or exception category

    Returns the matching exception dict, or None.
    """
    for exc in exceptions:
        exc_sub = exc.get("subject", "*")
        exc_ses = exc.get("session", "*")
        exc_task = exc.get("task")
        exc_category = exc.get("category", "")

        # Subject match
        if exc_sub != "*" and exc_sub != result.get("subject"):
            continue

        # Session match — support applies_to_sessions list
        applies_to = exc.get("applies_to_sessions")
        if applies_to:
            if result.get("session") not in applies_to and exc_ses != result.get("session"):
                continue
        elif exc_ses != "*" and exc_ses != result.get("session"):
            continue

        # Task match (if exception specifies a task)
        if exc_task and result.get("task") and exc_task != result.get("task"):
            continue

        # Category / check_name affinity
        check = result.get("check_name", "")
        if not _category_matches_check(exc_category, check, result):
            continue

        return exc

    return None


def _category_matches_check(category: str, check_name: str, result: dict) -> bool:
    """Determine if an exception category is relevant to a check result."""
    # Direct mappings
    mapping = {
        "dcm2bids": ["file_presence"],
        "behavioral": ["file_presence", "events_row_count", "events_columns"],
        "events": ["events_row_count", "events_columns", "events_timing"],
        "events_conversion": ["file_presence", "events_row_count", "events_columns"],
        "physio": ["physio_presence"],
        "physio_collection": ["physio_presence"],
        "eyetracking": ["eyetracking_presence"],
        "run_count": ["file_presence", "total_runs"],
        "volume_count": ["volume_count"],
    }

    valid_checks = mapping.get(category, [])
    if valid_checks:
        return check_name in valid_checks

    # If no mapping, match any check (generic exception)
    return True


def apply_exceptions(results: list[dict], exceptions: list[dict]) -> list[dict]:
    """
    Apply known exceptions to validation results.

    By default, matching exceptions downgrade fail/warn → info (design
    deviations that are fully accounted for).  Exceptions with
    ``data_missing = true`` instead keep the original status and annotate
    the message — the data is genuinely absent, even though the reason is
    known.

    Modifies results in place and returns them.
    """
    for r in results:
        if r["status"] in ("fail", "warn"):
            exc = match_exception(r, exceptions)
            if exc:
                desc = exc.get("description", "known exception")
                if exc.get("data_missing", False):
                    # Keep original status — data is genuinely missing
                    r["message"] = f"[KNOWN MISSING] {desc}" + (
                        f" | {r['message']}" if r["message"] else ""
                    )
                else:
                    r["status"] = "info"
                    r["message"] = f"[EXCEPTION] {desc}" + (
                        f" | {r['message']}" if r["message"] else ""
                    )
    return results


# ---------------------------------------------------------------------------
# Result storage
# ---------------------------------------------------------------------------

def store_results(conn: sqlite3.Connection, results: list[dict]):
    """Write validation results to the validation_results table."""
    cur = conn.cursor()
    cur.execute("DELETE FROM validation_results")
    now = datetime.datetime.now().isoformat()

    for r in results:
        cur.execute(
            """INSERT INTO validation_results
               (check_name, subject, session, task, run, status,
                expected, actual, message, checked_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (r["check_name"], r["subject"], r["session"],
             r["task"], r["run"], r["status"],
             r["expected"], r["actual"], r["message"], now)
        )

    conn.commit()


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_summary(results: list[dict]):
    """Print a summary table to the terminal."""
    counts = {"pass": 0, "fail": 0, "warn": 0, "info": 0}
    n_known_missing = 0
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
        if r["message"] and r["message"].startswith("[KNOWN MISSING]"):
            n_known_missing += 1

    n_new_warn = counts["warn"] - n_known_missing

    total = len(results)
    print(f"\n{'='*60}")
    print(f"  Validation Summary: {total} checks")
    print(f"{'='*60}")
    print(f"  PASS: {counts['pass']}")
    print(f"  FAIL: {counts['fail']}")
    print(f"  WARN: {n_new_warn}  (+{n_known_missing} known missing)")
    print(f"  INFO: {counts['info']}  (design exceptions)")
    print(f"{'='*60}")

    # Group failures and warnings by check
    issues = [r for r in results if r["status"] in ("fail", "warn")]
    if not issues:
        print("\n  No unresolved failures or warnings.\n")
        return

    # Split known-missing from new issues
    new_issues = [r for r in issues
                  if not (r["message"] and r["message"].startswith("[KNOWN MISSING]"))]
    known_missing = [r for r in issues
                     if r["message"] and r["message"].startswith("[KNOWN MISSING]")]

    if new_issues:
        print(f"\n  Unresolved issues ({len(new_issues)}):\n")

        by_check = {}
        for r in new_issues:
            by_check.setdefault(r["check_name"], []).append(r)

        for check, items in sorted(by_check.items()):
            print(f"  [{check}] ({len(items)} issues)")
            for r in items[:10]:  # cap per-check display
                run_str = f" run-{r['run']}" if r["run"] else ""
                task_str = f" {r['task']}" if r["task"] else ""
                print(f"    {r['status'].upper()}: {r['subject']} {r['session']}"
                      f"{task_str}{run_str}")
                if r["message"]:
                    print(f"           {r['message']}")
            if len(items) > 10:
                print(f"    ... and {len(items) - 10} more")
            print()

    if known_missing:
        print(f"\n  Known missing data ({len(known_missing)}):\n")

        by_check = {}
        for r in known_missing:
            by_check.setdefault(r["check_name"], []).append(r)

        for check, items in sorted(by_check.items()):
            # Group by unique description to avoid repetition
            by_desc = {}
            for r in items:
                desc = r["message"].replace("[KNOWN MISSING] ", "").split(" | ")[0]
                by_desc.setdefault(desc, []).append(r)

            print(f"  [{check}] ({len(items)} items)")
            for desc, group in by_desc.items():
                sessions = sorted({f"{r['subject']}/{r['session']}" for r in group})
                if len(sessions) <= 5:
                    print(f"    {desc}")
                    print(f"      Sessions: {', '.join(sessions)}")
                else:
                    print(f"    {desc}")
                    print(f"      {len(sessions)} subject/session pairs affected")
            print()


def export_tsv(results: list[dict], out_path: Path):
    """Export results to a TSV file."""
    cols = ["check_name", "subject", "session", "task", "run",
            "status", "expected", "actual", "message"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols, delimiter="\t",
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"Results exported to: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run MMMData manifest validation checks."
    )
    parser.add_argument(
        "--db", type=Path, default=DEFAULT_DB,
        help=f"Manifest database path (default: {DEFAULT_DB})"
    )
    parser.add_argument(
        "--schema", type=Path, default=None,
        help="Path to dataset_expectations.toml"
    )
    parser.add_argument(
        "--checks", nargs="*", default=None,
        help=f"Checks to run (default: all). Available: {', '.join(ALL_CHECKS)}"
    )
    parser.add_argument(
        "--subjects", nargs="*", default=None,
        help="Subjects to validate (default: all active)"
    )
    parser.add_argument(
        "--sessions", nargs="*", default=None,
        help="Sessions to validate (default: all)"
    )
    parser.add_argument(
        "--tsv", type=Path, default=None,
        help="Export results to a TSV file"
    )
    parser.add_argument(
        "--no-store", action="store_true",
        help="Skip writing results to validation_results table"
    )
    parser.add_argument(
        "--no-exceptions", action="store_true",
        help="Skip exception matching (report raw pass/fail/warn)"
    )
    args = parser.parse_args()

    # Load schema
    schema_path = args.schema or find_schema_path()
    print(f"Schema: {schema_path}")
    with open(schema_path, "rb") as f:
        schema = tomllib.load(f)

    # Connect to manifest
    if not args.db.exists():
        print(f"ERROR: Database not found: {args.db}", file=sys.stderr)
        sys.exit(1)
    print(f"Database: {args.db}")

    conn = sqlite3.connect(str(args.db))

    # Determine subjects
    subjects = args.subjects or schema.get("subjects", {}).get("active", [])
    sessions = args.sessions or []
    print(f"Subjects: {subjects}")
    if sessions:
        print(f"Sessions: {sessions}")

    # Determine which checks to run
    check_names = args.checks or list(ALL_CHECKS.keys())
    invalid = set(check_names) - set(ALL_CHECKS.keys())
    if invalid:
        print(f"ERROR: Unknown checks: {invalid}", file=sys.stderr)
        print(f"Available: {', '.join(ALL_CHECKS)}", file=sys.stderr)
        sys.exit(1)

    # Run checks
    all_results = []
    for name in check_names:
        print(f"  Running: {name}...", end=" ", flush=True)
        check_fn = ALL_CHECKS[name]
        results = check_fn(conn, schema, subjects, sessions)
        n_pass = sum(1 for r in results if r["status"] == "pass")
        n_fail = sum(1 for r in results if r["status"] == "fail")
        n_warn = sum(1 for r in results if r["status"] == "warn")
        print(f"{len(results)} results (pass={n_pass} fail={n_fail} warn={n_warn})")
        all_results.extend(results)

    # Apply exception matching
    if not args.no_exceptions:
        exceptions = load_exceptions(schema)
        print(f"\nApplying {len(exceptions)} known exceptions...")
        apply_exceptions(all_results, exceptions)

    # Report
    print_summary(all_results)

    # Store
    if not args.no_store:
        store_results(conn, all_results)
        print(f"Results stored in validation_results table ({len(all_results)} rows)")

    # Export TSV
    if args.tsv:
        export_tsv(all_results, args.tsv)

    conn.close()

    # Exit with error code if unresolved failures exist
    n_fail = sum(1 for r in all_results if r["status"] == "fail")
    sys.exit(1 if n_fail > 0 else 0)


if __name__ == "__main__":
    main()
