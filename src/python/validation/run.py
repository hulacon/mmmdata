#!/usr/bin/env python3
"""
Report MMMData expectation resolution from the Contract A catalog.

Catalog-backed since 2026-08-17: prints the four-state resolution
(present / missing / excepted / surplus) of declared expectations,
judged inside catalog.duckdb. The declaration is
``<bids_root>/expectations/dataset.toml``; rebuild the judgment with
``scripts/catalog_expectations.py`` after editing it.

Usage:
    python -m validation                              # issues only
    python -m validation --subjects sub-03            # one subject
    python -m validation --sessions ses-01 ses-02     # some sessions
    python -m validation --status missing surplus     # explicit states
    python -m validation --all                        # every row
    python -m validation --tsv report.tsv             # export TSV
"""

import argparse
import csv
import sys
from pathlib import Path

# Make core/ importable when run as a module from anywhere
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from . import orchestrate  # noqa: E402


def print_report(report: dict) -> None:
    print("Resolution summary (filtered subjects/sessions, all statuses):")
    for key in sorted(report["summary"]):
        print(f"  {key:20s} {report['summary'][key]}")
    print()

    rows = report["rows"]
    if not rows:
        print("No rows match the status filter — nothing to report.")
        return

    print(f"{len(rows)} reported unit(s):")
    header = ("status", "sub", "ses", "task", "suffix", "dataset",
              "observed", "declared", "notes")
    print("  " + "  ".join(f"{h:>9s}" if h in ("observed", "declared")
                           else h for h in header))
    for r in rows:
        declared = (f"{r['runs_min']}" if r["runs_min"] == r["runs_max"]
                    else f"{r['runs_min']}-{r['runs_max']}")
        status = r["status"]
        if r.get("disposition"):
            status = f"{status}({r['disposition']})"
        notes = (r.get("exception_notes") or "")[:60]
        print(f"  {status:18s} {r['sub'] or '-':3s} {r['ses'] or '-':3s} "
              f"{(r['task'] or '-'):14s} {(r['suffix'] or '-'):8s} "
              f"{r['dataset_relpath']:28s} "
              f"{str(r['observed_n']):>3s} {declared:>7s}  {notes}")


def write_tsv(report: dict, path: Path) -> None:
    rows = report["rows"]
    fields = ["status", "disposition", "sub", "ses", "datatype", "task",
              "suffix", "recording", "dataset_relpath", "observed_n",
              "runs_min", "runs_max", "exception_notes"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t",
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Report MMMData expectation resolution from the catalog."
    )
    parser.add_argument("--db", type=Path, default=None,
                        help="Path to catalog.duckdb (default: from config)")
    parser.add_argument("--subjects", nargs="+", default=None,
                        help="Subjects to report (e.g. sub-03 or 03)")
    parser.add_argument("--sessions", nargs="+", default=None,
                        help="Sessions to report (e.g. ses-01 or 01)")
    parser.add_argument("--status", nargs="+", default=None,
                        choices=orchestrate.available_statuses(),
                        help="Statuses to report (default: issues only)")
    parser.add_argument("--all", action="store_true",
                        help="Report every row, including present")
    parser.add_argument("--tsv", type=Path, default=None,
                        help="Also write the rows to a TSV file")
    args = parser.parse_args()

    db = args.db or orchestrate.default_catalog()
    report = orchestrate.run_validation(
        db,
        subjects=args.subjects,
        sessions=args.sessions,
        statuses=args.status,
        include_all=args.all,
    )

    print_report(report)
    if args.tsv:
        write_tsv(report, args.tsv)

    # Exit nonzero when unexplained problems exist (missing/surplus),
    # so the CLI is usable as a gate.
    hard = sum(v for k, v in report["summary"].items()
               if k in ("missing", "surplus"))
    return 1 if hard else 0


if __name__ == "__main__":
    sys.exit(main())
