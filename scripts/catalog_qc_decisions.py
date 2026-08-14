#!/usr/bin/env python3
"""Contract A QC tier: duckbrain QC decisions -> catalog.duckdb.

duckbrain owns the QC review process and the decision artifact (append-only
per-run JSON); the catalog ingests it. This reads every *_decision.json from
the known decision locations, oldest location first (the same search order as
duckbrain's ``decision_search_dirs``):

  1. <root>/derivatives/preprocessing_qc/          (legacy; mmmdata-written)
  2. <root>/derivatives/duckbrain/qc/decisions/    (current; duckbrain-written)

and materializes one row per run in a ``qc_decisions`` table: the latest
run-level verdict (the newest entry carrying no ``domain`` key — domain notes
share the file and must never read as the run's verdict), whether that verdict
is a sign-off an identifiable person made, and history counts.

The read semantics replicate duckbrain ``src/duckbrain/core/qc.py``
(``_history_of`` / ``load_decisions`` / ``is_signed_off``), which is the
reference implementation; both on-disk schemas are accepted and nothing is
ever rewritten. The catalog is derived, never canonical.

Rebuild is cheap and idempotent; run after every sweep (only replaces its own
table, never ``files``/``datasets``).

Runs in the shared catalog env:
  /gpfs/projects/hulacon/shared/envs/catalog/bin/python

Design: mmmdata-agents docs/workbench/contract-a-catalog/ (log 2026-08-14 QC
tier decision) and docs/constellation-contracts.md §3.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

import duckdb

# duckbrain's decision vocabulary (core/qc.py); replicated, not imported,
# because the catalog env deliberately has no duckbrain install.
SIGNED_OFF_DECISIONS = {"keep", "exclude", "investigate"}
AUTOMATED_REVIEWERS = {"auto-stub", "auto", "automated", ""}

DECISION_DIRS = (  # oldest first; later roots' entries are more recent
    "derivatives/preprocessing_qc",
    "derivatives/duckbrain/qc/decisions",
)

ENTITY_RE = {
    "sub": re.compile(r"sub-([0-9a-zA-Z]+)"),
    "ses": re.compile(r"ses-([0-9a-zA-Z]+)"),
    "task": re.compile(r"task-([0-9a-zA-Z]+)"),
    "run": re.compile(r"run-([0-9a-zA-Z]+)"),
}
SUFFIX_RE = re.compile(r"_([a-zA-Z0-9]+)$")


def history_of(data: dict) -> list[dict]:
    """A record's entries oldest-first, from either on-disk schema."""
    if isinstance(data.get("decisions"), list):
        return [e for e in data["decisions"] if isinstance(e, dict)]
    history = [e for e in data.get("history", []) if isinstance(e, dict)]
    if isinstance(data.get("latest"), dict):
        history.append(data["latest"])
    return history


def is_signed_off(entry: dict | None) -> bool:
    """True when the verdict is a call an identifiable person made."""
    if not entry or entry.get("automated"):
        return False
    if entry.get("decision") not in SIGNED_OFF_DECISIONS:
        return False
    reviewer = str(entry.get("reviewer") or "").strip().lower()
    return bool(reviewer) and reviewer not in AUTOMATED_REVIEWERS


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, type=pathlib.Path,
                    help="BIDS root (holds derivatives/ and inventory/)")
    ap.add_argument("--db", type=pathlib.Path, default=None,
                    help="catalog.duckdb (default <root>/inventory/catalog.duckdb)")
    args = ap.parse_args()

    root = args.root.resolve()
    db_path = args.db or root / "inventory" / "catalog.duckdb"

    # Merge per run_key across locations, oldest location first, so the
    # current location's entries land last and read as most recent.
    merged: dict[str, list[dict]] = {}
    sources: dict[str, list[str]] = {}
    n_files = 0
    unreadable: list[str] = []
    for rel_dir in DECISION_DIRS:
        base = root / rel_dir
        if not base.is_dir():
            continue
        for f in sorted(base.rglob("*_decision.json")):
            n_files += 1
            try:
                data = json.loads(f.read_text())
            except Exception:
                unreadable.append(str(f.relative_to(root)))
                continue
            run_key = data.get("run_key") or f.name.removesuffix(
                "_decision.json")
            merged.setdefault(run_key, []).extend(history_of(data))
            sources.setdefault(run_key, []).append(str(f.relative_to(root)))

    rows = []
    for run_key, entries in sorted(merged.items()):
        run_level = [e for e in entries if not e.get("domain")]
        domain_notes = [e for e in entries if e.get("domain")]
        latest = run_level[-1] if run_level else None
        ents = {k: (m.group(1) if (m := rx.search(run_key)) else None)
                for k, rx in ENTITY_RE.items()}
        m = SUFFIX_RE.search(run_key)
        rows.append([
            run_key, ents["sub"], ents["ses"], ents["task"], ents["run"],
            m.group(1) if m else None,
            latest.get("decision") if latest else None,
            latest.get("reviewer") if latest else None,
            bool(latest.get("automated")) if latest else None,
            is_signed_off(latest),
            latest.get("timestamp") if latest else None,
            latest.get("reason") if latest else None,
            len(run_level), len(domain_notes),
            ";".join(sources[run_key]),
        ])

    con = duckdb.connect(str(db_path))
    con.execute("""
        CREATE OR REPLACE TABLE qc_decisions (
            run_key VARCHAR, sub VARCHAR, ses VARCHAR, task VARCHAR,
            run VARCHAR, suffix VARCHAR,
            decision VARCHAR, reviewer VARCHAR, automated BOOLEAN,
            signed_off BOOLEAN, "timestamp" VARCHAR, reason VARCHAR,
            n_entries INTEGER, n_domain_notes INTEGER, source_files VARCHAR)""")
    con.executemany(
        "INSERT INTO qc_decisions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        rows)
    tally = con.execute("""
        SELECT decision, signed_off, count(*) FROM qc_decisions
        GROUP BY 1, 2 ORDER BY 1, 2""").fetchall()
    con.close()

    print(f"{n_files} decision files -> {len(rows)} runs -> {db_path}")
    for decision, signed, n in tally:
        print(f"  {str(decision):12s} signed_off={str(signed):5s} {n:5d}")
    if unreadable:
        print(f"unreadable ({len(unreadable)}):")
        for u in unreadable:
            print(f"  {u}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
