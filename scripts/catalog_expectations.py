#!/usr/bin/env python3
"""Contract A expectations tier: declared expectations -> catalog.duckdb.

Reads the dataset-owned declaration (<root>/expectations/dataset.toml),
expands it into per-unit expected rows keyed like observed `files` rows,
ingests the canonical sub-*_sessions.tsv files, loads disposition-tagged
exceptions, and defines the four-state `resolution` view:

  present  — observed count within [runs_min, runs_max]
  missing  — below runs_min, no status-eligible exception
  excepted — out of range, matched by a status-eligible exception
             (disposition: pending dominates accepted)
  surplus  — present but undeclared, or above runs_max with no exception

Judgment is the SQL join in the view — no engine code. Rebuild is cheap and
idempotent; run it after every catalog sweep (it only replaces its own
tables/views, never `files`/`datasets`).

Runs in the shared catalog env:
  /gpfs/projects/hulacon/shared/envs/catalog/bin/python

Design: mmmdata-agents docs/workbench/contract-a-catalog/ (log 2026-08-14
design forks) and docs/constellation-contracts.md §3.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import tomllib

import duckdb

# Entities that individuate acquisitions within a unit. Observed run counts
# are distinct combinations of these — collapsing space/res/den/hemi so a
# derivative run in three output spaces still counts once.
IDENTITY_ENTITIES = ["acq", "ce", "rec", "dir", "run", "echo", "part"]


def expand_units(decl: dict) -> list[dict]:
    """Session-type templates + per-subject overrides -> expected unit rows."""
    active = decl["subjects"]["active"]
    session_types = decl["session_types"]
    rows: list[dict] = []

    def add(dataset, sub, ses, datatype, task, suffix, desc, lo, hi):
        rows.append(dict(dataset_relpath=dataset, sub=sub, ses=ses,
                         datatype=datatype, task=task, suffix=suffix,
                         des=desc, runs_min=lo, runs_max=hi))

    for unit in decl["units"]:
        subjects = [unit["subject"]] if "subject" in unit else active
        sessions = unit.get("sessions") or session_types[unit["session_type"]]
        lo = unit.get("runs_min", unit.get("runs"))
        hi = unit.get("runs_max", unit.get("runs"))
        if lo is None or hi is None:
            raise SystemExit(f"unit missing runs/runs_min+runs_max: {unit}")
        task = unit.get("task")
        for sub in subjects:
            for ses in sessions:
                add(".", sub, ses, unit["datatype"], task, unit["suffix"],
                    None, lo, hi)
                if unit["suffix"] == "bold":
                    if unit.get("sbref", True):
                        add(".", sub, ses, "func", task, "sbref", None, lo, hi)
                    if unit.get("events", False):
                        add(".", sub, ses, "func", task, "events", None, lo, hi)
                    for deriv in decl["derivatives"]["complete_for_bold"]:
                        add(deriv, sub, ses, "func", task, "bold",
                            "preproc", lo, hi)
    return rows


def expand_exceptions(decl: dict) -> list[dict]:
    rows = []
    for exc in decl.get("exceptions", []):
        # Status rule: only exceptions naming task/datatype/suffix may flip a
        # unit to "excepted"; session-level notes are annotation-only.
        eligible = any(k in exc for k in ("task", "datatype", "suffix"))
        for ses in exc["sessions"]:
            rows.append(dict(
                sub=exc["subject"], ses=ses, task=exc.get("task"),
                datatype=exc.get("datatype"), suffix=exc.get("suffix"),
                category=exc["category"], disposition=exc["disposition"],
                status_eligible=eligible, description=exc["description"]))
    return rows


def scope_predicate(scopes: list[dict]) -> str:
    """One OR-branch per scoped dataset, over the observed `files` table."""
    branches = []
    for sc in scopes:
        conds = [f"dataset_relpath = '{sc['dataset']}'", "ext != '.json'"]
        if "datatypes" in sc:
            vals = ", ".join(f"'{v}'" for v in sc["datatypes"])
            conds.append(f"datatype IN ({vals})")
        if "suffixes" in sc:
            vals = ", ".join(f"'{v}'" for v in sc["suffixes"])
            conds.append(f"suffix IN ({vals})")
        if "descs" in sc:
            vals = ", ".join(f"'{v}'" for v in sc["descs"])
            conds.append(f"\"desc\" IN ({vals})")
        if "exclude_suffixes" in sc:
            vals = ", ".join(f"'{v}'" for v in sc["exclude_suffixes"])
            conds.append(f"suffix NOT IN ({vals})")
        branches.append("(" + " AND ".join(conds) + ")")
    return " OR ".join(branches)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, type=pathlib.Path,
                    help="BIDS root (holds expectations/ and inventory/)")
    ap.add_argument("--db", type=pathlib.Path, default=None,
                    help="catalog.duckdb (default <root>/inventory/catalog.duckdb)")
    ap.add_argument("--expectations", type=pathlib.Path, default=None,
                    help="declaration (default <root>/expectations/dataset.toml)")
    args = ap.parse_args()

    root = args.root.resolve()
    db_path = args.db or root / "inventory" / "catalog.duckdb"
    decl_path = args.expectations or root / "expectations" / "dataset.toml"
    decl = tomllib.loads(decl_path.read_text())

    units = expand_units(decl)
    exceptions = expand_exceptions(decl)
    con = duckdb.connect(str(db_path))

    # -- canonical session metadata (contracts §8.6: BIDS-tree TSVs win) --
    con.execute(f"""
        CREATE OR REPLACE TABLE sessions AS
        SELECT regexp_extract(filename, 'sub-([0-9a-zA-Z]+)_sessions', 1) AS sub,
               replace(session_id, 'ses-', '') AS ses, *
        FROM read_csv('{root}/sub-*/sub-*_sessions.tsv', delim='\t',
                      header=true, union_by_name=true, all_varchar=true,
                      filename=true)""")

    con.execute("""
        CREATE OR REPLACE TABLE expected_units (
            dataset_relpath VARCHAR, sub VARCHAR, ses VARCHAR,
            datatype VARCHAR, task VARCHAR, suffix VARCHAR, des VARCHAR,
            runs_min INTEGER, runs_max INTEGER)""")
    con.executemany(
        "INSERT INTO expected_units VALUES (?,?,?,?,?,?,?,?,?)",
        [[u["dataset_relpath"], u["sub"], u["ses"], u["datatype"], u["task"],
          u["suffix"], u["des"], u["runs_min"], u["runs_max"]] for u in units])

    con.execute("""
        CREATE OR REPLACE TABLE expectation_exceptions (
            sub VARCHAR, ses VARCHAR, task VARCHAR, datatype VARCHAR,
            suffix VARCHAR, category VARCHAR, disposition VARCHAR,
            status_eligible BOOLEAN, description VARCHAR)""")
    con.executemany(
        "INSERT INTO expectation_exceptions VALUES (?,?,?,?,?,?,?,?,?)",
        [[e["sub"], e["ses"], e["task"], e["datatype"], e["suffix"],
          e["category"], e["disposition"], e["status_eligible"],
          e["description"]] for e in exceptions])

    identity = ", ".join(
        f"coalesce(CAST({e} AS VARCHAR), '')" for e in IDENTITY_ENTITIES)
    con.execute(f"""
        CREATE OR REPLACE VIEW observed_units AS
        SELECT dataset_relpath, sub, ses, datatype, task, suffix,
               "desc" AS des,
               count(DISTINCT concat_ws('|', {identity})) AS observed_n
        FROM files
        WHERE {scope_predicate(decl["scope"])}
        GROUP BY ALL""")

    # Exception matching happens at unit grain; pending dominates accepted so
    # to-do items keep reporting even when an accepted note also matches.
    con.execute("""
        CREATE OR REPLACE VIEW resolution AS
        WITH joined AS (
            SELECT
                coalesce(e.dataset_relpath, o.dataset_relpath) AS dataset_relpath,
                coalesce(e.sub, o.sub) AS sub,
                coalesce(e.ses, o.ses) AS ses,
                coalesce(e.datatype, o.datatype) AS datatype,
                coalesce(e.task, o.task) AS task,
                coalesce(e.suffix, o.suffix) AS suffix,
                coalesce(e.des, o.des) AS des,
                e.runs_min, e.runs_max,
                coalesce(o.observed_n, 0) AS observed_n,
                e.dataset_relpath IS NOT NULL AS declared
            FROM expected_units e
            FULL OUTER JOIN observed_units o
              ON e.dataset_relpath = o.dataset_relpath
             AND e.sub = o.sub AND e.ses = o.ses
             AND coalesce(e.task, '') = coalesce(o.task, '')
             AND e.datatype = o.datatype AND e.suffix = o.suffix
             AND coalesce(e.des, '') = coalesce(o.des, '')
        ),
        annotated AS (
            SELECT j.*,
                max(CASE WHEN x.status_eligible THEN 1 ELSE 0 END) AS has_eligible_exc,
                max(CASE WHEN x.status_eligible AND x.disposition = 'pending'
                         THEN 1 ELSE 0 END) AS has_pending,
                string_agg(DISTINCT x.category || ': ' || x.description,
                           ' | ') AS exception_notes
            FROM joined j
            LEFT JOIN expectation_exceptions x
              -- dataset-level blanket notes (sub-* x ses-*, no unit narrowing)
              -- stay out of the per-unit annotation join: they would annotate
              -- every unit and drown the column in noise
              ON NOT (x.sub = '*' AND x.ses = '*' AND NOT x.status_eligible)
             AND (x.sub = '*' OR x.sub = j.sub)
             AND (x.ses = '*' OR x.ses = j.ses)
             AND (x.task IS NULL OR x.task = j.task)
             AND (x.datatype IS NULL OR x.datatype = j.datatype)
             AND (x.suffix IS NULL OR x.suffix = j.suffix)
            GROUP BY ALL
        )
        SELECT dataset_relpath, sub, ses, datatype, task, suffix, des,
               runs_min, runs_max, observed_n,
               CASE
                   WHEN NOT declared THEN 'surplus'
                   WHEN observed_n BETWEEN runs_min AND runs_max THEN 'present'
                   WHEN has_eligible_exc = 1 THEN 'excepted'
                   WHEN observed_n > runs_max THEN 'surplus'
                   ELSE 'missing'
               END AS status,
               CASE WHEN has_eligible_exc = 1 AND NOT
                         (observed_n BETWEEN runs_min AND runs_max)
                    THEN CASE WHEN has_pending = 1
                              THEN 'pending' ELSE 'accepted' END
               END AS disposition,
               exception_notes
        FROM annotated""")

    tally = con.execute("""
        SELECT status, coalesce(disposition, '') AS disposition, count(*)
        FROM resolution GROUP BY 1, 2 ORDER BY 1, 2""").fetchall()
    print(f"{len(units)} expected units, {len(exceptions)} exception rows, "
          f"{con.execute('SELECT count(*) FROM sessions').fetchone()[0]} "
          f"session rows -> {db_path}")
    for status, disposition, n in tally:
        print(f"  {status:9s} {disposition:9s} {n:5d}")
    not_ok = con.execute("""
        SELECT dataset_relpath, sub, ses, task, suffix, runs_min, runs_max,
               observed_n, status
        FROM resolution WHERE status IN ('missing', 'surplus')
        ORDER BY status, dataset_relpath, sub, ses, task""").fetchall()
    if not_ok:
        print("\nmissing / surplus:")
        for r in not_ok:
            print("  " + "  ".join(str(v) for v in r))
    con.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
