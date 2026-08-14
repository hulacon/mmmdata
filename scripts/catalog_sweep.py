#!/usr/bin/env python3
"""Contract A catalog sweep: index the full BIDS tree with bids2table.

Discovers every BIDS dataset under --root (raw + each derivative), indexes
each with bids2table, writes one parquet per dataset plus a consolidated
DuckDB catalog, and emits a JSON report with per-dataset row counts, wall
times, and failures. Datasets that fail to index (e.g. missing
dataset_description.json) are recorded, never fatal.

Runs in the shared catalog env:
  /gpfs/projects/hulacon/shared/envs/catalog/bin/python

Design: mmmdata-agents docs/workbench/contract-a-catalog/ (charter) and
docs/constellation-contracts.md §3.
"""

from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import sys
import time

# cloudpathlib/pathlib incompatibility shim (see mmmdata-agents
# docs/CLUSTER-TODO.md §1) — must run before importing bids2table proper.
import bids2table._pathlib as _b2t_pathlib

_b2t_pathlib.AnyPath = pathlib.Path

import bids2table as b2t  # noqa: E402
import duckdb  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402


def dataset_slug(root: pathlib.Path, dataset: pathlib.Path) -> str:
    rel = dataset.relative_to(root)
    return "raw" if str(rel) == "." else str(rel).replace("/", "__")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, type=pathlib.Path)
    ap.add_argument("--out-dir", required=True, type=pathlib.Path,
                    help="Where catalog.duckdb, catalog_parquet/, and the "
                         "sweep report land (normally the inventory dir)")
    ap.add_argument("--exclude", action="append", default=[],
                    help="Root-relative dataset path to skip (repeatable), "
                         "e.g. derivatives/bids_validation")
    args = ap.parse_args()

    root = args.root.resolve()
    parquet_dir = args.out_dir / "catalog_parquet"
    parquet_dir.mkdir(parents=True, exist_ok=True)

    report: dict = {
        "root": str(root),
        "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "bids2table_version": getattr(b2t, "__version__", "unknown"),
        "duckdb_version": duckdb.__version__,
        "excluded": args.exclude,
        "datasets": [],
    }
    t_sweep = time.monotonic()

    for dataset in sorted(b2t.find_bids_datasets(root)):
        dataset = pathlib.Path(dataset)
        rel = str(dataset.relative_to(root)) if dataset != root else "."
        entry = {
            "dataset": rel,
            "has_dataset_description":
                (dataset / "dataset_description.json").exists(),
        }
        if any(rel == ex or rel.startswith(ex.rstrip("/") + "/")
               for ex in args.exclude):
            entry["skipped"] = "excluded"
            report["datasets"].append(entry)
            print(f"    skip  {rel}  (excluded)", flush=True)
            continue

        t0 = time.monotonic()
        try:
            table = b2t.index_dataset(dataset)
            slug = dataset_slug(root, dataset)
            # bids2table emits its own `dataset` column; ours is the
            # root-relative path, the stable join key across the sweep
            table = table.append_column(
                "dataset_relpath", pa.array([rel] * table.num_rows, pa.string()))
            pq.write_table(table, parquet_dir / f"{slug}.parquet")
            entry.update(rows=table.num_rows,
                         seconds=round(time.monotonic() - t0, 1))
            print(f"{table.num_rows:8d}  {rel}  "
                  f"({entry['seconds']}s)", flush=True)
        except Exception as exc:  # record and continue: failures are data here
            entry.update(error=f"{type(exc).__name__}: {exc}",
                         seconds=round(time.monotonic() - t0, 1))
            print(f"    FAIL  {rel}: {entry['error']}", flush=True)
        report["datasets"].append(entry)

    report["sweep_seconds"] = round(time.monotonic() - t_sweep, 1)

    db_path = args.out_dir / "catalog.duckdb"
    t0 = time.monotonic()
    con = duckdb.connect(str(db_path))
    con.execute(
        "CREATE OR REPLACE TABLE files AS "
        "SELECT * FROM read_parquet(?, union_by_name=true)",
        [str(parquet_dir / "*.parquet")],
    )
    n_files = con.execute("SELECT count(*) FROM files").fetchone()[0]
    con.execute("CREATE OR REPLACE TABLE sweep_meta AS "
                "SELECT ? AS key, ? AS value", ["report", json.dumps(report)])
    con.close()
    report["duckdb_seconds"] = round(time.monotonic() - t0, 1)
    report["total_rows"] = n_files

    report_path = args.out_dir / "catalog_sweep_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    ok = sum(1 for d in report["datasets"] if "rows" in d)
    fail = sum(1 for d in report["datasets"] if "error" in d)
    print(f"\n{ok} datasets indexed, {fail} failed, {n_files} rows total "
          f"-> {db_path}\nreport: {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
