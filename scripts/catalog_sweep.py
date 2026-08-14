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


#: Path components that mark a discovered dataset as pipeline-internal
#: (scratch, logs, caches, staging inputs) rather than a canonical data
#: product. Rule-driven, not curated: nested trees like hippunfold/work,
#: hippunfold/logs, nordic/bids_input, pattern_similarity/cache come and go
#: with pipeline runs (decided 2026-08-14, contract-a-catalog log).
INTERNAL_COMPONENTS = frozenset(
    {"work", "logs", "log", "cache", ".cache", "tmp", "scratch",
     "bids_input"})


def dataset_kind(rel: str) -> str:
    """'internal' when any path component names pipeline scratch, else
    'canonical'. A release cut is a filter over canonical datasets — scope
    is a query, not an indexer decision."""
    if rel == ".":
        return "canonical"
    parts = set(rel.split("/"))
    return "internal" if parts & INTERNAL_COMPONENTS else "canonical"


def read_provenance(root: pathlib.Path, dataset: pathlib.Path) -> dict:
    """Provenance from dataset_description.json: pipeline, version, sources.

    Sources (SourceDatasets URLs, written tree-relative) are resolved to
    root-relative paths; the raw dataset resolves to "raw"."""
    dd = dataset / "dataset_description.json"
    if not dd.exists():
        return {}
    try:
        meta = json.loads(dd.read_text())
    except Exception as exc:
        return {"provenance_error": f"{type(exc).__name__}: {exc}"}
    generated_by = (meta.get("GeneratedBy") or [{}])[0]
    sources = []
    for src in meta.get("SourceDatasets") or []:
        url = src.get("URL", "")
        if "://" in url or url.startswith("doi:"):
            sources.append(url)  # DOI/remote URL: keep verbatim
            continue
        try:
            resolved = (dataset / url).resolve().relative_to(root)
            sources.append("raw" if str(resolved) == "." else str(resolved))
        except (ValueError, OSError):
            sources.append(url)  # unresolvable: keep verbatim
    return {
        "ds_name": meta.get("Name"),
        "dataset_type": meta.get("DatasetType"),
        "pipeline": generated_by.get("Name"),
        "pipeline_version": generated_by.get("Version"),
        "sources": ";".join(sources) or None,
    }


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
    for stale in parquet_dir.glob("*.parquet"):
        stale.unlink()  # datasets can vanish between sweeps; never union stale files

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
            "kind": dataset_kind(rel),
            "has_dataset_description":
                (dataset / "dataset_description.json").exists(),
            **read_provenance(root, dataset),
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
    con.execute(
        "CREATE OR REPLACE TABLE datasets ("
        " relpath VARCHAR, kind VARCHAR, name VARCHAR, dataset_type VARCHAR,"
        " pipeline VARCHAR, pipeline_version VARCHAR, sources VARCHAR,"
        " has_dataset_description BOOLEAN, skipped VARCHAR, error VARCHAR,"
        " n_rows BIGINT, index_seconds DOUBLE)")
    con.executemany(
        "INSERT INTO datasets VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        [[d["dataset"], d["kind"], d.get("ds_name"), d.get("dataset_type"),
          d.get("pipeline"), d.get("pipeline_version"), d.get("sources"),
          d["has_dataset_description"], d.get("skipped"), d.get("error"),
          d.get("rows"), d.get("seconds")]
         for d in report["datasets"]])
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
