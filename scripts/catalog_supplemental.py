#!/usr/bin/env python3
"""Contract A supplemental tier: files the indexer cannot see -> catalog.duckdb.

Walks every discovered dataset tree and records each file bids2table emitted
no row for, so *darkness itself is queryable*: "what does the catalog not
understand" becomes SQL over ``files_supplemental``, never a silent gap.
(Decided 2026-08-14, contract-a-catalog log: supplemental tier for blanket
coverage; schema-aware per-pipeline indexers only when a consumer question
demands one.)

Each row carries a ``category``:

  sidecar   — a .json whose stem pairs with an indexed row (bids2table emits
              no rows for JSON sidecars, so this also models the old
              ``has_json`` expectation)
  metadata  — dataset-level bookkeeping (dataset_description.json, README,
              CHANGES, LICENSE, .bidsignore, CITATION.cff)
  dark      — everything else: real files no catalog tier accounts for

Scope is explicit, not silent. For the raw dataset only ``sub-*``,
``phenotype/``, and root-level files are walked; ``stimuli/`` belongs to the
Contract B registry (§4.2), ``code/``/``inventory/``/``tmp_dcm2bids/`` are
not data (hygiene items stay on their own lists). Skipped top-level entries
are printed with file counts so the exclusion is visible in every run. A
subtree that is itself a discovered dataset is pruned — its files are its
own dataset's business.

Rebuild is cheap and idempotent; run after every sweep (needs the sweep's
``files`` + ``datasets`` tables; only replaces its own table).

Runs in the shared catalog env:
  /gpfs/projects/hulacon/shared/envs/catalog/bin/python

Design: mmmdata-agents docs/workbench/contract-a-catalog/ and
docs/constellation-contracts.md §3.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import re
import sys

import duckdb

#: Top-level entries of the *raw* dataset that are in scope for the walk.
RAW_SCOPE_PATTERNS = (re.compile(r"^sub-[0-9a-zA-Z]+$"),
                      re.compile(r"^phenotype$"),
                      # walked with nested datasets pruned, so only files
                      # loose under derivatives/ outside any dataset land here
                      re.compile(r"^derivatives$"))

METADATA_NAMES = {"dataset_description.json", "README", "README.md",
                  "CHANGES", "LICENSE", "LICENSE.txt", ".bidsignore",
                  "CITATION.cff"}

SUB_RE = re.compile(r"sub-([0-9a-zA-Z]+)")
SES_RE = re.compile(r"ses-([0-9a-zA-Z]+)")


def walk_dataset(ds_dir: pathlib.Path, root: pathlib.Path,
                 all_relpaths: set[str], is_raw: bool):
    """Yield files under ds_dir, pruning nested datasets (and, for the raw
    dataset, everything outside RAW_SCOPE_PATTERNS). Returns skip tallies."""
    skipped: dict[str, int] = {}
    if is_raw:
        entries = sorted(os.scandir(ds_dir), key=lambda e: e.name)
        for e in entries:
            if e.is_file(follow_symlinks=False):
                yield pathlib.Path(e.path)
            elif e.is_dir(follow_symlinks=False):
                rel = str(pathlib.Path(e.path).relative_to(root))
                if rel in all_relpaths:
                    continue  # a discovered dataset (e.g. derivatives/*)
                if any(p.match(e.name) for p in RAW_SCOPE_PATTERNS):
                    yield from _walk(pathlib.Path(e.path), root, all_relpaths)
                else:
                    n = sum(len(fs) for _, _, fs in os.walk(e.path))
                    skipped[e.name] = n
    else:
        yield from _walk(ds_dir, root, all_relpaths)
    walk_dataset.skipped = skipped  # type: ignore[attr-defined]


def _walk(top: pathlib.Path, root: pathlib.Path, all_relpaths: set[str]):
    for dirpath, dirnames, filenames in os.walk(top):
        dirnames[:] = sorted(
            d for d in dirnames
            if str((pathlib.Path(dirpath) / d).relative_to(root))
            not in all_relpaths)
        for f in sorted(filenames):
            yield pathlib.Path(dirpath) / f


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, type=pathlib.Path)
    ap.add_argument("--db", type=pathlib.Path, default=None,
                    help="catalog.duckdb (default <root>/inventory/catalog.duckdb)")
    args = ap.parse_args()

    root = args.root.resolve()
    db_path = args.db or root / "inventory" / "catalog.duckdb"
    con = duckdb.connect(str(db_path))

    datasets = con.execute(
        "SELECT relpath, skipped, kind FROM datasets").fetchall()
    all_relpaths = {d[0] for d in datasets}
    # Indexed stems per dataset: path minus ext, for membership tests and
    # for pairing .json sidecars with the row they describe.
    indexed: dict[str, set[str]] = {}
    stems: dict[str, set[str]] = {}
    for ds, path, ext in con.execute(
            "SELECT dataset_relpath, path, ext FROM files").fetchall():
        indexed.setdefault(ds, set()).add(path)
        if ext and path.endswith(ext):
            stems.setdefault(ds, set()).add(path[: -len(ext)])

    rows = []
    for rel, skipped, kind in sorted(datasets):
        if skipped:
            continue
        if kind == "internal":
            # pipeline scratch: dark by design, tagged on the datasets
            # table; walking it would flood the tier with Snakemake/work
            # files nobody will ever query for
            print(f"  kind-skip   {rel}  (internal)", flush=True)
            continue
        ds_dir = root if rel == "." else root / rel
        if not ds_dir.is_dir():
            continue
        is_raw = rel == "."
        ds_indexed = indexed.get(rel, set())
        ds_stems = stems.get(rel, set())
        gen = walk_dataset(ds_dir, root, all_relpaths - {rel}, is_raw)
        for f in gen:
            fp = str(f.relative_to(ds_dir))
            if fp in ds_indexed:
                continue
            if f.name in METADATA_NAMES:
                category = "metadata"
            elif fp.endswith(".json") and fp[:-5] in ds_stems:
                category = "sidecar"
            else:
                category = "dark"
            try:
                st = f.stat()
                size, mtime = st.st_size, int(st.st_mtime)
            except OSError:
                size, mtime = None, None
            sub = m.group(1) if (m := SUB_RE.search(fp)) else None
            ses = m.group(1) if (m := SES_RE.search(fp)) else None
            ext = "".join(f.suffixes[-2:]) if f.name.endswith(".gz") \
                else (f.suffix or None)
            rows.append([rel, fp, f.name, ext, category, sub, ses,
                         size, mtime])
        for name, n in getattr(walk_dataset, "skipped", {}).items():
            print(f"  scope-skip  {name}/  ({n} files; see docstring)",
                  flush=True)

    con.execute("""
        CREATE OR REPLACE TABLE files_supplemental (
            dataset_relpath VARCHAR, path VARCHAR, filename VARCHAR,
            ext VARCHAR, category VARCHAR, sub VARCHAR, ses VARCHAR,
            size_bytes BIGINT, mtime_epoch BIGINT)""")
    con.executemany(
        "INSERT INTO files_supplemental VALUES (?,?,?,?,?,?,?,?,?)", rows)
    tally = con.execute("""
        SELECT category, count(*), count(DISTINCT dataset_relpath)
        FROM files_supplemental GROUP BY 1 ORDER BY 1""").fetchall()
    dark_by_ds = con.execute("""
        SELECT dataset_relpath, count(*) FROM files_supplemental
        WHERE category = 'dark' GROUP BY 1 ORDER BY 2 DESC LIMIT 15""").fetchall()
    con.close()

    print(f"\n{len(rows)} supplemental rows -> {db_path}")
    for category, n, nds in tally:
        print(f"  {category:9s} {n:7d} rows across {nds} datasets")
    print("\ndark files by dataset (top 15):")
    for ds, n in dark_by_ds:
        print(f"  {n:7d}  {ds}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
