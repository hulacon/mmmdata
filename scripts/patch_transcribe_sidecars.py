#!/usr/bin/env python
"""Backfill the words-frame columns into `transcribe` sidecar declarations.

`transcribe` is the one aud2psy model that writes two frames -- segments and
words -- and `pipeline.py` declared `list(transcript_df.columns)`, the
segments frame, at all four of its sites. The words frame's own columns were
never declared, which left `transcribe_probability` emitted but undeclared in
every `*_transcript_words.csv` in the store: 1,060 families.

**No data is wrong.** The column is correctly prefixed, psytwill attributed it,
and it is in the aggregates. What is wrong is the sidecar's account of itself,
and re-extracting to fix a provenance field would be ~3.5 GPU-h of Whisper.
So this patches the declarations in place instead.

aud2psy is fixed for future runs (`_transcribe_columns`, 0.15.1). This script
reproduces exactly what that function would have written, by reading the
family's tables off disk rather than guessing: declared stays in its emitted
order, and words-only columns are appended. `stimulus_id` is excluded because
the CLI adds it when writing, and the model never declared it.

Idempotent: a family already carrying its words columns is left untouched, so
a second run reports 0 patched. Dry-run unless `--apply`.

    patch_transcribe_sidecars.py                # what would change
    patch_transcribe_sidecars.py --apply
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

MODEL = "transcribe"

#: Added to the CSV at write time by the campaign's `--stimulus-id`, never
#: declared by the model. Declaring it here would be a new inaccuracy.
NOT_DECLARED = {"stimulus_id"}


def family_tables(sidecar: dict, meta_path: Path) -> list[Path]:
    """The family's CSVs, from the sidecar's own `output` map.

    Authoritative in a way a prefix glob is not: one directory holds cells
    whose stems prefix each other, and `output` names each table outright.
    Paths are resolved relative to the sidecar so a moved tree still works.
    """
    out = sidecar.get("output") or {}
    tables = []
    for entry in out.values():
        if not isinstance(entry, dict) or "path" not in entry:
            continue
        path = Path(entry["path"])
        if not path.exists():
            path = meta_path.parent / path.name
        if path.exists() and path.suffix == ".csv":
            tables.append(path)
    return tables


def header(path: Path) -> list[str]:
    with open(path, newline="") as f:
        return next(csv.reader(f), [])


def emitted_columns(tables: list[Path]) -> list[str]:
    """Every column the family emits, in table order, first occurrence wins."""
    seen: list[str] = []
    for table in tables:
        for column in header(table):
            if column not in seen and column not in NOT_DECLARED:
                seen.append(column)
    return seen


def plan(root: Path) -> tuple[list[tuple[Path, list[str], list[str]]], int, int]:
    """(patches, n_transcribe_families, n_already_correct)."""
    patches, n_seen, n_ok = [], 0, 0
    for meta_path in sorted(root.rglob("*.meta.json")):
        try:
            sidecar = json.loads(meta_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(sidecar, dict):
            continue
        models = sidecar.get("models")
        if not isinstance(models, dict):
            # Some sidecars in the store spell `models` as a list of names.
            # They declare no columns at all, so there is nothing to reconcile.
            continue
        entry = models.get(MODEL)
        if not isinstance(entry, dict):
            continue
        n_seen += 1
        declared = list(entry.get("columns") or [])
        tables = family_tables(sidecar, meta_path)
        if not tables:
            continue
        emitted = emitted_columns(tables)
        missing = [c for c in emitted if c not in declared]
        if not missing:
            n_ok += 1
            continue
        patches.append((meta_path, declared, declared + missing))
    return patches, n_seen, n_ok


def apply(meta_path: Path, columns: list[str]) -> None:
    """Rewrite one sidecar atomically, preserving its `indent=2` shape."""
    sidecar = json.loads(meta_path.read_text())
    sidecar["models"][MODEL]["columns"] = columns
    tmp = meta_path.with_suffix(".meta.json.tmp")
    tmp.write_text(json.dumps(sidecar, indent=2))
    os.replace(tmp, meta_path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path,
                    default=Path("/gpfs/projects/hulacon/shared/mmmdata")
                    / "derivatives" / "stimuli_features")
    ap.add_argument("--apply", action="store_true",
                    help="write the sidecars (default: report only)")
    args = ap.parse_args()

    if not args.root.exists():
        print(f"ERROR  no feature store at {args.root}", file=sys.stderr)
        return 2

    patches, n_seen, n_ok = plan(args.root)
    print(f"{n_seen} families declare `{MODEL}`; "
          f"{n_ok} already complete, {len(patches)} to patch")

    added: dict[str, int] = {}
    for _, before, after in patches:
        for column in after[len(before):]:
            added[column] = added.get(column, 0) + 1
    for column, n in sorted(added.items(), key=lambda kv: -kv[1]):
        print(f"  +{n:>6}  {column}")

    if not patches:
        return 0
    if not args.apply:
        example, before, after = patches[0]
        print(f"\nExample: {example}")
        print(f"  before: {before}")
        print(f"  after:  {after}")
        print("\nDry run. Re-run with --apply to write.")
        return 0

    for meta_path, _, after in patches:
        apply(meta_path, after)
    print(f"\nPatched {len(patches)} sidecars.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
