#!/usr/bin/env python3
"""Copy PsychoPy .log and .psydat files to their matching locations in sourcedata.

For each .log/.psydat file in rawto_be_bids/alldata/free_recall_behavioral/,
finds the matching .csv already organized in sourcedata/sub-{id}/ses-{NN}/behavioral/
and copies the companion file to the same directory.

Usage:
    python copy_psychopy_companions.py --dry-run   # preview only
    python copy_psychopy_companions.py              # actually copy
"""

import argparse
import shutil
from pathlib import Path

SOURCEDATA = Path("/projects/hulacon/shared/mmmdata/sourcedata")
RAW_BEHAVIORAL = SOURCEDATA / "rawto_be_bids" / "alldata" / "free_recall_behavioral"

SUBJECT_MAP = {
    "sub_03": "sub-03",
    "sub_04": "sub-04",
    "sub_05": "sub-05",
}


def build_csv_index():
    """Build a lookup: csv_stem -> full path in organized sourcedata."""
    index = {}
    for bids_sub in SUBJECT_MAP.values():
        sub_dir = SOURCEDATA / bids_sub
        for csv_path in sub_dir.rglob("behavioral/*.csv"):
            index[csv_path.stem] = csv_path.parent
    return index


def find_companion_files():
    """Find all .log and .psydat files in rawto_be_bids."""
    companions = []
    for ext in ("*.log", "*.psydat"):
        companions.extend(RAW_BEHAVIORAL.rglob(ext))
    return sorted(companions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be copied without copying",
    )
    args = parser.parse_args()

    csv_index = build_csv_index()
    companions = find_companion_files()

    copied = 0
    skipped = 0
    no_match = 0

    for src in companions:
        stem = src.stem  # e.g. 3_1_1_mem_search_recall_2025-04-01_12h39.55.925
        dest_dir = csv_index.get(stem)

        if dest_dir is None:
            print(f"  NO MATCH: {src.name}")
            no_match += 1
            continue

        dest = dest_dir / src.name

        if dest.exists():
            skipped += 1
            continue

        if args.dry_run:
            print(f"  COPY: {src.name}")
            print(f"    -> {dest}")
        else:
            shutil.copy2(src, dest)
            print(f"  COPIED: {src.name} -> {dest_dir}")
        copied += 1

    print()
    print(f"Summary: {copied} {'would copy' if args.dry_run else 'copied'}, "
          f"{skipped} already exist, {no_match} no matching CSV found")


if __name__ == "__main__":
    main()
