#!/usr/bin/env python3
"""Orchestrator: read file_inventory.csv and run all converters.

Reads the inventory, groups files by conversion_type, and dispatches each
to the appropriate converter module. Supports dry-run, subject filtering,
and task filtering.

Usage:
    python run_all.py [--dry-run] [--subjects sub-03,sub-04] [--tasks encoding,math]
    python run_all.py --validate [--subjects sub-03]
"""

import argparse
import csv
import os
import sys
import traceback

from common import BIDS_ROOT, SOURCE_DIR

# Import converters
import timed_events
import psychopy_encoding
import psychopy_retrieval
import localizer_events
import behavioral_to_beh
import edf_to_physio
import physio_dcm
import validate

INVENTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "file_inventory.csv")


def load_inventory():
    """Load and return all rows from file_inventory.csv."""
    with open(INVENTORY, newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def filter_rows(rows, subjects=None, conversion_types=None):
    """Filter inventory rows by subject and/or conversion type."""
    if subjects:
        sub_patterns = []
        for s in subjects:
            num = s.replace("sub-", "").replace("sub_", "")
            sub_patterns.extend([f"sub_{num}", f"sub-{num}", f"sub-0{num}",
                                  f"sub_{num.zfill(2)}", f"sub-{num.zfill(2)}"])
        rows = [r for r in rows if any(p in r["source_file"] for p in sub_patterns)]

    if conversion_types:
        rows = [r for r in rows if r.get("conversion_type") in conversion_types]

    return rows


def process_timed_events(row, dry_run=False):
    """Process a timed_events file (behavioral + timing -> events.tsv)."""
    src = row["source_file"]
    dest = row["bids_destination"]

    # Skip non-BIDS destinations
    if not dest.startswith("sub-"):
        return True

    full_src = os.path.join(SOURCE_DIR, src)
    full_dest = os.path.join(BIDS_ROOT, dest)

    timing_csv = timed_events.find_timing_csv(full_src)
    return timed_events.convert_file(full_src, timing_csv, full_dest, dry_run=dry_run)


def process_psychopy_encoding(row, dry_run=False):
    """Process a psychopy_encoding file."""
    src = row["source_file"]
    dest = row["bids_destination"]
    if not dest.startswith("sub-"):
        return True

    full_src = os.path.join(SOURCE_DIR, src)
    full_dest = os.path.join(BIDS_ROOT, dest)
    return psychopy_encoding.convert_file(full_src, full_dest, dry_run=dry_run)


def process_psychopy_retrieval(row, dry_run=False):
    """Process a psychopy_retrieval file."""
    src = row["source_file"]
    dest = row["bids_destination"]
    if not dest.startswith("sub-"):
        return True

    full_src = os.path.join(SOURCE_DIR, src)
    full_dest = os.path.join(BIDS_ROOT, dest)
    return psychopy_retrieval.convert_file(full_src, full_dest, dry_run=dry_run)


def process_localizer(row, dry_run=False):
    """Process a localizer_events file."""
    src = row["source_file"]
    dest = row["bids_destination"]
    if not dest.startswith("sub-"):
        return True

    full_src = os.path.join(SOURCE_DIR, src)
    full_dest = os.path.join(BIDS_ROOT, dest)
    return localizer_events.convert_file(full_src, full_dest, dry_run=dry_run)


def process_behavioral(row, dry_run=False):
    """Process a behavioral_to_beh file."""
    src = row["source_file"]
    dest = row["bids_destination"]
    if not dest.startswith("sub-"):
        return True

    full_src = os.path.join(SOURCE_DIR, src)
    full_dest = os.path.join(BIDS_ROOT, dest)
    return behavioral_to_beh.convert_file(full_src, full_dest, dry_run=dry_run)


def process_edf(row, dry_run=False):
    """Process an edf_to_physio file."""
    src = row["source_file"]
    dest = row["bids_destination"]
    if not dest.startswith("sub-"):
        return True

    full_src = os.path.join(SOURCE_DIR, src)
    full_dest = os.path.join(BIDS_ROOT, dest)
    return edf_to_physio.convert_file(full_src, full_dest, dry_run=dry_run)


def process_physio_dcm(row, dry_run=False):
    """Process a physio_dcm file (PhysioLog DICOM -> BIDS physio)."""
    src = row["source_file"]
    dest = row["bids_destination"]
    if not dest.startswith("sub-"):
        return True

    # src is a directory path relative to sourcedata
    full_src = os.path.join(SOURCE_DIR, src)
    # dest is a BIDS base path (without _recording-*_physio suffix)
    full_dest = os.path.join(BIDS_ROOT, dest)
    return physio_dcm.convert_file(full_src, full_dest, dry_run=dry_run)


PROCESSORS = {
    "timed_events": process_timed_events,
    "psychopy_encoding": process_psychopy_encoding,
    "psychopy_retrieval": process_psychopy_retrieval,
    "localizer_events": process_localizer,
    "behavioral_to_beh": process_behavioral,
    "edf_to_physio": process_edf,
    "physio_dcm": process_physio_dcm,
}

# Types that don't produce output
SKIP_TYPES = {"timing_input", "supplementary", "no_conversion"}


def main():
    parser = argparse.ArgumentParser(description="Run all BIDS converters")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print actions without writing files")
    parser.add_argument("--subjects", default=None,
                        help="Comma-separated subject list (e.g. sub-03,sub-04)")
    parser.add_argument("--tasks", default=None,
                        help="Comma-separated conversion types to run")
    parser.add_argument("--validate", action="store_true",
                        help="Run validation against metainformation after conversion")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.validate:
        subjects_arg = args.subjects if args.subjects else None
        sys.exit(validate.main())

    rows = load_inventory()
    subjects = args.subjects.split(",") if args.subjects else None
    conv_types = args.tasks.split(",") if args.tasks else None

    if conv_types is None:
        conv_types = list(PROCESSORS.keys())

    rows = filter_rows(rows, subjects=subjects, conversion_types=conv_types)

    print(f"Processing {len(rows)} files")
    if args.dry_run:
        print("[DRY RUN MODE]")
    print()

    # Group by conversion type
    from collections import Counter, defaultdict
    by_type = defaultdict(list)
    for row in rows:
        ct = row.get("conversion_type", "")
        if ct in PROCESSORS:
            by_type[ct].append(row)
        elif ct in SKIP_TYPES:
            pass  # silently skip
        elif ct:
            print(f"WARNING: Unknown conversion_type '{ct}' for {row['source_file']}")

    success = 0
    failure = 0

    for ct in conv_types:
        type_rows = by_type.get(ct, [])
        if not type_rows:
            continue

        print(f"\n{'='*60}")
        print(f"Converting {ct} ({len(type_rows)} files)")
        print(f"{'='*60}")

        processor = PROCESSORS[ct]
        for row in type_rows:
            try:
                ok = processor(row, dry_run=args.dry_run)
                if ok:
                    success += 1
                else:
                    failure += 1
            except Exception as e:
                failure += 1
                print(f"ERROR processing {row['source_file']}: {e}", file=sys.stderr)
                if args.verbose:
                    traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"SUMMARY: {success} succeeded, {failure} failed, {success + failure} total")
    print(f"{'='*60}")

    return 0 if failure == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
