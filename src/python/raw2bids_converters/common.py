#!/usr/bin/env python3
"""Shared utilities for raw-to-BIDS behavioral data converters."""

import json
import os

import pandas as pd

# === Paths ===
BIDS_ROOT = "/gpfs/projects/hulacon/shared/mmmdata"
SOURCE_DIR = f"{BIDS_ROOT}/sourcedata"
METAINFO_DIR = f"{BIDS_ROOT}/sourcedata/metainformation"

# === Session offsets ===
FR_SESSION_OFFSET = 18   # free recall session N -> BIDS ses-(N+18)
CR_SESSION_OFFSET = 3    # cued recall session N -> BIDS ses-(N+3)
FINAL_SESSION = 30

# === BIDS missing value ===
NA = "n/a"

# === Subject directory name mappings ===
# Source dirs use both underscore (sub_03) and hyphen (sub-03)
SUBJECT_NUMS = [3, 4, 5]


def bids_sub(num):
    """Return BIDS subject label, e.g. 'sub-03'."""
    return f"sub-{num:02d}"


def bids_ses(num):
    """Return BIDS session label, e.g. 'ses-04'."""
    return f"ses-{num:02d}"


def bids_ses_fr(sess_num):
    """Map free recall session number to BIDS session label."""
    return bids_ses(sess_num + FR_SESSION_OFFSET)


def bids_ses_cr(sess_num):
    """Map cued recall session number to BIDS session label."""
    return bids_ses(sess_num + CR_SESSION_OFFSET)


def na_value(val):
    """Replace NaN/None/empty with BIDS 'n/a'."""
    if pd.isna(val) or val == "" or val is None:
        return NA
    return val


def write_events_tsv(df, output_path, dry_run=False):
    """Write a BIDS events TSV file.

    - Tab-separated, no index column
    - NaN/missing replaced with 'n/a'
    - Creates parent directories if needed
    """
    if dry_run:
        print(f"  [dry-run] Would write {len(df)} rows -> {output_path}")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = df.fillna(NA)
    df.to_csv(output_path, sep="\t", index=False)
    print(f"  Wrote {len(df)} rows -> {output_path}")


def write_beh_tsv(df, output_path, dry_run=False):
    """Write a BIDS behavioral TSV file (same format as events)."""
    write_events_tsv(df, output_path, dry_run=dry_run)


def write_json_sidecar(descriptions, output_path, dry_run=False):
    """Write a BIDS JSON sidecar file with column descriptions.

    Parameters
    ----------
    descriptions : dict
        Maps column names to description dicts, e.g.:
        {"onset": {"Description": "Event onset time", "Units": "s"}}
    output_path : str
        Path for JSON file.
    """
    if dry_run:
        print(f"  [dry-run] Would write JSON -> {output_path}")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(descriptions, f, indent=4)
        f.write("\n")
    print(f"  Wrote JSON -> {output_path}")


def bids_output_path(sub_num, ses_num, modality, filename):
    """Build a full BIDS output path.

    Parameters
    ----------
    sub_num : int
        Subject number (e.g. 3)
    ses_num : int
        BIDS session number (already mapped, e.g. 19 for free recall session 1)
    modality : str
        'func' or 'beh'
    filename : str
        BIDS filename (e.g. 'sub-03_ses-19_task-encoding_run-01_events.tsv')
    """
    sub = bids_sub(sub_num)
    ses = bids_ses(ses_num)
    return os.path.join(BIDS_ROOT, sub, ses, modality, filename)


def int_or_na(val):
    """Convert to int if numeric, else return 'n/a'."""
    if pd.isna(val) or val == "" or val is None:
        return NA
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return NA


def float_or_na(val):
    """Convert to float if numeric, else return 'n/a'."""
    if pd.isna(val) or val == "" or val is None:
        return NA
    try:
        return float(val)
    except (ValueError, TypeError):
        return NA
