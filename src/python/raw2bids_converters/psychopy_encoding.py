#!/usr/bin/env python3
"""Convert PsychoPy free recall encoding CSVs into BIDS _events.tsv files.

Handles files with conversion_type='psychopy_encoding' (60 files total).
These are PsychoPy output CSVs from the free recall encoding (movie watching)
task, sessions 1-10 (BIDS ses-19 to ses-28), 2 runs per session.

PsychoPy CSV structure (9 rows, ~77 columns):
  - Rows with movie_loop.thisN not NaN are trial rows (4 movies per run)
  - Scanner reference time: use_row.started on first trial row
  - Movie onset: movies.started
  - Movie duration: mov_len (seconds)

Usage:
    python psychopy_encoding.py <psychopy_csv> [<output_events_tsv>] [--dry-run]
"""

import argparse
import os
import re
import sys

import pandas as pd

from common import (
    NA, BIDS_ROOT, FR_SESSION_OFFSET,
    bids_ses_fr, bids_sub, float_or_na, int_or_na,
    write_events_tsv, write_json_sidecar,
)


def parse_filename(csv_path):
    """Extract subject, session, run from PsychoPy encoding filename.

    Pattern: {subj}_{sess}_{run}_mem_search_recall_{timestamp}.csv
    Example: 3_1_1_mem_search_recall_2025-04-01_12h39.55.925.csv
    """
    fname = os.path.basename(csv_path)
    m = re.match(r"(\d+)_(\d+)_(\d+)_mem_search_recall_", fname)
    if not m:
        raise ValueError(f"Cannot parse PsychoPy encoding filename: {fname}")
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def convert_file(psychopy_csv, output_tsv, dry_run=False):
    """Convert a PsychoPy encoding CSV to BIDS events TSV+JSON.

    Parameters
    ----------
    psychopy_csv : str
        Path to the PsychoPy output CSV.
    output_tsv : str
        Destination path for the BIDS events TSV.
    dry_run : bool
        If True, print what would be done without writing files.

    Returns
    -------
    bool
        True if conversion succeeded.
    """
    subj, sess, run = parse_filename(psychopy_csv)

    df = pd.read_csv(psychopy_csv)

    # Filter to trial rows: movie_loop.thisN is not NaN
    trial_rows = df[df["movie_loop.thisN"].notna()].copy()

    if len(trial_rows) == 0:
        print(f"  WARNING: No trial rows found in {psychopy_csv}", file=sys.stderr)
        return False

    # Find scanner reference time: use_row.started on the first trial row
    # (or earliest non-NaN value in the column)
    ref_candidates = df["use_row.started"].dropna()
    if ref_candidates.empty:
        print(f"  WARNING: No scanner reference (use_row.started) in {psychopy_csv}",
              file=sys.stderr)
        return False
    scanner_ref = float(ref_candidates.iloc[0])

    # Extract onset, duration, and trial metadata
    movie_started = trial_rows["movies.started"].astype(float)
    onsets = movie_started - scanner_ref

    # Duration from mov_len (clip length in seconds)
    durations = trial_rows["mov_len"].astype(float)

    # free_recall_position: position in subsequent free recall, or n/a
    recall_pos = trial_rows["free_recall_position"].apply(
        lambda x: int_or_na(x)
    )

    events = pd.DataFrame({
        "onset": onsets.values,
        "duration": durations.values,
        "trial_type": "movie",
        "movie_name": trial_rows["movie_name"].values,
        "condition": trial_rows["condition"].apply(int_or_na).values,
        "style": trial_rows["style"].values,
        "free_recall_position": recall_pos.values,
    })

    write_events_tsv(events, output_tsv, dry_run=dry_run)

    json_path = output_tsv.replace("_events.tsv", "_events.json")
    write_json_sidecar(SIDECAR, json_path, dry_run=dry_run)
    return True


SIDECAR = {
    "onset": {
        "Description": "Movie onset relative to scanner start (use_row.started)",
        "Units": "s",
    },
    "duration": {
        "Description": "Movie clip duration",
        "Units": "s",
    },
    "trial_type": {
        "Description": "Type of trial event",
        "Levels": {"movie": "Movie clip presentation"},
    },
    "movie_name": {"Description": "Name of movie clip"},
    "condition": {"Description": "Experimental condition code (1, 2, or 3)"},
    "style": {"Description": "Movie style descriptor (e.g. 'Animated, no speech')"},
    "free_recall_position": {
        "Description": "Position in subsequent free recall order (n/a if not recalled)"
    },
}


def main():
    parser = argparse.ArgumentParser(
        description="Convert PsychoPy encoding CSV to BIDS events TSV"
    )
    parser.add_argument("psychopy_csv", help="Path to PsychoPy encoding CSV")
    parser.add_argument("output_tsv", nargs="?", default=None,
                        help="Output events TSV path (auto-generated if omitted)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.output_tsv is None:
        subj, sess, run = parse_filename(args.psychopy_csv)
        sub = bids_sub(subj)
        ses = bids_ses_fr(sess)
        fname = f"{sub}_{ses}_task-NATencoding_run-{run:02d}_events.tsv"
        output = os.path.join(BIDS_ROOT, sub, ses, "func", fname)
    else:
        output = args.output_tsv

    print(f"Input: {args.psychopy_csv}")
    print(f"Output: {output}")
    convert_file(args.psychopy_csv, output, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
