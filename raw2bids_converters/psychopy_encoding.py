#!/usr/bin/env python3
"""Convert PsychoPy free recall encoding CSVs into BIDS _events.tsv files.

Handles files with conversion_type='psychopy_encoding' (60 files total).
These are PsychoPy output CSVs from the free recall encoding (movie watching)
task, sessions 1-10 (BIDS ses-19 to ses-28), 2 runs per session.

PsychoPy CSV structure (9 rows, ~77 columns):
  - Rows with movie_loop.thisN not NaN are trial rows (4 movies per run)
  - Scanner reference time: use_row.started on first trial row
  - Per trial: title display, fixation cross, movie presentation
  - Post-run: 20s blank period

Output: 13 events per run (4 trials x 3 events + 1 blank)
  - title: movie title display
  - fixation: fixation cross between title and movie
  - movie: movie clip presentation
  - blank: post-run blank period

Usage:
    python psychopy_encoding.py <psychopy_csv> [<output_events_tsv>] [--dry-run]
"""

import argparse
import os
import re
import sys

import pandas as pd

from common import (
    BIDS_ROOT, FR_SESSION_OFFSET,
    bids_ses_fr, bids_sub, int_or_na,
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

    Produces 13 events per run: title + fixation + movie for each of 4 trials,
    plus a post-run blank event.

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
    ref_candidates = df["use_row.started"].dropna()
    if ref_candidates.empty:
        print(f"  WARNING: No scanner reference (use_row.started) in {psychopy_csv}",
              file=sys.stderr)
        return False
    scanner_ref = float(ref_candidates.iloc[0])

    bids_ses_num = sess + FR_SESSION_OFFSET

    events_list = []
    for _, row in trial_rows.iterrows():
        trial_num = int(row["movie_loop.thisN"]) + 1

        # --- Title event ---
        title_start = float(row["movie_title.started"])
        title_stop = float(row["movie_title.stopped"])
        events_list.append({
            "onset": title_start - scanner_ref,
            "duration": title_stop - title_start,
            "trial_type": "title",
            "trial_num": trial_num,
            "subject": subj,
            "session": bids_ses_num,
            "run": run,
            "condition": "",
            "movie_name": "",
            "movie_length": "",
            "style": "",
            "free_recall_position": "",
            "movie_title.started": title_start - scanner_ref,
            "movie_title.stopped": title_stop - scanner_ref,
        })

        # --- Fixation event (skip if timing not recorded) ---
        fix_start_raw = row.get("fixation.started")
        fix_stop_raw = row.get("fixation.stopped")
        if pd.notna(fix_start_raw) and pd.notna(fix_stop_raw):
            fix_start = float(fix_start_raw)
            fix_stop = float(fix_stop_raw)
            events_list.append({
                "onset": fix_start - scanner_ref,
                "duration": fix_stop - fix_start,
                "trial_type": "fixation",
                "trial_num": trial_num,
                "subject": subj,
                "session": bids_ses_num,
                "run": run,
                "condition": "",
                "movie_name": "",
                "movie_length": "",
                "style": "",
                "free_recall_position": "",
                "movie_title.started": "",
                "movie_title.stopped": "",
            })

        # --- Movie event ---
        movie_start = float(row["movies.started"])
        movie_stop = float(row["movies.stopped"])
        events_list.append({
            "onset": movie_start - scanner_ref,
            "duration": movie_stop - movie_start,
            "trial_type": "movie",
            "trial_num": trial_num,
            "subject": subj,
            "session": bids_ses_num,
            "run": run,
            "condition": int_or_na(row["condition"]),
            "movie_name": row["movie_name"],
            "movie_length": float(row["mov_len"]),
            "style": row["style"] if pd.notna(row["style"]) else "",
            "free_recall_position": int_or_na(row["free_recall_position"]),
            "movie_title.started": "",
            "movie_title.stopped": "",
        })

    # --- Post-run blank event ---
    blank_start = df["blank_20.started"].dropna()
    stop_et = df["stop_eyetracking.stopped"].dropna()
    if not blank_start.empty and not stop_et.empty:
        events_list.append({
            "onset": float(blank_start.iloc[0]) - scanner_ref,
            "duration": float(stop_et.iloc[0]) - float(blank_start.iloc[0]),
            "trial_type": "blank",
            "trial_num": "",
            "subject": subj,
            "session": bids_ses_num,
            "run": run,
            "condition": "",
            "movie_name": "",
            "movie_length": "",
            "style": "",
            "free_recall_position": "",
            "movie_title.started": "",
            "movie_title.stopped": "",
        })

    events = pd.DataFrame(events_list)
    write_events_tsv(events, output_tsv, dry_run=dry_run)

    json_path = output_tsv.replace("_events.tsv", "_events.json")
    write_json_sidecar(SIDECAR, json_path, dry_run=dry_run)
    return True


SIDECAR = {
    "onset": {
        "Description": "Event onset time relative to scanner start (use_row.started)",
        "Units": "s",
    },
    "duration": {
        "Description": "Event duration",
        "Units": "s",
    },
    "trial_type": {
        "Description": "Type of event",
        "Levels": {
            "title": "Movie title display (movie_title routine)",
            "fixation": "Fixation cross between title and movie onset",
            "movie": "Movie presentation",
            "blank": "Post-run blank period",
        },
    },
    "trial_num": {
        "Description": "Trial number within the run (1-indexed)",
        "Units": "integer",
    },
    "subject": {
        "Description": "Subject identifier number",
        "Units": "integer",
    },
    "session": {
        "Description": "Session number for this subject",
        "Units": "integer",
    },
    "run": {
        "Description": "Run number within the session",
        "Units": "integer",
    },
    "condition": {
        "Description": "Experimental condition for the movie",
        "Levels": {
            "1": "Condition 1",
            "2": "Condition 2",
            "3": "Condition 3",
        },
    },
    "movie_name": {
        "Description": "Name of the movie presented (empty for non-movie events)",
    },
    "movie_length": {
        "Description": "Expected duration of the movie in seconds",
        "Units": "s",
    },
    "style": {
        "Description": "Movie style description (e.g., animated, live-action, speech content)",
    },
    "free_recall_position": {
        "Description": "Position in the free recall sequence for this movie",
    },
    "movie_title.started": {
        "Description": "Onset of the movie_title routine relative to scanner start (title events only)",
        "Units": "s",
    },
    "movie_title.stopped": {
        "Description": "Offset of the movie_title routine relative to scanner start (title events only)",
        "Units": "s",
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
