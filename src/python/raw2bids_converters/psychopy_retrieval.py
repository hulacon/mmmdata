#!/usr/bin/env python3
"""Convert PsychoPy free recall retrieval CSVs into BIDS _events.tsv files.

Handles files with conversion_type='psychopy_retrieval' (30 files total).
These are PsychoPy output CSVs from the free recall retrieval (recall) task,
sessions 1-10 (BIDS ses-19 to ses-28), 1 run per session.

PsychoPy CSV structure (11 rows, ~89 columns):
  - Rows with trials_recall.thisN not NaN are trial rows (typically 4-6 per run)
  - Scanner reference time: use_row.started on first trial row
  - Trial onset: recall1.started
  - Duration: time of recall period (key_resp_recall.rt or next trial onset)

Usage:
    python psychopy_retrieval.py <psychopy_csv> [<output_events_tsv>] [--dry-run]
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
    """Extract subject and session from PsychoPy retrieval filename.

    Pattern: {subj}_{sess}_free_recall_recall_{timestamp}.csv
    Example: 3_1_free_recall_recall_2025-04-01_13h31.05.742.csv
    """
    fname = os.path.basename(csv_path)
    m = re.match(r"(\d+)_(\d+)_free_recall_recall_", fname)
    if not m:
        raise ValueError(f"Cannot parse PsychoPy retrieval filename: {fname}")
    return int(m.group(1)), int(m.group(2))


def parse_recall_rt(val):
    """Parse key_resp_recall.rt which may be a string like '[119.86]'."""
    if pd.isna(val):
        return None
    s = str(val).strip("[] '\"")
    try:
        return float(s)
    except ValueError:
        return None


def convert_file(psychopy_csv, output_tsv, dry_run=False):
    """Convert a PsychoPy retrieval CSV to BIDS events TSV+JSON.

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
    subj, sess = parse_filename(psychopy_csv)

    df = pd.read_csv(psychopy_csv)

    # Filter to trial rows: trials_recall.thisN is not NaN
    trial_rows = df[df["trials_recall.thisN"].notna()].copy()

    if len(trial_rows) == 0:
        print(f"  WARNING: No trial rows found in {psychopy_csv}", file=sys.stderr)
        return False

    # Find scanner reference time
    ref_candidates = df["use_row.started"].dropna()
    if ref_candidates.empty:
        print(f"  WARNING: No scanner reference (use_row.started) in {psychopy_csv}",
              file=sys.stderr)
        return False
    scanner_ref = float(ref_candidates.iloc[0])

    # Trial onset: recall1.started relative to scanner reference
    recall_started = trial_rows["recall1.started"].astype(float)
    onsets = recall_started - scanner_ref

    # Duration: time until key_resp_recall ends (from recall1.started)
    # Use next trial's recall1.started - current recall1.started as fallback
    durations = []
    recall_vals = recall_started.values
    for i, idx in enumerate(trial_rows.index):
        # Try key_resp_recall.rt first (time from recall start to space bar)
        # This includes conditional response + image viewing + recall period
        rt = parse_recall_rt(trial_rows.loc[idx, "key_resp_recall.rt"])
        if rt is not None:
            # key_resp_recall.rt is relative to key_resp_recall.started,
            # which is when the image cue appears.
            # Full trial duration = image_stim.started - recall1.started + rt
            img_start = trial_rows.loc[idx, "image_stim.started"]
            if pd.notna(img_start):
                dur = float(img_start) - recall_vals[i] + rt
            else:
                dur = rt
        elif i < len(recall_vals) - 1:
            dur = recall_vals[i + 1] - recall_vals[i]
        else:
            dur = NA
        durations.append(dur)

    # Conditional response (confidence rating)
    cond_keys = trial_rows.get("key_resp_conditional.keys",
                               trial_rows.get("trials_recall.key_resp_conditional.keys"))
    cond_rt = trial_rows.get("key_resp_conditional.rt",
                             trial_rows.get("trials_recall.key_resp_conditional.rt"))

    events = pd.DataFrame({
        "onset": onsets.values,
        "duration": durations,
        "trial_type": "recall",
        "movie_name": trial_rows["movie_name"].values,
        "condition": trial_rows["condition"].apply(int_or_na).values,
        "style": trial_rows["style"].values,
        "free_recall_position": trial_rows["free_recall_position"].apply(int_or_na).values,
        "response": cond_keys.apply(int_or_na).values if cond_keys is not None else NA,
        "response_time": cond_rt.apply(float_or_na).values if cond_rt is not None else NA,
    })

    write_events_tsv(events, output_tsv, dry_run=dry_run)

    json_path = output_tsv.replace("_events.tsv", "_events.json")
    write_json_sidecar(SIDECAR, json_path, dry_run=dry_run)
    return True


SIDECAR = {
    "onset": {
        "Description": "Recall trial onset relative to scanner start (use_row.started)",
        "Units": "s",
    },
    "duration": {
        "Description": "Total trial duration (conditional response + cue viewing + recall)",
        "Units": "s",
    },
    "trial_type": {
        "Description": "Type of trial event",
        "Levels": {"recall": "Free recall trial for a previously viewed movie"},
    },
    "movie_name": {"Description": "Name of movie being recalled"},
    "condition": {"Description": "Experimental condition code"},
    "style": {"Description": "Movie style descriptor"},
    "free_recall_position": {"Description": "Recall order position"},
    "response": {
        "Description": "Participant conditional response (confidence rating)",
        "Units": "button number",
    },
    "response_time": {
        "Description": "Reaction time for conditional response",
        "Units": "s",
    },
}


def main():
    parser = argparse.ArgumentParser(
        description="Convert PsychoPy retrieval CSV to BIDS events TSV"
    )
    parser.add_argument("psychopy_csv", help="Path to PsychoPy retrieval CSV")
    parser.add_argument("output_tsv", nargs="?", default=None,
                        help="Output events TSV path (auto-generated if omitted)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.output_tsv is None:
        subj, sess = parse_filename(args.psychopy_csv)
        sub = bids_sub(subj)
        ses = bids_ses_fr(sess)
        fname = f"{sub}_{ses}_task-NATretrieval_events.tsv"
        output = os.path.join(BIDS_ROOT, sub, ses, "func", fname)
    else:
        output = args.output_tsv

    print(f"Input: {args.psychopy_csv}")
    print(f"Output: {output}")
    convert_file(args.psychopy_csv, output, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
