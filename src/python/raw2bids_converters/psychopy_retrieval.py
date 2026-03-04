#!/usr/bin/env python3
"""Convert PsychoPy free recall retrieval CSVs into BIDS _events.tsv files.

Handles files with conversion_type='psychopy_retrieval' (30 files total).
These are PsychoPy output CSVs from the free recall retrieval (recall) task,
sessions 1-10 (BIDS ses-19 to ses-28), 1 run per session.

PsychoPy CSV structure (11+ rows, ~89 columns):
  - Rows with trials_recall.thisN not NaN are trial rows (typically 4-8 per run)
  - Scanner reference time: use_row.started on first trial row
  - Per trial: blank ISI, prompt, conditional image cue, recall period
  - Post-run: congrats screen

Output: ~33 events per session (N trials x 4 events + 1 congrats)
  - blank: ~500ms ISI at start of each trial
  - prompt: instruction text display (participant presses 6 or 7)
  - cue: image cue display (shown if key=7, duration=0 if key=6)
  - recall: active free recall period
  - congrats: post-run congratulations screen

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
    bids_ses_fr, bids_sub, int_or_na,
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


def convert_file(psychopy_csv, output_tsv, dry_run=False):
    """Convert a PsychoPy retrieval CSV to BIDS events TSV+JSON.

    Produces ~33 events per session: blank + prompt + cue + recall for each
    trial, plus a post-run congrats event.

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

    # Find scanner reference time: wait.started marks the start of the
    # 12s dummy scan period, which is when the scanner begins acquiring
    # volumes. Use the last non-NaN value (the go_back loop may repeat
    # the wait routine if the researcher goes back to instructions).
    ref_candidates = df["wait.started"].dropna()
    if ref_candidates.empty:
        print(f"  WARNING: No scanner reference (wait.started) in {psychopy_csv}",
              file=sys.stderr)
        return False
    scanner_ref = float(ref_candidates.iloc[-1])

    bids_ses_num = sess + FR_SESSION_OFFSET

    events_list = []
    for _, row in trial_rows.iterrows():
        trial_num = int(row["trials_recall.thisN"]) + 1

        recall1_start = float(row["recall1.started"])
        prompt_start = float(row["text_recallany.started"])
        prompt_stop = float(row["text_recallany.stopped"])
        text_plus_start = float(row["text_plus.started"])

        # --- Blank (ISI, ~500ms) ---
        events_list.append({
            "onset": recall1_start - scanner_ref,
            "duration": prompt_start - recall1_start,
            "trial_type": "blank",
            "trial_num": trial_num,
            "subj_num": subj,
            "ses_num": bids_ses_num,
            "condition": "",
            "movie_name": "",
            "movie_length": "",
            "style": "",
            "free_recall_position": "",
            "cue_response": "",
            "recall1.started": "",
            "recall1.stopped": "",
        })

        # --- Prompt (instruction display) ---
        events_list.append({
            "onset": prompt_start - scanner_ref,
            "duration": prompt_stop - prompt_start,
            "trial_type": "prompt",
            "trial_num": trial_num,
            "subj_num": subj,
            "ses_num": bids_ses_num,
            "condition": "",
            "movie_name": "",
            "movie_length": "",
            "style": "",
            "free_recall_position": "",
            "cue_response": "",
            "recall1.started": "",
            "recall1.stopped": "",
        })

        # --- Determine cue response and timing ---
        # Primary: check if image was actually displayed (> 100ms)
        img_start = row.get("image_stim.started")
        img_stop = row.get("image_stim.stopped")
        has_real_image = (pd.notna(img_start) and pd.notna(img_stop)
                         and str(img_start).strip() not in ("", "None")
                         and str(img_stop).strip() not in ("", "None")
                         and float(img_stop) - float(img_start) > 0.1)

        if has_real_image:
            cue_response = 7
            cue_onset = float(img_start) - scanner_ref
            cue_dur = float(img_stop) - float(img_start)
        else:
            # Fallback: derive cue_response from key_resp_conditional.keys
            keys_val = row.get("key_resp_conditional.keys")
            if pd.isna(keys_val) or str(keys_val).strip() in ("", "None"):
                keys_val = row.get("trials_recall.key_resp_conditional.keys")

            if pd.notna(keys_val) and str(keys_val).strip() not in ("", "None"):
                key_str = str(keys_val).strip()
                if key_str in ("7", "7.0", "apostrophe"):
                    cue_response = 7
                elif key_str in ("6", "6.0"):
                    cue_response = 6
                else:
                    try:
                        cue_response = int(float(key_str))
                    except (ValueError, TypeError):
                        cue_response = ""
            else:
                cue_response = ""

            cue_onset = text_plus_start - scanner_ref
            cue_dur = 0.0

        # Movie metadata (for cue and recall events)
        movie_name = row["movie_name"] if pd.notna(row["movie_name"]) else ""
        condition = int_or_na(row["condition"])
        movie_length = float(row["mov_len"]) if pd.notna(row.get("mov_len")) else ""
        style = row["style"] if pd.notna(row.get("style")) else ""
        free_recall_pos = int_or_na(row["free_recall_position"])

        # --- Cue (conditional image display) ---
        events_list.append({
            "onset": cue_onset,
            "duration": cue_dur,
            "trial_type": "cue",
            "trial_num": trial_num,
            "subj_num": subj,
            "ses_num": bids_ses_num,
            "condition": condition,
            "movie_name": movie_name,
            "movie_length": movie_length,
            "style": style,
            "free_recall_position": free_recall_pos,
            "cue_response": cue_response,
            "recall1.started": "",
            "recall1.stopped": "",
        })

        # --- Recall (active recall period) ---
        recall1_stop_raw = row.get("recall1.stopped")
        recall1_stop = (float(recall1_stop_raw)
                        if pd.notna(recall1_stop_raw) else None)
        recall_onset = text_plus_start - scanner_ref
        recall_dur = (recall1_stop - text_plus_start
                      if recall1_stop is not None else NA)

        events_list.append({
            "onset": recall_onset,
            "duration": recall_dur,
            "trial_type": "recall",
            "trial_num": trial_num,
            "subj_num": subj,
            "ses_num": bids_ses_num,
            "condition": condition,
            "movie_name": movie_name,
            "movie_length": movie_length,
            "style": style,
            "free_recall_position": free_recall_pos,
            "cue_response": cue_response,
            "recall1.started": recall1_start - scanner_ref,
            "recall1.stopped": (recall1_stop - scanner_ref
                                if recall1_stop is not None else ""),
        })

    # --- Congrats event (post-run) ---
    congrats_start = df["congrats.started"].dropna()
    congrats_stop = df["congrats.stopped"].dropna()
    if not congrats_start.empty and not congrats_stop.empty:
        events_list.append({
            "onset": float(congrats_start.iloc[0]) - scanner_ref,
            "duration": float(congrats_stop.iloc[0]) - float(congrats_start.iloc[0]),
            "trial_type": "congrats",
            "trial_num": "",
            "subj_num": subj,
            "ses_num": bids_ses_num,
            "condition": "",
            "movie_name": "",
            "movie_length": "",
            "style": "",
            "free_recall_position": "",
            "cue_response": "",
            "recall1.started": "",
            "recall1.stopped": "",
        })

    events = pd.DataFrame(events_list)
    write_events_tsv(events, output_tsv, dry_run=dry_run)

    json_path = output_tsv.replace("_events.tsv", "_events.json")
    write_json_sidecar(SIDECAR, json_path, dry_run=dry_run)
    return True


SIDECAR = {
    "onset": {
        "Description": "Event onset time relative to scanner start (wait.started)",
        "Units": "s",
    },
    "duration": {
        "Description": "Event duration",
        "Units": "s",
    },
    "trial_type": {
        "Description": "Type of event",
        "Levels": {
            "blank": "Brief blank period (500ms ISI) at start of each recall trial",
            "prompt": "Recall instruction text display (participant presses 6 or 7)",
            "cue": "Image cue display (duration > 0 when participant pressed 7; duration = 0 when pressed 6)",
            "recall": "Active free recall period; ended by spacebar press",
            "congrats": "Post-run congratulations screen",
        },
    },
    "trial_num": {
        "Description": "Trial number within the session (1-indexed)",
        "Units": "integer",
    },
    "subj_num": {
        "Description": "Subject identifier number",
        "Units": "integer",
    },
    "ses_num": {
        "Description": "BIDS session number",
        "Units": "integer",
    },
    "condition": {
        "Description": "Experimental condition for the movie (recall events only)",
    },
    "movie_name": {
        "Description": "Name of the movie being recalled (recall events only)",
    },
    "movie_length": {
        "Description": "Duration of the movie in seconds (recall events only)",
        "Units": "s",
    },
    "style": {
        "Description": "Movie style description (recall events only)",
    },
    "free_recall_position": {
        "Description": "Order in which this movie was recalled within the session (cue/recall events only)",
    },
    "cue_response": {
        "Description": "Key pressed during prompt: 6 = skipped image cue, 7 = image cue shown (cue/recall events only)",
        "Levels": {
            "6": "Participant did not need the image cue (cue skipped)",
            "7": "Participant requested the image cue (cue shown)",
        },
    },
    "recall1.started": {
        "Description": "Onset of the recall1 routine relative to scanner start (wait.started; recall events only)",
        "Units": "s",
    },
    "recall1.stopped": {
        "Description": "Offset of the recall1 routine relative to scanner start (wait.started; recall events only)",
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
