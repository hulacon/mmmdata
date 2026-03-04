#!/usr/bin/env python3
"""Convert behavioral CSV + timing CSV pairs into BIDS _events.tsv files.

Handles these file types (conversion_type='timed_events'):
  - Cued recall encoding  (ses-04 to ses-18)
  - Cued recall math      (ses-04 to ses-18)
  - Cued recall retrieval (ses-04 to ses-18)
  - Free recall math      (ses-19 to ses-28)
  - Final cued recall     (ses-30)

Each source behavioral CSV is paired with a timing CSV that provides
scanner-relative onset/offset values. The behavioral CSV provides trial-level
stimulus and response data.

Usage:
    python timed_events.py <behavioral_csv> <timing_csv> <output_events_tsv> [--dry-run]
"""

import argparse
import os
import re
import sys

import pandas as pd

from common import (
    NA, BIDS_ROOT, CR_SESSION_OFFSET, FINAL_SESSION, FR_SESSION_OFFSET,
    bids_ses_cr, bids_ses_fr, bids_sub, float_or_na, int_or_na,
    write_events_tsv, write_json_sidecar,
)


# ============================================================================
# Detect task type from filename
# ============================================================================

def detect_task(behavioral_path):
    """Detect the task type from the behavioral CSV filename.

    Returns one of: 'cued_recall_encoding', 'cued_recall_math',
    'cued_recall_retrieval', 'free_recall_math', 'final_cued_recall'
    """
    fname = os.path.basename(behavioral_path)

    if fname.startswith("cued_recall_encoding_"):
        return "cued_recall_encoding"
    if fname.startswith("cued_recall_math_"):
        return "cued_recall_math"
    if fname.startswith("cued_recall_retrieval_"):
        return "cued_recall_retrieval"
    if fname.startswith("free_recall_math_"):
        return "free_recall_math"
    if fname.startswith("final_cued_recall_"):
        return "final_cued_recall"

    raise ValueError(f"Cannot detect task type from filename: {fname}")


def parse_subj_sess_run(behavioral_path):
    """Extract subject, session, run numbers from behavioral filename."""
    fname = os.path.basename(behavioral_path)
    m = re.search(r"subj(\d+)_sess(\d+)_run(\d+)", fname)
    if not m:
        raise ValueError(f"Cannot parse subj/sess/run from: {fname}")
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


# ============================================================================
# Conversion functions per task type
# ============================================================================

def convert_cued_recall_encoding(beh_df, timing_df, subj, sess, run):
    """Cued recall encoding -> BIDS events.

    Behavioral columns: subjId, session, run, trial, pairId, mmmId, nsdId,
        itmno, word, voiceId, sharedId, enCon, reCon, voice, resp, resp_RT
    Timing columns: sub_id, task_id, sess_id, run_id, trial_id,
        stim_image_start, stim_image_end, stim_word_start, stim_word_end,
        stim_fixation_start, stim_fixation_end

    onset = stim_image_start (scanner-relative)
    duration = stim_image_end - stim_image_start
    trial_type = "image"
    modality = "visual"
    """
    # Merge on trial number
    beh_df = beh_df.copy()
    timing_df = timing_df.copy()
    beh_df["trial_id"] = beh_df["trial"]
    merged = pd.merge(beh_df, timing_df, on="trial_id", suffixes=("", "_timing"))

    # Detect rest trials: pairId == 0 or word is NaN
    is_rest = merged["word"].isna() | (merged["pairId"].astype(float) == 0)

    onset_actual = merged["stim_image_start"].astype(float).values.copy()
    img_end = merged["stim_image_end"].astype(float).values
    duration_actual = (img_end - onset_actual).copy()

    # For rest trials, onset_actual = previous trial's stim_fixation_end
    fix_end = merged["stim_fixation_end"].astype(float).values
    for i in range(len(merged)):
        if is_rest.iloc[i] and i > 0:
            onset_actual[i] = fix_end[i - 1]
            duration_actual[i] = 3.0  # scheduled duration

    # Scheduled onset: 4.5s spacing starting at 9.0s
    n_trials = len(merged)
    onset_scheduled = [9.0 + 4.5 * i for i in range(n_trials)]

    # Build trial_type and modality based on rest status
    trial_type = ["rest" if is_rest.iloc[i] else "image" for i in range(n_trials)]
    modality = [NA if is_rest.iloc[i] else "visual" for i in range(n_trials)]

    # For rest trials, null out behavioral columns
    def maybe_na(series, rest_mask, converter=None):
        vals = []
        for i, v in enumerate(series):
            if rest_mask.iloc[i]:
                vals.append(NA)
            elif converter:
                vals.append(converter(v))
            else:
                vals.append(v)
        return vals

    events = pd.DataFrame({
        "onset": onset_scheduled,
        "duration": [3.0] * n_trials,
        "onset_actual": onset_actual,
        "duration_actual": duration_actual,
        "subj_num": subj,
        "ses_num": sess,
        "run_idx": run,
        "trial_type": trial_type,
        "modality": modality,
        "word": maybe_na(merged["word"], is_rest),
        "pairId": maybe_na(merged["pairId"], is_rest, int_or_na),
        "mmmId": maybe_na(merged["mmmId"], is_rest, int_or_na),
        "nsdId": maybe_na(merged["nsdId"], is_rest, int_or_na),
        "itmno": maybe_na(merged["itmno"], is_rest, int_or_na),
        "sharedId": maybe_na(merged["sharedId"], is_rest, int_or_na),
        "voiceId": maybe_na(merged["voiceId"], is_rest, int_or_na),
        "voice": maybe_na(merged["voice"], is_rest),
        "enCon": maybe_na(merged["enCon"], is_rest, int_or_na),
        "reCon": maybe_na(merged["reCon"], is_rest, int_or_na),
        "resp": maybe_na(merged["resp"], is_rest, int_or_na),
        "resp_RT": maybe_na(merged["resp_RT"], is_rest, float_or_na),
        "trial_id": merged["trial_id"].values,
    })
    return events


def convert_cued_recall_math(beh_df, timing_df, subj, sess, run):
    """Cued recall math -> BIDS events.

    Behavioral columns: problem, answer, trial, resp, resp_RT
    Timing columns: sub_id, task_id, sess_id, run_id, trial_id,
        stim_image_start, stim_image_end, stim_fixation_start, stim_fixation_end

    onset = stim_image_start
    duration = stim_image_end - stim_image_start
    """
    beh_df = beh_df.copy()
    timing_df = timing_df.copy()
    beh_df["trial_id"] = beh_df["trial"]
    merged = pd.merge(beh_df, timing_df, on="trial_id", suffixes=("", "_timing"))

    onset_actual = merged["stim_image_start"].astype(float)
    duration_actual = merged["stim_image_end"].astype(float) - onset_actual

    n_trials = len(merged)
    onset_scheduled = [6.0 + 4.5 * i for i in range(n_trials)]

    events = pd.DataFrame({
        "onset": onset_scheduled,
        "duration": [3.0] * n_trials,
        "onset_actual": onset_actual.values,
        "duration_actual": duration_actual.values,
        "subj_num": subj,
        "ses_num": sess,
        "run_idx": run,
        "trial_type": "math",
        "modality": "visual",
        "problem": merged["problem"].values,
        "correct_answer": merged["answer"].apply(int_or_na).values,
        "resp": merged["resp"].apply(int_or_na).values,
        "resp_RT": merged["resp_RT"].apply(float_or_na).values,
        "trial_id": merged["trial_id"].values,
    })
    return events


def convert_cued_recall_retrieval(beh_df, timing_df, subj, sess, run):
    """Cued recall retrieval -> BIDS events.

    Behavioral columns: subjId, session, run, trial, cueId, pairId, mmmId,
        nsdId, itmno, word, voiceId, voice, sharedId, enCon, reCon, resp, resp_RT
    Timing columns: sub_id, task_id, sess_id, run_id, trial_id,
        stim_image_start, stim_image_end, stim_word_start, stim_word_end,
        stim_fixation_start, stim_fixation_end

    onset = stim_word_start (word cue; stim_image columns are empty)
    duration = stim_word_end - stim_word_start

    Rest trials: behavioral fields are empty, timing duplicates previous trial.
    For rest trials, use fixation_end of previous trial as onset_actual,
    and scheduled duration (0.54s) as duration_actual.
    """
    beh_df = beh_df.copy()
    timing_df = timing_df.copy()
    beh_df["trial_id"] = beh_df["trial"]
    merged = pd.merge(beh_df, timing_df, on="trial_id", suffixes=("", "_timing"))

    # Detect rest trials: cueId is NaN/empty
    is_rest = merged["cueId"].isna() | (merged["cueId"].astype(str).str.strip() == "")

    # For normal trials, onset = stim_word_start
    onset_actual = merged["stim_word_start"].astype(float).values.copy()
    word_end = merged["stim_word_end"].astype(float).values
    duration_actual = (word_end - onset_actual).copy()

    # For rest trials, use previous trial's stim_fixation_end as onset
    fix_end = merged["stim_fixation_end"].astype(float).values
    for i in range(len(merged)):
        if is_rest.iloc[i] and i > 0:
            onset_actual[i] = fix_end[i - 1]
            duration_actual[i] = 0.54  # scheduled duration

    n_trials = len(merged)
    onset_scheduled = [9.0 + 4.5 * i for i in range(n_trials)]

    # Build trial_type: "word" for normal, "rest" for rest
    trial_type = ["rest" if is_rest.iloc[i] else "word" for i in range(n_trials)]
    modality = ["" if is_rest.iloc[i] else "auditory" for i in range(n_trials)]

    # For rest trials, null out behavioral columns
    def maybe_na(series, rest_mask, converter=None):
        vals = []
        for i, v in enumerate(series):
            if rest_mask.iloc[i]:
                vals.append(NA)
            elif converter:
                vals.append(converter(v))
            else:
                vals.append(v)
        return vals

    events = pd.DataFrame({
        "onset": onset_scheduled,
        "duration": [0.54] * n_trials,
        "onset_actual": onset_actual,
        "duration_actual": duration_actual,
        "subj_num": subj,
        "ses_num": sess,
        "run_idx": run,
        "trial_type": trial_type,
        "modality": modality,
        "word": maybe_na(merged["word"], is_rest),
        "pairId": maybe_na(merged["pairId"], is_rest, int_or_na),
        "mmmId": maybe_na(merged["mmmId"], is_rest, int_or_na),
        "nsdId": maybe_na(merged["nsdId"], is_rest, int_or_na),
        "itmno": maybe_na(merged["itmno"], is_rest, int_or_na),
        "sharedId": maybe_na(merged["sharedId"], is_rest, int_or_na),
        "voiceId": maybe_na(merged["voiceId"], is_rest, int_or_na),
        "voice": maybe_na(merged["voice"], is_rest),
        "enCon": maybe_na(merged["enCon"], is_rest, int_or_na),
        "reCon": maybe_na(merged["reCon"], is_rest, int_or_na),
        "resp": maybe_na(merged["resp"], is_rest, int_or_na),
        "resp_RT": maybe_na(merged["resp_RT"], is_rest, float_or_na),
        "cueId": maybe_na(merged["cueId"], is_rest, float_or_na),
        "trial_id": merged["trial_id"].values,
    })
    return events


def convert_free_recall_math(beh_df, timing_df, subj, sess, run):
    """Free recall math -> BIDS events.

    Same structure as cued recall math. Behavioral columns: problem, answer,
    trial, resp, resp_RT.
    """
    # Reuse cued recall math logic (identical column structure)
    return convert_cued_recall_math(beh_df, timing_df, subj, sess, run)


def convert_final_cued_recall(beh_df, timing_df, subj, sess, run):
    """Final cued recall -> BIDS events.

    Behavioral columns: subjId, session, run, trial, enSession, enRun,
        enTrial, pairId, mmmId, nsdId, itmno, word, voiceId, sharedId,
        enCon, reCon, voice, trial_accuracy, cueId, resp, resp_RT
    Timing columns: same as cued recall retrieval (stim_image empty, uses stim_word)

    Same structure as cued recall retrieval but with extra columns
    (enSession, enRun, enTrial, trial_accuracy).
    """
    beh_df = beh_df.copy()
    timing_df = timing_df.copy()
    beh_df["trial_id"] = beh_df["trial"]
    merged = pd.merge(beh_df, timing_df, on="trial_id", suffixes=("", "_timing"))

    onset_actual = merged["stim_word_start"].astype(float)
    fix_end = merged["stim_fixation_end"].astype(float)
    duration_actual = fix_end - onset_actual  # full trial duration

    n_trials = len(merged)

    events = pd.DataFrame({
        "onset": onset_actual.values,
        "duration": duration_actual.values,
        "trial_type": "word",
        "word": merged["word"].values,
        "voice": merged["voice"].values,
        "pairId": merged["pairId"].apply(int_or_na).values,
        "mmmId": merged["mmmId"].apply(int_or_na).values,
        "nsdId": merged["nsdId"].apply(int_or_na).values,
        "sharedId": merged["sharedId"].apply(int_or_na).values,
        "enCon": merged["enCon"].apply(int_or_na).values,
        "reCon": merged["reCon"].apply(int_or_na).values,
        "response": merged["resp"].apply(int_or_na).values,
        "response_time": merged["resp_RT"].apply(float_or_na).values,
        "trial_id": merged["trial_id"].values,
    })
    return events


# ============================================================================
# JSON sidecar definitions
# ============================================================================

SIDECAR_CUED_ENCODING = {
    "onset": {"Description": "Scheduled event onset time (design-based)", "Units": "s"},
    "duration": {"Description": "Scheduled event duration (design-based)", "Units": "s"},
    "onset_actual": {"Description": "Measured event onset time", "Units": "s"},
    "duration_actual": {"Description": "Measured event duration", "Units": "s"},
    "subj_num": {"Description": "Subject identifier number"},
    "ses_num": {"Description": "BIDS session number"},
    "run_idx": {"Description": "Run number within the session"},
    "trial_type": {
        "Description": "Type of trial event",
        "Levels": {
            "image": "Visual image presentation",
            "rest": "Empty trial (fixation cross, no stimulus)",
        },
    },
    "modality": {
        "Description": "Stimulus modality",
        "Levels": {
            "visual": "Visual image presentation",
            "": "None (rest)",
        },
    },
    "word": {"Description": "Word stimulus presented with the image"},
    "pairId": {"Description": "Unique identifier for stimulus pair"},
    "mmmId": {"Description": "Unique stimulus ID in multimodal memory dataset"},
    "nsdId": {"Description": "Identifier for NSD stimulus set"},
    "itmno": {"Description": "Internal stimulus number"},
    "sharedId": {"Description": "Shared stimulus ID across conditions"},
    "voiceId": {"Description": "Numeric code for voice identity in auditory trials"},
    "voice": {"Description": "Label for voice identity in auditory trials"},
    "enCon": {"Description": "Encoding condition (1=single, 2=repeats, 3=triplets)"},
    "reCon": {"Description": "Retrieval condition (1=within, 2=across)"},
    "resp": {"Description": "Participant response button", "Units": "button number"},
    "resp_RT": {"Description": "Reaction time for participant response", "Units": "s"},
    "trial_id": {"Description": "Sequential trial number within the run"},
}

SIDECAR_CUED_MATH = {
    "onset": {"Description": "Scheduled event onset time (design-based)", "Units": "s"},
    "duration": {"Description": "Scheduled event duration (design-based)", "Units": "s"},
    "onset_actual": {"Description": "Measured event onset time", "Units": "s"},
    "duration_actual": {"Description": "Measured event duration", "Units": "s"},
    "subj_num": {"Description": "Subject identifier number"},
    "ses_num": {"Description": "BIDS session number"},
    "run_idx": {"Description": "Run number within the session"},
    "trial_type": {
        "Description": "Type of trial event",
        "Levels": {"math": "Math distractor problem"},
    },
    "modality": {
        "Description": "Stimulus modality",
        "Levels": {"visual": "Visual math problem"},
    },
    "problem": {"Description": "Math problem string"},
    "correct_answer": {"Description": "Correct answer to the math problem"},
    "resp": {"Description": "Participant response button", "Units": "button number"},
    "resp_RT": {"Description": "Reaction time for participant response", "Units": "s"},
    "trial_id": {"Description": "Sequential trial number within the run"},
}

SIDECAR_CUED_RETRIEVAL = {
    "onset": {"Description": "Scheduled event onset time (design-based)", "Units": "s"},
    "duration": {"Description": "Scheduled event duration (design-based)", "Units": "s"},
    "onset_actual": {"Description": "Measured event onset time", "Units": "s"},
    "duration_actual": {"Description": "Measured event duration", "Units": "s"},
    "subj_num": {"Description": "Subject identifier number"},
    "ses_num": {"Description": "BIDS session number"},
    "run_idx": {"Description": "Run number within the session"},
    "trial_type": {
        "Description": "Type of trial event",
        "Levels": {
            "word": "Auditory word presentation",
            "rest": "Empty trial (fixation cross, no stimulus)",
        },
    },
    "modality": {
        "Description": "Stimulus modality",
        "Levels": {
            "auditory": "Auditory word presentation",
            "": "None (rest)",
        },
    },
    "word": {"Description": "Word stimulus presented, if applicable"},
    "pairId": {"Description": "Unique identifier for stimulus pair"},
    "mmmId": {"Description": "Unique stimulus ID in multimodal memory dataset"},
    "nsdId": {"Description": "Identifier for NSD stimulus set"},
    "itmno": {"Description": "Internal stimulus number"},
    "sharedId": {"Description": "Shared stimulus ID across conditions"},
    "voiceId": {"Description": "Numeric code for voice identity in auditory trials"},
    "voice": {"Description": "Label for voice identity in auditory trials"},
    "enCon": {"Description": "Encoding condition (1=single, 2=repeats, 3=triplets)"},
    "reCon": {"Description": "Retrieval condition (1=within, 2=across)"},
    "resp": {"Description": "Participant response button", "Units": "button number"},
    "resp_RT": {"Description": "Reaction time for participant response", "Units": "s"},
    "cueId": {"Description": "Retrieval cue identifier"},
    "trial_id": {"Description": "Sequential trial number within the run"},
}

SIDECAR_FINAL_CUED_RECALL = {
    "onset": {"Description": "Event onset time relative to scanner start", "Units": "s"},
    "duration": {"Description": "Full trial duration", "Units": "s"},
    "trial_type": {
        "Description": "Type of trial event",
        "Levels": {"word": "Auditory word cue presentation"},
    },
    "word": {"Description": "Cue word presented"},
    "voice": {"Description": "Voice identity label"},
    "pairId": {"Description": "Unique identifier for stimulus pair"},
    "mmmId": {"Description": "Unique stimulus ID in multimodal memory dataset"},
    "nsdId": {"Description": "Identifier for NSD stimulus set"},
    "sharedId": {"Description": "Shared stimulus ID across conditions"},
    "enCon": {"Description": "Encoding condition (1=single, 2=repeats, 3=triplets)"},
    "reCon": {"Description": "Retrieval condition (1=within, 2=across)"},
    "response": {"Description": "Participant response button", "Units": "button number"},
    "response_time": {"Description": "Reaction time for participant response", "Units": "s"},
    "trial_id": {"Description": "Sequential trial number within the run"},
}

# Free recall math uses same sidecar as cued recall math
SIDECAR_FREE_MATH = SIDECAR_CUED_MATH.copy()


def get_sidecar(task):
    """Return the appropriate JSON sidecar definition for a task."""
    return {
        "cued_recall_encoding": SIDECAR_CUED_ENCODING,
        "cued_recall_math": SIDECAR_CUED_MATH,
        "cued_recall_retrieval": SIDECAR_CUED_RETRIEVAL,
        "free_recall_math": SIDECAR_FREE_MATH,
        "final_cued_recall": SIDECAR_FINAL_CUED_RECALL,
    }[task]


# ============================================================================
# Main conversion entry point
# ============================================================================

def convert_file(behavioral_csv, timing_csv, output_tsv, dry_run=False):
    """Convert a behavioral+timing CSV pair to a BIDS events TSV+JSON.

    Parameters
    ----------
    behavioral_csv : str
        Path to the behavioral data CSV.
    timing_csv : str
        Path to the companion timing CSV.
    output_tsv : str
        Destination path for the BIDS events TSV.
    dry_run : bool
        If True, print what would be done without writing files.

    Returns
    -------
    bool
        True if conversion succeeded.
    """
    task = detect_task(behavioral_csv)
    subj, sess, run = parse_subj_sess_run(behavioral_csv)

    # Map raw behavioral session number to absolute BIDS session number
    if task.startswith("cued_recall"):
        bids_session = sess + CR_SESSION_OFFSET
    elif task == "free_recall_math":
        bids_session = sess + FR_SESSION_OFFSET
    elif task == "final_cued_recall":
        bids_session = FINAL_SESSION
    else:
        raise ValueError(f"Unknown task type for session mapping: {task}")

    beh_df = pd.read_csv(behavioral_csv)
    timing_df = pd.read_csv(timing_csv)

    converters = {
        "cued_recall_encoding": convert_cued_recall_encoding,
        "cued_recall_math": convert_cued_recall_math,
        "cued_recall_retrieval": convert_cued_recall_retrieval,
        "free_recall_math": convert_free_recall_math,
        "final_cued_recall": convert_final_cued_recall,
    }

    events_df = converters[task](beh_df, timing_df, subj, bids_session, run)
    write_events_tsv(events_df, output_tsv, dry_run=dry_run)

    # Write JSON sidecar (same path but .json extension)
    json_path = output_tsv.replace("_events.tsv", "_events.json")
    write_json_sidecar(get_sidecar(task), json_path, dry_run=dry_run)

    return True


def find_timing_csv(behavioral_csv):
    """Locate the companion timing CSV for a behavioral CSV.

    Searches in timing_data/ or timing/ subdirectory relative to the
    behavioral file, or in parent directories.
    """
    beh_dir = os.path.dirname(behavioral_csv)
    fname = os.path.basename(behavioral_csv)

    # Build timing filename: insert '_timing' before '.csv'
    timing_fname = fname.replace(".csv", "_timing.csv")

    # Search locations (in priority order)
    search_dirs = [
        beh_dir,                                # same directory as behavioral
        os.path.join(beh_dir, "timing_data"),
        os.path.join(beh_dir, "Timing"),        # sub_05 variant
        os.path.join(beh_dir, "timing"),         # final cued recall variant
    ]

    for d in search_dirs:
        candidate = os.path.join(d, timing_fname)
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(
        f"Cannot find timing CSV for {behavioral_csv}. "
        f"Searched for {timing_fname} in: {search_dirs}"
    )


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Convert behavioral+timing CSV pair to BIDS events TSV"
    )
    parser.add_argument("behavioral_csv", help="Path to behavioral data CSV")
    parser.add_argument("timing_csv", nargs="?", default=None,
                        help="Path to timing CSV (auto-detected if omitted)")
    parser.add_argument("output_tsv", nargs="?", default=None,
                        help="Output events TSV path (auto-generated if omitted)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print actions without writing files")
    args = parser.parse_args()

    timing = args.timing_csv or find_timing_csv(args.behavioral_csv)

    if args.output_tsv is None:
        # Auto-generate output path from behavioral filename
        task = detect_task(args.behavioral_csv)
        subj, sess, run = parse_subj_sess_run(args.behavioral_csv)
        sub = bids_sub(subj)

        if task.startswith("cued_recall"):
            ses = bids_ses_cr(sess)
            task_label = "TB" + task.replace("cued_recall_", "")
        elif task == "free_recall_math":
            ses = bids_ses_fr(sess)
            task_label = "NATmath"
        elif task == "final_cued_recall":
            ses = f"ses-{FINAL_SESSION:02d}"
            task_label = "FINretrieval"
        else:
            raise ValueError(f"Unknown task: {task}")

        if task_label.endswith("math"):
            fname = f"{sub}_{ses}_task-{task_label}_events.tsv"
        else:
            fname = f"{sub}_{ses}_task-{task_label}_run-{run:02d}_events.tsv"

        output = os.path.join(BIDS_ROOT, sub, ses, "func", fname)
    else:
        output = args.output_tsv

    print(f"Task: {detect_task(args.behavioral_csv)}")
    print(f"Input: {args.behavioral_csv}")
    print(f"Timing: {timing}")
    print(f"Output: {output}")
    convert_file(args.behavioral_csv, timing, output, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
