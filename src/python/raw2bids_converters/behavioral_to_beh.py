#!/usr/bin/env python3
"""Convert out-of-scanner behavioral CSVs into BIDS _beh.tsv files.

Handles files with conversion_type='behavioral_to_beh' (51 files total):
  - Cued recall outscan recognition (45 files, 15 per subject -> ses-04 to ses-18)
  - Final recognition (3 files, 1 per subject -> ses-30)
  - Final timeline sequence (3 files, 1 per subject -> ses-30)

These are self-paced tasks performed outside the scanner, so there are no
scanner-relative timings. The output goes in beh/ subdirectories.

For recognition tasks, onset is computed as cumulative response time
(each trial starts when the previous response ends).

Usage:
    python behavioral_to_beh.py <behavioral_csv> [<output_beh_tsv>] [--dry-run]
"""

import argparse
import os
import re
import sys

import pandas as pd

from common import (
    NA, BIDS_ROOT, CR_SESSION_OFFSET, FINAL_SESSION,
    bids_sub, bids_ses, bids_ses_cr, float_or_na, int_or_na,
    write_beh_tsv, write_json_sidecar,
)


def detect_task(csv_path):
    """Detect behavioral task type from filename."""
    fname = os.path.basename(csv_path)
    if "recognition_outscan" in fname:
        return "outscan_recognition"
    if "final_recognition" in fname:
        return "final_recognition"
    if "final_timeline" in fname:
        return "final_timeline"
    raise ValueError(f"Cannot detect task type from: {fname}")


def parse_subj_sess_run(csv_path):
    """Extract subject, session, run from behavioral filename."""
    fname = os.path.basename(csv_path)
    m = re.search(r"subj(\d+)_sess(\d+)_run(\d+)", fname)
    if not m:
        raise ValueError(f"Cannot parse subj/sess/run from: {fname}")
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def convert_outscan_recognition(csv_path, output_tsv, dry_run=False):
    """Convert cued recall outscan recognition -> BIDS beh TSV.

    Columns: subjId, session, run, trial, cueId, pairId, mmmId, nsdId, itmno,
    word, voiceId, voice, sharedId, enCon, reCon, mmmId_lure, nsdId_lure,
    image1, image2, correct_resp, resp, resp_RT, recog, trial_accuracy

    onset = cumulative resp_RT (self-paced)
    duration = resp_RT for current trial
    """
    subj, sess, run = parse_subj_sess_run(csv_path)
    bids_session = sess + CR_SESSION_OFFSET
    df = pd.read_csv(csv_path)

    # Compute cumulative onset from resp_RT
    rt_vals = df["resp_RT"].astype(float)
    onsets = rt_vals.cumsum().shift(1, fill_value=0.0)

    events = pd.DataFrame({
        "onset": onsets.values,
        "duration": rt_vals.values,
        "subj_num": subj,
        "ses_num": bids_session,
        "run_idx": run,
        "encoding_run": df["run"].apply(int_or_na).values,
        "trial_type": "recognition",
        "modality": "visual",
        "word": df["word"].values,
        "image1": df["image1"].values,
        "image2": df["image2"].values,
        "correct_resp": df["correct_resp"].apply(int_or_na).values,
        "resp": df["resp"].apply(int_or_na).values,
        "resp_RT": rt_vals.values,
        "trial_accuracy": df["trial_accuracy"].apply(float_or_na).values,
        "enCon": df["enCon"].apply(int_or_na).values,
        "reCon": df["reCon"].apply(int_or_na).values,
        "cueId": df["cueId"].apply(float_or_na).values,
        "pairId": df["pairId"].apply(int_or_na).values,
        "recog": df["recog"].apply(float_or_na).values,
        "trial_id": df["trial"].values,
    })

    write_beh_tsv(events, output_tsv, dry_run=dry_run)

    json_path = output_tsv.replace("_beh.tsv", "_beh.json")
    write_json_sidecar(SIDECAR_RECOGNITION, json_path, dry_run=dry_run)
    return True


def convert_final_recognition(csv_path, output_tsv, dry_run=False):
    """Convert final recognition -> BIDS beh TSV.

    Columns: subjId, session, run, trial, enSession, enRun, enTrial, pairId,
    mmmId, nsdId, itmno, word, voiceId, sharedId, enCon, reCon, voice,
    trial_accuracy, cueId, mmmId_lure, nsdId_lure, image1, image2,
    ans, resp, resp_RT, recog, accuracy
    """
    subj, sess, run = parse_subj_sess_run(csv_path)
    df = pd.read_csv(csv_path)

    rt_vals = df["resp_RT"].astype(float)
    onsets = rt_vals.cumsum().shift(1, fill_value=0.0)

    events = pd.DataFrame({
        "onset": onsets.values,
        "duration": rt_vals.values,
        "subj_num": subj,
        "ses_num": FINAL_SESSION,
        "run_idx": run,
        "trial_type": "recognition",
        "modality": "visual",
        "word": df["word"].values,
        "image1": df["image1"].values,
        "image2": df["image2"].values,
        "correct_resp": df["ans"].apply(int_or_na).values,
        "resp": df["resp"].apply(int_or_na).values,
        "resp_RT": rt_vals.values,
        "accuracy": df["accuracy"].apply(float_or_na).values,
        "trial_accuracy": df["trial_accuracy"].apply(float_or_na).values,
        "enCon": df["enCon"].apply(int_or_na).values,
        "reCon": df["reCon"].apply(int_or_na).values,
        "cueId": df["cueId"].apply(float_or_na).values,
        "pairId": df["pairId"].apply(int_or_na).values,
        "recog": df["recog"].apply(float_or_na).values,
        "enSession": df["enSession"].apply(int_or_na).values,
        "enRun": df["enRun"].apply(int_or_na).values,
        "enTrial": df["enTrial"].apply(int_or_na).values,
        "trial_id": df["trial"].values,
    })

    write_beh_tsv(events, output_tsv, dry_run=dry_run)

    json_path = output_tsv.replace("_beh.tsv", "_beh.json")
    write_json_sidecar(SIDECAR_FINAL_RECOGNITION, json_path, dry_run=dry_run)
    return True


def convert_final_timeline(csv_path, output_tsv, dry_run=False):
    """Convert final timeline sequence -> BIDS beh TSV.

    Columns: subjId, session, run, trial, enSession, enRun, enTrial, pairId,
    mmmId, nsdId, itmno, word, voiceId, sharedId, enCon, reCon, voice,
    trial_accuracy, cueId, timeline_RT, timeline_resp
    """
    subj, sess, run = parse_subj_sess_run(csv_path)
    df = pd.read_csv(csv_path)

    rt_vals = df["timeline_RT"].astype(float)
    onsets = rt_vals.cumsum().shift(1, fill_value=0.0)

    events = pd.DataFrame({
        "onset": onsets.values,
        "duration": rt_vals.values,
        "subj_num": subj,
        "ses_num": FINAL_SESSION,
        "run_idx": run,
        "trial_type": "timeline",
        "modality": "visual",
        "word": df["word"].values,
        "timeline_resp": df["timeline_resp"].apply(float_or_na).values,
        "timeline_RT": rt_vals.values,
        "trial_accuracy": df["trial_accuracy"].apply(float_or_na).values,
        "enCon": df["enCon"].apply(int_or_na).values,
        "reCon": df["reCon"].apply(int_or_na).values,
        "cueId": df["cueId"].apply(float_or_na).values,
        "pairId": df["pairId"].apply(int_or_na).values,
        "mmmId": df["mmmId"].apply(int_or_na).values,
        "nsdId": df["nsdId"].apply(int_or_na).values,
        "itmno": df["itmno"].apply(int_or_na).values,
        "sharedId": df["sharedId"].apply(int_or_na).values,
        "voiceId": df["voiceId"].apply(int_or_na).values,
        "voice": df["voice"].values,
        "enSession": df["enSession"].apply(int_or_na).values,
        "enRun": df["enRun"].apply(int_or_na).values,
        "enTrial": df["enTrial"].apply(int_or_na).values,
        "trial_id": df["trial"].values,
    })

    write_beh_tsv(events, output_tsv, dry_run=dry_run)

    json_path = output_tsv.replace("_beh.tsv", "_beh.json")
    write_json_sidecar(SIDECAR_TIMELINE, json_path, dry_run=dry_run)
    return True


def convert_file(csv_path, output_tsv, dry_run=False):
    """Convert a behavioral CSV to BIDS beh TSV+JSON."""
    task = detect_task(csv_path)
    converters = {
        "outscan_recognition": convert_outscan_recognition,
        "final_recognition": convert_final_recognition,
        "final_timeline": convert_final_timeline,
    }
    return converters[task](csv_path, output_tsv, dry_run)


# ============================================================================
# JSON sidecars
# ============================================================================

SIDECAR_RECOGNITION = {
    "onset": {"Description": "Cumulative onset (sum of prior response times)", "Units": "s"},
    "duration": {"Description": "Response time for this trial", "Units": "s"},
    "subj_num": {"Description": "Subject identifier number"},
    "ses_num": {"Description": "BIDS session number"},
    "run_idx": {"Description": "Recognition run number within the session"},
    "encoding_run": {"Description": "Encoding run from which this trial's stimulus pair was originally presented"},
    "trial_type": {
        "Description": "Type of trial",
        "Levels": {"recognition": "Two-alternative forced-choice recognition"},
    },
    "modality": {"Description": "Stimulus modality", "Levels": {"visual": "Visual stimuli"}},
    "word": {"Description": "Associated word for the stimulus pair"},
    "image1": {"Description": "First image filename"},
    "image2": {"Description": "Second image filename"},
    "correct_resp": {"Description": "Correct response (1 or 2, indicating which image)"},
    "resp": {
        "Description": "Participant response (confidence + choice)",
        "Levels": {
            "1": "High confidence left image (image1)",
            "2": "Low confidence left image (image1)",
            "3": "Low confidence right image (image2)",
            "4": "High confidence right image (image2)",
        },
        "Units": "button number",
    },
    "resp_RT": {"Description": "Reaction time", "Units": "s"},
    "trial_accuracy": {"Description": "Whether the trial was correct (1.0) or incorrect (0.0)"},
    "enCon": {"Description": "Encoding condition (1=single, 2=repeats, 3=triplets)"},
    "reCon": {"Description": "Retrieval condition (1=within, 2=across)"},
    "cueId": {"Description": "Cue type identifier (1=visual/image, 2=auditory/word)"},
    "pairId": {"Description": "Unique identifier for stimulus pair"},
    "recog": {"Description": "Recognition condition", "Levels": {"1": "Condition 1", "2": "Condition 2"}},
    "trial_id": {"Description": "Sequential trial number within the run"},
}

SIDECAR_FINAL_RECOGNITION = {
    "onset": {"Description": "Cumulative onset (sum of prior response times)", "Units": "s"},
    "duration": {"Description": "Response time for this trial", "Units": "s"},
    "subj_num": {"Description": "Subject identifier number"},
    "ses_num": {"Description": "BIDS session number"},
    "run_idx": {"Description": "Recognition run number within the session"},
    "trial_type": {
        "Description": "Type of trial",
        "Levels": {"recognition": "Two-alternative forced-choice recognition"},
    },
    "modality": {"Description": "Stimulus modality", "Levels": {"visual": "Visual stimuli"}},
    "word": {"Description": "Associated word for the stimulus pair"},
    "image1": {"Description": "First image filename"},
    "image2": {"Description": "Second image filename"},
    "correct_resp": {"Description": "Correct response (1 or 2)"},
    "resp": {
        "Description": "Participant response (confidence + choice)",
        "Levels": {
            "1": "High confidence left image (image1)",
            "2": "Low confidence left image (image1)",
            "3": "Low confidence right image (image2)",
            "4": "High confidence right image (image2)",
        },
        "Units": "button number",
    },
    "resp_RT": {"Description": "Reaction time", "Units": "s"},
    "accuracy": {"Description": "Final recognition accuracy (0.0 or 1.0)"},
    "trial_accuracy": {"Description": "Previous recognition accuracy (1.0=correct, 0.0=incorrect)"},
    "enCon": {"Description": "Encoding condition (1=single, 2=repeats, 3=triplets)"},
    "reCon": {"Description": "Retrieval condition (1=within, 2=across)"},
    "cueId": {"Description": "Cue type identifier (1=visual/image, 2=auditory/word)"},
    "pairId": {"Description": "Unique identifier for stimulus pair"},
    "recog": {"Description": "Recognition condition", "Levels": {"1": "Condition 1", "2": "Condition 2"}},
    "enSession": {"Description": "Original encoding session number"},
    "enRun": {"Description": "Original encoding run number"},
    "enTrial": {"Description": "Original encoding trial number"},
    "trial_id": {"Description": "Sequential trial number within the run"},
}

SIDECAR_TIMELINE = {
    "onset": {"Description": "Cumulative onset (sum of prior response times)", "Units": "s"},
    "duration": {"Description": "Response time for this trial", "Units": "s"},
    "subj_num": {"Description": "Subject identifier number"},
    "ses_num": {"Description": "BIDS session number"},
    "run_idx": {"Description": "Run number within the session"},
    "trial_type": {
        "Description": "Type of trial",
        "Levels": {"timeline": "Timeline sequence judgment"},
    },
    "modality": {"Description": "Stimulus modality", "Levels": {"visual": "Visual stimuli"}},
    "word": {"Description": "Associated word for the stimulus pair"},
    "timeline_resp": {"Description": "Timeline position response (0-1 scale)"},
    "timeline_RT": {"Description": "Response time for timeline judgment", "Units": "s"},
    "trial_accuracy": {"Description": "Whether the trial was correct (1.0 or 0.0)"},
    "enCon": {"Description": "Encoding condition (1=single, 2=repeats, 3=triplets)"},
    "reCon": {"Description": "Retrieval condition (1=within, 2=across)"},
    "cueId": {"Description": "Cue type identifier (1=visual/image, 2=auditory/word)"},
    "pairId": {"Description": "Unique identifier for stimulus pair"},
    "mmmId": {"Description": "Unique stimulus ID in multimodal memory dataset"},
    "nsdId": {"Description": "Identifier for NSD stimulus set"},
    "itmno": {"Description": "Internal stimulus number"},
    "sharedId": {"Description": "Shared stimulus ID across conditions"},
    "voiceId": {"Description": "Numeric code for voice identity in auditory trials"},
    "voice": {"Description": "Label for voice identity in auditory trials"},
    "enSession": {"Description": "Original encoding session number"},
    "enRun": {"Description": "Original encoding run number"},
    "enTrial": {"Description": "Original encoding trial number"},
    "trial_id": {"Description": "Sequential trial number within the run"},
}


def main():
    parser = argparse.ArgumentParser(
        description="Convert out-of-scanner behavioral CSV to BIDS beh TSV"
    )
    parser.add_argument("behavioral_csv", help="Path to behavioral data CSV")
    parser.add_argument("output_tsv", nargs="?", default=None,
                        help="Output beh TSV path (auto-generated if omitted)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.output_tsv is None:
        task = detect_task(args.behavioral_csv)
        subj, sess, run = parse_subj_sess_run(args.behavioral_csv)
        sub = bids_sub(subj)

        if task == "outscan_recognition":
            ses = bids_ses_cr(sess)
            fname = f"{sub}_{ses}_task-TB2AFC_run-{run:02d}_beh.tsv"
        elif task == "final_recognition":
            ses = bids_ses(FINAL_SESSION)
            fname = f"{sub}_{ses}_task-FIN2AFC_beh.tsv"
        elif task == "final_timeline":
            ses = bids_ses(FINAL_SESSION)
            fname = f"{sub}_{ses}_task-FINtimeline_beh.tsv"
        else:
            raise ValueError(f"Unknown task: {task}")

        output = os.path.join(BIDS_ROOT, sub, ses, "beh", fname)
    else:
        output = args.output_tsv

    print(f"Task: {detect_task(args.behavioral_csv)}")
    print(f"Input: {args.behavioral_csv}")
    print(f"Output: {output}")
    convert_file(args.behavioral_csv, output, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
