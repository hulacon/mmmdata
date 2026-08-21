#!/usr/bin/env python3
"""Convert localizer timing CSVs into BIDS _events.tsv files.

Handles files with conversion_type='localizer_events' (12 files total):
  - Auditory localizer (3 files, 1 per subject)
  - Motor localizer (6 files, 2 runs per subject)
  - Fixation / eyetracking calibration (3 files, 1 per subject)

All are final session files -> BIDS ses-30.

Auditory localizer format:
  Columns: sub_id, task_id, sess_id, run_id, trial_id, stim_start, stim_end,
           stim_fixation_start, stim_fixation_end
  Single trial with long auditory stimulus (~562s).

Motor localizer format:
  Columns: sub_id, task, onset, offset
  Block design with conditions: foot, mouth, saccade, hand, rest (20s blocks).

Fixation format:
  Columns: event, onset_s, offset_s. Two rows: a `sync` marker at 0 with no
  offset, and a `movie` row spanning the whole run. The run is one continuous
  ~75 s eyetracking-calibration sequence with no recorded internal structure --
  target positions and timings are not in this file, and would have to come
  from the EyeLink record.

  Note this replaces `mmmsourcedata/shared/conversion/eventfiles/
  events_FINfixation.py`, which was written but never run. That script could
  not be used as-is: it emits `onset_s`/`offset_s` rather than the `onset`/
  `duration` BIDS requires, and names its output with a `run-01` entity the
  BOLD does not carry, so the result would not have paired with any scan.

Usage:
    python localizer_events.py <timing_csv> [<output_events_tsv>] [--dry-run]
"""

import argparse
import os
import re
import sys

import pandas as pd

from common import (
    NA, BIDS_ROOT, FINAL_SESSION,
    bids_sub, bids_ses, float_or_na,
    write_events_tsv, write_json_sidecar,
)


def detect_localizer_type(csv_path):
    """Detect whether this is an auditory or motor localizer."""
    fname = os.path.basename(csv_path)
    if "auditory" in fname:
        return "auditory"
    if "motor" in fname:
        return "motor"
    if "calibration" in fname:
        return "fixation"
    raise ValueError(f"Cannot detect localizer type from: {fname}")


def parse_subj_run(csv_path):
    """Extract subject and run numbers from localizer filename."""
    fname = os.path.basename(csv_path)
    # auditory: localizer_auditory_subj3_sess1_run1_2025_Aug_15_1233_timing.csv
    # motor: localizer_motor_sub3_sess1_run1_2025_Aug_15_1201_timing.csv
    m = re.search(r"sub[j]?(\d+)_sess(\d+)_run(\d+)", fname)
    if not m:
        raise ValueError(f"Cannot parse subject/run from: {fname}")
    return int(m.group(1)), int(m.group(3))


def convert_auditory(csv_path, output_tsv, dry_run=False):
    """Convert auditory localizer timing CSV -> BIDS events TSV.

    Source has one row per trial with stim_start/stim_end and
    stim_fixation_start/stim_fixation_end. Output has two events:
    stimulus (auditory presentation) and fixation (post-stimulus).
    """
    subj, run = parse_subj_run(csv_path)
    df = pd.read_csv(csv_path)

    events_list = []
    for _, row in df.iterrows():
        stim_start = float(row["stim_start"])
        stim_end = float(row["stim_end"])
        fix_end = float(row["stim_fixation_end"])
        trial_id = int(row["trial_id"])

        events_list.append({
            "onset": stim_start,
            "duration": stim_end - stim_start,
            "subj_num": subj,
            "ses_num": FINAL_SESSION,
            "run_idx": run,
            "trial_type": "stimulus",
            "trial_id": trial_id,
        })
        events_list.append({
            "onset": stim_end,
            "duration": fix_end - stim_end,
            "subj_num": subj,
            "ses_num": FINAL_SESSION,
            "run_idx": run,
            "trial_type": "fixation",
            "trial_id": trial_id,
        })

    events = pd.DataFrame(events_list)
    write_events_tsv(events, output_tsv, dry_run=dry_run)

    json_path = output_tsv.replace("_events.tsv", "_events.json")
    write_json_sidecar(SIDECAR_AUDITORY, json_path, dry_run=dry_run)
    return True


def convert_motor(csv_path, output_tsv, dry_run=False):
    """Convert motor localizer timing CSV -> BIDS events TSV."""
    subj, run = parse_subj_run(csv_path)
    df = pd.read_csv(csv_path)

    events = pd.DataFrame({
        "onset": df["onset"].astype(float),
        "duration": df["offset"].astype(float) - df["onset"].astype(float),
        "subj_num": subj,
        "ses_num": FINAL_SESSION,
        "run_idx": run,
        "trial_type": df["task"],
    })

    write_events_tsv(events, output_tsv, dry_run=dry_run)

    json_path = output_tsv.replace("_events.tsv", "_events.json")
    write_json_sidecar(SIDECAR_MOTOR, json_path, dry_run=dry_run)
    return True


def convert_fixation(csv_path, output_tsv, dry_run=False):
    """Convert an eyetracking-calibration timing CSV -> BIDS events TSV.

    The source holds a `sync` marker and a single `movie` row covering the
    whole run. Only the calibration block becomes an event: the sync marker
    sits at onset 0 with no offset, so as a row it would be a zero-duration
    regressor at the run origin carrying no information the origin does not
    already carry. That it reads 0.000 is recorded in the sidecar instead,
    because it is the evidence the clock starts at the scanner sync.
    """
    subj, _ = parse_subj_run(csv_path)
    df = pd.read_csv(csv_path)

    block = df[df["event"] == "movie"]
    if len(block) != 1:
        raise ValueError(
            f"{csv_path}: expected exactly one 'movie' row, found {len(block)}"
        )
    row = block.iloc[0]
    onset, offset = float(row["onset_s"]), float(row["offset_s"])

    sync = df[df["event"] == "sync"]
    if len(sync) != 1 or float(sync.iloc[0]["onset_s"]) != 0.0:
        raise ValueError(f"{csv_path}: expected one sync marker at 0.0")

    events = pd.DataFrame([{
        "onset": onset,
        "duration": offset - onset,
        "subj_num": subj,
        "ses_num": FINAL_SESSION,
        "run_idx": 1,
        "trial_type": "calibration",
    }])

    write_events_tsv(events, output_tsv, dry_run=dry_run)
    json_path = output_tsv.replace("_events.tsv", "_events.json")
    write_json_sidecar(SIDECAR_FIXATION, json_path, dry_run=dry_run)
    return True


def convert_file(csv_path, output_tsv, dry_run=False):
    """Convert a localizer timing CSV to BIDS events TSV+JSON."""
    loc_type = detect_localizer_type(csv_path)
    if loc_type == "auditory":
        return convert_auditory(csv_path, output_tsv, dry_run)
    if loc_type == "fixation":
        return convert_fixation(csv_path, output_tsv, dry_run)
    return convert_motor(csv_path, output_tsv, dry_run)


SIDECAR_AUDITORY = {
    "onset": {"Description": "Event onset time relative to scanner start", "Units": "s"},
    "duration": {"Description": "Event duration", "Units": "s"},
    "subj_num": {"Description": "Subject identifier number"},
    "ses_num": {"Description": "BIDS session number"},
    "run_idx": {"Description": "Run number within the session"},
    "trial_type": {
        "Description": "Type of event",
        "Levels": {
            "stimulus": "Auditory localizer stimulus presentation",
            "fixation": "Post-stimulus fixation period",
        },
    },
    "trial_id": {"Description": "Sequential trial number within the run"},
}

SIDECAR_MOTOR = {
    "onset": {"Description": "Block onset time relative to scanner start", "Units": "s"},
    "duration": {"Description": "Block duration", "Units": "s"},
    "subj_num": {"Description": "Subject identifier number"},
    "ses_num": {"Description": "BIDS session number"},
    "run_idx": {"Description": "Run number within the session"},
    "trial_type": {
        "Description": "Motor task condition",
        "Levels": {
            "foot": "Foot movement block",
            "mouth": "Mouth movement block",
            "saccade": "Saccade (eye movement) block",
            "hand": "Hand movement block",
            "speak": "Speech production block",
            "rest": "Rest block (fixation)",
        },
    },
}


SIDECAR_FIXATION = {
    "onset": {
        "Description": (
            "Block onset relative to scanner start. The source records a sync "
            "marker at 0.000 s, so the task clock starts at the scanner sync "
            "and no shift is applied."
        ),
        "Units": "s",
    },
    "duration": {
        "Description": (
            "Block duration. The calibration sequence runs marginally past "
            "the 75.0 s acquisition (50 volumes x 1.5 s); the recorded value "
            "is kept rather than clipped, since it is what was presented."
        ),
        "Units": "s",
    },
    "subj_num": {"Description": "Subject identifier number"},
    "ses_num": {"Description": "BIDS session number"},
    "run_idx": {"Description": "Run number within the session"},
    "trial_type": {
        "Description": (
            "Type of event. The run is a single continuous block: the source "
            "records no internal structure, so individual calibration target "
            "positions and timings are not represented here and would have to "
            "come from the EyeLink record."
        ),
        "Levels": {"calibration": "Eyetracking calibration sequence"},
    },
}


def main():
    parser = argparse.ArgumentParser(
        description="Convert localizer timing CSV to BIDS events TSV"
    )
    parser.add_argument("timing_csv", help="Path to localizer timing CSV")
    parser.add_argument("output_tsv", nargs="?", default=None,
                        help="Output events TSV path (auto-generated if omitted)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.output_tsv is None:
        subj, run = parse_subj_run(args.timing_csv)
        loc_type = detect_localizer_type(args.timing_csv)
        sub = bids_sub(subj)
        ses = bids_ses(FINAL_SESSION)

        if loc_type == "auditory":
            fname = f"{sub}_{ses}_task-auditory_events.tsv"
        elif loc_type == "fixation":
            # The BOLD carries no run entity, so the events must not either.
            fname = f"{sub}_{ses}_task-fixation_events.tsv"
        else:
            fname = f"{sub}_{ses}_task-motor_run-{run:02d}_events.tsv"

        output = os.path.join(BIDS_ROOT, sub, ses, "func", fname)
    else:
        output = args.output_tsv

    print(f"Type: {detect_localizer_type(args.timing_csv)} localizer")
    print(f"Input: {args.timing_csv}")
    print(f"Output: {output}")
    convert_file(args.timing_csv, output, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
