#!/usr/bin/env python3
"""Convert localizer timing CSVs into BIDS _events.tsv files.

Handles files with conversion_type='localizer_events': the auditory, motor,
fixation and tone localizers. Per-task run counts are not listed here -- query
the catalog.

Auditory and fixation are final-session files -> BIDS ses-30. Tone and motor
take their BIDS session from the **source path**, never from the CSV's
`sess_id`: tone was collected in the localizer sessions, and motor moved there
too under the regularised protocol, so `sess_id` means different things for
different cohorts while the path always means the same thing.

Auditory localizer format:
  Columns: sub_id, task_id, sess_id, run_id, trial_id, stim_start, stim_end,
           stim_fixation_start, stim_fixation_end
  Single trial with long auditory stimulus (~562s).

Motor localizer format:
  Columns: sub_id, task, onset, offset
  Block design, 20 s blocks, six conditions: hand, foot, mouth, speak,
  saccade, rest. The order is frozen by a hardcoded seed in the stimulus
  program -- identical in every run of every subject. The run entity comes
  from the BOLD, not from the CSV filename, whose `runN` is a per-subject
  counter (see motor_target).

Fixation format:
  Columns: event, onset_s, offset_s. Two rows: a `sync` marker at 0 with no
  offset, and a `movie` row spanning the whole run. The run is one continuous
  ~75 s eyetracking-calibration sequence with no recorded internal structure --
  target positions and timings are not in this file, and would have to come
  from the EyeLink record.

Tone format:
  Columns: sub_id, task_id, sess_id, run_id, tone_file, trial_id,
  stim_tone_start, stim_tone_end, stim_fixation_start, stim_fixation_end.
  15 trials of a 32 s cycle: a ~25.75 s pure-tone sweep then fixation to the
  boundary, 480 s total. The sweep direction is a session-level property --
  ses-02 is `pure_tones_low_to_high_filtered.wav` and ses-03 is
  `pure_tones_high_to_low_filtered.wav` for every subject -- so it is carried
  as a `direction` column.

  `stim_fixation_start` duplicates `stim_tone_start` in every row of every
  file, so it is ignored; the fixation period runs from `stim_tone_end` to
  `stim_fixation_end`, as it does for the auditory localizer.

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
    NA, BIDS_ROOT, FINAL_SESSION, SOURCE_DIR,
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
    if "tone" in fname:
        return "tone"
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


def ses_from_path(csv_path):
    """BIDS session number from the sourcedata path.

    Used for tone, whose CSV `sess_id` counts 1, 2 against BIDS ses-02, ses-03.
    The path is the authority; guessing from `sess_id` would be off by one.
    """
    m = re.search(r"/ses-(\d+)/", os.path.abspath(csv_path))
    if not m:
        raise ValueError(f"No /ses-NN/ component in: {csv_path}")
    return int(m.group(1))


def subj_from_path(csv_path):
    """BIDS subject number from the sourcedata path.

    The path is the authority. A subject number written into a source
    filename has been wrong before in this dataset, so it is only used to
    cross-check this one.
    """
    m = re.search(r"/sub-(\d+)/", os.path.abspath(csv_path))
    if not m:
        raise ValueError(f"No /sub-NN/ component in: {csv_path}")
    return int(m.group(1))


def run_duration_s(bold_path):
    """Acquisition length in seconds, or None when the BOLD is absent.

    Imported lazily so auditory/motor/fixation conversions do not require
    nibabel; only tone needs it, to truncate a run that was stopped early.
    """
    if not os.path.exists(bold_path):
        return None
    try:
        import nibabel as nb
    except ImportError:
        raise ImportError(
            "nibabel is needed to truncate tone events against the acquisition. "
            "Use the mmmdata-agents env, or pass an already-complete run."
        )
    img = nb.load(bold_path)
    return img.shape[3] * float(img.header.get_zooms()[3])


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


MOTOR_DESIGN_S = 600.0      # 30 blocks x 20 s, which exactly fills the run
MOTOR_TR_S = 1.5
# Ben's acceptance bar (2026-09-08): a margin under 500 ms is delta = 0. The
# measured runs sit at -0.25 to -0.39 s, so anything reaching this threshold is
# a new finding, not the known flip-latency overrun.
MOTOR_MARGIN_TOL_S = 0.5


def motor_target(csv_path):
    """Resolve (subj, ses, run_entity) for a motor CSV from the path + the BOLD.

    Two things a motor source filename cannot be trusted for:

    * `sessN` counts differently per cohort. The first cohort ran motor only
      in the final session and their CSVs all say `sess1`; from the
      regularised protocol onward `sessN` already is the BIDS session. The
      sourcedata path carries the BIDS session for both, so it is the
      authority.
    * `runN` is a per-subject counter, not a within-session index. Under the
      regularised protocol a session's only motor run can be named `run2`,
      and its BOLD carries no `run` entity at all.

    So the run entity is read off whichever BOLD exists, and an absent BOLD is
    an error naming both candidates rather than a guess.
    """
    subj_in_name, run_in_name = parse_subj_run(csv_path)
    subj = subj_from_path(csv_path)
    if subj != subj_in_name:
        raise ValueError(
            f"{csv_path}: the filename says sub{subj_in_name} but the path "
            f"says sub-{subj:02d}. The path wins; move or rename the file."
        )
    ses = ses_from_path(csv_path)

    stem = f"{bids_sub(subj)}_{bids_ses(ses)}_task-motor"
    func = os.path.join(BIDS_ROOT, bids_sub(subj), bids_ses(ses), "func")
    without_run = os.path.join(func, f"{stem}_bold.nii.gz")
    with_run = os.path.join(func, f"{stem}_run-{run_in_name:02d}_bold.nii.gz")

    if os.path.exists(without_run):
        return subj, ses, None
    if os.path.exists(with_run):
        return subj, ses, run_in_name
    raise FileNotFoundError(
        f"No motor BOLD for sub-{subj:02d} ses-{ses:02d}; looked for\n"
        f"  {without_run}\n  {with_run}\n"
        "The BOLD decides whether the events carry a run entity, so run this "
        "after BIDSification or pass an explicit output path."
    )


def motor_margin_s(subj, ses, run_entity, events_end_s):
    """`nvols x TR - events_end`: the check that would have caught fLoc.

    A whole-TR offset can only hide inside surplus acquisition. Motor has
    none -- 30 blocks x 20 s exactly fills a 400-volume run, and the measured
    span overruns it slightly through per-block flip latency -- so a surplus
    of a TR or more means the task clock does not start at the trigger and
    must be explained before events are written.

    Returns the margin in seconds, or None when the BOLD is absent.
    """
    stem = f"{bids_sub(subj)}_{bids_ses(ses)}_task-motor"
    if run_entity is not None:
        stem += f"_run-{run_entity:02d}"
    bold = os.path.join(BIDS_ROOT, bids_sub(subj), bids_ses(ses), "func",
                        f"{stem}_bold.nii.gz")
    acq = run_duration_s(bold)
    if acq is None:
        return None

    margin = acq - events_end_s
    if margin >= MOTOR_MARGIN_TOL_S:
        raise ValueError(
            f"{stem}: the acquisition ({acq:.3f} s) exceeds the events span "
            f"({events_end_s:.3f} s) by {margin:.3f} s, at or past the "
            f"{MOTOR_MARGIN_TOL_S:.1f} s acceptance bar. A surplus has to be "
            "explained by a named feature of the stimulus program before "
            "these events are trustworthy -- it is the shape of the fLoc "
            "countdown bug. Check the program for a lead-in, then set an "
            "explicit shift here."
        )
    if margin <= -MOTOR_TR_S:
        print(f"  WARNING: {stem} overruns its acquisition by "
              f"{-margin:.3f} s (more than one TR); the scan may have been "
              "stopped early, which motor has not seen before.")
    return margin


def convert_motor(csv_path, output_tsv, dry_run=False):
    """Convert motor localizer timing CSV -> BIDS events TSV.

    Subject and session come from the sourcedata path; the run entity comes
    from `output_tsv`, which `main` builds from the BOLD via `motor_target`.
    Onsets are the measured values verbatim -- the program resets its clock on
    the scanner sync with no lead-in, so the task clock is the scan clock.
    """
    subj_in_name, _ = parse_subj_run(csv_path)
    subj = subj_from_path(csv_path)
    if subj != subj_in_name:
        raise ValueError(
            f"{csv_path}: the filename says sub{subj_in_name} but the path "
            f"says sub-{subj:02d}. The path wins; move or rename the file."
        )
    ses = ses_from_path(csv_path)
    m = re.search(r"_run-(\d+)_", os.path.basename(output_tsv))
    run_entity = int(m.group(1)) if m else None

    df = pd.read_csv(csv_path)
    onset = df["onset"].astype(float)
    offset = df["offset"].astype(float)

    events = pd.DataFrame({
        "onset": onset,
        "duration": offset - onset,
        "subj_num": subj,
        "ses_num": ses,
        "run_idx": 1 if run_entity is None else run_entity,
        "trial_type": df["task"],
    })

    margin = motor_margin_s(subj, ses, run_entity, float(offset.max()))

    write_events_tsv(events, output_tsv, dry_run=dry_run)

    json_path = output_tsv.replace("_events.tsv", "_events.json")
    write_json_sidecar(motor_sidecar(csv_path, margin), json_path,
                       dry_run=dry_run)
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


TONE_DIRECTIONS = {
    "pure_tones_low_to_high_filtered.wav": "low_to_high",
    "pure_tones_high_to_low_filtered.wav": "high_to_low",
}


def convert_tone(csv_path, output_tsv, dry_run=False):
    """Convert a tonotopy localizer timing CSV -> BIDS events TSV.

    Two events per trial, matching the auditory localizer: the tone sweep, then
    the fixation period running to the 32 s cycle boundary.

    Events are truncated against the actual acquisition. sub-03's ses-03 run was
    stopped mid-scan (441.0 s of a 480.0 s design), so the last sweep was never
    acquired and the one before it was cut in half. Copying the CSV wholesale
    would put two sweeps' worth of regressors outside the data.
    """
    subj, _ = parse_subj_run(csv_path)
    ses = ses_from_path(csv_path)
    df = pd.read_csv(csv_path)

    tone_files = set(df["tone_file"].map(os.path.basename))
    if len(tone_files) != 1:
        raise ValueError(f"{csv_path}: expected one tone file, found {tone_files}")
    tone_file = tone_files.pop()
    if tone_file not in TONE_DIRECTIONS:
        raise ValueError(
            f"{csv_path}: unrecognised tone file {tone_file}. Add it to "
            f"TONE_DIRECTIONS with its sweep direction before converting."
        )
    direction = TONE_DIRECTIONS[tone_file]

    events_list = []
    for _, row in df.iterrows():
        tone_start = float(row["stim_tone_start"])
        tone_end = float(row["stim_tone_end"])
        fix_end = float(row["stim_fixation_end"])
        trial_id = int(row["trial_id"])
        for trial_type, onset, end in (
            ("tone", tone_start, tone_end),
            ("fixation", tone_end, fix_end),
        ):
            events_list.append({
                "onset": onset,
                "duration": end - onset,
                "subj_num": subj,
                "ses_num": ses,
                "run_idx": 1,
                "trial_type": trial_type,
                "trial_id": trial_id,
                "direction": direction,
            })

    events = pd.DataFrame(events_list)

    run_s = run_duration_s(output_tsv.replace("_events.tsv", "_bold.nii.gz"))
    truncated = 0
    if run_s is None:
        print(f"  WARNING: no BOLD beside {os.path.basename(output_tsv)} — "
              f"events not truncated against an acquisition")
    else:
        design_s = float((events["onset"] + events["duration"]).max())
        # A millisecond of overhang is the design landing on the last volume,
        # not a stopped scan. Clipping that would report a truncation on every
        # complete run.
        tol = 1e-3
        keep = events["onset"] < run_s - tol
        dropped = int((~keep).sum())
        events = events[keep].copy()
        over = events["onset"] + events["duration"] > run_s + tol
        truncated = int(over.sum())
        events.loc[over, "duration"] = run_s - events.loc[over, "onset"]
        if dropped or truncated:
            print(f"  Run is {run_s:.1f}s against a {design_s:.1f}s design: "
                  f"dropped {dropped} events past the end, clipped {truncated}")

    sidecar = dict(SIDECAR_TONE)
    sidecar["StimulusFile"] = tone_file
    if run_s is not None and truncated:
        sidecar["duration"] = dict(SIDECAR_TONE["duration"])
        sidecar["duration"]["Description"] += (
            f" This run was stopped early: events are truncated at the "
            f"{run_s:.1f} s acquisition, and {truncated} event(s) straddling "
            f"the end carry a clipped duration rather than the presented one."
        )

    write_events_tsv(events, output_tsv, dry_run=dry_run)
    json_path = output_tsv.replace("_events.tsv", "_events.json")
    write_json_sidecar(sidecar, json_path, dry_run=dry_run)
    return True


def convert_file(csv_path, output_tsv, dry_run=False):
    """Convert a localizer timing CSV to BIDS events TSV+JSON."""
    loc_type = detect_localizer_type(csv_path)
    if loc_type == "auditory":
        return convert_auditory(csv_path, output_tsv, dry_run)
    if loc_type == "fixation":
        return convert_fixation(csv_path, output_tsv, dry_run)
    if loc_type == "tone":
        return convert_tone(csv_path, output_tsv, dry_run)
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


def motor_sidecar(csv_path, margin_s):
    """SIDECAR_MOTOR plus this run's provenance and its timing check."""
    sc = dict(SIDECAR_MOTOR)
    sc["Description"] = (
        "Motor localizer adapted from Tang et al. (2023) / LeBel et al. "
        "(2023): 30 blocks of 20 s (600.0 s per run) over six conditions "
        "-- hand, foot, mouth, speak, saccade, rest -- five blocks each. "
        "The block order is frozen: the stimulus program shuffles the "
        "condition list under a hardcoded seed, so every run of every "
        "subject presents the identical sequence. Treat run-to-run and "
        "subject-to-subject order as fixed, not as a randomisation."
    )
    sc["StimulusPresentation"] = {
        "SoftwareName": "PsychoPy (hand-coded localizer_motor.py)",
    }
    sc["Sources"] = [os.path.relpath(csv_path, os.path.dirname(SOURCE_DIR))]
    sc["TimingOrigin"] = (
        "Onsets are the measured values from the source record, unshifted. "
        "The program waits on the scanner sync key and resets its clock "
        "immediately afterwards, with no countdown, lead-in or launchScan, "
        "so the task clock is the scan clock."
    )
    if margin_s is not None:
        sc["TimingVerification"] = (
            f"nvols x TR - events_end = {margin_s:+.3f} s, inside the "
            f"{MOTOR_MARGIN_TOL_S:.1f} s acceptance bar, so the events are "
            "taken to start at the first volume (delta = 0). The design fills "
            "the acquisition exactly, so there is no surplus for a whole-TR "
            "offset to hide in; the small overrun is per-block flip latency "
            "accumulated across the run."
        )
    return sc


SIDECAR_TONE = {
    "onset": {
        "Description": (
            "Event onset on the task clock. The design is 480.0 s against a "
            "480.0 s acquisition (320 volumes x 1.5 s), consistent with the "
            "program having started at the scanner trigger; unlike the fLoc "
            "localizer the source records no trigger pulses, so that "
            "alignment is corroborated by the durations rather than measured."
        ),
        "Units": "s",
    },
    "duration": {"Description": "Event duration.", "Units": "s"},
    "subj_num": {"Description": "Subject identifier number"},
    "ses_num": {
        "Description": (
            "BIDS session number, taken from the source path. The source CSV's "
            "own `sess_id` counts 1, 2 against BIDS ses-02, ses-03."
        ),
    },
    "run_idx": {
        "Description": (
            "Run number within the session. One tone run per session for "
            "sub-03 and sub-04, so always 1; the BIDS filename carries no run "
            "entity. The source CSV's `run_id` tracks the session number, not "
            "a within-session counter."
        ),
    },
    "trial_type": {
        "Description": "Type of event",
        "Levels": {
            "tone": "Pure-tone sweep (~25.75 s)",
            "fixation": "Fixation to the 32 s cycle boundary",
        },
    },
    "trial_id": {"Description": "Sequential trial number within the run (1-15)"},
    "direction": {
        "Description": (
            "Sweep direction of the tone stimulus, a session-level property: "
            "ses-02 is low-to-high and ses-03 high-to-low for every subject. "
            "Constant within a run, carried as a column so it survives "
            "concatenation across runs."
        ),
        "Levels": {
            "low_to_high": "pure_tones_low_to_high_filtered.wav",
            "high_to_low": "pure_tones_high_to_low_filtered.wav",
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
        elif loc_type == "tone":
            # One tone acquisition per session for sub-03/04, no run entity.
            ses = bids_ses(ses_from_path(args.timing_csv))
            fname = f"{sub}_{ses}_task-tone_events.tsv"
        elif loc_type == "fixation":
            # The BOLD carries no run entity, so the events must not either.
            fname = f"{sub}_{ses}_task-fixation_events.tsv"
        else:
            # The BOLD decides the run entity: the first cohort ran two motor
            # runs in one session, later subjects one per session with no
            # entity at all.
            subj, ses_num, run_entity = motor_target(args.timing_csv)
            sub, ses = bids_sub(subj), bids_ses(ses_num)
            run_part = "" if run_entity is None else f"_run-{run_entity:02d}"
            fname = f"{sub}_{ses}_task-motor{run_part}_events.tsv"

        output = os.path.join(BIDS_ROOT, sub, ses, "func", fname)
    else:
        output = args.output_tsv

    print(f"Type: {detect_localizer_type(args.timing_csv)} localizer")
    print(f"Input: {args.timing_csv}")
    print(f"Output: {output}")
    convert_file(args.timing_csv, output, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
