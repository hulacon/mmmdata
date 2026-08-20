#!/usr/bin/env python3
"""Generate file_inventory.csv from current sourcedata contents.

Walks sourcedata/{subject}/{session}/behavioral/ and eyetracking/ directories,
classifies each file, and maps it to a BIDS destination with the correct
prefixed task names (TBencoding, NATencoding, FINretrieval, etc.).

If edf_triage.csv exists alongside this script, EDF files are cross-referenced
against it: files with decision='exclude' are marked conversion_type='edf_excluded'
and their description updated with the reason.

Usage:
    python generate_inventory.py [--output file_inventory.csv]
"""

import argparse
import csv
import os
import re
from pathlib import Path

BIDS_ROOT = "/gpfs/projects/hulacon/shared/mmmdata"
SOURCE_ROOT = os.path.join(BIDS_ROOT, "sourcedata")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EDF_TRIAGE_CSV = os.path.join(SCRIPT_DIR, "edf_triage.csv")
PHYSIO_TRIAGE_CSV = os.path.join(SCRIPT_DIR, "physio_triage.csv")

SUBJECTS = [3, 4, 5]
CR_SESSION_OFFSET = 3   # cued recall: source sess N -> BIDS ses-(N+3)
FR_SESSION_OFFSET = 18  # free recall: source sess N -> BIDS ses-(N+18)
FINAL_SESSION = 30

# Cued recall BIDS sessions: ses-04 through ses-18 (source sess 1-15)
CR_BIDS_SESSIONS = range(4, 19)
# Free recall BIDS sessions: ses-19 through ses-28 (source sess 1-10)
FR_BIDS_SESSIONS = range(19, 29)


def bids_sub(num):
    return f"sub-{num:02d}"


def bids_ses(num):
    return f"ses-{num:02d}"


# ──────────────────────────────────────────────────────────
# Filename parsers
# ──────────────────────────────────────────────────────────

# Cued recall: cued_recall_{task}_subj{N}_sess{S}_run{R}[_timing].csv
RE_CUED = re.compile(
    r"cued_recall_(encoding|math|retrieval|recognition_outscan)"
    r"_subj(\d+)_sess(\d+)_run(\d+)(_timing)?\.csv$"
)

# Free recall math: free_recall_math_subj{N}_sess{S}_run{R}[_timing].csv
RE_FR_MATH = re.compile(
    r"free_recall_math_subj(\d+)_sess(\d+)_run(\d+)(_timing)?\.csv$"
)

# PsychoPy encoding: {subj}_{sess}_{run}_mem_search_recall_{datetime}.{ext}
RE_PSYCHOPY_ENC = re.compile(
    r"(\d+)_(\d+)_(\d+)_mem_search_recall_[\d\-_h.]+\.(csv|log|psydat)$"
)

# PsychoPy retrieval: {subj}_{sess}_free_recall_recall_{datetime}.{ext}
RE_PSYCHOPY_RET = re.compile(
    r"(\d+)_(\d+)_free_recall_recall_[\d\-_h.]+\.(csv|log|psydat)$"
)

# Final cued recall: final_cued_recall_subj{N}_sess{S}_run{R}[_timing].csv
RE_FINAL_CR = re.compile(
    r"final_cued_recall_subj(\d+)_sess(\d+)_run(\d+)(_timing)?\.csv$"
)

# Final recognition: final_recognition_subj{N}_sess{S}_run{R}.csv
RE_FINAL_RECOG = re.compile(
    r"final_recognition_subj(\d+)_sess(\d+)_run(\d+)\.csv$"
)

# Final timeline: final_timeline_subj{N}_sess{S}_run{R}.csv
RE_FINAL_TIMELINE = re.compile(
    r"final_timeline_subj(\d+)_sess(\d+)_run(\d+)\.csv$"
)

# Motor localizer: localizer_motor_sub{N}_sess{S}_run{R}_{datetime}_timing.csv
RE_MOTOR_LOC = re.compile(
    r"localizer_motor_sub(\d+)_sess(\d+)_run(\d+)_[\d_A-Za-z]+_timing\.csv$"
)

# Auditory localizer: localizer_auditory_subj{N}_sess{S}_run{R}_{datetime}_timing.csv
RE_AUDITORY_LOC = re.compile(
    r"localizer_auditory_subj(\d+)_sess(\d+)_run(\d+)_[\d_A-Za-z]+_timing\.csv$"
)

# Eyetracking calibration: eyetracking_calibration_subj{N}_sess{S}_run{R}_{datetime}_timing.csv
RE_ET_CALIB = re.compile(
    r"eyetracking_calibration_subj(\d+)_sess(\d+)_run(\d+)_[\d_A-Za-z]+_timing\.csv$"
)

# EDF files: s{subj}s{sess}r{run}{phase}_{date}_{time}.EDF
# phase: m = encoding (memory/movie), r = retrieval (recall)
RE_EDF = re.compile(
    r"s(\d+)s(\d+)r(\d+)([mr])_[\d_]+\.EDF$"
)

# Known EDF files with anomalous naming (manually mapped)
# Maps filename prefix → (subj, sess, run, phase)
EDF_ANOMALIES = {
    # sub-04/ses-22: "s4s4s1r" has extra 's'; actually encoding run 1
    "s4s4s1r_": (4, 4, 1, "m"),
    # sub-04/ses-24: "s4s6r1" missing phase letter; actually retrieval run 1
    "s4s6r1_2025": (4, 6, 1, "r"),
    # sub-03/ses-24: all 3 files labeled 'r' (retrieval) but BOLD acq times
    # and trigger counts prove the first two are encoding runs:
    #   s3s6r1r_12_05 → 772 triggers ≈ 775 vols NATencoding run-01 (acq 12:06)
    #   s3s6r2r_12_26 → 600 triggers ≈ 602 vols NATencoding run-02 (acq 12:27)
    #   s3s6r1r_12_53 → 1048 triggers ≈ 1054 vols NATretrieval (acq 12:57)
    "s3s6r1r_2025_05_01_12_05": (3, 6, 1, "m"),
    "s3s6r2r_2025": (3, 6, 2, "m"),
    "s3s6r1r_2025_05_01_12_53": (3, 6, 1, "r"),
}

# AOI files: TRIAL_{NNNN}_ROUTINE_{NN}.ias
RE_AOI = re.compile(r"TRIAL_\d+_ROUTINE_\d+\.ias$")


# ──────────────────────────────────────────────────────────
# Classification and BIDS mapping
# ──────────────────────────────────────────────────────────

def classify_cued_recall_file(filename, subj_num, bids_ses_num):
    """Classify a cued recall behavioral file."""
    m = RE_CUED.match(filename)
    if not m:
        return None
    task, subj_str, sess_str, run_str, is_timing = m.groups()
    run = int(run_str)
    sub = bids_sub(subj_num)
    ses = bids_ses(bids_ses_num)

    if task == "encoding":
        if is_timing:
            return {
                "description": f"Cued recall encoding timing, {sub} {ses} run {run}",
                "bids_destination": "n/a (timing input)",
                "conversion_type": "timing_input",
            }
        return {
            "description": f"Cued recall encoding behavioral, {sub} {ses} run {run}",
            "bids_destination": f"{sub}/{ses}/func/{sub}_{ses}_task-TBencoding_run-{run:02d}_events.tsv",
            "conversion_type": "timed_events",
        }
    elif task == "math":
        if is_timing:
            return {
                "description": f"Cued recall math timing, {sub} {ses}",
                "bids_destination": "n/a (timing input)",
                "conversion_type": "timing_input",
            }
        return {
            "description": f"Cued recall math behavioral, {sub} {ses}",
            "bids_destination": f"{sub}/{ses}/func/{sub}_{ses}_task-TBmath_events.tsv",
            "conversion_type": "timed_events",
        }
    elif task == "retrieval":
        if is_timing:
            return {
                "description": f"Cued recall retrieval timing, {sub} {ses} run {run}",
                "bids_destination": "n/a (timing input)",
                "conversion_type": "timing_input",
            }
        return {
            "description": f"Cued recall retrieval behavioral, {sub} {ses} run {run}",
            "bids_destination": f"{sub}/{ses}/func/{sub}_{ses}_task-TBretrieval_run-{run:02d}_events.tsv",
            "conversion_type": "timed_events",
        }
    elif task == "recognition_outscan":
        return {
            "description": f"Cued recall recognition (out-of-scanner), {sub} {ses} run {run}",
            "bids_destination": f"{sub}/{ses}/beh/{sub}_{ses}_task-TB2AFC_run-{run:02d}_beh.tsv",
            "conversion_type": "behavioral_to_beh",
        }
    return None


def classify_free_recall_behavioral(filename, subj_num, bids_ses_num):
    """Classify a free recall session behavioral file."""
    sub = bids_sub(subj_num)
    ses = bids_ses(bids_ses_num)

    # Free recall math
    m = RE_FR_MATH.match(filename)
    if m:
        subj_str, sess_str, run_str, is_timing = m.groups()
        if is_timing:
            return {
                "description": f"Free recall math timing, {sub} {ses}",
                "bids_destination": "n/a (timing input)",
                "conversion_type": "timing_input",
            }
        return {
            "description": f"Free recall math behavioral, {sub} {ses}",
            "bids_destination": f"{sub}/{ses}/func/{sub}_{ses}_task-NATmath_events.tsv",
            "conversion_type": "timed_events",
        }

    # PsychoPy encoding CSV
    m = RE_PSYCHOPY_ENC.match(filename)
    if m:
        subj_str, sess_str, run_str, ext = m.groups()
        run = int(run_str)
        if ext == "csv":
            return {
                "description": f"PsychoPy encoding data, {sub} {ses} run {run}",
                "bids_destination": f"{sub}/{ses}/func/{sub}_{ses}_task-NATencoding_run-{run:02d}_events.tsv",
                "conversion_type": "psychopy_encoding",
            }
        else:
            return {
                "description": f"PsychoPy encoding {ext} file, {sub} {ses} run {run}",
                "bids_destination": f"n/a (PsychoPy session file)",
                "conversion_type": "no_conversion",
            }

    # PsychoPy retrieval CSV
    m = RE_PSYCHOPY_RET.match(filename)
    if m:
        subj_str, sess_str, ext = m.groups()
        if ext == "csv":
            return {
                "description": f"PsychoPy retrieval data, {sub} {ses}",
                "bids_destination": f"{sub}/{ses}/func/{sub}_{ses}_task-NATretrieval_events.tsv",
                "conversion_type": "psychopy_retrieval",
            }
        else:
            return {
                "description": f"PsychoPy retrieval {ext} file, {sub} {ses}",
                "bids_destination": f"n/a (PsychoPy session file)",
                "conversion_type": "no_conversion",
            }

    return None


def classify_eyetracking_file(filename, subj_num, bids_ses_num, subdir=""):
    """Classify an eyetracking file (EDF, AOI, or audio)."""
    sub = bids_sub(subj_num)
    ses = bids_ses(bids_ses_num)

    # EDF files — check anomalies first, then standard regex
    edf_match = None
    for prefix, (a_subj, a_sess, a_run, a_phase) in EDF_ANOMALIES.items():
        if filename.startswith(prefix) and filename.endswith(".EDF"):
            edf_match = (a_subj, a_sess, a_run, a_phase)
            break

    if edf_match is None:
        m = RE_EDF.match(filename)
        if m:
            edf_match = (int(m.group(1)), int(m.group(2)),
                         int(m.group(3)), m.group(4))

    if edf_match:
        _, _, run, phase = edf_match
        if phase == "m":
            return {
                "description": f"Eyetracking encoding EDF, {sub} {ses} run {run}",
                "bids_destination": (
                    f"{sub}/{ses}/func/{sub}_{ses}_task-NATencoding"
                    f"_run-{run:02d}_recording-eye_physio.tsv.gz"
                ),
                "conversion_type": "edf_to_physio",
            }
        else:
            return {
                "description": f"Eyetracking retrieval EDF, {sub} {ses}",
                "bids_destination": (
                    f"{sub}/{ses}/func/{sub}_{ses}_task-NATretrieval"
                    f"_recording-eye_physio.tsv.gz"
                ),
                "conversion_type": "edf_to_physio",
            }

    # AOI files
    if RE_AOI.match(filename):
        return {
            "description": f"AOI definition, {sub} {ses}",
            "bids_destination": "n/a (supplementary AOI data)",
            "conversion_type": "supplementary",
        }

    # .EDF.tmp files (incomplete recordings)
    if filename.endswith(".EDF.tmp"):
        return {
            "description": f"Incomplete EDF recording, {sub} {ses}",
            "bids_destination": "n/a (incomplete recording)",
            "conversion_type": "no_conversion",
        }

    return None


def classify_final_session_file(filename, subj_num, subdir):
    """Classify a final session (ses-30) file."""
    sub = bids_sub(subj_num)
    ses = bids_ses(FINAL_SESSION)

    if subdir == "final_cued_recall":
        m = RE_FINAL_CR.match(filename)
        if m:
            subj_str, sess_str, run_str, is_timing = m.groups()
            run = int(run_str)
            if is_timing:
                return {
                    "description": f"Final cued recall timing, {sub} {ses} run {run}",
                    "bids_destination": "n/a (timing input)",
                    "conversion_type": "timing_input",
                }
            return {
                "description": f"Final cued recall behavioral, {sub} {ses} run {run}",
                "bids_destination": (
                    f"{sub}/{ses}/func/{sub}_{ses}_task-FINretrieval"
                    f"_run-{run:02d}_events.tsv"
                ),
                "conversion_type": "timed_events",
            }

    elif subdir == "final_cued_recall/timing":
        m = RE_FINAL_CR.match(filename)
        if m:
            subj_str, sess_str, run_str, is_timing = m.groups()
            run = int(run_str)
            return {
                "description": f"Final cued recall timing, {sub} {ses} run {run}",
                "bids_destination": "n/a (timing input)",
                "conversion_type": "timing_input",
            }

    elif subdir == "final_recognition":
        m = RE_FINAL_RECOG.match(filename)
        if m:
            return {
                "description": f"Final recognition behavioral, {sub} {ses}",
                "bids_destination": f"{sub}/{ses}/beh/{sub}_{ses}_task-FIN2AFC_beh.tsv",
                "conversion_type": "behavioral_to_beh",
            }

    elif subdir == "final_timeline_sequence":
        m = RE_FINAL_TIMELINE.match(filename)
        if m:
            return {
                "description": f"Final timeline behavioral, {sub} {ses}",
                "bids_destination": f"{sub}/{ses}/beh/{sub}_{ses}_task-FINtimeline_beh.tsv",
                "conversion_type": "behavioral_to_beh",
            }

    elif subdir == "motor_localizer":
        m = RE_MOTOR_LOC.match(filename)
        if m:
            subj_str, sess_str, run_str = m.group(1), m.group(2), m.group(3)
            run = int(run_str)
            return {
                "description": f"Motor localizer timing, {sub} {ses} run {run}",
                "bids_destination": (
                    f"{sub}/{ses}/func/{sub}_{ses}_task-motor"
                    f"_run-{run:02d}_events.tsv"
                ),
                "conversion_type": "localizer_events",
            }

    elif subdir == "auditory_localizer":
        m = RE_AUDITORY_LOC.match(filename)
        if m:
            return {
                "description": f"Auditory localizer timing, {sub} {ses}",
                "bids_destination": (
                    f"{sub}/{ses}/func/{sub}_{ses}_task-auditory_events.tsv"
                ),
                "conversion_type": "localizer_events",
            }

    elif subdir == "eyetracking":
        m = RE_ET_CALIB.match(filename)
        if m:
            return {
                "description": f"Eyetracking calibration timing, {sub} {ses}",
                "bids_destination": "n/a (calibration data)",
                "conversion_type": "supplementary",
            }

    return None


def determine_bids_session(ses_num):
    """Determine session type from BIDS session number."""
    if 4 <= ses_num <= 18:
        return "cued_recall"
    elif 19 <= ses_num <= 28:
        return "free_recall"
    elif ses_num == 30:
        return "final"
    else:
        return "other"


def walk_subject(subj_num):
    """Walk all sessions for a subject and classify files."""
    rows = []
    sub_dir = os.path.join(SOURCE_ROOT, bids_sub(subj_num))

    for ses_num in list(range(4, 19)) + list(range(19, 29)) + [30]:
        ses_dir = os.path.join(sub_dir, bids_ses(ses_num))
        if not os.path.isdir(ses_dir):
            continue

        session_type = determine_bids_session(ses_num)

        # ── Behavioral directory ──
        beh_dir = os.path.join(ses_dir, "behavioral")
        if os.path.isdir(beh_dir):
            if session_type == "final":
                # Final session has subdirectories
                for subdir in sorted(os.listdir(beh_dir)):
                    subdir_path = os.path.join(beh_dir, subdir)
                    if os.path.isdir(subdir_path):
                        for fname in sorted(os.listdir(subdir_path)):
                            fpath = os.path.join(subdir_path, fname)
                            if os.path.isfile(fpath):
                                rel_path = os.path.relpath(fpath, SOURCE_ROOT)
                                info = classify_final_session_file(
                                    fname, subj_num, subdir
                                )
                                # Check for timing subdirectory
                                if info is None and subdir == "final_cued_recall":
                                    # Could be the timing/ subdir
                                    if os.path.isdir(fpath):
                                        continue
                                if info is None:
                                    info = {
                                        "description": f"Unclassified final session file: {fname}",
                                        "bids_destination": "n/a (unclassified)",
                                        "conversion_type": "no_conversion",
                                    }
                                rows.append({
                                    "source_file": rel_path,
                                    **info,
                                })
                        # Check for nested subdirectories (e.g., final_cued_recall/timing/)
                        for nested in sorted(os.listdir(subdir_path)):
                            nested_path = os.path.join(subdir_path, nested)
                            if os.path.isdir(nested_path):
                                for fname in sorted(os.listdir(nested_path)):
                                    fpath = os.path.join(nested_path, fname)
                                    if os.path.isfile(fpath):
                                        rel_path = os.path.relpath(
                                            fpath, SOURCE_ROOT
                                        )
                                        nested_subdir = f"{subdir}/{nested}"
                                        info = classify_final_session_file(
                                            fname, subj_num, nested_subdir
                                        )
                                        if info is None:
                                            info = {
                                                "description": f"Unclassified: {fname}",
                                                "bids_destination": "n/a (unclassified)",
                                                "conversion_type": "no_conversion",
                                            }
                                        rows.append({
                                            "source_file": rel_path,
                                            **info,
                                        })
                    elif os.path.isfile(subdir_path):
                        # Files directly in behavioral/ for ses-30
                        # (shouldn't normally happen, but handle gracefully)
                        rel_path = os.path.relpath(subdir_path, SOURCE_ROOT)
                        rows.append({
                            "source_file": rel_path,
                            "description": f"Unclassified ses-30 file: {subdir}",
                            "bids_destination": "n/a (unclassified)",
                            "conversion_type": "no_conversion",
                        })
            else:
                # Cued recall or free recall — flat directory of files
                for fname in sorted(os.listdir(beh_dir)):
                    fpath = os.path.join(beh_dir, fname)
                    if not os.path.isfile(fpath):
                        continue
                    rel_path = os.path.relpath(fpath, SOURCE_ROOT)

                    if session_type == "cued_recall":
                        info = classify_cued_recall_file(
                            fname, subj_num, ses_num
                        )
                    elif session_type == "free_recall":
                        info = classify_free_recall_behavioral(
                            fname, subj_num, ses_num
                        )
                    else:
                        info = None

                    if info is None:
                        info = {
                            "description": f"Unclassified behavioral file: {fname}",
                            "bids_destination": "n/a (unclassified)",
                            "conversion_type": "no_conversion",
                        }
                    rows.append({"source_file": rel_path, **info})

        # ── Eyetracking directory ──
        et_dir = os.path.join(ses_dir, "eyetracking")
        if os.path.isdir(et_dir):
            for fname in sorted(os.listdir(et_dir)):
                fpath = os.path.join(et_dir, fname)
                if os.path.isdir(fpath):
                    # AOI subdirectory
                    if fname == "aoi":
                        for aoi_file in sorted(os.listdir(fpath)):
                            aoi_path = os.path.join(fpath, aoi_file)
                            if os.path.isfile(aoi_path):
                                rel_path = os.path.relpath(
                                    aoi_path, SOURCE_ROOT
                                )
                                info = classify_eyetracking_file(
                                    aoi_file, subj_num, ses_num, "aoi"
                                )
                                if info is None:
                                    info = {
                                        "description": f"Unclassified AOI file: {aoi_file}",
                                        "bids_destination": "n/a (unclassified)",
                                        "conversion_type": "no_conversion",
                                    }
                                rows.append({"source_file": rel_path, **info})
                    continue

                if not os.path.isfile(fpath):
                    continue
                rel_path = os.path.relpath(fpath, SOURCE_ROOT)
                info = classify_eyetracking_file(fname, subj_num, ses_num)
                if info is None:
                    info = {
                        "description": f"Unclassified eyetracking file: {fname}",
                        "bids_destination": "n/a (unclassified)",
                        "conversion_type": "no_conversion",
                    }
                rows.append({"source_file": rel_path, **info})

    return rows


def load_edf_triage():
    """Load edf_triage.csv into a dict keyed by source_file path.

    Returns empty dict if the file doesn't exist.
    """
    if not os.path.isfile(EDF_TRIAGE_CSV):
        return {}
    triage = {}
    with open(EDF_TRIAGE_CSV, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            triage[row["source_file"]] = row
    return triage


def apply_edf_triage(rows, triage):
    """Apply EDF triage decisions to inventory rows.

    - Excluded files: conversion_type -> 'edf_excluded', description updated
    - Included pupil_only files: description notes gaze unavailable
    """
    if not triage:
        return rows
    for row in rows:
        if row.get("conversion_type") != "edf_to_physio":
            continue
        src = row["source_file"]
        tri = triage.get(src)
        if tri is None:
            continue
        if tri["decision"] == "exclude":
            gaze_pct = tri["gaze_valid_pct"]
            pupil_pct = tri["pupil_valid_pct"]
            row["conversion_type"] = "edf_excluded"
            row["description"] += (
                f" [EXCLUDED: gaze {gaze_pct}% pupil {pupil_pct}% valid in scan window]"
            )
        elif tri["channels"] == "pupil_only":
            gaze_pct = tri["gaze_valid_pct"]
            row["description"] += f" [pupil_only: gaze {gaze_pct}% valid]"
    return rows


# ──────────────────────────────────────────────────────────
# PhysioLog (scanner physio) inventory from physio_triage.csv
# ──────────────────────────────────────────────────────────

# Series name (with Series_NN_ prefix and _PhysioLog suffix stripped)
# -> (bids_task, run_number_or_None)
RE_PHYSIO_SERIES = re.compile(
    r"Series_\d+_(.+?)_PhysioLog$"
)

# Task patterns: order matters (most specific first)
PHYSIO_TASK_MAP = [
    (re.compile(r"cued_recall_encoding_run(\d+)$"), "TBencoding", True),
    (re.compile(r"cued_recall_retrieval_run(\d+)$"), "TBretrieval", True),
    (re.compile(r"cued_recall_math$"), "TBmath", False),
    (re.compile(r"cued_recall_resting$"), "TBresting", False),
    (re.compile(r"free_recall_encoding_run(\d+)$"), "NATencoding", True),
    (re.compile(r"free_recall_retrieval_run(\d+)(_attempt\d+)?$"), "NATretrieval", True),
    (re.compile(r"free_recall_math$"), "NATmath", False),
    (re.compile(r"free_recall_resting$"), "NATresting", False),
    (re.compile(r"final_cued_recall_run(\d+)$"), "FINretrieval", True),
    (re.compile(r"Resting_baseline$"), "INITresting", False),
    (re.compile(r"Resting$"), "FINresting", False),
    (re.compile(r"fixation_calibration$"), "fixation", False),
    (re.compile(r"localizer_prf_run(\d+)$"), "prf", True),
    (re.compile(r"localizer_floc_run(\d+)$"), "floc", True),
    (re.compile(r"localizer_tone(?:_run(\d+))?$"), "tone", True),
    (re.compile(r"localizer_auditory(?:_run(\d+))?$"), "auditory", True),
    (re.compile(r"localizer_motor_run(\d+)$"), "motor", True),
]


def _physio_series_to_bids(series_name):
    """Map a PhysioLog series name to (bids_task, run_number_or_None).

    Returns None if the series can't be mapped (e.g. fixation_calibration
    which may be skipped).
    """
    m = RE_PHYSIO_SERIES.match(series_name)
    if not m:
        return None
    task_part = m.group(1)

    for pattern, bids_task, has_run in PHYSIO_TASK_MAP:
        pm = pattern.match(task_part)
        if pm:
            run = None
            if has_run and pm.lastindex and pm.group(1):
                run = int(pm.group(1))
            return bids_task, run
    return None


def load_physio_triage():
    """Load physio_triage.csv and generate inventory rows for convertible files.

    Only COMPLETE and PARTIAL files are included.

    Returns
    -------
    list of dict
        Inventory rows with source_file, description, bids_destination,
        conversion_type='physio_dcm'.
    """
    if not os.path.isfile(PHYSIO_TRIAGE_CSV):
        return []

    rows = []
    with open(PHYSIO_TRIAGE_CSV, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            status = row["status"]
            if status not in ("COMPLETE", "PARTIAL"):
                continue

            sub = row["sub"]
            ses = row["ses"]
            series = row["series"]
            source_path = row["source_path"]

            mapping = _physio_series_to_bids(series)
            if mapping is None:
                continue

            bids_task, run = mapping

            # Build BIDS base path (without _recording-*_physio suffix)
            if run is not None:
                bids_base = (
                    f"{sub}/{ses}/func/{sub}_{ses}_task-{bids_task}"
                    f"_run-{run:02d}"
                )
            else:
                bids_base = f"{sub}/{ses}/func/{sub}_{ses}_task-{bids_task}"

            rows.append({
                "source_file": source_path,
                "description": (
                    f"Scanner physio ({status.lower()}), {sub} {ses} "
                    f"task-{bids_task}"
                    + (f" run-{run:02d}" if run else "")
                ),
                "bids_destination": bids_base,
                "conversion_type": "physio_dcm",
            })

    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Generate file_inventory.csv from sourcedata"
    )
    parser.add_argument(
        "--output", "-o",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "file_inventory.csv"),
        help="Output CSV path (default: file_inventory.csv in script directory)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print progress info",
    )
    args = parser.parse_args()

    all_rows = []
    for subj_num in SUBJECTS:
        if args.verbose:
            print(f"Processing {bids_sub(subj_num)}...")
        rows = walk_subject(subj_num)
        all_rows.extend(rows)
        if args.verbose:
            print(f"  Found {len(rows)} files")

    # Apply EDF triage decisions
    triage = load_edf_triage()
    if triage:
        all_rows = apply_edf_triage(all_rows, triage)
        n_excluded = sum(1 for r in all_rows if r["conversion_type"] == "edf_excluded")
        n_edf = sum(1 for r in all_rows if r["conversion_type"] in ("edf_to_physio", "edf_excluded"))
        print(f"EDF triage applied: {n_excluded}/{n_edf} files excluded")
    else:
        print(f"WARNING: {EDF_TRIAGE_CSV} not found, skipping EDF triage")

    # Add scanner physio rows from physio_triage.csv
    physio_rows = load_physio_triage()
    if physio_rows:
        all_rows.extend(physio_rows)
        print(f"Scanner physio: {len(physio_rows)} convertible PhysioLog files added")
    else:
        print(f"WARNING: {PHYSIO_TRIAGE_CSV} not found or empty, skipping scanner physio")

    # Write CSV
    fieldnames = ["source_file", "description", "bids_destination", "conversion_type"]
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Wrote {len(all_rows)} rows to {args.output}")

    # Summary by conversion type
    from collections import Counter
    counts = Counter(r["conversion_type"] for r in all_rows)
    print("\nBy conversion_type:")
    for ct, n in sorted(counts.items()):
        print(f"  {ct}: {n}")

    # Check for unclassified
    unclassified = [r for r in all_rows if "unclassified" in r["description"].lower()]
    if unclassified:
        print(f"\nWARNING: {len(unclassified)} unclassified files:")
        for r in unclassified:
            print(f"  {r['source_file']}")


if __name__ == "__main__":
    main()
