#!/usr/bin/env python3
"""Convert EyeLink EDF files into BIDS _recording-eye_physio.tsv.gz files.

Handles files with conversion_type='edf_to_physio' from the inventory.
These are EyeLink eyetracking recordings from the free recall sessions
(encoding and retrieval), BIDS ses-19 to ses-28.

Requires: eyelinkio (pip install eyelinkio)

EDF naming: s{Subj}s{Sess}r{Run}{m|r}_YYYY_MM_DD_HH_MM.EDF
  m = encoding (memory), r = recall (retrieval)

Scanner triggers: Input value 255 in the EDF discrete event stream marks
each BOLD volume onset (every 1.5s = TR). These define the scan window
and provide StartTime alignment.

Output format:
  - _recording-eye_physio.tsv.gz: continuous eye position + pupil samples
    trimmed to the scan window (first trigger to last trigger).
    Always 3 columns (eye1_x_coordinate, eye1_y_coordinate, eye1_pupil_size),
    with gaze columns as n/a when calibration was lost.
    No header, tab-separated, gzipped.
  - _recording-eye_physio.json: sampling rate, start time relative to first
    BOLD volume, column names.

Usage:
    python edf_to_physio.py <edf_file> [<output_physio_tsv_gz>] [--dry-run]
"""

import argparse
import gzip
import json
import os
import re
import sys

import numpy as np

from common import (
    BIDS_ROOT, FR_SESSION_OFFSET,
    bids_sub, bids_ses_fr,
    write_json_sidecar,
)

# Trigger value for scanner volume onset in EyeLink input channel
TRIGGER_VALUE = 255


def parse_edf_filename(edf_path):
    """Extract subject, session, run, phase from EDF filename.

    Returns (subj, sess, run, phase) where phase is 'encoding' or 'retrieval'.
    """
    fname = os.path.basename(edf_path)

    # Standard: s3s1r1m_2025_04_01_12_39.EDF
    m = re.match(r"s(\d+)s(\d+)r(\d+)([mr])_\d{4}_\d{2}_\d{2}_\d{2}_\d{2}\.EDF$", fname)
    if m:
        subj = int(m.group(1))
        sess = int(m.group(2))
        run = int(m.group(3))
        phase = "encoding" if m.group(4) == "m" else "retrieval"
        return subj, sess, run, phase

    # Missing suffix: s4s6r1_2025_05_14_14_15.EDF (infer from directory)
    m = re.match(r"s(\d+)s(\d+)r(\d+)_\d{4}_\d{2}_\d{2}_\d{2}_\d{2}\.EDF$", fname)
    if m:
        subj = int(m.group(1))
        sess = int(m.group(2))
        run = int(m.group(3))
        phase = "encoding" if "/Encoding/" in edf_path or "/encoding/" in edf_path else "retrieval"
        return subj, sess, run, phase

    # Extra 's': s4s4s1r_2025_04_30_12_38.EDF
    m = re.match(r"s(\d+)s(\d+)s(\d+)[mr]?_\d{4}_\d{2}_\d{2}_\d{2}_\d{2}\.EDF$", fname)
    if m:
        subj = int(m.group(1))
        sess = int(m.group(2))
        run = int(m.group(3))
        phase = "encoding" if "/Encoding/" in edf_path or "/encoding/" in edf_path else "retrieval"
        return subj, sess, run, phase

    raise ValueError(f"Cannot parse EDF filename: {fname}")


def extract_triggers(edf):
    """Extract scanner trigger times (seconds) from EDF discrete events.

    Returns array of trigger onset times (input value 255).
    """
    disc = edf.get("discrete", {})
    inputs = disc.get("inputs", np.array([]))
    if len(inputs) == 0:
        return np.array([])
    inp_times = np.array([x[0] for x in inputs])
    inp_vals = np.array([x[1] for x in inputs])
    return inp_times[inp_vals == TRIGGER_VALUE]


def convert_file(edf_path, output_tsv_gz, dry_run=False):
    """Convert an EDF file to BIDS physio TSV.GZ + JSON sidecar.

    The output is trimmed to the scan window defined by scanner triggers.
    StartTime is set to 0.0 (first sample = first trigger = first BOLD volume).
    Always outputs 3 columns: eye1_x_coordinate, eye1_y_coordinate,
    eye1_pupil_size. Missing gaze (isnan) and missing pupil (==0) are
    written as 'n/a'.

    Parameters
    ----------
    edf_path : str
        Path to the EDF file.
    output_tsv_gz : str
        Destination path for the physio TSV.GZ file.
    dry_run : bool
        If True, print what would be done without writing files.

    Returns
    -------
    bool
        True if conversion succeeded.
    """
    subj, sess, run, phase = parse_edf_filename(edf_path)

    if dry_run:
        print(f"  [dry-run] Would convert EDF -> {output_tsv_gz}")
        json_path = output_tsv_gz.replace("_physio.tsv.gz", "_physio.json")
        print(f"  [dry-run] Would write JSON -> {json_path}")
        return True

    try:
        from eyelinkio import read_edf
    except ImportError:
        print("ERROR: eyelinkio not installed. Run: pip install eyelinkio",
              file=sys.stderr)
        return False

    # Read the EDF file
    edf = read_edf(edf_path)

    # Extract sample data
    # eyelinkio 0.3 returns samples as ndarray (n_fields, n_samples) with
    # field names in info["sample_fields"]. Earlier versions used a dict.
    raw_samples = edf["samples"]
    sfreq = edf["info"]["sfreq"]
    sample_fields = edf["info"].get("sample_fields", [])

    # Normalise to a dict keyed by field name
    if isinstance(raw_samples, np.ndarray):
        samples = {name: raw_samples[i] for i, name in enumerate(sample_fields)}
    elif isinstance(raw_samples, dict):
        samples = raw_samples
    else:
        print(f"  WARNING: Unexpected samples type {type(raw_samples)} in {edf_path}",
              file=sys.stderr)
        return False

    # Determine which eye(s) were recorded
    eye_info = edf["info"].get("eye", None)

    xpos = samples.get("xpos")
    ypos = samples.get("ypos")
    ps = samples.get("ps", samples.get("pupil_size"))

    if xpos is None or ypos is None or ps is None:
        print(f"  WARNING: Missing sample channels in {edf_path}", file=sys.stderr)
        return False

    n_total = len(xpos)

    # --- Extract triggers and define scan window ---
    triggers = extract_triggers(edf)
    if len(triggers) < 2:
        print(f"  WARNING: Only {len(triggers)} triggers in {edf_path}, "
              "cannot define scan window", file=sys.stderr)
        return False

    first_trig_s = triggers[0]
    last_trig_s = triggers[-1]
    first_i = max(0, min(int(first_trig_s * sfreq), n_total - 1))
    last_i = max(first_i, min(int(last_trig_s * sfreq), n_total - 1))

    # Trim to scan window
    xpos = xpos[first_i : last_i + 1]
    ypos = ypos[first_i : last_i + 1]
    ps = ps[first_i : last_i + 1]
    n_samples = len(xpos)

    # --- Build column data (always 3 columns) ---
    col_names = ["eye1_x_coordinate", "eye1_y_coordinate", "eye1_pupil_size"]
    columns = [xpos, ypos, ps]
    data = np.column_stack(columns)

    # --- Missing data detection ---
    # Gaze missing: isnan (EyeLink reports NaN when gaze can't be computed)
    # Pupil missing: == 0 (pupil not detected, e.g. blinks or lost tracking)
    missing_mask = np.zeros(data.shape, dtype=bool)
    missing_mask[:, 0] = np.isnan(data[:, 0])   # x
    missing_mask[:, 1] = np.isnan(data[:, 1])   # y
    missing_mask[:, 2] = data[:, 2] == 0         # pupil

    # --- Format as strings ---
    data_str = np.empty(data.shape, dtype=object)
    for col in range(data.shape[1]):
        for row in range(data.shape[0]):
            if missing_mask[row, col]:
                data_str[row, col] = "n/a"
            else:
                data_str[row, col] = f"{data[row, col]:.4f}"

    # --- Write gzipped TSV (no header per BIDS physio spec) ---
    os.makedirs(os.path.dirname(output_tsv_gz), exist_ok=True)
    with gzip.open(output_tsv_gz, "wt") as f:
        for row in data_str:
            f.write("\t".join(row) + "\n")
    print(f"  Wrote {n_samples} samples (scan window) -> {output_tsv_gz}")

    # --- Determine recorded eye (BIDS values: "Left", "Right", "Both") ---
    _EYE_MAP = {"LEFT_EYE": "Left", "RIGHT_EYE": "Right", "BINOCULAR": "Both"}
    if eye_info:
        recorded_eye = _EYE_MAP.get(str(eye_info).upper(), str(eye_info).capitalize())
    else:
        recorded_eye = "Left" if len(col_names) <= 3 else "Both"

    # --- Write JSON sidecar ---
    # StartTime = 0.0 because we trimmed the data to start at the first trigger,
    # which corresponds to the first BOLD volume.
    json_path = output_tsv_gz.replace("_physio.tsv.gz", "_physio.json")
    sidecar = {
        "SamplingFrequency": float(sfreq),
        "StartTime": 0.0,
        "Columns": col_names,
        "Manufacturer": "SR Research",
        "ManufacturersModelName": "EyeLink",
        "RecordedEye": recorded_eye,
        "EyeTrackingMethod": "video-based",
        "ScanWindow": {
            "Description": "Data trimmed to scan window defined by scanner triggers (input=255). "
                           "First sample corresponds to first BOLD volume onset.",
            "FirstTriggerTime": round(float(first_trig_s), 3),
            "LastTriggerTime": round(float(last_trig_s), 3),
            "NumTriggers": len(triggers),
        },
    }

    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(sidecar, f, indent=4)
        f.write("\n")
    print(f"  Wrote JSON -> {json_path}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Convert EyeLink EDF to BIDS physio TSV.GZ"
    )
    parser.add_argument("edf_file", help="Path to EDF file")
    parser.add_argument("output_tsv_gz", nargs="?", default=None,
                        help="Output physio TSV.GZ path (auto-generated if omitted)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.output_tsv_gz is None:
        subj, sess, run, phase = parse_edf_filename(args.edf_file)
        sub = bids_sub(subj)
        ses = bids_ses_fr(sess)

        if phase == "encoding":
            fname = f"{sub}_{ses}_task-NATencoding_run-{run:02d}_recording-eye_physio.tsv.gz"
        else:
            fname = f"{sub}_{ses}_task-NATretrieval_recording-eye_physio.tsv.gz"

        output = os.path.join(BIDS_ROOT, sub, ses, "func", fname)
    else:
        output = args.output_tsv_gz

    print(f"Input: {args.edf_file}")
    print(f"Output: {output}")
    convert_file(args.edf_file, output, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
