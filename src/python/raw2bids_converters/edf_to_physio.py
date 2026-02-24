#!/usr/bin/env python3
"""Convert EyeLink EDF files into BIDS _recording-eye_physio.tsv.gz files.

Handles files with conversion_type='edf_to_physio' (87 complete EDF files).
These are EyeLink eyetracking recordings from the free recall sessions
(encoding and retrieval), BIDS ses-19 to ses-28.

Requires: eyelinkio (pip install eyelinkio)

EDF naming: s{Subj}s{Sess}r{Run}{m|r}_YYYY_MM_DD_HH_MM.EDF
  m = encoding (memory), r = recall (retrieval)

Output format:
  - _recording-eye_physio.tsv.gz: continuous eye position + pupil samples
    (no header, tab-separated, gzipped)
  - _recording-eye_physio.json: sampling rate, start time, column names

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


def convert_file(edf_path, output_tsv_gz, dry_run=False):
    """Convert an EDF file to BIDS physio TSV.GZ + JSON sidecar.

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

    # Build column data arrays
    columns = []
    col_names = []

    xpos = samples.get("xpos")
    ypos = samples.get("ypos")
    ps = samples.get("ps", samples.get("pupil_size"))

    if xpos is not None and ypos is not None:
        columns.append(xpos)
        col_names.append("eye1_x_coordinate")
        columns.append(ypos)
        col_names.append("eye1_y_coordinate")
        if ps is not None:
            columns.append(ps)
            col_names.append("eye1_pupil_size")

    if not columns:
        print(f"  WARNING: No sample data found in {edf_path}", file=sys.stderr)
        return False

    # Stack into (n_samples, n_columns)
    data = np.column_stack(columns)
    n_samples = data.shape[0]

    # Replace NaN/missing with 'n/a'
    # EyeLink uses ~1e8 for missing gaze and 0 for missing pupil size
    missing_mask = np.isnan(data) | (np.abs(data) > 1e6)
    # Mark pupil_size=0 as missing (pupil not detected, e.g. during blinks)
    for i, name in enumerate(col_names):
        if "pupil" in name:
            missing_mask[:, i] |= data[:, i] == 0

    data_str = np.empty(data.shape, dtype=object)
    for i in range(data.shape[1]):
        for j in range(data.shape[0]):
            if missing_mask[j, i]:
                data_str[j, i] = "n/a"
            else:
                data_str[j, i] = f"{data[j, i]:.4f}"

    # Write gzipped TSV (no header per BIDS physio spec)
    os.makedirs(os.path.dirname(output_tsv_gz), exist_ok=True)
    with gzip.open(output_tsv_gz, "wt") as f:
        for row in data_str:
            f.write("\t".join(row) + "\n")
    print(f"  Wrote {n_samples} samples -> {output_tsv_gz}")

    # Determine recorded eye (BIDS values: "Left", "Right", "Both")
    _EYE_MAP = {"LEFT_EYE": "Left", "RIGHT_EYE": "Right", "BINOCULAR": "Both"}
    if eye_info:
        recorded_eye = _EYE_MAP.get(str(eye_info).upper(), str(eye_info).capitalize())
    else:
        recorded_eye = "Left" if len(col_names) <= 3 else "Both"

    # Write JSON sidecar
    json_path = output_tsv_gz.replace("_physio.tsv.gz", "_physio.json")
    sidecar = {
        "SamplingFrequency": float(sfreq),
        "StartTime": 0.0,  # relative to first BOLD volume (needs alignment)
        "Columns": col_names,
        "Manufacturer": "SR Research",
        "ManufacturersModelName": "EyeLink",
        "RecordedEye": recorded_eye,
        "EyeTrackingMethod": "video-based",
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
