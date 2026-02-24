#!/usr/bin/env python3
"""Triage all 87 EDF files: extract scanner triggers, compute scan-window
quality for gaze and pupil channels, and write edf_triage.csv.

Viability threshold: a channel must have >=60% valid samples within the
scan window (first to last scanner trigger) to be considered viable.

Decision logic:
  - gaze viable OR pupil viable  -> include (channels = gaze+pupil | pupil_only)
  - neither viable               -> exclude

Output: raw2bids_converters/edf_triage.csv

Designed to run as a SLURM job (~10 min, <200 MB RAM, single core).
"""

import csv
import gc
import glob
import os
import re
import sys
import time

import numpy as np
from eyelinkio import read_edf

# --- Config ---
SOURCEDATA = "/gpfs/projects/hulacon/shared/mmmdata/sourcedata"
OUTPUT_CSV = "/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/raw2bids_converters/edf_triage.csv"
VIABILITY_THRESHOLD = 0.60  # >=60% valid to keep a channel

# --- EDF filename parsing (mirrors edf_to_physio.py) ---
def parse_edf_path(edf_path):
    """Return (sub, ses, run, phase) from an EDF path."""
    fname = os.path.basename(edf_path)
    # Derive sub/ses from directory structure
    parts = edf_path.split("/sourcedata/")[1].split("/")
    sub = parts[0]  # sub-XX
    ses = parts[1]  # ses-YY

    # Standard: s3s1r1m_2025_04_01_12_39.EDF
    m = re.match(r"s\d+s\d+r(\d+)([mr])_", fname)
    if m:
        return sub, ses, int(m.group(1)), "encoding" if m.group(2) == "m" else "retrieval"

    # Missing suffix: s4s6r1_2025_05_14_14_15.EDF
    m = re.match(r"s\d+s\d+r(\d+)_\d{4}_", fname)
    if m:
        phase = "encoding" if "/Encoding/" in edf_path or "/encoding/" in edf_path else "retrieval"
        return sub, ses, int(m.group(1)), phase

    # Extra 's': s4s4s1r_2025_04_30_12_38.EDF
    m = re.match(r"s\d+s\d+s(\d+)[mr]?_", fname)
    if m:
        phase = "encoding" if "/Encoding/" in edf_path or "/encoding/" in edf_path else "retrieval"
        return sub, ses, int(m.group(1)), phase

    return sub, ses, 0, "unknown"


def triage_one(edf_path):
    """Read one EDF, return a dict of triage metrics."""
    sub, ses, run, phase = parse_edf_path(edf_path)
    row = {
        "source_file": os.path.relpath(edf_path, SOURCEDATA),
        "sub": sub, "ses": ses, "phase": phase, "run": run,
        "sfreq": "", "n_samples": "", "duration_s": "",
        "n_triggers": 0, "scan_start_s": "", "scan_dur_s": "",
        "gaze_valid_pct": "", "pupil_valid_pct": "",
        "gaze_viable": False, "pupil_viable": False,
        "decision": "exclude", "channels": "",
        "note": "",
    }

    try:
        edf = read_edf(edf_path)
    except Exception as e:
        row["note"] = f"read_error: {e}"
        return row

    sfreq = edf["info"]["sfreq"]
    raw = edf["samples"]
    fields = edf["info"].get("sample_fields", [])
    if isinstance(raw, np.ndarray) and raw.ndim > 1:
        samples = {name: raw[i] for i, name in enumerate(fields)}
    else:
        samples = raw

    xpos = samples.get("xpos", np.array([]))
    pupil = samples.get("ps", samples.get("pupil_size", np.array([])))
    n_total = len(xpos)
    duration = n_total / sfreq

    row["sfreq"] = int(sfreq)
    row["n_samples"] = n_total
    row["duration_s"] = round(duration, 1)

    # Extract triggers (input value 255 = scanner volume onset)
    disc = edf["discrete"]
    inputs = disc.get("inputs", np.array([]))
    if len(inputs) > 0:
        inp_times = np.array([x[0] for x in inputs])
        inp_vals = np.array([x[1] for x in inputs])
        trig = inp_times[inp_vals == 255]
    else:
        trig = np.array([])

    row["n_triggers"] = len(trig)

    if len(trig) < 2:
        row["note"] = "insufficient triggers"
        # Fall back to full-recording metrics
        gaze_valid = float(~np.isnan(xpos)).mean() if n_total else 0
        pupil_valid = float(pupil > 0).mean() if n_total else 0
        row["gaze_valid_pct"] = round(100 * gaze_valid, 1)
        row["pupil_valid_pct"] = round(100 * pupil_valid, 1)
        row["gaze_viable"] = gaze_valid >= VIABILITY_THRESHOLD
        row["pupil_viable"] = pupil_valid >= VIABILITY_THRESHOLD
    else:
        first_s = trig[0]
        last_s = trig[-1]
        scan_dur = last_s - first_s
        first_i = max(0, min(int(first_s * sfreq), n_total - 1))
        last_i = max(first_i, min(int(last_s * sfreq), n_total - 1))

        row["scan_start_s"] = round(first_s, 1)
        row["scan_dur_s"] = round(scan_dur, 1)

        scan_x = xpos[first_i : last_i + 1]
        scan_p = pupil[first_i : last_i + 1]
        n_scan = len(scan_x)

        if n_scan > 0:
            gaze_valid = (~np.isnan(scan_x)).sum() / n_scan
            pupil_valid = (scan_p > 0).sum() / n_scan
        else:
            gaze_valid = 0.0
            pupil_valid = 0.0

        row["gaze_valid_pct"] = round(100 * gaze_valid, 1)
        row["pupil_valid_pct"] = round(100 * pupil_valid, 1)
        row["gaze_viable"] = gaze_valid >= VIABILITY_THRESHOLD
        row["pupil_viable"] = pupil_valid >= VIABILITY_THRESHOLD

    # Decision
    if row["gaze_viable"] and row["pupil_viable"]:
        row["decision"] = "include"
        row["channels"] = "gaze+pupil"
    elif row["gaze_viable"]:
        row["decision"] = "include"
        row["channels"] = "gaze_only"
    elif row["pupil_viable"]:
        row["decision"] = "include"
        row["channels"] = "pupil_only"
    else:
        row["decision"] = "exclude"
        row["channels"] = ""

    # Cleanup
    del edf, raw, samples, xpos, pupil, disc, inputs
    gc.collect()

    return row


def main():
    # Find all EDF files (exclude .EDF.tmp)
    pattern = os.path.join(SOURCEDATA, "sub-*/ses-*/eyetracking/*.EDF")
    edf_files = sorted(f for f in glob.glob(pattern) if not f.endswith(".EDF.tmp"))
    print(f"Found {len(edf_files)} EDF files")

    fieldnames = [
        "source_file", "sub", "ses", "phase", "run", "sfreq",
        "n_samples", "duration_s", "n_triggers", "scan_start_s",
        "scan_dur_s", "gaze_valid_pct", "pupil_valid_pct",
        "gaze_viable", "pupil_viable", "decision", "channels", "note",
    ]

    rows = []
    t0 = time.time()
    for i, fp in enumerate(edf_files, 1):
        t_file = time.time()
        row = triage_one(fp)
        elapsed = time.time() - t_file
        status = f"[{i:>2}/{len(edf_files)}] {row['decision']:>7} "
        status += f"gaze={row['gaze_valid_pct']:>5}% pupil={row['pupil_valid_pct']:>5}%"
        status += f"  {row['channels']:<12} ({elapsed:.1f}s) {os.path.basename(fp)}"
        print(status)
        rows.append(row)

    # Write CSV
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Summary
    elapsed_total = time.time() - t0
    n_include = sum(1 for r in rows if r["decision"] == "include")
    n_exclude = sum(1 for r in rows if r["decision"] == "exclude")
    n_gaze_pupil = sum(1 for r in rows if r["channels"] == "gaze+pupil")
    n_pupil_only = sum(1 for r in rows if r["channels"] == "pupil_only")
    n_gaze_only = sum(1 for r in rows if r["channels"] == "gaze_only")

    print(f"\n{'='*60}")
    print(f"Total: {len(rows)} files in {elapsed_total:.0f}s")
    print(f"  Include: {n_include}  (gaze+pupil: {n_gaze_pupil}, "
          f"pupil_only: {n_pupil_only}, gaze_only: {n_gaze_only})")
    print(f"  Exclude: {n_exclude}")
    print(f"Output: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
