#!/usr/bin/env python3
"""Convert Siemens PhysioLog DICOMs into BIDS _physio.tsv.gz files.

Handles files with conversion_type='physio_dcm'. These are Siemens PMU
physiological recordings embedded in DICOM private tag (7FE1,1010).

Each PhysioLog DICOM contains 5 concatenated log sections:
  ECG  (SampleTime=1ms,  4 channels: ECG1-ECG4)
  PULS (SampleTime=2ms,  1 channel: pulse oximetry)
  RESP (SampleTime=8ms,  1 channel: respiratory belt)
  EXT  (SampleTime=8ms,  external trigger)
  ACQUISITION_INFO (volume/slice timing)

Output per BIDS spec for physio data:
  - _recording-cardiac_physio.tsv.gz  + .json  (ECG channel 1)
  - _recording-pulse_physio.tsv.gz    + .json  (PULS)
  - _recording-respiratory_physio.tsv.gz + .json (RESP)

Requires: pydicom, numpy

Usage:
    python physio_dcm.py <dicom_dir> [<output_dir>] [--dry-run]
"""

import argparse
import gzip
import json
import os
import re
import sys

import numpy as np

from common import BIDS_ROOT, SOURCE_DIR


# Sampling rates derived from SampleTime (ms per sample)
# SampleTime=1 -> 1000 Hz, SampleTime=2 -> 500 Hz, SampleTime=8 -> 125 Hz
RECORDINGS = {
    "cardiac": {"section": "ECG", "sample_time": 1, "sfreq": 1000.0,
                "channel": "ECG1"},
    "pulse":   {"section": "PULS", "sample_time": 2, "sfreq": 500.0,
                "channel": "PULS"},
    "respiratory": {"section": "RESP", "sample_time": 8, "sfreq": 125.0,
                    "channel": "RESP"},
}


def _find_section_pos(text, sec_name):
    """Return character offset of a standalone section header, or -1."""
    idx = text.find(f"\n{sec_name}\n")
    if idx >= 0:
        return idx + 1
    idx = text.find(f"{sec_name}\n")
    if idx >= 0:
        return idx
    return -1


def parse_pmu_text(text):
    """Parse the full PMU text into per-section data.

    Returns
    -------
    sections : dict
        {section_name: {"sample_time": int, "timestamps": ndarray, "values": ndarray}}
    acq_info : dict
        {"num_volumes": int, "num_slices": int, "vol_start_tics": dict}
    """
    # Locate all sections
    sec_names = ["ECG", "PULS", "RESP", "EXT", "ACQUISITION_INFO"]
    sec_pos = {}
    for sn in sec_names:
        p = _find_section_pos(text, sn)
        if p >= 0:
            sec_pos[sn] = p

    sorted_secs = sorted(sec_pos.items(), key=lambda x: x[1])
    sections = {}

    for i, (sn, sp) in enumerate(sorted_secs):
        if sn == "ACQUISITION_INFO":
            continue
        se = sorted_secs[i + 1][1] if i + 1 < len(sorted_secs) else len(text)
        chunk = text[sp:se]

        # Trim null padding
        null_idx = chunk.find('\x00')
        if null_idx >= 0:
            chunk = chunk[:null_idx]

        # Parse SampleTime
        st_m = re.search(r'SampleTime\s*=\s*(\d+)', chunk)
        sample_time = int(st_m.group(1)) if st_m else 0

        # Parse data rows: ACQ_TIME_TICS  CHANNEL  VALUE  [SIGNAL]
        timestamps = []
        values = {}
        for line in chunk.split('\n'):
            m = re.match(r'\s+(\d+)\s+(\w+)\s+(\d+)', line)
            if m:
                tic = int(m.group(1))
                ch = m.group(2)
                val = int(m.group(3))
                if ch not in values:
                    values[ch] = []
                values[ch].append((tic, val))

        sections[sn] = {
            "sample_time": sample_time,
            "channels": values,
        }

    # Parse ACQUISITION_INFO
    acq_info = {"num_volumes": 0, "num_slices": 0, "vol_start_tics": {}}
    if "ACQUISITION_INFO" in sec_pos:
        acq_start = sec_pos["ACQUISITION_INFO"]
        acq_chunk = text[acq_start:acq_start + 50000]
        null_idx = acq_chunk.find('\x00')
        if null_idx >= 0:
            acq_chunk = acq_chunk[:null_idx]

        nv = re.search(r'NumVolumes\s*=\s*(\d+)', acq_chunk)
        ns = re.search(r'NumSlices\s*=\s*(\d+)', acq_chunk)
        if nv:
            acq_info["num_volumes"] = int(nv.group(1))
        if ns:
            acq_info["num_slices"] = int(ns.group(1))

        # Extract first-slice start time for each volume
        for line in acq_chunk.split('\n'):
            m = re.match(r'\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)', line)
            if m:
                vol = int(m.group(1))
                slc = int(m.group(2))
                start_tic = int(m.group(3))
                if slc == 0 and vol not in acq_info["vol_start_tics"]:
                    acq_info["vol_start_tics"][vol] = start_tic

    return sections, acq_info


def _resample_channel(tics_values, sample_time_ms):
    """Convert irregularly-spaced (tic, value) pairs to a uniform time series.

    Parameters
    ----------
    tics_values : list of (tic, value) tuples
        Raw data from the PMU section.
    sample_time_ms : int
        Nominal sampling interval in ms (from SampleTime header).

    Returns
    -------
    first_tic : int
        Tic of the first sample.
    values : ndarray
        Uniformly-sampled values (nearest-neighbor from raw tics).
    """
    if not tics_values:
        return 0, np.array([])

    tics = np.array([t for t, v in tics_values])
    vals = np.array([v for t, v in tics_values])

    first_tic = int(tics[0])
    last_tic = int(tics[-1])

    # Tic spacing in tics (2.5ms per tic -> sample_time_ms / 2.5 tics per sample)
    tic_step = sample_time_ms / 2.5
    n_samples = int(np.round((last_tic - first_tic) / tic_step)) + 1

    uniform_tics = np.linspace(first_tic, first_tic + (n_samples - 1) * tic_step,
                               n_samples)
    # Nearest-neighbor lookup
    indices = np.searchsorted(tics, uniform_tics, side='right') - 1
    indices = np.clip(indices, 0, len(vals) - 1)
    uniform_vals = vals[indices]

    return first_tic, uniform_vals


def convert_file(dicom_dir, output_base, dry_run=False):
    """Convert a PhysioLog DICOM directory to BIDS physio files.

    Parameters
    ----------
    dicom_dir : str
        Path to the PhysioLog DICOM directory (contains one .dcm file).
    output_base : str
        Base path for output files. The recording entity and suffix will be
        appended: e.g. if output_base is
          sub-03/ses-04/func/sub-03_ses-04_task-CRencoding_run-01
        then output files will be:
          ..._recording-cardiac_physio.tsv.gz
          ..._recording-pulse_physio.tsv.gz
          ..._recording-respiratory_physio.tsv.gz
    dry_run : bool
        If True, print what would be done without writing.

    Returns
    -------
    bool
        True if conversion succeeded.
    """
    # Find the DICOM file
    dcm_files = [f for f in os.listdir(dicom_dir) if not f.startswith('.')]
    if not dcm_files:
        print(f"  WARNING: No files in {dicom_dir}", file=sys.stderr)
        return False

    fpath = os.path.join(dicom_dir, dcm_files[0])

    try:
        import pydicom
    except ImportError:
        print("ERROR: pydicom not installed. Run: pip install pydicom",
              file=sys.stderr)
        return False

    ds = pydicom.dcmread(fpath, force=True)
    try:
        raw = ds[0x7fe1, 0x1010].value
    except (KeyError, AttributeError):
        print(f"  WARNING: No private tag (7FE1,1010) in {fpath}",
              file=sys.stderr)
        return False

    text = raw.decode('latin-1', errors='replace')

    # Check for Info.log-only files
    if "Info.log" in text[:500] and _find_section_pos(text, "ECG") < 0:
        print(f"  SKIP: Info.log-only (no physio data) in {dicom_dir}")
        return False

    sections, acq_info = parse_pmu_text(text)

    # Determine StartTime: time of first BOLD volume in seconds
    # relative to physio recording start
    vol0_tic = acq_info["vol_start_tics"].get(0, None)

    wrote_any = False
    for rec_name, rec_cfg in RECORDINGS.items():
        sec_name = rec_cfg["section"]
        if sec_name not in sections:
            continue

        sec = sections[sec_name]
        ch_name = rec_cfg["channel"]
        if ch_name not in sec["channels"]:
            continue

        tics_values = sec["channels"][ch_name]
        if not tics_values:
            continue

        sample_time_ms = rec_cfg["sample_time"]
        sfreq = rec_cfg["sfreq"]

        first_tic, values = _resample_channel(tics_values, sample_time_ms)

        if len(values) == 0:
            continue

        # Compute StartTime relative to first BOLD volume
        if vol0_tic is not None:
            start_time = (first_tic - vol0_tic) * 2.5 / 1000  # tics to seconds
        else:
            start_time = 0.0

        # Output paths
        tsv_gz_path = f"{output_base}_recording-{rec_name}_physio.tsv.gz"
        json_path = f"{output_base}_recording-{rec_name}_physio.json"

        if dry_run:
            print(f"  [dry-run] Would write {len(values)} samples -> {tsv_gz_path}")
            print(f"  [dry-run] Would write JSON -> {json_path}")
            wrote_any = True
            continue

        os.makedirs(os.path.dirname(tsv_gz_path), exist_ok=True)

        # Write gzipped TSV (no header per BIDS physio spec)
        with gzip.open(tsv_gz_path, "wt") as f:
            for v in values:
                f.write(f"{v}\n")
        print(f"  Wrote {len(values)} samples -> {tsv_gz_path}")

        # Write JSON sidecar
        col_name = {
            "cardiac": "cardiac",
            "pulse": "cardiac",  # BIDS: pulse ox measures cardiac rhythm
            "respiratory": "respiratory",
        }[rec_name]

        sidecar = {
            "SamplingFrequency": sfreq,
            "StartTime": round(start_time, 4),
            "Columns": [col_name],
            "Manufacturer": "Siemens",
            "ManufacturersModelName": "PMU",
        }
        with open(json_path, "w") as f:
            json.dump(sidecar, f, indent=4)
            f.write("\n")
        print(f"  Wrote JSON -> {json_path}")
        wrote_any = True

    return wrote_any


def main():
    parser = argparse.ArgumentParser(
        description="Convert Siemens PhysioLog DICOM to BIDS physio TSV.GZ"
    )
    parser.add_argument("dicom_dir", help="Path to PhysioLog DICOM directory")
    parser.add_argument("output_base", nargs="?", default=None,
                        help="Output base path (without _recording-*_physio suffix)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print(f"Input: {args.dicom_dir}")
    if args.output_base:
        print(f"Output base: {args.output_base}")
        convert_file(args.dicom_dir, args.output_base, dry_run=args.dry_run)
    else:
        print("ERROR: output_base required (auto-generation not yet implemented)")
        sys.exit(1)


if __name__ == "__main__":
    main()
