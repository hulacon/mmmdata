#!/usr/bin/env python3
"""Generate physio_triage.csv by parsing every PhysioLog DICOM in sourcedata.

`physio_triage.csv` gates which scanner physio recordings get converted:
`generate_inventory.py` emits inventory rows only for COMPLETE and PARTIAL.
The table was previously hand-maintained and covered sub-03/04/05 only, so a
new subject's physio was silently absent from BIDS -- the files converted fine,
nothing asked for them.

Status is a property of the recording, not a judgement call:

    no waveform sections (Info.log only)  -> INFO_ONLY
    ratio >= 0.9                          -> COMPLETE
    ratio >= 0.5                          -> PARTIAL
    otherwise                             -> TRUNCATED

where ratio = recorded duration / expected duration, and expected duration
comes from the volume count and TR recorded in the log's own ACQUISITION_INFO.

Run with --check to verify the rule reproduces the existing table rather than
rewriting it; that is what licences trusting it on a new cohort.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from statistics import median

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from raw2bids_converters.common import SOURCE_DIR  # noqa: E402
from raw2bids_converters.physio_dcm import parse_pmu_text, _find_section_pos  # noqa: E402

TIC_MS = 2.5          # Siemens PMU tick
WAVE_SECTIONS = ("ECG", "EXT", "PULS", "RESP")
COMPLETE_MIN = 0.9
PARTIAL_MIN = 0.5

FIELDS = ["sub", "ses", "series", "size_mb", "status", "num_volumes", "tr_ms",
          "expected_dur", "rec_dur", "ratio", "sections", "source_path"]


def _read_pmu_text(series_dir: str) -> tuple[str | None, float]:
    """Return the PMU text blob and the DICOM's size in MB."""
    import pydicom
    files = [f for f in os.listdir(series_dir) if not f.startswith(".")]
    if not files:
        return None, 0.0
    fpath = os.path.join(series_dir, files[0])
    size_mb = round(os.path.getsize(fpath) / 1e6, 1)
    ds = pydicom.dcmread(fpath, force=True)
    try:
        raw = ds[0x7fe1, 0x1010].value
    except (KeyError, AttributeError):
        return None, size_mb
    return raw.decode("latin-1", errors="replace"), size_mb


def triage_series(series_dir: str) -> dict | None:
    """Compute the triage row for one PhysioLog series directory."""
    text, size_mb = _read_pmu_text(series_dir)
    if text is None:
        return None

    # Info.log-only files carry no waveform at all.
    if "Info.log" in text[:500] and _find_section_pos(text, "ECG") < 0:
        return {"size_mb": size_mb, "status": "INFO_ONLY", "num_volumes": "",
                "tr_ms": "", "expected_dur": "", "rec_dur": 0, "ratio": 0,
                "sections": ""}

    sections, acq = parse_pmu_text(text)
    present = [s for s in WAVE_SECTIONS if s in sections]
    if not present:
        return {"size_mb": size_mb, "status": "INFO_ONLY", "num_volumes": "",
                "tr_ms": "", "expected_dur": "", "rec_dur": 0, "ratio": 0,
                "sections": ""}

    num_volumes = acq.get("num_volumes", 0)

    # TR from the spacing of consecutive volume starts, in PMU ticks.
    tics = [acq["vol_start_tics"][k] for k in sorted(acq["vol_start_tics"])]
    diffs = [b - a for a, b in zip(tics, tics[1:])]
    tr_ms = round(median(diffs) * TIC_MS, 1) if diffs else 0.0

    # Recorded duration from the longest waveform section.
    rec_dur = 0.0
    for s in present:
        sec = sections[s]
        n = max((len(v) for v in sec["channels"].values()), default=0)
        rec_dur = max(rec_dur, n * sec["sample_time"] * TIC_MS / 1000.0)
    rec_dur = round(rec_dur, 1)

    expected_dur = round(num_volumes * tr_ms / 1000.0, 1)
    ratio = round(rec_dur / expected_dur, 3) if expected_dur else 0.0

    if ratio >= COMPLETE_MIN:
        status = "COMPLETE"
    elif ratio >= PARTIAL_MIN:
        status = "PARTIAL"
    else:
        status = "TRUNCATED"

    return {"size_mb": size_mb, "status": status, "num_volumes": num_volumes,
            "tr_ms": tr_ms, "expected_dur": expected_dur, "rec_dur": rec_dur,
            "ratio": ratio, "sections": ",".join(present)}


def walk(source_root: str, subjects: list[str]) -> list[dict]:
    rows = []
    for sub in subjects:
        sub_dir = os.path.join(source_root, sub)
        if not os.path.isdir(sub_dir):
            continue
        for ses in sorted(os.listdir(sub_dir)):
            dicom_dir = os.path.join(sub_dir, ses, "dicom")
            if not os.path.isdir(dicom_dir):
                continue
            for series in sorted(os.listdir(dicom_dir)):
                if not series.endswith("_PhysioLog"):
                    continue
                res = triage_series(os.path.join(dicom_dir, series))
                if res is None:
                    continue
                rows.append({
                    "sub": sub, "ses": ses, "series": series,
                    "source_path": f"{sub}/{ses}/dicom/{series}",
                    **res,
                })
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subjects", help="Comma-separated (default: all in sourcedata)")
    p.add_argument("--output", "-o", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "physio_triage.csv"))
    p.add_argument("--check", action="store_true",
                   help="Compare against the existing table; write nothing")
    p.add_argument("--append", action="store_true",
                   help="Keep existing rows, add only subjects absent from the table")
    args = p.parse_args(argv)

    source_root = SOURCE_DIR
    if args.subjects:
        subjects = args.subjects.split(",")
    else:
        subjects = sorted(d for d in os.listdir(source_root) if d.startswith("sub-"))

    rows = walk(source_root, subjects)
    print(f"parsed {len(rows)} PhysioLog series across {len(subjects)} subject(s)")

    existing = []
    if os.path.isfile(args.output):
        with open(args.output, newline="") as f:
            existing = list(csv.DictReader(f))

    if args.check:
        old = {(r["sub"], r["ses"], r["series"]): r["status"] for r in existing}
        new = {(r["sub"], r["ses"], r["series"]): r["status"] for r in rows}
        shared = set(old) & set(new)
        disagree = [k for k in shared if old[k] != new[k]]
        print(f"overlap {len(shared)}; status disagreements: {len(disagree)}")
        for k in disagree[:15]:
            print(f"   {k}: table={old[k]} recomputed={new[k]}")
        return 1 if disagree else 0

    if args.append:
        have = {r["sub"] for r in existing}
        rows = [r for r in rows if r["sub"] not in have]
        print(f"appending {len(rows)} row(s) for {sorted({r['sub'] for r in rows})}")
        rows = existing + rows

    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
