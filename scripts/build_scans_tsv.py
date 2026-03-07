#!/usr/bin/env python3
"""
Generate BIDS-compliant scans.tsv files for every subject/session pair.

For each session, walks the datatype directories (anat, func, dwi, fmap, beh)
and produces a sub-XX_ses-YY_scans.tsv with one row per primary file (NIfTI,
events TSV, physio TSV.gz, beh TSV). Columns include the BIDS-required
`filename` plus custom metadata columns documented in the companion
scans.json sidecar.

Usage:
    python build_scans_tsv.py                  # all subjects, all sessions
    python build_scans_tsv.py --subjects sub-03
    python build_scans_tsv.py --subjects sub-03 --sessions ses-01 ses-04
    python build_scans_tsv.py --dry-run         # print output, don't write
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import nibabel as nib

# ---------------------------------------------------------------------------
# Path setup — use config if importable, else fall back to well-known path
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent

sys.path.insert(0, str(_REPO_ROOT / "src" / "python"))
try:
    from core.config import load_config
    _config = load_config(config_dir=_REPO_ROOT / "config")
    BIDS_ROOT = Path(_config["paths"]["bids_project_dir"])
except Exception:
    BIDS_ROOT = Path("/gpfs/projects/hulacon/shared/mmmdata")

# ---------------------------------------------------------------------------
# Which suffixes are "primary" rows in scans.tsv (one row per primary file)
# ---------------------------------------------------------------------------
# Per BIDS spec: filename column uses paths relative to the session directory.
# We include NIfTI images (the core data files) as rows. Sidecars (.json),
# companion files (.bval/.bvec), sbref, events, and physio are captured as
# boolean columns on the parent NIfTI row.
PRIMARY_NIFTI_SUFFIXES = {"bold", "T1w", "T2w", "dwi", "epi", "sbref"}

# Regex to parse BIDS entities from a filename
ENTITY_RE = re.compile(
    r"(?:sub-(?P<sub>[^_]+))"
    r"(?:_ses-(?P<ses>[^_]+))?"
    r"(?:_task-(?P<task>[^_]+))?"
    r"(?:_acq-(?P<acq>[^_]+))?"
    r"(?:_dir-(?P<dir>[^_]+))?"
    r"(?:_run-(?P<run>[^_]+))?"
    r"(?:_recording-(?P<rec>[^_]+))?"
    r"_(?P<suffix>[a-zA-Z0-9]+)"
)


def parse_entities(fname: str) -> dict:
    """Extract BIDS entities from a filename stem."""
    m = ENTITY_RE.match(Path(fname).name.split(".")[0])
    if not m:
        return {}
    return {k: v for k, v in m.groupdict().items() if v is not None}


def get_nifti_info(nii_path: Path) -> dict:
    """Read NIfTI header for shape and zooms."""
    try:
        img = nib.load(str(nii_path))
        shape = img.header.get_data_shape()
        zooms = img.header.get_zooms()
        n_volumes = int(shape[3]) if len(shape) > 3 else 1
        return {
            "n_volumes": n_volumes,
            "nx": int(shape[0]),
            "ny": int(shape[1]),
            "nz": int(shape[2]),
            "voxel_x": round(float(zooms[0]), 3),
            "voxel_y": round(float(zooms[1]), 3),
            "voxel_z": round(float(zooms[2]), 3),
        }
    except Exception as e:
        print(f"  WARNING: could not read NIfTI header for {nii_path}: {e}",
              file=sys.stderr)
        return {}


def read_json_sidecar(json_path: Path) -> dict:
    """Read a JSON sidecar, return empty dict on failure."""
    try:
        with open(json_path) as f:
            return json.load(f)
    except Exception:
        return {}


def count_tsv_rows(tsv_path: Path) -> int:
    """Count data rows (excluding header) in a TSV file."""
    try:
        with open(tsv_path) as f:
            return sum(1 for _ in f) - 1  # subtract header
    except Exception:
        return 0


def find_matching_file(session_dir: Path, nii_path: Path, target_suffix: str,
                       target_ext: str, recording: str = None) -> Path | None:
    """Find a file matching the same entities but different suffix/recording."""
    stem = nii_path.name
    # Strip .nii.gz
    base = stem.replace(".nii.gz", "")
    entities = parse_entities(base)
    if not entities:
        return None

    # Build the expected filename
    parts = [f"sub-{entities['sub']}"]
    if "ses" in entities:
        parts.append(f"ses-{entities['ses']}")
    if "task" in entities:
        parts.append(f"task-{entities['task']}")
    if "acq" in entities:
        parts.append(f"acq-{entities['acq']}")
    if "dir" in entities:
        parts.append(f"dir-{entities['dir']}")
    if "run" in entities:
        parts.append(f"run-{entities['run']}")
    if recording:
        parts.append(f"recording-{recording}")
    parts.append(target_suffix)

    expected_name = "_".join(parts) + target_ext
    # Determine datatype directory
    datatype_dir = nii_path.parent
    # For physio/events, they're typically in the same datatype dir
    candidate = datatype_dir / expected_name
    if candidate.exists():
        return candidate
    return None


def build_scans_row(session_dir: Path, nii_path: Path) -> dict:
    """Build a single scans.tsv row for a NIfTI file."""
    rel_path = nii_path.relative_to(session_dir)
    entities = parse_entities(nii_path.name)
    suffix = entities.get("suffix", "")

    row = {
        "filename": str(rel_path),
    }

    # --- JSON sidecar ---
    json_path = nii_path.with_name(nii_path.name.replace(".nii.gz", ".json"))
    has_json = json_path.exists()
    row["has_json"] = has_json
    sidecar = read_json_sidecar(json_path) if has_json else {}

    # --- Acquisition time ---
    row["acq_time"] = sidecar.get("AcquisitionTime", "n/a")

    # --- NIfTI header info ---
    nii_info = get_nifti_info(nii_path)
    row["n_volumes"] = nii_info.get("n_volumes", "n/a")

    # --- Duration ---
    acq_dur = sidecar.get("AcquisitionDuration")
    tr = sidecar.get("RepetitionTime")
    n_vol = nii_info.get("n_volumes")
    if acq_dur is not None:
        row["duration_s"] = round(float(acq_dur), 2)
    elif tr is not None and n_vol is not None and n_vol > 1:
        row["duration_s"] = round(float(tr) * int(n_vol), 2)
    else:
        row["duration_s"] = "n/a"

    # --- Events (only for bold) ---
    if suffix == "bold":
        events_tsv = find_matching_file(session_dir, nii_path, "events", ".tsv")
        row["n_events"] = count_tsv_rows(events_tsv) if events_tsv else "n/a"
        events_json = find_matching_file(session_dir, nii_path, "events", ".json")
        row["has_events_json"] = events_json is not None and events_json.exists()
    else:
        row["n_events"] = "n/a"
        row["has_events_json"] = "n/a"

    # --- Physio channels (only for bold) ---
    if suffix == "bold":
        for rec in ("cardiac", "pulse", "respiratory"):
            physio = find_matching_file(
                session_dir, nii_path, "physio", ".tsv.gz", recording=rec
            )
            row[f"physio_{rec}"] = physio is not None and physio.exists()
        # Eyetracking
        eye = find_matching_file(
            session_dir, nii_path, "physio", ".tsv.gz", recording="eye"
        )
        row["eyetracking"] = eye is not None and eye.exists()
    else:
        for rec in ("cardiac", "pulse", "respiratory"):
            row[f"physio_{rec}"] = "n/a"
        row["eyetracking"] = "n/a"

    # --- SBRef (only for bold) ---
    if suffix == "bold":
        sbref = find_matching_file(session_dir, nii_path, "sbref", ".nii.gz")
        row["has_sbref"] = sbref is not None and sbref.exists()
    else:
        row["has_sbref"] = "n/a"

    return row


def build_session_scans(session_dir: Path) -> list[dict]:
    """Build all scans.tsv rows for a single session directory."""
    rows = []

    # Collect all NIfTI files across datatypes
    nifti_files = sorted(session_dir.rglob("*.nii.gz"))

    for nii_path in nifti_files:
        entities = parse_entities(nii_path.name)
        suffix = entities.get("suffix", "")

        # Only include primary data files, not sbref (tracked as column on bold)
        if suffix == "sbref":
            continue

        row = build_scans_row(session_dir, nii_path)
        rows.append(row)

    return rows


COLUMNS = [
    "filename",
    "acq_time",
    "n_volumes",
    "duration_s",
    "n_events",
    "physio_cardiac",
    "physio_pulse",
    "physio_respiratory",
    "eyetracking",
    "has_sbref",
    "has_json",
    "has_events_json",
]


def format_value(val) -> str:
    """Format a value for TSV output."""
    if isinstance(val, bool):
        return "true" if val else "false"
    return str(val)


def write_scans_tsv(session_dir: Path, rows: list[dict], dry_run: bool = False):
    """Write a scans.tsv file for a session."""
    sub = session_dir.parent.name
    ses = session_dir.name
    out_path = session_dir / f"{sub}_{ses}_scans.tsv"

    if not rows:
        print(f"  SKIP {sub}/{ses}: no NIfTI files found")
        return

    if dry_run:
        print(f"\n--- {out_path.relative_to(session_dir.parent.parent)} ---")
        print("\t".join(COLUMNS))
        for row in rows:
            print("\t".join(format_value(row.get(c, "n/a")) for c in COLUMNS))
        return

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=COLUMNS, delimiter="\t",
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({c: format_value(row.get(c, "n/a")) for c in COLUMNS})

    print(f"  WROTE {out_path.relative_to(session_dir.parent.parent)} "
          f"({len(rows)} rows)")


def main():
    parser = argparse.ArgumentParser(
        description="Generate BIDS scans.tsv files for every subject/session."
    )
    parser.add_argument(
        "--bids-dir", type=Path, default=BIDS_ROOT,
        help=f"BIDS root directory (default: {BIDS_ROOT})"
    )
    parser.add_argument(
        "--subjects", nargs="+", default=None,
        help="Subject labels to process (e.g. sub-03 sub-04). Default: all."
    )
    parser.add_argument(
        "--sessions", nargs="+", default=None,
        help="Session labels to process (e.g. ses-01 ses-04). Default: all."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print output to stdout instead of writing files."
    )
    args = parser.parse_args()

    bids_dir = args.bids_dir.resolve()
    if not bids_dir.exists():
        print(f"ERROR: BIDS directory not found: {bids_dir}", file=sys.stderr)
        sys.exit(1)

    # Discover subjects
    if args.subjects:
        sub_dirs = sorted(bids_dir / s for s in args.subjects)
    else:
        sub_dirs = sorted(d for d in bids_dir.iterdir()
                          if d.is_dir() and d.name.startswith("sub-"))

    total_files = 0
    total_sessions = 0

    for sub_dir in sub_dirs:
        if not sub_dir.exists():
            print(f"WARNING: subject directory not found: {sub_dir}",
                  file=sys.stderr)
            continue

        print(f"\n{sub_dir.name}")

        # Discover sessions
        if args.sessions:
            ses_dirs = sorted(sub_dir / s for s in args.sessions)
        else:
            ses_dirs = sorted(d for d in sub_dir.iterdir()
                              if d.is_dir() and d.name.startswith("ses-"))

        for ses_dir in ses_dirs:
            if not ses_dir.exists():
                continue

            rows = build_session_scans(ses_dir)
            write_scans_tsv(ses_dir, rows, dry_run=args.dry_run)
            total_files += len(rows)
            total_sessions += 1

    print(f"\nDone: {total_sessions} sessions, {total_files} scan entries.")


if __name__ == "__main__":
    main()
