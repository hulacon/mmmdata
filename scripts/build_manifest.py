#!/usr/bin/env python3
"""
Build the MMMData manifest database (SQLite).

Ingests scans.tsv files, NIfTI headers, events metadata, physio metadata,
sourcedata file listings, derivative file listings, and session metadata
into a single queryable SQLite database.

Usage:
    python build_manifest.py                   # full rebuild
    python build_manifest.py --db manifest.db  # custom output path
    python build_manifest.py --skip-nifti      # skip slow NIfTI reads
"""

import argparse
import csv
import json
import os
import re
import sqlite3
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup
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

DEFAULT_DB = BIDS_ROOT / "inventory" / "manifest.db"

# ---------------------------------------------------------------------------
# BIDS entity parsing
# ---------------------------------------------------------------------------
ENTITY_RE = re.compile(
    r"(?:sub-(?P<sub>[^_/]+))"
    r"(?:_ses-(?P<ses>[^_/]+))?"
    r"(?:_task-(?P<task>[^_/]+))?"
    r"(?:_acq-(?P<acq>[^_/]+))?"
    r"(?:_dir-(?P<dir>[^_/]+))?"
    r"(?:_run-(?P<run>[^_/]+))?"
    r"(?:_recording-(?P<rec>[^_/]+))?"
    r"(?:_desc-(?P<desc>[^_/]+))?"
    r"_(?P<suffix>[a-zA-Z0-9]+)"
)


def parse_entities(fname: str) -> dict:
    stem = Path(fname).name.split(".")[0]
    m = ENTITY_RE.match(stem)
    if not m:
        return {}
    return {k: v for k, v in m.groupdict().items() if v is not None}


def get_extension(fname: str) -> str:
    """Get full extension (.nii.gz, .tsv.gz, .json, etc.)."""
    name = Path(fname).name
    if name.endswith(".nii.gz"):
        return ".nii.gz"
    if name.endswith(".tsv.gz"):
        return ".tsv.gz"
    return Path(name).suffix


def get_datatype(rel_path: str) -> str:
    """Extract datatype from relative path (first directory component)."""
    parts = Path(rel_path).parts
    if parts:
        first = parts[0]
        if first in ("anat", "func", "dwi", "fmap", "beh", "perf"):
            return first
    return "unknown"


# ---------------------------------------------------------------------------
# Schema creation
# ---------------------------------------------------------------------------
SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    path TEXT PRIMARY KEY,
    subject TEXT,
    session TEXT,
    datatype TEXT,
    task TEXT,
    run TEXT,
    suffix TEXT,
    format TEXT,
    size_bytes INTEGER,
    mtime TEXT
);

CREATE TABLE IF NOT EXISTS nifti_meta (
    path TEXT PRIMARY KEY REFERENCES files(path),
    nx INTEGER, ny INTEGER, nz INTEGER, nt INTEGER,
    voxel_x REAL, voxel_y REAL, voxel_z REAL,
    tr REAL,
    acq_duration REAL
);

CREATE TABLE IF NOT EXISTS events_meta (
    path TEXT PRIMARY KEY REFERENCES files(path),
    n_rows INTEGER,
    columns TEXT,
    onset_min REAL, onset_max REAL,
    duration_min REAL, duration_max REAL,
    end_max REAL
);

CREATE TABLE IF NOT EXISTS physio_meta (
    path TEXT PRIMARY KEY REFERENCES files(path),
    recording TEXT,
    sampling_rate REAL,
    n_columns INTEGER
);

CREATE TABLE IF NOT EXISTS sourcedata (
    path TEXT PRIMARY KEY,
    subject TEXT,
    session TEXT,
    category TEXT,
    file_count INTEGER
);

CREATE TABLE IF NOT EXISTS derivatives (
    path TEXT PRIMARY KEY,
    pipeline TEXT,
    subject TEXT,
    session TEXT,
    source_suffix TEXT,
    description TEXT
);

CREATE TABLE IF NOT EXISTS session_metadata (
    subject TEXT,
    session TEXT,
    acq_date TEXT,
    session_type TEXT,
    earbud_used TEXT,
    physio_used TEXT,
    eyetracking_used TEXT,
    experiment_start_time TEXT,
    sleep_hours REAL,
    mood TEXT,
    stress TEXT,
    session_note TEXT,
    scan_note TEXT,
    PRIMARY KEY (subject, session)
);

CREATE TABLE IF NOT EXISTS validation_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    check_name TEXT,
    subject TEXT,
    session TEXT,
    task TEXT,
    run TEXT,
    status TEXT,
    expected TEXT,
    actual TEXT,
    message TEXT,
    checked_at TEXT
);
"""


def create_schema(conn: sqlite3.Connection):
    conn.executescript(SCHEMA)
    conn.commit()


# ---------------------------------------------------------------------------
# 1. Ingest scans.tsv → files table
# ---------------------------------------------------------------------------
def ingest_scans_tsv(conn: sqlite3.Connection, bids_dir: Path):
    """Walk all scans.tsv files and populate the files table."""
    cursor = conn.cursor()
    count = 0

    for scans_tsv in sorted(bids_dir.rglob("*_scans.tsv")):
        session_dir = scans_tsv.parent
        sub = session_dir.parent.name
        ses = session_dir.name

        with open(scans_tsv, newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                rel_filename = row.get("filename", "")
                abs_path = session_dir / rel_filename
                # Store path relative to BIDS root
                try:
                    bids_rel = str(abs_path.relative_to(bids_dir))
                except ValueError:
                    bids_rel = str(abs_path)

                entities = parse_entities(Path(rel_filename).name)
                ext = get_extension(rel_filename)
                datatype = get_datatype(rel_filename)

                # Get file stats
                size = 0
                mtime = ""
                if abs_path.exists():
                    stat = abs_path.stat()
                    size = stat.st_size
                    mtime = str(stat.st_mtime)

                cursor.execute(
                    """INSERT OR REPLACE INTO files
                       (path, subject, session, datatype, task, run, suffix, format, size_bytes, mtime)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (bids_rel, sub, ses, datatype,
                     entities.get("task"), entities.get("run"),
                     entities.get("suffix"), ext, size, mtime)
                )
                count += 1

    conn.commit()
    print(f"  files: {count} rows from scans.tsv")


# ---------------------------------------------------------------------------
# 2. Ingest companion files (sbref, events, physio, json, beh) → files table
# ---------------------------------------------------------------------------
def ingest_companion_files(conn: sqlite3.Connection, bids_dir: Path):
    """Add non-NIfTI companion files to the files table."""
    cursor = conn.cursor()
    count = 0
    companion_globs = [
        "sub-*/ses-*/**/*_sbref.nii.gz",
        "sub-*/ses-*/**/*.json",
        "sub-*/ses-*/**/*_events.tsv",
        "sub-*/ses-*/**/*_physio.tsv.gz",
        "sub-*/ses-*/**/*_beh.tsv",
        "sub-*/ses-*/**/*.bval",
        "sub-*/ses-*/**/*.bvec",
    ]

    for pattern in companion_globs:
        for fpath in sorted(bids_dir.glob(pattern)):
            bids_rel = str(fpath.relative_to(bids_dir))

            # Skip if already in files table
            existing = cursor.execute(
                "SELECT 1 FROM files WHERE path = ?", (bids_rel,)
            ).fetchone()
            if existing:
                continue

            parts = fpath.relative_to(bids_dir).parts
            sub = parts[0] if len(parts) > 0 else None
            ses = parts[1] if len(parts) > 1 else None
            entities = parse_entities(fpath.name)
            ext = get_extension(fpath.name)
            datatype = get_datatype("/".join(parts[2:])) if len(parts) > 2 else "unknown"

            stat = fpath.stat()
            cursor.execute(
                """INSERT OR REPLACE INTO files
                   (path, subject, session, datatype, task, run, suffix, format, size_bytes, mtime)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (bids_rel, sub, ses, datatype,
                 entities.get("task"), entities.get("run"),
                 entities.get("suffix"), ext, stat.st_size, str(stat.st_mtime))
            )
            count += 1

    conn.commit()
    print(f"  companion files: {count} additional rows")


# ---------------------------------------------------------------------------
# 3. NIfTI metadata
# ---------------------------------------------------------------------------
def ingest_nifti_meta(conn: sqlite3.Connection, bids_dir: Path):
    """Read NIfTI headers for all .nii.gz files in the files table."""
    try:
        import nibabel as nib
    except ImportError:
        print("  nifti_meta: SKIPPED (nibabel not available)")
        return

    cursor = conn.cursor()
    rows = cursor.execute(
        "SELECT path FROM files WHERE format = '.nii.gz'"
    ).fetchall()
    count = 0

    for (rel_path,) in rows:
        abs_path = bids_dir / rel_path
        if not abs_path.exists():
            continue

        try:
            img = nib.load(str(abs_path))
            shape = img.header.get_data_shape()
            zooms = img.header.get_zooms()

            nx, ny, nz = int(shape[0]), int(shape[1]), int(shape[2])
            nt = int(shape[3]) if len(shape) > 3 else 1
            vx = round(float(zooms[0]), 3)
            vy = round(float(zooms[1]), 3)
            vz = round(float(zooms[2]), 3)

            # Read TR and duration from JSON sidecar
            json_path = abs_path.with_name(
                abs_path.name.replace(".nii.gz", ".json")
            )
            tr = None
            acq_dur = None
            if json_path.exists():
                try:
                    with open(json_path) as f:
                        sidecar = json.load(f)
                    tr = sidecar.get("RepetitionTime")
                    acq_dur = sidecar.get("AcquisitionDuration")
                except Exception:
                    pass

            cursor.execute(
                """INSERT OR REPLACE INTO nifti_meta
                   (path, nx, ny, nz, nt, voxel_x, voxel_y, voxel_z, tr, acq_duration)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (rel_path, nx, ny, nz, nt, vx, vy, vz, tr, acq_dur)
            )
            count += 1
        except Exception as e:
            print(f"    WARNING: {rel_path}: {e}", file=sys.stderr)

    conn.commit()
    print(f"  nifti_meta: {count} rows")


# ---------------------------------------------------------------------------
# 4. Events metadata
# ---------------------------------------------------------------------------
def ingest_events_meta(conn: sqlite3.Connection, bids_dir: Path):
    """Parse events.tsv files for row counts, columns, onset/duration ranges."""
    cursor = conn.cursor()
    rows = cursor.execute(
        "SELECT path FROM files WHERE suffix = 'events' AND format = '.tsv'"
    ).fetchall()
    count = 0

    for (rel_path,) in rows:
        abs_path = bids_dir / rel_path
        if not abs_path.exists():
            continue

        try:
            with open(abs_path, newline="") as f:
                reader = csv.DictReader(f, delimiter="\t")
                columns = reader.fieldnames or []
                onsets = []
                durations = []
                ends = []
                n_rows = 0
                for row in reader:
                    n_rows += 1
                    onset_val = None
                    dur_val = None
                    if "onset" in row and row["onset"] not in ("n/a", ""):
                        try:
                            onset_val = float(row["onset"])
                            onsets.append(onset_val)
                        except ValueError:
                            pass
                    if "duration" in row and row["duration"] not in ("n/a", ""):
                        try:
                            dur_val = float(row["duration"])
                            durations.append(dur_val)
                        except ValueError:
                            pass
                    if onset_val is not None and dur_val is not None:
                        ends.append(onset_val + dur_val)

            cursor.execute(
                """INSERT OR REPLACE INTO events_meta
                   (path, n_rows, columns, onset_min, onset_max,
                    duration_min, duration_max, end_max)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (rel_path, n_rows, json.dumps(columns),
                 min(onsets) if onsets else None,
                 max(onsets) if onsets else None,
                 min(durations) if durations else None,
                 max(durations) if durations else None,
                 max(ends) if ends else None)
            )
            count += 1
        except Exception as e:
            print(f"    WARNING: {rel_path}: {e}", file=sys.stderr)

    conn.commit()
    print(f"  events_meta: {count} rows")


# ---------------------------------------------------------------------------
# 5. Physio metadata
# ---------------------------------------------------------------------------
def ingest_physio_meta(conn: sqlite3.Connection, bids_dir: Path):
    """Parse physio JSON sidecars for sampling rate and recording type."""
    cursor = conn.cursor()
    rows = cursor.execute(
        "SELECT path FROM files WHERE suffix = 'physio' AND format = '.json'"
    ).fetchall()
    count = 0

    for (rel_path,) in rows:
        abs_path = bids_dir / rel_path
        if not abs_path.exists():
            continue

        entities = parse_entities(Path(rel_path).name)
        recording = entities.get("rec", "unknown")

        try:
            with open(abs_path) as f:
                sidecar = json.load(f)

            sampling_rate = sidecar.get("SamplingFrequency")
            columns = sidecar.get("Columns", [])

            # Use the .tsv.gz path as the key (not the .json)
            tsv_path = rel_path.replace("_physio.json", "_physio.tsv.gz")

            cursor.execute(
                """INSERT OR REPLACE INTO physio_meta
                   (path, recording, sampling_rate, n_columns)
                   VALUES (?, ?, ?, ?)""",
                (tsv_path, recording, sampling_rate, len(columns))
            )
            count += 1
        except Exception as e:
            print(f"    WARNING: {rel_path}: {e}", file=sys.stderr)

    conn.commit()
    print(f"  physio_meta: {count} rows")


# ---------------------------------------------------------------------------
# 6. Sourcedata
# ---------------------------------------------------------------------------
def ingest_sourcedata(conn: sqlite3.Connection, bids_dir: Path):
    """Walk sourcedata directories and catalogue contents."""
    cursor = conn.cursor()
    sd_root = bids_dir / "sourcedata"
    if not sd_root.exists():
        print("  sourcedata: SKIPPED (directory not found)")
        return

    count = 0
    categories = ("dicom", "behavioral", "audio", "eyetracking", "other")

    for sub_dir in sorted(sd_root.iterdir()):
        if not sub_dir.is_dir() or not sub_dir.name.startswith("sub-"):
            continue
        sub = sub_dir.name

        for ses_dir in sorted(sub_dir.iterdir()):
            if not ses_dir.is_dir() or not ses_dir.name.startswith("ses-"):
                continue
            ses = ses_dir.name

            for cat_dir in sorted(ses_dir.iterdir()):
                if not cat_dir.is_dir():
                    continue
                cat_name = cat_dir.name
                if cat_name not in categories:
                    cat_name = "other"

                if cat_name == "dicom":
                    # Count DICOM series directories
                    for series_dir in sorted(cat_dir.iterdir()):
                        if not series_dir.is_dir():
                            continue
                        n_files = sum(1 for f in series_dir.iterdir() if f.is_file())
                        rel = str(series_dir.relative_to(bids_dir))
                        cursor.execute(
                            """INSERT OR REPLACE INTO sourcedata
                               (path, subject, session, category, file_count)
                               VALUES (?, ?, ?, ?, ?)""",
                            (rel, sub, ses, "dicom", n_files)
                        )
                        count += 1
                else:
                    # Count files in category directory
                    n_files = sum(1 for f in cat_dir.iterdir() if f.is_file())
                    rel = str(cat_dir.relative_to(bids_dir))
                    cursor.execute(
                        """INSERT OR REPLACE INTO sourcedata
                           (path, subject, session, category, file_count)
                           VALUES (?, ?, ?, ?, ?)""",
                        (rel, sub, ses, cat_name, n_files)
                    )
                    count += 1

    conn.commit()
    print(f"  sourcedata: {count} rows")


# ---------------------------------------------------------------------------
# 7. Derivatives
# ---------------------------------------------------------------------------
DERIVATIVE_PIPELINES = {"fmriprep", "mriqc"}


def ingest_derivatives(conn: sqlite3.Connection, bids_dir: Path):
    """Walk derivative directories and catalogue key output files."""
    cursor = conn.cursor()
    deriv_root = bids_dir / "derivatives"
    if not deriv_root.exists():
        print("  derivatives: SKIPPED (directory not found)")
        return

    count = 0

    for pipeline in sorted(DERIVATIVE_PIPELINES):
        pipe_dir = deriv_root / pipeline
        if not pipe_dir.exists():
            continue

        for fpath in sorted(pipe_dir.rglob("sub-*")):
            if not fpath.is_file():
                continue

            rel = str(fpath.relative_to(bids_dir))
            entities = parse_entities(fpath.name)
            if not entities:
                continue

            sub = f"sub-{entities['sub']}" if "sub" in entities else None
            ses = f"ses-{entities['ses']}" if "ses" in entities else None

            cursor.execute(
                """INSERT OR REPLACE INTO derivatives
                   (path, pipeline, subject, session, source_suffix, description)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (rel, pipeline, sub, ses,
                 entities.get("suffix"), entities.get("desc"))
            )
            count += 1

    conn.commit()
    print(f"  derivatives: {count} rows")


# ---------------------------------------------------------------------------
# 8. Session metadata
# ---------------------------------------------------------------------------
def ingest_session_metadata(conn: sqlite3.Connection, bids_dir: Path):
    """Parse sub-XX_sessions.tsv files into session_metadata table."""
    cursor = conn.cursor()
    count = 0

    for sessions_tsv in sorted(bids_dir.glob("sub-*/sub-*_sessions.tsv")):
        sub = sessions_tsv.parent.name

        with open(sessions_tsv, newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                ses = row.get("session_id", "")

                def val(key):
                    v = row.get(key, "")
                    return v if v and v != "n/a" else None

                def float_val(key):
                    v = val(key)
                    if v is None:
                        return None
                    try:
                        return float(v)
                    except ValueError:
                        return None

                cursor.execute(
                    """INSERT OR REPLACE INTO session_metadata
                       (subject, session, acq_date, session_type,
                        earbud_used, physio_used, eyetracking_used,
                        experiment_start_time, sleep_hours, mood, stress,
                        session_note, scan_note)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (sub, ses, val("acq_time"), val("session_type"),
                     val("earbud_used"), val("physio_used"),
                     val("eyetracking_used"), val("experiment_start_time"),
                     float_val("sleep_hours"), val("mood"), val("stress"),
                     val("session_note"), val("scan_note"))
                )
                count += 1

    conn.commit()
    print(f"  session_metadata: {count} rows")


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------
def print_summary(conn: sqlite3.Connection):
    """Print a quick summary of the manifest contents."""
    cursor = conn.cursor()
    print("\n--- Manifest Summary ---")

    tables = [
        "files", "nifti_meta", "events_meta", "physio_meta",
        "sourcedata", "derivatives", "session_metadata"
    ]
    for table in tables:
        n = cursor.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  {table}: {n} rows")

    # Files by datatype
    print("\n  Files by datatype:")
    for row in cursor.execute(
        "SELECT datatype, COUNT(*) FROM files GROUP BY datatype ORDER BY datatype"
    ):
        print(f"    {row[0]}: {row[1]}")

    # Files by subject
    print("\n  Files by subject:")
    for row in cursor.execute(
        "SELECT subject, COUNT(*) FROM files GROUP BY subject ORDER BY subject"
    ):
        print(f"    {row[0]}: {row[1]}")

    # Derivatives by pipeline
    print("\n  Derivatives by pipeline:")
    for row in cursor.execute(
        "SELECT pipeline, COUNT(*) FROM derivatives GROUP BY pipeline ORDER BY pipeline"
    ):
        print(f"    {row[0]}: {row[1]}")

    # Sourcedata by category
    print("\n  Sourcedata by category:")
    for row in cursor.execute(
        "SELECT category, COUNT(*), SUM(file_count) FROM sourcedata GROUP BY category ORDER BY category"
    ):
        print(f"    {row[0]}: {row[1]} entries, {row[2]} files")


# ---------------------------------------------------------------------------
# Summary export
# ---------------------------------------------------------------------------
def export_summary(conn: sqlite3.Connection, out_path: Path):
    """Export a human-readable manifest summary as markdown."""
    cursor = conn.cursor()
    lines = []

    lines.append("# MMMData Manifest Summary\n")
    lines.append(f"*Auto-generated by `build_manifest.py`*\n")

    # --- Table row counts ---
    lines.append("## Database Overview\n")
    lines.append("| Table | Rows |")
    lines.append("|-------|------|")
    for table in ("files", "nifti_meta", "events_meta", "physio_meta",
                  "sourcedata", "derivatives", "session_metadata"):
        n = cursor.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        lines.append(f"| {table} | {n:,} |")
    lines.append("")

    # --- Session × bold run matrix ---
    lines.append("## Bold Runs per Session\n")

    subjects = [r[0] for r in cursor.execute(
        "SELECT DISTINCT subject FROM files ORDER BY subject"
    )]

    for sub in subjects:
        lines.append(f"### {sub}\n")

        # Get all sessions and tasks for this subject
        sessions = [r[0] for r in cursor.execute(
            "SELECT DISTINCT session FROM files WHERE subject=? ORDER BY session",
            (sub,)
        )]

        tasks = [r[0] for r in cursor.execute(
            """SELECT DISTINCT task FROM files
               WHERE subject=? AND suffix='bold' AND task IS NOT NULL
               ORDER BY task""",
            (sub,)
        )]

        if not tasks:
            lines.append("*No bold runs found.*\n")
            continue

        # Build matrix: session rows × task columns, value = run count
        header = "| Session | " + " | ".join(tasks) + " |"
        sep = "|---------|" + "|".join("---:" for _ in tasks) + "|"
        lines.append(header)
        lines.append(sep)

        for ses in sessions:
            cells = []
            for task in tasks:
                n = cursor.execute(
                    """SELECT COUNT(*) FROM files
                       WHERE subject=? AND session=? AND task=? AND suffix='bold'""",
                    (sub, ses, task)
                ).fetchone()[0]
                cells.append(str(n) if n > 0 else "—")
            lines.append(f"| {ses} | " + " | ".join(cells) + " |")
        lines.append("")

    # --- Physio coverage ---
    lines.append("## Physio Coverage (bold runs with physio files)\n")
    lines.append("| Subject | Session | cardiac | pulse | respiratory | eye |")
    lines.append("|---------|---------|---------|-------|-------------|-----|")

    for sub in subjects:
        sessions = [r[0] for r in cursor.execute(
            """SELECT DISTINCT session FROM files
               WHERE subject=? AND suffix='bold' ORDER BY session""",
            (sub,)
        )]
        for ses in sessions:
            bold_count = cursor.execute(
                "SELECT COUNT(*) FROM files WHERE subject=? AND session=? AND suffix='bold'",
                (sub, ses)
            ).fetchone()[0]
            if bold_count == 0:
                continue

            counts = {}
            for rec in ("cardiac", "pulse", "respiratory", "eye"):
                n = cursor.execute(
                    """SELECT COUNT(*) FROM physio_meta pm
                       JOIN files f ON pm.path = f.path
                       WHERE f.subject=? AND f.session=? AND pm.recording=?""",
                    (sub, ses, rec)
                ).fetchone()[0]
                counts[rec] = n

            lines.append(
                f"| {sub} | {ses} | {counts['cardiac']}/{bold_count} "
                f"| {counts['pulse']}/{bold_count} "
                f"| {counts['respiratory']}/{bold_count} "
                f"| {counts['eye']}/{bold_count} |"
            )
    lines.append("")

    # --- Events coverage ---
    lines.append("## Events File Coverage\n")
    lines.append("| Subject | Session | Task | Runs with events | Typical row count |")
    lines.append("|---------|---------|------|-----------------|-------------------|")

    for sub in subjects:
        for row in cursor.execute(
            """SELECT f.session, f.task, COUNT(*), GROUP_CONCAT(em.n_rows)
               FROM files f
               JOIN events_meta em ON em.path = f.path
               WHERE f.subject=? AND f.suffix='events'
               GROUP BY f.session, f.task
               ORDER BY f.session, f.task""",
            (sub,)
        ):
            ses, task, n_files, row_counts = row
            # Compute typical (mode/median)
            counts = [int(x) for x in row_counts.split(",") if x]
            typical = sorted(counts)[len(counts) // 2] if counts else "—"
            lines.append(f"| {sub} | {ses} | {task} | {n_files} | {typical} |")
    lines.append("")

    # --- Sourcedata summary ---
    lines.append("## Sourcedata Summary\n")
    lines.append("| Category | Entries | Total files |")
    lines.append("|----------|---------|-------------|")
    for row in cursor.execute(
        """SELECT category, COUNT(*), SUM(file_count)
           FROM sourcedata GROUP BY category ORDER BY category"""
    ):
        lines.append(f"| {row[0]} | {row[1]:,} | {row[2]:,} |")
    lines.append("")

    # --- Derivatives summary ---
    lines.append("## Derivatives Summary\n")
    lines.append("| Pipeline | Files | Subjects with output |")
    lines.append("|----------|-------|---------------------|")
    for row in cursor.execute(
        """SELECT pipeline, COUNT(*), COUNT(DISTINCT subject)
           FROM derivatives GROUP BY pipeline ORDER BY pipeline"""
    ):
        lines.append(f"| {row[0]} | {row[1]:,} | {row[2]} |")
    lines.append("")

    # --- Flagged issues ---
    lines.append("## Potential Issues\n")

    # Bold runs missing sbref
    lines.append("### Bold runs missing SBRef\n")
    missing_sbref = cursor.execute(
        """SELECT f.subject, f.session, f.task, f.run
           FROM files f
           WHERE f.suffix='bold'
           AND NOT EXISTS (
               SELECT 1 FROM files s
               WHERE s.subject=f.subject AND s.session=f.session
               AND s.task=f.task AND (s.run=f.run OR (s.run IS NULL AND f.run IS NULL))
               AND s.suffix='sbref'
           )
           ORDER BY f.subject, f.session, f.task"""
    ).fetchall()
    if missing_sbref:
        for row in missing_sbref:
            run_str = f" run-{row[3]}" if row[3] else ""
            lines.append(f"- {row[0]} {row[1]} {row[2]}{run_str}")
    else:
        lines.append("*None — all bold runs have matching SBRef files.*")
    lines.append("")

    # Bold runs missing JSON sidecar
    lines.append("### Bold runs missing JSON sidecar\n")
    missing_json = cursor.execute(
        """SELECT f.subject, f.session, f.task, f.run
           FROM files f
           WHERE f.suffix='bold' AND f.format='.nii.gz'
           AND NOT EXISTS (
               SELECT 1 FROM files j
               WHERE j.subject=f.subject AND j.session=f.session
               AND j.task=f.task AND (j.run=f.run OR (j.run IS NULL AND f.run IS NULL))
               AND j.suffix='bold' AND j.format='.json'
           )
           ORDER BY f.subject, f.session, f.task"""
    ).fetchall()
    if missing_json:
        for row in missing_json:
            run_str = f" run-{row[3]}" if row[3] else ""
            lines.append(f"- {row[0]} {row[1]} {row[2]}{run_str}")
    else:
        lines.append("*None — all bold NIfTIs have matching JSON sidecars.*")
    lines.append("")

    # Sessions missing from session_metadata
    lines.append("### Sessions in files but missing from session_metadata\n")
    missing_meta = cursor.execute(
        """SELECT DISTINCT f.subject, f.session
           FROM files f
           WHERE NOT EXISTS (
               SELECT 1 FROM session_metadata sm
               WHERE sm.subject=f.subject AND sm.session=f.session
           )
           ORDER BY f.subject, f.session"""
    ).fetchall()
    if missing_meta:
        for row in missing_meta:
            lines.append(f"- {row[0]} {row[1]}")
    else:
        lines.append("*None — all sessions have metadata.*")
    lines.append("")

    content = "\n".join(lines) + "\n"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(content)
    print(f"\nSummary exported to: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Build the MMMData manifest database (SQLite)."
    )
    parser.add_argument(
        "--bids-dir", type=Path, default=BIDS_ROOT,
        help=f"BIDS root directory (default: {BIDS_ROOT})"
    )
    parser.add_argument(
        "--db", type=Path, default=None,
        help=f"Output database path (default: <bids-dir>/inventory/manifest.db)"
    )
    parser.add_argument(
        "--skip-nifti", action="store_true",
        help="Skip NIfTI header reading (faster, but no nifti_meta table)"
    )
    parser.add_argument(
        "--summary-only", action="store_true",
        help="Only export summary from existing DB (no rebuild)"
    )
    args = parser.parse_args()

    bids_dir = args.bids_dir.resolve()
    db_path = args.db or (bids_dir / "inventory" / "manifest.db")

    if not bids_dir.exists():
        print(f"ERROR: BIDS directory not found: {bids_dir}", file=sys.stderr)
        sys.exit(1)

    summary_path = bids_dir / "inventory" / "manifest_summary.md"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    if args.summary_only:
        if not db_path.exists():
            print(f"ERROR: Database not found: {db_path}", file=sys.stderr)
            sys.exit(1)
        conn = sqlite3.connect(str(db_path))
        export_summary(conn, summary_path)
        conn.close()
        return

    # Remove old DB for clean rebuild
    if db_path.exists():
        db_path.unlink()

    print(f"Building manifest: {db_path}")
    print(f"BIDS root: {bids_dir}\n")

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

    create_schema(conn)

    # Ingest in order
    print("Ingesting...")
    ingest_scans_tsv(conn, bids_dir)
    ingest_companion_files(conn, bids_dir)

    if not args.skip_nifti:
        ingest_nifti_meta(conn, bids_dir)
    else:
        print("  nifti_meta: SKIPPED (--skip-nifti)")

    ingest_events_meta(conn, bids_dir)
    ingest_physio_meta(conn, bids_dir)
    ingest_sourcedata(conn, bids_dir)
    ingest_derivatives(conn, bids_dir)
    ingest_session_metadata(conn, bids_dir)

    # Create useful indexes
    conn.executescript("""
        CREATE INDEX IF NOT EXISTS idx_files_subject ON files(subject);
        CREATE INDEX IF NOT EXISTS idx_files_session ON files(subject, session);
        CREATE INDEX IF NOT EXISTS idx_files_task ON files(task);
        CREATE INDEX IF NOT EXISTS idx_files_suffix ON files(suffix);
        CREATE INDEX IF NOT EXISTS idx_files_datatype ON files(datatype);
        CREATE INDEX IF NOT EXISTS idx_deriv_pipeline ON derivatives(pipeline);
        CREATE INDEX IF NOT EXISTS idx_deriv_subject ON derivatives(subject, session);
        CREATE INDEX IF NOT EXISTS idx_source_subject ON sourcedata(subject, session);
        CREATE INDEX IF NOT EXISTS idx_session_subject ON session_metadata(subject);
    """)
    conn.commit()

    print_summary(conn)

    # Export markdown summary
    export_summary(conn, summary_path)

    conn.close()

    db_size = db_path.stat().st_size / 1024
    print(f"\nDone. Database size: {db_size:.0f} KB")


if __name__ == "__main__":
    main()
