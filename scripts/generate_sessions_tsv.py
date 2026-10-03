#!/usr/bin/env python3
"""Generate BIDS sessions.tsv files from the scan log.

Reads mmm_scanlog.xlsx and merges:
  - BySession sheet (scan dates, equipment, notes)
  - scan_questionaire sheet (session-level questionnaire responses)
  - Pipeline exception registry (compiled from plan docs + config JSONs)

Outputs per-subject sessions.tsv files to <source_dir>/sub-XX/ (every
column, every logged session). With --bids, also writes the BIDS-root copy
<bids_root>/sub-XX/sub-XX_sessions.tsv: pipeline-only columns stripped, and
only sessions that exist in the BIDS tree (an aborted session the log numbers
31, with no data, stays out).

The scan log is the lab's live Google Sheet, exported to xlsx; a stale export
is the failure to watch for, so the latest scan date per subject is printed.

Usage:
    python generate_sessions_tsv.py [--scanlog PATH] [--subjects sub-##,...] [--bids]
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src" / "python") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src" / "python"))
from core.config import load_config  # noqa: E402

_PATHS = load_config(config_dir=_REPO_ROOT / "config")["paths"]
BIDS_ROOT = Path(_PATHS["bids_project_dir"])
OUTDIR = Path(_PATHS["source_dir"])
SCANLOG = OUTDIR / "shared" / "scan_logs" / "mmm_scanlog.xlsx"

# Columns that track the pipeline, not the session; kept in the source-side
# copy, stripped from the BIDS-root copy.
PIPELINE_COLS = [
    "dcm2bids_exception",
    "behavioral_exception",
    "physio_exception",
    "exception_verified",
]

# ── Subject / session mapping ────────────────────────────────────────────────

# BySession writes "mmm_##"; scan_questionaire writes "Sub##". Pilots
# ("mmm-test_##") and note rows match neither.
_BYSESSION_ID = re.compile(r"^mmm_(\d+)$")
_QUESTIONNAIRE_ID = re.compile(r"^Sub(\d+)$")


def _map_id(value, pattern: re.Pattern) -> str | None:
    m = pattern.match(str(value).strip())
    return f"sub-{int(m.group(1)):02d}" if m else None

# Questionnaire columns: (Excel header substring, TSV column name)
QUESTIONNAIRE_COLS = [
    ("Start time of experiment", "experiment_start_time"),
    ("Are you currently taking medications", "medications"),
    ("How many hours did you sleep last night", "sleep_hours"),
    ("how many hours of sleep did you average", "sleep_hours_average"),
    ("How well did you sleep last night", "sleep_quality"),
    ("Have you had caffeine", "caffeine_3h"),
    ("How is your mood today", "mood"),
    ("How hungry are you", "hunger"),
    ("What is your stress level", "stress"),
    ("How comfortable were you", "comfort_post_scan"),
    ("Feedback from subject", "subject_feedback"),
    ("Subject behavioral performance", "performance_note"),
    ("Notes (data incomplete", "session_note"),
]

SESSION_TYPE_MAP = {
    "structural": "anat",
    "localizer1": "localizer",
    "localizer2": "localizer",
    "cued_recall": "cued_recall",
    "free_recall": "free_recall",
    "Final Free Recall": "final_free_recall",
    "Final Cued Recall": "final",
}

# ── Exception registry ──────────────────────────────────────────────────────
#
# Compiled from all available sources. Keys are (subject, session) tuples.
# Each value is a dict with optional keys: dcm2bids, behavioral, physio,
# verified.  Missing keys default to empty / "n/a".

EXCEPTIONS: dict[tuple[str, str], dict[str, str]] = {
    # ═══ SUB-03 ══════════════════════════════════════════════════════════════
    ("sub-03", "ses-02"): {
        "dcm2bids": (
            "custom task list: prf (3 runs) + auditory + tone; "
            "2 fmap groups (first, second)"
        ),
        "verified": "false",
    },
    ("sub-03", "ses-03"): {
        "dcm2bids": (
            "custom task list: floc (6 runs) + prf (3 runs) + tone; "
            "3 fmap groups"
        ),
        "verified": "false",
    },
    ("sub-03", "ses-07"): {
        "dcm2bids": (
            "check for extra DICOM series from restart; "
            "LCNI name mismatch (labeled MMM03_sess04CR)"
        ),
        "verified": "false",
    },
    ("sub-03", "ses-10"): {
        "dcm2bids": (
            "3 fmap groups (re-entry after encoding); "
            "explicit series numbers in overrides.toml"
        ),
        "verified": "false",
    },
    ("sub-03", "ses-11"): {
        "dcm2bids": "check for extra DICOM series from retrieval restart",
        "verified": "false",
    },
    ("sub-03", "ses-12"): {
        "dcm2bids": "check for extra DICOM series from retrieval restart",
        "verified": "false",
    },
    ("sub-03", "ses-13"): {
        "dcm2bids": "LCNI name mismatch (labeled MMM_03_sess10); verify DICOM headers",
        "verified": "false",
    },
    ("sub-03", "ses-18"): {
        "dcm2bids": "extra AP fieldmap reported; may have been deleted at scan time",
        "verified": "false",
    },
    ("sub-03", "ses-19"): {
        "behavioral": "voice recording didn't save; no audio WAV for this session",
    },
    ("sub-03", "ses-20"): {
        "dcm2bids": "extra scout + AP/PA pair; use second set, skip first",
        "behavioral": "recording saved only first 12MB; use voice memos backup",
        "verified": "false",
    },
    ("sub-03", "ses-21"): {
        "behavioral": "voice recording issue; use voice memos backup",
    },
    ("sub-03", "ses-24"): {
        "dcm2bids": (
            "check for extra DICOM series from restarts "
            "(volume crash; movie run 1 + recall run restarted)"
        ),
        "behavioral": "check for duplicate run files from restarts",
        "verified": "false",
    },
    ("sub-03", "ses-28"): {
        "dcm2bids": "extra anatomical scans at end (1 MPRAGE + 1 Diff)",
        "verified": "false",
    },
    ("sub-03", "ses-30"): {
        "dcm2bids": (
            "SeriesNumber disambiguation on FINretrieval runs 1-2 "
            "(original run-1 discarded; 'DON'T USE ANY OF THE RUN 1s'); "
            "auditory localizer run deleted"
        ),
        "behavioral": "timeline beh file: use re-run only (first run crashed and data not saved)",
        "physio": "skip PhysioLog for discarded run-1 series",
        "verified": "false",
    },
    # ═══ SUB-04 ══════════════════════════════════════════════════════════════
    ("sub-04", "ses-02"): {
        "dcm2bids": (
            "custom task list: prf (3 runs) + auditory + tone; "
            "2 fmap groups"
        ),
        "verified": "false",
    },
    ("sub-04", "ses-03"): {
        "dcm2bids": (
            "fLOC failed; custom task list: prf (3 runs) + tone only; "
            "2 fmap groups; LCNI subject mismatch "
            "(labeled MMM_003_sess03, actually sub-04)"
        ),
        "verified": "false",
    },
    ("sub-04", "ses-04"): {
        "dcm2bids": (
            "6 makeup fLOC runs prepended (from ses-03 failure); "
            "fLOC split across 2 fmap groups + standard TB tasks"
        ),
        "behavioral": "no math events file (math not scanned this session)",
        "verified": "false",
    },
    ("sub-04", "ses-05"): {
        "behavioral": "no math events file (math not scanned this session)",
    },
    ("sub-04", "ses-08"): {
        "dcm2bids": "encoding run 3 ran repeated; check for extra DICOM series",
        "verified": "false",
    },
    ("sub-04", "ses-09"): {
        "dcm2bids": "LCNI name note (labeled MMM_04_sess06CR); verify DICOM headers",
        "verified": "false",
    },
    ("sub-04", "ses-11"): {
        "dcm2bids": "encoding restart (voice muffled); check for extra DICOM series",
        "verified": "false",
    },
    ("sub-04", "ses-14"): {
        "dcm2bids": "encoding run 2 restarted; check for extra DICOM series",
        "verified": "false",
    },
    ("sub-04", "ses-16"): {
        "physio": "no respiratory data (battery dead); pulse oximetry still recorded",
    },
    ("sub-04", "ses-20"): {
        "dcm2bids": "computer crashed mid-session; check for extra/incomplete DICOM series",
        "behavioral": "audio recording on voice memos, not scan computer",
        "verified": "false",
    },
    ("sub-04", "ses-22"): {
        "behavioral": "EDF naming anomaly: s4s4s1r (mistyped, actually encoding run 1); handled in inventory",
    },
    ("sub-04", "ses-24"): {
        "behavioral": "EDF naming anomaly: s4s6r1 (missing phase letter, actually retrieval); handled in inventory",
    },
    ("sub-04", "ses-25"): {
        "behavioral": "3 .EDF.tmp files (incomplete eye tracking recordings); skipped in inventory",
    },
    ("sub-04", "ses-28"): {
        "dcm2bids": "extra anatomical scans at end (1 MPRAGE + 1 Diff)",
        "verified": "false",
    },
    ("sub-04", "ses-30"): {
        "dcm2bids": "single fmap group; SeriesDescription matching strategy",
        "verified": "false",
    },
    # ═══ SUB-05 ══════════════════════════════════════════════════════════════
    ("sub-05", "ses-02"): {
        "dcm2bids": (
            "custom task list: prf (3 runs) + floc (3 runs) + tone; "
            "2 fmap groups; scan stopped during tone (bathroom break); "
            "floc_run1 aborted after 5 volumes (Series 27) and restarted "
            "(Series 30) — requires run_series override to skip abort"
        ),
        "verified": "false",
    },
    ("sub-05", "ses-03"): {
        "dcm2bids": (
            "custom task list: prf (3 runs) + floc (3 runs) + tone (2 runs) + auditory; "
            "3 fmap groups (re-entry between tasks)"
        ),
        "verified": "false",
    },
    ("sub-05", "ses-05"): {
        "dcm2bids": (
            "re-entry after retrieval run 3 (bathroom break); "
            "extra fieldmap set"
        ),
        "verified": "false",
    },
    ("sub-05", "ses-06"): {
        "dcm2bids": "retrieval runs 2 and 4 restarted; check for extra DICOM series",
        "verified": "false",
    },
    ("sub-05", "ses-08"): {
        "dcm2bids": "retrieval run 3 restarted; check for extra DICOM series",
        "verified": "false",
    },
    ("sub-05", "ses-09"): {
        "dcm2bids": "encoding run 2 stopped after ~30s (poor audio), restarted; check for extra DICOM series",
        "verified": "false",
    },
    ("sub-05", "ses-10"): {
        "dcm2bids": (
            "accidentally ran fixation (stopped after 30s); "
            "encoding run 2 restarted; check for extra DICOM series"
        ),
        "verified": "false",
    },
    ("sub-05", "ses-12"): {
        "dcm2bids": "retrieval restart (words muffled); check for extra DICOM series",
        "verified": "false",
    },
    ("sub-05", "ses-13"): {
        "dcm2bids": "encoding run 1 restarted; check for extra DICOM series",
        "verified": "false",
    },
    ("sub-05", "ses-17"): {
        "dcm2bids": "disregard first AP fieldmap; explicit fmap series numbers in overrides.toml",
        "verified": "false",
    },
    ("sub-05", "ses-18"): {
        "dcm2bids": "retrieval run 1 restarted; check for extra DICOM series",
        "verified": "false",
    },
    ("sub-05", "ses-19"): {
        "dcm2bids": (
            "2-3 non-usable encoding runs (volume issues); "
            "4th encoding run is good; scouts redone; bathroom break"
        ),
        "verified": "false",
    },
    ("sub-05", "ses-20"): {
        "behavioral": "forgot to switch persaio and add mic adaptor; audio may be affected",
    },
    ("sub-05", "ses-23"): {
        "physio": "respiration not working this session; respiratory channel missing or unusable",
    },
    ("sub-05", "ses-24"): {
        "dcm2bids": "restroom break; reran anatomical and fieldmaps; check for extra series",
        "verified": "false",
    },
    ("sub-05", "ses-26"): {
        "dcm2bids": "headphones not correctly positioned; scouts restarted; check for extra scout series",
        "verified": "false",
    },
    ("sub-05", "ses-28"): {
        "dcm2bids": (
            "T1 MPRAGE not working; used alternative protocol (mprage_p2 from Kuhl lab); "
            "4 setters run, only last one accurate; verify correct series"
        ),
        "verified": "false",
    },
    ("sub-05", "ses-30"): {
        "dcm2bids": (
            "2 fmap groups (re-scout before retrieval run 3 due to head pain); "
            "hybrid fmap matching (SeriesDescription + SeriesNumber); "
            "screen glitches during auditory (audio still recorded)"
        ),
        "physio": "extra PhysioLog for fixation + run 3 deleted at scan time",
        "verified": "false",
    },
}


def _resolve_questionnaire_col(headers: list[str], prefix: str) -> str | None:
    """Find the Excel column header matching a prefix substring."""
    for h in headers:
        if h and prefix in h:
            return h
    return None


def _read_questionnaire(scanlog_path: Path) -> pd.DataFrame:
    """Read scan_questionaire sheet and normalize to (participant_id, date)."""
    qdf = pd.read_excel(scanlog_path, sheet_name="scan_questionaire")

    # Drop rows with no subject ID (trailing empties)
    qdf = qdf.dropna(subset=[qdf.columns[0]]).copy()

    # Map subject IDs
    qdf["participant_id"] = qdf.iloc[:, 0].map(lambda v: _map_id(v, _QUESTIONNAIRE_ID))
    qdf = qdf[qdf["participant_id"].notna()].copy()

    # Normalize date to date-only for joining
    qdf["_join_date"] = pd.to_datetime(qdf.iloc[:, 2]).dt.normalize()

    # Rename questionnaire columns to BIDS-friendly names
    headers = list(qdf.columns)
    rename = {}
    for prefix, bids_name in QUESTIONNAIRE_COLS:
        src = _resolve_questionnaire_col(headers, prefix)
        if src and src != bids_name:
            rename[src] = bids_name
    qdf = qdf.rename(columns=rename)

    # Format experiment_start_time as HH:MM string
    if "experiment_start_time" in qdf.columns:
        def _fmt_time(v):
            if pd.isna(v) or v is None:
                return "n/a"
            if hasattr(v, "strftime"):
                return v.strftime("%H:%M")
            return str(v)
        qdf["experiment_start_time"] = qdf["experiment_start_time"].apply(_fmt_time)

    # Keep only the columns we need
    keep = ["participant_id", "_join_date"] + [
        c for _, c in QUESTIONNAIRE_COLS if c in qdf.columns
    ]
    return qdf[keep]


def _default_subjects() -> list[str]:
    """Subjects with a directory in the BIDS root: excludes pilots and dropped subjects."""
    return sorted(p.name for p in BIDS_ROOT.glob("sub-*") if p.is_dir())


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scanlog", type=Path, default=SCANLOG,
                    help=f"xlsx export of the scan log (default: {SCANLOG})")
    ap.add_argument("--subjects", default=None,
                    help="Comma-separated, e.g. sub-06,sub-07 (default: every sub-* in the BIDS root)")
    ap.add_argument("--bids", action="store_true",
                    help="Also write the BIDS-root copy (pipeline columns stripped, "
                         "sessions present in the tree only)")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Write under this directory instead (DIR/sub-XX/ and, with --bids, "
                         "DIR/bids/sub-XX/) -- for comparing against the live files")
    args = ap.parse_args(argv)
    src_out = args.out_dir or OUTDIR
    bids_out = (args.out_dir / "bids") if args.out_dir else BIDS_ROOT
    subjects = args.subjects.split(",") if args.subjects else _default_subjects()

    # ── Read BySession sheet ─────────────────────────────────────────────
    df = pd.read_excel(args.scanlog, sheet_name="BySession")

    # Map subject IDs; drops pilots and protocol-reminder rows
    df["participant_id"] = df["Subject ID"].map(lambda v: _map_id(v, _BYSESSION_ID))
    df = df[df["participant_id"].isin(subjects)].copy()
    missing = sorted(set(subjects) - set(df["participant_id"]))
    if missing:
        raise SystemExit(f"{args.scanlog}: no BySession rows for {missing}; is the export stale?")

    # Map session IDs
    df["session_id"] = df["Session #"].apply(lambda n: f"ses-{int(n):02d}")

    # One row per (subject, session). The log is typed by hand; a repeated
    # session number is an entry error to correct at the source, never to
    # resolve here by picking one.
    dup = df[df.duplicated(["participant_id", "session_id"], keep=False)]
    if not dup.empty:
        raise SystemExit(
            f"{args.scanlog}: the same session is logged twice -- correct the scan log:\n"
            + dup[["Subject ID", "Session #", "Scan date", "Scan start time"]].to_string()
        )
    latest = df.groupby("participant_id")["Scan date"].max()
    print("Latest scan date per subject:", ", ".join(
        f"{s} {pd.Timestamp(d).date()}" for s, d in latest.items()))

    # Map session types
    df["session_type"] = df["Scan session name"].map(SESSION_TYPE_MAP)

    # Format date
    df["acq_time"] = pd.to_datetime(df["Scan date"]).dt.strftime("%Y-%m-%d")

    # Join key for questionnaire merge
    df["_join_date"] = pd.to_datetime(df["Scan date"]).dt.normalize()

    # Boolean equipment columns
    df["earbud_used"] = df["Earbud used?"].apply(
        lambda x: "true" if x == 1 else "false"
    )
    df["physio_used"] = df["Physio (pulse, resp) used?"].apply(
        lambda x: "true" if x == 1 else "false"
    )
    df["eyetracking_used"] = df["Eyetracking used?"].apply(
        lambda x: "true" if x == 1 else "false"
    )

    # Scan note — clean up whitespace, replace NaN with empty
    df["scan_note"] = (
        df["Note"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.replace("\n", " ")
        .str.replace("\r", "")
    )

    # ── Merge questionnaire data ─────────────────────────────────────────
    qdf = _read_questionnaire(args.scanlog)
    q_cols = [c for _, c in QUESTIONNAIRE_COLS if c in qdf.columns]

    df = df.merge(
        qdf, on=["participant_id", "_join_date"], how="left", suffixes=("", "_q")
    )
    df.drop(columns=["_join_date"], inplace=True)

    # Fill missing questionnaire values with n/a (ses-01 through ses-03)
    # and clean up numeric formatting (4.0 → 4 for integer-valued floats)
    def _clean_val(v):
        if pd.isna(v) or v is None:
            return "n/a"
        if isinstance(v, float) and v == int(v):
            return str(int(v))
        s = str(v).strip()
        if s in ("", "None", "nan"):
            return "n/a"
        return s

    for col in q_cols:
        df[col] = df[col].apply(_clean_val)

    # Add exception columns from registry
    for col in [
        "dcm2bids_exception",
        "behavioral_exception",
        "physio_exception",
        "exception_verified",
    ]:
        df[col] = ""

    for idx, row in df.iterrows():
        key = (row["participant_id"], row["session_id"])
        exc = EXCEPTIONS.get(key, {})
        df.at[idx, "dcm2bids_exception"] = exc.get("dcm2bids", "")
        df.at[idx, "behavioral_exception"] = exc.get("behavioral", "")
        df.at[idx, "physio_exception"] = exc.get("physio", "")
        has_exception = any(
            exc.get(k) for k in ("dcm2bids", "behavioral", "physio")
        )
        df.at[idx, "exception_verified"] = (
            exc.get("verified", "n/a") if has_exception else "n/a"
        )

    # Select and order output columns — questionnaire columns between
    # equipment flags and pipeline-tracking columns (pipeline columns get
    # stripped when copying to BIDS root)
    out_cols = [
        "session_id",
        "acq_time",
        "session_type",
        "earbud_used",
        "physio_used",
        "eyetracking_used",
        *q_cols,
        "scan_note",
        *PIPELINE_COLS,
    ]

    # Write per-subject TSV files
    for subj in sorted(df["participant_id"].unique()):
        subj_df = (
            df[df["participant_id"] == subj][out_cols]
            .sort_values("session_id")
            .reset_index(drop=True)
        )
        outpath = src_out / subj / f"{subj}_sessions.tsv"
        outpath.parent.mkdir(parents=True, exist_ok=True)
        subj_df.to_csv(outpath, sep="\t", index=False, lineterminator="\n")
        print(f"Wrote {outpath} ({len(subj_df)} sessions)")

        if args.bids:
            present = {p.name for p in (BIDS_ROOT / subj).glob("ses-*") if p.is_dir()}
            bids_df = subj_df[subj_df["session_id"].isin(present)].drop(columns=PIPELINE_COLS)
            left_out = sorted(set(subj_df["session_id"]) - present)
            bids_path = bids_out / subj / f"{subj}_sessions.tsv"
            bids_path.parent.mkdir(parents=True, exist_ok=True)
            bids_df.to_csv(bids_path, sep="\t", index=False, lineterminator="\n")
            print(f"Wrote {bids_path} ({len(bids_df)} sessions"
                  + (f"; not in the tree: {', '.join(left_out)})" if left_out else ")"))


if __name__ == "__main__":
    main()
