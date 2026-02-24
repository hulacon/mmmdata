#!/usr/bin/env python3
"""Generate BIDS sessions.tsv files from the scan log.

Reads mmm_scanlog.xlsx and merges:
  - BySession sheet (scan dates, equipment, notes)
  - scan_questionaire sheet (session-level questionnaire responses)
  - Pipeline exception registry (compiled from plan docs + config JSONs)

Outputs per-subject sessions.tsv files to sourcedata/sub-XX/.
These become the canonical source of truth for session-level metadata
and are copied to the BIDS root during BIDSification (with pipeline-only
columns stripped).

Usage:
    python generate_sessions_tsv.py
"""

from pathlib import Path

import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────

BIDS_ROOT = Path("/gpfs/projects/hulacon/shared/mmmdata")
SCANLOG = BIDS_ROOT / "sourcedata/shared/scan_logs/mmm_scanlog.xlsx"
OUTDIR = BIDS_ROOT / "sourcedata"

# ── Subject / session mapping ────────────────────────────────────────────────

SUBJECT_MAP = {"mmm_03": "sub-03", "mmm_04": "sub-04", "mmm_05": "sub-05"}

# scan_questionaire tab uses a different ID format
QUESTIONNAIRE_SUBJECT_MAP = {
    "Sub03": "sub-03",
    "Sub04": "sub-04",
    "Sub05": "sub-05",
}

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
            "2 fmap groups; scan stopped during tone (bathroom break)"
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
    qdf["participant_id"] = qdf.iloc[:, 0].map(QUESTIONNAIRE_SUBJECT_MAP)
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


def main() -> None:
    # ── Read BySession sheet ─────────────────────────────────────────────
    df = pd.read_excel(SCANLOG, sheet_name="BySession")

    # Filter to complete subjects, drop protocol-reminder rows
    df = df[df["Subject ID"].isin(SUBJECT_MAP)].copy()

    # Map subject IDs
    df["participant_id"] = df["Subject ID"].map(SUBJECT_MAP)

    # Map session IDs
    df["session_id"] = df["Session #"].apply(lambda n: f"ses-{int(n):02d}")

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
    qdf = _read_questionnaire(SCANLOG)
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
        "dcm2bids_exception",
        "behavioral_exception",
        "physio_exception",
        "exception_verified",
    ]

    # Write per-subject TSV files
    for subj in sorted(df["participant_id"].unique()):
        subj_df = (
            df[df["participant_id"] == subj][out_cols]
            .sort_values("session_id")
            .reset_index(drop=True)
        )
        outpath = OUTDIR / subj / f"{subj}_sessions.tsv"
        subj_df.to_csv(outpath, sep="\t", index=False)
        print(f"Wrote {outpath} ({len(subj_df)} sessions)")

    print("\nDone. Remember to copy sessions.json sidecar alongside the TSVs.")


if __name__ == "__main__":
    main()
