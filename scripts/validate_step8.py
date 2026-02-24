#!/usr/bin/env python3
"""Step 8 validation: compare new BIDS output against reference in derivatives/bids_validation/.

Walks the reference directory, maps each file to its expected new BIDS output
path, and compares data values. Produces a TSV report and terminal summary.

Categories:
  A — Same column structure (full comparison)
  A_OFFSET — Same columns but known scanner-reference offset on timing columns
             (onset offset computed per-file and subtracted before comparison)
  B — Shared column subset (compare overlapping columns)
  C — Different structure (row count only, flag for manual review)
  D — No reference (new output files with no counterpart)

Usage:
    python validate_step8.py [--verbose]
"""

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# === Paths ===
BIDS_ROOT = Path("/gpfs/projects/hulacon/shared/mmmdata")
REF_DIR = BIDS_ROOT / "derivatives" / "bids_validation" / "eventfiles"
REPORT_PATH = Path(__file__).parent / "validation_report.tsv"
SUBJECTS = ["sub-03", "sub-04", "sub-05"]

# === Tolerances ===
TIMING_TOL = 0.001  # seconds — float precision only

# === Columns to skip in comparison (known-different) ===
SKIP_COLS = {"session", "run", "subject"}

# === Categorical columns (exact match when present) ===
CATEGORICAL = {"trial_type", "word", "pairId", "mmmId", "trial_id", "modality",
               "voice", "voiceId", "enCon", "reCon", "nsdId", "itmno",
               "sharedId", "cueId", "correct_resp", "correct_answer",
               "trial_accuracy", "recog", "image1", "image2", "problem",
               "condition", "style", "movie_name", "free_recall_position"}

# === Timing columns (numeric tolerance) ===
TIMING = {"onset", "duration", "onset_actual", "duration_actual", "resp_RT",
          "response_time", "timeline_RT"}

# === Response columns (exact match, but n/a ↔ empty equivalence) ===
RESPONSE = {"resp", "response", "timeline_resp", "accuracy"}

# === Columns subject to scanner-reference offset (A_OFFSET category) ===
# These columns contain absolute timestamps relative to scanner start.
# The reference uses movie_instructions.stopped / recall_instructions.stopped;
# our converters use use_row.started. Both valid, but produce a constant offset.
OFFSET_TIMING_COLS = {"onset", "movie_title.started", "movie_title.stopped",
                      "recall1.started", "recall1.stopped"}


def ref_to_bids_path(ref_path):
    """Map a reference file path to the expected new BIDS output path.

    Returns (bids_path, category) or (None, "SKIP").
    """
    rel = ref_path.relative_to(REF_DIR)
    sub = rel.parts[0]  # sub-XX
    ses = rel.parts[1]  # ses-YY
    fname = rel.parts[2]

    # Parse task and run from filename
    m = re.match(
        r"(sub-\d+)_(ses-\d+)_task-(\w+?)(?:_run-(\d+))?_(events)\.(tsv)",
        fname,
    )
    if not m:
        return None, "SKIP"

    task = m.group(3)
    run = m.group(4)  # may be None

    # --- Mapping rules ---

    # Fixation: no events file in BIDS output
    if task == "fixation":
        return None, "SKIP"

    # Math tasks: drop _run-01
    if task in ("TBmath", "NATmath"):
        new_fname = f"{sub}_{ses}_task-{task}_events.tsv"
        return BIDS_ROOT / sub / ses / "func" / new_fname, "A"

    # Auditory localizer: rename task, drop run
    if task == "auditorylocalizer":
        new_fname = f"{sub}_{ses}_task-auditory_events.tsv"
        return BIDS_ROOT / sub / ses / "func" / new_fname, "A"

    # Motor localizer: rename task, keep run
    if task == "motorlocalizer":
        run_str = f"_run-{run}" if run else ""
        new_fname = f"{sub}_{ses}_task-motor{run_str}_events.tsv"
        return BIDS_ROOT / sub / ses / "func" / new_fname, "B"

    # TB2AFC → beh directory, _beh.tsv suffix, KEEP run
    if task == "TB2AFC":
        run_str = f"_run-{run}" if run else ""
        new_fname = f"{sub}_{ses}_task-{task}{run_str}_beh.tsv"
        return BIDS_ROOT / sub / ses / "beh" / new_fname, "B"

    # FIN2AFC and FINtimeline → beh directory, _beh.tsv suffix, DROP run
    if task in ("FIN2AFC", "FINtimeline"):
        new_fname = f"{sub}_{ses}_task-{task}_beh.tsv"
        return BIDS_ROOT / sub / ses / "beh" / new_fname, "B"

    # FINretrieval: same filename but different column structure
    if task == "FINretrieval":
        run_str = f"_run-{run}" if run else ""
        new_fname = f"{sub}_{ses}_task-{task}{run_str}_events.tsv"
        return BIDS_ROOT / sub / ses / "func" / new_fname, "B"

    # NATencoding, NATretrieval: same columns but different scanner reference
    if task in ("NATencoding", "NATretrieval"):
        run_str = f"_run-{run}" if run else ""
        new_fname = f"{sub}_{ses}_task-{task}{run_str}_events.tsv"
        return BIDS_ROOT / sub / ses / "func" / new_fname, "A_OFFSET"

    # Everything else (TBencoding, TBretrieval): direct mapping
    run_str = f"_run-{run}" if run else ""
    new_fname = f"{sub}_{ses}_task-{task}{run_str}_events.tsv"
    return BIDS_ROOT / sub / ses / "func" / new_fname, "A"


def normalize_val(val):
    """Normalize a cell value for comparison: treat empty, NaN, 'n/a' as equivalent."""
    if pd.isna(val):
        return "n/a"
    s = str(val).strip()
    if s in ("", "nan", "NaN", "None"):
        return "n/a"
    return s


def compare_column(ref_vals, new_vals, col_name):
    """Compare a single column. Returns (status, detail_str)."""
    n = len(ref_vals)
    if n != len(new_vals):
        return "ROW_COUNT", f"ref={n}, new={len(new_vals)}"

    # Decide comparison mode
    if col_name in TIMING:
        return _compare_numeric(ref_vals, new_vals, col_name, TIMING_TOL)
    elif col_name in CATEGORICAL or col_name in RESPONSE:
        return _compare_exact(ref_vals, new_vals, col_name)
    else:
        # Default: try numeric first, fall back to exact
        try:
            ref_num = pd.to_numeric(ref_vals.map(normalize_val).replace("n/a", np.nan),
                                     errors="raise")
            new_num = pd.to_numeric(new_vals.map(normalize_val).replace("n/a", np.nan),
                                     errors="raise")
            return _compare_numeric_series(ref_num, new_num, col_name, TIMING_TOL)
        except (ValueError, TypeError):
            return _compare_exact(ref_vals, new_vals, col_name)


def _compare_numeric(ref_vals, new_vals, col_name, tol):
    """Compare numeric column with tolerance."""
    ref_norm = ref_vals.map(normalize_val).replace("n/a", np.nan)
    new_norm = new_vals.map(normalize_val).replace("n/a", np.nan)
    ref_num = pd.to_numeric(ref_norm, errors="coerce")
    new_num = pd.to_numeric(new_norm, errors="coerce")
    return _compare_numeric_series(ref_num, new_num, col_name, tol)


def _compare_numeric_series(ref_num, new_num, col_name, tol):
    """Compare two numeric series with tolerance."""
    both_valid = ref_num.notna() & new_num.notna()
    both_na = ref_num.isna() & new_num.isna()
    one_na = (ref_num.isna() ^ new_num.isna())

    na_mismatches = one_na.sum()
    if both_valid.any():
        diffs = (ref_num[both_valid] - new_num[both_valid]).abs()
        max_diff = diffs.max()
        n_exceed = (diffs > tol).sum()
    else:
        max_diff = 0.0
        n_exceed = 0

    if n_exceed == 0 and na_mismatches == 0:
        return "MATCH", f"max_diff={max_diff:.6f}"
    issues = []
    if n_exceed > 0:
        issues.append(f"{n_exceed} vals exceed tol (max={max_diff:.6f})")
    if na_mismatches > 0:
        issues.append(f"{na_mismatches} n/a mismatches")
    return "MISMATCH", "; ".join(issues)


def _compare_exact(ref_vals, new_vals, col_name):
    """Compare column values exactly (with n/a normalization)."""
    ref_norm = ref_vals.map(normalize_val)
    new_norm = new_vals.map(normalize_val)
    mismatches = (ref_norm != new_norm).sum()
    if mismatches == 0:
        return "MATCH", ""
    # Sample first few mismatches
    idx = ref_norm != new_norm
    samples = []
    for i in idx[idx].index[:3]:
        samples.append(f"row {i}: ref='{ref_norm[i]}' new='{new_norm[i]}'")
    return "MISMATCH", f"{mismatches} mismatches; " + "; ".join(samples)


def compare_files(ref_path, new_path, category, verbose=False):
    """Compare reference and new BIDS output files.

    Returns a result dict.
    """
    result = {
        "reference": str(ref_path),
        "new_output": str(new_path),
        "category": category,
        "status": "MATCH",
        "issues": [],
    }

    try:
        ref_df = pd.read_csv(ref_path, sep="\t", dtype=str)
        new_df = pd.read_csv(new_path, sep="\t", dtype=str)
    except Exception as e:
        result["status"] = "ERROR"
        result["issues"].append(f"Read error: {e}")
        return result

    # Category C: always manual review (different converter approach)
    if category == "C":
        result["status"] = "MANUAL_REVIEW"
        result["issues"].append(f"Row count: ref={len(ref_df)}, new={len(new_df)}")
        result["issues"].append(f"ref_cols={sorted(ref_df.columns)}")
        result["issues"].append(f"new_cols={sorted(new_df.columns)}")
        return result

    # Row count
    if len(ref_df) != len(new_df):
        result["status"] = "MISMATCH"
        result["issues"].append(f"Row count: ref={len(ref_df)}, new={len(new_df)}")
        return result

    # Determine which columns to compare
    ref_cols = set(ref_df.columns)
    new_cols = set(new_df.columns)
    compare_cols = ref_cols & new_cols - SKIP_COLS

    extra_in_ref = ref_cols - new_cols - SKIP_COLS
    extra_in_new = new_cols - ref_cols - SKIP_COLS
    if extra_in_ref:
        result["issues"].append(f"Cols only in ref: {sorted(extra_in_ref)}")
    if extra_in_new:
        result["issues"].append(f"Cols only in new: {sorted(extra_in_new)}")

    # For A_OFFSET: compute scanner-reference offset from first onset row
    onset_offset = 0.0
    if category == "A_OFFSET" and "onset" in compare_cols:
        try:
            ref_onset0 = pd.to_numeric(ref_df["onset"], errors="coerce").dropna().iloc[0]
            new_onset0 = pd.to_numeric(new_df["onset"], errors="coerce").dropna().iloc[0]
            onset_offset = float(new_onset0 - ref_onset0)
            result["issues"].append(f"Scanner-ref offset: {onset_offset:.3f}s")
        except (IndexError, ValueError):
            result["issues"].append("Could not compute onset offset")

    # Compare each shared column
    col_mismatches = []
    for col in sorted(compare_cols):
        if col in SKIP_COLS:
            continue

        ref_col = ref_df[col]
        new_col = new_df[col]

        # For A_OFFSET: adjust offset-timing columns before comparison
        if category == "A_OFFSET" and col in OFFSET_TIMING_COLS and onset_offset != 0.0:
            new_num = pd.to_numeric(new_col.map(normalize_val).replace("n/a", np.nan),
                                    errors="coerce")
            adjusted = new_num - onset_offset
            # Convert back to string series for comparison
            new_col = adjusted.map(lambda x: str(x) if pd.notna(x) else "n/a")

        status, detail = compare_column(ref_col, new_col, col)
        if status == "MISMATCH":
            col_mismatches.append(f"{col}: {detail}")

    if col_mismatches:
        result["status"] = "MISMATCH"
        result["issues"].extend(col_mismatches)

    return result


def find_all_new_output_files():
    """Find all non-physio TSV files in the new BIDS output."""
    files = set()
    for sub in SUBJECTS:
        for ses_dir in sorted((BIDS_ROOT / sub).iterdir()):
            if not ses_dir.is_dir() or not ses_dir.name.startswith("ses-"):
                continue
            for mod in ("func", "beh"):
                mod_dir = ses_dir / mod
                if not mod_dir.is_dir():
                    continue
                for f in mod_dir.iterdir():
                    if f.suffix == ".tsv" and "physio" not in f.name:
                        files.add(f)
    return files


def main():
    parser = argparse.ArgumentParser(description="Step 8 validation")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # --- Discover reference files ---
    ref_files = sorted(REF_DIR.rglob("*_events.tsv"))
    print(f"Found {len(ref_files)} reference files")

    # --- Compare each reference against new output ---
    results = []
    matched_new = set()  # track which new files were matched

    for ref_path in ref_files:
        new_path, category = ref_to_bids_path(ref_path)

        if category == "SKIP":
            results.append({
                "reference": str(ref_path),
                "new_output": "",
                "category": "SKIP",
                "status": "SKIPPED",
                "issues": ["No BIDS events file expected (e.g. fixation)"],
            })
            continue

        if new_path is None or not new_path.exists():
            results.append({
                "reference": str(ref_path),
                "new_output": str(new_path) if new_path else "",
                "category": category,
                "status": "MISSING_NEW",
                "issues": ["New BIDS output file not found"],
            })
            continue

        matched_new.add(new_path)
        r = compare_files(ref_path, new_path, category, args.verbose)
        results.append(r)

    # --- Find new output files with no reference ---
    all_new = find_all_new_output_files()
    unmatched = sorted(all_new - matched_new)
    for f in unmatched:
        results.append({
            "reference": "",
            "new_output": str(f),
            "category": "D",
            "status": "NO_REFERENCE",
            "issues": ["No reference file for comparison"],
        })

    # --- Write report TSV ---
    rows = []
    for r in results:
        rows.append({
            "reference": r["reference"],
            "new_output": r["new_output"],
            "category": r["category"],
            "status": r["status"],
            "issues": " | ".join(r["issues"]),
        })
    report_df = pd.DataFrame(rows)
    report_df.to_csv(REPORT_PATH, sep="\t", index=False)
    print(f"\nReport written to {REPORT_PATH}")

    # --- Print summary ---
    print(f"\n{'='*70}")
    print("VALIDATION SUMMARY")
    print(f"{'='*70}")

    counts = report_df["status"].value_counts()
    for status in ["MATCH", "MISMATCH", "MATCH_WITH_OFFSET", "MANUAL_REVIEW",
                    "MISSING_NEW", "NO_REFERENCE", "SKIPPED", "ERROR"]:
        if status in counts.index:
            print(f"  {status:20s} {counts[status]:4d}")
    print(f"  {'TOTAL':20s} {len(results):4d}")

    # Show mismatches
    mismatches = report_df[report_df["status"] == "MISMATCH"]
    if len(mismatches) > 0:
        print(f"\n{'='*70}")
        print("MISMATCHES")
        print(f"{'='*70}")
        for _, row in mismatches.iterrows():
            ref_name = Path(row["reference"]).name if row["reference"] else "?"
            new_name = Path(row["new_output"]).name if row["new_output"] else "?"
            print(f"\n  {ref_name}")
            print(f"  → {new_name}")
            for issue in row["issues"].split(" | "):
                print(f"    - {issue}")

    # Show missing
    missing = report_df[report_df["status"] == "MISSING_NEW"]
    if len(missing) > 0:
        print(f"\n{'='*70}")
        print("MISSING NEW FILES")
        print(f"{'='*70}")
        for _, row in missing.iterrows():
            ref_name = Path(row["reference"]).name if row["reference"] else "?"
            print(f"  {ref_name}")
            if row["new_output"]:
                print(f"    expected: {row['new_output']}")

    # Show no-reference summary
    no_ref = report_df[report_df["status"] == "NO_REFERENCE"]
    if len(no_ref) > 0:
        print(f"\n{'='*70}")
        print(f"NO REFERENCE ({len(no_ref)} files)")
        print(f"{'='*70}")
        # Group by task pattern
        task_counts = {}
        for _, row in no_ref.iterrows():
            fname = Path(row["new_output"]).name
            m = re.search(r"task-(\w+?)(?:_run-\d+)?_(?:events|beh)", fname)
            task = m.group(1) if m else "unknown"
            task_counts[task] = task_counts.get(task, 0) + 1
        for task, count in sorted(task_counts.items()):
            print(f"  task-{task}: {count} files")

    # Show manual review
    manual = report_df[report_df["status"] == "MANUAL_REVIEW"]
    if len(manual) > 0:
        print(f"\n{'='*70}")
        print(f"MANUAL REVIEW NEEDED ({len(manual)} files)")
        print(f"{'='*70}")
        for _, row in manual.iterrows():
            ref_name = Path(row["reference"]).name if row["reference"] else "?"
            print(f"  {ref_name}")

    return 1 if len(mismatches) > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
