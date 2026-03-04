#!/usr/bin/env python3
"""
Compare onset times between BIDS root events files and validation events files.

Systematically identifies onset time offsets between the two versions of events
files across all subjects, sessions, tasks, and runs.

The validation events are considered the ground truth.
"""

import os
import re
import csv
from pathlib import Path
from collections import defaultdict
import statistics

BIDS_ROOT = Path("/projects/hulacon/shared/mmmdata")
VALIDATION_DIR = BIDS_ROOT / "derivatives" / "bids_validation" / "eventfiles"

# All known tasks
TASKS = [
    "TBencoding", "TBretrieval", "TBmath", "TB2AFC",
    "NATencoding", "NATretrieval", "NATmath",
    "FINretrieval", "FIN2AFC", "FINtimeline",
    "auditorylocalizer", "motorlocalizer", "fixation",
]


def read_first_onset(filepath):
    """Read the onset value from the first data row of a TSV file."""
    try:
        with open(filepath, "r") as f:
            reader = csv.DictReader(f, delimiter="\t")
            row = next(reader)
            return float(row["onset"])
    except Exception as e:
        return None


def read_all_onsets(filepath):
    """Read all onset values from a TSV file."""
    onsets = []
    try:
        with open(filepath, "r") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                try:
                    onsets.append(float(row["onset"]))
                except (ValueError, KeyError):
                    pass
    except Exception:
        pass
    return onsets


def get_header(filepath):
    """Get the header (column names) of a TSV file."""
    try:
        with open(filepath, "r") as f:
            return f.readline().strip().split("\t")
    except Exception:
        return []


def get_nrows(filepath):
    """Count data rows in a TSV file."""
    try:
        with open(filepath, "r") as f:
            return sum(1 for _ in f) - 1  # subtract header
    except Exception:
        return 0


def find_matching_pairs():
    """
    Find all events file pairs between BIDS root and validation.

    Handles naming differences:
    - Some BIDS root files lack run numbers (e.g., task-TBmath_events.tsv)
      while validation has run-01 (task-TBmath_run-01_events.tsv)
    - Some files have matching names in both locations.
    """
    pairs = []
    unmatched_validation = []
    unmatched_bids = []

    # First, build index of all BIDS root events files
    bids_events = {}  # key: (sub, ses, task, run_or_none) -> filepath
    for sub_dir in sorted(BIDS_ROOT.glob("sub-*")):
        if not sub_dir.is_dir() or "derivatives" in str(sub_dir):
            continue
        sub = sub_dir.name
        for ses_dir in sorted(sub_dir.glob("ses-*")):
            ses = ses_dir.name
            func_dir = ses_dir / "func"
            if not func_dir.is_dir():
                continue
            for f in sorted(func_dir.glob("*_events.tsv")):
                fname = f.name
                # Parse task and run
                task_match = re.search(r"task-(\w+?)(?:_run-(\d+))?_events\.tsv", fname)
                if task_match:
                    task = task_match.group(1)
                    run = task_match.group(2)  # None if no run
                    bids_events[(sub, ses, task, run)] = f

    # Build index of all validation events files
    val_events = {}
    for sub_dir in sorted(VALIDATION_DIR.glob("sub-*")):
        sub = sub_dir.name
        for ses_dir in sorted(sub_dir.glob("ses-*")):
            ses = ses_dir.name
            for f in sorted(ses_dir.glob("*_events.tsv")):
                fname = f.name
                task_match = re.search(r"task-(\w+?)(?:_run-(\d+))?_events\.tsv", fname)
                if task_match:
                    task = task_match.group(1)
                    run = task_match.group(2)
                    val_events[(sub, ses, task, run)] = f

    # Match pairs
    matched_val_keys = set()
    for key, bids_path in sorted(bids_events.items()):
        sub, ses, task, run = key
        # Try exact match first
        if key in val_events:
            pairs.append((key, bids_path, val_events[key]))
            matched_val_keys.add(key)
        else:
            # If BIDS has no run number, try matching validation run-01
            if run is None:
                alt_key = (sub, ses, task, "01")
                if alt_key in val_events:
                    pairs.append((key, bids_path, val_events[alt_key]))
                    matched_val_keys.add(alt_key)
                else:
                    unmatched_bids.append((key, bids_path))
            else:
                # If BIDS has a run number but no match
                alt_key = (sub, ses, task, None)
                if alt_key in val_events:
                    pairs.append((key, bids_path, val_events[alt_key]))
                    matched_val_keys.add(alt_key)
                else:
                    unmatched_bids.append((key, bids_path))

    # Find unmatched validation files
    for key, val_path in sorted(val_events.items()):
        if key not in matched_val_keys:
            unmatched_validation.append((key, val_path))

    return pairs, unmatched_bids, unmatched_validation


def check_full_file_differences(bids_path, val_path, onset_offset):
    """
    For files with no onset offset, check if there are other differences.
    For files with onset offset, check if the offset is consistent across all rows.

    Returns a dict with difference info.
    """
    diffs = {}

    bids_onsets = read_all_onsets(bids_path)
    val_onsets = read_all_onsets(val_path)

    # Check row count
    bids_nrows = get_nrows(bids_path)
    val_nrows = get_nrows(val_path)
    if bids_nrows != val_nrows:
        diffs["row_count"] = f"BIDS={bids_nrows}, val={val_nrows}"

    # Check headers
    bids_header = get_header(bids_path)
    val_header = get_header(val_path)
    if bids_header != val_header:
        bids_only = set(bids_header) - set(val_header)
        val_only = set(val_header) - set(bids_header)
        header_diffs = []
        if bids_only:
            header_diffs.append(f"BIDS-only cols: {bids_only}")
        if val_only:
            header_diffs.append(f"val-only cols: {val_only}")
        # Check for renamed columns
        renamed = []
        for bc in bids_only:
            for vc in val_only:
                if bc.replace("_id", "").replace("_num", "") == vc.replace("_id", "").replace("_num", ""):
                    renamed.append(f"{bc} -> {vc}")
        if renamed:
            header_diffs.append(f"possibly renamed: {renamed}")
        diffs["header"] = "; ".join(header_diffs)

    # If there's an onset offset, check consistency across all rows
    if abs(onset_offset) > 0.01:
        min_len = min(len(bids_onsets), len(val_onsets))
        if min_len > 0:
            offsets = [val_onsets[i] - bids_onsets[i] for i in range(min_len)]
            offset_min = min(offsets)
            offset_max = max(offsets)
            offset_mean = statistics.mean(offsets)
            if offset_max - offset_min < 0.001:
                diffs["offset_consistency"] = f"CONSTANT offset across all {min_len} rows: {offset_mean:.6f}s"
            else:
                diffs["offset_consistency"] = (
                    f"VARIABLE offset across {min_len} rows: "
                    f"min={offset_min:.6f}, max={offset_max:.6f}, "
                    f"mean={offset_mean:.6f}, range={offset_max - offset_min:.6f}"
                )
    else:
        # No onset offset - check for other column value diffs
        # Compare a few data columns beyond onset
        try:
            with open(bids_path, "r") as bf, open(val_path, "r") as vf:
                breader = csv.DictReader(bf, delimiter="\t")
                vreader = csv.DictReader(vf, delimiter="\t")
                common_cols = set(breader.fieldnames or []) & set(vreader.fieldnames or [])
                non_onset_diffs = set()
                for i, (brow, vrow) in enumerate(zip(breader, vreader)):
                    for col in common_cols:
                        if col == "onset":
                            continue
                        bval = brow.get(col, "")
                        vval = vrow.get(col, "")
                        if bval != vval:
                            # Try numeric comparison for floating point
                            try:
                                if abs(float(bval) - float(vval)) > 1e-6:
                                    non_onset_diffs.add(col)
                            except (ValueError, TypeError):
                                non_onset_diffs.add(col)
                    if i > 5:  # sample first few rows
                        break
                if non_onset_diffs:
                    diffs["other_col_diffs"] = list(non_onset_diffs)
        except Exception as e:
            diffs["comparison_error"] = str(e)

    return diffs


def main():
    print("=" * 100)
    print("ONSET TIME COMPARISON: BIDS Root vs. Validation Events Files")
    print("=" * 100)

    pairs, unmatched_bids, unmatched_val = find_matching_pairs()

    print(f"\nFound {len(pairs)} matching file pairs")
    print(f"Unmatched BIDS root files: {len(unmatched_bids)}")
    print(f"Unmatched validation files: {len(unmatched_val)}")

    if unmatched_bids:
        print(f"\n--- Unmatched BIDS root files (no validation counterpart) ---")
        for key, path in unmatched_bids[:20]:
            print(f"  {key}")
        if len(unmatched_bids) > 20:
            print(f"  ... and {len(unmatched_bids) - 20} more")

    if unmatched_val:
        print(f"\n--- Unmatched validation files (no BIDS root counterpart) ---")
        for key, path in unmatched_val[:20]:
            print(f"  {key}")
        if len(unmatched_val) > 20:
            print(f"  ... and {len(unmatched_val) - 20} more")

    # Compare onset times
    results = []
    errors = []

    for key, bids_path, val_path in pairs:
        sub, ses, task, run = key
        bids_onset = read_first_onset(bids_path)
        val_onset = read_first_onset(val_path)

        if bids_onset is None or val_onset is None:
            errors.append((key, bids_path, val_path, bids_onset, val_onset))
            continue

        offset = val_onset - bids_onset
        results.append({
            "key": key,
            "sub": sub,
            "ses": ses,
            "task": task,
            "run": run,
            "bids_onset": bids_onset,
            "val_onset": val_onset,
            "offset": offset,
            "bids_path": bids_path,
            "val_path": val_path,
        })

    # Separate offset vs no-offset
    offset_results = [r for r in results if abs(r["offset"]) > 0.01]
    no_offset_results = [r for r in results if abs(r["offset"]) <= 0.01]

    print(f"\n{'=' * 100}")
    print(f"RESULTS SUMMARY")
    print(f"{'=' * 100}")
    print(f"Total compared: {len(results)}")
    print(f"Files with onset offset (|diff| > 0.01s): {len(offset_results)}")
    print(f"Files with NO onset offset: {len(no_offset_results)}")
    print(f"Errors (couldn't read onset): {len(errors)}")

    # ===== GROUP BY TASK =====
    print(f"\n{'=' * 100}")
    print(f"OFFSET BREAKDOWN BY TASK")
    print(f"{'=' * 100}")

    by_task = defaultdict(list)
    by_task_all = defaultdict(list)
    for r in results:
        by_task_all[r["task"]].append(r)
    for r in offset_results:
        by_task[r["task"]].append(r)

    for task in sorted(set(r["task"] for r in results)):
        total = len(by_task_all[task])
        with_offset = len(by_task[task])
        without_offset = total - with_offset
        print(f"\n  Task: {task}")
        print(f"    Total files: {total}, WITH offset: {with_offset}, WITHOUT offset: {without_offset}")
        if with_offset > 0:
            offsets = [r["offset"] for r in by_task[task]]
            print(f"    Offset range: {min(offsets):.6f} to {max(offsets):.6f}")
            print(f"    Offset mean: {statistics.mean(offsets):.6f}")
            if len(offsets) > 1:
                print(f"    Offset stdev: {statistics.stdev(offsets):.6f}")
            # Check if constant
            if max(offsets) - min(offsets) < 0.01:
                print(f"    => CONSTANT offset across all affected files: ~{statistics.mean(offsets):.6f}s")
            else:
                print(f"    => VARIABLE offset values!")

    # ===== GROUP BY SUBJECT =====
    print(f"\n{'=' * 100}")
    print(f"OFFSET BREAKDOWN BY SUBJECT")
    print(f"{'=' * 100}")

    by_sub = defaultdict(list)
    by_sub_all = defaultdict(list)
    for r in results:
        by_sub_all[r["sub"]].append(r)
    for r in offset_results:
        by_sub[r["sub"]].append(r)

    for sub in sorted(by_sub_all.keys()):
        total = len(by_sub_all[sub])
        with_offset = len(by_sub[sub])
        print(f"\n  Subject: {sub}")
        print(f"    Total: {total}, WITH offset: {with_offset}, WITHOUT: {total - with_offset}")
        if with_offset > 0:
            offsets = [r["offset"] for r in by_sub[sub]]
            print(f"    Offset range: {min(offsets):.6f} to {max(offsets):.6f}")
            # Break down by task within subject
            sub_by_task = defaultdict(list)
            for r in by_sub[sub]:
                sub_by_task[r["task"]].append(r)
            for task in sorted(sub_by_task.keys()):
                task_offsets = [r["offset"] for r in sub_by_task[task]]
                print(f"      {task}: {len(task_offsets)} files, "
                      f"offset range [{min(task_offsets):.6f}, {max(task_offsets):.6f}]")

    # ===== GROUP BY SESSION =====
    print(f"\n{'=' * 100}")
    print(f"OFFSET BREAKDOWN BY SUBJECT x SESSION")
    print(f"{'=' * 100}")

    by_sub_ses = defaultdict(list)
    for r in offset_results:
        by_sub_ses[(r["sub"], r["ses"])].append(r)

    for (sub, ses) in sorted(by_sub_ses.keys()):
        entries = by_sub_ses[(sub, ses)]
        offsets = [r["offset"] for r in entries]
        tasks = sorted(set(r["task"] for r in entries))
        offset_str = f"[{min(offsets):.6f}, {max(offsets):.6f}]" if len(offsets) > 1 else f"{offsets[0]:.6f}"
        print(f"  {sub} {ses}: {len(entries)} files, offset={offset_str}, tasks={tasks}")

    # ===== CHECK IF OFFSET IS CONSISTENT WITHIN EACH FILE =====
    print(f"\n{'=' * 100}")
    print(f"DETAILED OFFSET ANALYSIS (consistency within each file)")
    print(f"{'=' * 100}")

    for r in offset_results:
        details = check_full_file_differences(r["bids_path"], r["val_path"], r["offset"])
        print(f"\n  {r['sub']} {r['ses']} task-{r['task']} run-{r['run'] or 'none'}")
        print(f"    First-row offset: {r['offset']:.6f}s (BIDS={r['bids_onset']:.6f}, val={r['val_onset']:.6f})")
        for dk, dv in details.items():
            print(f"    {dk}: {dv}")

    # ===== FILES WITHOUT OFFSET: CHECK FOR OTHER DIFFERENCES =====
    print(f"\n{'=' * 100}")
    print(f"FILES WITHOUT ONSET OFFSET: checking for other differences")
    print(f"{'=' * 100}")

    other_diff_count = 0
    header_only_diff_count = 0
    identical_count = 0
    other_diff_types = defaultdict(int)

    for r in no_offset_results:
        details = check_full_file_differences(r["bids_path"], r["val_path"], r["offset"])
        if details:
            has_non_header = any(k != "header" for k in details)
            if has_non_header:
                other_diff_count += 1
                for k in details:
                    if k != "header":
                        other_diff_types[k] += 1
                print(f"\n  {r['sub']} {r['ses']} task-{r['task']} run-{r['run'] or 'none'}")
                print(f"    Onset offset: {r['offset']:.6f} (effectively zero)")
                for dk, dv in details.items():
                    print(f"    {dk}: {dv}")
            else:
                header_only_diff_count += 1
        else:
            identical_count += 1

    print(f"\n  Summary of non-offset files:")
    print(f"    Fully identical (data & headers): {identical_count}")
    print(f"    Header-only differences (column naming): {header_only_diff_count}")
    print(f"    Other differences (beyond header): {other_diff_count}")
    if other_diff_types:
        print(f"    Types of other differences: {dict(other_diff_types)}")

    # ===== CHECK PATTERN: Is offset = first dummy scan duration? =====
    print(f"\n{'=' * 100}")
    print(f"OFFSET PATTERN ANALYSIS")
    print(f"{'=' * 100}")

    # Check if offsets correlate with specific values
    unique_offsets = set()
    for r in offset_results:
        unique_offsets.add(round(r["offset"], 3))

    print(f"\n  Unique offset values (rounded to 3 decimal places): {len(unique_offsets)}")
    for uo in sorted(unique_offsets):
        count = sum(1 for r in offset_results if abs(r["offset"] - uo) < 0.01)
        print(f"    {uo:.3f}s : {count} files")

    # Check if offset per task is constant
    print(f"\n  Is the offset constant per task?")
    for task in sorted(by_task.keys()):
        offsets = [r["offset"] for r in by_task[task]]
        if len(offsets) == 0:
            continue
        spread = max(offsets) - min(offsets)
        if spread < 0.01:
            print(f"    {task}: YES (constant at ~{statistics.mean(offsets):.6f}s)")
        else:
            print(f"    {task}: NO (spread={spread:.6f}s)")
            # Show per-session breakdown
            by_ses = defaultdict(list)
            for r in by_task[task]:
                by_ses[(r["sub"], r["ses"])].append(r["offset"])
            for (sub, ses), offs in sorted(by_ses.items()):
                print(f"      {sub} {ses}: {[f'{o:.6f}' for o in offs]}")

    # ===== ERRORS =====
    if errors:
        print(f"\n{'=' * 100}")
        print(f"ERRORS (could not read onset)")
        print(f"{'=' * 100}")
        for key, bp, vp, bo, vo in errors:
            print(f"  {key}: BIDS onset={bo}, val onset={vo}")

    # ===== FINAL SUMMARY TABLE =====
    print(f"\n{'=' * 100}")
    print(f"FINAL SUMMARY TABLE: All offset files")
    print(f"{'=' * 100}")
    print(f"{'Subject':<10} {'Session':<10} {'Task':<20} {'Run':<6} {'BIDS onset':>14} {'Val onset':>14} {'Offset':>14}")
    print("-" * 90)
    for r in sorted(offset_results, key=lambda x: (x["sub"], x["ses"], x["task"], x["run"] or "")):
        print(f"{r['sub']:<10} {r['ses']:<10} {r['task']:<20} {r['run'] or 'none':<6} "
              f"{r['bids_onset']:>14.6f} {r['val_onset']:>14.6f} {r['offset']:>14.6f}")


if __name__ == "__main__":
    main()
