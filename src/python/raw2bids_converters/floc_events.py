#!/usr/bin/env python3
"""Convert Stanford fLoc localizer .mat logs into BIDS _events.tsv files.

Source: one Psychtoolbox workspace per run under
`mmmsourcedata/sub-XX/ses-YY/behavioral/<prefix>_<date>_fLoc_oddball_run<N>.mat`.
Each holds `theSubject.trials` (600 trials x 0.5 s = 300 s, struct-of-arrays
with `block`, `onset`, `cond`, `task`, `img`) and `theData` (per-trial `keys`,
`rt`, `resp`).

Design: 75 miniblocks of 8 images, 4 s each. Ten categories in five domains,
plus blank baseline blocks. The subject presses a button for the oddball, a
phase-scrambled image spliced into a block (`trials.task == 1`, and its `img`
begins with `scrambled`).

Emitted events are block-level, matching `localizer_events.py` and the
archived GLM plan. Per-trial image identity is not copied here; it stays
recoverable from the source .mat.

Two properties of the source were measured before writing this converter
(see docs/workbench/localizer-glm/log.md, 2026-08-21):

  1. Timing origin. The scanner trigger is logged as an ordinary keypress
     (`'`). Its median inter-pulse interval is 1.504 s (the TR) and the pulse
     train extrapolates back to t = 0, so the experiment clock starts on a
     trigger and block onsets need no shift. Onsets below come from the
     measured `theSubject.timePerTrial`, not the nominal schedule; the two
     differ by at most 22 ms over a run.

  2. Response key. `theData.falseAlarms` is not usable -- it counts the
     trigger pulses, which is why it sits near 190 in every run. The real
     button is `6^`. `n_response` below counts only that key.

Run mapping: the operator restarted the fLoc program between scanner runs and
encoded the target BIDS runs in the output prefix -- `mmm03run456` produced
BIDS run-04, run-05 and run-06, in its own run1/run2/run3 order. A bare
prefix (`mmm04`) means the script index is the BIDS run index.

Usage:
    python floc_events.py [--dry-run] [--subject 03] [<mat_path> ...]
"""

import argparse
import glob
import os
import re
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import scipy.io as sio

from common import (
    NA, BIDS_ROOT, SOURCE_DIR,
    bids_sub, bids_ses, bids_output_path,
    write_events_tsv, write_json_sidecar,
)

TASK = "floc"
TRIAL_DUR = 0.5
BLOCK_TRIALS = 8
RESPONSE_KEY = "6^"

# cond code -> (category, domain). Verified against the `img` filenames in all
# 18 runs: every cond maps to exactly one category with no exceptions.
CONDITIONS = {
    0:  ("baseline",   "baseline"),
    1:  ("word",       "character"),
    2:  ("number",     "character"),
    3:  ("body",       "body"),
    4:  ("limb",       "body"),
    5:  ("adult",      "face"),
    6:  ("child",      "face"),
    7:  ("corridor",   "place"),
    8:  ("house",      "place"),
    9:  ("car",        "object"),
    10: ("instrument", "object"),
}

MAT_GLOB = "sub-*/ses-*/behavioral/*fLoc_oddball_run*.mat"
# Prefixes seen so far: `mmm03`, `mmm03run456` (sub-03/04/05, restarts encoded)
# and a bare `6` (sub-06/07, one uninterrupted set of three).
MAT_RE = re.compile(
    r"^(?:mmm)?(?P<subj>\d+)(?:run(?P<runs>\d+))?_.*_fLoc_oddball_run(?P<script>\d+)\.mat$"
)


def entities_from_path(path):
    """Read (subj_num, ses_num) out of the sourcedata path.

    The path is authoritative for identity; the filename prefix is only
    cross-checked against it. Two entries in the scan log (OPEN-QUESTIONS
    Q1, Q2) record subject mislabels at acquisition, so a silent disagreement
    between the two is exactly what we want surfaced.
    """
    m = re.search(r"/sub-(\d+)/ses-(\d+)/", path)
    if not m:
        raise ValueError(f"No /sub-NN/ses-NN/ component in: {path}")
    return int(m.group(1)), int(m.group(2))


def parse_mat_name(path):
    """Map a source .mat filename to its BIDS run number.

    The prefix carries the BIDS runs a restart was meant to cover; the
    trailing `_run<N>` indexes into them. `mmm03run23_..._run2.mat` is the
    second of runs [2, 3], so BIDS run-03. A prefix with no run digits means
    the script index is the BIDS run index.
    """
    fname = os.path.basename(path)
    m = MAT_RE.match(fname)
    if not m:
        raise ValueError(f"Cannot parse fLoc mat filename: {fname}")

    path_subj, _ = entities_from_path(path)
    if int(m.group("subj")) != path_subj:
        print(f"  WARNING: {fname} names subject {m.group('subj')} but sits "
              f"under {bids_sub(path_subj)}; trusting the path")

    script_run = int(m.group("script"))
    runs = [int(c) for c in m.group("runs")] if m.group("runs") else None
    if runs is None:
        return script_run
    if not 1 <= script_run <= len(runs):
        raise ValueError(
            f"{fname}: script run {script_run} outside the range the prefix "
            f"declares ({runs}). A file may be missing, or the prefix is wrong."
        )
    return runs[script_run - 1]


def discover_mats(subject=None):
    """Find fLoc source .mat files for sessions that exist in the BIDS tree.

    Subjects still being collected (sub-06, sub-07 as of 2026-08) have source
    .mat files but no BIDS session to write into. They are skipped, loudly.
    """
    paths = sorted(glob.glob(os.path.join(SOURCE_DIR, MAT_GLOB)))
    if subject is not None:
        want = bids_sub(int(subject))
        paths = [p for p in paths if f"/{want}/" in p]

    keep, skipped = [], defaultdict(int)
    for p in paths:
        subj, ses = entities_from_path(p)
        func_dir = os.path.join(BIDS_ROOT, bids_sub(subj), bids_ses(ses), "func")
        if os.path.isdir(func_dir):
            keep.append(p)
        else:
            skipped[(subj, ses)] += 1
    for (subj, ses), n in sorted(skipped.items()):
        print(f"  Skipping {n} .mat under {bids_sub(subj)} {bids_ses(ses)}: "
              f"no BIDS func/ directory (not BIDSified yet)")
    return keep


def resolve_runs(paths):
    """Return [(path, subj, ses, run)], after checking each session is complete.

    A session's runs must be exactly 1..N with no duplicates. A duplicate
    means two source files claim the same BIDS run; a gap means a source file
    is missing. Either way the mapping is not trustworthy and we stop.
    """
    resolved = []
    for p in paths:
        subj, ses = entities_from_path(p)
        run = parse_mat_name(p)
        resolved.append((p, subj, ses, run))

    by_session = defaultdict(list)
    for p, subj, ses, run in resolved:
        by_session[(subj, ses)].append((run, p))

    for (subj, ses), items in sorted(by_session.items()):
        runs = sorted(r for r, _ in items)
        expected = list(range(1, len(runs) + 1))
        if runs != expected:
            detail = "\n".join(
                f"    run-{r:02d} <- {os.path.basename(p)}" for r, p in sorted(items)
            )
            raise ValueError(
                f"{bids_sub(subj)} {bids_ses(ses)}: resolved runs {runs}, "
                f"expected {expected}. Mapping:\n{detail}\n"
                f"  Fix the prefix table in this module's docstring before writing events."
            )
    return resolved


def build_events(mat_path, subj, ses, run):
    """Collapse a run's 600 trials into 75 block-level BIDS events."""
    m = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    subject, data = m["theSubject"], m["theData"]
    trials = subject.trials

    block = np.atleast_1d(trials.block).astype(int)
    cond = np.atleast_1d(trials.cond).astype(int)
    oddball = np.atleast_1d(trials.task).astype(int)
    img = np.array([str(x) for x in np.atleast_1d(trials.img)])
    onset = np.atleast_1d(subject.timePerTrial).astype(float)
    total_time = float(subject.totalTime)
    keys = np.array([str(k) for k in np.atleast_1d(data.keys)])

    n = len(block)
    if not (len(cond) == len(onset) == len(img) == len(keys) == n):
        raise ValueError(f"{mat_path}: ragged trial arrays ({n} blocks vs others)")

    # The oddball is a scrambled image; the flag and the filename must agree.
    scrambled = np.char.startswith(img, "scrambled")
    if not np.array_equal(oddball == 1, scrambled):
        raise ValueError(
            f"{mat_path}: trials.task disagrees with the scrambled images "
            f"({int((oddball == 1).sum())} flagged, {int(scrambled.sum())} scrambled)"
        )

    pressed = np.char.find(keys, RESPONSE_KEY) >= 0

    rows = []
    block_ids = sorted(set(block.tolist()))
    for i, b in enumerate(block_ids):
        sel = np.where(block == b)[0]
        if len(sel) != BLOCK_TRIALS:
            raise ValueError(
                f"{mat_path}: block {b} has {len(sel)} trials, expected {BLOCK_TRIALS}"
            )
        codes = set(cond[sel].tolist())
        if len(codes) != 1:
            raise ValueError(f"{mat_path}: block {b} mixes conditions {sorted(codes)}")
        code = codes.pop()
        if code not in CONDITIONS:
            raise ValueError(f"{mat_path}: block {b} has unknown cond {code}")
        trial_type, domain = CONDITIONS[code]

        start = float(onset[sel[0]])
        # Contiguous coverage, as the motor localizer events do: a block runs
        # until the next one starts, and the last runs to the end of the task.
        if i + 1 < len(block_ids):
            nxt = np.where(block == block_ids[i + 1])[0][0]
            duration = float(onset[nxt]) - start
        else:
            duration = total_time - start

        rows.append({
            "onset": round(start, 3),
            "duration": round(duration, 3),
            "subj_num": subj,
            "ses_num": ses,
            "run_idx": run,
            "trial_type": trial_type,
            "domain": domain,
            "n_oddball": int(oddball[sel].sum()),
            "n_response": int(pressed[sel].sum()),
        })

    df = pd.DataFrame(rows)
    if len(df) != 75:
        raise ValueError(f"{mat_path}: built {len(df)} blocks, expected 75")
    return df


def sidecar(mat_path):
    """Column descriptions, plus the provenance a reader needs to trust these."""
    return {
        "onset": {
            "Description": (
                "Block onset, measured from theSubject.timePerTrial. The "
                "experiment clock starts on a scanner trigger, so this is "
                "relative to the first volume with no shift applied."
            ),
            "Units": "s",
        },
        "duration": {
            "Description": (
                "Block duration, taken as the interval to the next block "
                "onset (the final block runs to theSubject.totalTime). "
                "Nominally 4 s: 8 images at 0.5 s."
            ),
            "Units": "s",
        },
        "subj_num": {"Description": "Subject number."},
        "ses_num": {"Description": "BIDS session number."},
        "run_idx": {"Description": "BIDS run number."},
        "trial_type": {
            "Description": "Stimulus category of the block.",
            "Levels": {
                name: f"{domain} domain" if name != "baseline" else "blank fixation block"
                for name, domain in CONDITIONS.values()
            },
        },
        "domain": {
            "Description": (
                "Category domain, for the standard fLoc contrasts "
                "(e.g. face > all others)."
            ),
            "Levels": {
                "character": "word, number",
                "body": "body, limb",
                "face": "adult, child",
                "place": "corridor, house",
                "object": "car, instrument",
                "baseline": "blank fixation",
            },
        },
        "n_oddball": {
            "Description": (
                "Phase-scrambled oddball images in this block, the target of "
                "the detection task."
            ),
        },
        "n_response": {
            "Description": (
                "Button presses in this block, counting only key '6^'. The "
                "source .mat also logs the scanner trigger as a keypress, so "
                "its own falseAlarms field is inflated and is not used here."
            ),
        },
        "Description": (
            "Stanford VPNL fLoc localizer, 75 miniblocks of 8 images at 0.5 s "
            "(4 s per block, 300 s per run). Ten categories in five domains "
            "plus blank baseline; the subject detects a phase-scrambled "
            "oddball. Collapsed to block level from the run's Psychtoolbox "
            "workspace by mmmdata raw2bids_converters/floc_events.py; "
            "per-trial image identity remains in the source .mat. Onsets are "
            "measured presentation times and are already relative to the "
            "first volume -- the logged scanner-trigger train (median 1.504 s "
            "interval) extrapolates back to t=0, so no shift is applied."
        ),
        "StimulusPresentation": {
            "SoftwareName": "Psychtoolbox-3 (Stanford VPNL fLoc)",
        },
        "Sources": [os.path.relpath(mat_path, os.path.dirname(SOURCE_DIR))],
    }


def convert(mat_path, subj, ses, run, dry_run=False):
    df = build_events(mat_path, subj, ses, run)
    stem = f"{bids_sub(subj)}_{bids_ses(ses)}_task-{TASK}_run-{run:02d}"
    tsv = bids_output_path(subj, ses, "func", f"{stem}_events.tsv")
    json_path = bids_output_path(subj, ses, "func", f"{stem}_events.json")

    bold = bids_output_path(subj, ses, "func", f"{stem}_bold.nii.gz")
    if not os.path.exists(bold):
        print(f"  WARNING: no BOLD at {bold} -- writing events anyway")

    print(f"  {os.path.basename(mat_path)} -> {stem}")
    write_events_tsv(df, tsv, dry_run=dry_run)
    write_json_sidecar(sidecar(mat_path), json_path, dry_run=dry_run)
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("mats", nargs="*", help="Source .mat paths (default: all)")
    ap.add_argument("--subject", help="Restrict to one subject, e.g. 03")
    ap.add_argument("--dry-run", action="store_true", help="Report without writing")
    args = ap.parse_args()

    paths = args.mats or discover_mats(args.subject)
    if not paths:
        print("No fLoc .mat files found.", file=sys.stderr)
        return 1

    resolved = resolve_runs(paths)
    print(f"Converting {len(resolved)} fLoc runs"
          f"{' [dry-run]' if args.dry_run else ''}")

    total = 0
    for path, subj, ses, run in sorted(resolved, key=lambda r: (r[1], r[2], r[3])):
        df = convert(path, subj, ses, run, dry_run=args.dry_run)
        total += len(df)
    print(f"Done: {len(resolved)} runs, {total} blocks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
