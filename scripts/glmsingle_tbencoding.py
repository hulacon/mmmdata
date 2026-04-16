#!/usr/bin/env python3
"""
glmsingle_tbencoding.py — Run GLMsingle across all TBencoding sessions.

Concatenates all TBencoding runs across sessions into a single GLMsingle
call with a 1000-condition design matrix (one column per unique mmmId in the
experiment). Uses sessionindicator to let GLMsingle handle session-wise
polynomial drift and scaling.

Key design features:
  - 1000-column design matrix: every stimulus gets its own condition, so
    cross-session repetitions (especially the 6 anchor items × 42 reps)
    contribute to HRF fitting and reliability estimation.
  - sessionindicator: tells GLMsingle which runs belong to which scanning
    session, enabling proper session-wise nuisance modeling.
  - No external confound regression: GLMsingle's GLMdenoise (Type C) learns
    noise regressors from the data itself. The authors recommend against
    pre-filtering with motion params / aCompCor (see GLMsingle wiki).

Usage:
    python glmsingle_tbencoding.py --subject sub-03
    python glmsingle_tbencoding.py --subject sub-03 --sessions ses-04 ses-05
    python glmsingle_tbencoding.py --subject sub-03 --spike-regressors
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

# ── paths ────────────────────────────────────────────────────────────────────

BIDS_ROOT = Path("/gpfs/projects/hulacon/shared/mmmdata")
FMRIPREP_DEFAULT = BIDS_ROOT / "derivatives" / "fmriprep"
OUTPUT_BASE = BIDS_ROOT / "derivatives" / "glmsingle"

SPACE = "MNI152NLin2009cAsym_res-2"
TASK = "TBencoding"
TB_SESSIONS = [f"ses-{i:02d}" for i in range(4, 18)]  # ses-04 through ses-17

# Experiment parameters
TR = 1.5       # seconds
STIMDUR = 3.0  # seconds (image presentation duration)
FD_THRESHOLD = 0.5  # mm, for optional spike regressors


# ── path helpers ─────────────────────────────────────────────────────────────

def bold_path(fmriprep_dir, subject, session, run):
    return (
        fmriprep_dir / subject / session / "func"
        / f"{subject}_{session}_task-{TASK}_{run}_space-{SPACE}_desc-preproc_bold.nii.gz"
    )


def confounds_path(fmriprep_dir, subject, session, run):
    return (
        fmriprep_dir / subject / session / "func"
        / f"{subject}_{session}_task-{TASK}_{run}_desc-confounds_timeseries.tsv"
    )


def events_path(subject, session, run):
    return (
        BIDS_ROOT / subject / session / "func"
        / f"{subject}_{session}_task-{TASK}_{run}_events.tsv"
    )


def detect_runs(fmriprep_dir, subject, session):
    """Auto-detect available TBencoding runs for a subject/session."""
    func_dir = fmriprep_dir / subject / session / "func"
    pattern = f"{subject}_{session}_task-{TASK}_run-*_space-{SPACE}_desc-preproc_bold.nii.gz"
    return sorted([
        p.name.split("_")[3]  # extract run-XX
        for p in func_dir.glob(pattern)
    ])


# ── data loading ─────────────────────────────────────────────────────────────

def discover_sessions(fmriprep_dir, subject, sessions=None):
    """Discover all TBencoding sessions with fMRIPrep output.

    Returns list of (session, [runs]) tuples.
    """
    if sessions is None:
        sessions = TB_SESSIONS

    found = []
    for ses in sessions:
        runs = detect_runs(fmriprep_dir, subject, ses)
        if runs:
            found.append((ses, runs))
            print(f"  {ses}: {len(runs)} runs")
        else:
            print(f"  {ses}: no TBencoding runs found, skipping")
    return found


def load_all_events(subject, session_runs):
    """Load events from all sessions/runs.

    Args:
        subject: e.g. "sub-03"
        session_runs: list of (session, [runs]) tuples

    Returns:
        all_events: list of DataFrames (one per run, image trials only),
                    with 'session' and 'run' columns added
    """
    all_events = []
    for session, runs in session_runs:
        for run in runs:
            path = events_path(subject, session, run)
            df = pd.read_csv(path, sep="\t")
            img = df[df["trial_type"] == "image"].copy()
            img["session"] = session
            img["run"] = run
            all_events.append(img)
    return all_events


def build_condition_mapping(all_events):
    """Build global mmmId → column index mapping across ALL sessions.

    Returns:
        cond_map: dict mapping mmmId (str) to column index (int)
        condition_key: DataFrame with columns [col_index, mmmId, n_presentations]
    """
    seen = {}
    for events_df in all_events:
        for mmm_id in events_df["mmmId"].values:
            mmm_id_str = str(mmm_id)
            if mmm_id_str not in seen:
                seen[mmm_id_str] = len(seen)

    cond_map = seen
    n_conditions = len(cond_map)

    all_ids = []
    for events_df in all_events:
        all_ids.extend(str(x) for x in events_df["mmmId"].values)
    counts = Counter(all_ids)

    rows = []
    for mmm_id, col_idx in sorted(cond_map.items(), key=lambda x: x[1]):
        rows.append({
            "col_index": col_idx,
            "mmmId": mmm_id,
            "n_presentations": counts[mmm_id],
        })
    condition_key = pd.DataFrame(rows)

    n_repeated = (condition_key["n_presentations"] > 1).sum()
    rep_dist = Counter(condition_key["n_presentations"].values)
    print(f"  {n_conditions} unique conditions, {n_repeated} with repetitions")
    print(f"  Repetition distribution: "
          + ", ".join(f"{n_items}×{n_reps}reps"
                      for n_reps, n_items in sorted(rep_dist.items())))

    return cond_map, condition_key


def build_design_matrices(all_events, cond_map, n_volumes_per_run, run_labels):
    """Build condition-based design matrices for GLMsingle.

    Each run's design matrix has shape (n_volumes, n_conditions).
    The same condition column is shared across sessions for repeated stimuli.
    """
    n_conditions = len(cond_map)
    designs = []
    trial_rows = []

    for run_idx, (events_df, n_vols) in enumerate(
        zip(all_events, n_volumes_per_run)
    ):
        design = np.zeros((n_vols, n_conditions), dtype=np.float32)

        for _, trial in events_df.iterrows():
            mmm_id_str = str(trial["mmmId"])
            col_idx = cond_map[mmm_id_str]
            onset_vol = int(np.round(trial["onset"] / TR))

            if 0 <= onset_vol < n_vols:
                design[onset_vol, col_idx] = 1.0

            trial_rows.append({
                "session": trial.get("session", ""),
                "run": trial.get("run", ""),
                "run_idx": run_idx,
                "onset": trial["onset"],
                "duration": trial["duration"],
                "mmmId": mmm_id_str,
                "col_index": col_idx,
                "word": trial.get("word", ""),
                "trial_type": trial.get("trial_type", "image"),
            })

        n_active = (design.sum(axis=0) > 0).sum()
        print(f"  {run_labels[run_idx]}: design ({n_vols} × {n_conditions}), "
              f"{n_active} active conditions")
        designs.append(design)

    trial_info = pd.DataFrame(trial_rows)
    return designs, trial_info


def build_spike_regressors(fmriprep_dir, subject, session, run, n_volumes):
    """Build spike (motion censoring) regressors for one run."""
    path = confounds_path(fmriprep_dir, subject, session, run)
    df = pd.read_csv(path, sep="\t")

    fd = df["framewise_displacement"].values
    fd = np.nan_to_num(fd, nan=0.0)
    outlier_trs = np.where(fd > FD_THRESHOLD)[0]

    assert len(df) == n_volumes, (
        f"Confounds ({len(df)}) != BOLD ({n_volumes}) for {session}/{run}"
    )

    if len(outlier_trs) > 0:
        spikes = np.zeros((n_volumes, len(outlier_trs)), dtype=np.float64)
        for i, tr_idx in enumerate(outlier_trs):
            spikes[tr_idx, i] = 1.0
        return spikes
    return None


# ── GLMsingle runner ─────────────────────────────────────────────────────────

def run_glmsingle(fmriprep_dir, subject, session_runs, output_dir,
                   use_spike_regressors=False):
    """Run GLMsingle across all TBencoding sessions for one subject.

    Args:
        fmriprep_dir: Path to fMRIPrep derivatives directory
        subject: e.g. "sub-03"
        session_runs: list of (session, [runs]) tuples
        output_dir: Path for GLMsingle outputs
        use_spike_regressors: if True, pass spike regressors for high-motion TRs
    """
    from glmsingle.glmsingle import GLM_single

    # Flatten session_runs into ordered run list
    run_list = []  # (session, run) tuples
    run_labels = []
    session_indices = []  # 1-indexed session number for sessionindicator
    for ses_idx, (session, runs) in enumerate(session_runs):
        for run in runs:
            run_list.append((session, run))
            run_labels.append(f"{session}/{run}")
            session_indices.append(ses_idx + 1)

    n_total_runs = len(run_list)
    n_sessions = len(session_runs)

    print(f"\n{'=' * 70}")
    print(f"GLMsingle: {subject} — {n_sessions} sessions, {n_total_runs} runs")
    print(f"{'=' * 70}")

    # ── Load BOLD data ──
    print("\nLoading BOLD data...")
    data_list = []
    n_volumes_per_run = []
    for session, run in run_list:
        bp = bold_path(fmriprep_dir, subject, session, run)
        if not bp.exists():
            print(f"  ERROR: {bp} not found")
            sys.exit(1)
        img = nib.load(str(bp))
        data = img.get_fdata(dtype=np.float32)
        n_vols = data.shape[-1]
        print(f"  {session}/{run}: {data.shape}")
        data_list.append(data)
        n_volumes_per_run.append(n_vols)

    # ── Load events and build design matrices ──
    print("\nLoading events...")
    all_events = load_all_events(subject, session_runs)
    print(f"  {len(all_events)} run event files, "
          f"{sum(len(e) for e in all_events)} total trials")

    print("\nBuilding condition mapping...")
    cond_map, condition_key = build_condition_mapping(all_events)

    print("\nBuilding design matrices...")
    designs, trial_info = build_design_matrices(
        all_events, cond_map, n_volumes_per_run, run_labels
    )

    # ── Build sessionindicator ──
    sessionindicator = np.array(session_indices, dtype=int).reshape(1, -1)
    print(f"\nsessionindicator: {sessionindicator.shape} — "
          f"{n_sessions} sessions, {n_total_runs} runs")

    # ── Optionally build spike regressors ──
    extra_regressors = None
    spike_counts = {}
    if use_spike_regressors:
        print("\nBuilding spike regressors...")
        extra_regressors = []
        for i, (session, run) in enumerate(run_list):
            spikes = build_spike_regressors(
                fmriprep_dir, subject, session, run, n_volumes_per_run[i]
            )
            n_spikes = 0 if spikes is None else spikes.shape[1]
            spike_counts[run_labels[i]] = n_spikes
            if spikes is None:
                spikes = np.zeros((n_volumes_per_run[i], 0), dtype=np.float64)
            extra_regressors.append(spikes)
            if n_spikes > 0:
                print(f"    {run_labels[i]}: {n_spikes} spike regressors")
        total_spikes = sum(spike_counts.values())
        print(f"  Total spike regressors: {total_spikes}")
    else:
        print("\nNo external confound regression (per GLMsingle recommendation).")

    # ── Configure and run GLMsingle ──
    glmsingle_outdir = output_dir / "glmsingle_outputs"
    figuredir = output_dir / "glmsingle_figures"

    print(f"\nOutput: {glmsingle_outdir}")
    print(f"Figures: {figuredir}")

    params = {
        "wantlibrary": 1,
        "wantglmdenoise": 1,
        "wantfracridge": 1,
        "wantfileoutputs": [1, 1, 1, 1],
        "wantmemoryoutputs": [0, 0, 0, 0],
        "sessionindicator": sessionindicator,
    }
    if extra_regressors is not None:
        params["extra_regressors"] = extra_regressors

    print(f"\nGLMsingle configuration:")
    print(f"  TR = {TR}s, stimdur = {STIMDUR}s")
    print(f"  {n_total_runs} runs across {n_sessions} sessions")
    print(f"  {len(cond_map)} conditions (design matrix columns)")
    print(f"  sessionindicator: {session_indices}")
    if extra_regressors is not None:
        print(f"  extra_regressors (spike only): "
              f"{[r.shape for r in extra_regressors]}")
    else:
        print(f"  extra_regressors: None")
    print(f"  FitHRF + GLMdenoise + Ridge Regression")

    glm = GLM_single(params)

    print("\nRunning GLMsingle (this will take a while with 42 runs)...")
    results = glm.fit(
        design=designs,
        data=data_list,
        stimdur=STIMDUR,
        tr=TR,
        outputdir=str(glmsingle_outdir),
        figuredir=str(figuredir),
    )
    print("GLMsingle complete.")

    # ── Save supplementary outputs ──
    condition_key.to_csv(output_dir / "condition_key.csv", index=False)
    trial_info.to_csv(output_dir / "trial_info.csv", index=False)

    metadata = {
        "subject": subject,
        "task": TASK,
        "fmriprep_dir": str(fmriprep_dir),
        "sessions": [sr[0] for sr in session_runs],
        "runs_per_session": {sr[0]: sr[1] for sr in session_runs},
        "n_sessions": n_sessions,
        "n_total_runs": n_total_runs,
        "run_labels": run_labels,
        "session_indices": session_indices,
        "tr": TR,
        "stimdur": STIMDUR,
        "n_conditions": len(cond_map),
        "n_trials_total": len(trial_info),
        "n_repeated_conditions": int(
            (condition_key["n_presentations"] > 1).sum()
        ),
        "confound_strategy": (
            "spike_regressors_only" if use_spike_regressors
            else "none (GLMdenoise handles denoising)"
        ),
        "fd_threshold": FD_THRESHOLD if use_spike_regressors else None,
        "spike_counts": spike_counts if use_spike_regressors else None,
    }
    with open(output_dir / "run_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved condition_key.csv, trial_info.csv, run_metadata.json")
    print(f"GLMsingle file outputs in: {glmsingle_outdir}")
    return results


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--subject", required=True,
        help="Subject ID (e.g. sub-03)",
    )
    parser.add_argument(
        "--sessions", nargs="+", default=None,
        help="Specific sessions to include (default: all TB sessions ses-04..ses-17)",
    )
    parser.add_argument(
        "--fmriprep-dir", type=Path, default=None,
        help="Path to fMRIPrep derivatives (default: derivatives/fmriprep)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Override output directory (default: derivatives/glmsingle/{subject})",
    )
    parser.add_argument(
        "--spike-regressors", action="store_true", default=False,
        help="Include spike regressors for high-motion TRs (FD > 0.5mm). "
             "Default: no external confounds per GLMsingle recommendation.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    subject = args.subject
    fmriprep_dir = args.fmriprep_dir or FMRIPREP_DEFAULT

    print(f"Discovering TBencoding sessions for {subject}...")
    print(f"fMRIPrep dir: {fmriprep_dir}")
    session_runs = discover_sessions(fmriprep_dir, subject, args.sessions)

    if not session_runs:
        print("ERROR: No TBencoding sessions found")
        sys.exit(1)

    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        output_dir = OUTPUT_BASE / subject

    output_dir.mkdir(parents=True, exist_ok=True)
    run_glmsingle(fmriprep_dir, subject, session_runs, output_dir,
                   use_spike_regressors=args.spike_regressors)

    print("\nDone.")


if __name__ == "__main__":
    main()
