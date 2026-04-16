#!/usr/bin/env python3
"""
nordic_glmsingle_pilot.py — Run GLMsingle on TBencoding data for NORDIC pilot.

Compares original vs NORDIC-denoised fMRIPrep outputs by running GLMsingle
on 3 TBencoding runs from ses-04 for a single subject.

Key design choices:
  - Condition-based design matrix (77 columns = unique mmmIds) so GLMsingle
    can cross-validate HRF, GLMdenoise, and ridge regression using the 53
    stimuli repeated 3x within the session.
  - No external confound regression by default. GLMsingle's GLMdenoise step
    (Type C) learns noise regressors from the data itself; the authors
    recommend against pre-filtering with motion params / aCompCor as it can
    introduce bias and cause complications (see GLMsingle wiki). Optional
    --spike-regressors flag adds censoring regressors for high-motion TRs.
  - Output: single-trial betas for all 183 presentations, with condition
    mapping in DESIGNINFO.npy['stimorder'].

Usage:
    python nordic_glmsingle_pilot.py --subject sub-03 --pipeline original
    python nordic_glmsingle_pilot.py --subject sub-03 --pipeline nordic
    python nordic_glmsingle_pilot.py --subject sub-03 --pipeline both
"""

import argparse
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

# ── paths ────────────────────────────────────────────────────────────────────

BIDS_ROOT = Path("/gpfs/projects/hulacon/shared/mmmdata")
FMRIPREP_ORIG = BIDS_ROOT / "derivatives" / "fmriprep"
FMRIPREP_NORDIC = BIDS_ROOT / "derivatives" / "fmriprep_nordic"
OUTPUT_BASE = BIDS_ROOT / "derivatives" / "nordic" / "validation" / "glmsingle"

SPACE = "MNI152NLin2009cAsym_res-2"
TASK = "TBencoding"

# Experiment parameters
TR = 1.5       # seconds
STIMDUR = 3.0  # seconds (image presentation duration)
FD_THRESHOLD = 0.5  # mm, for optional spike regressors


# ── path helpers ─────────────────────────────────────────────────────────────

def bold_path(deriv_root, subject, session, run):
    return (
        deriv_root / subject / session / "func"
        / f"{subject}_{session}_task-{TASK}_{run}_space-{SPACE}_desc-preproc_bold.nii.gz"
    )


def mask_path(deriv_root, subject, session, run):
    return (
        deriv_root / subject / session / "func"
        / f"{subject}_{session}_task-{TASK}_{run}_space-{SPACE}_desc-brain_mask.nii.gz"
    )


def confounds_path(deriv_root, subject, session, run):
    return (
        deriv_root / subject / session / "func"
        / f"{subject}_{session}_task-{TASK}_{run}_desc-confounds_timeseries.tsv"
    )


def events_path(subject, session, run):
    return (
        BIDS_ROOT / subject / session / "func"
        / f"{subject}_{session}_task-{TASK}_{run}_events.tsv"
    )


def detect_runs(deriv_root, subject, session):
    """Auto-detect available TBencoding runs for a subject/session."""
    func_dir = deriv_root / subject / session / "func"
    pattern = f"{subject}_{session}_task-{TASK}_run-*_space-{SPACE}_desc-preproc_bold.nii.gz"
    runs = sorted([
        p.name.split("_")[3]  # extract run-XX
        for p in func_dir.glob(pattern)
    ])
    return runs


# ── data loading ─────────────────────────────────────────────────────────────

def load_events(subject, session, runs):
    """Load events from all runs. Returns list of DataFrames (image trials only)."""
    all_events = []
    for run in runs:
        path = events_path(subject, session, run)
        df = pd.read_csv(path, sep="\t")
        img = df[df["trial_type"] == "image"].copy()
        img["run"] = run
        all_events.append(img)
        print(f"  {run}: {len(img)} image trials, {len(df) - len(img)} rest trials")
    return all_events


def build_condition_mapping(all_events):
    """Build global mmmId → column index mapping from all runs' events.

    Returns:
        cond_map: dict mapping mmmId (str) to column index (int)
        condition_key: DataFrame with columns [col_index, mmmId, n_presentations]
    """
    # Collect all mmmIds in order of first appearance
    seen = {}
    for events_df in all_events:
        for mmm_id in events_df["mmmId"].values:
            mmm_id_str = str(mmm_id)
            if mmm_id_str not in seen:
                seen[mmm_id_str] = len(seen)

    cond_map = seen
    n_conditions = len(cond_map)

    # Count presentations per condition
    all_ids = []
    for events_df in all_events:
        all_ids.extend(str(x) for x in events_df["mmmId"].values)

    from collections import Counter
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
    print(f"  {n_conditions} unique conditions, {n_repeated} with repetitions")

    return cond_map, condition_key


def build_design_matrices(all_events, cond_map, n_volumes_per_run, runs):
    """Build condition-based design matrices for GLMsingle.

    Each run's design matrix has shape (n_volumes, n_conditions).
    Repeated stimuli share a column across runs.

    Args:
        all_events: list of DataFrames (one per run, image trials only)
        cond_map: dict mapping mmmId (str) to column index
        n_volumes_per_run: list of int, volumes per run
        runs: list of run IDs (e.g. ["run-01", "run-02", "run-03"])

    Returns:
        designs: list of numpy arrays (one per run)
        trial_info: DataFrame with per-trial metadata
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

            # Mark onset TR only — GLMsingle handles stimulus duration
            # via the stimdur parameter passed to glm.fit().
            if 0 <= onset_vol < n_vols:
                design[onset_vol, col_idx] = 1.0

            trial_rows.append({
                "run": runs[run_idx],
                "run_idx": run_idx,
                "onset": trial["onset"],
                "duration": trial["duration"],
                "mmmId": mmm_id_str,
                "col_index": col_idx,
                "word": trial.get("word", ""),
                "trial_type": trial.get("trial_type", "image"),
            })

        n_active = (design.sum(axis=0) > 0).sum()
        print(f"  {runs[run_idx]}: design ({n_vols} x {n_conditions}), "
              f"{n_active} active conditions")
        designs.append(design)

    trial_info = pd.DataFrame(trial_rows)
    return designs, trial_info


def build_spike_regressors(deriv_root, subject, session, run, n_volumes):
    """Build spike (motion censoring) regressors for one run.

    Returns numpy array (n_volumes, n_spikes) with one column per outlier TR
    where framewise displacement exceeds FD_THRESHOLD, or None if no outliers.
    """
    path = confounds_path(deriv_root, subject, session, run)
    df = pd.read_csv(path, sep="\t")

    fd = df["framewise_displacement"].values
    fd = np.nan_to_num(fd, nan=0.0)
    outlier_trs = np.where(fd > FD_THRESHOLD)[0]

    assert len(df) == n_volumes, (
        f"Confounds ({len(df)}) != BOLD ({n_volumes}) for {run}"
    )

    if len(outlier_trs) > 0:
        spikes = np.zeros((n_volumes, len(outlier_trs)), dtype=np.float64)
        for i, tr_idx in enumerate(outlier_trs):
            spikes[tr_idx, i] = 1.0
        print(f"    {run}: {len(outlier_trs)} spike regressors "
              f"(FD > {FD_THRESHOLD}mm)")
        return spikes
    else:
        print(f"    {run}: 0 outlier TRs")
        return None


# ── GLMsingle runner ─────────────────────────────────────────────────────────

def run_glmsingle(subject, session, pipeline, deriv_root, output_dir,
                   use_spike_regressors=False):
    """Run GLMsingle for one subject/session/pipeline combination.

    Args:
        subject: e.g. "sub-03"
        session: e.g. "ses-04"
        pipeline: "original" or "nordic"
        deriv_root: Path to fMRIPrep derivatives
        output_dir: Path for GLMsingle outputs
        use_spike_regressors: if True, pass spike regressors for high-motion
            TRs via extra_regressors. Default False per GLMsingle recommendation
            to avoid external confound regression.
    """
    from glmsingle.glmsingle import GLM_single

    runs = detect_runs(deriv_root, subject, session)
    if not runs:
        print(f"ERROR: No TBencoding runs found for {subject}/{session}")
        sys.exit(1)

    print(f"\n{'=' * 70}")
    print(f"GLMsingle: {subject} / {session} / {pipeline} ({len(runs)} runs)")
    print(f"{'=' * 70}")

    # ── Load BOLD data ──
    print("\nLoading BOLD data...")
    data_list = []
    n_volumes_per_run = []
    for run in runs:
        bp = bold_path(deriv_root, subject, session, run)
        if not bp.exists():
            print(f"  ERROR: {bp} not found")
            sys.exit(1)
        img = nib.load(str(bp))
        data = img.get_fdata(dtype=np.float32)
        n_vols = data.shape[-1]
        print(f"  {run}: {data.shape}")
        data_list.append(data)
        n_volumes_per_run.append(n_vols)

    # ── Load events and build design matrices ──
    print("\nLoading events...")
    all_events = load_events(subject, session, runs)

    print("\nBuilding condition mapping...")
    cond_map, condition_key = build_condition_mapping(all_events)

    print("\nBuilding design matrices...")
    designs, trial_info = build_design_matrices(
        all_events, cond_map, n_volumes_per_run, runs
    )

    # ── Optionally build spike regressors ──
    extra_regressors = None
    spike_counts = {}
    if use_spike_regressors:
        print("\nBuilding spike regressors...")
        extra_regressors = []
        for run_idx, run in enumerate(runs):
            spikes = build_spike_regressors(
                deriv_root, subject, session, run, n_volumes_per_run[run_idx]
            )
            extra_regressors.append(spikes)
            spike_counts[run] = 0 if spikes is None else spikes.shape[1]
        # Replace None entries with empty arrays for GLMsingle
        for i, reg in enumerate(extra_regressors):
            if reg is None:
                extra_regressors[i] = np.zeros(
                    (n_volumes_per_run[i], 0), dtype=np.float64
                )
    else:
        print("\nNo external confound regression (per GLMsingle recommendation).")
        print("  GLMdenoise will learn noise regressors from the data.")

    # ── Configure and run GLMsingle ──
    glmsingle_outdir = output_dir / "glmsingle_outputs"
    figuredir = output_dir / "glmsingle_figures"

    # GLMsingle deletes and recreates outputdir, so use a subdirectory
    print(f"\nOutput: {glmsingle_outdir}")
    print(f"Figures: {figuredir}")

    params = {
        "wantlibrary": 1,
        "wantglmdenoise": 1,
        "wantfracridge": 1,
        "wantfileoutputs": [1, 1, 1, 1],
        "wantmemoryoutputs": [0, 0, 0, 0],  # save memory; read from disk
    }
    if extra_regressors is not None:
        params["extra_regressors"] = extra_regressors

    print(f"\nGLMsingle configuration:")
    print(f"  TR = {TR}s, stimdur = {STIMDUR}s")
    print(f"  {len(designs)} runs, {len(cond_map)} conditions")
    if extra_regressors is not None:
        print(f"  extra_regressors (spike only): "
              f"{[r.shape for r in extra_regressors]}")
    else:
        print(f"  extra_regressors: None")
    print(f"  FitHRF + GLMdenoise + Ridge Regression")

    glm = GLM_single(params)

    print("\nRunning GLMsingle (this may take a while)...")
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
        "pipeline": pipeline,
        "session": session,
        "task": TASK,
        "runs": runs,
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
        "deriv_root": str(deriv_root),
    }
    with open(output_dir / "run_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved condition_key.csv, trial_info.csv, run_metadata.json")
    print(f"GLMsingle file outputs in: {glmsingle_outdir}")
    return results


# ── main ─────────────────────────────────────────────────────────────────────

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
        "--session", required=True,
        help="Session ID (e.g. ses-04)",
    )
    parser.add_argument(
        "--pipeline", required=True, choices=["original", "nordic", "both"],
        help="Which fMRIPrep pipeline to use",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Override output directory (default: derivatives/nordic/validation/glmsingle/{subject}/{pipeline})",
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
    session = args.session

    pipelines = (
        ["original", "nordic"] if args.pipeline == "both"
        else [args.pipeline]
    )

    for pipeline in pipelines:
        deriv_root = FMRIPREP_ORIG if pipeline == "original" else FMRIPREP_NORDIC

        if args.output_dir is not None:
            output_dir = args.output_dir
        else:
            output_dir = OUTPUT_BASE / subject / pipeline

        output_dir.mkdir(parents=True, exist_ok=True)
        run_glmsingle(subject, session, pipeline, deriv_root, output_dir,
                       use_spike_regressors=args.spike_regressors)

    print("\nDone.")


if __name__ == "__main__":
    main()
