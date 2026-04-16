#!/usr/bin/env python3
"""
isc_confound_comparison.py — Evaluate 21 confound regression models on ISC/WSC.

Computes inter-subject correlation (ISC) and within-subject correlation (WSC)
for NATencoding movie-watching fMRI data across confound regression models.

ISC: pairwise correlations between subjects watching the same movie in the
     same session (all movies eligible).
WSC: pairwise correlations across sessions for repeated movies within a
     subject ("The Bench", "From Dad To Son"). Test-retest reliability.

Cortical ROIs (V1, MT+, EAC, IFG) use fsaverage6 surface BOLD.
Hippocampus uses MNI152 volumetric BOLD + FreeSurfer aseg labels.

Usage:
    python isc_confound_comparison.py [--sessions ses-19 ses-20 ...] [--output-dir PATH]
    python isc_confound_comparison.py --all-sessions  # ses-19 through ses-28

Requires: neuroconda3 environment
    /home/bhutch/.conda/envs/neuroconda3/bin/python isc_confound_comparison.py
"""

import argparse
import json
import sys
import time
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# ── natsort shim (not installed in neuroconda3) ──────────────────────────
import re
import types

def _natsorted(seq, **kwargs):
    """Natural sort: split strings into (text, number) chunks."""
    def _key(s):
        return [int(c) if c.isdigit() else c.lower()
                for c in re.split(r'(\d+)', str(s))]
    return sorted(seq, key=_key)

natsort_shim = types.ModuleType("natsort")
natsort_shim.natsorted = _natsorted
sys.modules["natsort"] = natsort_shim

# ── import Nastase confound utilities ─────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
ISC_CONFOUNDS_DIR = SCRIPT_DIR.parent.parent.parent / "isc-confounds"
sys.path.insert(0, str(ISC_CONFOUNDS_DIR))
from extract_confounds import load_confounds, extract_confounds

# Load model specifications (generated from model_specification.py)
with open(SCRIPT_DIR / "model_meta.json") as _f:
    model_meta = json.load(_f)

# ── paths ─────────────────────────────────────────────────────────────────

BIDS_ROOT = Path("/gpfs/projects/hulacon/shared/mmmdata")
FMRIPREP = BIDS_ROOT / "derivatives" / "fmriprep"

SUBJECTS = ["sub-03", "sub-04", "sub-05"]
ALL_SESSIONS = [f"ses-{i}" for i in range(19, 29)]  # ses-19 through ses-28
ENCODING_RUNS = ["run-01", "run-02"]

ROI_MASK_DIR = ISC_CONFOUNDS_DIR / "afni" / "tpl-fsaverage6"
CORTICAL_ROIS = ["V1", "MT+", "EAC", "IFG"]
HEMIS = ["L", "R"]

ASEG_HIPPO_LABELS = [17, 53]  # L and R hippocampus

# Movies repeated across all sessions (for WSC)
REPEATED_MOVIES = {"the bench", "from dad to son"}

MNI_SPACE = "MNI152NLin2009cAsym_res-2"

# ── path helpers ──────────────────────────────────────────────────────────


def surface_bold_path(subject, session, run, hemi):
    return (
        FMRIPREP / subject / session / "func"
        / f"{subject}_{session}_task-NATencoding_{run}_hemi-{hemi}_space-fsaverage6_bold.func.gii"
    )


def volume_bold_path(subject, session, run):
    return (
        FMRIPREP / subject / session / "func"
        / f"{subject}_{session}_task-NATencoding_{run}_space-{MNI_SPACE}_desc-preproc_bold.nii.gz"
    )


def volume_mask_path(subject, session, run):
    return (
        FMRIPREP / subject / session / "func"
        / f"{subject}_{session}_task-NATencoding_{run}_space-{MNI_SPACE}_desc-brain_mask.nii.gz"
    )


def confounds_tsv_path(subject, session, run):
    return (
        FMRIPREP / subject / session / "func"
        / f"{subject}_{session}_task-NATencoding_{run}_desc-confounds_timeseries.tsv"
    )


def events_path(subject, session, run):
    return (
        BIDS_ROOT / subject / session / "func"
        / f"{subject}_{session}_task-NATencoding_{run}_events.tsv"
    )


def get_tr(subject, session, run):
    json_path = (
        BIDS_ROOT / subject / session / "func"
        / f"{subject}_{session}_task-NATencoding_{run}_bold.json"
    )
    with open(json_path) as f:
        return json.load(f)["RepetitionTime"]


# ── data loading ──────────────────────────────────────────────────────────


def load_gifti_bold(gifti_path):
    """Load a surface BOLD GIFTI as (n_vertices, n_timepoints)."""
    gii = nib.load(str(gifti_path))
    return np.column_stack([d.data for d in gii.darrays])


def load_roi_masks():
    """Load cortical ROI masks for both hemispheres.

    Returns dict: {(roi_name, hemi): boolean array (40962,)}
    """
    masks = {}
    for roi in CORTICAL_ROIS:
        for hemi in HEMIS:
            npy_path = ROI_MASK_DIR / f"tpl-fsaverage6_hemi-{hemi}_desc-{roi}_mask.npy"
            mask = np.load(str(npy_path)).astype(bool)
            masks[(roi, hemi)] = mask
            print(f"  Cortical ROI {roi} hemi-{hemi}: {mask.sum()} vertices")
    return masks


def load_hippo_mask(bold_ref_img, subject):
    """Load bilateral hippocampus mask from FreeSurfer aseg, resampled to BOLD space.

    Returns boolean mask in BOLD voxel space, or None if aseg not found.
    """
    from nilearn.image import resample_to_img

    aseg_path = FMRIPREP / "sourcedata" / "freesurfer" / subject / "mri" / "aparc+aseg.mgz"
    if not aseg_path.exists():
        print(f"  WARNING: aseg not found for {subject}, skipping hippocampus")
        return None

    aseg_img = nib.load(str(aseg_path))
    aseg_resampled = resample_to_img(aseg_img, bold_ref_img, interpolation="nearest")
    aseg_data = np.asarray(aseg_resampled.dataobj).astype(int)
    hippo_mask = np.isin(aseg_data, ASEG_HIPPO_LABELS)
    print(f"  Hippocampus ({subject}): {hippo_mask.sum()} voxels")
    return hippo_mask


# ── processing ────────────────────────────────────────────────────────────


def regress_confounds(data, confound_matrix):
    """Remove confounds from data via OLS.

    data: (n_features, n_timepoints)
    confound_matrix: (n_timepoints, n_confounds) or None
    Returns residuals with same shape.
    """
    if confound_matrix is None or confound_matrix.shape[1] == 0:
        return data

    X = np.nan_to_num(confound_matrix.copy(), nan=0.0)
    n_tp = X.shape[0]
    X = np.column_stack([np.ones(n_tp), X])

    betas = np.linalg.lstsq(X, data.T, rcond=None)[0]
    return data - (X @ betas).T


def extract_movie_timeseries(roi_ts, events_df, tr):
    """Extract per-movie z-scored timeseries from a 1D ROI timeseries.

    Returns dict: {movie_name_lower: {"display_name": str, "ts": 1D array}}
    """
    movies = events_df[events_df["trial_type"] == "movie"]
    result = {}

    for _, row in movies.iterrows():
        name = row["movie_name"]
        onset_vol = int(np.round(row["onset"] / tr))
        dur_vols = int(np.round(row["duration"] / tr))
        end_vol = min(onset_vol + dur_vols, len(roi_ts))

        ts = roi_ts[onset_vol:end_vol].copy().astype(np.float64)
        if ts.std() > 0:
            ts = (ts - ts.mean()) / ts.std()

        key = name.lower()
        result[key] = {"display_name": name, "ts": ts}

    return result


def compute_pairwise_correlations(ts_dict):
    """Compute all pairwise Pearson r values.

    ts_dict: {label: 1D array}
    Returns list of (pair_label, r)
    """
    labels = sorted(ts_dict.keys())
    if len(labels) < 2:
        return []

    min_len = min(len(ts_dict[l]) for l in labels)
    pairs = []
    for l1, l2 in combinations(labels, 2):
        ts1 = ts_dict[l1][:min_len]
        ts2 = ts_dict[l2][:min_len]
        if ts1.std() > 0 and ts2.std() > 0:
            r = np.corrcoef(ts1, ts2)[0, 1]
            pairs.append((f"{l1}|{l2}", r))
    return pairs


def get_confound_matrix(confounds_df, confounds_meta, model_id):
    """Extract confound matrix for a model, or None for model 0."""
    spec = model_meta[model_id]
    if not spec["confounds"] and not any(
        k in spec for k in ["a_comp_cor", "t_comp_cor", "c_comp_cor", "w_comp_cor"]
    ):
        return None
    return extract_confounds(confounds_df, confounds_meta, spec).values


# ── main ──────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--sessions", nargs="+", default=None,
        help="Sessions to process (default: ses-19 only)",
    )
    parser.add_argument(
        "--all-sessions", action="store_true",
        help="Process all 10 sessions (ses-19 through ses-28)",
    )
    parser.add_argument(
        "--no-hippocampus", action="store_true",
        help="Skip hippocampus volumetric analysis",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=BIDS_ROOT / "derivatives" / "qc" / "isc_confound_comparison",
    )
    args = parser.parse_args()

    if args.all_sessions:
        sessions = ALL_SESSIONS
    elif args.sessions:
        sessions = args.sessions
    else:
        sessions = ["ses-19"]

    do_hippo = not args.no_hippocampus
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    fig_dir = out / "figures"
    fig_dir.mkdir(exist_ok=True)

    models = sorted(model_meta.keys(), key=int)

    print(f"Sessions: {sessions}")
    print(f"Models: {len(models)} ({models[0]}–{models[-1]})")
    print(f"Subjects: {SUBJECTS}")
    print(f"Hippocampus: {'yes' if do_hippo else 'no'}")
    print(f"Output: {out}")
    print()

    # ── Load cortical ROI masks (constant across subjects) ───────────────
    print("Loading cortical ROI masks...")
    cortical_masks = load_roi_masks()

    # ── Load hippocampus masks (per-subject, need a BOLD ref) ────────────
    hippo_masks = {}  # {subject: 3D bool array}
    if do_hippo:
        print("\nLoading hippocampus masks...")
        for subject in SUBJECTS:
            # Find first available volumetric BOLD as reference
            for session in sessions:
                for run in ENCODING_RUNS:
                    ref_path = volume_bold_path(subject, session, run)
                    if ref_path.exists():
                        ref_img = nib.load(str(ref_path))
                        hippo_masks[subject] = load_hippo_mask(ref_img, subject)
                        break
                if subject in hippo_masks:
                    break

    # ── Build ROI list ───────────────────────────────────────────────────
    # Each entry: (roi_label, space, hemi_or_none)
    # roi_label used in output; space determines which BOLD to load
    roi_specs = []
    for roi in CORTICAL_ROIS:
        for hemi in HEMIS:
            roi_specs.append((roi, hemi, "surface"))
    if do_hippo and hippo_masks:
        roi_specs.append(("Hippocampus", "bilateral", "volume"))

    print(f"\nROIs: {len(roi_specs)} ({len(CORTICAL_ROIS)} cortical × 2 hemis"
          f"{' + hippocampus' if do_hippo and hippo_masks else ''})")

    # ══════════════════════════════════════════════════════════════════════
    # PHASE A: Extract movie timeseries per (model, roi, subject, session, run)
    #
    # Memory-efficient: load one run at a time, extract all ROI timeseries,
    # then discard BOLD. Store only 1D movie timeseries.
    # ══════════════════════════════════════════════════════════════════════

    # movie_ts[model_id][(roi, hemi, subject, session, movie_key)] = 1D array
    # fd_values[(subject, session, run)] = float
    # movie_display_names[movie_key] = str

    print("\n" + "=" * 70)
    print("Extracting movie timeseries (per run, memory-efficient)...")
    print("=" * 70)

    movie_ts = {m: {} for m in models}
    fd_values = {}
    movie_display_names = {}
    t0 = time.time()

    for si, session in enumerate(sessions):
        for subject in SUBJECTS:
            for run in ENCODING_RUNS:
                # Check existence
                gii_path = surface_bold_path(subject, session, run, "L")
                if not gii_path.exists():
                    print(f"  SKIP — {subject} {session} {run} (no surface BOLD)")
                    continue

                events_df = pd.read_csv(events_path(subject, session, run), sep="\t")
                tr = get_tr(subject, session, run)

                # Load confounds once per run
                conf_fn = str(confounds_tsv_path(subject, session, run))
                confounds_df, confounds_meta = load_confounds(conf_fn)
                fd_values[(subject, session, run)] = confounds_df[
                    "framewise_displacement"
                ].mean(skipna=True)

                # Pre-extract confound matrices for all models
                conf_mats = {}
                for model_id in models:
                    conf_mats[model_id] = get_confound_matrix(
                        confounds_df, confounds_meta, model_id
                    )

                # ── Surface cortical ROIs ────────────────────────────────
                for hemi in HEMIS:
                    gii_path = surface_bold_path(subject, session, run, hemi)
                    bold_surf = load_gifti_bold(gii_path)  # (40962, n_trs)

                    for roi in CORTICAL_ROIS:
                        mask = cortical_masks[(roi, hemi)]

                        for model_id in models:
                            residuals = regress_confounds(bold_surf[mask], conf_mats[model_id])
                            roi_ts = residuals.mean(axis=0)
                            mts = extract_movie_timeseries(roi_ts, events_df, tr)

                            for movie_key, info in mts.items():
                                ts_key = (roi, hemi, subject, session, movie_key)
                                movie_ts[model_id][ts_key] = info["ts"]
                                movie_display_names[movie_key] = info["display_name"]

                    del bold_surf  # free surface BOLD memory

                # ── Volumetric hippocampus ────────────────────────────────
                if do_hippo and subject in hippo_masks and hippo_masks[subject] is not None:
                    vol_path = volume_bold_path(subject, session, run)
                    if vol_path.exists():
                        vol_img = nib.load(str(vol_path))
                        mask_img = nib.load(str(volume_mask_path(subject, session, run)))
                        vol_data = vol_img.get_fdata(dtype=np.float32)
                        brain_mask = mask_img.get_fdata().astype(bool)
                        valid = hippo_masks[subject] & brain_mask
                        hippo_voxels = vol_data[valid]  # (n_voxels, n_trs)

                        for model_id in models:
                            residuals = regress_confounds(hippo_voxels, conf_mats[model_id])
                            roi_ts = residuals.mean(axis=0)
                            mts = extract_movie_timeseries(roi_ts, events_df, tr)

                            for movie_key, info in mts.items():
                                ts_key = ("Hippocampus", "bilateral", subject, session, movie_key)
                                movie_ts[model_id][ts_key] = info["ts"]
                                movie_display_names[movie_key] = info["display_name"]

                        del vol_data, hippo_voxels

                print(f"  {subject} {session} {run} done "
                      f"[{time.time() - t0:.0f}s elapsed]")

    print(f"\nExtraction complete. {time.time() - t0:.0f}s total.")
    n_ts = len(movie_ts[models[0]])
    print(f"Timeseries per model: {n_ts}")

    # ══════════════════════════════════════════════════════════════════════
    # PHASE B: Compute ISC (across subjects within session)
    # ══════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 70)
    print("Computing ISC...")
    print("=" * 70)

    isc_results = []

    for model_id in models:
        ts_store = movie_ts[model_id]

        for roi, hemi, space in roi_specs:
            for session in sessions:
                # Group by movie: {movie_key: {subject: ts}}
                movies_this_session = {}
                for subject in SUBJECTS:
                    for movie_key in movie_display_names:
                        ts_key = (roi, hemi, subject, session, movie_key)
                        if ts_key in ts_store:
                            movies_this_session.setdefault(movie_key, {})[subject] = ts_store[ts_key]

                for movie_key, subj_ts in movies_this_session.items():
                    pairs = compute_pairwise_correlations(subj_ts)
                    for pair_label, r in pairs:
                        isc_results.append({
                            "model": int(model_id),
                            "roi": roi,
                            "hemi": hemi,
                            "movie": movie_display_names[movie_key],
                            "session": session,
                            "subject_pair": pair_label,
                            "r": r,
                        })

    isc_df = pd.DataFrame(isc_results)
    isc_df.to_csv(out / "isc_by_model.tsv", sep="\t", index=False)
    print(f"Saved {len(isc_df)} ISC observations to isc_by_model.tsv")

    # ══════════════════════════════════════════════════════════════════════
    # PHASE C: Compute WSC (across sessions within subject, repeated movies)
    # ══════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 70)
    print("Computing WSC (within-subject correlation for repeated movies)...")
    print("=" * 70)

    wsc_results = []

    for model_id in models:
        ts_store = movie_ts[model_id]

        for roi, hemi, space in roi_specs:
            for subject in SUBJECTS:
                for movie_key in REPEATED_MOVIES:
                    # Collect timeseries across sessions
                    session_ts = {}
                    for session in sessions:
                        ts_key = (roi, hemi, subject, session, movie_key)
                        if ts_key in ts_store:
                            session_ts[session] = ts_store[ts_key]

                    if len(session_ts) < 2:
                        continue

                    pairs = compute_pairwise_correlations(session_ts)
                    for pair_label, r in pairs:
                        wsc_results.append({
                            "model": int(model_id),
                            "roi": roi,
                            "hemi": hemi,
                            "movie": movie_display_names.get(movie_key, movie_key),
                            "subject": subject,
                            "session_pair": pair_label,
                            "r": r,
                        })

    wsc_df = pd.DataFrame(wsc_results)
    if not wsc_df.empty:
        wsc_df.to_csv(out / "wsc_by_model.tsv", sep="\t", index=False)
        print(f"Saved {len(wsc_df)} WSC observations to wsc_by_model.tsv")
    else:
        print("No WSC observations (need multiple sessions with repeated movies)")

    # ══════════════════════════════════════════════════════════════════════
    # PHASE D: Summary table (ISC + WSC, Fisher z-transform)
    # ══════════════════════════════════════════════════════════════════════

    summary_rows = []

    for (model, roi, hemi), grp in isc_df.groupby(["model", "roi", "hemi"]):
        z = np.arctanh(grp["r"].values)
        summary_rows.append({
            "model": model, "roi": roi, "hemi": hemi, "track": "ISC",
            "mean_r": np.tanh(z.mean()), "sd_r": z.std(), "n_obs": len(grp),
        })

    if not wsc_df.empty:
        for (model, roi, hemi), grp in wsc_df.groupby(["model", "roi", "hemi"]):
            z = np.arctanh(grp["r"].values)
            summary_rows.append({
                "model": model, "roi": roi, "hemi": hemi, "track": "WSC",
                "mean_r": np.tanh(z.mean()), "sd_r": z.std(), "n_obs": len(grp),
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out / "summary_by_model.tsv", sep="\t", index=False)
    print(f"\nSaved summary ({len(summary_df)} rows) to summary_by_model.tsv")

    # ══════════════════════════════════════════════════════════════════════
    # PHASE E: ISC–FD and WSC–FD correlations (Analysis 2)
    # ══════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 70)
    print("Computing correlation–FD relationships...")
    print("=" * 70)

    # ISC–FD: for each subject × session, compute mean ISC and mean FD
    isc_fd_results = []
    for model_id_int in sorted(isc_df["model"].unique()):
        for roi in isc_df["roi"].unique():
            for hemi in isc_df[isc_df["roi"] == roi]["hemi"].unique():
                sub = isc_df[
                    (isc_df["model"] == model_id_int) &
                    (isc_df["roi"] == roi) &
                    (isc_df["hemi"] == hemi)
                ]
                isc_vals, fd_vals = [], []
                for session in sessions:
                    for subject in SUBJECTS:
                        sess = sub[sub["session"] == session]
                        subj_pairs = sess[sess["subject_pair"].str.contains(subject, regex=False)]
                        if subj_pairs.empty:
                            continue
                        mean_isc = np.tanh(np.arctanh(subj_pairs["r"].values).mean())
                        fd_list = [fd_values[k] for k in
                                   ((subject, session, r) for r in ENCODING_RUNS)
                                   if k in fd_values]
                        if not fd_list:
                            continue
                        isc_vals.append(mean_isc)
                        fd_vals.append(np.mean(fd_list))

                if len(isc_vals) >= 3:
                    rho, p = spearmanr(isc_vals, fd_vals)
                    isc_fd_results.append({
                        "model": model_id_int, "roi": roi, "hemi": hemi,
                        "spearman_r": rho, "p": p, "n": len(isc_vals),
                    })

    if isc_fd_results:
        isc_fd_df = pd.DataFrame(isc_fd_results)
        isc_fd_df.to_csv(out / "isc_fd_correlation.tsv", sep="\t", index=False)
        print(f"Saved ISC-FD correlations to isc_fd_correlation.tsv")

    # WSC–FD: for each session-pair, mean FD = average of both sessions
    wsc_fd_results = []
    if not wsc_df.empty:
        for model_id_int in sorted(wsc_df["model"].unique()):
            for roi in wsc_df["roi"].unique():
                for hemi in wsc_df[wsc_df["roi"] == roi]["hemi"].unique():
                    sub = wsc_df[
                        (wsc_df["model"] == model_id_int) &
                        (wsc_df["roi"] == roi) &
                        (wsc_df["hemi"] == hemi)
                    ]
                    wsc_vals, fd_vals = [], []
                    for _, row in sub.iterrows():
                        s1, s2 = row["session_pair"].split("|")
                        subject = row["subject"]
                        fd1 = np.mean([fd_values.get((subject, s1, r), np.nan)
                                       for r in ENCODING_RUNS])
                        fd2 = np.mean([fd_values.get((subject, s2, r), np.nan)
                                       for r in ENCODING_RUNS])
                        if np.isnan(fd1) or np.isnan(fd2):
                            continue
                        wsc_vals.append(row["r"])
                        fd_vals.append((fd1 + fd2) / 2)

                    if len(wsc_vals) >= 3:
                        rho, p = spearmanr(wsc_vals, fd_vals)
                        wsc_fd_results.append({
                            "model": model_id_int, "roi": roi, "hemi": hemi,
                            "spearman_r": rho, "p": p, "n": len(wsc_vals),
                        })

    if wsc_fd_results:
        wsc_fd_df = pd.DataFrame(wsc_fd_results)
        wsc_fd_df.to_csv(out / "wsc_fd_correlation.tsv", sep="\t", index=False)
        print(f"Saved WSC-FD correlations to wsc_fd_correlation.tsv")

    # ══════════════════════════════════════════════════════════════════════
    # PHASE F: Visualization
    # ══════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 70)
    print("Generating figures...")
    print("=" * 70)

    plot_isc_wsc_by_model(summary_df, fig_dir)
    plot_isc_vs_wsc_scatter(summary_df, fig_dir)
    if isc_fd_results:
        plot_fd_correlation(pd.DataFrame(isc_fd_results), "ISC", fig_dir)
    if wsc_fd_results:
        plot_fd_correlation(pd.DataFrame(wsc_fd_results), "WSC", fig_dir)

    print(f"\nDone. Results in {out}/")
    print(f"Total time: {time.time() - t0:.0f}s")


# ══════════════════════════════════════════════════════════════════════════
# Plotting functions
# ══════════════════════════════════════════════════════════════════════════


def plot_isc_wsc_by_model(summary_df, fig_dir):
    """Horizontal bar plot of mean ISC and WSC per model, one panel per ROI."""
    tracks = sorted(summary_df["track"].unique())
    rois = sorted(summary_df["roi"].unique())
    n_rois = len(rois)
    n_tracks = len(tracks)

    fig, axes = plt.subplots(n_tracks, n_rois,
                              figsize=(4 * n_rois, 4 * n_tracks + 1),
                              sharey=True, squeeze=False)

    models = sorted(summary_df["model"].unique())
    y_pos = np.arange(len(models))
    width = 0.6

    for ti, track in enumerate(tracks):
        for ri, roi in enumerate(rois):
            ax = axes[ti, ri]
            sub = summary_df[(summary_df["track"] == track) & (summary_df["roi"] == roi)]
            mean_by_model = sub.groupby("model")["mean_r"].mean()

            vals = [mean_by_model.get(m, 0) for m in models]
            colors = ["steelblue" if v >= 0 else "salmon" for v in vals]
            ax.barh(y_pos, vals, height=width, color=colors, edgecolor="gray", linewidth=0.5)
            ax.set_yticks(y_pos)
            ax.set_yticklabels([str(m) for m in models], fontsize=8)
            ax.axvline(0, color="black", linewidth=0.5)

            if 0 in mean_by_model.index:
                ax.axvline(mean_by_model[0], color="red", linewidth=0.8,
                            linestyle="--", alpha=0.7, label="Model 0")
                ax.legend(fontsize=6, loc="lower right")

            if ti == 0:
                ax.set_title(roi, fontsize=11)
            if ri == 0:
                ax.set_ylabel(f"{track}\nConfound Model", fontsize=10)
            if ti == n_tracks - 1:
                ax.set_xlabel("Mean r", fontsize=9)

    fig.suptitle("ISC / WSC by Confound Regression Model (averaged across hemispheres)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(fig_dir / "isc_wsc_by_model.png", dpi=150)
    plt.close(fig)
    print(f"  Saved isc_wsc_by_model.png")


def plot_isc_vs_wsc_scatter(summary_df, fig_dir):
    """Scatter plot: mean ISC vs mean WSC per model, one panel per ROI."""
    if "WSC" not in summary_df["track"].values:
        return

    rois = sorted(summary_df["roi"].unique())
    n_rois = len(rois)
    fig, axes = plt.subplots(1, n_rois, figsize=(4 * n_rois, 4), squeeze=False)

    models = sorted(summary_df["model"].unique())

    for ri, roi in enumerate(rois):
        ax = axes[0, ri]
        isc_sub = summary_df[(summary_df["track"] == "ISC") & (summary_df["roi"] == roi)]
        wsc_sub = summary_df[(summary_df["track"] == "WSC") & (summary_df["roi"] == roi)]

        isc_by_model = isc_sub.groupby("model")["mean_r"].mean()
        wsc_by_model = wsc_sub.groupby("model")["mean_r"].mean()

        for m in models:
            if m in isc_by_model.index and m in wsc_by_model.index:
                ax.scatter(isc_by_model[m], wsc_by_model[m], s=40, zorder=3)
                ax.annotate(str(m), (isc_by_model[m], wsc_by_model[m]),
                            fontsize=7, ha="left", va="bottom")

        ax.set_xlabel("Mean ISC")
        ax.set_ylabel("Mean WSC")
        ax.set_title(roi)

        # Identity line
        lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
                max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(lims, lims, "k--", alpha=0.3, linewidth=0.8)

    fig.suptitle("ISC vs WSC by Confound Model", fontsize=13)
    fig.tight_layout()
    fig.savefig(fig_dir / "isc_vs_wsc_scatter.png", dpi=150)
    plt.close(fig)
    print(f"  Saved isc_vs_wsc_scatter.png")


def plot_fd_correlation(fd_df, track_name, fig_dir):
    """Bar plot of Spearman r(correlation, FD) per model."""
    rois = sorted(fd_df["roi"].unique())
    n_rois = len(rois)

    fig, axes = plt.subplots(1, n_rois, figsize=(4 * n_rois, 8), sharey=True)
    if n_rois == 1:
        axes = [axes]

    models = sorted(fd_df["model"].unique())
    y_pos = np.arange(len(models))

    for ax, roi in zip(axes, rois):
        roi_df = fd_df[fd_df["roi"] == roi]
        mean_by_model = roi_df.groupby("model")["spearman_r"].mean()

        vals = [mean_by_model.get(m, 0) for m in models]
        colors = ["salmon" if v < 0 else "steelblue" for v in vals]
        ax.barh(y_pos, vals, color=colors, edgecolor="gray", linewidth=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([str(m) for m in models])
        ax.set_xlabel(f"Spearman r({track_name}, FD)")
        ax.set_title(roi)
        ax.axvline(0, color="black", linewidth=0.5)

    axes[0].set_ylabel("Confound Model")
    fname = f"{track_name.lower()}_fd_by_model.png"
    fig.suptitle(f"{track_name}–FD Correlation by Confound Model", fontsize=13)
    fig.tight_layout()
    fig.savefig(fig_dir / fname, dpi=150)
    plt.close(fig)
    print(f"  Saved {fname}")


if __name__ == "__main__":
    main()
