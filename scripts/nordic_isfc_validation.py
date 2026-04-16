#!/usr/bin/env python3
"""
nordic_isfc_validation.py — ISFC comparison for original vs NORDIC fMRIPrep outputs.

Computes inter-subject functional correlation (ISFC) within ROIs for each
movie clip in the NATencoding task. Movies are shown in different orders
across subjects, so timeseries are aligned by movie onset/duration from
events files.

Usage:
    python nordic_isfc_validation.py [--output-dir PATH]
"""

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
from nilearn.image import resample_to_img
import numpy as np
import pandas as pd

# ── paths ────────────────────────────────────────────────────────────────────

BIDS_ROOT = Path("/gpfs/projects/hulacon/shared/mmmdata")
FMRIPREP_ORIG = BIDS_ROOT / "derivatives" / "fmriprep"
FMRIPREP_NORDIC = BIDS_ROOT / "derivatives" / "fmriprep_nordic"

SUBJECTS = ["sub-03", "sub-04", "sub-05"]
SESSION = "ses-19"
SPACE = "MNI152NLin2009cAsym_res-2"
ENCODING_RUNS = ["run-01", "run-02"]

# ── ROI definitions (same as nordic_validation.py) ──────────────────────────

SCHAEFER_ATLAS = (
    BIDS_ROOT / "derivatives" / "atlases" / "tpl-MNI152NLin2009cAsym" / "anat"
    / "tpl-MNI152NLin2009cAsym_atlas-Schaefer2018_seg-17n_scale-400_res-2_dseg.nii.gz"
)
SCHAEFER_LUT = (
    BIDS_ROOT / "derivatives" / "atlases" / "tpl-MNI152NLin2009cAsym" / "anat"
    / "tpl-MNI152NLin2009cAsym_atlas-Schaefer2018_seg-17n_scale-400_res-2_dseg.tsv"
)

SCHAEFER_ROIS = {
    "V1 (VisCent)": "VisCent_ExStr",
    "Angular gyrus": "DefaultA_IPL",
    "vmPFC": "DefaultA_PFCm",
}

ASEG_HIPPO_LABELS = [17, 53]


# ── helpers ──────────────────────────────────────────────────────────────────

def bold_path(deriv_root, subject, run):
    return (
        deriv_root / subject / SESSION / "func"
        / f"{subject}_{SESSION}_task-NATencoding_{run}_space-{SPACE}_desc-preproc_bold.nii.gz"
    )


def mask_path(deriv_root, subject, run):
    return (
        deriv_root / subject / SESSION / "func"
        / f"{subject}_{SESSION}_task-NATencoding_{run}_space-{SPACE}_desc-brain_mask.nii.gz"
    )


def events_path(subject, run):
    return (
        BIDS_ROOT / subject / SESSION / "func"
        / f"{subject}_{SESSION}_task-NATencoding_{run}_events.tsv"
    )


def get_tr(subject, run):
    json_path = (
        BIDS_ROOT / subject / SESSION / "func"
        / f"{subject}_{SESSION}_task-NATencoding_{run}_bold.json"
    )
    with open(json_path) as f:
        return json.load(f)["RepetitionTime"]


def load_roi_masks(bold_ref_img, subject):
    """Build ROI boolean masks in MNI BOLD space."""
    rois = {}

    # Schaefer cortical parcels
    schaefer_img = nib.load(str(SCHAEFER_ATLAS))
    schaefer_resampled = resample_to_img(
        schaefer_img, bold_ref_img, interpolation="nearest"
    )
    schaefer_data = np.asarray(schaefer_resampled.dataobj).astype(int)
    lut = pd.read_csv(SCHAEFER_LUT, sep="\t")

    for roi_name, pattern in SCHAEFER_ROIS.items():
        matching = lut[lut["name"].str.contains(pattern, na=False)]
        indices = set(matching["index"].values)
        mask = np.isin(schaefer_data, list(indices))
        rois[roi_name] = mask
        print(f"  ROI '{roi_name}': {len(indices)} parcels, {mask.sum()} voxels")

    # Hippocampus from FreeSurfer aseg
    aseg_mgz = FMRIPREP_ORIG / "sourcedata" / "freesurfer" / subject / "mri" / "aparc+aseg.mgz"
    if aseg_mgz.exists():
        aseg_img = nib.load(str(aseg_mgz))
        aseg_resampled = resample_to_img(
            aseg_img, bold_ref_img, interpolation="nearest"
        )
        aseg_data = np.asarray(aseg_resampled.dataobj).astype(int)
        hippo_mask = np.isin(aseg_data, ASEG_HIPPO_LABELS)
        rois["Hippocampus"] = hippo_mask
        print(f"  ROI 'Hippocampus': {hippo_mask.sum()} voxels")

    return rois


def extract_movie_timeseries(bold_img, mask_data, roi_mask, events_df, tr):
    """
    Extract mean ROI timeseries for each movie clip.

    Returns dict: movie_name -> 1D array of mean ROI timeseries.
    """
    data = bold_img.get_fdata(dtype=np.float32)
    valid = roi_mask & mask_data
    voxel_ts = data[valid]  # (n_voxels, n_timepoints)

    # Mean across voxels -> single ROI timeseries
    roi_ts = voxel_ts.mean(axis=0)  # (n_timepoints,)

    movies = events_df[events_df["trial_type"] == "movie"].copy()
    movie_ts = {}

    for _, row in movies.iterrows():
        name = row["movie_name"]
        onset_vol = int(np.round(row["onset"] / tr))
        dur_vols = int(np.round(row["duration"] / tr))
        end_vol = min(onset_vol + dur_vols, len(roi_ts))

        ts = roi_ts[onset_vol:end_vol].copy()
        # Z-score within movie
        if ts.std() > 0:
            ts = (ts - ts.mean()) / ts.std()
        movie_ts[name] = ts

    return movie_ts


def compute_isfc_for_movie(ts_dict):
    """
    Compute pairwise ISFC for a single movie across subjects.

    ts_dict: {subject: 1D timeseries array}
    Returns mean pairwise Pearson correlation.
    """
    subjects = list(ts_dict.keys())
    if len(subjects) < 2:
        return np.nan

    # Truncate to shortest timeseries (minor TR differences)
    min_len = min(len(ts_dict[s]) for s in subjects)
    correlations = []

    for s1, s2 in combinations(subjects, 2):
        ts1 = ts_dict[s1][:min_len]
        ts2 = ts_dict[s2][:min_len]
        if ts1.std() > 0 and ts2.std() > 0:
            r = np.corrcoef(ts1, ts2)[0, 1]
            correlations.append(r)

    return np.mean(correlations) if correlations else np.nan


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path,
                        default=BIDS_ROOT / "derivatives" / "nordic" / "validation" / "isfc",
                        help="Directory for output plots and summary")
    args = parser.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    # Load ROI masks once (using first subject's first run mask as reference)
    ref_subject = SUBJECTS[0]
    ref_mask_file = mask_path(FMRIPREP_ORIG, ref_subject, ENCODING_RUNS[0])
    ref_mask_img = nib.load(str(ref_mask_file))
    print("Loading ROI masks...")
    roi_masks = load_roi_masks(ref_mask_img, ref_subject)

    # For each pipeline (orig, nordic), for each subject, for each run:
    # extract per-movie ROI timeseries
    results = []

    for pipeline_name, deriv_root in [("Original", FMRIPREP_ORIG),
                                       ("NORDIC", FMRIPREP_NORDIC)]:
        print(f"\n{'='*60}")
        print(f"Pipeline: {pipeline_name}")
        print(f"{'='*60}")

        # movie_ts[roi_name][movie_name][subject] = 1D timeseries
        movie_ts = {roi: {} for roi in roi_masks}

        for subject in SUBJECTS:
            for run in ENCODING_RUNS:
                bold_file = bold_path(deriv_root, subject, run)
                mask_file = mask_path(deriv_root, subject, run)
                events_file = events_path(subject, run)

                if not bold_file.exists():
                    print(f"  SKIP — {bold_file.name} not found")
                    continue

                print(f"  Loading {subject} {run}...")
                bold_img = nib.load(str(bold_file))
                mask_img = nib.load(str(mask_file))
                mask_data = mask_img.get_fdata().astype(bool)
                events_df = pd.read_csv(events_file, sep="\t")
                tr = get_tr(subject, run)

                for roi_name, roi_mask in roi_masks.items():
                    ts_per_movie = extract_movie_timeseries(
                        bold_img, mask_data, roi_mask, events_df, tr
                    )
                    for movie_name, ts in ts_per_movie.items():
                        if movie_name not in movie_ts[roi_name]:
                            movie_ts[roi_name][movie_name] = {}
                        movie_ts[roi_name][movie_name][subject] = ts

        # Compute ISFC per movie per ROI
        for roi_name in roi_masks:
            for movie_name, subj_ts in movie_ts[roi_name].items():
                isfc = compute_isfc_for_movie(subj_ts)
                n_subjects = len(subj_ts)
                min_len = min(len(v) for v in subj_ts.values()) if subj_ts else 0
                results.append({
                    "pipeline": pipeline_name,
                    "roi": roi_name,
                    "movie": movie_name,
                    "isfc": isfc,
                    "n_subjects": n_subjects,
                    "n_timepoints": min_len,
                })
                print(f"  {roi_name} / {movie_name}: ISFC = {isfc:.4f} "
                      f"(n={n_subjects}, T={min_len})")

    # ── Summary table ────────────────────────────────────────────────────
    df = pd.DataFrame(results)
    df.to_csv(out / "isfc_results.tsv", sep="\t", index=False)

    # Pivot for comparison
    pivot = df.pivot_table(index=["roi", "movie"], columns="pipeline",
                           values="isfc").reset_index()
    if "Original" in pivot.columns and "NORDIC" in pivot.columns:
        pivot["improvement"] = pivot["NORDIC"] - pivot["Original"]
        pivot.to_csv(out / "isfc_comparison.tsv", sep="\t", index=False)

    # ── Summary by ROI ───────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("ISFC Summary (mean across movies)")
    print(f"{'='*60}")
    roi_summary = df.groupby(["pipeline", "roi"])["isfc"].mean().unstack("pipeline")
    if "Original" in roi_summary.columns and "NORDIC" in roi_summary.columns:
        roi_summary["improvement"] = roi_summary["NORDIC"] - roi_summary["Original"]
    print(roi_summary.to_string())
    roi_summary.to_csv(out / "isfc_roi_summary.tsv", sep="\t")

    # ── Plot ─────────────────────────────────────────────────────────────
    roi_names = sorted(df["roi"].unique())
    movies = sorted(df["movie"].unique())

    fig, axes = plt.subplots(1, len(roi_names), figsize=(5 * len(roi_names), 5),
                              sharey=True)
    if len(roi_names) == 1:
        axes = [axes]

    width = 0.35
    x = np.arange(len(movies))

    for ax, roi_name in zip(axes, roi_names):
        orig_vals = []
        nordic_vals = []
        for movie in movies:
            orig_row = df[(df["pipeline"] == "Original") & (df["roi"] == roi_name)
                          & (df["movie"] == movie)]
            nordic_row = df[(df["pipeline"] == "NORDIC") & (df["roi"] == roi_name)
                            & (df["movie"] == movie)]
            orig_vals.append(orig_row["isfc"].values[0] if len(orig_row) else np.nan)
            nordic_vals.append(nordic_row["isfc"].values[0] if len(nordic_row) else np.nan)

        ax.bar(x - width / 2, orig_vals, width, label="Original", color="gray")
        ax.bar(x + width / 2, nordic_vals, width, label="NORDIC", color="steelblue")
        ax.set_title(roi_name, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(movies, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("ISFC (Pearson r)")
        ax.legend(fontsize=8)
        ax.axhline(0, color="black", linewidth=0.5)

    fig.suptitle("Inter-Subject Functional Correlation: Original vs NORDIC", fontsize=13)
    fig.tight_layout()
    fig.savefig(out / "isfc_comparison.png", dpi=150)
    plt.close(fig)

    print(f"\nResults saved to {out}/")


if __name__ == "__main__":
    main()
