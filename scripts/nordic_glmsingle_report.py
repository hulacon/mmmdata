#!/usr/bin/env python3
"""
nordic_glmsingle_report.py — Compare GLMsingle results: original vs NORDIC.

Loads GLMsingle outputs from the NORDIC pilot, computes voxel reliability
and R², and generates comparison figures + summary TSV.

Metrics:
  - Voxel reliability: per-voxel Pearson r of beta profile across 1-vs-2
    split halves of repeated stimulus presentations, averaged over all
    possible splits (matches GLMsingle paper, Prince et al. 2022).
  - R²: variance explained by the GLM, from GLMsingle's built-in output.

Usage:
    python nordic_glmsingle_report.py
    python nordic_glmsingle_report.py --output-dir /path/to/output
"""

import argparse
from collections import defaultdict
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
GLMSINGLE_BASE = BIDS_ROOT / "derivatives" / "nordic" / "validation" / "glmsingle"

SPACE = "MNI152NLin2009cAsym_res-2"
SUBJECTS = ["sub-03", "sub-04", "sub-05"]
PIPELINES = ["original", "nordic"]
BETA_TYPES = [
    ("TYPEA_ONOFF", "b1: AssumeHRF"),
    ("TYPEB_FITHRF", "b2: FitHRF"),
    ("TYPEC_FITHRF_GLMDENOISE", "b3: +GLMdenoise"),
    ("TYPED_FITHRF_GLMDENOISE_RR", "b4: +RidgeReg"),
]

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


def aseg_path(subject):
    return FMRIPREP_ORIG / "sourcedata" / "freesurfer" / subject / "mri" / "aparc+aseg.mgz"


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
        print(f"  ROI '{roi_name}': {mask.sum()} voxels")

    # Hippocampus from FreeSurfer aseg
    aseg_mgz = aseg_path(subject)
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


# ── GLMsingle output loading ────────────────────────────────────────────────

def load_glmsingle_output(subject, pipeline, beta_type):
    """Load full GLMsingle output dict for a beta type.

    Returns:
        data: dict with keys like 'betasmd', 'R2', etc. (or None)
        stimorder: 1D array mapping each trial to its condition index
    """
    base = GLMSINGLE_BASE / subject / pipeline / "glmsingle_outputs"
    beta_file = base / f"{beta_type}.npy"

    if not beta_file.exists():
        print(f"  WARNING: {beta_file} not found")
        return None, None

    data = np.load(str(beta_file), allow_pickle=True).item()

    # Load design info for stimorder
    design_info = np.load(
        str(base / "DESIGNINFO.npy"), allow_pickle=True
    ).item()
    stimorder = design_info["stimorder"]

    return data, stimorder


# ── metrics ──────────────────────────────────────────────────────────────────

def _vectorized_pearson_r(A, B):
    """Compute per-row Pearson r between matrices A and B.

    Args:
        A, B: (n_voxels, n_conditions) arrays

    Returns:
        r: (n_voxels,) array of correlation coefficients
    """
    A_dm = A - A.mean(axis=1, keepdims=True)
    B_dm = B - B.mean(axis=1, keepdims=True)
    num = (A_dm * B_dm).sum(axis=1)
    denom = np.sqrt((A_dm**2).sum(axis=1) * (B_dm**2).sum(axis=1))
    r = np.full(A.shape[0], np.nan)
    valid = denom > 1e-10
    r[valid] = num[valid] / denom[valid]
    return r


def compute_voxel_reliability(betas, stimorder, mask=None):
    """Compute voxel reliability (split-half Pearson r of beta profiles).

    For each voxel, correlates the condition-beta profile from one split
    against the profile from the remaining presentations. Averages across
    all possible 1-vs-rest splits (up to 3 for conditions with 3 reps).

    This matches the "voxel reliability" metric from Prince et al. (2022).

    Args:
        betas: (X, Y, Z, n_trials)
        stimorder: 1D array (n_trials,) mapping trial -> condition index
        mask: optional 3D boolean array

    Returns:
        reliability_map: 3D array (same spatial dims as betas)
    """
    spatial_shape = betas.shape[:3]
    n_trials = betas.shape[3]

    cond_trials = defaultdict(list)
    for trial_idx, cond_idx in enumerate(stimorder):
        cond_trials[cond_idx].append(trial_idx)

    # Keep only repeated conditions (>= 2 presentations)
    repeated = {k: v for k, v in cond_trials.items() if len(v) >= 2}
    if len(repeated) == 0:
        print("  WARNING: no repeated conditions for reliability")
        return np.full(spatial_shape, np.nan)

    n_reps_list = [len(v) for v in repeated.values()]
    max_splits = max(n_reps_list)
    print(f"  reliability: {len(repeated)} repeated conditions, "
          f"mean {np.mean(n_reps_list):.1f} reps, {max_splits} splits")

    # Flatten spatial dims
    betas_flat = betas.reshape(-1, n_trials)  # (n_voxels, n_trials)
    n_voxels = betas_flat.shape[0]

    if mask is not None:
        voxel_indices = np.where(mask.flatten())[0]
    else:
        voxel_indices = np.arange(n_voxels)

    betas_sel = betas_flat[voxel_indices]  # (n_sel_voxels, n_trials)
    n_sel = len(voxel_indices)
    n_conds = len(repeated)
    cond_list = sorted(repeated.keys())

    all_split_r = []

    for split_idx in range(max_splits):
        half_a = np.full((n_sel, n_conds), np.nan)
        half_b = np.full((n_sel, n_conds), np.nan)

        for c_idx, cond_idx in enumerate(cond_list):
            trial_indices = repeated[cond_idx]
            n_reps = len(trial_indices)

            if split_idx >= n_reps:
                continue

            singleton_idx = trial_indices[split_idx]
            remaining_indices = [
                t for i, t in enumerate(trial_indices) if i != split_idx
            ]

            half_a[:, c_idx] = betas_sel[:, singleton_idx]
            half_b[:, c_idx] = betas_sel[:, remaining_indices].mean(axis=1)

        # Mask out conditions that were NaN for this split
        valid_conds = np.isfinite(half_a[0])
        if valid_conds.sum() < 3:
            continue

        r = _vectorized_pearson_r(
            half_a[:, valid_conds], half_b[:, valid_conds]
        )
        all_split_r.append(r)

    if not all_split_r:
        return np.full(spatial_shape, np.nan)

    # Average across splits
    stacked = np.array(all_split_r)  # (n_splits, n_sel_voxels)
    mean_r = np.nanmean(stacked, axis=0)

    reliability_flat = np.full(n_voxels, np.nan)
    reliability_flat[voxel_indices] = mean_r
    return reliability_flat.reshape(spatial_shape)


def save_nifti(data, ref_img, out_path):
    """Save a 3D array as a NIfTI file using ref_img's affine/header."""
    img = nib.Nifti1Image(data.astype(np.float32), ref_img.affine, ref_img.header)
    nib.save(img, str(out_path))


def roi_summary(metric_map, roi_masks):
    """Extract mean/median metric within each ROI."""
    summary = {}
    for roi_name, mask in roi_masks.items():
        values = metric_map[mask]
        values = values[np.isfinite(values)]
        if len(values) > 0:
            summary[roi_name] = {
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "n_voxels": int(len(values)),
            }
        else:
            summary[roi_name] = {"mean": np.nan, "median": np.nan, "n_voxels": 0}
    return summary


# ── figures ──────────────────────────────────────────────────────────────────

def _get_roi_names(all_results):
    """Extract ROI names from the first available result with data."""
    for sub in all_results.values():
        for pipe in sub.values():
            for beta in pipe.values():
                if beta.get("reliability"):
                    return list(beta["reliability"].keys())
                if beta.get("R2"):
                    return list(beta["R2"].keys())
    return []


def figure_reliability_comparison(all_results, output_dir):
    """Figure 1: voxel reliability for original vs NORDIC, b2 + b4, per ROI."""
    roi_names = _get_roi_names(all_results)

    fig, axes = plt.subplots(1, len(roi_names), figsize=(4 * len(roi_names), 5),
                             sharey=True)
    if len(roi_names) == 1:
        axes = [axes]

    x = np.arange(len(SUBJECTS))
    width = 0.18

    for roi_idx, roi_name in enumerate(roi_names):
        ax = axes[roi_idx]

        for p_idx, (pipeline, color) in enumerate([
            ("original", "#4472C4"),
            ("nordic", "#C00000"),
        ]):
            # b2 (light) and b4 (solid) for each pipeline
            for b_idx, (beta_key, alpha, label_suffix) in enumerate([
                ("TYPEB_FITHRF", 0.4, " b2"),
                ("TYPED_FITHRF_GLMDENOISE_RR", 1.0, " b4"),
            ]):
                vals = []
                for subject in SUBJECTS:
                    v = (all_results.get(subject, {})
                         .get(pipeline, {})
                         .get(beta_key, {})
                         .get("reliability", {})
                         .get(roi_name, {})
                         .get("median", np.nan))
                    vals.append(v)

                offset = (p_idx * 2 + b_idx - 1.5) * width
                label = f"{pipeline}{label_suffix}" if roi_idx == 0 else None
                ax.bar(x + offset, vals, width, alpha=alpha,
                       color=color, label=label, edgecolor="black",
                       linewidth=0.5)

        ax.set_title(roi_name, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace("sub-", "s") for s in SUBJECTS])
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("Voxel reliability (median Pearson r)")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Voxel reliability: Original vs NORDIC", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "figure_reliability_comparison.png",
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved figure_reliability_comparison.png")


def figure_reliability_progression(all_results, output_dir):
    """Figure 2: voxel reliability across b2→b4, per pipeline, avg across subjects."""
    roi_names = _get_roi_names(all_results)

    fig, axes = plt.subplots(1, len(roi_names), figsize=(4 * len(roi_names), 5),
                             sharey=True)
    if len(roi_names) == 1:
        axes = [axes]

    # Skip TYPEA_ONOFF (no per-trial betas)
    metric_beta_types = [bt for bt in BETA_TYPES if bt[0] != "TYPEA_ONOFF"]
    beta_keys = [bt[0] for bt in metric_beta_types]
    beta_labels = [bt[1] for bt in metric_beta_types]
    x = np.arange(len(beta_keys))

    colors = {"original": "#4472C4", "nordic": "#C00000"}

    for roi_idx, roi_name in enumerate(roi_names):
        ax = axes[roi_idx]

        for pipeline in PIPELINES:
            means = []
            sems = []
            for beta_key in beta_keys:
                vals = []
                for subject in SUBJECTS:
                    v = (all_results.get(subject, {})
                         .get(pipeline, {})
                         .get(beta_key, {})
                         .get("reliability", {})
                         .get(roi_name, {})
                         .get("median", np.nan))
                    if np.isfinite(v):
                        vals.append(v)
                means.append(np.mean(vals) if vals else np.nan)
                sems.append(np.std(vals) / np.sqrt(len(vals)) if len(vals) > 1 else 0)

            label = pipeline if roi_idx == 0 else None
            ax.errorbar(x, means, yerr=sems, marker="o", capsize=3,
                        color=colors[pipeline], label=label, linewidth=2)

        ax.set_title(roi_name, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(beta_labels, fontsize=8, rotation=30, ha="right")
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("Voxel reliability (median r, mean across subjects)")
    axes[0].legend(fontsize=9)
    fig.suptitle("GLMsingle beta version progression: voxel reliability",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "figure_reliability_progression.png",
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved figure_reliability_progression.png")


def figure_r2_comparison(all_results, output_dir):
    """Figure 3: R² for each beta type × pipeline, per ROI, avg across subjects."""
    roi_names = _get_roi_names(all_results)

    fig, axes = plt.subplots(1, len(roi_names), figsize=(4 * len(roi_names), 5),
                             sharey=True)
    if len(roi_names) == 1:
        axes = [axes]

    beta_keys = [bt[0] for bt in BETA_TYPES]
    beta_labels = [bt[1] for bt in BETA_TYPES]
    x = np.arange(len(beta_keys))

    colors = {"original": "#4472C4", "nordic": "#C00000"}

    for roi_idx, roi_name in enumerate(roi_names):
        ax = axes[roi_idx]

        for pipeline in PIPELINES:
            means = []
            sems = []
            for beta_key in beta_keys:
                vals = []
                for subject in SUBJECTS:
                    v = (all_results.get(subject, {})
                         .get(pipeline, {})
                         .get(beta_key, {})
                         .get("R2", {})
                         .get(roi_name, {})
                         .get("mean", np.nan))
                    if np.isfinite(v):
                        vals.append(v)
                means.append(np.mean(vals) if vals else np.nan)
                sems.append(np.std(vals) / np.sqrt(len(vals)) if len(vals) > 1 else 0)

            label = pipeline if roi_idx == 0 else None
            ax.errorbar(x, means, yerr=sems, marker="o", capsize=3,
                        color=colors[pipeline], label=label, linewidth=2)

        ax.set_title(roi_name, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(beta_labels, fontsize=8, rotation=30, ha="right")
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("R² (mean across subjects)")
    axes[0].legend(fontsize=9)
    fig.suptitle("GLMsingle R²: Original vs NORDIC", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "figure_r2_comparison.png",
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved figure_r2_comparison.png")


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir", type=Path, default=GLMSINGLE_BASE,
        help="Base output directory (default: derivatives/nordic/validation/glmsingle/)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir

    # Structure: all_results[subject][pipeline][beta_type] = {
    #   "reliability": roi_summary, "R2": roi_summary
    # }
    all_results = {}
    summary_rows = []

    for subject in SUBJECTS:
        print(f"\n{'=' * 60}")
        print(f"Processing {subject}")
        all_results[subject] = {}

        # Load a reference BOLD image for ROI mask creation
        ref_bold_path = (
            FMRIPREP_ORIG / subject / "ses-04" / "func"
            / f"{subject}_ses-04_task-TBencoding_run-01_space-{SPACE}_desc-preproc_bold.nii.gz"
        )
        ref_img = nib.load(str(ref_bold_path))
        ref_3d = nib.Nifti1Image(
            ref_img.dataobj[..., 0], ref_img.affine, ref_img.header
        )

        print("Loading ROI masks...")
        roi_masks = load_roi_masks(ref_3d, subject)

        # Build a brain mask from fMRIPrep
        brain_mask_path = (
            FMRIPREP_ORIG / subject / "ses-04" / "func"
            / f"{subject}_ses-04_task-TBencoding_run-01_space-{SPACE}_desc-brain_mask.nii.gz"
        )
        brain_mask = nib.load(str(brain_mask_path)).get_fdata().astype(bool)

        for pipeline in PIPELINES:
            print(f"\n  Pipeline: {pipeline}")
            all_results[subject][pipeline] = {}

            for beta_type, beta_label in BETA_TYPES:
                print(f"\n    {beta_label} ({beta_type})")

                data, stimorder = load_glmsingle_output(
                    subject, pipeline, beta_type
                )
                if data is None:
                    print(f"    Skipping — output not found")
                    continue

                betas = data["betasmd"]
                r2_map = data.get("R2")

                print(f"    Betas shape: {betas.shape}, "
                      f"{len(stimorder)} trials, "
                      f"{len(set(stimorder))} conditions")

                # ── R² (available for all beta types) ──
                r2_roi = {}
                if r2_map is not None:
                    r2_roi = roi_summary(r2_map, roi_masks)

                # ── Voxel reliability (only for per-trial betas) ──
                reliability_roi = {}
                if betas.shape[3] >= len(stimorder):
                    print(f"    Computing voxel reliability...")
                    reliability_map = compute_voxel_reliability(
                        betas, stimorder, mask=brain_mask
                    )
                    reliability_roi = roi_summary(reliability_map, roi_masks)
                else:
                    print(f"    Skipping reliability — not per-trial betas "
                          f"({betas.shape[3]} vols vs {len(stimorder)} trials)")

                all_results[subject][pipeline][beta_type] = {
                    "reliability": reliability_roi,
                    "R2": r2_roi,
                }

                # ── Save NIfTI maps ──
                nii_dir = (GLMSINGLE_BASE / subject / pipeline
                           / "nifti")
                nii_dir.mkdir(parents=True, exist_ok=True)
                if r2_map is not None:
                    save_nifti(
                        r2_map, ref_3d,
                        nii_dir / f"{beta_type}_R2.nii.gz",
                    )
                if reliability_roi:
                    save_nifti(
                        reliability_map, ref_3d,
                        nii_dir / f"{beta_type}_reliability.nii.gz",
                    )

                # Collect summary rows
                for roi_name in roi_masks:
                    row = {
                        "subject": subject,
                        "pipeline": pipeline,
                        "beta_type": beta_type,
                        "beta_label": beta_label,
                        "roi": roi_name,
                        "n_voxels": r2_roi.get(roi_name, {}).get("n_voxels", 0),
                        "R2_mean": r2_roi.get(roi_name, {}).get("mean", np.nan),
                        "R2_median": r2_roi.get(roi_name, {}).get("median", np.nan),
                    }
                    if reliability_roi:
                        row["reliability_mean"] = reliability_roi.get(
                            roi_name, {}
                        ).get("mean", np.nan)
                        row["reliability_median"] = reliability_roi.get(
                            roi_name, {}
                        ).get("median", np.nan)
                    else:
                        row["reliability_mean"] = np.nan
                        row["reliability_median"] = np.nan
                    summary_rows.append(row)

                # Print ROI summary
                for roi_name in roi_masks:
                    r2_val = r2_roi.get(roi_name, {}).get("mean", np.nan)
                    rel_val = reliability_roi.get(roi_name, {}).get("median", np.nan)
                    parts = [f"R²={r2_val:.1f}%"]
                    if np.isfinite(rel_val):
                        parts.append(f"reliability={rel_val:.3f}")
                    print(f"      {roi_name}: {', '.join(parts)}")

    # ── Save summary TSV ──
    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_dir / "glmsingle_metrics_summary.tsv"
    summary_df.to_csv(summary_path, sep="\t", index=False, float_format="%.4f")
    print(f"\nSaved {summary_path}")

    # ── Generate figures ──
    print("\nGenerating figures...")
    figure_reliability_comparison(all_results, output_dir)
    figure_reliability_progression(all_results, output_dir)
    figure_r2_comparison(all_results, output_dir)

    # ── Print go/no-go summary ──
    beta_key = "TYPED_FITHRF_GLMDENOISE_RR"

    print(f"\n{'=' * 60}")
    print("GO / NO-GO: Voxel reliability (b4, median r)")
    print(f"{'=' * 60}")
    b4_rel = summary_df[summary_df["beta_type"] == beta_key].copy()
    if not b4_rel.empty and "reliability_median" in b4_rel.columns:
        pivot = b4_rel.pivot_table(
            index=["subject", "roi"],
            columns="pipeline",
            values="reliability_median",
        )
        if "original" in pivot.columns and "nordic" in pivot.columns:
            pivot["change"] = pivot["nordic"] - pivot["original"]
            print(pivot.to_string(float_format="%.3f"))

    print(f"\n{'=' * 60}")
    print("R² by beta type (mean across ROIs)")
    print(f"{'=' * 60}")
    if not summary_df.empty:
        r2_pivot = summary_df.pivot_table(
            index=["subject", "beta_label"],
            columns="pipeline",
            values="R2_mean",
        )
        if "original" in r2_pivot.columns and "nordic" in r2_pivot.columns:
            r2_pivot["change"] = r2_pivot["nordic"] - r2_pivot["original"]
            print(r2_pivot.to_string(float_format="%.2f"))

    print("\nDone.")


if __name__ == "__main__":
    main()
