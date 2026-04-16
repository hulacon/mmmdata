#!/usr/bin/env python3
"""
glmsingle_qc.py — Quality control report for GLMsingle output.

Generates per-subject QC diagnostics for a single GLMsingle run, including:
  - R² progression across beta types (B → C → D)
  - Voxel reliability for repeated conditions
  - HRF index distribution and spatial map
  - Noise pool and PC selection summary
  - FRACvalue (ridge regression regularization) distribution
  - Run-wise FIR timecourse sanity check
  - Per-ROI metric summaries
  - Anomaly detection (negative R², NaN betas, etc.)

Reads GLMsingle outputs from a directory structured as:
  {input_dir}/glmsingle_outputs/  (*.npy files)
  {input_dir}/glmsingle_figures/  (diagnostic PNGs from GLMsingle)
  {input_dir}/condition_key.csv
  {input_dir}/trial_info.csv
  {input_dir}/run_metadata.json

Usage:
    python glmsingle_qc.py --input-dir derivatives/glmsingle/sub-03/ses-04
    python glmsingle_qc.py --input-dir derivatives/glmsingle/sub-03/ses-04 \
                           --output-dir derivatives/qc/glmsingle/sub-03/ses-04
"""

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import nibabel as nib
from nilearn.image import resample_to_img
import numpy as np
import pandas as pd

# ── paths ────────────────────────────────────────────────────────────────────

BIDS_ROOT = Path("/gpfs/projects/hulacon/shared/mmmdata")
FMRIPREP = BIDS_ROOT / "derivatives" / "fmriprep"

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

SPACE = "MNI152NLin2009cAsym_res-2"

BETA_TYPES = [
    ("TYPEA_ONOFF", "A: AssumeHRF"),
    ("TYPEB_FITHRF", "B: FitHRF"),
    ("TYPEC_FITHRF_GLMDENOISE", "C: +GLMdenoise"),
    ("TYPED_FITHRF_GLMDENOISE_RR", "D: +RidgeReg"),
]


# ── ROI loading ──────────────────────────────────────────────────────────────

def load_roi_masks(bold_ref_img, subject):
    """Build ROI boolean masks in BOLD space."""
    rois = {}

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

    aseg_mgz = FMRIPREP / "sourcedata" / "freesurfer" / subject / "mri" / "aparc+aseg.mgz"
    if aseg_mgz.exists():
        aseg_img = nib.load(str(aseg_mgz))
        aseg_resampled = resample_to_img(
            aseg_img, bold_ref_img, interpolation="nearest"
        )
        aseg_data = np.asarray(aseg_resampled.dataobj).astype(int)
        rois["Hippocampus"] = np.isin(aseg_data, ASEG_HIPPO_LABELS)

    return rois


# ── GLMsingle loading ────────────────────────────────────────────────────────

def load_npy(path):
    """Load a .npy file as a dict."""
    return np.load(str(path), allow_pickle=True).item()


# ── reliability (same as nordic_glmsingle_report.py) ─────────────────────────

def _vectorized_pearson_r(A, B):
    A_dm = A - A.mean(axis=1, keepdims=True)
    B_dm = B - B.mean(axis=1, keepdims=True)
    num = (A_dm * B_dm).sum(axis=1)
    denom = np.sqrt((A_dm**2).sum(axis=1) * (B_dm**2).sum(axis=1))
    r = np.full(A.shape[0], np.nan)
    valid = denom > 1e-10
    r[valid] = num[valid] / denom[valid]
    return r


def compute_voxel_reliability(betas, stimorder, mask=None):
    """Split-half reliability of condition-beta profiles (Prince et al. 2022)."""
    spatial_shape = betas.shape[:3]
    n_trials = betas.shape[3]

    cond_trials = defaultdict(list)
    for trial_idx, cond_idx in enumerate(stimorder):
        cond_trials[cond_idx].append(trial_idx)

    repeated = {k: v for k, v in cond_trials.items() if len(v) >= 2}
    if len(repeated) == 0:
        return np.full(spatial_shape, np.nan), 0

    max_splits = max(len(v) for v in repeated.values())

    betas_flat = betas.reshape(-1, n_trials)
    n_voxels = betas_flat.shape[0]

    if mask is not None:
        voxel_indices = np.where(mask.flatten())[0]
    else:
        voxel_indices = np.arange(n_voxels)

    betas_sel = betas_flat[voxel_indices]
    n_conds = len(repeated)
    cond_list = sorted(repeated.keys())

    all_split_r = []
    for split_idx in range(max_splits):
        half_a = np.full((len(voxel_indices), n_conds), np.nan)
        half_b = np.full((len(voxel_indices), n_conds), np.nan)

        for c_idx, cond_idx in enumerate(cond_list):
            trial_indices = repeated[cond_idx]
            if split_idx >= len(trial_indices):
                continue
            singleton_idx = trial_indices[split_idx]
            remaining = [t for i, t in enumerate(trial_indices) if i != split_idx]
            half_a[:, c_idx] = betas_sel[:, singleton_idx]
            half_b[:, c_idx] = betas_sel[:, remaining].mean(axis=1)

        valid_conds = np.isfinite(half_a[0])
        if valid_conds.sum() < 3:
            continue
        r = _vectorized_pearson_r(half_a[:, valid_conds], half_b[:, valid_conds])
        all_split_r.append(r)

    if not all_split_r:
        return np.full(spatial_shape, np.nan), len(repeated)

    mean_r = np.nanmean(np.array(all_split_r), axis=0)
    reliability_flat = np.full(n_voxels, np.nan)
    reliability_flat[voxel_indices] = mean_r
    return reliability_flat.reshape(spatial_shape), len(repeated)


def get_r2(data_dict):
    """Extract R² map from a GLMsingle output dict (handles TypeA's 'onoffR2')."""
    if "R2" in data_dict:
        return data_dict["R2"]
    if "onoffR2" in data_dict:
        return data_dict["onoffR2"]
    return None


# ── QC report generation ─────────────────────────────────────────────────────

def roi_stats(data_3d, roi_masks):
    """Extract stats for each ROI from a 3D map."""
    results = {}
    for name, mask in roi_masks.items():
        vals = data_3d[mask]
        vals = vals[np.isfinite(vals)]
        if len(vals) > 0:
            results[name] = {
                "n_voxels": len(vals),
                "mean": float(np.mean(vals)),
                "median": float(np.median(vals)),
                "std": float(np.std(vals)),
                "pct5": float(np.percentile(vals, 5)),
                "pct95": float(np.percentile(vals, 95)),
            }
        else:
            results[name] = {"n_voxels": 0}
    return results


def generate_qc_report(input_dir, output_dir, subject, session):
    """Generate full QC report for one GLMsingle run."""

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir = input_dir / "glmsingle_outputs"

    # Load metadata
    metadata_path = input_dir / "run_metadata.json"
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)

    # Load DESIGNINFO
    design_info = load_npy(outputs_dir / "DESIGNINFO.npy")
    stimorder = design_info["stimorder"]
    tr = design_info["tr"]

    # Load condition key
    cond_key = None
    cond_key_path = input_dir / "condition_key.csv"
    if cond_key_path.exists():
        cond_key = pd.read_csv(cond_key_path)

    # Get a brain mask from fMRIPrep — try the given session first, then
    # fall back to scanning for any available TBencoding mask (needed when
    # session is a label like "all-sessions" rather than a real ses-XX).
    run_01 = "run-01"
    task = metadata.get("task", "TBencoding")
    brain_mask = None
    ref_3d = None

    # Determine fMRIPrep directory from metadata or use default
    fmriprep_dir = Path(metadata["fmriprep_dir"]) if metadata.get("fmriprep_dir") else FMRIPREP

    candidate_sessions = [session]
    if metadata.get("sessions"):
        candidate_sessions = metadata["sessions"]
    for try_ses in candidate_sessions:
        mask_path = (
            fmriprep_dir / subject / try_ses / "func"
            / f"{subject}_{try_ses}_task-{task}_{run_01}_space-{SPACE}_desc-brain_mask.nii.gz"
        )
        if mask_path.exists():
            mask_img = nib.load(str(mask_path))
            brain_mask = mask_img.get_fdata().astype(bool)
            ref_3d = mask_img
            print(f"Using brain mask from {try_ses}")
            break

    if brain_mask is None:
        print("WARNING: No brain mask found — ROI and masked metrics will be unavailable")

    # Load ROI masks
    roi_masks = {}
    if ref_3d is not None:
        print("Loading ROI masks...")
        roi_masks = load_roi_masks(ref_3d, subject)
        for name, m in roi_masks.items():
            print(f"  {name}: {m.sum()} voxels")

    # ── Load all beta types ──
    all_data = {}
    for beta_type, label in BETA_TYPES:
        npy_path = outputs_dir / f"{beta_type}.npy"
        if npy_path.exists():
            all_data[beta_type] = load_npy(npy_path)
            print(f"Loaded {label}")
        else:
            print(f"Skipped {label} (not found)")

    if not all_data:
        print("ERROR: No GLMsingle outputs found")
        return

    # Use TypeD (most complete) as primary reference
    primary_key = "TYPED_FITHRF_GLMDENOISE_RR"
    if primary_key not in all_data:
        primary_key = list(all_data.keys())[-1]
    primary = all_data[primary_key]

    # ── Begin PDF report ──
    pdf_path = output_dir / f"glmsingle_qc_{subject}_{session}.pdf"
    summary_rows = []

    with PdfPages(str(pdf_path)) as pdf:

        # ── Page 1: Overview ──
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis("off")
        overview_text = [
            f"GLMsingle QC Report",
            f"Subject: {subject}    Session: {session}",
            f"Task: {metadata.get('task', 'unknown')}",
            f"TR: {tr}s    Stim duration: {design_info.get('stimdur', '?')}s",
            f"Runs: {len(design_info.get('numtrialrun', []))}",
            f"Conditions: {len(design_info.get('condcounts', []))}",
            f"Total trials: {len(stimorder)}",
            f"Confound strategy: {metadata.get('confound_strategy', 'unknown')}",
            "",
        ]
        # Count repeated conditions
        cond_trials = defaultdict(int)
        for c in stimorder:
            cond_trials[c] += 1
        n_repeated = sum(1 for v in cond_trials.values() if v >= 2)
        overview_text.append(f"Repeated conditions (≥2 presentations): {n_repeated}")
        overview_text.append(f"Unique conditions with 1 presentation: {sum(1 for v in cond_trials.values() if v == 1)}")

        if primary_key in all_data:
            d = all_data[primary_key]
            overview_text.append(f"")
            overview_text.append(f"GLMdenoise PCs used: {d.get('pcnum', '?')}")
            overview_text.append(f"Noise pool voxels: {d['noisepool'].sum() if 'noisepool' in d else '?'}")

        ax.text(0.05, 0.95, "\n".join(overview_text),
                transform=ax.transAxes, fontsize=11, verticalalignment="top",
                fontfamily="monospace")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # ── Page 2: R² progression (B → C → D) ──
        print("Generating R² progression plots...")
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Whole-brain R² histograms
        ax = axes[0, 0]
        for beta_type, label in BETA_TYPES:
            if beta_type not in all_data:
                continue
            r2 = get_r2(all_data[beta_type])
            if brain_mask is not None:
                vals = r2[brain_mask]
            else:
                vals = r2[r2 != 0]
            vals = vals[np.isfinite(vals)]
            ax.hist(vals, bins=100, alpha=0.5, label=f"{label} (med={np.median(vals):.1f}%)",
                    density=True, range=(-5, 50))
        ax.set_xlabel("R² (%)")
        ax.set_ylabel("Density")
        ax.set_title("Whole-brain R² distribution")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # R² improvement: C vs B
        ax = axes[0, 1]
        if "TYPEB_FITHRF" in all_data and "TYPEC_FITHRF_GLMDENOISE" in all_data:
            r2_b = get_r2(all_data["TYPEB_FITHRF"])
            r2_c = get_r2(all_data["TYPEC_FITHRF_GLMDENOISE"])
            diff = r2_c - r2_b
            if brain_mask is not None:
                diff_vals = diff[brain_mask]
            else:
                diff_vals = diff[diff != 0]
            diff_vals = diff_vals[np.isfinite(diff_vals)]
            ax.hist(diff_vals, bins=100, alpha=0.7, color="green",
                    range=(-10, 20))
            ax.axvline(0, color="black", linestyle="--", linewidth=0.5)
            ax.set_xlabel("ΔR² (C - B)")
            ax.set_title(f"GLMdenoise improvement (med={np.median(diff_vals):.2f}%)")
            pct_improved = (diff_vals > 0).mean() * 100
            ax.text(0.95, 0.95, f"{pct_improved:.0f}% voxels improved",
                    transform=ax.transAxes, ha="right", va="top", fontsize=9)
        ax.grid(alpha=0.3)

        # R² improvement: D vs C
        ax = axes[1, 0]
        if "TYPEC_FITHRF_GLMDENOISE" in all_data and primary_key in all_data:
            r2_c = get_r2(all_data["TYPEC_FITHRF_GLMDENOISE"])
            r2_d = get_r2(all_data[primary_key])
            diff = r2_d - r2_c
            if brain_mask is not None:
                diff_vals = diff[brain_mask]
            else:
                diff_vals = diff[diff != 0]
            diff_vals = diff_vals[np.isfinite(diff_vals)]
            ax.hist(diff_vals, bins=100, alpha=0.7, color="purple",
                    range=(-10, 20))
            ax.axvline(0, color="black", linestyle="--", linewidth=0.5)
            ax.set_xlabel("ΔR² (D - C)")
            ax.set_title(f"Ridge regression improvement (med={np.median(diff_vals):.2f}%)")
            pct_improved = (diff_vals > 0).mean() * 100
            ax.text(0.95, 0.95, f"{pct_improved:.0f}% voxels improved",
                    transform=ax.transAxes, ha="right", va="top", fontsize=9)
        ax.grid(alpha=0.3)

        # Per-ROI R² bar chart across beta types
        ax = axes[1, 1]
        if roi_masks:
            roi_names = list(roi_masks.keys())
            x = np.arange(len(roi_names))
            width = 0.2
            for i, (beta_type, label) in enumerate(BETA_TYPES):
                if beta_type not in all_data:
                    continue
                r2 = get_r2(all_data[beta_type])
                means = []
                for roi_name in roi_names:
                    vals = r2[roi_masks[roi_name]]
                    vals = vals[np.isfinite(vals)]
                    means.append(np.mean(vals) if len(vals) > 0 else 0)
                ax.bar(x + i * width - 1.5 * width, means, width,
                       label=label, alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(roi_names, fontsize=8, rotation=20, ha="right")
            ax.set_ylabel("Mean R² (%)")
            ax.set_title("R² by ROI and beta type")
            ax.legend(fontsize=7)
            ax.grid(axis="y", alpha=0.3)

        fig.suptitle(f"R² Progression — {subject} / {session}", fontsize=13)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # ── Page 3: Voxel reliability ──
        print("Computing voxel reliability...")
        reliability_maps = {}
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Compute reliability for B, C, D
        rel_beta_types = [bt for bt in BETA_TYPES if bt[0] != "TYPEA_ONOFF"]
        for beta_type, label in rel_beta_types:
            if beta_type not in all_data:
                continue
            betas = all_data[beta_type]["betasmd"]
            rel_map, n_rep = compute_voxel_reliability(
                betas, stimorder, mask=brain_mask
            )
            reliability_maps[beta_type] = rel_map
            print(f"  {label}: {n_rep} repeated conditions")

        # Reliability histograms
        ax = axes[0, 0]
        for beta_type, label in rel_beta_types:
            if beta_type not in reliability_maps:
                continue
            rel = reliability_maps[beta_type]
            if brain_mask is not None:
                vals = rel[brain_mask]
            else:
                vals = rel.flatten()
            vals = vals[np.isfinite(vals)]
            ax.hist(vals, bins=100, alpha=0.5,
                    label=f"{label} (med={np.median(vals):.3f})",
                    density=True, range=(-0.3, 0.6))
        ax.set_xlabel("Voxel reliability (Pearson r)")
        ax.set_ylabel("Density")
        ax.set_title("Whole-brain reliability distribution")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # Reliability improvement: D vs B
        ax = axes[0, 1]
        if "TYPEB_FITHRF" in reliability_maps and primary_key in reliability_maps:
            rel_b = reliability_maps["TYPEB_FITHRF"]
            rel_d = reliability_maps[primary_key]
            diff = rel_d - rel_b
            if brain_mask is not None:
                diff_vals = diff[brain_mask]
            else:
                diff_vals = diff.flatten()
            diff_vals = diff_vals[np.isfinite(diff_vals)]
            ax.hist(diff_vals, bins=100, alpha=0.7, color="teal", range=(-0.2, 0.3))
            ax.axvline(0, color="black", linestyle="--", linewidth=0.5)
            ax.set_xlabel("Δ reliability (D - B)")
            ax.set_title(f"Full pipeline improvement (med={np.median(diff_vals):.4f})")
            pct_improved = (diff_vals > 0).mean() * 100
            ax.text(0.95, 0.95, f"{pct_improved:.0f}% voxels improved",
                    transform=ax.transAxes, ha="right", va="top", fontsize=9)
        ax.grid(alpha=0.3)

        # Per-ROI reliability
        ax = axes[1, 0]
        if roi_masks:
            roi_names = list(roi_masks.keys())
            x = np.arange(len(roi_names))
            width = 0.25
            for i, (beta_type, label) in enumerate(rel_beta_types):
                if beta_type not in reliability_maps:
                    continue
                rel = reliability_maps[beta_type]
                medians = []
                for roi_name in roi_names:
                    vals = rel[roi_masks[roi_name]]
                    vals = vals[np.isfinite(vals)]
                    medians.append(np.median(vals) if len(vals) > 0 else 0)
                ax.bar(x + i * width - width, medians, width,
                       label=label, alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(roi_names, fontsize=8, rotation=20, ha="right")
            ax.set_ylabel("Median reliability (r)")
            ax.set_title("Reliability by ROI and beta type")
            ax.legend(fontsize=7)
            ax.grid(axis="y", alpha=0.3)

        # Reliability vs R² scatter (TypeD)
        ax = axes[1, 1]
        if primary_key in all_data and primary_key in reliability_maps:
            r2 = get_r2(all_data[primary_key])
            rel = reliability_maps[primary_key]
            if brain_mask is not None:
                r2_vals = r2[brain_mask]
                rel_vals = rel[brain_mask]
            else:
                r2_vals = r2.flatten()
                rel_vals = rel.flatten()
            valid = np.isfinite(r2_vals) & np.isfinite(rel_vals)
            # Subsample for plotting
            n_plot = min(10000, valid.sum())
            if valid.sum() > 0:
                idx = np.random.default_rng(42).choice(
                    np.where(valid)[0], size=n_plot, replace=False
                )
                ax.scatter(r2_vals[idx], rel_vals[idx], s=1, alpha=0.2, color="gray")
                ax.set_xlabel("R² (%)")
                ax.set_ylabel("Reliability (r)")
                ax.set_title("R² vs reliability (TypeD, 10k voxels)")
                ax.grid(alpha=0.3)

        fig.suptitle(f"Voxel Reliability — {subject} / {session}", fontsize=13)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # ── Page 4: HRF index & noise pool ──
        print("Generating HRF and denoising diagnostics...")
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # HRF index distribution
        ax = axes[0, 0]
        if "TYPEB_FITHRF" in all_data:
            hrf_idx = all_data["TYPEB_FITHRF"]["HRFindex"]
            if brain_mask is not None:
                vals = hrf_idx[brain_mask]
            else:
                vals = hrf_idx[hrf_idx >= 0]
            n_library = int(vals.max()) + 1
            ax.hist(vals, bins=np.arange(-0.5, n_library + 0.5, 1),
                    alpha=0.7, color="steelblue", edgecolor="black", linewidth=0.3)
            ax.set_xlabel("HRF library index")
            ax.set_ylabel("Voxel count")
            ax.set_title(f"HRF index distribution ({n_library} library HRFs)")
            ax.grid(axis="y", alpha=0.3)

        # xvaltrend (cross-validation of PC count)
        ax = axes[0, 1]
        if primary_key in all_data and "xvaltrend" in all_data[primary_key]:
            xvt = all_data[primary_key]["xvaltrend"]
            n_pcs = len(xvt)
            pcnum = all_data[primary_key].get("pcnum", None)
            ax.plot(range(n_pcs), xvt, "o-", color="darkred", markersize=4)
            if pcnum is not None:
                ax.axvline(pcnum, color="green", linestyle="--",
                           label=f"Selected: {pcnum} PCs")
                ax.legend()
            ax.set_xlabel("Number of PCs")
            ax.set_ylabel("Cross-validation metric")
            ax.set_title("GLMdenoise PC cross-validation")
            ax.grid(alpha=0.3)

        # Noise pool size
        ax = axes[1, 0]
        if primary_key in all_data and "noisepool" in all_data[primary_key]:
            np_mask = all_data[primary_key]["noisepool"]
            np_count = np_mask.sum()
            brain_count = brain_mask.sum() if brain_mask is not None else np_mask.size
            ax.bar(["Noise pool", "Brain"], [np_count, brain_count],
                   color=["salmon", "lightblue"], edgecolor="black")
            ax.set_ylabel("Voxel count")
            ax.set_title(f"Noise pool: {np_count} / {brain_count} "
                         f"({100 * np_count / max(brain_count, 1):.1f}%)")
            ax.grid(axis="y", alpha=0.3)

        # FRACvalue distribution (ridge regularization)
        ax = axes[1, 1]
        if primary_key in all_data and "FRACvalue" in all_data[primary_key]:
            frac = all_data[primary_key]["FRACvalue"]
            if brain_mask is not None:
                vals = frac[brain_mask]
            else:
                vals = frac[frac > 0]
            vals = vals[np.isfinite(vals) & (vals > 0)]
            if len(vals) > 0:
                ax.hist(vals, bins=50, alpha=0.7, color="darkorange",
                        edgecolor="black", linewidth=0.3)
                ax.set_xlabel("FRAC value (fraction of full OLS)")
                ax.set_ylabel("Voxel count")
                ax.set_title(f"Ridge regularization (med={np.median(vals):.2f})")
                ax.grid(alpha=0.3)

        fig.suptitle(f"HRF & Denoising — {subject} / {session}", fontsize=13)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # ── Page 5: FIR timecourses ──
        print("Generating FIR timecourse plots...")
        fir_path = outputs_dir / "RUNWISEFIR.npy"
        if fir_path.exists():
            fir_data = load_npy(fir_path)
            # fir_data keys: firR2 (n_runs, X, Y, Z), firtcs (n_runs, X, Y, Z, 1, n_tp),
            #                firavg (n_runs, n_tp), firgrandavg (n_tp,)
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # Left: grand average + per-run FIR (already computed by GLMsingle)
            ax = axes[0]
            if "firgrandavg" in fir_data:
                grandavg = fir_data["firgrandavg"]
                t = np.arange(len(grandavg)) * tr
                ax.plot(t, grandavg, "ko-", linewidth=2, label="Grand avg")
            if "firavg" in fir_data:
                firavg = fir_data["firavg"]
                for run_i in range(firavg.shape[0]):
                    t = np.arange(firavg.shape[1]) * tr
                    ax.plot(t, firavg[run_i], "o--", markersize=3,
                            alpha=0.6, label=f"run-{run_i+1:02d}")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Response (% signal change)")
            ax.set_title("FIR timecourse — task-responsive voxels")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
            ax.axhline(0, color="black", linewidth=0.5)

            # Right: FIR for top 1% R² voxels per ROI
            ax = axes[1]
            if ("firtcs" in fir_data and "TYPEB_FITHRF" in all_data
                    and roi_masks):
                firtcs = fir_data["firtcs"]  # (n_runs, X, Y, Z, 1, n_tp)
                # Average across runs, squeeze singleton dim
                fir_mean = firtcs.mean(axis=0).squeeze(axis=3)  # (X, Y, Z, n_tp)
                n_tp = fir_mean.shape[-1]
                t = np.arange(n_tp) * tr

                for roi_name, roi_mask in roi_masks.items():
                    roi_fir = fir_mean[roi_mask]  # (n_roi_voxels, n_tp)
                    if roi_fir.shape[0] > 0:
                        # Mean across ROI voxels
                        mean_tc = np.nanmean(roi_fir, axis=0)
                        ax.plot(t, mean_tc, "o-", markersize=3, label=roi_name)

                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Response (% signal change)")
                ax.set_title("FIR timecourse by ROI (run-averaged)")
                ax.legend(fontsize=8)
                ax.grid(alpha=0.3)
                ax.axhline(0, color="black", linewidth=0.5)

            fig.suptitle(f"FIR Diagnostics — {subject} / {session}", fontsize=13)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        # ── Page 6: Anomaly detection ──
        print("Running anomaly checks...")
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.axis("off")
        anomaly_lines = [f"Anomaly Report — {subject} / {session}", ""]

        for beta_type, label in BETA_TYPES:
            if beta_type not in all_data:
                continue
            d = all_data[beta_type]
            r2 = get_r2(d)
            betas = d.get("betasmd")

            anomaly_lines.append(f"── {label} ──")

            # Negative R²
            if brain_mask is not None:
                r2_brain = r2[brain_mask]
            else:
                r2_brain = r2.flatten()
            r2_brain = r2_brain[np.isfinite(r2_brain)]
            n_neg = (r2_brain < 0).sum()
            pct_neg = 100 * n_neg / max(len(r2_brain), 1)
            flag = " ⚠" if pct_neg > 50 else ""
            anomaly_lines.append(f"  Negative R²: {n_neg} voxels ({pct_neg:.1f}%){flag}")

            # NaN betas
            if betas is not None:
                n_nan_voxels = np.any(np.isnan(betas), axis=3).sum()
                flag = " ⚠" if n_nan_voxels > 100 else ""
                anomaly_lines.append(f"  Voxels with NaN betas: {n_nan_voxels}{flag}")

                # Zero-variance betas (voxels where all betas are identical)
                if brain_mask is not None:
                    betas_brain = betas[brain_mask]
                else:
                    betas_brain = betas.reshape(-1, betas.shape[3])
                beta_var = np.nanvar(betas_brain, axis=1)
                n_zero_var = (beta_var < 1e-10).sum()
                flag = " ⚠" if n_zero_var > 100 else ""
                anomaly_lines.append(f"  Zero-variance beta voxels: {n_zero_var}{flag}")

            # R² summary
            anomaly_lines.append(
                f"  R² brain: mean={np.mean(r2_brain):.2f}%, "
                f"median={np.median(r2_brain):.2f}%, "
                f"max={np.max(r2_brain):.1f}%"
            )
            anomaly_lines.append("")

        # Global checks
        anomaly_lines.append("── Global checks ──")
        if primary_key in all_data:
            d = all_data[primary_key]
            pcnum = d.get("pcnum", None)
            if pcnum == 0:
                anomaly_lines.append("  ⚠ GLMdenoise selected 0 PCs — denoising had no effect")
            elif pcnum is not None:
                anomaly_lines.append(f"  GLMdenoise PCs: {pcnum} (OK)")

            if "noisepool" in d:
                np_pct = 100 * d["noisepool"].sum() / max(brain_mask.sum() if brain_mask is not None else 1, 1)
                if np_pct < 5:
                    anomaly_lines.append(f"  ⚠ Noise pool very small: {np_pct:.1f}% of brain")
                elif np_pct > 80:
                    anomaly_lines.append(f"  ⚠ Noise pool very large: {np_pct:.1f}% of brain")
                else:
                    anomaly_lines.append(f"  Noise pool: {np_pct:.1f}% of brain (OK)")

        ax.text(0.05, 0.95, "\n".join(anomaly_lines),
                transform=ax.transAxes, fontsize=9, verticalalignment="top",
                fontfamily="monospace")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # ── Page 7: ROI summary table ──
        print("Building ROI summary table...")
        if roi_masks:
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.axis("off")

            table_data = []
            col_labels = ["ROI", "N voxels", "Beta Type",
                          "R² mean", "R² med", "Rel. mean", "Rel. med"]

            for roi_name in roi_masks:
                for beta_type, label in BETA_TYPES:
                    if beta_type not in all_data:
                        continue
                    r2 = get_r2(all_data[beta_type])
                    r2_vals = r2[roi_masks[roi_name]]
                    r2_vals = r2_vals[np.isfinite(r2_vals)]

                    rel_mean = rel_med = ""
                    if beta_type in reliability_maps:
                        rel = reliability_maps[beta_type]
                        rel_vals = rel[roi_masks[roi_name]]
                        rel_vals = rel_vals[np.isfinite(rel_vals)]
                        if len(rel_vals) > 0:
                            rel_mean = f"{np.mean(rel_vals):.4f}"
                            rel_med = f"{np.median(rel_vals):.4f}"

                    row = [
                        roi_name, str(len(r2_vals)), label,
                        f"{np.mean(r2_vals):.2f}" if len(r2_vals) > 0 else "—",
                        f"{np.median(r2_vals):.2f}" if len(r2_vals) > 0 else "—",
                        rel_mean or "—", rel_med or "—",
                    ]
                    table_data.append(row)
                    summary_rows.append({
                        "subject": subject, "session": session,
                        "roi": roi_name, "beta_type": beta_type,
                        "beta_label": label,
                        "n_voxels": len(r2_vals),
                        "R2_mean": np.mean(r2_vals) if len(r2_vals) > 0 else np.nan,
                        "R2_median": np.median(r2_vals) if len(r2_vals) > 0 else np.nan,
                        "reliability_mean": float(rel_mean) if rel_mean else np.nan,
                        "reliability_median": float(rel_med) if rel_med else np.nan,
                    })

            if table_data:
                table = ax.table(cellText=table_data, colLabels=col_labels,
                                 loc="center", cellLoc="center")
                table.auto_set_font_size(False)
                table.set_fontsize(8)
                table.scale(1, 1.3)
                ax.set_title(f"ROI Summary — {subject} / {session}", fontsize=12, pad=20)

            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    print(f"\nSaved PDF report: {pdf_path}")

    # ── Save summary TSV ──
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        tsv_path = output_dir / f"glmsingle_qc_{subject}_{session}.tsv"
        summary_df.to_csv(tsv_path, sep="\t", index=False, float_format="%.4f")
        print(f"Saved summary TSV: {tsv_path}")

    # ── Save reliability NIfTIs ──
    if ref_3d is not None:
        nifti_dir = output_dir / "nifti"
        nifti_dir.mkdir(exist_ok=True)
        for beta_type, label in rel_beta_types:
            if beta_type in reliability_maps:
                out_path = nifti_dir / f"{beta_type}_reliability.nii.gz"
                img = nib.Nifti1Image(
                    reliability_maps[beta_type].astype(np.float32),
                    ref_3d.affine, ref_3d.header
                )
                nib.save(img, str(out_path))
                print(f"Saved {out_path.name}")

        # Also save TypeD R² as NIfTI
        if primary_key in all_data:
            r2_out = nifti_dir / f"{primary_key}_R2.nii.gz"
            img = nib.Nifti1Image(
                get_r2(all_data[primary_key]).astype(np.float32),
                ref_3d.affine, ref_3d.header
            )
            nib.save(img, str(r2_out))
            print(f"Saved {r2_out.name}")

    # ── Copy GLMsingle's own diagnostic figures ──
    src_figs = input_dir / "glmsingle_figures"
    if src_figs.exists():
        dst_figs = output_dir / "glmsingle_figures"
        if dst_figs.exists():
            shutil.rmtree(dst_figs)
        shutil.copytree(src_figs, dst_figs)
        print(f"Copied {len(list(dst_figs.glob('*.png')))} GLMsingle diagnostic figures")

    print("\nQC complete.")


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input-dir", type=Path, required=True,
        help="GLMsingle output directory (contains glmsingle_outputs/, etc.)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="QC output directory (default: derivatives/qc/glmsingle/{subject}/{session})",
    )
    parser.add_argument(
        "--subject", required=True,
        help="Subject ID (e.g. sub-03)",
    )
    parser.add_argument(
        "--session", required=True,
        help="Session ID (e.g. ses-04)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.output_dir is None:
        output_dir = (
            BIDS_ROOT / "derivatives" / "qc" / "glmsingle"
            / args.subject / args.session
        )
    else:
        output_dir = args.output_dir

    generate_qc_report(
        input_dir=args.input_dir,
        output_dir=output_dir,
        subject=args.subject,
        session=args.session,
    )


if __name__ == "__main__":
    main()
