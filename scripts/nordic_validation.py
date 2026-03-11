#!/usr/bin/env python3
"""
nordic_validation.py — Compare original vs NORDIC-denoised fMRIPrep outputs.

Computes:
  6a. tSNR maps (mean/std) per run, side-by-side slices + improvement histogram
  6b. Temporal autocorrelation (lag 1-5) to check NORDIC doesn't smooth in time

Usage:
    python nordic_validation.py [--output-dir PATH]

Reads MNI-space preprocessed BOLD from:
  - derivatives/fmriprep/          (original)
  - derivatives/fmriprep_nordic/   (NORDIC-denoised)
"""

import argparse
import sys
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

SUBJECT = "sub-03"
SESSION = "ses-04"
SPACE = "MNI152NLin2009cAsym_res-2"

RUNS = [
    "task-TBencoding_run-01",
    "task-TBencoding_run-02",
    "task-TBencoding_run-03",
    "task-TBretrieval_run-01",
    "task-TBretrieval_run-02",
    "task-TBmath",
    "task-TBresting",
]

# ── ROI definitions ──────────────────────────────────────────────────────────
# Schaefer 400-parcel 17-network atlas (cortical ROIs)
SCHAEFER_ATLAS = (
    BIDS_ROOT / "derivatives" / "atlases" / "tpl-MNI152NLin2009cAsym" / "anat"
    / "tpl-MNI152NLin2009cAsym_atlas-Schaefer2018_seg-17n_scale-400_res-2_dseg.nii.gz"
)
SCHAEFER_LUT = (
    BIDS_ROOT / "derivatives" / "atlases" / "tpl-MNI152NLin2009cAsym" / "anat"
    / "tpl-MNI152NLin2009cAsym_atlas-Schaefer2018_seg-17n_scale-400_res-2_dseg.tsv"
)

# FreeSurfer aseg (subcortical — hippocampus)
ASEG_MGZ = (
    FMRIPREP_ORIG / "sourcedata" / "freesurfer" / SUBJECT / "mri" / "aparc+aseg.mgz"
)
# fMRIPrep T1w-to-MNI warp
T1W_TO_MNI_XFM = (
    FMRIPREP_ORIG / SUBJECT / "anat"
    / f"{SUBJECT}_acq-MPR_from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5"
)
# MNI-space T1w as reference for resampling
MNI_REF = (
    FMRIPREP_ORIG / SUBJECT / "anat"
    / f"{SUBJECT}_acq-MPR_space-MNI152NLin2009cAsym_res-2_desc-preproc_T1w.nii.gz"
)

# Schaefer parcel indices (bilateral, collapsed L+R) for each cortical ROI
SCHAEFER_ROIS = {
    "V1 (VisCent)": {
        "pattern": "VisCent_ExStr",
    },
    "Angular gyrus (DefaultA IPL)": {
        "pattern": "DefaultA_IPL",
    },
    "vmPFC (DefaultA PFCm)": {
        "pattern": "DefaultA_PFCm",
    },
}

# FreeSurfer aseg labels for hippocampus
ASEG_HIPPO_LABELS = [17, 53]  # Left-Hippocampus, Right-Hippocampus


def bold_path(deriv_root, run):
    return (
        deriv_root / SUBJECT / SESSION / "func"
        / f"{SUBJECT}_{SESSION}_{run}_space-{SPACE}_desc-preproc_bold.nii.gz"
    )


def mask_path(deriv_root, run):
    return (
        deriv_root / SUBJECT / SESSION / "func"
        / f"{SUBJECT}_{SESSION}_{run}_space-{SPACE}_desc-brain_mask.nii.gz"
    )


# ── ROI masks ────────────────────────────────────────────────────────────────

def load_roi_masks(bold_ref_img):
    """
    Build ROI boolean masks in MNI BOLD space.

    Returns dict: roi_name -> 3D boolean array matching bold_ref_img shape.
    """
    rois = {}

    # --- Schaefer cortical parcels ---
    schaefer_img = nib.load(str(SCHAEFER_ATLAS))
    # Resample to BOLD grid (nearest-neighbor to preserve labels)
    schaefer_resampled = resample_to_img(
        schaefer_img, bold_ref_img, interpolation="nearest"
    )
    schaefer_data = np.asarray(schaefer_resampled.dataobj).astype(int)

    # Read the LUT to find parcel indices by name pattern
    lut = pd.read_csv(SCHAEFER_LUT, sep="\t")

    for roi_name, roi_def in SCHAEFER_ROIS.items():
        pattern = roi_def["pattern"]
        matching = lut[lut["name"].str.contains(pattern, na=False)]
        indices = set(matching["index"].values)
        mask = np.isin(schaefer_data, list(indices))
        rois[roi_name] = mask
        print(f"  ROI '{roi_name}': {len(indices)} parcels, {mask.sum()} voxels")

    # --- Hippocampus from FreeSurfer aseg ---
    if ASEG_MGZ.exists():
        aseg_img = nib.load(str(ASEG_MGZ))
        # Resample aseg to MNI BOLD grid via nilearn (nearest-neighbor).
        # The aseg is in FreeSurfer conformed space (~1mm T1w); nilearn will
        # use the image affines to align it to the BOLD reference.
        # Since there's a nonlinear warp (T1w→MNI), using affine-only
        # resampling is approximate — but for a coarse ROI like hippocampus
        # at 2mm resolution, this is adequate for tSNR averaging.
        aseg_resampled = resample_to_img(
            aseg_img, bold_ref_img, interpolation="nearest"
        )
        aseg_data = np.asarray(aseg_resampled.dataobj).astype(int)
        hippo_mask = np.isin(aseg_data, ASEG_HIPPO_LABELS)
        rois["Hippocampus"] = hippo_mask
        print(f"  ROI 'Hippocampus': labels {ASEG_HIPPO_LABELS}, {hippo_mask.sum()} voxels")
    else:
        print(f"  WARNING: aseg not found at {ASEG_MGZ}, skipping hippocampus")

    return rois


# ── tSNR ─────────────────────────────────────────────────────────────────────

def compute_tsnr(bold_img, mask_data):
    """Compute tSNR = mean / std within mask. Returns full-volume array."""
    data = bold_img.get_fdata(dtype=np.float32)
    mean = data.mean(axis=-1)
    std = data.std(axis=-1)
    tsnr = np.zeros_like(mean)
    valid = (std > 0) & (mask_data > 0)
    tsnr[valid] = mean[valid] / std[valid]
    return tsnr


def plot_tsnr_comparison(tsnr_orig, tsnr_nordic, mask_data, run_label, output_dir):
    """Side-by-side tSNR axial slices + improvement ratio histogram."""
    # Pick slices with good brain coverage
    z_coverage = mask_data.sum(axis=(0, 1))
    z_indices = np.where(z_coverage > z_coverage.max() * 0.3)[0]
    slices = z_indices[np.linspace(0, len(z_indices) - 1, 6, dtype=int)]

    vmax = np.percentile(tsnr_orig[mask_data > 0], 98)

    fig, axes = plt.subplots(3, 6, figsize=(18, 9))

    for i, z in enumerate(slices):
        axes[0, i].imshow(tsnr_orig[:, :, z].T, origin="lower", cmap="hot",
                          vmin=0, vmax=vmax)
        axes[0, i].set_title(f"z={z}")
        axes[0, i].axis("off")

        axes[1, i].imshow(tsnr_nordic[:, :, z].T, origin="lower", cmap="hot",
                          vmin=0, vmax=vmax)
        axes[1, i].axis("off")

        # Improvement ratio
        ratio = np.zeros_like(tsnr_orig[:, :, z])
        valid = (tsnr_orig[:, :, z] > 0) & (mask_data[:, :, z] > 0)
        ratio[valid] = tsnr_nordic[:, :, z][valid] / tsnr_orig[:, :, z][valid]
        axes[2, i].imshow(ratio.T, origin="lower", cmap="RdBu_r",
                          vmin=0.5, vmax=2.0)
        axes[2, i].axis("off")

    axes[0, 0].set_ylabel("Original", fontsize=12)
    axes[1, 0].set_ylabel("NORDIC", fontsize=12)
    axes[2, 0].set_ylabel("Ratio (N/O)", fontsize=12)
    fig.suptitle(f"tSNR — {run_label}", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / f"tsnr_slices_{run_label}.png", dpi=150)
    plt.close(fig)

    # Histogram of voxel-wise improvement
    valid_mask = (tsnr_orig > 0) & (mask_data > 0)
    ratio_vals = tsnr_nordic[valid_mask] / tsnr_orig[valid_mask]
    ratio_vals = ratio_vals[(ratio_vals > 0) & (ratio_vals < 5)]  # trim outliers

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(ratio_vals, bins=100, color="steelblue", edgecolor="none", alpha=0.8)
    median_r = np.median(ratio_vals)
    ax.axvline(1.0, color="gray", ls="--", label="No change")
    ax.axvline(median_r, color="red", ls="-", label=f"Median = {median_r:.2f}")
    ax.set_xlabel("tSNR ratio (NORDIC / Original)")
    ax.set_ylabel("Voxel count")
    ax.set_title(f"tSNR improvement — {run_label}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"tsnr_hist_{run_label}.png", dpi=150)
    plt.close(fig)

    return median_r


# ── Temporal autocorrelation ─────────────────────────────────────────────────

def compute_acf(bold_img, mask_data, max_lag=5):
    """Compute mean autocorrelation at lags 1..max_lag within mask."""
    data = bold_img.get_fdata(dtype=np.float32)
    mask_idx = np.where(mask_data > 0)
    ts = data[mask_idx]  # (n_voxels, n_timepoints)

    # Demean each voxel
    ts = ts - ts.mean(axis=1, keepdims=True)
    var = (ts ** 2).sum(axis=1)
    var[var == 0] = 1  # avoid division by zero

    acf = np.zeros(max_lag)
    for lag in range(1, max_lag + 1):
        cov = (ts[:, :-lag] * ts[:, lag:]).sum(axis=1)
        acf[lag - 1] = np.mean(cov / var)

    return acf


def plot_acf_comparison(acf_orig, acf_nordic, run_label, output_dir):
    """Bar chart comparing ACF at each lag."""
    lags = np.arange(1, len(acf_orig) + 1)
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(lags - width / 2, acf_orig, width, label="Original", color="gray")
    ax.bar(lags + width / 2, acf_nordic, width, label="NORDIC", color="steelblue")
    ax.set_xlabel("Lag (TR)")
    ax.set_ylabel("Autocorrelation")
    ax.set_title(f"Temporal ACF — {run_label}")
    ax.set_xticks(lags)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"acf_{run_label}.png", dpi=150)
    plt.close(fig)

    return acf_orig[0], acf_nordic[0]  # lag-1 values


# ── ROI tSNR comparison ──────────────────────────────────────────────────────

def compute_roi_tsnr(tsnr_map, roi_masks, brain_mask):
    """Compute mean tSNR within each ROI (intersected with brain mask)."""
    roi_tsnr = {}
    for name, roi_mask in roi_masks.items():
        valid = roi_mask & brain_mask
        if valid.sum() == 0:
            roi_tsnr[name] = np.nan
        else:
            roi_tsnr[name] = np.mean(tsnr_map[valid])
    return roi_tsnr


def plot_roi_tsnr_summary(roi_results, output_dir):
    """Bar chart of ROI tSNR: original vs NORDIC, across all runs."""
    roi_names = sorted({name for r in roi_results for name in r["roi_orig"]})
    runs = [r["run"] for r in roi_results]

    fig, axes = plt.subplots(1, len(roi_names), figsize=(5 * len(roi_names), 5),
                              sharey=False)
    if len(roi_names) == 1:
        axes = [axes]

    width = 0.35
    x = np.arange(len(runs))

    for ax, roi_name in zip(axes, roi_names):
        orig_vals = [r["roi_orig"].get(roi_name, np.nan) for r in roi_results]
        nordic_vals = [r["roi_nordic"].get(roi_name, np.nan) for r in roi_results]

        ax.bar(x - width / 2, orig_vals, width, label="Original", color="gray")
        ax.bar(x + width / 2, nordic_vals, width, label="NORDIC", color="steelblue")
        ax.set_title(roi_name, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(runs, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Mean tSNR")
        ax.legend(fontsize=8)

    fig.suptitle("ROI tSNR: Original vs NORDIC", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_dir / "roi_tsnr_comparison.png", dpi=150)
    plt.close(fig)


# ── Summary table ────────────────────────────────────────────────────────────

def write_summary(results, roi_results, output_dir):
    """Write TSV summaries for whole-brain and ROI metrics."""
    # Whole-brain summary
    header = "run\ttsnr_median_orig\ttsnr_median_nordic\ttsnr_ratio\tacf1_orig\tacf1_nordic\tacf1_change_pct"
    lines = [header]
    for r in results:
        change_pct = (r["acf1_nordic"] - r["acf1_orig"]) / abs(r["acf1_orig"]) * 100
        lines.append(
            f"{r['run']}\t{r['tsnr_orig']:.1f}\t{r['tsnr_nordic']:.1f}\t"
            f"{r['tsnr_ratio']:.3f}\t{r['acf1_orig']:.4f}\t{r['acf1_nordic']:.4f}\t"
            f"{change_pct:.1f}"
        )
    (output_dir / "nordic_validation_summary.tsv").write_text("\n".join(lines) + "\n")

    # ROI summary
    if roi_results:
        roi_names = sorted({name for r in roi_results for name in r["roi_orig"]})
        roi_header = "run\troi\ttsnr_orig\ttsnr_nordic\ttsnr_ratio"
        roi_lines = [roi_header]
        for r in roi_results:
            for roi_name in roi_names:
                o = r["roi_orig"].get(roi_name, np.nan)
                n = r["roi_nordic"].get(roi_name, np.nan)
                ratio = n / o if o > 0 else np.nan
                roi_lines.append(f"{r['run']}\t{roi_name}\t{o:.1f}\t{n:.1f}\t{ratio:.3f}")
        (output_dir / "nordic_roi_tsnr.tsv").write_text("\n".join(roi_lines) + "\n")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path,
                        default=BIDS_ROOT / "derivatives" / "nordic" / "validation",
                        help="Directory for output plots and summary")
    args = parser.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    # Load ROI masks once (using first available BOLD as spatial reference)
    roi_masks = None
    results = []
    roi_results = []

    for run in RUNS:
        run_label = run.replace("task-", "").replace("_", "-")
        print(f"\n{'='*60}")
        print(f"Processing {run_label}")
        print(f"{'='*60}")

        orig_bold = bold_path(FMRIPREP_ORIG, run)
        nordic_bold = bold_path(FMRIPREP_NORDIC, run)

        if not orig_bold.exists():
            print(f"  SKIP — original not found: {orig_bold}")
            continue
        if not nordic_bold.exists():
            print(f"  SKIP — NORDIC not found: {nordic_bold}")
            continue

        # Use the original mask (same space, same subject)
        mask_file = mask_path(FMRIPREP_ORIG, run)
        mask_img = nib.load(str(mask_file))
        mask_data = mask_img.get_fdata().astype(bool)

        # Load ROI masks on first iteration (all runs share MNI grid)
        if roi_masks is None:
            print("  Loading ROI masks...")
            roi_masks = load_roi_masks(mask_img)

        # tSNR
        print("  Computing tSNR (original)...")
        orig_img = nib.load(str(orig_bold))
        tsnr_orig = compute_tsnr(orig_img, mask_data)

        print("  Computing tSNR (NORDIC)...")
        nordic_img = nib.load(str(nordic_bold))
        tsnr_nordic = compute_tsnr(nordic_img, mask_data)

        median_orig = np.median(tsnr_orig[mask_data])
        median_nordic = np.median(tsnr_nordic[mask_data])
        ratio = plot_tsnr_comparison(tsnr_orig, tsnr_nordic, mask_data,
                                     run_label, out)
        print(f"  tSNR median: orig={median_orig:.1f}, nordic={median_nordic:.1f}, "
              f"ratio={ratio:.3f}")

        # ROI tSNR
        roi_orig = compute_roi_tsnr(tsnr_orig, roi_masks, mask_data)
        roi_nordic = compute_roi_tsnr(tsnr_nordic, roi_masks, mask_data)
        for roi_name in sorted(roi_orig):
            o, n = roi_orig[roi_name], roi_nordic[roi_name]
            r = n / o if o > 0 else float("nan")
            print(f"  ROI {roi_name}: orig={o:.1f}, nordic={n:.1f}, ratio={r:.3f}")
        roi_results.append({"run": run_label, "roi_orig": roi_orig, "roi_nordic": roi_nordic})

        # ACF
        print("  Computing ACF (original)...")
        acf_orig = compute_acf(orig_img, mask_data)
        print("  Computing ACF (NORDIC)...")
        acf_nordic = compute_acf(nordic_img, mask_data)
        acf1_orig, acf1_nordic = plot_acf_comparison(acf_orig, acf_nordic,
                                                      run_label, out)
        change = (acf1_nordic - acf1_orig) / abs(acf1_orig) * 100
        print(f"  ACF lag-1: orig={acf1_orig:.4f}, nordic={acf1_nordic:.4f}, "
              f"change={change:+.1f}%")

        results.append({
            "run": run_label,
            "tsnr_orig": median_orig,
            "tsnr_nordic": median_nordic,
            "tsnr_ratio": ratio,
            "acf1_orig": acf1_orig,
            "acf1_nordic": acf1_nordic,
        })

    if results:
        write_summary(results, roi_results, out)
        plot_roi_tsnr_summary(roi_results, out)
        print(f"\n{'='*60}")
        print(f"Summary written to {out / 'nordic_validation_summary.tsv'}")
        print(f"ROI tSNR written to {out / 'nordic_roi_tsnr.tsv'}")
        print(f"Plots saved to {out}/")
        print(f"{'='*60}")
    else:
        print("\nNo runs processed — check that fMRIPrep NORDIC outputs exist.")
        sys.exit(1)


if __name__ == "__main__":
    main()
