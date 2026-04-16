#!/usr/bin/env python3
"""
extract_roi_timeseries.py — Extract volumetric ROI timeseries for ISC benchmark.

Extracts mean ROI timeseries from MNI152NLin2009cAsym-space fMRIPrep BOLD
for two datasets:

  mmmdata:  Local NATencoding runs (sub-03/04/05, ses-19..28, run-01/02)
  pixar:    ds000228 adult subjects (sub-pixar123..155), downloaded from
            OpenNeuro S3, extracted, then deleted to minimize footprint.

ROIs are defined using Harvard-Oxford atlases (shipped with nilearn):
  - V1:           Intracalcarine Cortex (cortical label 24)
  - EAC:          Heschl's Gyrus (cortical label 45)
  - MT+:          Middle Temporal Gyrus, temporooccipital part (cortical label 13)
  - IFG:          IFG pars triangularis + opercularis (cortical labels 5, 6)
  - Hippocampus:  Left (subcortical label 9) + Right (subcortical label 19)

Cortical ROIs are split into L/R hemispheres using x < 0 (MNI convention:
left hemisphere = negative x).

Output per run:
  <out>/<dataset>/<subject>/<stem>_roi-timeseries.npz
      keys: V1_L, V1_R, EAC_L, EAC_R, MT+_L, MT+_R, IFG_L, IFG_R,
            Hippocampus_L, Hippocampus_R   (each shape: n_timepoints,)
  <out>/<dataset>/<subject>/<stem>_confounds.tsv
      (copy of fMRIPrep confounds file)

Usage:
    python extract_roi_timeseries.py --dataset mmmdata
    python extract_roi_timeseries.py --dataset pixar
    python extract_roi_timeseries.py --dataset both

Requires: neuroconda3 environment
    /home/bhutch/.conda/envs/neuroconda3/bin/python extract_roi_timeseries.py
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

# ── paths ─────────────────────────────────────────────────────────────────

BIDS_ROOT = Path("/gpfs/projects/hulacon/shared/mmmdata")
FMRIPREP = BIDS_ROOT / "derivatives" / "fmriprep"
OUTPUT_DIR = BIDS_ROOT / "derivatives" / "qc" / "isc_benchmark"

MNI_SPACE = "MNI152NLin2009cAsym_res-2"

# Pixar S3 base (public, no auth)
PIXAR_S3 = "https://openneuro-derivatives.s3.amazonaws.com/fmriprep/ds000228-fmriprep"
PIXAR_ADULTS = [f"sub-pixar{i:03d}" for i in range(123, 156)]  # 33 adults

# MMMData layout
MMMDATA_SUBJECTS = ["sub-03", "sub-04", "sub-05"]
MMMDATA_SESSIONS = [f"ses-{i}" for i in range(19, 29)]
MMMDATA_RUNS = ["run-01", "run-02"]

# ── ROI definitions (Harvard-Oxford atlas labels) ────────────────────────

# Cortical atlas: fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
CORTICAL_ROIS = {
    "V1":  [24],      # Intracalcarine Cortex
    "EAC": [45],      # Heschl's Gyrus (primary auditory cortex)
    "MT+": [13],      # Middle Temporal Gyrus, temporooccipital part
    "IFG": [5, 6],    # pars triangularis + pars opercularis
}

# Subcortical atlas: fetch_atlas_harvard_oxford("sub-maxprob-thr25-2mm")
SUBCORTICAL_ROIS = {
    "Hippocampus": {"L": [9], "R": [19]},
}


# ── atlas loading ────────────────────────────────────────────────────────

def load_roi_masks():
    """Build volumetric ROI masks from Harvard-Oxford atlases.

    Returns:
        masks: dict mapping (roi_name, hemi) -> boolean 3D array
        affine: 4x4 affine of the atlas (MNI 2mm)
    """
    from nilearn.datasets import fetch_atlas_harvard_oxford

    # Cortical atlas
    cort = fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
    cort_maps = cort.maps
    cort_img = cort_maps if isinstance(cort_maps, nib.Nifti1Image) else nib.load(cort_maps)
    cort_data = np.asarray(cort_img.dataobj).astype(int)
    affine = cort_img.affine

    # Compute x-coordinates for hemisphere split
    i_coords = np.arange(cort_data.shape[0])
    x_coords = affine[0, 0] * i_coords + affine[0, 3]  # MNI x per voxel column
    left_hemi = x_coords < 0   # shape: (n_x,)
    right_hemi = x_coords >= 0

    masks = {}
    for roi_name, label_ids in CORTICAL_ROIS.items():
        roi_mask = np.isin(cort_data, label_ids)
        n_total = roi_mask.sum()

        # Split by hemisphere (broadcast along x-axis)
        mask_l = roi_mask & left_hemi[:, None, None]
        mask_r = roi_mask & right_hemi[:, None, None]
        masks[(roi_name, "L")] = mask_l
        masks[(roi_name, "R")] = mask_r
        print(f"  ROI {roi_name}: {n_total} voxels (L={mask_l.sum()}, R={mask_r.sum()})")

    # Subcortical atlas
    sub = fetch_atlas_harvard_oxford("sub-maxprob-thr25-2mm")
    sub_maps = sub.maps
    sub_img = sub_maps if isinstance(sub_maps, nib.Nifti1Image) else nib.load(sub_maps)
    sub_data = np.asarray(sub_img.dataobj).astype(int)

    for roi_name, hemi_labels in SUBCORTICAL_ROIS.items():
        for hemi, label_ids in hemi_labels.items():
            mask = np.isin(sub_data, label_ids)
            masks[(roi_name, hemi)] = mask
            print(f"  ROI {roi_name} hemi-{hemi}: {mask.sum()} voxels")

    return masks, affine


def resample_masks_to_bold(masks, atlas_affine, bold_img):
    """Resample atlas-space masks to BOLD resolution if needed.

    Returns dict of boolean masks in BOLD voxel space.
    """
    from nilearn.image import resample_to_img

    bold_shape = bold_img.shape[:3]
    resampled = {}

    # Build a single atlas-like image for resampling reference check
    atlas_shape = list(masks.values())[0].shape
    if atlas_shape == bold_shape and np.allclose(atlas_affine, bold_img.affine):
        # Same space, no resampling needed
        return masks

    print("  Resampling ROI masks to BOLD space...")
    for key, mask in masks.items():
        mask_img = nib.Nifti1Image(mask.astype(np.float32), atlas_affine)
        resampled_img = resample_to_img(mask_img, bold_img, interpolation="nearest")
        resampled[key] = np.asarray(resampled_img.dataobj) > 0.5
    return resampled


# ── timeseries extraction ────────────────────────────────────────────────

def extract_timeseries(bold_path, masks):
    """Extract mean ROI timeseries from a 4D BOLD NIfTI.

    Returns dict: {(roi_name, hemi): 1D array of shape (n_timepoints,)}
    """
    bold_img = nib.load(str(bold_path))
    bold_data = bold_img.get_fdata(dtype=np.float32)

    timeseries = {}
    for key, mask in masks.items():
        voxels = bold_data[mask]  # (n_voxels, n_timepoints)
        if voxels.shape[0] == 0:
            print(f"  WARNING: 0 voxels for {key}, skipping")
            continue
        timeseries[key] = voxels.mean(axis=0)
    return timeseries


def save_run(timeseries, confounds_src, out_dir, stem):
    """Save extracted timeseries and confounds for one run."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save timeseries as .npz with keys like "V1_L", "EAC_R", etc.
    npz_data = {}
    for (roi, hemi), ts in timeseries.items():
        npz_data[f"{roi}_{hemi}"] = ts
    npz_path = out_dir / f"{stem}_roi-timeseries.npz"
    np.savez_compressed(npz_path, **npz_data)

    # Copy confounds TSV
    conf_dst = out_dir / f"{stem}_confounds.tsv"
    shutil.copy2(str(confounds_src), str(conf_dst))
    # Copy JSON sidecar if it exists
    json_src = confounds_src.with_suffix("").with_suffix(".json")
    if json_src.exists():
        shutil.copy2(str(json_src), str(conf_dst.with_suffix("").with_suffix(".json")))

    return npz_path


# ── MMMData extraction ───────────────────────────────────────────────────

def extract_mmmdata(masks, atlas_affine, output_dir):
    """Extract ROI timeseries from all MMMData NATencoding runs."""
    out_base = output_dir / "mmmdata"
    n_saved = 0

    for sub in MMMDATA_SUBJECTS:
        for ses in MMMDATA_SESSIONS:
            for run in MMMDATA_RUNS:
                stem = f"{sub}_{ses}_task-NATencoding_{run}"
                out_dir = out_base / sub

                # Skip if already extracted
                npz_path = out_dir / f"{stem}_roi-timeseries.npz"
                if npz_path.exists():
                    print(f"  SKIP (exists): {stem}")
                    n_saved += 1
                    continue

                bold_path = (
                    FMRIPREP / sub / ses / "func"
                    / f"{stem}_space-{MNI_SPACE}_desc-preproc_bold.nii.gz"
                )
                conf_path = (
                    FMRIPREP / sub / ses / "func"
                    / f"{stem}_desc-confounds_timeseries.tsv"
                )
                if not bold_path.exists():
                    print(f"  MISSING: {bold_path.name}")
                    continue

                print(f"  Extracting: {stem}")
                bold_img = nib.load(str(bold_path))
                run_masks = resample_masks_to_bold(masks, atlas_affine, bold_img)
                timeseries = extract_timeseries(bold_path, run_masks)
                save_run(timeseries, conf_path, out_dir, stem)
                n_saved += 1

    print(f"MMMData: {n_saved} runs saved to {out_base}")


# ── Pixar download + extraction ──────────────────────────────────────────

def download_file(url, dest):
    """Download a file using curl (available on HPC, no extra deps)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["curl", "-s", "-f", "-L", "-o", str(dest), url],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Download failed ({result.returncode}): {url}\n{result.stderr}")
    return dest


def extract_pixar(masks, atlas_affine, output_dir):
    """Download, extract, and delete Pixar adult BOLD data one subject at a time."""
    out_base = output_dir / "pixar"
    n_saved = 0
    n_skipped = 0
    n_failed = 0

    for sub in PIXAR_ADULTS:
        stem = f"{sub}_task-pixar"
        out_dir = out_base / sub

        # Skip if already extracted
        npz_path = out_dir / f"{stem}_roi-timeseries.npz"
        if npz_path.exists():
            print(f"  SKIP (exists): {sub}")
            n_skipped += 1
            continue

        print(f"  Processing: {sub}")

        # Files to download
        bold_fname = f"{stem}_space-{MNI_SPACE}_desc-preproc_bold.nii.gz"
        conf_fname = f"{stem}_desc-confounds_timeseries.tsv"
        conf_json_fname = f"{stem}_desc-confounds_timeseries.json"

        with tempfile.TemporaryDirectory(prefix=f"pixar_{sub}_") as tmpdir:
            tmpdir = Path(tmpdir)

            try:
                # Download BOLD + confounds
                bold_url = f"{PIXAR_S3}/{sub}/func/{bold_fname}"
                conf_url = f"{PIXAR_S3}/{sub}/func/{conf_fname}"
                conf_json_url = f"{PIXAR_S3}/{sub}/func/{conf_json_fname}"

                bold_path = download_file(bold_url, tmpdir / bold_fname)
                conf_path = download_file(conf_url, tmpdir / conf_fname)
                try:
                    download_file(conf_json_url, tmpdir / conf_json_fname)
                except RuntimeError:
                    pass  # JSON sidecar is optional

                # Check file sizes (guard against empty/annex-pointer files)
                bold_size_mb = bold_path.stat().st_size / (1024 * 1024)
                if bold_size_mb < 10:
                    print(f"  WARNING: {sub} BOLD only {bold_size_mb:.1f} MB — likely a pointer file, skipping")
                    n_failed += 1
                    continue

                print(f"    Downloaded BOLD: {bold_size_mb:.0f} MB")

                # Extract timeseries
                bold_img = nib.load(str(bold_path))
                run_masks = resample_masks_to_bold(masks, atlas_affine, bold_img)
                timeseries = extract_timeseries(bold_path, run_masks)
                save_run(timeseries, conf_path, out_dir, stem)
                n_saved += 1
                print(f"    Saved: {npz_path}")

            except Exception as e:
                print(f"  ERROR ({sub}): {e}", file=sys.stderr)
                n_failed += 1
                continue

            # BOLD deleted automatically when tmpdir is cleaned up

    print(f"Pixar: {n_saved} saved, {n_skipped} skipped, {n_failed} failed")


# ── save atlas masks for reproducibility ─────────────────────────────────

def save_atlas_masks(masks, affine, output_dir):
    """Save the ROI masks as NIfTI files for inspection/reproducibility."""
    mask_dir = output_dir / "roi_masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    for (roi, hemi), mask in masks.items():
        fname = f"roi-{roi}_hemi-{hemi}_space-{MNI_SPACE}_mask.nii.gz"
        img = nib.Nifti1Image(mask.astype(np.uint8), affine)
        nib.save(img, str(mask_dir / fname))

    print(f"ROI masks saved to {mask_dir}")


# ── main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset", choices=["mmmdata", "pixar", "both"], default="both",
        help="Which dataset(s) to extract (default: both)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR})",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Harvard-Oxford atlas ROI masks...")
    masks, atlas_affine = load_roi_masks()

    print("\nSaving ROI masks for reproducibility...")
    save_atlas_masks(masks, atlas_affine, output_dir)

    if args.dataset in ("mmmdata", "both"):
        print("\n=== Extracting MMMData ROI timeseries ===")
        extract_mmmdata(masks, atlas_affine, output_dir)

    if args.dataset in ("pixar", "both"):
        print("\n=== Extracting Pixar ROI timeseries ===")
        extract_pixar(masks, atlas_affine, output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
