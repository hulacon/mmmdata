#!/usr/bin/env python3
"""Compare hippocampal segmentations across FreeSurfer, HippUnfold, and HSF.

First-pass comparison: Dice coefficients and volumes in native T1w space.

Usage:
    python compare_hippo_segmentations.py [--subjects sub-03 sub-04 sub-05]

Requires: nibabel, numpy, scipy (all in mmmdata venv).
FSL flirt must be on PATH for HSF T2w→T1w registration.
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────
BIDS_ROOT = Path("/gpfs/projects/hulacon/shared/mmmdata")
FMRIPREP = BIDS_ROOT / "derivatives" / "fmriprep"
HIPPUNFOLD = BIDS_ROOT / "derivatives" / "hippunfold" / "hippunfold"
HIPPUNFOLD_OBLCOR = BIDS_ROOT / "derivatives" / "hippunfold_oblcor" / "hippunfold"
HSF_DIR = BIDS_ROOT / "derivatives" / "hsf"
HSF_OBLCOR_DIR = BIDS_ROOT / "derivatives" / "hsf_oblcor"

# ── Label schemes ─────────────────────────────────────────────────────────
# HippUnfold (multihist7 atlas): label → name
HU_LABELS = {1: "Sub", 2: "CA1", 3: "CA2", 4: "CA3", 5: "CA4", 6: "DG", 7: "SRLM", 8: "Cyst"}

# HSF (ca_mode=1/2/3): label → name
HSF_LABELS = {1: "DG", 2: "CA1", 3: "CA2", 4: "CA3", 5: "Sub"}

# Shared subfields for cross-pipeline Dice (HippUnfold label → HSF label)
SHARED_SUBFIELDS = {
    "CA1": (2, 2),   # HU label, HSF label
    "CA2": (3, 3),
    "CA3": (4, 4),
    "DG":  (6, 1),
    "Sub": (1, 5),
}

# FreeSurfer aseg: whole hippocampus only
FS_LABELS = {"L": 17, "R": 53}


def dice(mask_a, mask_b):
    """Dice coefficient between two binary masks."""
    intersection = np.sum(mask_a & mask_b)
    total = np.sum(mask_a) + np.sum(mask_b)
    if total == 0:
        return float("nan")
    return 2.0 * intersection / total


def volume_mm3(mask, voxel_vol):
    """Volume in mm³."""
    return float(np.sum(mask)) * voxel_vol


def load_nifti(path):
    """Load NIfTI, return (data_array, voxel_volume_mm3, nibabel_image)."""
    img = nib.load(str(path))
    data = np.asarray(img.dataobj).astype(np.float32)
    voxvol = np.abs(np.linalg.det(img.affine[:3, :3]))
    return data, voxvol, img


def get_freesurfer_hippo_t1w(subject):
    """Extract FreeSurfer hippocampus labels resampled to fMRIPrep T1w space.

    Uses mri_vol2vol to resample aseg.mgz to the fMRIPrep T1w reference grid.
    Returns dict with 'L' and 'R' binary masks + voxel volume.
    """
    fs_dir = FMRIPREP / "sourcedata" / "freesurfer" / subject / "mri"
    aseg_mgz = fs_dir / "aseg.mgz"
    rawavg = fs_dir / "rawavg.mgz"  # original T1w (for header reference)

    # fMRIPrep T1w reference
    t1w_ref = FMRIPREP / subject / "anat" / f"{subject}_acq-MPR_desc-preproc_T1w.nii.gz"

    if not aseg_mgz.exists():
        print(f"  WARNING: {aseg_mgz} not found, skipping FreeSurfer")
        return None

    # Use mri_label2vol or just load aseg and resample
    # Simpler: use mri_vol2vol to go from FS conformed → T1w native
    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
        tmp_nii = tmp.name

    # mri_vol2vol: resample aseg to T1w space using nearest-neighbor
    cmd = [
        "mri_vol2vol",
        "--mov", str(aseg_mgz),
        "--targ", str(t1w_ref),
        "--regheader",
        "--interp", "nearest",
        "--o", tmp_nii,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  WARNING: mri_vol2vol failed: {result.stderr[:200]}")
        # Fallback: try loading aseg directly with nibabel
        return _load_aseg_nibabel(aseg_mgz, t1w_ref)

    data, voxvol, _ = load_nifti(tmp_nii)
    Path(tmp_nii).unlink(missing_ok=True)

    masks = {}
    for hemi, label in FS_LABELS.items():
        masks[hemi] = data == label

    return {"masks": masks, "voxvol": voxvol}


def _load_aseg_nibabel(aseg_mgz, t1w_ref):
    """Fallback: load aseg.mgz with nibabel, resample to T1w grid."""
    from scipy.ndimage import affine_transform

    aseg_img = nib.load(str(aseg_mgz))
    aseg_data = np.asarray(aseg_img.dataobj).astype(np.float32)

    t1w_img = nib.load(str(t1w_ref))
    t1w_shape = t1w_img.shape[:3]

    # Compute transform: aseg voxel → world → T1w voxel
    aseg_to_world = aseg_img.affine
    world_to_t1w = np.linalg.inv(t1w_img.affine)
    aseg_to_t1w = world_to_t1w @ aseg_to_world

    # scipy affine_transform uses the inverse mapping (output→input)
    t1w_to_aseg = np.linalg.inv(aseg_to_t1w)

    resampled = affine_transform(
        aseg_data,
        t1w_to_aseg[:3, :3],
        offset=t1w_to_aseg[:3, 3],
        output_shape=t1w_shape,
        order=0,  # nearest neighbor
    )

    voxvol = np.abs(np.linalg.det(t1w_img.affine[:3, :3]))
    masks = {}
    for hemi, label in FS_LABELS.items():
        masks[hemi] = resampled == label

    return {"masks": masks, "voxvol": voxvol}


def get_hippunfold_t1w(subject, deriv_dir=None):
    """Load HippUnfold T1w-space dseg for both hemispheres.

    Returns dict with per-hemi label arrays + voxel volume.
    """
    if deriv_dir is None:
        deriv_dir = HIPPUNFOLD
    result = {}
    for hemi in ("L", "R"):
        dseg_path = (
            deriv_dir / subject / "ses-01" / "anat"
            / f"{subject}_ses-01_hemi-{hemi}_space-T1w_desc-subfields_atlas-multihist7_dseg.nii.gz"
        )
        if not dseg_path.exists():
            print(f"  WARNING: {dseg_path} not found")
            return None
        data, voxvol, _ = load_nifti(dseg_path)
        result[hemi] = data
    result["voxvol"] = voxvol
    return result


def find_hsf_output(subject, hsf_dir):
    """Find HSF segmentation output file.

    HSF writes output alongside the input or in the working directory.
    Need to discover the actual output location.
    """
    # Check common HSF output patterns
    search_dirs = [
        hsf_dir / subject,
        # HSF may write next to input
        BIDS_ROOT / subject / "ses-01" / "anat",
    ]
    patterns = ["*seg*.nii*", "*hsf*.nii*", "*mask*.nii*"]

    for d in search_dirs:
        if not d.exists():
            continue
        for pat in patterns:
            matches = sorted(d.glob(pat))
            if matches:
                return matches

    # Also check Hydra output dirs inside the HSF output dir
    hsf_sub = hsf_dir / subject
    if hsf_sub.exists():
        matches = sorted(hsf_sub.rglob("*.nii*"))
        if matches:
            return matches

    return []


def register_t2w_to_t1w(subject, t2w_acq="SPC"):
    """Register T2w to T1w using flirt 6-DOF, return transform matrix path."""
    if t2w_acq == "SPC":
        t2w_path = BIDS_ROOT / subject / "ses-01" / "anat" / f"{subject}_ses-01_acq-SPC_T2w.nii.gz"
    else:
        t2w_path = BIDS_ROOT / subject / "ses-01" / "anat" / f"{subject}_ses-01_acq-oblcor_T2w.nii.gz"

    t1w_path = FMRIPREP / subject / "anat" / f"{subject}_acq-MPR_desc-preproc_T1w.nii.gz"

    if not t2w_path.exists() or not t1w_path.exists():
        print(f"  WARNING: Missing {t2w_path} or {t1w_path}")
        return None, None

    omat = tempfile.NamedTemporaryFile(suffix=".mat", delete=False).name

    cmd = [
        "flirt",
        "-in", str(t2w_path),
        "-ref", str(t1w_path),
        "-omat", omat,
        "-dof", "6",
        "-cost", "mutualinfo",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  WARNING: flirt failed: {result.stderr[:200]}")
        return None, None

    return omat, t1w_path


def apply_flirt_to_labels(input_nii, ref_nii, omat, output_nii=None):
    """Apply flirt transform to a label image using nearest-neighbor."""
    if output_nii is None:
        output_nii = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False).name

    cmd = [
        "flirt",
        "-in", str(input_nii),
        "-ref", str(ref_nii),
        "-applyxfm",
        "-init", str(omat),
        "-interp", "nearestneighbour",
        "-out", str(output_nii),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  WARNING: flirt apply failed: {result.stderr[:200]}")
        return None
    return output_nii


def load_hsf_to_t1w(subject, hsf_dir, t2w_acq="SPC"):
    """Find HSF output, register to T1w space, return label array + voxvol.

    Returns dict with 'data' (3D label array in T1w space), 'voxvol', or None.
    """
    hsf_files = find_hsf_output(subject, hsf_dir)
    if not hsf_files:
        print(f"  HSF ({t2w_acq}): No output files found")
        return None

    # Find the segmentation file
    seg_file = None
    for f in hsf_files:
        if "seg" in f.name.lower():
            seg_file = f
            break
    if seg_file is None:
        print(f"  HSF ({t2w_acq}): No seg file among {[f.name for f in hsf_files]}")
        return None

    print(f"  HSF ({t2w_acq}) seg file: {seg_file.name}")

    omat, t1w_ref = register_t2w_to_t1w(subject, t2w_acq)
    if omat is None:
        return None

    resampled = apply_flirt_to_labels(seg_file, t1w_ref, omat)
    Path(omat).unlink(missing_ok=True)
    if resampled is None:
        return None

    data, voxvol, _ = load_nifti(resampled)
    Path(resampled).unlink(missing_ok=True)

    labels_found = np.unique(data[data > 0]).astype(int)
    print(f"    Labels in T1w space: {labels_found}")

    return {"data": data, "voxvol": voxvol}


def print_subfield_dice(label_a, label_b, voxvol_a, voxvol_b, map_a, map_b,
                        name_a, name_b, subfields):
    """Print per-subfield Dice table between two label arrays.

    map_a/map_b: dict mapping subfield name → integer label for each pipeline.
    """
    print(f"\n  {name_a} vs {name_b} — per-subfield Dice:")
    for hemi in ("L", "R"):
        print(f"    {hemi}:")
        a_data = label_a[hemi] if isinstance(label_a, dict) else label_a
        b_data = label_b[hemi] if isinstance(label_b, dict) else label_b
        for sf_name in subfields:
            a_label = map_a[sf_name]
            b_label = map_b[sf_name]
            a_mask = a_data == a_label
            b_mask = b_data == b_label
            d = dice(a_mask, b_mask)
            a_v = volume_mm3(a_mask, voxvol_a)
            b_v = volume_mm3(b_mask, voxvol_b)
            print(f"      {sf_name:4s}: Dice={d:.3f}  {name_a}={a_v:.0f}mm³  {name_b}={b_v:.0f}mm³")


def print_whole_hippo_dice(label_a, label_b, voxvol_a, voxvol_b, name_a, name_b,
                           a_is_binary=False, b_is_binary=False,
                           a_exclude=None, b_exclude=None):
    """Print whole-hippocampus Dice between two segmentations.

    For label maps, whole hippo = any label > 0 minus excluded labels.
    For binary masks (like FreeSurfer per-hemi), use as-is.
    """
    print(f"\n  {name_a} vs {name_b} — whole hippocampus:")
    for hemi in ("L", "R"):
        if isinstance(label_a, dict) and hemi in label_a:
            a_data = label_a[hemi]
        elif isinstance(label_a, dict) and "data" in label_a:
            a_data = label_a["data"]
        else:
            a_data = label_a

        if isinstance(label_b, dict) and hemi in label_b:
            b_data = label_b[hemi]
        elif isinstance(label_b, dict) and "data" in label_b:
            b_data = label_b["data"]
        else:
            b_data = label_b

        if a_is_binary:
            a_mask = a_data.astype(bool)
        else:
            a_mask = a_data > 0
            if a_exclude:
                for ex in a_exclude:
                    a_mask = a_mask & (a_data != ex)

        if b_is_binary:
            b_mask = b_data.astype(bool)
        else:
            b_mask = b_data > 0
            if b_exclude:
                for ex in b_exclude:
                    b_mask = b_mask & (b_data != ex)

        d = dice(a_mask, b_mask)
        a_v = volume_mm3(a_mask, voxvol_a)
        b_v = volume_mm3(b_mask, voxvol_b)
        print(f"    {hemi}: Dice={d:.3f}  {name_a}={a_v:.0f}mm³  {name_b}={b_v:.0f}mm³")


def process_subject(subject):
    """Run all comparisons for one subject."""
    print(f"\n{'='*60}")
    print(f"  {subject}")
    print(f"{'='*60}")

    # ── Load all segmentations ─────────────────────────────────────────
    hu_spc = get_hippunfold_t1w(subject, HIPPUNFOLD)
    hu_oblcor = get_hippunfold_t1w(subject, HIPPUNFOLD_OBLCOR)
    fs = get_freesurfer_hippo_t1w(subject)
    hsf_spc = load_hsf_to_t1w(subject, HSF_DIR, "SPC")
    hsf_oblcor = load_hsf_to_t1w(subject, HSF_OBLCOR_DIR, "oblcor")

    results = {"subject": subject}

    # Subfield name → label mappings for each pipeline
    hu_sf_map = {"CA1": 2, "CA2": 3, "CA3": 4, "DG": 6, "Sub": 1}
    hsf_sf_map = {"CA1": 2, "CA2": 3, "CA3": 4, "DG": 1, "Sub": 5}
    shared_sf_names = list(hu_sf_map.keys())

    # ── 1. FreeSurfer vs HippUnfold (whole hippocampus) ────────────────
    if fs and hu_spc:
        print("\n  FreeSurfer vs HippUnfold (SPC) — whole hippocampus:")
        for hemi in ("L", "R"):
            fs_mask = fs["masks"][hemi]
            hu_mask = (hu_spc[hemi] > 0) & (hu_spc[hemi] != 8)
            d = dice(fs_mask, hu_mask)
            fs_vol = volume_mm3(fs_mask, fs["voxvol"])
            hu_vol = volume_mm3(hu_mask, hu_spc["voxvol"])
            print(f"    {hemi}: Dice={d:.3f}  FS_vol={fs_vol:.0f}mm³  HU_vol={hu_vol:.0f}mm³")
            results[f"fs_vs_hu_{hemi}_dice"] = d
            results[f"fs_{hemi}_vol"] = fs_vol
            results[f"hu_spc_{hemi}_vol"] = hu_vol

    # ── 2. FreeSurfer vs HSF (whole hippocampus) ───────────────────────
    if fs and hsf_spc:
        print("\n  FreeSurfer vs HSF (SPC) — whole hippocampus:")
        for hemi in ("L", "R"):
            fs_mask = fs["masks"][hemi]
            hsf_mask = hsf_spc["data"] > 0
            d = dice(fs_mask, hsf_mask)
            fs_vol = volume_mm3(fs_mask, fs["voxvol"])
            hsf_vol = volume_mm3(hsf_mask, hsf_spc["voxvol"])
            print(f"    {hemi}: Dice={d:.3f}  FS_vol={fs_vol:.0f}mm³  HSF_vol={hsf_vol:.0f}mm³")
            results[f"fs_vs_hsf_{hemi}_dice"] = d

    # ── 3. HSF vs HippUnfold (per-subfield + whole) ───────────────────
    if hsf_spc and hu_spc:
        print("\n  HSF vs HippUnfold (SPC) — per-subfield Dice:")
        for hemi in ("L", "R"):
            print(f"    {hemi}:")
            for sf in shared_sf_names:
                hu_mask = hu_spc[hemi] == hu_sf_map[sf]
                hsf_mask = hsf_spc["data"] == hsf_sf_map[sf]
                d = dice(hu_mask, hsf_mask)
                hu_v = volume_mm3(hu_mask, hu_spc["voxvol"])
                hsf_v = volume_mm3(hsf_mask, hsf_spc["voxvol"])
                print(f"      {sf:4s}: Dice={d:.3f}  HU={hu_v:.0f}mm³  HSF={hsf_v:.0f}mm³")

        print("\n  HSF vs HippUnfold (SPC) — whole hippocampus:")
        for hemi in ("L", "R"):
            hu_mask = (hu_spc[hemi] > 0) & (hu_spc[hemi] != 8)
            hsf_mask = hsf_spc["data"] > 0
            d = dice(hu_mask, hsf_mask)
            print(f"    {hemi}: Dice={d:.3f}")

    # ── 4. HippUnfold subfield volumes (SPC) ──────────────────────────
    if hu_spc:
        print("\n  HippUnfold (SPC) subfield volumes (mm³):")
        for hemi in ("L", "R"):
            vols = []
            for sf in shared_sf_names:
                v = volume_mm3(hu_spc[hemi] == hu_sf_map[sf], hu_spc["voxvol"])
                vols.append(f"{sf}={v:.0f}")
            print(f"    {hemi}: {', '.join(vols)}")

    # ── 5. FreeSurfer vs HippUnfold oblcor (whole hippocampus) ────────
    if fs and hu_oblcor:
        print("\n  FreeSurfer vs HippUnfold (oblcor) — whole hippocampus:")
        for hemi in ("L", "R"):
            fs_mask = fs["masks"][hemi]
            hu_mask = (hu_oblcor[hemi] > 0) & (hu_oblcor[hemi] != 8)
            d = dice(fs_mask, hu_mask)
            fs_vol = volume_mm3(fs_mask, fs["voxvol"])
            hu_vol = volume_mm3(hu_mask, hu_oblcor["voxvol"])
            print(f"    {hemi}: Dice={d:.3f}  FS_vol={fs_vol:.0f}mm³  HU_vol={hu_vol:.0f}mm³")

    # ── 6. FreeSurfer vs HSF oblcor (whole hippocampus) ───────────────
    if fs and hsf_oblcor:
        print("\n  FreeSurfer vs HSF (oblcor) — whole hippocampus:")
        for hemi in ("L", "R"):
            fs_mask = fs["masks"][hemi]
            hsf_mask = hsf_oblcor["data"] > 0
            d = dice(fs_mask, hsf_mask)
            fs_vol = volume_mm3(fs_mask, fs["voxvol"])
            hsf_vol = volume_mm3(hsf_mask, hsf_oblcor["voxvol"])
            print(f"    {hemi}: Dice={d:.3f}  FS_vol={fs_vol:.0f}mm³  HSF_vol={hsf_vol:.0f}mm³")

    # ── 7. HSF oblcor vs HippUnfold oblcor (per-subfield + whole) ─────
    if hsf_oblcor and hu_oblcor:
        print("\n  HSF vs HippUnfold (oblcor) — per-subfield Dice:")
        for hemi in ("L", "R"):
            print(f"    {hemi}:")
            for sf in shared_sf_names:
                hu_mask = hu_oblcor[hemi] == hu_sf_map[sf]
                hsf_mask = hsf_oblcor["data"] == hsf_sf_map[sf]
                d = dice(hu_mask, hsf_mask)
                hu_v = volume_mm3(hu_mask, hu_oblcor["voxvol"])
                hsf_v = volume_mm3(hsf_mask, hsf_oblcor["voxvol"])
                print(f"      {sf:4s}: Dice={d:.3f}  HU={hu_v:.0f}mm³  HSF={hsf_v:.0f}mm³")

        print("\n  HSF vs HippUnfold (oblcor) — whole hippocampus:")
        for hemi in ("L", "R"):
            hu_mask = (hu_oblcor[hemi] > 0) & (hu_oblcor[hemi] != 8)
            hsf_mask = hsf_oblcor["data"] > 0
            d = dice(hu_mask, hsf_mask)
            print(f"    {hemi}: Dice={d:.3f}")

    # ── 8. Cross-pipeline cross-acquisition ───────────────────────────
    if hsf_spc and hu_oblcor:
        print("\n  HSF (SPC) vs HippUnfold (oblcor) — whole hippocampus:")
        for hemi in ("L", "R"):
            hu_mask = (hu_oblcor[hemi] > 0) & (hu_oblcor[hemi] != 8)
            hsf_mask = hsf_spc["data"] > 0
            d = dice(hu_mask, hsf_mask)
            print(f"    {hemi}: Dice={d:.3f}")

    if hsf_oblcor and hu_spc:
        print("\n  HSF (oblcor) vs HippUnfold (SPC) — whole hippocampus:")
        for hemi in ("L", "R"):
            hu_mask = (hu_spc[hemi] > 0) & (hu_spc[hemi] != 8)
            hsf_mask = hsf_oblcor["data"] > 0
            d = dice(hu_mask, hsf_mask)
            print(f"    {hemi}: Dice={d:.3f}")

    # ── 9. HippUnfold SPC vs oblcor (per-subfield) ────────────────────
    if hu_spc and hu_oblcor:
        print("\n  HippUnfold SPC vs oblcor — per-subfield Dice:")
        for hemi in ("L", "R"):
            print(f"    {hemi}:")
            for sf in shared_sf_names:
                spc_mask = hu_spc[hemi] == hu_sf_map[sf]
                obl_mask = hu_oblcor[hemi] == hu_sf_map[sf]
                d = dice(spc_mask, obl_mask)
                spc_v = volume_mm3(spc_mask, hu_spc["voxvol"])
                obl_v = volume_mm3(obl_mask, hu_oblcor["voxvol"])
                print(f"      {sf:4s}: Dice={d:.3f}  SPC={spc_v:.0f}mm³  oblcor={obl_v:.0f}mm³")

    # ── 10. HSF SPC vs oblcor (per-subfield) ──────────────────────────
    if hsf_spc and hsf_oblcor:
        print("\n  HSF SPC vs oblcor — per-subfield Dice:")
        for hemi in ("L", "R"):
            print(f"    {hemi}:")
            for sf in shared_sf_names:
                spc_mask = hsf_spc["data"] == hsf_sf_map[sf]
                obl_mask = hsf_oblcor["data"] == hsf_sf_map[sf]
                d = dice(spc_mask, obl_mask)
                spc_v = volume_mm3(spc_mask, hsf_spc["voxvol"])
                obl_v = volume_mm3(obl_mask, hsf_oblcor["voxvol"])
                print(f"      {sf:4s}: Dice={d:.3f}  SPC={spc_v:.0f}mm³  oblcor={obl_v:.0f}mm³")

        print("\n  HSF SPC vs oblcor — whole hippocampus:")
        for hemi in ("L", "R"):
            spc_mask = hsf_spc["data"] > 0
            obl_mask = hsf_oblcor["data"] > 0
            d = dice(spc_mask, obl_mask)
            print(f"    {hemi}: Dice={d:.3f}")

    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subjects", nargs="+", default=["sub-03", "sub-04", "sub-05"],
        help="Subjects to compare",
    )
    args = parser.parse_args()

    print("Hippocampal Segmentation Comparison")
    print("=" * 60)
    print(f"Pipelines: FreeSurfer (aseg), HippUnfold v1.5.2, HSF v1.2.3")
    print(f"Space: Native T1w (fMRIPrep reference)")
    print(f"Subjects: {', '.join(args.subjects)}")

    all_results = []
    for subject in args.subjects:
        r = process_subject(subject)
        all_results.append(r)

    # ── Summary table ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  SUMMARY: FreeSurfer vs HippUnfold (whole hippocampus)")
    print(f"{'='*60}")
    print(f"  {'Subject':<8} {'Hemi':<5} {'Dice':<8} {'FS vol':<10} {'HU vol':<10}")
    print(f"  {'-'*8} {'-'*5} {'-'*8} {'-'*10} {'-'*10}")
    for r in all_results:
        for hemi in ("L", "R"):
            d = r.get(f"fs_vs_hu_{hemi}_dice", float("nan"))
            fv = r.get(f"fs_{hemi}_vol", float("nan"))
            hv = r.get(f"hu_spc_{hemi}_vol", float("nan"))
            print(f"  {r['subject']:<8} {hemi:<5} {d:<8.3f} {fv:<10.0f} {hv:<10.0f}")


if __name__ == "__main__":
    main()
