#!/usr/bin/env python3
"""Export the HRF-selection outputs of one GLMsingle arm as small NIfTIs.

GLMsingle keeps ``HRFindex`` (the 1-of-20 library kernel each voxel chose),
``FitHRFR2`` (the R2 of every kernel at every voxel, from which the choice was
made), ``R2`` and ``meanvol`` inside a multi-GB pickled dict (``TYPEB_FITHRF``).
Anything that wants to compare HRF choices across arms should not have to load
betas to do it, so this writes the four as NIfTIs under ``<arm>/hrf/``:

    sub-##_arm-<arm>_space-<space>_desc-hrfindex_dseg.nii.gz   (int16)
    sub-##_arm-<arm>_space-<space>_desc-fithrfr2_stat.nii.gz   (4D, 20 kernels)
    sub-##_arm-<arm>_space-<space>_desc-hrfr2_stat.nii.gz
    sub-##_arm-<arm>_space-<space>_desc-meanvol_bold.nii.gz

The grid comes from ``--reference``, a NIfTI on the fit's grid (the arm has no
affine of its own); a shape mismatch is refused. Loading the pickle needs RAM
for the whole dict (~16 G for a 42-run TB fit) — run it under sbatch.

Usage:
    python glmsingle_export_hrf.py --subject sub-## --arm enc \
        --reference <NIfTI on the fit grid> [--tree derivatives/glmsingle_tb]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "python"))

SPACE = "MNI152NLin2009cAsym_res-2"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--reference", required=True,
                    help="NIfTI on the fit's grid, supplies shape + affine")
    ap.add_argument("--tree", default=None,
                    help="GLMsingle tree; default <output_dir>/glmsingle_tb")
    ap.add_argument("--space", default=SPACE)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    import nibabel as nib

    if args.tree is None:
        from core.config import load_config
        tree = Path(load_config()["paths"]["output_dir"]) / "glmsingle_tb"
    else:
        tree = Path(args.tree)
    sub = args.subject if args.subject.startswith("sub-") else f"sub-{args.subject}"
    arm_dir = tree / sub / args.arm
    src = arm_dir / "glmsingle_outputs" / "TYPEB_FITHRF.npy"
    if not src.exists():
        sys.exit(f"ERROR: {src} missing; nothing to export")
    out_dir = arm_dir / "hrf"
    stem = f"{sub}_arm-{args.arm}_space-{args.space}"
    targets = {
        "HRFindex": out_dir / f"{stem}_desc-hrfindex_dseg.nii.gz",
        "FitHRFR2": out_dir / f"{stem}_desc-fithrfr2_stat.nii.gz",
        "R2": out_dir / f"{stem}_desc-hrfr2_stat.nii.gz",
        "meanvol": out_dir / f"{stem}_desc-meanvol_bold.nii.gz",
    }
    if all(p.exists() for p in targets.values()) and not args.force:
        print(f"{sub}/{args.arm}: all four exports exist; --force to rewrite")
        return 0

    ref = nib.load(args.reference)
    print(f"{sub}/{args.arm}: loading {src} ({src.stat().st_size / 1e9:.1f} GB pickle)")
    t0 = time.time()
    d = np.load(src, allow_pickle=True).item()
    print(f"  loaded in {time.time() - t0:.0f} s; keys {sorted(d)}")
    out_dir.mkdir(parents=True, exist_ok=True)
    for key, path in targets.items():
        arr = np.asarray(d[key])
        if arr.shape[:3] != tuple(ref.shape[:3]):
            sys.exit(f"ERROR: {key} shape {arr.shape} does not match the reference "
                     f"grid {ref.shape[:3]}")
        dtype = np.int16 if key == "HRFindex" else np.float32
        nib.Nifti1Image(arr.astype(dtype), ref.affine).to_filename(str(path))
        print(f"  {key} {arr.shape} -> {path}")
    del d
    return 0


if __name__ == "__main__":
    sys.exit(main())
