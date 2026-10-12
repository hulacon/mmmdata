#!/usr/bin/env python3
"""Aggregate ``null_design.py`` outputs into voxel false-positive rates, overall and by tissue class.

One subject at a time. Every rest run's ``*_hits.npz`` is read (a missing run is an error,
never a smaller denominator), and per cell — design × contrast — each voxel's FPR is

    hits / valid fits      over that voxel's runs × draws,

where a fit in which the voxel's noise model was masked does not count. A voxel with no valid
fit at all is excluded and counted. The cell's statistics are the median voxel FPR (and the
pooled mean, hits / fits), over the whole mask and within each tissue class.

Tissue classes partition the mask, from fMRIPrep's T1w ``label-{GM,WM,CSF}_probseg`` resampled
linearly to the BOLD grid: CSF is where CSF is the argmax; **CSF-adjacent** is CSF dilated by one
voxel (26-neighbourhood, CSF itself included, adjacency to CSF outside the mask counts); every
other mask voxel is GM or WM by argmax.

The pass/fail reading is not here: the thresholds belong to the record that pre-registered
them. This script reports the numbers and, with ``--maps``, writes each cell's voxel-FPR map.

Usage (compute node; the run list and mask are rebuilt exactly as ``null_design.py`` does):
    .venv/bin/python scripts/braintwill_glm/score_null.py --subject sub-## \\
        --task <rest task> [<rest task> ...] --hits-dir /path/to/null_design/out --out /path/score.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from adapter import config_bids_root  # noqa: E402
from null_design import CONTRASTS, DESIGNS  # noqa: E402

from neuroimaging.data_quality_voxelmaps import _anat_one  # noqa: E402
from neuroimaging.glm.reference import reference_config  # noqa: E402
from neuroimaging.io import find_fmriprep_runs, mask_intersection  # noqa: E402

TISSUES = ("GM", "WM", "CSF")
CLASSES = ("GM", "WM", "CSF-adjacent")


def tissue_classes(fmriprep_tree: Path, subject: str, mask_img) -> np.ndarray:
    """Class label per mask voxel (index into ``CLASSES``), in the mask's C order."""
    from nilearn.image import resample_to_img
    from scipy.ndimage import binary_dilation

    sub_dir = Path(fmriprep_tree) / f"sub-{subject}"
    prob = np.stack([
        np.asarray(resample_to_img(str(_anat_one(sub_dir, subject, f"_label-{t}_probseg.nii.gz")), mask_img,
                                   interpolation="linear", force_resample=True, copy_header=True).dataobj)
        for t in TISSUES
    ], axis=-1)
    arg = prob.argmax(axis=-1)
    csf_adjacent = binary_dilation(arg == TISSUES.index("CSF"), structure=np.ones((3, 3, 3), bool))
    label = np.where(arg == TISSUES.index("WM"), CLASSES.index("WM"), CLASSES.index("GM"))
    label[csf_adjacent] = CLASSES.index("CSF-adjacent")
    return label[np.asarray(mask_img.dataobj).astype(bool)]


def accumulate(paths, n_vox: int, cells) -> tuple[dict, dict, list[int]]:
    """Per cell, each voxel's significant-and-valid count and valid-fit count, summed over runs × draws."""
    hits = {cell: np.zeros(n_vox, np.int32) for cell in cells}
    fits = {cell: np.zeros(n_vox, np.int32) for cell in cells}
    draws_per_run = []
    for path in paths:
        with np.load(path) as z:
            draws = sorted({k.rsplit("_d", 1)[1] for k in z.files if not k.endswith("_ok")})
            draws_per_run.append(len(draws))
            for (kind, c) in cells:
                for d in draws:
                    key = f"{kind}_{c}_d{d}"
                    sig = np.unpackbits(z[key], count=n_vox).astype(bool)
                    ok = np.unpackbits(z[f"{key}_ok"], count=n_vox).astype(bool)
                    hits[(kind, c)] += sig & ok
                    fits[(kind, c)] += ok
    return hits, fits, draws_per_run


def summarise(hits: np.ndarray, fits: np.ndarray) -> dict:
    keep = fits > 0
    fpr = hits[keep] / fits[keep]
    return {"voxels": int(keep.sum()), "excluded_no_valid_fit": int((~keep).sum()),
            "median_fpr": float(np.median(fpr)) if keep.any() else None,
            "pooled_mean_fpr": float(hits[keep].sum() / fits[keep].sum()) if keep.any() else None,
            "fpr_p95": float(np.quantile(fpr, 0.95)) if keep.any() else None,
            "fits_min": int(fits[keep].min()) if keep.any() else 0, "fits_max": int(fits.max())}


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subject", required=True)
    p.add_argument("--task", required=True, nargs="+", help="resting task label(s), as given to null_design.py")
    p.add_argument("--hits-dir", type=Path, required=True, help="null_design.py --out-dir")
    p.add_argument("--noise", default="ar3-bc")
    p.add_argument("--space", default="T1w")
    p.add_argument("--regime", default="reference")
    p.add_argument("--out", type=Path, required=True, help="score JSON")
    p.add_argument("--maps", type=Path, help="directory for per-cell voxel-FPR and class NIfTIs")
    return p.parse_args(argv)


def main(argv=None) -> int:
    import nibabel as nib

    args = parse_args(argv)
    subject = args.subject.removeprefix("sub-")
    bids_root = config_bids_root()
    cfg = dataclasses.replace(reference_config(args.regime), space=args.space)
    runs = []
    for task in args.task:
        runs += sorted(find_fmriprep_runs(subject=subject, task=task, variant=cfg.variant, space=args.space,
                                          bids_root=bids_root), key=lambda r: (r.session, r.run or ""))
    if not runs:
        raise SystemExit(f"ERROR: no {args.task} runs for sub-{subject} in {args.space}")
    mask_img, mask = mask_intersection(runs)
    n_vox = int(mask.sum())
    fmriprep_tree = Path(runs[0].bold).parents[3]
    label = tissue_classes(fmriprep_tree, subject, mask_img)

    stems = [f"sub-{subject}_{r.entity_prefix.split('_', 1)[1]}_noise-{args.noise}" for r in runs]
    paths = [args.hits_dir / f"sub-{subject}" / f"{s}_hits.npz" for s in stems]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise SystemExit(f"ERROR: {len(missing)} of {len(paths)} runs have no hits file, e.g. {missing[:3]}")

    cells = [(kind, c) for kind in DESIGNS for c in CONTRASTS]
    hits, fits, draws_per_run = accumulate(paths, n_vox, cells)

    record = {"subject": subject, "noise": args.noise, "space": args.space, "runs": len(runs),
              "draws_per_run": sorted(set(draws_per_run)), "mask_voxels": n_vox,
              "class_voxels": {name: int((label == i).sum()) for i, name in enumerate(CLASSES)},
              "class_rule": "partition; CSF-adjacent = dilate(argmax==CSF, 26-nbhd); rest GM/WM by argmax",
              "cells": {}}
    if args.maps:
        args.maps.mkdir(parents=True, exist_ok=True)
        vol = np.zeros(mask.shape, np.int16)
        vol[mask] = label + 1
        nib.Nifti1Image(vol, mask_img.affine).to_filename(args.maps / f"sub-{subject}_desc-r1classes_dseg.nii.gz")
    for (kind, c) in cells:
        h, f = hits[(kind, c)], fits[(kind, c)]
        entry = {"all": summarise(h, f)}
        for i, name in enumerate(CLASSES):
            entry[name] = summarise(h[label == i], f[label == i])
        gm, csfa = entry["GM"]["median_fpr"], entry["CSF-adjacent"]["median_fpr"]
        entry["csf_adjacent_minus_gm"] = None if gm is None or csfa is None else csfa - gm
        record["cells"][f"{kind}_{c}"] = entry
        if args.maps:
            vol = np.full(mask.shape, np.nan, np.float32)
            with np.errstate(invalid="ignore", divide="ignore"):
                vol[mask] = np.where(f > 0, h / np.maximum(f, 1), np.nan)
            nib.Nifti1Image(vol, mask_img.affine).to_filename(
                args.maps / f"sub-{subject}_design-{kind}_contrast-{c}_noise-{args.noise}_fpr.nii.gz")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=1))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
