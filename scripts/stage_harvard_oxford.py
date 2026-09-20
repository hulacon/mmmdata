#!/usr/bin/env python3
"""Stage the Harvard-Oxford maxprob atlases on the shared res-2 grid.

Resamples FSL's Harvard-Oxford maxprob-thr25 2 mm cortical and subcortical
atlases (the copies nilearn fetches) ONCE, nearest-neighbour, onto the voxel
grid of the Schaefer ``dseg`` files already in ``derivatives/atlases``, and
writes them beside them with BIDS-style names, label TSVs, and sidecars that
state the source template.

Why: every analysis that used Harvard-Oxford fetched it at run time and
resampled it per call onto whichever BOLD image it had in hand. FSL ships
the atlas in MNI152NLin6Asym; the dataset's MNI outputs are
MNI152NLin2009cAsym, and ``resample_to_img`` crosses that boundary silently.
Staging once makes the crossing explicit (in the sidecar) and the grid fixed.

No cross-template registration is applied. This is a resampling of the
MNI152NLin6Asym labels onto the MNI152NLin2009cAsym res-2 voxel grid, which
is exactly what the per-call path did implicitly.

Usage::

    python scripts/stage_harvard_oxford.py [--atlases-dir DIR]
        [--reference-npz NPZ] [--overwrite]

Checks, each a hard failure:

* every Schaefer dseg in the atlases tree shares one grid, and the output
  matches it (shape + affine)
* the output matches the grid recorded in a pattern-similarity cache npz
  (``grid_shape`` / ``affine``) when ``--reference-npz`` is given
* the written label TSVs round-trip nilearn's label lists

Then prints a per-ROI voxel-count table, source grid vs staged vs the old
per-call fetch+resample route, for the six benchmark ROIs plus VTC.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

TEMPLATE = "MNI152NLin2009cAsym"
SOURCE_TEMPLATE = "MNI152NLin6Asym"
ANAT = f"tpl-{TEMPLATE}/anat"

# atlas entity -> (nilearn fetch name, human name)
ATLASES = {
    "HOCPA": ("cort-maxprob-thr25-2mm", "Harvard-Oxford cortical structural atlas"),
    "HOSPA": ("sub-maxprob-thr25-2mm", "Harvard-Oxford subcortical structural atlas"),
}
DESC = "th25"

_DEFAULT_ATLASES_DIR = Path(
    "/gpfs/projects/hulacon/shared/mmmdata/derivatives/atlases"
)
_PS_SHARED = Path(__file__).resolve().parent / "pattern_similarity" / "shared.py"

# VTC per the neural-rotation-pilot charter (Ye et al. 2020 reading of
# Harvard-Oxford maxprob-thr25): Temporal Fusiform ant/post, Parahippocampal
# ant/post, Inferior Temporal ant/post/temporooccipital. Reported here only;
# it is not one of the six benchmark ROIs in pattern_similarity/shared.py.
VTC_CORTICAL_LABELS = {
    14: "Inferior Temporal Gyrus, anterior division",
    15: "Inferior Temporal Gyrus, posterior division",
    16: "Inferior Temporal Gyrus, temporooccipital part",
    34: "Parahippocampal Gyrus, anterior division",
    35: "Parahippocampal Gyrus, posterior division",
    37: "Temporal Fusiform Cortex, anterior division",
    38: "Temporal Fusiform Cortex, posterior division",
}


def staged_name(atlas: str, ext: str) -> str:
    return f"tpl-{TEMPLATE}_atlas-{atlas}_res-2_desc-{DESC}_dseg{ext}"


def reference_grid(atlases_dir: Path) -> nib.Nifti1Image:
    """One Schaefer dseg, after asserting every Schaefer dseg shares its grid."""
    files = sorted((atlases_dir / ANAT).glob("*_atlas-Schaefer2018_*_dseg.nii.gz"))
    if not files:
        raise FileNotFoundError(
            f"no Schaefer dseg under {atlases_dir / ANAT}: the reference grid is "
            "missing, so there is nothing to resample onto"
        )
    ref = nib.load(files[0])
    for f in files[1:]:
        img = nib.load(f)
        if img.shape != ref.shape or not np.allclose(img.affine, ref.affine):
            raise RuntimeError(f"Schaefer grids differ: {files[0].name} vs {f.name}")
    return ref


def fetch(atlas_key: str):
    from nilearn.datasets import fetch_atlas_harvard_oxford

    atlas = fetch_atlas_harvard_oxford(ATLASES[atlas_key][0])
    img = atlas.maps if hasattr(atlas.maps, "affine") else nib.load(atlas.maps)
    # nilearn >= 0.13 returns an in-memory image; the on-disk path is .filename
    src_path = getattr(atlas, "filename", None) or img.get_filename()
    return img, list(atlas.labels), src_path


def resample_labels(src: nib.Nifti1Image, ref: nib.Nifti1Image) -> nib.Nifti1Image:
    from nilearn.image import resample_to_img

    out = resample_to_img(
        src, ref, interpolation="nearest", force_resample=True, copy_header=True
    )
    data = np.rint(np.asarray(out.dataobj)).astype(np.uint8)
    img = nib.Nifti1Image(data, ref.affine)
    img.set_data_dtype(np.uint8)
    return img


def write_tsv(path: Path, labels: list[str]) -> None:
    # BIDS dseg.tsv: one row per non-background label; index is the voxel value
    with open(path, "w") as fh:
        fh.write("index\tname\n")
        for i, name in enumerate(labels):
            if i == 0:
                continue
            fh.write(f"{i}\t{name}\n")


def read_tsv(path: Path) -> list[str]:
    rows = [l.rstrip("\n").split("\t") for l in open(path)][1:]
    n = max(int(r[0]) for r in rows) + 1
    labels = ["Background"] * n
    for idx, name in rows:
        labels[int(idx)] = name
    return labels


def sidecar(atlas_key: str, src: nib.Nifti1Image, ref: nib.Nifti1Image, src_path) -> dict:
    return {
        "Name": ATLASES[atlas_key][1],
        "Atlas": atlas_key,
        "Description": (
            f"FSL {ATLASES[atlas_key][0]} (maximum-probability labels, 25% threshold), "
            f"resampled once with nearest-neighbour interpolation from its native "
            f"{SOURCE_TEMPLATE} 2 mm grid onto the {TEMPLATE} res-2 grid shared by "
            "the Schaefer dseg files in this tree and by the fMRIPrep MNI outputs."
        ),
        "SourceTemplate": SOURCE_TEMPLATE,
        "SourceFile": str(src_path),
        "SourceGrid": {"shape": list(src.shape), "affine": src.affine.tolist()},
        "TargetTemplate": TEMPLATE,
        "TargetGrid": {"shape": list(ref.shape), "affine": ref.affine.tolist()},
        "Interpolation": "nearest",
        "CrossTemplateRegistration": "none",
        "Caveat": (
            f"Labels are defined in {SOURCE_TEMPLATE}; no {SOURCE_TEMPLATE}->{TEMPLATE} "
            "warp was applied. The two templates differ by roughly a voxel at 2 mm. "
            "This reproduces what nilearn resample_to_img did implicitly at every "
            "call site before the atlas was staged."
        ),
        "GeneratedBy": {"Name": Path(__file__).name},
    }


def description_json() -> dict:
    return {
        "Name": "Harvard-Oxford cortical and subcortical structural atlases",
        "Description": (
            "Probabilistic atlases covering 48 cortical and 21 subcortical "
            "structural areas, derived from structural data and segmentations "
            "provided by the Harvard Center for Morphometric Analysis. Staged "
            "here as maximum-probability label volumes at the 25% threshold, "
            "resampled once from FSL's MNI152NLin6Asym copy onto the "
            "MNI152NLin2009cAsym res-2 grid without cross-template registration; "
            "see each dseg's JSON sidecar."
        ),
        "License": "FSL atlas licence (non-commercial); see FSL distribution",
        "ReferencesAndLinks": [
            "https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/Atlases",
            "https://doi.org/10.1016/j.schres.2005.11.020",
            "https://doi.org/10.1016/j.biopsych.2006.06.027",
            "https://doi.org/10.1016/j.neuroimage.2007.03.048",
        ],
        "Species": "homo sapiens",
        "DerivedFrom": "T1w structural segmentations (37 subjects)",
        "LevelType": "group",
        "SourceTemplate": "MNI152NLin6Asym",
        "SourceDistribution": "FSL, via nilearn.datasets.fetch_atlas_harvard_oxford",
    }


def assert_grid(img: nib.Nifti1Image, shape, affine, what: str) -> None:
    if tuple(img.shape) != tuple(shape) or not np.allclose(img.affine, affine):
        raise RuntimeError(
            f"grid mismatch against {what}: staged {img.shape} "
            f"{img.affine[:3, 3]} vs {tuple(shape)} {np.asarray(affine)[:3, 3]}"
        )


def roi_table(atlases_dir: Path, ref: nib.Nifti1Image, src_imgs: dict) -> list[tuple]:
    """Six benchmark ROIs + VTC: voxels on the FSL grid, staged, and per-call."""
    spec = importlib.util.spec_from_file_location("ps_shared", _PS_SHARED)
    ps = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = ps
    spec.loader.exec_module(ps)

    fetched, fetched_affine = ps.load_bilateral_roi_masks(source="fetch")
    percall = ps.resample_masks_to_bold(fetched, fetched_affine, ref)
    staged, staged_affine = ps.load_bilateral_roi_masks(
        source="staged", atlases_dir=atlases_dir
    )
    assert np.allclose(staged_affine, ref.affine)

    rows = []
    for roi in ps.PATTERN_ROI_NAMES:
        rows.append((roi, int(fetched[roi].sum()), int(staged[roi].sum()),
                     int(percall[roi].sum())))

    vtc_vals = list(VTC_CORTICAL_LABELS)
    src_cort = np.asarray(src_imgs["HOCPA"].dataobj).astype(int)
    stg_cort = np.asarray(
        nib.load(atlases_dir / ANAT / staged_name("HOCPA", ".nii.gz")).dataobj
    ).astype(int)
    vtc_src = np.isin(src_cort, vtc_vals)
    vtc_percall = ps.resample_masks_to_bold({"VTC": vtc_src}, fetched_affine, ref)["VTC"]
    rows.append(("VTC", int(vtc_src.sum()), int(np.isin(stg_cort, vtc_vals).sum()),
                 int(vtc_percall.sum())))
    return rows


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--atlases-dir", type=Path, default=_DEFAULT_ATLASES_DIR)
    p.add_argument("--reference-npz", type=Path, default=None,
                   help="pattern-similarity cache npz whose grid_shape/affine must match")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)

    out_dir = args.atlases_dir / ANAT
    ref = reference_grid(args.atlases_dir)
    print(f"reference grid: shape {ref.shape}, origin {ref.affine[:3, 3]}")

    npz_grid = None
    if args.reference_npz:
        z = np.load(args.reference_npz, allow_pickle=True)
        npz_grid = (tuple(int(v) for v in z["grid_shape"]), np.asarray(z["affine"]))
        assert_grid(ref, *npz_grid, what=f"cache npz {args.reference_npz.name}")
        print(f"reference grid matches cache npz {args.reference_npz.name}")

    src_imgs = {}
    for key in ATLASES:
        nii = out_dir / staged_name(key, ".nii.gz")
        if nii.exists() and not args.overwrite:
            raise FileExistsError(f"{nii} exists; pass --overwrite to replace it")
        src, labels, src_path = fetch(key)
        src_imgs[key] = src
        print(f"{key}: source {src.shape} origin {src.affine[:3, 3]}, "
              f"{len(labels) - 1} labels, from {src_path}")
        out = resample_labels(src, ref)
        assert_grid(out, ref.shape, ref.affine, what="Schaefer reference dseg")
        if npz_grid:
            assert_grid(out, *npz_grid, what="cache npz")

        src_vals = set(np.unique(np.asarray(src.dataobj)).astype(int).tolist())
        out_vals = set(np.unique(np.asarray(out.dataobj)).astype(int).tolist())
        if not out_vals <= src_vals:
            raise RuntimeError(f"{key}: resampling invented labels {out_vals - src_vals}")
        lost = src_vals - out_vals
        if lost:
            print(f"  WARNING {key}: labels present in source but empty after resampling: {sorted(lost)}")

        nib.save(out, nii)
        write_tsv(out_dir / staged_name(key, ".tsv"), labels)
        if read_tsv(out_dir / staged_name(key, ".tsv")) != labels:
            raise RuntimeError(f"{key}: label TSV does not round-trip")
        with open(out_dir / staged_name(key, ".json"), "w") as fh:
            json.dump(sidecar(key, src, ref, src_path), fh, indent=2)
        print(f"  wrote {nii.name} (+ .tsv, .json)")

    with open(args.atlases_dir / "atlas-HarvardOxford_description.json", "w") as fh:
        json.dump(description_json(), fh, indent=2)

    rows = roi_table(args.atlases_dir, ref, src_imgs)
    print("\nvoxels per ROI  (source = FSL MNI152NLin6Asym 2 mm grid; staged = this tree; "
          "per-call = old fetch+resample onto a res-2 BOLD grid)")
    print(f"{'ROI':<12}{'source':>8}{'staged':>8}{'per-call':>10}{'staged-percall':>16}")
    for roi, a, b, c in rows:
        print(f"{roi:<12}{a:>8}{b:>8}{c:>10}{b - c:>16}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
