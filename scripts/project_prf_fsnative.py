#!/usr/bin/env python3
"""
project_prf_fsnative.py — project fitted pRF parameter volumes to fsnative.

Stage 1 of the projection -> test-retest leg (workbench prf-retinotopy,
HANDOFF 2026-08-27). The fits live in the native functional volume
(`fit_prf.py`, one session = one fit unit); this script samples those
parameter volumes at the subject's FreeSurfer surface vertices, NSD-style
(`cvnlab/nsddatapaper` `analysis_prf_maps.m`: fit in the volume, project the
parameters, never the timeseries).

Per session the chain, applied to VERTEX COORDINATES (so images are never
resampled):

    fsnative surface (tkr-RAS)
      -> fsnative scanner-RAS      via orig.mgz vox2ras / vox2ras-tkr
      -> T1w RAS                   via fMRIPrep from-T1w_to-fsnative ITK xfm
      -> native func (boldref) RAS via fMRIPrep from-boldref_to-T1w ITK xfm
      -> func voxel ijk            via the parameter volume's affine

The two ITK affines are used in the direction they map POINTS: an fMRIPrep
`from-A_to-B` transform resamples images A->B, which is exactly the map of
coordinates B->A. nitransforms handles the ITK LPS convention; do not replace
it with hand-rolled affine algebra. A built-in check verifies the chain
end-to-end every run: mid-depth vertices sampled against the run's func
brain mask must land inside it almost always — an inverted or misordered
chain sends vertices into space and fails loudly.

Sampling: each parameter volume is sampled at three cortical depths
(white->pial fractions 0.25 / 0.5 / 0.75, NSD-style depth average) with
trilinear interpolation made NaN-aware by normalised convolution (sample
value*valid and valid separately, divide; vertices whose interpolation
weight is mostly NaN voxels come out NaN rather than biased toward 0).

POLAR ANGLE never touches interpolation directly: the angle volume is split
into cos/sin volumes, each projected like any scalar (including the depth
average), and recombined with atan2 at the very end. Interpolating degrees
across the 0/360 wrap manufactures fake reversals, and reversals ARE the
V1/V2/V3 borders.

Outputs, one GIFTI shape file per parameter x hemisphere x polarity
(format DECIDED here: .shape.gii — viewable in freeview/NiiVue, carries the
anatomical-structure tag, and stays paired with the fsnative surfaces):

    derivatives/prf/sub-XX/ses-0Y/
      sub-XX_ses-0Y_task-prf_space-fsnative_hemi-{L,R}_desc-<param>_{prf,negprf}.shape.gii
      sub-XX_ses-0Y_task-prf_space-fsnative_{prf,negprf}.json

Usage:
    python project_prf_fsnative.py --subject 03 --session 02
    python project_prf_fsnative.py --all            # every fitted session x polarity
    python project_prf_fsnative.py --subject 03 --session 02 --polarity negprf
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_REPO_ROOT / "src" / "python"))

try:
    from core.config import load_config
    _config = load_config(config_dir=_REPO_ROOT / "config")
    DERIV_ROOT = Path(_config["paths"]["output_dir"])
except Exception:  # pragma: no cover
    DERIV_ROOT = Path("/gpfs/projects/hulacon/shared/mmmdata/derivatives")

PARAMS = ("R2", "angle", "eccentricity", "size", "sigma", "exponent", "gain")
DEPTHS = (0.25, 0.5, 0.75)      # white->pial fractions, averaged (NSD-style)
WEIGHT_FLOOR = 0.5              # min valid interpolation weight, else NaN
MASK_HIT_FLOOR = 0.80           # end-to-end chain check (see module docstring)
HEMIS = {"L": "lh", "R": "rh"}
STRUCTURE = {"L": "CortexLeft", "R": "CortexRight"}


# ---------------------------------------------------------------------------
# path resolution
# ---------------------------------------------------------------------------

def fsnative_xfm(subject):
    """fMRIPrep's from-T1w_to-fsnative ITK affine.

    Longitudinal subjects carry it under sub-XX/anat/; single-anat subjects
    under sub-XX/ses-*/anat/ with a ses- entity in the name. Accept either,
    refuse ambiguity.
    """
    fp = DERIV_ROOT / "fmriprep" / f"sub-{subject}"
    hits = sorted(fp.glob("anat/*from-T1w_to-fsnative_mode-image_xfm.txt"))
    hits += sorted(fp.glob("ses-*/anat/*from-T1w_to-fsnative_mode-image_xfm.txt"))
    if not hits:
        sys.exit(f"ERROR: no from-T1w_to-fsnative xfm under {fp}/(ses-*/)anat/")
    if len(hits) > 1:
        sys.exit(f"ERROR: {len(hits)} from-T1w_to-fsnative xfms for sub-{subject}: "
                 f"{[str(h) for h in hits]} — pick one explicitly in the code")
    return hits[0]


def coreg_xfm(subject, session):
    """Per-session boldref->T1w coreg affine (one serves all runs: within a
    session the three pRF runs share the native grid exactly — measured, see
    fit_prf.py)."""
    p = (DERIV_ROOT / "fmriprep" / f"sub-{subject}" / f"ses-{session}" / "func"
         / f"sub-{subject}_ses-{session}_task-prf_run-01"
           "_from-boldref_to-T1w_mode-image_desc-coreg_xfm.txt")
    if not p.exists():
        sys.exit(f"ERROR: no coreg xfm at {p}")
    return p


def func_brain_mask(subject, session):
    p = (DERIV_ROOT / "fmriprep" / f"sub-{subject}" / f"ses-{session}" / "func"
         / f"sub-{subject}_ses-{session}_task-prf_run-01_desc-brain_mask.nii.gz")
    if not p.exists():
        sys.exit(f"ERROR: no func brain mask at {p}")
    return p


def freesurfer_dir(subject):
    d = DERIV_ROOT / "fmriprep" / "sourcedata" / "freesurfer" / f"sub-{subject}"
    if not d.is_dir():
        sys.exit(f"ERROR: no FreeSurfer subject dir at {d}")
    return d


def param_volume(subject, session, param, polarity):
    p = (DERIV_ROOT / "prf" / f"sub-{subject}" / f"ses-{session}"
         / f"sub-{subject}_ses-{session}_task-prf_space-func"
           f"_desc-{param}_{polarity}.nii.gz")
    if not p.exists():
        sys.exit(f"ERROR: no parameter volume at {p}")
    return p


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------

def tkr_to_scanner(fs_dir):
    """4x4 mapping FreeSurfer surface (tkr) RAS -> fsnative scanner RAS,
    from orig.mgz's two vox2ras matrices."""
    import nibabel as nib
    orig = nib.load(str(fs_dir / "mri" / "orig.mgz"))
    return orig.header.get_vox2ras() @ np.linalg.inv(orig.header.get_vox2ras_tkr())


def depth_coords(fs_dir, fs_hemi, depths=DEPTHS):
    """Vertex coordinates at each cortical depth, tkr RAS, (n_depth, n_vert, 3).

    white and pial share vertex correspondence in FreeSurfer, so the depth
    point is a plain linear blend per vertex.
    """
    import nibabel as nib
    white, _ = nib.freesurfer.read_geometry(str(fs_dir / "surf" / f"{fs_hemi}.white"))
    pial, _ = nib.freesurfer.read_geometry(str(fs_dir / "surf" / f"{fs_hemi}.pial"))
    if white.shape != pial.shape:
        sys.exit(f"ERROR: {fs_hemi} white/pial vertex counts differ "
                 f"({white.shape[0]} vs {pial.shape[0]})")
    return np.stack([white + f * (pial - white) for f in depths]), white.shape[0]


def apply_affine(mat, pts):
    return pts @ mat[:3, :3].T + mat[:3, 3]


def vertex_voxel_coords(subject, session, fs_hemi):
    """(n_depth, n_vert, 3) voxel ijk in the session's native func grid."""
    import nibabel as nib
    import nitransforms as nt

    fs_dir = freesurfer_dir(subject)
    coords, n_vert = depth_coords(fs_dir, fs_hemi)
    flat = coords.reshape(-1, 3)

    flat = apply_affine(tkr_to_scanner(fs_dir), flat)          # tkr -> fsnative RAS
    # from-T1w_to-fsnative maps points fsnative -> T1w; from-boldref_to-T1w
    # maps points T1w -> boldref (the image-resampling direction, reversed).
    flat = nt.linear.load(str(fsnative_xfm(subject)), fmt="itk").map(flat)
    flat = nt.linear.load(str(coreg_xfm(subject, session)), fmt="itk").map(flat)

    ref = nib.load(str(param_volume(subject, session, "R2", "prf")))
    ijk = apply_affine(np.linalg.inv(ref.affine), flat)
    return ijk.reshape(len(DEPTHS), n_vert, 3), ref


# ---------------------------------------------------------------------------
# sampling
# ---------------------------------------------------------------------------

def sample_volume(vol, ijk_depths):
    """NaN-aware trilinear sampling with depth averaging.

    vol         : 3D array (may contain NaN)
    ijk_depths  : (n_depth, n_vert, 3)
    returns     : (n_vert,) float32, NaN where valid weight < WEIGHT_FLOOR
    """
    from scipy.ndimage import map_coordinates
    valid = np.isfinite(vol)
    filled = np.where(valid, vol, 0.0).astype(np.float64)
    w = valid.astype(np.float64)

    num = np.zeros(ijk_depths.shape[1])
    den = np.zeros(ijk_depths.shape[1])
    for d in range(ijk_depths.shape[0]):
        c = ijk_depths[d].T  # (3, n_vert)
        num += map_coordinates(filled, c, order=1, mode="constant", cval=0.0)
        den += map_coordinates(w, c, order=1, mode="constant", cval=0.0)
    out = np.full(ijk_depths.shape[1], np.nan, dtype=np.float32)
    good = den >= WEIGHT_FLOOR * ijk_depths.shape[0]
    out[good] = (num[good] / den[good]).astype(np.float32)
    return out


def chain_check(subject, session, ijk_depths):
    """End-to-end transform verification: mid-depth vertices vs the func
    brain mask. See module docstring."""
    import nibabel as nib
    from scipy.ndimage import map_coordinates
    mask = np.asarray(nib.load(str(func_brain_mask(subject, session))).dataobj,
                      dtype=np.float64)
    mid = ijk_depths[len(DEPTHS) // 2].T
    hit = map_coordinates(mask, mid, order=1, mode="constant", cval=0.0)
    frac = float((hit > 0.5).mean())
    if frac < MASK_HIT_FLOOR:
        sys.exit(
            f"ERROR: only {100 * frac:.1f}% of mid-depth vertices land in the "
            f"func brain mask (floor {100 * MASK_HIT_FLOOR:.0f}%).\n"
            "       The transform chain is inverted or misordered — re-read the\n"
            "       direction note in the module docstring before touching the\n"
            "       affine algebra.")
    return frac


# ---------------------------------------------------------------------------
# outputs
# ---------------------------------------------------------------------------

def write_shape_gii(path, values, hemi):
    import nibabel as nib
    da = nib.gifti.GiftiDataArray(
        np.asarray(values, dtype=np.float32),
        intent="NIFTI_INTENT_SHAPE",
        datatype="NIFTI_TYPE_FLOAT32")
    img = nib.gifti.GiftiImage(darrays=[da])
    img.meta["AnatomicalStructurePrimary"] = STRUCTURE[hemi]
    nib.save(img, str(path))


def project_session(subject, session, polarity):
    out_dir = DERIV_ROOT / "prf" / f"sub-{subject}" / f"ses-{session}"
    base = f"sub-{subject}_ses-{session}_task-prf_space-fsnative"
    written = []
    stats = {}

    for hemi, fs_hemi in HEMIS.items():
        ijk, _ = vertex_voxel_coords(subject, session, fs_hemi)
        frac = chain_check(subject, session, ijk)
        print(f"  hemi-{hemi}: {ijk.shape[1]} vertices, "
              f"{100 * frac:.1f}% in func brain mask")

        import nibabel as nib
        vols = {p: np.asarray(
                    nib.load(str(param_volume(subject, session, p, polarity))).dataobj,
                    dtype=np.float64)
                for p in PARAMS}

        surf = {}
        for p in PARAMS:
            if p == "angle":
                rad = np.radians(vols[p])
                cos_s = sample_volume(np.cos(rad), ijk)
                sin_s = sample_volume(np.sin(rad), ijk)
                surf[p] = (np.degrees(np.arctan2(sin_s, cos_s)) % 360.0
                           ).astype(np.float32)
                surf[p][~np.isfinite(cos_s)] = np.nan
            else:
                surf[p] = sample_volume(vols[p], ijk)

        for p in PARAMS:
            path = out_dir / f"{base}_hemi-{hemi}_desc-{p}_{polarity}.shape.gii"
            write_shape_gii(path, surf[p], hemi)
            written.append(path.name)

        n_good = int(np.nansum(surf["R2"] > 10.0))
        stats[f"hemi-{hemi}"] = {
            "vertices": int(ijk.shape[1]),
            "mask_hit_fraction": round(frac, 4),
            "vertices_R2_gt_10": n_good,
        }
        print(f"           vertices with R2 > 10%: {n_good}")

    vol_sidecar = json.loads(
        (out_dir / f"sub-{subject}_ses-{session}_task-prf_space-func_{polarity}.json"
         ).read_text())
    meta = {
        "Description": ("pRF parameters projected from the native functional "
                        "volume to the subject's fsnative surface."),
        "Subject": f"sub-{subject}", "Session": f"ses-{session}",
        "Space": "fsnative (fMRIPrep FreeSurfer surfaces)",
        "SourceSpace": "func (native boldref; fit_prf.py outputs)",
        "Method": ("Vertex coordinates mapped fsnative->T1w->boldref via the "
                   "fMRIPrep ITK affines (nitransforms); parameter volumes "
                   "sampled trilinearly at cortical depths "
                   f"{list(DEPTHS)} (white->pial fractions) and depth-averaged; "
                   "NaN-aware normalised interpolation "
                   f"(valid-weight floor {WEIGHT_FLOOR})."),
        "AngleHandling": ("Projected as cos/sin separately (interpolation AND "
                          "depth average), recombined with atan2 — degrees "
                          "never interpolated across the 0/360 wrap."),
        "AngleConvention": vol_sidecar.get("AngleConvention"),
        "HRFKind": vol_sidecar.get("HRFKind"),
        "FieldOfViewDeg": vol_sidecar.get("FieldOfViewDeg"),
        "ChainCheck": stats,
        "Provenance": "mmmdata/scripts/project_prf_fsnative.py; workbench prf-retinotopy",
        "Maps": written,
    }
    sidecar = out_dir / f"{base}_{polarity}.json"
    sidecar.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"  wrote {len(written)} shape.gii + {sidecar.name}")


def discover_sessions():
    """Every (subject, session) with a fitted prf sidecar on disk."""
    pairs = []
    for sc in sorted((DERIV_ROOT / "prf").glob(
            "sub-*/ses-*/sub-*_ses-*_task-prf_space-func_prf.json")):
        sub = sc.name.split("_")[0].split("-")[1]
        ses = sc.name.split("_")[1].split("-")[1]
        pairs.append((sub, ses))
    if not pairs:
        sys.exit(f"ERROR: no fitted pRF sessions under {DERIV_ROOT / 'prf'}")
    return pairs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject", help="bare label, e.g. 03")
    ap.add_argument("--session", help="bare label, e.g. 02")
    ap.add_argument("--polarity", choices=("prf", "negprf", "both"), default="both")
    ap.add_argument("--all", action="store_true",
                    help="project every fitted session found on disk")
    args = ap.parse_args()

    if args.all:
        pairs = discover_sessions()
    elif args.subject and args.session:
        pairs = [(args.subject, args.session)]
    else:
        ap.error("--subject and --session are required (or use --all)")

    polarities = ("prf", "negprf") if args.polarity == "both" else (args.polarity,)
    for sub, ses in pairs:
        for pol in polarities:
            print(f"sub-{sub} ses-{ses} {pol}")
            project_session(sub, ses, pol)
    return 0


if __name__ == "__main__":
    sys.exit(main())
