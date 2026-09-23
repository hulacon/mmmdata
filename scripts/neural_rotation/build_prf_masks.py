#!/usr/bin/env python3
"""Rung (iv) of the neural-rotation ladder: pRF-defined voxel populations.

From the analyzePRF pooled fits (``derivatives/prf/sub-##/``, space-T1w,
R² in percent, unthresholded) three masks per R² threshold t:

  pos        R²_pos > t
  negstrict  R²_neg > t  and  (R²_neg - R²_pos) >= --klink-margin (percentage
             points; the Klink relative criterion that turns two independent
             fits into a partition -- the negative fit is otherwise nested
             inside the positive one at any absolute cut)
  union      pos  or  R²_neg > t

each restricted to fitted centres inside the stimulus (eccentricity <=
``StimulusRadiusDeg`` from the fit's sidecar): under this backend off-screen
centres are unbounded extrapolations whose parameters are meaningless.

Written in T1w, then warped to the GLMsingle grid (MNI152NLin2009cAsym
res-2) with the subject's fMRIPrep transform through antsApplyTransforms
(nearest neighbour for masks). The pRF maps the level-3 predicted map needs
-- R², eccentricity and angle (as cos/sin so the wrap survives
interpolation) for both polarities -- are warped alongside (linear).

Output tree (``<output_dir>/functional_rois/sub-##/``):

  space-T1w/sub-##_task-prf_desc-{pos,negstrict,union}_thr-<t>_mask.nii.gz
  space-MNI152NLin2009cAsym_res-2/
      sub-##_task-prf_desc-{pos,negstrict,union}_thr-<t>_mask.nii.gz
      sub-##_task-prf_desc-{R2,eccentricity,anglecos,anglesin}_{prf,negprf}.nii.gz
      sub-##_task-prf_desc-fitted_mask.nii.gz    where the T1w fit had a value
  sub-##_task-prf_masks.json                      counts, nesting, parameters
  sub-##_task-prf_masks.tsv                       one row per threshold x mask x space

Thresholds are given in percent and spelled in filenames with ``p`` for the
decimal point (2.5 -> ``thr-2p5``).

Requires ``antsApplyTransforms`` on PATH (Talapas: ``module load ants``)
unless ``--no-warp``.

Usage:
    python build_prf_masks.py --subject sub-## [--threshold-pct 2.5 10] [--dry-run]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).resolve().parent.parent


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


tb = _load_module("glmsingle_tb", SCRIPTS / "glmsingle_tb.py")

TREE = "functional_rois"
MNI_DIR = f"space-{tb.SPACE}"
MASKS = ("pos", "negstrict", "union")
POLARITIES = ("prf", "negprf")
WARPED_MAPS = ("R2", "eccentricity", "anglecos", "anglesin")
KLINK_MARGIN_DEFAULT = 5.0


def thr_token(t: float) -> str:
    s = f"{t:g}".replace(".", "p")
    return f"thr-{s}"


def prf_path(prf_dir: Path, subject: str, desc: str, pol: str) -> Path:
    return prf_dir / subject / f"{subject}_task-prf_space-T1w_desc-{desc}_{pol}.nii.gz"


def load_maps(prf_dir: Path, subject: str) -> tuple[dict, nib.Nifti1Image, dict]:
    """{(desc, pol): ndarray} for R2/eccentricity/angle, the reference image, sidecar."""
    side = prf_dir / subject / f"{subject}_task-prf_space-T1w_prf.json"
    if not side.exists():
        sys.exit(f"ERROR: pRF sidecar missing: {side}")
    meta = json.load(open(side))
    maps, ref = {}, None
    for pol in POLARITIES:
        for desc in ("R2", "eccentricity", "angle"):
            p = prf_path(prf_dir, subject, desc, pol)
            if not p.exists():
                sys.exit(f"ERROR: pRF map missing: {p}")
            img = nib.load(str(p))
            if ref is None:
                ref = img
            elif img.shape != ref.shape or not np.allclose(img.affine, ref.affine):
                sys.exit(f"ERROR: {p.name} grid differs from {ref.get_filename()}")
            maps[(desc, pol)] = np.asarray(img.dataobj, dtype=np.float64)
    return maps, ref, meta


def cut_masks(maps: dict, thresholds: list, radius: float, margin: float) -> dict:
    """{(t, name): bool ndarray}; NaN (outside the fit mask) never passes."""
    with np.errstate(invalid="ignore"):
        r2p, r2n = maps[("R2", "prf")], maps[("R2", "negprf")]
        onscreen_p = maps[("eccentricity", "prf")] <= radius
        onscreen_n = maps[("eccentricity", "negprf")] <= radius
        out = {}
        for t in thresholds:
            pos = (r2p > t) & onscreen_p
            neg = (r2n > t) & onscreen_n
            out[(t, "pos")] = pos
            out[(t, "negstrict")] = neg & ((r2n - r2p) >= margin)
            out[(t, "union")] = pos | neg
            out[(t, "_neg_any")] = neg          # for the nesting fraction only
    return out


def nesting_rows(masks: dict, thresholds: list, subject: str) -> list:
    rows = []
    for t in thresholds:
        pos, neg, strict = masks[(t, "pos")], masks[(t, "_neg_any")], masks[(t, "negstrict")]
        n_neg = int(neg.sum())
        rows.append({"subject": subject, "threshold_pct": t,
                     "n_pos": int(pos.sum()), "n_neg_any": n_neg,
                     "n_neg_strict": int(strict.sum()), "n_union": int(masks[(t, "union")].sum()),
                     "n_neg_in_pos": int((neg & pos).sum()),
                     "nesting_frac": float((neg & pos).sum() / n_neg) if n_neg else float("nan")})
    return rows


def find_xfm(fmriprep_dir: Path, subject: str) -> Path:
    hits = sorted((fmriprep_dir / subject / "anat").glob(
        f"{subject}_*from-T1w_to-{tb.SPACE.replace('_res-2', '')}_mode-image_xfm.h5"))
    if not hits:
        sys.exit(f"ERROR: no T1w->MNI transform for {subject} under {fmriprep_dir}/{subject}/anat")
    return hits[0]


def find_ref(fmriprep_dir: Path, subject: str) -> Path:
    hits = sorted((fmriprep_dir / subject / "anat").glob(
        f"{subject}_*space-{tb.SPACE.replace('_res-2', '')}_res-2_desc-brain_mask.nii.gz"))
    if not hits:
        sys.exit(f"ERROR: no MNI res-2 brain mask for {subject} under {fmriprep_dir}/{subject}/anat")
    return hits[0]


def ants_apply(src: Path, ref: Path, xfm: Path, dst: Path, interp: str) -> None:
    cmd = ["antsApplyTransforms", "-d", "3", "-i", str(src), "-r", str(ref),
           "-t", str(xfm), "-n", interp, "-o", str(dst), "--float"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not dst.exists():
        sys.exit(f"ERROR: antsApplyTransforms failed for {src.name}:\n{' '.join(cmd)}\n{r.stderr}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject", required=True, help="e.g. sub-##")
    ap.add_argument("--threshold-pct", nargs="+", type=float, default=[2.5, 10.0],
                    help="R² cuts in percent (default: the charter's 2.5 + 10 sensitivity)")
    ap.add_argument("--klink-margin", type=float, default=KLINK_MARGIN_DEFAULT,
                    help="negstrict needs R²_neg - R²_pos >= this many percentage points")
    ap.add_argument("--radius-deg", type=float, default=None,
                    help="eccentricity ceiling (default: the sidecar's StimulusRadiusDeg)")
    ap.add_argument("--prf-dir", default=None, help="override <output_dir>/prf")
    ap.add_argument("--fmriprep-dir", default=None)
    ap.add_argument("--out-root", default=None, help=f"override <output_dir>/{TREE}")
    ap.add_argument("--no-warp", action="store_true", help="write T1w masks only")
    ap.add_argument("--dry-run", action="store_true", help="cut and count; write nothing")
    args = ap.parse_args()

    cfg = tb.load_config()
    output_dir = Path(cfg["output_dir"])
    prf_dir = Path(args.prf_dir) if args.prf_dir else output_dir / "prf"
    fmriprep_dir = Path(args.fmriprep_dir) if args.fmriprep_dir else output_dir / "fmriprep"
    out_root = Path(args.out_root) if args.out_root else output_dir / TREE
    sub_dir = out_root / args.subject
    thresholds = sorted(args.threshold_pct)

    print(f"=== pRF masks (rung iv): {args.subject}; thresholds {thresholds} % ===")
    maps, ref_t1w, meta = load_maps(prf_dir, args.subject)
    if meta.get("Thresholded") or meta.get("RefineThresholdR2Pct"):
        sys.exit(f"ERROR: {args.subject} pRF fit is gated (RefineThresholdR2Pct="
                 f"{meta.get('RefineThresholdR2Pct')}); rung (iv) needs the unthresholded fit")
    radius = args.radius_deg if args.radius_deg is not None else meta.get("StimulusRadiusDeg")
    if radius is None:
        sys.exit("ERROR: no StimulusRadiusDeg in the pRF sidecar; pass --radius-deg")
    fitted = np.isfinite(maps[("R2", "prf")])
    print(f"T1w grid {ref_t1w.shape}, {int(fitted.sum())} fitted voxels; radius {radius} deg; "
          f"Klink margin {args.klink_margin} pp")

    masks = cut_masks(maps, thresholds, radius, args.klink_margin)
    nesting = nesting_rows(masks, thresholds, args.subject)
    print(pd.DataFrame(nesting).to_string(index=False))
    if args.dry_run:
        print("DRY RUN — nothing written.")
        return

    if not args.no_warp and shutil.which("antsApplyTransforms") is None:
        sys.exit("ERROR: antsApplyTransforms not on PATH (Talapas: `module load ants`), "
                 "or pass --no-warp")

    t1w_dir, mni_dir = sub_dir / "space-T1w", sub_dir / MNI_DIR
    t1w_dir.mkdir(parents=True, exist_ok=True)
    rows, written = [], {}
    for t in thresholds:
        for name in MASKS:
            m = masks[(t, name)]
            p = t1w_dir / f"{args.subject}_task-prf_desc-{name}_{thr_token(t)}_mask.nii.gz"
            nib.save(nib.Nifti1Image(m.astype(np.uint8), ref_t1w.affine), p)
            written[(t, name)] = p
            rows.append({"subject": args.subject, "threshold_pct": t, "mask": name,
                         "space": "T1w", "n_vox": int(m.sum())})
    p_fitted = t1w_dir / f"{args.subject}_task-prf_desc-fitted_mask.nii.gz"
    nib.save(nib.Nifti1Image(fitted.astype(np.uint8), ref_t1w.affine), p_fitted)

    if not args.no_warp:
        mni_dir.mkdir(parents=True, exist_ok=True)
        ref_mni, xfm = find_ref(fmriprep_dir, args.subject), find_xfm(fmriprep_dir, args.subject)
        for (t, name), src in written.items():
            dst = mni_dir / src.name
            ants_apply(src, ref_mni, xfm, dst, "NearestNeighbor")
            rows.append({"subject": args.subject, "threshold_pct": t, "mask": name,
                         "space": tb.SPACE, "n_vox": int((np.asarray(nib.load(str(dst)).dataobj) > 0.5).sum())})
        ants_apply(p_fitted, ref_mni, xfm, mni_dir / p_fitted.name, "NearestNeighbor")
        # the parameter maps: NaN -> 0 before warping (ANTs propagates NaN),
        # angle as cos/sin so the 0/360 wrap does not average to 180
        for pol in POLARITIES:
            comp = {"R2": maps[("R2", pol)], "eccentricity": maps[("eccentricity", pol)],
                    "anglecos": np.cos(np.deg2rad(maps[("angle", pol)])),
                    "anglesin": np.sin(np.deg2rad(maps[("angle", pol)]))}
            for desc, arr in comp.items():
                tmp = t1w_dir / f"{args.subject}_task-prf_desc-{desc}_{pol}.nii.gz"
                nib.save(nib.Nifti1Image(np.nan_to_num(arr).astype(np.float32), ref_t1w.affine), tmp)
                ants_apply(tmp, ref_mni, xfm, mni_dir / tmp.name, "Linear")
                tmp.unlink()
        print(f"warped {len(written) + 1} masks and {len(POLARITIES) * len(WARPED_MAPS)} maps "
              f"to {mni_dir}")

    table = pd.DataFrame(rows)
    table.to_csv(sub_dir / f"{args.subject}_task-prf_masks.tsv", sep="\t", index=False)
    with open(sub_dir / f"{args.subject}_task-prf_masks.json", "w") as f:
        json.dump({
            "Subject": args.subject, "Source": str(prf_dir / args.subject),
            "SourceBackend": meta.get("Backend"), "ThresholdsPct": thresholds,
            "KlinkMarginPct": args.klink_margin, "StimulusRadiusDeg": radius,
            "StimulusRadiusNote": meta.get("StimulusRadiusNote"),
            "Definitions": {"pos": "R2_pos > t, eccentricity_pos <= radius",
                            "negstrict": "R2_neg > t, eccentricity_neg <= radius, "
                                         "R2_neg - R2_pos >= KlinkMarginPct",
                            "union": "pos or (R2_neg > t, eccentricity_neg <= radius)"},
            "Nesting": nesting,
            "Warp": None if args.no_warp else {
                "Tool": "antsApplyTransforms", "Transform": str(find_xfm(fmriprep_dir, args.subject)),
                "Reference": str(find_ref(fmriprep_dir, args.subject)),
                "MaskInterpolation": "NearestNeighbor", "MapInterpolation": "Linear",
                "AngleNote": "warped as cos/sin; recover angle = atan2(sin, cos)"},
        }, f, indent=2)
    print(table.pivot_table(index=["threshold_pct", "mask"], columns="space", values="n_vox").to_string())
    print(f"wrote {sub_dir}/{args.subject}_task-prf_masks.{{tsv,json}}")


if __name__ == "__main__":
    main()
