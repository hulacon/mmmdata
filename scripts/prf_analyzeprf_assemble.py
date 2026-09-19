#!/usr/bin/env python3
"""
prf_analyzeprf_assemble.py — analyzePRF chunk results -> pRF parameter volumes.

Step 3 of the analyzePRF-backed pRF fit. Reads the chunk files written by
prf_analyzeprf_fit.m, checks they cover the exported mask exactly once,
converts analyzePRF's pixel-unit outputs to the visual-field degrees
fit_prf.py releases, and writes the same seven maps under the same names --
so project_prf_fsnative.py, the viewer bundles and every downstream reader
run unchanged:

    sub-XX_task-prf_space-T1w_desc-{R2,angle,eccentricity,size,sigma,exponent,gain}_{prf,negprf}.nii.gz
    sub-XX_task-prf_space-T1w_{prf,negprf}.json

Unit conventions (checked against fit_prf.to_visual, and measured to agree
with the Python fit to 0.3 deg angle / 0.06 deg ecc on 2026-09-18):
    angle        analyzePRF `ang` as is: degrees CCW from the right horizontal
                 meridian, 0-360, row 0 = top of screen
    eccentricity `ecc` px * FOV/res
    size         `rfsize` px * FOV/res  (= sigma/sqrt(n), NSD prf_size)
    sigma        rfsize * sqrt(expt) * FOV/res  (the raw Gaussian width)
    exponent     `expt`
    gain         `gain`, in the data's units (PSC here; NSD's were raw)
    R2           `R2`, percent, every voxel, unthresholded

The existing product is never overwritten in place: pass --archive-root and
every file of that suffix already in the output directory is MOVED there
first (nothing is deleted); without it the script refuses when files exist.

Usage:
    python prf_analyzeprf_assemble.py --export-dir WORK/sub-03 --chunk-dir WORK/sub-03/prf \
        --subject 03 --polarity prf --out-root derivatives/prf \
        --archive-root derivatives/prf_pythoncss \
        --analyzeprf-commit <sha> --knkutils-commit <sha>
"""

import argparse
import json
import re
import shutil
import sys
from datetime import date
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import fit_prf  # noqa: E402
from prf_analyzeprf_export import export_paths, export_stem  # noqa: E402

MAP_NAMES = ("R2", "angle", "eccentricity", "size", "sigma", "exponent", "gain")


def load_chunks(chunk_dir):
    """Every chunk-NNNN.mat in order, as dicts of 1-D arrays."""
    import scipy.io as sio

    files = sorted(Path(chunk_dir).glob("chunk-[0-9][0-9][0-9][0-9].mat"))
    if not files:
        sys.exit(f"ERROR: no chunk-NNNN.mat files in {chunk_dir}\n"
                 "       Run prf_analyzeprf_fit.m first (fit_prf_analyzeprf.sbatch).")
    chunks = []
    for f in files:
        m = sio.loadmat(str(f), squeeze_me=False)
        chunks.append({k: np.asarray(m[k]).reshape(-1) if k != "params" else np.asarray(m[k])
                       for k in ("vxs", "ang", "ecc", "expt", "rfsize", "R2", "gain",
                                 "numiters", "params")})
    return chunks


def chunks_to_results(chunks, n_vox, fov_deg, res):
    """Stitch chunks into fit_prf.py's result vectors (n_vox each), in degrees.

    Coverage is asserted: each 1-based voxel index appears exactly once. A gap
    means a chunk is missing (the job stopped early -- requeue it, the fit
    resumes) and is an error, never a NaN row.
    """
    seen = np.zeros(n_vox, dtype=np.int64)
    out = {k: np.full(n_vox, np.nan, dtype=np.float64) for k in
           ("ang", "ecc", "expt", "rfsize", "R2", "gain", "numiters")}
    for c in chunks:
        idx = c["vxs"].astype(np.int64) - 1
        if idx.min() < 0 or idx.max() >= n_vox:
            sys.exit(f"ERROR: chunk voxel indices {idx.min() + 1}..{idx.max() + 1} "
                     f"fall outside the exported mask of {n_vox} voxels")
        seen[idx] += 1
        for k in out:
            out[k][idx] = c[k].astype(np.float64)
    missing = int((seen == 0).sum())
    dup = int((seen > 1).sum())
    if missing or dup:
        first = int(np.argmax(seen == 0)) + 1 if missing else None
        sys.exit(f"ERROR: chunks cover the mask incompletely: {missing} voxels "
                 f"never fitted (first: index {first}), {dup} fitted twice.\n"
                 "       Requeue the fit job; existing chunks are kept and the "
                 "missing ones are computed.")
    deg_per_px = fov_deg / res
    return {
        "R2": out["R2"].astype(np.float32),
        "angle": out["ang"] % 360.0,
        "eccentricity": out["ecc"] * deg_per_px,
        "size": out["rfsize"] * deg_per_px,
        "sigma": out["rfsize"] * np.sqrt(out["expt"]) * deg_per_px,
        "exponent": out["expt"],
        "gain": out["gain"],
    }, out["numiters"]


def archive_existing(out_dir, base, suffix, archive_root, subject):
    """Move every existing file of this product (volume AND fsnative) aside."""
    pattern = re.compile(rf"^{re.escape(base.split('_space-')[0])}_.*_{suffix}\.(nii\.gz|json|shape\.gii)$")
    existing = sorted(p for p in Path(out_dir).glob("*") if p.is_file() and pattern.match(p.name))
    if not existing:
        return []
    if archive_root is None:
        sys.exit(f"ERROR: {len(existing)} files of suffix `{suffix}` already exist in "
                 f"{out_dir} (e.g. {existing[0].name}).\n"
                 "       Nothing is deleted: pass --archive-root DIR to move them "
                 "there first, or --out-root elsewhere for a pilot.")
    dest = Path(archive_root) / f"sub-{subject}"
    dest.mkdir(parents=True, exist_ok=True)
    dd = Path(archive_root) / "dataset_description.json"
    if not dd.exists():
        dd.write_text(json.dumps({
            "Name": "MMMData pRF fits, Python CSS backend (archived)",
            "BIDSVersion": "1.9.0", "DatasetType": "derivative",
            "GeneratedBy": [{"Name": "fit_prf.py",
                             "Description": "Python port of analyzePRF's CSS model with a "
                                            "5% grid-R2 refinement gate; superseded by the "
                                            "analyzePRF backend (prf_analyzeprf_*), archived "
                                            f"{date.today().isoformat()} per the "
                                            "nothing-is-deleted rule."}],
        }, indent=2) + "\n")
    moved = []
    for p in existing:
        target = dest / p.name
        if target.exists():
            sys.exit(f"ERROR: archive target already exists: {target}")
        shutil.move(str(p), str(target))
        moved.append(p.name)
    return moved


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export-dir", required=True, help="prf_analyzeprf_export.py's --out-dir")
    ap.add_argument("--chunk-dir", required=True, help="prf_analyzeprf_fit.m's outdir")
    ap.add_argument("--subject", required=True)
    ap.add_argument("--space", default="T1w")
    ap.add_argument("--polarity", choices=("prf", "negprf"), required=True)
    ap.add_argument("--out-root", required=True,
                    help="derivatives/prf for the product; anywhere else for a pilot")
    ap.add_argument("--archive-root", default=None,
                    help="where existing files of this suffix are MOVED before writing")
    ap.add_argument("--analyzeprf-commit", default="unrecorded")
    ap.add_argument("--knkutils-commit", default="unrecorded")
    args = ap.parse_args()

    import nibabel as nib

    paths = export_paths(args.export_dir, args.subject, args.space)
    for p in paths.values():
        if not p.exists():
            sys.exit(f"ERROR: export file missing: {p}\n"
                     "       Run prf_analyzeprf_export.py first.")
    export_meta = json.loads(paths["json"].read_text())
    manifest_path = Path(args.chunk_dir) / "fit-manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if not manifest:
        print(f"  WARNING: no fit-manifest.json in {args.chunk_dir}; the fit did not "
              "finish its last chunk loop -- coverage is checked below regardless")
    mask_img = nib.load(str(paths["mask"]))
    mask = np.asarray(mask_img.dataobj) > 0
    n_vox = int(mask.sum())
    if n_vox != export_meta["MaskVoxels"]:
        sys.exit(f"ERROR: mask has {n_vox} voxels but the export sidecar says "
                 f"{export_meta['MaskVoxels']}; the export is inconsistent")
    negate = args.polarity == "negprf"
    if manifest and bool(manifest.get("negate")) != negate:
        sys.exit(f"ERROR: chunks in {args.chunk_dir} were fitted with negate="
                 f"{manifest.get('negate')} but --polarity {args.polarity} was asked")

    chunks = load_chunks(args.chunk_dir)
    results, numiters = chunks_to_results(chunks, n_vox, export_meta["FieldOfViewDeg"],
                                          export_meta["ApertureResolution"])
    print(f"  {len(chunks)} chunks, {n_vox} voxels; R2 median "
          f"{np.nanmedian(results['R2']):.2f}%, >10%: {int((results['R2'] > 10).sum())}, "
          f"|gain|>1e6: {int((np.abs(results['gain']) > 1e6).sum())}")

    fov = export_meta["FieldOfViewDeg"]
    meta = dict(export_meta)
    meta.update({
        "Description": (f"CSS pRF fit in space-{args.space}, analyzePRF backend"
                        + (" on SIGN-FLIPPED PSC BOLD: negative-pRF characterisation."
                           if negate else ".")),
        "Backend": "analyzePRF (MATLAB, cvnlab/analyzePRF + cvnlab/knkutils)",
        "BackendCommits": {"analyzePRF": args.analyzeprf_commit,
                           "knkutils": args.knkutils_commit},
        "BackendCall": manifest.get("call", "analyzePRF(stimulus, data, tr, "
                                            "struct('seedmode',2,'maxiter',100,'display','off'))"),
        "BackendNote": (
            "NSD's analysis_prf.m call, verbatim: super-grid seed (seedmode 2), "
            "maxiter 100, free exponent, analyzePRF's default HRF and polynomial "
            "degree, EVERY mask voxel optimised, nothing gated. Supersedes the "
            "Python port (fit_prf.py), which refined only voxels above 5% grid R2 "
            "and so wrote R2 = 0 below it; DECIDED 2026-09-16, workbench "
            "prf-retinotopy."),
        "MATLABVersion": manifest.get("matlab"),
        "SeedMode": manifest.get("seedmode", 2),
        "MaxIter": manifest.get("maxiter", 100),
        "ExponentLowerBound": manifest.get("exptlowerbound"),
        "Model": "analyzePRF CSS: gain * conv((S.g)^n, HRF) + polynomials",
        "HRF": "analyzePRF default getcanonicalhrf(tr, tr), as used by the fit",
        "HRFKind": "kay",
        "PolynomialDegreePerRun": manifest.get("maxpolydeg"),
        "FieldOfViewCaveat": ("Design geometry only; the subtended angle at the "
                              "scanner was never recorded. Eccentricity and size "
                              "scale LINEARLY with this value. Polar angle does not."),
        "SizeDefinition": (
            "sigma/sqrt(n), matching NSD prf_size (analyzePRF rfsize). NOT the raw "
            "sigma that most published size ranges report -- compare `sigma` to "
            "those and `size` to NSD. NSD's stimulus was 8.4 deg in diameter "
            f"against our {fov} deg, and pRF size grows with eccentricity, so a "
            "size comparison to NSD also needs an eccentricity-matched cut."),
        "SigmaExponentNote": (
            "Only size (sigma/sqrt(n)) is well identified by the CSS model; "
            "analyzePRF lets the exponent run to its lower bound with sigma "
            "shrinking to match, so `sigma` and `exponent` individually differ "
            "from the Python fit's while `size`, angle, eccentricity and R2 agree "
            "(measured 2026-09-18). Interpret the pair through `size`."),
        "GainNote": (
            "gain is in the data's units (percent signal change of the cleaned, "
            "averaged pseudo-runs), not NSD's raw units. analyzePRF's super-grid "
            "seeding gives noise voxels astronomically large gains (|gain| > 1e6 "
            "on ~1/3 of voxels with R2 < 5%); never read gain without the R2 mask."),
        "StimulusRadiusDeg": fov / 2.0,
        "StimulusRadiusNote": (
            "Fitted pRF centres beyond this radius are extrapolations: the "
            "stimulus never reached them. Maps are emitted UNMASKED (NSD "
            "releases unthresholded maps too); apply this radius in any "
            "summary, figure or ROI drawn from them."),
        "CentreBoundsNote": (
            "analyzePRF optimises with Levenberg-Marquardt, which ignores the "
            "parameter box, so fitted centres are UNBOUNDED: in the sub-04 pilot "
            "512 of 9,565 R2>10% voxels sat beyond 31.5 deg (max 427 deg), all "
            "of them extrapolations from the stimulus edge. The Python fit "
            "bounded them (max 21 deg). Apply StimulusRadiusDeg before any "
            "eccentricity summary; correlations across the unmasked map are "
            "dominated by these voxels."),
        "Thresholded": False,
        "RefineThresholdR2Pct": None,
        "RefineThresholdNote": "none: every voxel is optimised (analyzePRF behaviour)",
        "AngleConvention": "degrees CCW from right horizontal meridian, 0-360",
        "IterationsMedian": float(np.nanmedian(numiters)),
        "Provenance": ("mmmdata/scripts/prf_analyzeprf_{export.py,fit.m,assemble.py} "
                       "via fit_prf_analyzeprf.sbatch; workbench prf-retinotopy"),
    })
    if negate:
        meta["SignFlipped"] = True
        meta["SignFlipReference"] = ("Negative-pRF approach per "
                                     "https://www.biorxiv.org/content/10.1101/"
                                     "2024.09.27.615397v2")
    meta.pop("MaskShape", None)
    meta.pop("DataUnits", None)

    out_dir = Path(args.out_root) / f"sub-{args.subject}"
    base = export_stem(args.subject, args.space)
    moved = archive_existing(out_dir, base, args.polarity, args.archive_root, args.subject)
    if moved:
        print(f"  archived {len(moved)} existing `{args.polarity}` files to "
              f"{Path(args.archive_root) / f'sub-{args.subject}'}")
        meta["Supersedes"] = {"ArchivedTo": str(Path(args.archive_root) / f"sub-{args.subject}"),
                              "Files": moved}
    written, sidecar = fit_prf.write_maps(results, mask_img, mask, out_dir, base, meta,
                                          suffix=args.polarity)
    print(f"  wrote {len(written)} maps + {sidecar.name} to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
