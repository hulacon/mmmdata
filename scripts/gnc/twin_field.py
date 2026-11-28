#!/usr/bin/env python3
"""Gradient-nonlinearity (GNC) field from ND/DIS3D anatomical twins.

The scanner corrects anatomical scans for gradient nonlinearity (DIS3D) but not
the EPI. Each anatomical series also has an uncorrected twin (series name
ending ``_ND``) reconstructed from the same raw data. Registering ND to DIS3D
recovers the scanner's own correction field over the head, in scanner
coordinates, with no contrast or noise difference between the images.

Subcommands (all outputs go under ``--out``, a scratch directory; the source
tree holds PII, so nothing here is written into the BIDS root or a repo):

    pairs     convert one session's anatomical ND and DIS3D series with dcm2niix
              and pair them (same series base name, same acquisition time,
              identical grid); writes pairs.json
    register  ND -> DIS3D BSplineSyN for every pair, plus a SynthStrip brain
              mask of each DIS3D image; writes <pair>/field.nii.gz, the
              displacement (mm, ITK/LPS frame) at each DIS3D voxel
    floor     within-session repeatability: two pairs at one FOV placement
              sample the same field at the same scanner positions, so their
              voxelwise difference inside both brain masks is the estimation
              floor; writes floor.json

Registration settings are fixed here, before any result is read, so they
cannot be tuned against the repeatability precondition they are scored by.

Needs ANTs on PATH (loaded before the venv) and mri_synthstrip, either on PATH or
as SYNTHSTRIP_CMD (e.g. 'apptainer exec <freesurfer .sif> mri_synthstrip'):
    module load ants/2.5.2 freesurfer/8.2.0

Usage:
    python twin_field.py pairs    --sub ## --ses ## --out <dir>
    python twin_field.py register --out <dir> [--jobs N]
    python twin_field.py floor    --out <dir> [--pair-a <name> --pair-b <name>]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src" / "python") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src" / "python"))

from core.config import load_config  # noqa: E402

# Anatomical series that carry an ND twin. The coronal T2 is a partial, oblique
# FOV: it samples the same field in another orientation.
TWIN_SERIES = re.compile(r"^Series_\d+_(ABCD_T1w_MPR_vNav|ABCD_T2w_SPC_vNav|T2_coronal_1\.8)(_ND)?$")
DCM2NIIX_CONTAINER = "dcm2bids-3.2.0.sif"  # the build the BIDS conversion uses

# Fixed before any result was read; changing them means re-measuring the floor. BSplineSyN because
# the GNC field is a low-order spherical-harmonic field: a 26 mm control-point
# mesh can represent it and suppresses voxel-scale registration noise. No
# affine stage: the linear part of the field is part of what we measure.
SYN_TRANSFORM = "BSplineSyN[0.1,26,0,3]"
SYN_METRIC_RADIUS = 4
SYN_CONVERGENCE = "[100x70x50x20,1e-7,10]"
SYN_SHRINK = "6x4x2x1"
SYN_SMOOTH = "3x2x1x0vox"
# Brain-mask command. The freesurfer/8.2.0 module is an apptainer wrapper set
# that does not wrap mri_synthstrip, so the sbatch passes the image-exec form.
SYNTHSTRIP = os.environ.get("SYNTHSTRIP_CMD", "mri_synthstrip").split()


def _run(cmd: list[str], log: Path) -> None:
    with log.open("a") as fh:
        fh.write("$ " + " ".join(cmd) + "\n")
        fh.flush()
        res = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
    if res.returncode:
        sys.exit(f"command failed ({res.returncode}); see {log}: {' '.join(cmd[:3])} ...")


def cmd_pairs(args) -> None:
    cfg = load_config()["paths"]
    dicom_dir = Path(cfg["source_dir"]) / f"sub-{args.sub}" / f"ses-{args.ses}" / "dicom"
    if not dicom_dir.is_dir():
        sys.exit(f"no DICOM directory at {dicom_dir}")
    container = Path(cfg["singularity_dir"]) / DCM2NIIX_CONTAINER
    if not container.is_file():
        sys.exit(f"dcm2niix container missing: {container}")
    out = args.out
    nii_dir = out / "nifti"
    nii_dir.mkdir(parents=True, exist_ok=True)
    log = out / "pairs.log"

    series = sorted(p for p in dicom_dir.iterdir() if p.is_dir() and TWIN_SERIES.match(p.name))
    if not series:
        sys.exit(f"no twin-eligible anatomical series under {dicom_dir}")
    records = []
    for s in series:
        m = TWIN_SERIES.match(s.name)
        stem = s.name
        if not (nii_dir / f"{stem}.nii.gz").exists():
            _run(["apptainer", "exec", "-B", "/gpfs", str(container), "dcm2niix", "-z", "y", "-b", "y",
                  "-x", "n", "-f", stem, "-o", str(nii_dir), str(s)], log)
        side = json.loads((nii_dir / f"{stem}.json").read_text())
        img = nib.load(nii_dir / f"{stem}.nii.gz")
        records.append({
            "series": stem, "base": m.group(1), "nd": bool(m.group(2)),
            "acq_time": side.get("AcquisitionTime"), "series_number": side.get("SeriesNumber"),
            "image_type": side.get("ImageType"), "gnc": side.get("NonlinearGradientCorrection"),
            "affine": img.affine.tolist(), "shape": list(img.shape),
        })

    pairs, problems = [], []
    nds = [r for r in records if r["nd"]]
    for nd in nds:
        mates = [r for r in records if not r["nd"] and r["base"] == nd["base"] and r["acq_time"] == nd["acq_time"]]
        if len(mates) != 1:
            problems.append(f"{nd['series']}: {len(mates)} DIS3D mates at acquisition time {nd['acq_time']}")
            continue
        d = mates[0]
        if nd["shape"] != d["shape"] or not np.allclose(nd["affine"], d["affine"], atol=1e-4):
            problems.append(f"{nd['series']} vs {d['series']}: grids differ")
            continue
        if nd["gnc"] is not False or d["gnc"] is not True:
            problems.append(f"{nd['series']} vs {d['series']}: NonlinearGradientCorrection "
                            f"{nd['gnc']} / {d['gnc']}, expected false / true")
            continue
        pairs.append({"name": nd["series"].removesuffix("_ND").split("_", 2)[-1] + f"_s{nd['series_number']:02d}",
                      "nd": nd["series"], "dis3d": d["series"], "acq_time": nd["acq_time"],
                      "shape": nd["shape"], "affine": nd["affine"]})
    manifest = {"sub": args.sub, "ses": args.ses, "dicom_dir": str(dicom_dir),
                "dcm2niix_container": str(container), "pairs": pairs, "problems": problems}
    (out / "pairs.json").write_text(json.dumps(manifest, indent=2))
    print(f"{len(pairs)} pairs, {len(problems)} problems -> {out / 'pairs.json'}")
    for p in problems:
        print("  PROBLEM:", p)
    if not pairs:
        sys.exit("no usable pairs")


def _register_one(out: Path, pair: dict, threads: int) -> str:
    nii = out / "nifti"
    pdir = out / pair["name"]
    pdir.mkdir(exist_ok=True)
    log = pdir / "register.log"
    fixed, moving = nii / f"{pair['dis3d']}.nii.gz", nii / f"{pair['nd']}.nii.gz"
    mask = pdir / "brainmask.nii.gz"
    if not mask.exists():
        _run([*SYNTHSTRIP, "-i", str(fixed), "-m", str(mask)], log)
    prefix = pdir / "nd2dis3d_"
    warp = Path(f"{prefix}0Warp.nii.gz")
    env = dict(os.environ, ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=str(threads))
    if not warp.exists():
        cmd = ["antsRegistration", "-d", "3", "--float", "0", "-v", "1",
               "-o", f"[{prefix},{prefix}Warped.nii.gz]", "-n", "Linear", "-w", "[0.005,0.995]",
               "-u", "0", "-t", SYN_TRANSFORM,
               "-m", f"CC[{fixed},{moving},1,{SYN_METRIC_RADIUS}]",
               "-c", SYN_CONVERGENCE, "-f", SYN_SHRINK, "-s", SYN_SMOOTH]
        with log.open("a") as fh:
            fh.write("$ " + " ".join(cmd) + "\n")
            fh.flush()
            res = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)
        if res.returncode:
            sys.exit(f"{pair['name']}: antsRegistration failed; see {log}")
    # ANTs writes the displacement as a 5D (x, y, z, 1, 3) vector image in ITK's LPS frame.
    w = nib.load(warp)
    field = np.asarray(w.dataobj, dtype=np.float32).reshape(w.shape[:3] + (3,))
    nib.save(nib.Nifti1Image(field, w.affine), pdir / "field.nii.gz")
    # Sanity: how well the warp makes ND match DIS3D, before and after, inside the brain.
    m = np.asarray(nib.load(mask).dataobj) > 0
    f = np.asarray(nib.load(fixed).dataobj, dtype=np.float64)[m]
    mv = np.asarray(nib.load(moving).dataobj, dtype=np.float64)[m]
    wp = np.asarray(nib.load(f"{prefix}Warped.nii.gz").dataobj, dtype=np.float64)[m]
    mag = np.linalg.norm(field[m], axis=1)
    summary = {"pair": pair["name"], "settings": {"transform": SYN_TRANSFORM, "metric": f"CC r={SYN_METRIC_RADIUS}",
                                                   "convergence": SYN_CONVERGENCE, "shrink": SYN_SHRINK, "smooth": SYN_SMOOTH},
               "brain_voxels": int(m.sum()),
               "r_before": float(np.corrcoef(f, mv)[0, 1]), "r_after": float(np.corrcoef(f, wp)[0, 1]),
               "field_mm": {"median": float(np.median(mag)), "p95": float(np.percentile(mag, 95)), "max": float(mag.max())}}
    (pdir / "register.json").write_text(json.dumps(summary, indent=2))
    return f"{pair['name']}: r {summary['r_before']:.5f} -> {summary['r_after']:.5f}, |d| median {summary['field_mm']['median']:.3f} mm"


def cmd_register(args) -> None:
    manifest = json.loads((args.out / "pairs.json").read_text())
    if shutil.which("antsRegistration") is None:
        sys.exit("antsRegistration must be on PATH (module load ants/2.5.2 before the venv)")
    if shutil.which(SYNTHSTRIP[0]) is None:
        sys.exit(f"{SYNTHSTRIP[0]} not found; set SYNTHSTRIP_CMD (the freesurfer/8.2.0 module wraps no "
                 "mri_synthstrip: use 'apptainer exec <its .sif> mri_synthstrip')")
    pairs = manifest["pairs"]
    threads = max(1, args.threads // max(1, args.jobs))
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        for line in ex.map(lambda p: _register_one(args.out, p, threads), pairs):
            print(line)


def cmd_floor(args) -> None:
    out = args.out
    if not (args.pair_a and args.pair_b):
        t1 = [p["name"] for p in json.loads((out / "pairs.json").read_text())["pairs"] if p["name"].startswith("ABCD_T1w")]
        if len(t1) != 2:
            sys.exit(f"expected two T1w pairs to compare by default, found {t1}; pass --pair-a/--pair-b")
        args.pair_a, args.pair_b = t1
    a, b = out / args.pair_a, out / args.pair_b
    fa, fb = nib.load(a / "field.nii.gz"), nib.load(b / "field.nii.gz")
    if fa.shape != fb.shape or not np.allclose(fa.affine, fb.affine, atol=1e-4):
        sys.exit("the two pairs are not on one grid (different FOV placement): the floor needs one placement")
    m = (np.asarray(nib.load(a / "brainmask.nii.gz").dataobj) > 0) & (np.asarray(nib.load(b / "brainmask.nii.gz").dataobj) > 0)
    da, db = np.asarray(fa.dataobj)[m], np.asarray(fb.dataobj)[m]
    diff = np.linalg.norm(da - db, axis=1)
    mag = np.linalg.norm(0.5 * (da + db), axis=1)
    res = {"pair_a": args.pair_a, "pair_b": args.pair_b, "voxels": int(m.sum()),
           "delta_mm": {"median": float(np.median(diff)), "p95": float(np.percentile(diff, 95)), "max": float(diff.max())},
           "per_axis_delta_sd_mm": [float(v) for v in (da - db).std(axis=0)],
           "field_mm": {"median": float(np.median(mag)), "p95": float(np.percentile(mag, 95))},
           "precondition": {"bar_median_mm": 0.05, "pass": bool(np.median(diff) <= 0.05)}}
    (out / "floor.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pairs")
    p.add_argument("--sub", required=True, help="bare label")
    p.add_argument("--ses", required=True, help="bare label")
    p.add_argument("--out", required=True, type=Path)
    r = sub.add_parser("register")
    r.add_argument("--out", required=True, type=Path)
    r.add_argument("--jobs", type=int, default=1, help="pairs registered in parallel")
    r.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    f = sub.add_parser("floor")
    f.add_argument("--out", required=True, type=Path)
    f.add_argument("--pair-a", help="default: the session's two T1w pairs")
    f.add_argument("--pair-b")
    args = ap.parse_args(argv)
    {"pairs": cmd_pairs, "register": cmd_register, "floor": cmd_floor}[args.cmd](args)


if __name__ == "__main__":
    main()
