#!/usr/bin/env python3
"""
prf_splithalf_export.py — one SESSION's pRF runs, exported for analyzePRF on
the pooled product's own fit mask, for a session split-half of the pooled fit.

The pooled pRF product (derivatives/prf, scripts/fit_prf_analyzeprf.sbatch)
fits all of a subject's pRF runs from two sessions: per-run PSC and Legendre
drift removal, the runs of each stimulus setnum averaged into one pseudo-run,
and the pseudo-runs fitted jointly by analyzePRF with NSD's verbatim call.
This script builds the SAME input from ONE session's runs:

* the same design, cleaning and averaging (``fit_prf`` functions, unchanged);
* the same fit mask and grid: the mask is rebuilt from ALL of the pooled
  fit's runs on the pooled fit's grid reference, and checked to have the
  pooled sidecar's voxel count, so the two halves and the pooled map share
  voxels one-to-one.

A session typically holds both setnums, one of them twice, so each half has
the pooled fit's pseudo-run count with fewer runs averaged into each. The
sidecar records which runs fed which pseudo-run.

The fit and assembly are the pooled product's own steps, untouched:
``scripts/prf_analyzeprf_fit.m`` and ``scripts/prf_analyzeprf_assemble.py``
(driven by ``prf_splithalf.sbatch``).

Usage:
    python prf_splithalf_export.py --subject ## --session ## --out-dir <work>/sub-##/ses-##
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import fit_prf  # noqa: E402
from prf_analyzeprf_export import cell_row, export_paths  # noqa: E402

SPACE = "T1w"


def pooled_sidecar(derivatives, subject):
    p = Path(derivatives) / fit_prf.OUTPUT_TREE / f"sub-{subject}" / f"sub-{subject}_task-prf_space-{SPACE}_prf.json"
    if not p.exists():
        sys.exit(f"ERROR: no pooled pRF sidecar at {p}; the split half is defined against it")
    return json.loads(p.read_text())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject", required=True, help="bare label")
    ap.add_argument("--session", required=True, help="bare label of the half's session")
    ap.add_argument("--derivatives", default=None, help="default: config output_dir")
    ap.add_argument("--aperture-dir", default=str(fit_prf.BIDS_ROOT / "stimuli" / "prf"))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    import nibabel as nib
    import scipy.io as sio

    if args.derivatives is None:
        from core.config import load_config

        args.derivatives = load_config()["paths"]["output_dir"]
    subject, session = args.subject, args.session
    paths = export_paths(args.out_dir, subject, SPACE)
    if all(p.exists() for p in paths.values()) and not args.force:
        print(f"export exists, skipping: {paths['mat']}")
        return 0

    pooled = pooled_sidecar(args.derivatives, subject)
    pooled_units = [(r.split("_")[0].split("-")[1], int(r.split("_")[1].split("-")[1])) for r in pooled["Runs"]]
    grid_session = pooled["GridReference"].split("_")[0].split("-")[1]
    units = [u for u in pooled_units if u[0] == session]
    if not units:
        sys.exit(f"ERROR: the pooled fit has no ses-{session} runs: {pooled['Runs']}")
    print(f"sub-{subject} half ses-{session}: {len(units)} runs {[f'{s}/{r:02d}' for s, r in units]}")

    designs = fit_prf.build_run_designs(subject, units, args.aperture_dir)
    S, run_index, groups, setnums = fit_prf.group_designs(designs, "average")
    for g, setnum in zip(groups, setnums):
        print(f"    set-{setnum}: averaging {len(g)} runs "
              + ", ".join(f"ses-{designs[i]['session']}/run-{designs[i]['run']:02d}" for i in g))
    if sorted(setnums) != sorted(int(k) for k in pooled["RunsBySetNumber"]):
        sys.exit(f"ERROR: half ses-{session} has setnums {setnums}, the pooled fit "
                 f"{sorted(pooled['RunsBySetNumber'])}; the half would not fit the same model")

    mask, reference, grid_run, resampled = fit_prf.build_fit_mask(subject, pooled_units, SPACE, grid_session)
    if int(mask.sum()) != int(pooled["MaskVoxels"]):
        sys.exit(f"ERROR: rebuilt mask has {int(mask.sum())} voxels, the pooled fit {pooled['MaskVoxels']}")
    blocks = fit_prf.pooled_blocks(subject, designs, groups, SPACE, mask, reference)

    n_tr = fit_prf.N_TR
    stimulus = [np.ascontiguousarray(np.transpose(S[k * n_tr:(k + 1) * n_tr], (1, 2, 0)).astype(np.float32))
                for k in range(len(groups))]
    data = [np.ascontiguousarray(b.T.astype(np.float32)) for b in blocks]
    del blocks

    meta = {
        "Description": ("One session's pRF pseudo-runs exported for analyzePRF on the pooled "
                        "product's fit mask: one half of a session split of the pooled fit."),
        "Subject": f"sub-{subject}",
        "Sessions": [f"ses-{session}"],
        "FitUnit": "session half of the pooled subject-level fit (split-half reliability)",
        "PooledFitSessions": pooled.get("Sessions"),
        "Runs": [f"ses-{s_}_run-{r_:02d}" for s_, r_ in units],
        "SetNumbers": setnums,
        "RunsBySetNumber": {str(setnum): [f"ses-{designs[i]['session']}_run-{designs[i]['run']:02d}" for i in g]
                            for g, setnum in zip(groups, setnums)},
        "PoolingMethod": pooled.get("PoolingMethod"),
        "Space": f"{SPACE} (fMRIPrep output space)",
        "GridReference": f"ses-{grid_session}_run-{grid_run:02d}",
        "ResampledRuns": resampled,
        "MaskSource": "rebuilt from every run of the pooled fit on its grid reference; voxel count checked",
        "ConfoundModel": "none",
        "TR": fit_prf.TR, "VolumesPerRun": n_tr,
        "ApertureResolution": fit_prf.APERTURE_RES,
        "FieldOfViewDeg": fit_prf.FOV_DEG,
        "MaskVoxels": int(mask.sum()),
        "Provenance": "mmmdata/scripts/functional_space/prf_splithalf_export.py",
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_mat = paths["mat"].with_suffix(f".tmp{os.getpid()}.mat")
    sio.savemat(str(tmp_mat), {
        "stimulus": cell_row(stimulus), "data": cell_row(data), "tr": float(fit_prf.TR),
        "setnums": np.asarray(setnums), "fov_deg": float(fit_prf.FOV_DEG),
        "aperture_res": int(fit_prf.APERTURE_RES), "n_vox": int(mask.sum()),
    }, do_compression=True)
    os.replace(tmp_mat, paths["mat"])
    mask_img = nib.Nifti1Image(mask.astype(np.uint8), reference.affine)
    mask_img.header.set_data_dtype(np.uint8)
    tmp_mask = paths["mask"].with_name(paths["mask"].name.replace(".nii.gz", f".tmp{os.getpid()}.nii.gz"))
    nib.save(mask_img, str(tmp_mask))
    os.replace(tmp_mask, paths["mask"])
    paths["json"].write_text(json.dumps(meta, indent=2) + "\n")
    print(f"  wrote {paths['mat'].name}: {len(groups)} pseudo-runs x {mask.sum()} voxels")
    return 0


if __name__ == "__main__":
    sys.exit(main())
