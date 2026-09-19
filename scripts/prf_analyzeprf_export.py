#!/usr/bin/env python3
"""
prf_analyzeprf_export.py — hand fit_prf.py's pooled pseudo-runs to analyzePRF.

Step 1 of the analyzePRF-backed pRF fit (mmmdata-agents workbench
prf-retinotopy, DECIDED 2026-09-16: pRF data are handled EXACTLY as
analyzePRF handles them, so the fit is analyzePRF itself, not the Python
port). This script builds the SAME inputs the Python fit uses -- the two
averaged pseudo-runs, on the same fit mask, from `fit_prf.py`'s own
functions -- and writes them as a MATLAB file with analyzePRF's calling
convention:

    stimulus   1 x n_runs cell, each res x res x N_TR (0-1 aperture, TR-binned)
    data       1 x n_runs cell, each n_vox x N_TR (per-run PSC, per-run
               Legendre 0-3 removed, averaged over that setnum's runs)
    tr         seconds

plus the fit mask as a NIfTI (so the assembler can put voxels back) and a
JSON sidecar with the provenance keys `fit_prf.py` would have written.
Nothing here is a fitting choice: pooling, cleaning and the mask are the
2026-09-09 decisions, unchanged. The chain is

    prf_analyzeprf_export.py  ->  prf_analyzeprf_fit.m  ->  prf_analyzeprf_assemble.py

driven by fit_prf_analyzeprf.sbatch.

Usage:
    python prf_analyzeprf_export.py --subject 03 --sessions 02 03 --space T1w \
        --out-dir /path/to/work/sub-03
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import fit_prf  # noqa: E402


def export_stem(subject, space):
    return f"sub-{subject}_task-prf_space-{space}"


def export_paths(out_dir, subject, space):
    """The three files this script writes, keyed by role."""
    stem = export_stem(subject, space)
    out_dir = Path(out_dir)
    return {"mat": out_dir / f"{stem}_desc-pooled_analyzeprfinput.mat",
            "mask": out_dir / f"{stem}_desc-fit_mask.nii.gz",
            "json": out_dir / f"{stem}_desc-pooled_analyzeprfinput.json"}


def cell_row(items):
    """A 1 x n object array, which scipy writes as a MATLAB 1 x n cell."""
    c = np.empty((1, len(items)), dtype=object)
    for k, item in enumerate(items):
        c[0, k] = item
    return c


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject", required=True, help="bare label, e.g. 03")
    ap.add_argument("--sessions", nargs="+", required=True, metavar="SES",
                    help="sessions to pool (bare labels)")
    ap.add_argument("--space", default="T1w",
                    help="fMRIPrep output space the runs share (default T1w)")
    ap.add_argument("--runs", type=int, nargs="+",
                    help="default: all pRF runs found in every session")
    ap.add_argument("--grid-session", default=None,
                    help="session whose grid every run is resampled onto "
                         "(default: the last session given, as fit_prf.py)")
    ap.add_argument("--aperture-dir", default=str(fit_prf.BIDS_ROOT / "stimuli" / "prf"))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--force", action="store_true",
                    help="rewrite an existing export (default: skip, so the "
                         "positive and negative array tasks can share one)")
    args = ap.parse_args()

    import nibabel as nib
    import scipy.io as sio

    subject, sessions, space = args.subject, args.sessions, args.space
    if len(sessions) < 2:
        ap.error("--sessions needs at least two sessions: this is the pooled, "
                 "subject-level product (per-session fits stay with fit_prf.py)")
    paths = export_paths(args.out_dir, subject, space)
    if all(p.exists() for p in paths.values()) and not args.force:
        print(f"export exists, skipping: {paths['mat']}")
        return 0

    units = []
    for ses in sessions:
        runs = args.runs or fit_prf.detect_runs(subject, ses, space)
        units.extend((ses, r) for r in runs)
    print(f"sub-{subject} ses-{'+'.join(sessions)} in space-{space}: "
          f"{len(units)} runs {[f'{s}/{r:02d}' for s, r in units]}")

    print("  building design")
    designs = fit_prf.build_run_designs(subject, units, args.aperture_dir)
    S, run_index, groups, setnums = fit_prf.group_designs(designs, "average")
    for g, setnum in zip(groups, setnums):
        members = ", ".join(f"ses-{designs[i]['session']}/run-{designs[i]['run']:02d}"
                            for i in g)
        print(f"    set-{setnum}: averaging {len(g)} runs ({members})")

    grid_session = args.grid_session or sessions[-1]
    if grid_session not in sessions:
        ap.error(f"--grid-session {grid_session} is not among {sessions}")
    mask, reference, grid_run, resampled = fit_prf.build_fit_mask(
        subject, units, space, grid_session)
    blocks = fit_prf.pooled_blocks(subject, designs, groups, space, mask, reference)

    n_tr = fit_prf.N_TR
    # analyzePRF wants stimulus as rows x cols x time and data as voxels x time.
    stimulus = [np.ascontiguousarray(
        np.transpose(S[k * n_tr:(k + 1) * n_tr], (1, 2, 0)).astype(np.float32))
        for k in range(len(groups))]
    data = [np.ascontiguousarray(b.T.astype(np.float32)) for b in blocks]
    del blocks

    meta = {
        "Description": ("Pooled pRF pseudo-runs exported for analyzePRF, built by "
                        "fit_prf.py's own functions (same design, mask, cleaning "
                        "and averaging as the Python fit)."),
        "Subject": f"sub-{subject}",
        "Sessions": [f"ses-{x}" for x in sessions],
        "FitUnit": "subject (sessions pooled)",
        "Runs": [f"ses-{s_}_run-{r_:02d}" for s_, r_ in units],
        "SetNumbers": setnums,
        "RunsBySetNumber": {
            str(setnum): [f"ses-{designs[i]['session']}_run-{designs[i]['run']:02d}"
                          for i in g]
            for g, setnum in zip(groups, setnums)},
        "PoolingMethod": (
            "average the runs of each setnum after per-run PSC and detrend, then "
            "fit the resulting pseudo-runs jointly -- NSD's route "
            "(cvnlab/nsddatapaper main/glm_prf.m: 'average the 3 reps of each "
            "stimulus type up front'). Cleaning precedes averaging because our "
            "runs span sessions, which NSD's did not."),
        "Space": f"{space} (fMRIPrep output space)",
        "GridReference": f"ses-{grid_session}_run-{grid_run:02d}",
        "ResampledRuns": resampled,
        "ResampledRunsNote": (
            "Runs whose grid differed from the reference and were resampled onto "
            "it (pure grid change within one space; no registration). Expected "
            "only for sub-03, whose ses-01/ses-02 used a smaller FOV."),
        "ConfoundModel": "none",
        "DataUnits": ("percent signal change per run, per-run Legendre 0-"
                      f"{fit_prf.POLY_DEGREE} projected out, then averaged per setnum"),
        "TR": fit_prf.TR, "VolumesPerRun": n_tr,
        "ApertureResolution": fit_prf.APERTURE_RES,
        "FieldOfViewDeg": fit_prf.FOV_DEG,
        "MaskVoxels": int(mask.sum()),
        "MaskShape": list(mask.shape),
        "Provenance": "mmmdata/scripts/prf_analyzeprf_export.py; workbench prf-retinotopy",
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Written to temporaries and renamed, so a concurrent task that finds the
    # final names can trust them and one that races only wastes its own work.
    tmp_mat = paths["mat"].with_suffix(f".tmp{os.getpid()}.mat")
    sio.savemat(str(tmp_mat), {
        "stimulus": cell_row(stimulus), "data": cell_row(data),
        "tr": float(fit_prf.TR), "setnums": np.asarray(setnums),
        "fov_deg": float(fit_prf.FOV_DEG),
        "aperture_res": int(fit_prf.APERTURE_RES),
        "n_vox": int(mask.sum()),
    }, do_compression=True)
    os.replace(tmp_mat, paths["mat"])
    mask_img = nib.Nifti1Image(mask.astype(np.uint8), reference.affine)
    mask_img.header.set_data_dtype(np.uint8)
    # nibabel picks the format from the extension, so the temporary keeps .nii.gz
    tmp_mask = paths["mask"].with_name(paths["mask"].name.replace(".nii.gz", f".tmp{os.getpid()}.nii.gz"))
    nib.save(mask_img, str(tmp_mask))
    os.replace(tmp_mask, paths["mask"])
    tmp_json = paths["json"].with_suffix(f".tmp{os.getpid()}.json")
    tmp_json.write_text(json.dumps(meta, indent=2) + "\n")
    os.replace(tmp_json, paths["json"])

    print(f"  wrote {paths['mat'].name} ({paths['mat'].stat().st_size / 1e6:.0f} MB): "
          f"{len(groups)} runs x {mask.sum()} voxels x {n_tr} TRs, "
          f"stimulus {fit_prf.APERTURE_RES}x{fit_prf.APERTURE_RES}")
    print(f"  wrote {paths['mask'].name}, {paths['json'].name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
