#!/usr/bin/env python3
"""Export the HRF-selection outputs of one GLMsingle arm as small NIfTIs.

GLMsingle keeps ``HRFindex`` (the 1-of-20 library kernel each voxel chose),
``FitHRFR2`` (the R2 of every kernel at every voxel, from which the choice was
made), ``R2`` and ``meanvol`` inside a multi-GB pickled dict (``TYPEB_FITHRF``),
together with their per-run forms: ``HRFindexrun`` (the kernel each RUN of the
same fit would have chosen for the voxel), ``R2run`` and ``FitHRFR2run``.
Anything that wants to compare HRF choices across arms, or across runs inside
one arm, should not have to load betas to do it, so this writes them as NIfTIs
under ``<arm>/hrf/``:

    sub-##_arm-<arm>_space-<space>_desc-hrfindex_dseg.nii.gz      (int16)
    sub-##_arm-<arm>_space-<space>_desc-fithrfr2_stat.nii.gz      (4D, 20 kernels)
    sub-##_arm-<arm>_space-<space>_desc-hrfr2_stat.nii.gz
    sub-##_arm-<arm>_space-<space>_desc-meanvol_bold.nii.gz
    sub-##_arm-<arm>_space-<space>_desc-hrfindexrun_dseg.nii.gz   (4D, one volume per run)
    sub-##_arm-<arm>_space-<space>_desc-hrfr2run_stat.nii.gz      (4D, per run)
    sub-##_arm-<arm>_space-<space>_desc-hrfmarginrun_stat.nii.gz  (4D; best minus second-best
                                                                    kernel R2 per run)
    sub-##_arm-<arm>_space-<space>_desc-hrfsecondrun_dseg.nii.gz  (4D; the per-run runner-up kernel)

``FitHRFR2run`` itself (grid x runs x 20 kernels, ~3.6 GB for a 42-run fit) is
written only inside a brain mask and only when ``--mask`` is given, as

    sub-##_arm-<arm>_space-<space>_desc-fithrfr2run_masked.npz
        data (voxels, runs, 20) float32, index (voxels,) flat C-order indices
        into shape, shape (3,), run_labels (runs,) from run_metadata.json

(~0.8 GB for 42 runs x ~230k voxels) so kernel selection can be replayed on
any pooling of runs (Track B of the glm-strategy pass 2). Per-run choices are a genuine within-fit replicate: GLMsingle
fits each kernel to all runs jointly, but trial regressors and polynomials are
run-specific, so a run's R2 depends only on that run's own data.

The grid comes from ``--reference``, a NIfTI on the fit's grid (the arm has no
affine of its own); a shape mismatch is refused. Loading the pickle needs RAM
for the whole dict (~16 G for a 42-run TB fit) plus one copy of ``FitHRFR2run``
— run it under sbatch. Existing exports are kept; only missing ones are
written unless ``--force``.

Usage:
    python glmsingle_export_hrf.py --subject sub-## --arm enc \
        --reference <NIfTI on the fit grid> [--tree derivatives/glmsingle_tb]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_SRC = str(Path(__file__).resolve().parents[1] / "src" / "python")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

SPACE = "MNI152NLin2009cAsym_res-2"
#: dict key -> (desc, suffix, dtype)
EXPORTS = {
    "HRFindex": ("hrfindex", "dseg", np.int16),
    "FitHRFR2": ("fithrfr2", "stat", np.float32),
    "R2": ("hrfr2", "stat", np.float32),
    "meanvol": ("meanvol", "bold", np.float32),
    "HRFindexrun": ("hrfindexrun", "dseg", np.int16),
    "R2run": ("hrfr2run", "stat", np.float32),
    "HRFmarginrun": ("hrfmarginrun", "stat", np.float32),  # derived from FitHRFR2run
    "HRFsecondrun": ("hrfsecondrun", "dseg", np.int16),  # derived from FitHRFR2run
}


def per_run_margin(fit_run: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """From ``FitHRFR2run`` (..., runs, kernels): best index, runner-up index, best - runner-up R2.

    NaN R2 (voxels GLMsingle never fitted) yield NaN margin and index 0.
    """
    fr = np.where(np.isnan(fit_run), -np.inf, fit_run)
    best = np.argmax(fr, axis=-1)
    best_val = np.take_along_axis(fr, best[..., None], axis=-1)[..., 0]
    fr2 = fr.copy()
    np.put_along_axis(fr2, best[..., None], -np.inf, axis=-1)
    second = np.argmax(fr2, axis=-1)
    second_val = np.take_along_axis(fr2, second[..., None], axis=-1)[..., 0]
    ok = np.isfinite(best_val) & np.isfinite(second_val)
    margin = np.where(ok, best_val - second_val, np.nan).astype(np.float32)
    return best, second, margin


def export_paths(out_dir: Path, stem: str) -> dict[str, Path]:
    return {k: out_dir / f"{stem}_desc-{desc}_{suffix}.nii.gz" for k, (desc, suffix, _) in EXPORTS.items()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--reference", required=True,
                    help="NIfTI on the fit's grid, supplies shape + affine")
    ap.add_argument("--tree", default=None,
                    help="GLMsingle tree; default <output_dir>/glmsingle_tb")
    ap.add_argument("--space", default=SPACE)
    ap.add_argument("--mask", nargs="*", default=None,
                    help="brain mask NIfTI(s) on the fit grid, intersected; enables the masked FitHRFR2run export")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

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
    targets = export_paths(out_dir, f"{sub}_arm-{args.arm}_space-{args.space}")
    todo = {k: p for k, p in targets.items() if args.force or not p.exists()}
    npz_path = out_dir / f"{sub}_arm-{args.arm}_space-{args.space}_desc-fithrfr2run_masked.npz"
    need_npz = args.mask is not None and (args.force or not npz_path.exists())
    if not todo and not need_npz:
        print(f"{sub}/{args.arm}: all {len(targets)} exports exist" + (" (+ masked FitHRFR2run)" if args.mask else "")
              + "; --force to rewrite")
        return 0

    ref = nib.load(args.reference)
    mask = None
    if need_npz:
        for mp in args.mask:
            m = np.asarray(nib.load(mp).dataobj).astype(bool)
            if m.shape != tuple(ref.shape[:3]):
                sys.exit(f"ERROR: mask {mp} shape {m.shape} does not match the reference grid {ref.shape[:3]}")
            mask = m if mask is None else (mask & m)
        if not args.mask:
            sys.exit("ERROR: --mask given without any mask file")
    print(f"{sub}/{args.arm}: loading {src} ({src.stat().st_size / 1e9:.1f} GB pickle) for {sorted(todo)}")
    t0 = time.time()
    d = np.load(src, allow_pickle=True).item()
    print(f"  loaded in {time.time() - t0:.0f} s; keys {sorted(d)}")
    out_dir.mkdir(parents=True, exist_ok=True)
    if "HRFmarginrun" in todo or "HRFsecondrun" in todo or need_npz:
        fit_run = np.asarray(d["FitHRFR2run"])
        if fit_run.shape[:3] != tuple(ref.shape[:3]):
            sys.exit(f"ERROR: FitHRFR2run shape {fit_run.shape} does not match the reference grid {ref.shape[:3]}")
        if "HRFmarginrun" in todo or "HRFsecondrun" in todo:
            best, second, margin = per_run_margin(fit_run)
            d["HRFmarginrun"], d["HRFsecondrun"] = margin, second
            if "HRFindexrun" in d:
                mism = int(np.sum(best != np.asarray(d["HRFindexrun"]).astype(int)))
                print(f"  argmax(FitHRFR2run) vs HRFindexrun: {mism} voxel-runs differ"
                      + ("" if mism == 0 else " (NaN or tied R2; the margin is NaN or 0 there)"))
        if need_npz:
            n_runs, n_h = fit_run.shape[3], fit_run.shape[4]
            flat = np.flatnonzero(mask)
            labels = []
            meta = arm_dir / "run_metadata.json"
            if meta.exists():
                labels = json.loads(meta.read_text()).get("run_labels", [])
                if labels and len(labels) != n_runs:
                    sys.exit(f"ERROR: {meta} lists {len(labels)} runs but FitHRFR2run has {n_runs}")
            np.savez(npz_path, data=fit_run.reshape(-1, n_runs, n_h)[flat].astype(np.float32),
                     index=flat.astype(np.int64), shape=np.array(mask.shape), run_labels=np.array(labels))
            print(f"  FitHRFR2run masked ({flat.size} voxels x {n_runs} runs x {n_h} kernels) -> {npz_path}")
        del fit_run
    for key, path in todo.items():
        arr = np.asarray(d[key])
        if arr.shape[:3] != tuple(ref.shape[:3]):
            sys.exit(f"ERROR: {key} shape {arr.shape} does not match the reference "
                     f"grid {ref.shape[:3]}")
        nib.Nifti1Image(arr.astype(EXPORTS[key][2]), ref.affine).to_filename(str(path))
        print(f"  {key} {arr.shape} -> {path}")
    del d
    return 0


if __name__ == "__main__":
    sys.exit(main())
