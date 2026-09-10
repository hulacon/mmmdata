#!/usr/bin/env python3
"""ROI pattern caches for the TB single-trial fits (encoding + retrieval).

Slices each GLMsingle beta type (TYPEB / TYPEC / TYPED) of one
``derivatives/glmsingle_tb/<sub>/<arm>/`` fit to the six bilateral
Harvard-Oxford ROIs of the pattern-similarity benchmarks and writes one npz per
beta type in the ``extract_betas.py`` cache format, so encoding and retrieval
betas are sliceable by the same code:

  derivatives/pattern_similarity/cache/glmsingle_tb/<sub>/<arm>/
      <sub>_arm-<arm>_desc-type{b,c,d}_roipatterns.npz

Arrays per file:
  patterns_<ROI>   (V_roi, N_trials) float32 — NaN where the ROI voxel lies
                   outside the GLMsingle brain mask (mutually-finite convention)
  voxidx_<ROI>     flat C-order voxel indices on the res-2 grid (fixed per ROI)
  R2_<ROI>, HRFindex_<ROI>, meanvol_<ROI>   per-voxel fit diagnostics of that
                   beta type; meanvol is the run-mean BOLD GLMsingle divided by
                   to express betas in percent signal change, so a near-zero
                   meanvol marks an edge/air voxel whose betas are unbounded
  one entry per trial column: trial_index, session, run (int), task, subgroup,
      mmmId (str), condition_id, col_index, onset, duration, word, pairId,
      sharedId, enCon, reCon, resp, resp_RT
  scalars: subject, arm, beta_type, roi_names, grid_shape, affine, source_dir

Column order is ``trial_info.csv`` row order: GLMsingle's ``betasmd`` has one
column PER TRIAL (not per condition — ``col_index`` addresses the design
matrix only). Matching between arms is on (session, run, onset).

Design record: mmmdata-agents docs/workbench/retrieval-modeling/ (2026-09-10).

Usage:
    python extract_roi_betas.py --subject sub-## --arm pooled --dry-run
    python extract_roi_betas.py --subject sub-## --arm ret-image
    python extract_roi_betas.py --subject sub-## --arm enc --types D
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import sys
import time
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


# The runner (config + path helpers + the output tree name) and the
# pattern-similarity ROI definitions are the two single sources reused here.
tb = _load_module("glmsingle_tb", SCRIPTS / "glmsingle_tb.py")
ps = _load_module("pattern_similarity_shared",
                  SCRIPTS / "pattern_similarity" / "shared.py")

BETA_FILES = {
    "B": "TYPEB_FITHRF.npy",
    "C": "TYPEC_FITHRF_GLMDENOISE.npy",
    "D": "TYPED_FITHRF_GLMDENOISE_RR.npy",
}
CACHE_TREE = "pattern_similarity"          # derivatives/<CACHE_TREE>/cache/...
TRIAL_COLS = ["session", "run", "task", "subgroup", "mmmId", "condition_id",
              "col_index", "onset", "duration", "word", "pairId", "sharedId",
              "enCon", "reCon", "resp", "resp_RT"]
ANCHORS = [str(i) for i in range(995, 1001)]   # the 6 super-repeat items


def ensure_cache_dataset_description(cache_root: Path, fit_root: Path,
                                     fmriprep_dir: Path) -> None:
    """Make the recreated cache tree catalog-legible on first touch."""
    dd = cache_root / "dataset_description.json"
    if dd.exists():
        return
    cache_root.mkdir(parents=True, exist_ok=True)
    with open(dd, "w") as f:
        json.dump({
            "Name": "Pattern-similarity ROI caches (GLMsingle TB betas, "
                    "Harvard-Oxford bilateral ROIs)",
            "BIDSVersion": "1.8.0",
            "DatasetType": "derivative",
            "GeneratedBy": [{
                "Name": "extract_roi_betas.py",
                "Description": "mmmdata/scripts/retrieval_modeling/"
                               "extract_roi_betas.py; design record in "
                               "mmmdata-agents docs/workbench/retrieval-modeling/",
            }],
            "SourceDatasets": [{"URL": str(fit_root)}, {"URL": str(fmriprep_dir)}],
        }, f, indent=2)
    print(f"Wrote {dd}")


def load_trial_info(fit_dir: Path) -> pd.DataFrame:
    ti = pd.read_csv(fit_dir / "trial_info.csv")
    missing = [c for c in TRIAL_COLS if c not in ti.columns]
    if missing:
        sys.exit(f"ERROR: {fit_dir / 'trial_info.csv'} lacks columns {missing}")
    ti["run_int"] = ti["run"].astype(str).str.replace("run-", "", regex=False).astype(int)
    ti["mmmId"] = ti["mmmId"].map(tb.norm_mmm)
    return ti


def reference_bold(fmriprep_dir: Path, subject: str, ti: pd.DataFrame):
    """The fMRIPrep BOLD of the first trial's run: the grid the fit ran on."""
    r = ti.iloc[0]
    p = tb.bold_path(fmriprep_dir, subject, r["session"], r["task"], r["run"])
    if not p.exists():
        sys.exit(f"ERROR: reference BOLD missing: {p}")
    return nib.load(str(p))


def roi_masks_on_grid(ref_img):
    masks_atlas, atlas_affine = ps.load_bilateral_roi_masks()
    masks = ps.resample_masks_to_bold(masks_atlas, atlas_affine, ref_img)
    vox_idx = {roi: np.flatnonzero(m.ravel(order="C")) for roi, m in masks.items()}
    return masks, vox_idx


def load_beta_dict(fit_dir: Path, beta_type: str) -> dict:
    f = fit_dir / "glmsingle_outputs" / BETA_FILES[beta_type]
    if not f.exists():
        sys.exit(f"ERROR: GLMsingle output missing: {f}")
    t0 = time.time()
    d = np.load(str(f), allow_pickle=True).item()
    print(f"  loaded {f.name} in {time.time() - t0:.0f}s; keys={sorted(d.keys())}",
          flush=True)
    return d


def slice_type(d: dict, beta_type: str, masks: dict, vox_idx: dict,
               ti: pd.DataFrame, ref_img, subject: str, arm: str,
               fit_dir: Path) -> dict:
    betas = np.asarray(d["betasmd"])
    if betas.ndim != 4:
        sys.exit(f"ERROR: expected 4-D betasmd, got shape {betas.shape}")
    if betas.shape[:3] != ref_img.shape[:3]:
        sys.exit(f"ERROR: betas grid {betas.shape[:3]} != BOLD grid {ref_img.shape[:3]}")
    if betas.shape[-1] != len(ti):
        sys.exit(f"ERROR: betasmd has {betas.shape[-1]} columns but trial_info "
                 f"has {len(ti)} rows — per-trial layout violated")

    out = {
        "subject": subject, "arm": arm, "beta_type": f"TYPE{beta_type}",
        "roi_names": np.array(ps.PATTERN_ROI_NAMES),
        "grid_shape": np.array(betas.shape[:3]), "affine": ref_img.affine,
        "source_dir": str(fit_dir),
        "trial_index": np.arange(len(ti)),
        "run": ti["run_int"].to_numpy(),
    }
    for c in TRIAL_COLS:
        if c == "run":
            continue
        v = ti[c]
        out[c] = (v.astype(str).to_numpy() if v.dtype == object or c in ("mmmId", "condition_id")
                  else v.to_numpy())

    r2 = np.asarray(d["R2"], dtype=np.float32).reshape(-1) if "R2" in d else None
    hrf = np.asarray(d["HRFindex"]).reshape(-1) if "HRFindex" in d else None
    mv = np.asarray(d["meanvol"], dtype=np.float32).reshape(-1) if "meanvol" in d else None
    for roi in ps.PATTERN_ROI_NAMES:
        pat = betas[masks[roi]].astype(np.float32)
        out[f"patterns_{roi}"] = pat
        out[f"voxidx_{roi}"] = vox_idx[roi]
        if r2 is not None:
            out[f"R2_{roi}"] = r2[vox_idx[roi]]
        if hrf is not None:
            out[f"HRFindex_{roi}"] = hrf[vox_idx[roi]]
        if mv is not None:
            out[f"meanvol_{roi}"] = mv[vox_idx[roi]]
    return out


def report(out: dict, ti: pd.DataFrame) -> None:
    n = len(ti)
    print(f"  {n} trial columns; subgroups {ti['subgroup'].value_counts().to_dict()}")
    anchors = ti["mmmId"].isin(ANCHORS).sum()
    print(f"  anchor (super-repeat) trials: {anchors}")
    for roi in ps.PATTERN_ROI_NAMES:
        pat = out[f"patterns_{roi}"]
        finite_vox = np.isfinite(pat).all(axis=1).sum()
        print(f"  {roi:12s} V={pat.shape[0]:5d}  finite-in-all-trials={finite_vox:5d}  "
              f"beta sd(median over trials)={np.nanmedian(np.nanstd(pat, axis=0)):.3g}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject", required=True, help="e.g. sub-##")
    ap.add_argument("--arm", required=True, choices=sorted(tb.ARM_SPECS))
    ap.add_argument("--types", nargs="+", default=["B", "C", "D"],
                    choices=sorted(BETA_FILES), help="beta types to cache")
    ap.add_argument("--fmriprep-dir", default=None)
    ap.add_argument("--fit-root", default=None,
                    help=f"Override derivatives/{tb.OUTPUT_TREE}")
    ap.add_argument("--cache-root", default=None,
                    help=f"Override derivatives/{CACHE_TREE}")
    ap.add_argument("--dry-run", action="store_true",
                    help="Resolve inputs, build ROI masks, report; load no betas")
    args = ap.parse_args()

    cfg = tb.load_config()
    bids_root = Path(cfg["bids_project_dir"])
    fmriprep_dir = Path(args.fmriprep_dir) if args.fmriprep_dir else bids_root / "derivatives" / "fmriprep"
    fit_root = Path(args.fit_root) if args.fit_root else bids_root / "derivatives" / tb.OUTPUT_TREE
    cache_root = Path(args.cache_root) if args.cache_root else bids_root / "derivatives" / CACHE_TREE
    fit_dir = fit_root / args.subject / args.arm
    out_dir = cache_root / "cache" / "glmsingle_tb" / args.subject / args.arm

    print(f"=== ROI cache: {args.subject} / {args.arm} / types {args.types} ===")
    print(f"fit:    {fit_dir}\ncache:  {out_dir}")
    if not (fit_dir / "trial_info.csv").exists():
        sys.exit(f"ERROR: no fit at {fit_dir} (trial_info.csv missing)")

    ti = load_trial_info(fit_dir)
    ref_img = reference_bold(fmriprep_dir, args.subject, ti)
    masks, vox_idx = roi_masks_on_grid(ref_img)
    print(f"grid {ref_img.shape[:3]}; ROI voxels "
          f"{ {roi: int(len(v)) for roi, v in vox_idx.items()} }")
    print(f"{len(ti)} trials; subgroups {ti['subgroup'].value_counts().to_dict()}; "
          f"sessions {ti['session'].nunique()}; runs {ti.groupby(['session', 'task', 'run']).ngroups}")

    if args.dry_run:
        print("DRY RUN — no betas loaded, nothing written.")
        return

    ensure_cache_dataset_description(cache_root, fit_root, fmriprep_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for t in args.types:
        print(f"\n[TYPE{t}]", flush=True)
        d = load_beta_dict(fit_dir, t)
        out = slice_type(d, t, masks, vox_idx, ti, ref_img, args.subject, args.arm, fit_dir)
        del d
        gc.collect()
        report(out, ti)
        p = out_dir / f"{args.subject}_arm-{args.arm}_desc-type{t.lower()}_roipatterns.npz"
        np.savez_compressed(p, **out)
        print(f"  wrote {p} ({p.stat().st_size / 1e6:.0f} MB)", flush=True)
        del out
        gc.collect()
    print("\nDone.")


if __name__ == "__main__":
    main()
