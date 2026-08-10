#!/usr/bin/env python3
"""Phases 4/5 — GLMsingle ROI beta caches (pattern-similarity analysis).

Loads the TYPED (FITHRF + GLMdenoise + Ridge) single-trial betas once per
(subject, pipeline), slices them to the 6 bilateral Harvard-Oxford ROIs, and
writes per-ROI pattern matrices + trial/chunk index arrays:

  --paradigm TB  (Phase 4): canonical TBencoding fits under
      derivatives/glmsingle{,_nordic}/ — full-4D betasmd (X, Y, Z, 2562)
      + trial_info.csv -> session/run/mmmId indices.
  --paradigm NAT (Phase 5): NAT chunk fits under
      derivatives/glmsingle_nat{,_nordic}/ (scripts/glmsingle_natencoding.py)
      — masked-2D betasmd (V_brain, ...) + brain_mask_index.npy
      + chunk_info.csv -> session/run/movie_name/chunk_idx indices.
      (full_4d layout also handled, per run_metadata.json.)

Voxel sets are the fixed atlas-defined ROI voxels on the res-2 grid (same
order as the Phase 3 rawTR caches); ROI voxels without a beta (outside the
GLMsingle brain mask) are NaN, per the mutually-finite-voxel convention.

Output (float32 npz, all 6 ROIs per file):
  derivatives/pattern_similarity/cache/glmsingle/{pipeline}/{sub}/
    {sub}_task-{TBencoding|NATencoding}_desc-typed_roipatterns.npz

Plan: docs/doc/pattern-similarity-plan.md, Phases 4-5.

Usage:
    python extract_betas.py --paradigm TB  --sub sub-03 --pipeline original
    python extract_betas.py --paradigm NAT --sub sub-03 --pipeline nordic
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from shared import (  # noqa: E402
    CACHE_DIR, GLMSINGLE_NAT_DIRS, PATTERN_ROI_NAMES, PIPELINES, SUBJECTS,
    bold_path, glmsingle_outputs_dir, load_bilateral_roi_masks,
    load_trial_metadata, resample_masks_to_bold,
)

BETA_CACHE = CACHE_DIR / "glmsingle"
TYPED_NAME = "TYPED_FITHRF_GLMDENOISE_RR.npy"


def load_typed_betas(outputs_dir: Path) -> np.ndarray:
    """Load TYPED betasmd (layout-agnostic; caller interprets shape)."""
    f = outputs_dir / TYPED_NAME
    if not f.exists():
        raise FileNotFoundError(f"GLMsingle output missing: {f}")
    return np.load(str(f), allow_pickle=True).item()["betasmd"]


def roi_grid_masks(ref_img):
    """ROI masks + flat C-order voxel indices on the reference res-2 grid."""
    masks_atlas, atlas_affine = load_bilateral_roi_masks()
    masks = resample_masks_to_bold(masks_atlas, atlas_affine, ref_img)
    vox_idx = {roi: np.flatnonzero(m.ravel(order="C"))
               for roi, m in masks.items()}
    return masks, vox_idx


def extract_tb(sub: str, pipeline: str) -> dict:
    outputs_dir = glmsingle_outputs_dir(sub, pipeline)
    print(f"Loading TB TYPED betas: {outputs_dir}")
    t0 = time.time()
    betas = load_typed_betas(outputs_dir)  # (X, Y, Z, n_trials)
    print(f"  {betas.shape} ({betas.nbytes / 1e9:.1f} GB) "
          f"in {time.time() - t0:.0f}s")

    meta = load_trial_metadata(sub, pipeline)
    assert len(meta) == betas.shape[-1], (
        f"trial_info rows ({len(meta)}) != n_trials ({betas.shape[-1]})"
    )

    ref_ses, ref_run = meta.iloc[0][["session", "run"]]
    ref_img = nib.load(str(bold_path(sub, ref_ses, int(ref_run),
                                     "TBencoding", pipeline)))
    assert ref_img.shape[:3] == betas.shape[:3], "betas grid != BOLD grid"
    masks, vox_idx = roi_grid_masks(ref_img)

    out = {
        "session": meta["session"].to_numpy(),
        "run": meta["run"].to_numpy(),
        "mmmId": meta["mmm_id"].astype(str).to_numpy(),
    }
    for roi in PATTERN_ROI_NAMES:
        out[f"patterns_{roi}"] = betas[masks[roi]].astype(np.float32)
        out[f"voxidx_{roi}"] = vox_idx[roi]
    return out


def extract_nat(sub: str, pipeline: str) -> dict:
    nat_dir = GLMSINGLE_NAT_DIRS[pipeline] / sub
    with open(nat_dir / "run_metadata.json") as fh:
        run_meta = json.load(fh)
    layout = run_meta["data_layout"]
    print(f"Loading NAT TYPED betas: {nat_dir} (layout: {layout})")
    t0 = time.time()
    betas = load_typed_betas(nat_dir / "glmsingle_outputs")
    print(f"  {betas.shape} ({betas.nbytes / 1e9:.1f} GB) "
          f"in {time.time() - t0:.0f}s")

    chunk_info = pd.read_csv(nat_dir / "chunk_info.csv")
    n_chunks = betas.shape[-1]
    assert len(chunk_info) == n_chunks, (
        f"chunk_info rows ({len(chunk_info)}) != n betas ({n_chunks})"
    )

    mask_img = nib.load(str(nat_dir / "brain_mask.nii.gz"))
    masks, vox_idx = roi_grid_masks(mask_img)

    if layout == "masked_2d":
        # (V_brain, [1, 1,] n) -> (V_brain, n); scatter into fixed ROI voxel sets
        betas2d = betas.reshape(betas.shape[0], n_chunks)
        mask_idx = np.load(nat_dir / "brain_mask_index.npy")
        assert betas2d.shape[0] == len(mask_idx), (
            f"betas voxels ({betas2d.shape[0]}) != mask index ({len(mask_idx)})"
        )

        def roi_patterns(roi):
            ridx = vox_idx[roi]
            pos = np.searchsorted(mask_idx, ridx)
            pos_c = np.clip(pos, 0, len(mask_idx) - 1)
            present = mask_idx[pos_c] == ridx
            pat = np.full((len(ridx), n_chunks), np.nan, dtype=np.float32)
            pat[present] = betas2d[pos_c[present]].astype(np.float32)
            return pat
    else:
        assert betas.ndim == 4 and betas.shape[:3] == mask_img.shape, (
            f"full_4d betas grid {betas.shape[:3]} != mask grid {mask_img.shape}"
        )

        def roi_patterns(roi):
            return betas[masks[roi]].astype(np.float32)

    out = {
        "session": chunk_info["session"].to_numpy(),
        "run": (chunk_info["run"].astype(str)
                .str.replace("run-", "", regex=False).astype(int).to_numpy()),
        "movie_name": chunk_info["movie_name"].to_numpy(),
        "chunk_idx": chunk_info["chunk_idx"].to_numpy(),
        "col_index": chunk_info["col_index"].to_numpy(),
        "onset_s": chunk_info["onset"].to_numpy(),
    }
    for roi in PATTERN_ROI_NAMES:
        out[f"patterns_{roi}"] = roi_patterns(roi)
        out[f"voxidx_{roi}"] = vox_idx[roi]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paradigm", required=True, choices=["TB", "NAT"])
    ap.add_argument("--sub", required=True, choices=SUBJECTS)
    ap.add_argument("--pipeline", required=True, choices=sorted(PIPELINES))
    args = ap.parse_args()

    task = {"TB": "TBencoding", "NAT": "NATencoding"}[args.paradigm]
    print(f"=== GLMsingle ROI cache: {args.sub} / {args.pipeline} / {task} ===")
    arrays = (extract_tb if args.paradigm == "TB" else extract_nat)(
        args.sub, args.pipeline)
    arrays["roi_names"] = np.array(PATTERN_ROI_NAMES)

    out_dir = BETA_CACHE / args.pipeline / args.sub
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{args.sub}_task-{task}_desc-typed_roipatterns.npz"
    np.savez_compressed(p, **arrays)

    n_cols = arrays[f"patterns_{PATTERN_ROI_NAMES[0]}"].shape[1]
    counts = {roi: arrays[f"patterns_{roi}"].shape[0]
              for roi in PATTERN_ROI_NAMES}
    print(f"Wrote {p}\n  {n_cols} patterns/ROI; voxel counts: {counts}")
    print("\nDone.")


if __name__ == "__main__":
    main()
