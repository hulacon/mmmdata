#!/usr/bin/env python3
"""TB benchmark evaluation: within-stim vs between-stim β-pattern correlations.

For each (subject, pipeline), reads the canonical multi-session GLMsingle TYPED
output (`derivatives/glmsingle{,_nordic}/<sub>/glmsingle_outputs/TYPED_*.npy`)
plus its sibling `trial_info.csv`, and computes per-ROI per-session
within/between cross-run trial-pair correlations.

Pair selection per the locked design (docs/nordic-benchmark-plan.md):
  - within-stim pair: same mmm_id, **different runs**, same session
  - between-stim pair: different mmm_id, **different runs**, same session
  - same-run pairs (within or between stim): excluded
  - cross-session pairs: excluded

Output:
  derivatives/nordic/benchmark/tb/{sub}/{pipeline}/pair_correlations.tsv
    columns: subject, session, stream=TB, roi, pipeline,
             within_r, between_r, discriminability, n_within, n_between
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from shared import (  # noqa: E402
    BENCHMARK_OUT, MNI_SPACE, ROI_KEYS, SUMMARY_COLS, TB_SESSIONS,
    bold_path, glmsingle_outputs_dir, load_roi_masks, load_trial_metadata,
    resample_masks_to_bold,
)


def load_glmsingle_betas(glmsingle_dir: Path):
    """Load the TYPED (FITHRF + GLMdenoise + Ridge) β maps."""
    f = glmsingle_dir / "TYPED_FITHRF_GLMDENOISE_RR.npy"
    if not f.exists():
        raise FileNotFoundError(f"GLMsingle output missing: {f}")
    data = np.load(str(f), allow_pickle=True).item()
    betas = data["betasmd"]  # (X, Y, Z, n_trials)
    return betas


def session_pair_stats(roi_betas: np.ndarray,
                       runs: np.ndarray,
                       mmm_ids: np.ndarray) -> tuple[float, float, int, int]:
    """For one ROI × one session, compute mean within/between r and pair counts.

    Args:
        roi_betas: (n_voxels, n_trials_session)
        runs:      (n_trials_session,) integer run number per trial
        mmm_ids:   (n_trials_session,) string mmm_id per trial

    Returns: (within_r, between_r, n_within, n_between)
    """
    n = roi_betas.shape[1]
    if n < 2:
        return (np.nan, np.nan, 0, 0)
    # Drop voxels with any NaN across trials in this session
    finite_voxels = np.all(np.isfinite(roi_betas), axis=1)
    if not finite_voxels.any():
        return (np.nan, np.nan, 0, 0)
    X = roi_betas[finite_voxels]  # (V', n)
    # Pearson correlation across trials' β patterns
    corr = np.corrcoef(X.T)        # (n, n) symmetric

    # Pair masks (upper triangle to avoid double-counting)
    iu, ju = np.triu_indices(n, k=1)
    same_run = runs[iu] == runs[ju]
    same_stim = mmm_ids[iu] == mmm_ids[ju]
    within_mask = same_stim & ~same_run
    between_mask = ~same_stim & ~same_run
    rs = corr[iu, ju]

    within_r = float(rs[within_mask].mean()) if within_mask.any() else np.nan
    between_r = float(rs[between_mask].mean()) if between_mask.any() else np.nan
    return (within_r, between_r, int(within_mask.sum()), int(between_mask.sum()))


def evaluate(sub: str, pipeline: str, output_root: Path) -> Path:
    out_dir = output_root / "tb" / sub / pipeline
    glmsingle_dir = glmsingle_outputs_dir(sub, pipeline)
    pair_out = out_dir / "pair_correlations.tsv"

    print(f"=== TB eval: {sub} / {pipeline} ===")
    print(f"  betas:    {glmsingle_dir}")
    t0 = time.time()
    betas = load_glmsingle_betas(glmsingle_dir)
    print(f"Loaded betas {betas.shape} ({betas.nbytes/1e9:.1f} GB) in {time.time()-t0:.0f}s")

    metadata = load_trial_metadata(sub, pipeline)
    if len(metadata) != betas.shape[3]:
        raise ValueError(f"metadata rows ({len(metadata)}) != n_trials in betas ({betas.shape[3]})")

    # Use the first BOLD in this pipeline as the reference grid for ROI resampling
    ref_ses, ref_run = metadata.iloc[0][["session", "run"]]
    ref_bold = nib.load(str(bold_path(sub, ref_ses, int(ref_run), "TBencoding", pipeline)))
    masks_atlas, atlas_affine = load_roi_masks()
    masks = resample_masks_to_bold(masks_atlas, atlas_affine, ref_bold)
    print(f"ROI masks resampled to BOLD grid {betas.shape[:3]}")

    # Sanity: mask shapes match betas spatial dims
    for k in ROI_KEYS:
        assert masks[k].shape == betas.shape[:3], f"{k} mask {masks[k].shape} != betas {betas.shape[:3]}"

    rows = []
    for ses in TB_SESSIONS:
        ses_idx = metadata.index[metadata["session"] == ses].to_numpy()
        if len(ses_idx) == 0:
            continue
        ses_runs = metadata.loc[ses_idx, "run"].to_numpy()
        ses_mmm = metadata.loc[ses_idx, "mmm_id"].astype(str).to_numpy()
        # ses_betas: (X, Y, Z, n_trials_session)
        ses_betas = betas[..., ses_idx]
        for roi, hemi in ROI_KEYS:
            mask = masks[(roi, hemi)]
            roi_betas = ses_betas[mask, :]   # (V_roi, n_trials_session)
            within_r, between_r, n_within, n_between = session_pair_stats(
                roi_betas, ses_runs, ses_mmm,
            )
            disc = within_r - between_r if np.isfinite(within_r) and np.isfinite(between_r) else np.nan
            rows.append({
                "subject": sub,
                "session": ses,
                "stream": "TB",
                "roi": f"{roi}_{hemi}",
                "pipeline": pipeline,
                "within_r": within_r,
                "between_r": between_r,
                "discriminability": disc,
                "n_within": n_within,
                "n_between": n_between,
            })

    pair_out.parent.mkdir(parents=True, exist_ok=True)
    with open(pair_out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=SUMMARY_COLS, delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {pair_out} ({len(rows)} rows)")

    # Quick aggregate summary printed for sanity
    df = pd.DataFrame(rows)
    print("\nMean discriminability per ROI (across 14 sessions):")
    print(df.groupby("roi")["discriminability"].agg(["mean", "std", "count"]).round(4))
    return pair_out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sub", required=True, choices=["sub-03", "sub-04", "sub-05"])
    ap.add_argument("--pipeline", required=True, choices=["original", "nordic"])
    ap.add_argument("--output-root", type=Path, default=BENCHMARK_OUT)
    args = ap.parse_args()

    evaluate(args.sub, args.pipeline, args.output_root)
    print("\nDone.")


if __name__ == "__main__":
    main()
