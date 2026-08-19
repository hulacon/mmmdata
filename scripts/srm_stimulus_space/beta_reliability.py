#!/usr/bin/env python3
"""Split-half voxel reliability of TB GLMsingle TYPED betas, per subject.

Diagnostic for the srm-stimulus-space workbench (sub-05's TB-glmsingle
weak spot): for every item with >=2 presentations, take its first and
second presentation betas (chronological); voxel reliability = Pearson r
across items between the two presentation vectors; report the median over
valid ROI voxels. Betas are session-z-scored per voxel first (the A1
convention; --raw skips it). Benchmarks for interpretation live in the
agents memory note 'GLMsingle reliability benchmarks' (NSD 7T ~0.15-0.25
median TypeD in visual cortex; MMMData sub-03 V1 ~0.12 measured 2026-04).

Usage:
    python beta_reliability.py [--pipeline original|nordic|both] [--raw]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from a1_encoding import ROI_NAMES, SUBJECTS, _session_zscore, beta_cache


def voxel_reliability(sub: str, pipeline: str, zscore: bool) -> dict:
    f = (beta_cache(pipeline) / sub /
         f"{sub}_task-TBencoding_desc-typed_roipatterns.npz")
    d = np.load(f, allow_pickle=True)
    mmm = d["mmmId"].astype(str)
    session = d["session"].astype(str)
    # first/second presentation column index per item, chronological
    order = np.lexsort((d["run"].astype(int), session))
    first, second = {}, {}
    for col in order:
        item = mmm[col]
        if item not in first:
            first[item] = col
        elif item not in second:
            second[item] = col
    items = sorted(second)
    i1 = np.array([first[i] for i in items])
    i2 = np.array([second[i] for i in items])

    out = {"subject": sub, "pipeline": pipeline, "n_items": len(items)}
    for roi in ROI_NAMES:
        pat = d[f"patterns_{roi}"].astype(np.float64)
        if zscore:
            pat = _session_zscore(pat, session)
        a, b = pat[:, i1], pat[:, i2]                    # (V, n_items)
        valid = np.isfinite(a).all(1) & np.isfinite(b).all(1) & \
            (a.std(1) > 0) & (b.std(1) > 0)
        ac = a[valid] - a[valid].mean(1, keepdims=True)
        bc = b[valid] - b[valid].mean(1, keepdims=True)
        r = (ac * bc).sum(1) / np.sqrt((ac**2).sum(1) * (bc**2).sum(1))
        out[roi] = float(np.median(r))
        out[f"{roi}_p90"] = float(np.percentile(r, 90))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pipeline", choices=["original", "nordic", "both"],
                    default="both")
    ap.add_argument("--raw", action="store_true",
                    help="skip session z-scoring")
    args = ap.parse_args()

    pipelines = (["original", "nordic"] if args.pipeline == "both"
                 else [args.pipeline])
    rows = [voxel_reliability(sub, pipe, not args.raw)
            for pipe in pipelines for sub in SUBJECTS]
    df = pd.DataFrame(rows)
    label = "raw" if args.raw else "session-z"
    print(f"Median split-half voxel reliability, TYPED TB betas ({label}):")
    cols = ["subject", "pipeline", "n_items"] + ROI_NAMES
    print(df[cols].round(4).to_string(index=False))
    print("\n90th-percentile voxel reliability:")
    print(df[["subject", "pipeline"]
             + [f"{r}_p90" for r in ROI_NAMES]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
