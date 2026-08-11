#!/usr/bin/env python3
"""Phase 2 verification — NAT GLMsingle fits (pattern-similarity analysis).

Checks, per (subject, pipeline) fit under derivatives/glmsingle_nat{,_nordic}/:
  1. condition_key: exactly 115 repeated conditions (n_presentations == 10),
     all belonging to the two repeated movies (52 + 63 chunks); every other
     condition is one-shot.
  2. chunk_info rows == sum of condition_key n_presentations, and per-condition
     presentation counts in chunk_info match condition_key exactly.
  3. chunk_info identical across the two pipelines (per subject).
  4. TYPED betas trailing dim == chunk_info rows (loads the 4.5 GB file;
     skip with --skip-betas).
  5. Mean split-half reliability of repeated-movie chunk betas in EVC > 0
     (first-5 vs last-5 presentations, per-voxel r across the 115 conditions).

Writes derivatives/pattern_similarity/qc/phase2_natfit_verification.tsv.

Plan: docs/doc/pattern-similarity-plan.md, Phase 2 "Verify".

Usage:
    python verify_natfits.py [--skip-betas] [--subjects sub-03 ...]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from shared import (  # noqa: E402
    GLMSINGLE_NAT_DIRS, QC_DIR, REPEATED_MOVIES, SUBJECTS,
)
from extract_betas import extract_nat  # noqa: E402

EXPECTED_REPEATED_CHUNKS = {"The Bench": 52, "From Dad To Son": 63}


def check(rows, name, ok, detail=""):
    rows.append({"check": name, "status": "PASS" if ok else "FAIL",
                 "detail": detail})
    print(f"  [{'OK' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    return ok


def split_half_reliability(patterns, col_index, repeated_cols):
    """Per-voxel r between first-5 and last-5 presentation means, across conds."""
    half_a, half_b = [], []
    for col in repeated_cols:
        idx = np.flatnonzero(col_index == col)  # chunk_info is session-ordered
        half_a.append(patterns[:, idx[:5]].mean(axis=1))
        half_b.append(patterns[:, idx[5:]].mean(axis=1))
    a = np.column_stack(half_a)  # (V, n_conds)
    b = np.column_stack(half_b)
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    denom = np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (a * b).sum(axis=1) / denom
    return r


def verify_fit(sub, pipeline, rows, skip_betas):
    nat_dir = GLMSINGLE_NAT_DIRS[pipeline] / sub
    tag = f"{sub}/{pipeline}"
    key = pd.read_csv(nat_dir / "condition_key.csv")
    info = pd.read_csv(nat_dir / "chunk_info.csv")

    rep = key[key["n_presentations"] == 10]
    rep_by_movie = rep.groupby("movie_name")["chunk_idx"].size().to_dict()
    check(rows, f"{tag}: 115 repeated conditions x10", len(rep) == 115,
          f"n={len(rep)}")
    check(rows, f"{tag}: repeated chunks per movie",
          rep_by_movie == EXPECTED_REPEATED_CHUNKS, str(rep_by_movie))
    check(rows, f"{tag}: non-repeated are one-shot",
          (key.loc[key["n_presentations"] != 10, "n_presentations"] == 1).all(),
          f"n_conditions={len(key)}")

    counts = info.groupby("col_index").size()
    key_counts = key.set_index("col_index")["n_presentations"]
    check(rows, f"{tag}: chunk_info rows == sum presentations",
          len(info) == key["n_presentations"].sum(), f"rows={len(info)}")
    check(rows, f"{tag}: per-condition counts match condition_key",
          counts.sort_index().equals(key_counts.sort_index()))

    if skip_betas:
        return None
    arrays = extract_nat(sub, pipeline)  # asserts betas trailing dim == rows
    check(rows, f"{tag}: betas trailing dim == chunk_info rows", True,
          f"n={len(info)}")
    r = split_half_reliability(arrays["patterns_EVC"], arrays["col_index"],
                               rep["col_index"].to_numpy())
    mean_r = float(np.nanmean(r))
    check(rows, f"{tag}: EVC split-half reliability > 0", mean_r > 0,
          f"mean r={mean_r:.3f} over {np.isfinite(r).sum()} voxels")
    return mean_r


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-betas", action="store_true",
                    help="metadata checks only (no 4.5 GB beta loads)")
    ap.add_argument("--subjects", nargs="+", default=SUBJECTS)
    args = ap.parse_args()

    rows = []
    for sub in args.subjects:
        print(f"=== {sub} ===")
        for pipeline in sorted(GLMSINGLE_NAT_DIRS):
            verify_fit(sub, pipeline, rows, args.skip_betas)
        infos = [pd.read_csv(GLMSINGLE_NAT_DIRS[p] / sub / "chunk_info.csv")
                 for p in sorted(GLMSINGLE_NAT_DIRS)]
        check(rows, f"{sub}: chunk_info identical across pipelines",
              infos[0].equals(infos[1]))

    out = QC_DIR / "phase2_natfit_verification.tsv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, sep="\t", index=False)
    n_fail = sum(r["status"] == "FAIL" for r in rows)
    print(f"\n{len(rows)} checks, {n_fail} failures -> {out}")
    if n_fail:
        sys.exit(1)
    print("Verification PASSED")


if __name__ == "__main__":
    main()
