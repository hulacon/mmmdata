#!/usr/bin/env python3
"""Run a single multi-session GLMsingle fit for one (subject, pipeline) over
all 42 TBencoding runs. Output: per-trial β maps in MNI res-2.

Usage:
    python tb_glmsingle_fit.py --sub sub-03 --pipeline original
    python tb_glmsingle_fit.py --sub sub-03 --pipeline original --dry-run

Design matrix per run: 210 TRs × 1000 conditions (NSD shared1000 union).
sessionindicator: 1×42, 1-indexed by session (1=ses-04, ..., 14=ses-17).

Confound regression: none, per GLMsingle authors' recommendation
(GLMdenoise learns noise regressors from data).
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from shared import (  # noqa: E402
    BENCHMARK_OUT, N_VOLS_TB, STIMDUR, TB_SESSIONS, TR,
    bold_path, list_tb_runs, read_tb_trials,
)


def build_designs_and_metadata(sub: str, pipeline: str):
    """Build 42 per-run design matrices over union stim set + trial metadata.

    Returns:
        designs: list of 42 ndarrays, each (210, n_union_stim), dtype float32
        sessionindicator: ndarray of shape (1, 42), int, 1-indexed by session
        trial_metadata: list of dicts with subject/session/run/trial_idx_in_fit/mmm_id/onset_s
        union_stim: sorted list of mmm_id strings (column order in designs)
    """
    runs = list_tb_runs(sub, pipeline)
    assert len(runs) == 42, f"expected 42 TB runs for {sub}/{pipeline}, got {len(runs)}"

    # First pass: collect union of mmm_ids
    union = set()
    per_run_trials = []
    for ses, r in runs:
        trials = read_tb_trials(sub, ses, r)
        per_run_trials.append(trials)
        union.update(t.mmm_id for t in trials)
    union_stim = sorted(union, key=lambda s: int(s) if s.isdigit() else s)
    stim_to_col = {s: i for i, s in enumerate(union_stim)}
    n_cond = len(union_stim)

    # Second pass: build designs + chronological trial metadata
    designs = []
    sessionindicator = np.zeros((1, len(runs)), dtype=int)
    trial_metadata = []
    trial_idx = 0
    for run_i, ((ses, r), trials) in enumerate(zip(runs, per_run_trials)):
        sessionindicator[0, run_i] = TB_SESSIONS.index(ses) + 1  # 1..14

        D = np.zeros((N_VOLS_TB, n_cond), dtype=np.float32)
        for t in trials:
            if t.onset_tr >= N_VOLS_TB:
                raise ValueError(f"trial onset_tr {t.onset_tr} >= {N_VOLS_TB} for {sub}/{ses}/run-{r}")
            D[t.onset_tr, stim_to_col[t.mmm_id]] = 1.0
            trial_metadata.append({
                "subject": sub,
                "session": ses,
                "run": r,
                "trial_idx_in_fit": trial_idx,
                "mmm_id": t.mmm_id,
                "onset_s": t.onset_s,
                "onset_tr": t.onset_tr,
                "session_ordinal": TB_SESSIONS.index(ses) + 1,
            })
            trial_idx += 1
        designs.append(D)

    return designs, sessionindicator, trial_metadata, union_stim


def write_trial_metadata(metadata: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["subject", "session", "run", "trial_idx_in_fit", "mmm_id",
            "onset_s", "onset_tr", "session_ordinal"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, delimiter="\t")
        w.writeheader()
        w.writerows(metadata)


def load_bold_runs(sub: str, pipeline: str, runs: list[tuple[str, int]]) -> list[np.ndarray]:
    """Load all BOLDs into memory as float32. ~36 GB for 42 runs in MNI res-2."""
    data = []
    for ses, r in runs:
        p = bold_path(sub, ses, r, "TBencoding", pipeline)
        img = nib.load(str(p))
        arr = np.asarray(img.dataobj, dtype=np.float32)
        if arr.shape[3] != N_VOLS_TB:
            raise ValueError(f"{p.name}: expected {N_VOLS_TB} vols, got {arr.shape[3]}")
        data.append(arr)
    return data


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sub", required=True, choices=["sub-03", "sub-04", "sub-05"])
    ap.add_argument("--pipeline", required=True, choices=["original", "nordic"])
    ap.add_argument("--dry-run", action="store_true",
                    help="Build designs + metadata but skip BOLD load and fit.")
    ap.add_argument("--output-root", type=Path, default=BENCHMARK_OUT,
                    help=f"Output root (default: {BENCHMARK_OUT})")
    args = ap.parse_args()

    out_dir = args.output_root / "tb" / args.sub / args.pipeline
    glmsingle_dir = out_dir / "glmsingle"
    metadata_path = out_dir / "trial_metadata.tsv"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== TB GLMsingle fit: {args.sub} / {args.pipeline} ===")
    print(f"Output dir: {out_dir}")

    t0 = time.time()
    designs, sessionindicator, trial_metadata, union_stim = build_designs_and_metadata(
        args.sub, args.pipeline,
    )
    print(f"Designs built in {time.time()-t0:.1f}s")
    print(f"  n_runs           = {len(designs)}")
    print(f"  design[0].shape  = {designs[0].shape}  (TRs × union_stim)")
    print(f"  union_stim count = {len(union_stim)}")
    print(f"  total trials     = {len(trial_metadata)}")
    print(f"  sessionindicator = {sessionindicator.tolist()[0]}")

    # Sanity: every design has the same column count, binary, exactly 61 ones per run
    n_cond = designs[0].shape[1]
    for i, D in enumerate(designs):
        assert D.shape == (N_VOLS_TB, n_cond), f"design[{i}].shape = {D.shape}"
        assert set(np.unique(D).tolist()) <= {0.0, 1.0}, f"design[{i}] has non-binary entries"
        assert int(D.sum()) == 61, f"design[{i}].sum() = {int(D.sum())}, expected 61 trials"
    uniq_per_run = [int((D.sum(0) > 0).sum()) for D in designs]
    print(f"  unique stim per run: min={min(uniq_per_run)}, max={max(uniq_per_run)} "
          f"(< 61 means within-run repeats present; encoded as multiple onsets in same column)")

    # Write trial metadata regardless of dry-run
    write_trial_metadata(trial_metadata, metadata_path)
    print(f"Wrote {metadata_path} ({len(trial_metadata)} rows)")

    if args.dry_run:
        print("\nDry run complete — skipping BOLD load + GLMsingle fit.")
        return

    # Load BOLDs
    print("\nLoading BOLDs...")
    t0 = time.time()
    runs = list_tb_runs(args.sub, args.pipeline)
    data = load_bold_runs(args.sub, args.pipeline, runs)
    total_gb = sum(d.nbytes for d in data) / 1e9
    print(f"Loaded {len(data)} BOLDs in {time.time()-t0:.0f}s, total {total_gb:.1f} GB")
    print(f"  data[0].shape = {data[0].shape}")

    # GLMsingle fit
    from glmsingle.glmsingle import GLM_single
    params = {
        "wantlibrary": 1,           # fit HRF library per voxel
        "wantglmdenoise": 1,        # learn noise regressors from data
        "wantfracridge": 1,         # ridge-regression with fractional shrinkage
        "wantfileoutputs": [0, 0, 0, 1],   # save only TYPED (final, denoised + ridge)
        "wantmemoryoutputs": [0, 0, 0, 1],
        "sessionindicator": sessionindicator,
    }
    print("\nGLMsingle params:")
    for k, v in params.items():
        if k == "sessionindicator":
            print(f"  {k}: shape={v.shape}, unique={sorted(set(v.flatten().tolist()))}")
        else:
            print(f"  {k}: {v}")

    print(f"\nFitting GLMsingle → {glmsingle_dir}")
    t0 = time.time()
    glm = GLM_single(params)
    glm.fit(designs, data, STIMDUR, TR, outputdir=str(glmsingle_dir))
    print(f"Fit complete in {(time.time()-t0)/60:.1f} min")

    print("\nDone.")


if __name__ == "__main__":
    main()
