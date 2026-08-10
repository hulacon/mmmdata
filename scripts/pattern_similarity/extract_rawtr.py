#!/usr/bin/env python3
"""Phase 3 — rawTR ROI pattern extraction (pattern-similarity analysis).

Per (pipeline, subject, session, run): load preproc BOLD + brain mask, slice
to the 6 bilateral Harvard-Oxford ROIs *before* regression, voxelwise
Model-8 residualization (6 motion + CSF + WM + GS + cosine), per-voxel
z-score over the run, then extract:

  - TBencoding: per image trial, pattern = mean of vols onset_tr+3, +4
    (+4.5 s HRF shift) -> per-ROI (V_roi, 61) + mmmId/onset metadata.
  - NATencoding: per movie, per-TR patterns shifted +3 TRs
    -> per-ROI (V_roi, n_movie_TRs) + movie_name/tr_index metadata.

Voxel sets are fixed per ROI on the res-2 grid (atlas mask only); voxels
outside a run's brain mask are NaN so downstream pair correlations can use
the mutually-finite-voxel convention (plan D6) with column alignment
guaranteed across runs, sessions, and pipelines.

Output (float32 npz, all 6 ROIs per file):
  derivatives/pattern_similarity/cache/rawtr/{pipeline}/{sub}/
    {sub}_{ses}_task-{task}_run-{run:02d}_desc-model8_roipatterns.npz

Plan: docs/doc/pattern-similarity-plan.md, Phase 3.

Usage:
    python extract_rawtr.py --sub sub-03 --pipeline original          # all 62 runs
    python extract_rawtr.py --sub sub-03 --pipeline original \
        --task TBencoding --ses ses-04 --run 1                        # single run
    python extract_rawtr.py --verify                                  # post-hoc checks
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
    CACHE_DIR, HRF_SHIFT_TRS, NAT_RUNS, NAT_SESSIONS, PATTERN_ROI_NAMES,
    PIPELINES, SUBJECTS, TB_RUNS, TB_SESSIONS, TR,
    bold_path, confounds_path, events_path, load_bilateral_roi_masks,
    mask_path, read_tb_trials, resample_masks_to_bold,
)

# ── Nastase & Hasson confound utilities (Model 8) ────────────────────────
# Mirrors scripts/nordic_benchmark/nat_eval.py (natsort shim + vendored
# code/isc-confounds); not imported from there to avoid a `shared` module
# name collision between the two script packages.
import re as _re
import types as _types


def _natsorted(seq, **_kwargs):
    def _key(s):
        return [int(c) if c.isdigit() else c.lower()
                for c in _re.split(r"(\d+)", str(s))]
    return sorted(seq, key=_key)


if "natsort" not in sys.modules:
    _natsort_shim = _types.ModuleType("natsort")
    _natsort_shim.natsorted = _natsorted
    sys.modules["natsort"] = _natsort_shim

_SCRIPT_DIR = Path(__file__).resolve().parent
_ISC_CONFOUNDS_CODE = _SCRIPT_DIR.parent.parent.parent / "isc-confounds"
_MODEL_META_PATH = _SCRIPT_DIR.parent / "isc_confounds" / "model_meta.json"
sys.path.insert(0, str(_ISC_CONFOUNDS_CODE))
from extract_confounds import extract_confounds, load_confounds  # noqa: E402

with open(_MODEL_META_PATH) as _fh:
    MODEL_META = json.load(_fh)

MODEL_ID = "8"            # winner of the 21-model ISC benchmark
CONFOUND_LABEL = "model8"

RAWTR_CACHE = CACHE_DIR / "rawtr"

TASK_RUNS = {
    "TBencoding": [(ses, run) for ses in TB_SESSIONS for run in TB_RUNS],
    "NATencoding": [(ses, run) for ses in NAT_SESSIONS for run in NAT_RUNS],
}
N_RUNS_TOTAL = sum(len(v) for v in TASK_RUNS.values())  # 62
N_TB_TRIALS = 61


def confound_matrix_for_run(confounds_p: Path) -> np.ndarray:
    """Model-8 (T x k) confound design for a run."""
    df, meta = load_confounds(str(confounds_p))
    return extract_confounds(df, meta, MODEL_META[MODEL_ID]).values


def regress_confounds(ts: np.ndarray, confound_matrix: np.ndarray) -> np.ndarray:
    """OLS-residualize (T, V) timeseries against a confound design (+intercept)."""
    X = np.nan_to_num(np.asarray(confound_matrix, dtype=np.float64), nan=0.0)
    X = np.column_stack([np.ones(X.shape[0]), X])
    betas = np.linalg.lstsq(X, ts, rcond=None)[0]
    return ts - X @ betas


def out_path(sub: str, ses: str, run: int, task: str, pipeline: str) -> Path:
    return (RAWTR_CACHE / pipeline / sub /
            f"{sub}_{ses}_task-{task}_run-{run:02d}_desc-{CONFOUND_LABEL}_roipatterns.npz")


def build_roi_grid_masks(ref_bold_img):
    """ROI masks + flat C-order voxel indices on the BOLD res-2 grid."""
    masks_atlas, atlas_affine = load_bilateral_roi_masks()
    masks = resample_masks_to_bold(masks_atlas, atlas_affine, ref_bold_img)
    vox_idx = {roi: np.flatnonzero(m.ravel(order="C"))
               for roi, m in masks.items()}
    return masks, vox_idx


def extract_run_patterns(sub: str, ses: str, run: int, task: str,
                         pipeline: str, roi_masks: dict) -> dict:
    """Residualized, z-scored per-ROI patterns + metadata for one run."""
    bold_img = nib.load(str(bold_path(sub, ses, run, task, pipeline)))
    ref_shape = next(iter(roi_masks.values())).shape
    assert bold_img.shape[:3] == ref_shape, (
        f"{sub} {ses} run-{run:02d}: BOLD grid {bold_img.shape[:3]} != "
        f"ROI mask grid {ref_shape}"
    )
    bold = bold_img.get_fdata(dtype=np.float32)  # (X, Y, Z, T)
    n_vols = bold.shape[-1]
    brain = np.asarray(
        nib.load(str(mask_path(sub, ses, run, task, pipeline))).dataobj) > 0

    cmat = confound_matrix_for_run(confounds_path(sub, ses, run, task, pipeline))
    assert cmat.shape[0] == n_vols, (
        f"confounds rows ({cmat.shape[0]}) != n_vols ({n_vols})"
    )

    # Per-ROI: residualize + z-score valid (in-brain, non-constant) voxels;
    # everything else stays NaN so pairwise finite-filters handle it.
    roi_z = {}
    for roi in PATTERN_ROI_NAMES:
        m = roi_masks[roi]
        data = bold[m].astype(np.float64)          # (V_roi, T), fixed voxel order
        valid = brain[m] & (data.std(axis=1) > 0)
        z = np.full(data.shape, np.nan, dtype=np.float32)
        if valid.any():
            resid = regress_confounds(data[valid].T, cmat).T  # (V_valid, T)
            sd = resid.std(axis=1, keepdims=True)
            sd[sd == 0] = np.nan
            z[valid] = ((resid - resid.mean(axis=1, keepdims=True)) / sd
                        ).astype(np.float32)
        roi_z[roi] = z

    out = {"n_vols": np.int64(n_vols)}

    if task == "TBencoding":
        trials = read_tb_trials(sub, ses, run)
        assert len(trials) == N_TB_TRIALS, (
            f"{sub} {ses} run-{run:02d}: {len(trials)} image trials, "
            f"expected {N_TB_TRIALS}"
        )
        cols = [t.onset_tr + HRF_SHIFT_TRS for t in trials]
        assert max(cols) + 1 < n_vols, "HRF-shifted trial window exceeds run"
        for roi in PATTERN_ROI_NAMES:
            z = roi_z[roi]
            out[f"patterns_{roi}"] = np.stack(
                [z[:, c:c + 2].mean(axis=1) for c in cols], axis=1)
        out["mmmId"] = np.array([t.mmm_id for t in trials])
        out["onset_s"] = np.array([t.onset_s for t in trials])
    else:
        events = pd.read_csv(events_path(sub, ses, run, task), sep="\t")
        movies = events[events["trial_type"] == "movie"].sort_values("onset")
        assert len(movies) >= 2, (
            f"{sub} {ses} run-{run:02d}: <2 movies (same-run sampler needs >=2)"
        )
        windows, names, tr_idx = [], [], []
        for _, row in movies.iterrows():
            onset_vol = int(np.round(row["onset"] / TR))
            dur_tr = int(np.round(row["duration"] / TR))
            start = onset_vol + HRF_SHIFT_TRS
            end = min(start + dur_tr, n_vols)
            assert end > start, f"movie {row['movie_name']}: empty shifted window"
            windows.append((start, end))
            names.extend([row["movie_name"]] * (end - start))
            tr_idx.extend(range(end - start))
        for roi in PATTERN_ROI_NAMES:
            z = roi_z[roi]
            out[f"patterns_{roi}"] = np.concatenate(
                [z[:, s:e] for s, e in windows], axis=1)
        out["movie_name"] = np.array(names)
        out["tr_index"] = np.array(tr_idx, dtype=np.int64)
    return out


def process_subject(sub: str, pipeline: str, tasks: list[str],
                    only_ses: str | None, only_run: int | None):
    run_list = [(task, ses, run)
                for task in tasks
                for ses, run in TASK_RUNS[task]
                if (only_ses is None or ses == only_ses)
                and (only_run is None or run == only_run)]
    print(f"=== rawTR extraction: {sub} / {pipeline} — {len(run_list)} runs ===")

    ref_task, ref_ses, ref_run = run_list[0]
    ref_img = nib.load(str(bold_path(sub, ref_ses, ref_run, ref_task, pipeline)))
    roi_masks, vox_idx = build_roi_grid_masks(ref_img)
    print("ROI voxel counts:",
          {roi: len(vox_idx[roi]) for roi in PATTERN_ROI_NAMES})

    out_dir = RAWTR_CACHE / pipeline / sub
    out_dir.mkdir(parents=True, exist_ok=True)

    for task, ses, run in run_list:
        t0 = time.time()
        arrays = extract_run_patterns(sub, ses, run, task, pipeline, roi_masks)
        arrays["roi_names"] = np.array(PATTERN_ROI_NAMES)
        arrays["confound_model"] = np.array(CONFOUND_LABEL)
        for roi in PATTERN_ROI_NAMES:
            arrays[f"voxidx_{roi}"] = vox_idx[roi]
        p = out_path(sub, ses, run, task, pipeline)
        np.savez_compressed(p, **arrays)
        n_cols = arrays[f"patterns_{PATTERN_ROI_NAMES[0]}"].shape[1]
        print(f"  {ses} {task} run-{run:02d}: {n_cols} patterns/ROI "
              f"({time.time() - t0:.0f}s) -> {p.name}")


# ── post-hoc verification (plan Phase 3 "Verify") ────────────────────────

def verify():
    """Login-node checks over all (sub x pipeline) caches; prints a report."""
    rng = np.random.default_rng(0)
    failures = 0
    for pipeline in PIPELINES:
        for sub in SUBJECTS:
            files = sorted((RAWTR_CACHE / pipeline / sub).glob("*_roipatterns.npz"))
            status = "OK" if len(files) == N_RUNS_TOTAL else "INCOMPLETE"
            if status != "OK":
                failures += 1
            print(f"{sub}/{pipeline}: {len(files)}/{N_RUNS_TOTAL} caches [{status}]")
            if not files:
                continue

            # TB: 61 trials per run
            tb = [f for f in files if "task-TBencoding" in f.name]
            for f in tb:
                n = np.load(f)["mmmId"].shape[0]
                if n != N_TB_TRIALS:
                    print(f"  FAIL {f.name}: {n} trials != {N_TB_TRIALS}")
                    failures += 1

            # NAT: TR counts consistent with events (allowing end-of-run clip)
            for f in [f for f in files if "task-NATencoding" in f.name]:
                d = np.load(f)
                ses, run = f.name.split("_")[1], int(f.name.split("run-")[1][:2])
                ev = pd.read_csv(events_path(sub, ses, run, "NATencoding"), sep="\t")
                mov = ev[ev["trial_type"] == "movie"]
                expected = int(np.round(mov["duration"] / TR).sum())
                got = d["tr_index"].shape[0]
                if not (expected - HRF_SHIFT_TRS <= got <= expected):
                    print(f"  FAIL {f.name}: {got} TRs, expected "
                          f"~{expected} (movies: {len(mov)})")
                    failures += 1

            # Spot check: residual ROI-mean ~uncorrelated with confounds
            if tb:
                f = tb[rng.integers(len(tb))]
                d = np.load(f)
                ses, run = f.name.split("_")[1], int(f.name.split("run-")[1][:2])
                # Rebuild z-scored run timeseries for one ROI via re-extraction
                ref_img = nib.load(str(bold_path(sub, ses, run, "TBencoding", pipeline)))
                roi_masks, _ = build_roi_grid_masks(ref_img)
                bold = ref_img.get_fdata(dtype=np.float32)
                m = roi_masks["EVC"]
                brain = np.asarray(nib.load(str(
                    mask_path(sub, ses, run, "TBencoding", pipeline))).dataobj) > 0
                data = bold[m].astype(np.float64)
                valid = brain[m] & (data.std(axis=1) > 0)
                cmat = confound_matrix_for_run(
                    confounds_path(sub, ses, run, "TBencoding", pipeline))
                resid_mean = regress_confounds(data[valid].T, cmat).T.mean(axis=0)
                cmat_f = np.nan_to_num(cmat, nan=0.0)
                keep = cmat_f.std(axis=0) > 0
                rs = np.abs([np.corrcoef(resid_mean, c)[0, 1]
                             for c in cmat_f.T[keep]])
                worst = float(rs.max())
                tag = "OK" if worst < 1e-6 else "FAIL"
                if tag == "FAIL":
                    failures += 1
                print(f"  spot-check {f.name} EVC: max |r(resid_mean, confound)| "
                      f"= {worst:.2e} [{tag}]")
    print(f"\nVerification {'PASSED' if failures == 0 else f'FAILED ({failures})'}")
    return failures


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sub", choices=SUBJECTS)
    ap.add_argument("--pipeline", choices=sorted(PIPELINES))
    ap.add_argument("--task", choices=sorted(TASK_RUNS), nargs="+",
                    default=["TBencoding", "NATencoding"])
    ap.add_argument("--ses", help="restrict to one session (testing)")
    ap.add_argument("--run", type=int, help="restrict to one run (testing)")
    ap.add_argument("--verify", action="store_true",
                    help="check existing caches instead of extracting")
    args = ap.parse_args()

    if args.verify:
        sys.exit(1 if verify() else 0)
    if not (args.sub and args.pipeline):
        ap.error("--sub and --pipeline are required unless --verify")
    process_subject(args.sub, args.pipeline, args.task, args.ses, args.run)
    print("\nDone.")


if __name__ == "__main__":
    main()
