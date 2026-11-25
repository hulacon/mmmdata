#!/usr/bin/env python3
"""Plain least-squares-separate (LSS) single-trial betas for a TB fit's trials,
restricted to the voxels of an existing ROI pattern cache.

A reference estimator beside GLMsingle's TYPEB/C/D: no HRF library, no
GLMdenoise, no ridge. For every trial, one OLS model per run holds that trial's
regressor, one regressor for all other trials of the run, the confounds and an
intercept (Mumford et al. 2012). Confounds and HRF follow the GLM bake-off
precedent (``neuroimaging.glm.config``): nilearn's SPM HRF; six motion
parameters + the first six aCompCor components + fMRIPrep's cosine drift
regressors + non-steady-state spikes (the ``acompcor`` preset). Fit by OLS.

Trials, their order and their metadata come from the GLMsingle fit's
``trial_info.csv`` (same session / run / onset / duration), so the output's
columns line up one-to-one with the GLMsingle caches. Voxels are the
``voxidx_<ROI>`` sets of a template cache (``--template``), so ROI voxel order
matches too. Betas are expressed in percent of each voxel's run-mean signal,
like GLMsingle's.

Output: an npz in the ``extract_roi_betas.py`` cache format
(``patterns_<ROI>``, ``voxidx_<ROI>``, ``meanvol_<ROI>``, trial metadata,
``beta_type = "LSS"``), readable by ``item_reliability.py``.

Usage:
    python lss_roi_betas.py --fit-dir <derivatives/glmsingle_tb/sub-##/enc> \\
        --template <cache.npz> [--template <ladder cache.npz> --rois VTC ...] \\
        --out <sub-##_arm-enc_desc-lss_roipatterns.npz>
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

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src" / "python"))
from neuroimaging.glm.config import DEFAULT_CONFIG  # noqa: E402
from neuroimaging.io import fmriprep_runs_on_disk, load_confounds  # noqa: E402

META = ["trial_index", "session", "run", "task", "subgroup", "mmmId",
        "condition_id", "col_index", "onset", "duration", "word", "pairId",
        "sharedId", "enCon", "reCon", "resp", "resp_RT"]


def hrf_regressors(onsets, durations, n_scans, tr):
    """(n_scans, n_trials) SPM-HRF-convolved boxcars, frame times = k * TR."""
    from nilearn.glm.first_level import compute_regressor
    ft = np.arange(n_scans) * tr
    cols = []
    for o, d in zip(onsets, durations):
        reg, _ = compute_regressor(np.array([[o], [d], [1.0]]), "spm", ft,
                                   oversampling=50)
        cols.append(reg[:, 0])
    return np.stack(cols, axis=1)


def lss(Y, H, C):
    """Y (T, V) data; H (T, n) trial regressors; C (T, k) confounds incl.
    intercept. -> (n, V) LSS betas. Frisch–Waugh: residualise on C first,
    then per trial a two-column OLS [h_i, sum of the others]."""
    Q, _ = np.linalg.qr(C)
    Yr = Y - Q @ (Q.T @ Y)
    Hr = H - Q @ (Q.T @ H)
    S = Hr.sum(axis=1)
    B = np.empty((H.shape[1], Y.shape[1]))
    for i in range(H.shape[1]):
        h = Hr[:, i]
        o = S - h
        X = np.stack([h, o], axis=1)
        G = X.T @ X
        B[i] = np.linalg.solve(G, X.T @ Yr)[0]
    return B


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fit-dir", required=True, type=Path,
                    help="GLMsingle fit dir holding trial_info.csv + run_metadata.json")
    ap.add_argument("--template", required=True, type=Path, action="append",
                    help="ROI cache(s) whose voxidx_<ROI> sets to use (repeatable)")
    ap.add_argument("--rois", nargs="*", default=None,
                    help="ROI names to keep (default: every ROI in the templates)")
    ap.add_argument("--confounds", default="acompcor",
                    help="confound preset of neuroimaging.glm.config")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)

    meta = json.loads((args.fit_dir / "run_metadata.json").read_text())
    trials = pd.read_csv(args.fit_dir / "trial_info.csv")
    tr = float(meta["tr"])
    subject = meta["subject"]

    # ROI voxel sets from the template caches
    rois, grid_shape, affine = {}, None, None
    for tp in args.template:
        d = np.load(tp, allow_pickle=True)
        grid_shape = tuple(d["grid_shape"]) if grid_shape is None else grid_shape
        affine = d["affine"] if affine is None else affine
        if tuple(d["grid_shape"]) != grid_shape:
            sys.exit(f"{tp} is on grid {tuple(d['grid_shape'])}, not {grid_shape}")
        for r in d["roi_names"]:
            r = str(r)
            if (args.rois is None or r in args.rois) and r not in rois:
                rois[r] = d[f"voxidx_{r}"]
    if args.rois:
        absent = sorted(set(args.rois) - set(rois))
        if absent:
            sys.exit(f"ROIs not in any template: {absent}")
    union = np.unique(np.concatenate(list(rois.values())))
    pos = {r: np.searchsorted(union, v) for r, v in rois.items()}
    print(f"{subject}: {len(trials)} trials, {len(rois)} ROIs, "
          f"{len(union)} voxels", flush=True)

    cfg = DEFAULT_CONFIG.with_confounds(args.confounds)
    sub = subject.removeprefix("sub-")
    betas = np.full((len(union), len(trials)), np.nan, dtype=np.float32)
    meanv = np.zeros((len(union), 0))
    run_keys = trials[["session", "run", "task"]].drop_duplicates().to_numpy()
    for ses, run, task in run_keys:
        t0 = time.time()
        rows = np.flatnonzero((trials["session"] == ses) & (trials["run"] == run)
                              & (trials["task"] == task))
        found = fmriprep_runs_on_disk(subject=sub, session=ses.removeprefix("ses-"),
                                      task=task, run=run.removeprefix("run-"))
        if len(found) != 1:
            sys.exit(f"{subject} {ses} {task} {run}: expected 1 fMRIPrep run, "
                     f"found {len(found)}")
        fr = found[0]
        img = nib.load(fr.bold)
        if img.shape[:3] != grid_shape:
            sys.exit(f"{fr.bold} grid {img.shape[:3]} != template grid {grid_shape}")
        data = np.asarray(img.dataobj, dtype=np.float32)
        Y = data.reshape(-1, data.shape[3])[union].T.astype(np.float64)  # (T, V)
        del data
        n = Y.shape[0]
        conf = load_confounds(fr)
        cols = cfg.confound_columns(list(conf.columns))
        C = conf[cols].fillna(0.0).to_numpy()
        C = np.column_stack([C, np.ones(n)])
        H = hrf_regressors(trials.loc[rows, "onset"].to_numpy(),
                           trials.loc[rows, "duration"].to_numpy(), n, tr)
        mu = Y.mean(axis=0)
        B = lss(Y, H, C)
        with np.errstate(divide="ignore", invalid="ignore"):
            B = 100.0 * B / mu
        betas[:, rows] = B.T.astype(np.float32)
        meanv = np.column_stack([meanv, mu])
        print(f"  {ses} {task} {run}: {len(rows)} trials, {n} vols, "
              f"{len(cols)} confounds, {time.time() - t0:.0f}s", flush=True)

    mv = meanv.mean(axis=1).astype(np.float32)
    out = {"subject": subject, "arm": meta.get("arm", ""), "beta_type": "LSS",
           "roi_names": np.array(list(rois)), "grid_shape": np.array(grid_shape),
           "affine": affine, "source_dir": str(args.fit_dir),
           "estimator": f"LSS OLS, spm HRF, confounds={args.confounds} "
                        f"({cfg.acompcor_n} aCompCor) + cosine + non-steady-state"}
    for c in META:
        if c in trials.columns:
            v = trials[c].to_numpy()
            out[c] = v.astype(str).astype(object) if c in ("mmmId", "condition_id") else v
    if "trial_index" not in trials.columns:
        out["trial_index"] = np.arange(len(trials))
    out["run"] = trials["run"].str.removeprefix("run-").astype(int).to_numpy()
    for r, p in pos.items():
        out[f"patterns_{r}"] = betas[p]
        out[f"voxidx_{r}"] = rois[r]
        out[f"meanvol_{r}"] = mv[p]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
