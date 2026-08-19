#!/usr/bin/env python3
"""Cross-validated low-level tuning check (srm-stimulus-space diagnostic).

Per (subject, pipeline, representation): correlate each EVC voxel's TB
response with image luminance (legacy viz2psy `luminance_mean`) across
trials in half the sessions (even-indexed), select the top-N positively
correlated voxels, and report their median luminance correlation in the
held-out (odd-indexed) sessions — and the reverse fold. A stable
low-level visual signal shows up as positive held-out r on selected
voxels; the random-voxel baseline and the EAC control ROI calibrate what
"zero" looks like. No encoding model, no embedding — the most
assumption-free stimulus-driven-signal check available.

Representations: TB GLMsingle TYPED betas (session-z) and TB rawTR trial
patterns (Model-8 residualized, z-scored per run) from the
pattern-similarity caches.

Usage:
    python lowlevel_check.py [--top-n 100] [--rois EVC EAC]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from a1_encoding import (
    BIDS_ROOT, FEAT_ROOT, REGISTRY, SUBJECTS,
    _session_zscore, beta_cache, rawtr_cache,
)

LEGACY_SCORES = BIDS_ROOT / "stimuli" / "shared1000" / "viz2psy_scores.csv"
FEATURE = "luminance_mean"  # overridden by --feature
RNG = np.random.default_rng(20260819)


def luminance_by_mmmid(feature: str = FEATURE) -> dict[str, float]:
    reg = pd.read_csv(REGISTRY / "shared1000.tsv", sep="\t")
    scores = pd.read_csv(LEGACY_SCORES, usecols=["filename", feature])
    scores["stimulus_id"] = scores["filename"].str.replace(
        r"\.png$", "", regex=True)
    merged = reg.merge(scores, on="stimulus_id", validate="1:1")
    return dict(zip(merged["mmmId"].astype(str), merged[feature]))


def load_trials(sub: str, pipeline: str, rep: str, roi: str):
    """(V, n_trials) responses + per-trial mmmId + session labels."""
    if rep == "glmsingle":
        f = (beta_cache(pipeline) / sub /
             f"{sub}_task-TBencoding_desc-typed_roipatterns.npz")
        d = np.load(f, allow_pickle=True)
        session = d["session"].astype(str)
        return (_session_zscore(d[f"patterns_{roi}"].astype(np.float64),
                                session),
                d["mmmId"].astype(str), session)
    files = sorted((rawtr_cache(pipeline) / sub).glob(
        f"{sub}_ses-*_task-TBencoding_run-*_desc-model8_roipatterns.npz"))
    pats, mmm, ses = [], [], []
    for f in files:
        d = np.load(f, allow_pickle=True)
        pats.append(d[f"patterns_{roi}"].astype(np.float64))
        mmm.append(d["mmmId"].astype(str))
        ses.extend([f.name.split("_")[1]] * pats[-1].shape[1])
    return np.hstack(pats), np.concatenate(mmm), np.array(ses)


def voxel_feature_corr(resp: np.ndarray, feat: np.ndarray) -> np.ndarray:
    """Per-voxel Pearson r between response and feature, NaN-aware."""
    r = np.full(resp.shape[0], np.nan)
    fz = (feat - feat.mean()) / feat.std()
    for v in range(resp.shape[0]):
        y = resp[v]
        ok = np.isfinite(y)
        if ok.sum() < 50 or y[ok].std() == 0:
            continue
        yz = (y[ok] - y[ok].mean()) / y[ok].std()
        r[v] = (yz * fz[ok]).mean()
    return r


def check(sub: str, pipeline: str, rep: str, roi: str, top_n: int,
          lum: dict[str, float]) -> list[dict]:
    resp, mmm, session = load_trials(sub, pipeline, rep, roi)
    feat = np.array([lum[i] for i in mmm])
    sessions = sorted(np.unique(session))
    folds = {"even": np.isin(session, sessions[0::2]),
             "odd": np.isin(session, sessions[1::2])}

    rows = []
    for train, test in (("even", "odd"), ("odd", "even")):
        r_train = voxel_feature_corr(resp[:, folds[train]],
                                     feat[folds[train]])
        r_test = voxel_feature_corr(resp[:, folds[test]], feat[folds[test]])
        valid = np.isfinite(r_train) & np.isfinite(r_test)
        order = np.argsort(np.where(valid, r_train, -np.inf))[::-1]
        top = order[:top_n]
        rand = RNG.choice(np.flatnonzero(valid), size=top_n, replace=False)
        rows.append({
            "subject": sub, "pipeline": pipeline, "rep": rep, "roi": roi,
            "fold": f"{train}->{test}", "n_valid_vox": int(valid.sum()),
            "sel_r_train": float(np.median(r_train[top])),
            "sel_r_heldout": float(np.median(r_test[top])),
            "rand_r_heldout": float(np.median(r_test[rand])),
        })
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--top-n", type=int, default=100)
    ap.add_argument("--feature", default=FEATURE)
    ap.add_argument("--rois", nargs="+", default=["EVC", "EAC"])
    ap.add_argument("--pipelines", nargs="+",
                    default=["original", "nordic"])
    args = ap.parse_args()

    lum = luminance_by_mmmid(args.feature)
    print(f"feature={args.feature}, {len(lum)} images, top_n={args.top_n}\n")
    rows = []
    for rep in ("rawtr", "glmsingle"):
        for pipeline in args.pipelines:
            for sub in SUBJECTS:
                for roi in args.rois:
                    rows.extend(check(sub, pipeline, rep, roi,
                                      args.top_n, lum))
    df = pd.DataFrame(rows)
    # average the two fold directions for the headline table
    g = df.groupby(["rep", "pipeline", "roi", "subject"])[
        ["sel_r_train", "sel_r_heldout", "rand_r_heldout"]].mean().round(4)
    print(g.to_string())


if __name__ == "__main__":
    main()
