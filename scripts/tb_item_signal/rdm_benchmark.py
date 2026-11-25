#!/usr/bin/env python3
"""Second-order (RSA) benchmark of item structure between datasets, per ROI.

Reads ROI pattern caches in the ``extract_roi_betas.py`` format (and any
adapter writing the same arrays, e.g. ``nsd_shared_cache.py``) in two groups:
``--target`` (the data being benchmarked, >= 1 cache) and ``--reference``
(the benchmark, >= 2 caches). The caches need not share voxels or a space;
only the ROI names and the item ids (``mmmId``) must mean the same thing.

Per ROI:
  1. each cache's patterns are cleaned and z-scored per voxel within session
     with ``item_reliability.py``'s rules; anchors (``sharedId == 1``) are
     dropped;
  2. item-mean patterns over all of an item's exposures, on ONE item set
     common to every cache (``--ref-min-reps``: also require that many
     exposures in every reference cache);
  3. RDM = 1 - Pearson r between item means;
  4. cells whose two items share a run (session, task, run) in ANY cache are
     masked out of every comparison: shared run noise makes same-run items
     look alike, and one mask keeps every pair of RDMs on the same cells.
     Between subjects, independent run noise only attenuates r; the mask
     matters when caches share a presentation sequence (NSD's subjects saw
     shared1000 in one order), where sequence-locked state is common to both.
     ``--mask-on session`` widens it to same-session pairs (sequence-locked
     session state); ``none`` keeps every cell (a check);
  5. Spearman r between every pair of cache RDMs over the unmasked cells.

Reported per ROI (long TSV, one row per metric x label):
  target_ref    mean r between a target RDM and the reference RDMs
                (label = the target, or ``all`` for the mean over targets)
  ref_ref       mean r between distinct reference RDMs: the benchmark's own
                between-subject agreement, the ceiling for ``target_ref``
                (label = ``all``, or a reference's mean r with the others)
  fraction      target_ref / ref_ref (same labels as target_ref)
  split_half    per cache, r between two RDMs each built from ONE exposure per
                item, the two exposures in different runs (items with such a
                pair only; cells masked when either half's items share a run).
                A per-exposure reliability index, on a cache-specific item
                subset; not used in ``fraction``.
CIs: percentile bootstrap over items (``--n-boot``; cells pairing an item with
its own duplicate are dropped), every quantity recomputed on each draw.
Null for target_ref: the target RDMs' item labels permuted (``--n-perm``)
against fixed reference RDMs and a fixed mask.

Usage:
    python rdm_benchmark.py --target <mmm_sub-##.npz> ... \\
        --reference <nsd_subj##.npz> ... --rois EVC_hcp VTC_hcp --out <out.tsv>
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from item_reliability import clean_voxels, cross_run_pairs, load_trials, session_zscore


# ── per-cache preparation ────────────────────────────────────────────────────

def cache_label(d, path: Path) -> str:
    parts = [str(d[k]) for k in ("arm", "subject") if k in d.files]
    return ":".join(parts) if parts else path.stem


def roi_patterns(d, t, roi, meanvol_frac, beta_cap, zscore=True):
    """-> (Z: voxels x trials, cleaned and session-z, n_vox_kept, n_vox_roi)."""
    P = d[f"patterns_{roi}"].astype(np.float64)
    mv = d[f"meanvol_{roi}"] if f"meanvol_{roi}" in d.files else None
    ok = clean_voxels(P, mv, meanvol_frac, beta_cap)
    P = P[ok]
    Z = session_zscore(P, t["session"].to_numpy()) if zscore else P
    return Z, int(ok.sum()), int(len(ok))


def item_means(Z, t, items):
    """-> items x voxels, the mean over each item's (non-anchor) exposures."""
    reg = t[~t["anchor"]]
    cols = reg.groupby("mmmId").indices          # positions within `reg`
    pos = reg.index.to_numpy()                   # -> columns of Z
    return np.stack([Z[:, pos[cols[i]]].mean(axis=1) for i in items])


def corun(t, items, on="run_key") -> np.ndarray:
    """items x items bool: the two items share at least one run (or session)."""
    reg = t[~t["anchor"] & t["mmmId"].isin(items)]
    runs = pd.Index(reg[on].unique())
    R = np.zeros((len(items), len(runs)), np.float32)
    ii = pd.Index(items).get_indexer(reg["mmmId"])
    R[ii, runs.get_indexer(reg[on])] = 1
    return (R @ R.T) > 0


def rdm(M) -> np.ndarray:
    X = M - M.mean(axis=1, keepdims=True)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = np.inf
    X = X / n
    return 1.0 - X @ X.T


# ── second-order comparison ──────────────────────────────────────────────────

def std_ranks(v) -> np.ndarray:
    r = rankdata(v)
    r -= r.mean()
    s = np.linalg.norm(r)
    return r / s if s > 0 else r


def spearman_matrix(vecs) -> np.ndarray:
    V = np.stack([std_ranks(v) for v in vecs])
    return V @ V.T


def summarise_S(S, n_t):
    """S: (targets + refs)^2 Spearman matrix, targets first.
    -> dict of named scalars (per-target and overall)."""
    T = S[:n_t, n_t:]
    Rr = S[n_t:, n_t:]
    n_r = Rr.shape[0]
    off = ~np.eye(n_r, dtype=bool)
    ref_ref = Rr[off].mean()
    out = {("target_ref", "all"): T.mean(), ("ref_ref", "all"): ref_ref,
           ("fraction", "all"): T.mean() / ref_ref}
    for k in range(n_t):
        out[("target_ref", k)] = T[k].mean()
        out[("fraction", k)] = T[k].mean() / ref_ref
    for k in range(n_r):
        out[("ref_ref", n_t + k)] = Rr[k][off[k]].mean()
    return out


def split_half(Z, t, items):
    """Per-exposure RDM reliability on items with a cross-run exposure pair."""
    _, first = cross_run_pairs(t, "any")
    item_set = set(items)
    keep = [(i, x, y) for i, x, y in first if i in item_set]
    if len(keep) < 10:
        return np.nan, len(keep)
    cx = np.array([x for _, x, _ in keep])
    cy = np.array([y for _, _, y in keep])
    rk = t["run_key"].to_numpy()
    same = (rk[cx][:, None] == rk[cx][None, :]) | (rk[cy][:, None] == rk[cy][None, :])
    iu = np.triu_indices(len(keep), 1)
    m = ~same[iu]
    a, b = rdm(Z[:, cx].T)[iu][m], rdm(Z[:, cy].T)[iu][m]
    return float(std_ranks(a) @ std_ranks(b)), len(keep)


# ── main ─────────────────────────────────────────────────────────────────────

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--target", nargs="+", required=True, type=Path)
    ap.add_argument("--reference", nargs="+", required=True, type=Path)
    ap.add_argument("--rois", nargs="+", required=True)
    ap.add_argument("--ref-min-reps", type=int, default=1,
                    help="keep items with at least this many exposures in every reference cache")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--meanvol-frac", type=float, default=0.25)
    ap.add_argument("--beta-cap", type=float, default=100.0)
    ap.add_argument("--no-zscore", action="store_true")
    ap.add_argument("--mask-on", default="run", choices=["run", "session", "none"],
                    help="mask cells whose items share a run (default), a session, or nothing")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--out-matrix", type=Path, default=None,
                    help="also write every ROI's cache x cache Spearman matrix (long TSV)")
    args = ap.parse_args(argv)
    if len(args.reference) < 2:
        sys.exit("need >= 2 reference caches (ref_ref is between references)")

    paths = list(args.target) + list(args.reference)
    n_t = len(args.target)
    for p in paths:
        if not p.is_file():
            sys.exit(f"cache not found: {p}")
    ds = [np.load(p, allow_pickle=True) for p in paths]
    ts = [load_trials(d) for d in ds]
    labels = [cache_label(d, p) for d, p in zip(ds, paths)]
    if len(set(labels)) != len(labels):
        sys.exit(f"cache labels collide: {labels}")
    for d, lab in zip(ds, labels):
        miss = [r for r in args.rois if f"patterns_{r}" not in d.files]
        if miss:
            sys.exit(f"{lab}: ROIs not in cache: {miss}")

    # one item set for every cache
    sets = []
    for k, t in enumerate(ts):
        n = t[~t["anchor"]].groupby("mmmId").size()
        if k >= n_t:
            n = n[n >= args.ref_min_reps]
        sets.append(set(n.index))
    items = sorted(set.intersection(*sets), key=lambda s: (len(s), s))
    n_items = len(items)
    if n_items < 20:
        sys.exit(f"only {n_items} items common to every cache")
    mask = np.zeros((n_items, n_items), bool)
    if args.mask_on != "none":
        on = "run_key" if args.mask_on == "run" else "session"
        for t in ts:
            mask |= corun(t, items, on)
    iu = np.triu_indices(n_items, 1)
    keep = ~mask[iu]
    ci, cj = iu[0][keep], iu[1][keep]
    print(f"{n_t} target + {len(args.reference)} reference caches | {n_items} common items | "
          f"{keep.sum()}/{keep.size} cells after the --mask-on {args.mask_on} mask", flush=True)

    rows, mats = [], []
    for roi in args.rois:
        t0 = time.time()
        rng = np.random.default_rng(args.seed)
        Ds, nvox, sh = [], [], []
        for d, t in zip(ds, ts):
            Z, nk, nr = roi_patterns(d, t, roi, args.meanvol_frac, args.beta_cap,
                                     not args.no_zscore)
            if nk < 10:
                sys.exit(f"{roi}: {nk} voxels after cleaning in one cache")
            Ds.append(rdm(item_means(Z, t, items)).astype(np.float32))
            nvox.append((nk, nr))
            sh.append(split_half(Z, t, items))

        S = spearman_matrix([D[ci, cj] for D in Ds])
        obs = summarise_S(S, n_t)

        # bootstrap over items
        boot = {k: np.empty(args.n_boot) for k in obs}
        for b in range(args.n_boot):
            idx = rng.integers(0, n_items, n_items)
            a, c = idx[iu[0]], idx[iu[1]]
            ok = (a != c) & ~mask[a, c]
            Sb = spearman_matrix([D[a[ok], c[ok]] for D in Ds])
            for k, v in summarise_S(Sb, n_t).items():
                boot[k][b] = v

        # permutation null for target_ref: relabel target items
        Vref = np.stack([std_ranks(D[ci, cj]) for D in Ds[n_t:]])
        null_all = np.empty(args.n_perm)
        null_t = np.empty((n_t, args.n_perm))
        for p in range(args.n_perm):
            for k in range(n_t):
                pm = rng.permutation(n_items)
                null_t[k, p] = (Vref @ std_ranks(Ds[k][pm[ci], pm[cj]])).mean()
            null_all[p] = null_t[:, p].mean()

        base = {"roi": roi, "mask_on": args.mask_on, "n_items": n_items, "n_cells": int(len(ci)),
                "n_boot": args.n_boot, "n_perm": args.n_perm}
        for (metric, who), v in obs.items():
            lab = "all" if who == "all" else labels[who]
            bv = boot[(metric, who)]
            row = {**base, "metric": metric, "label": lab, "value": v,
                   "boot_lo": np.nanpercentile(bv, 2.5), "boot_hi": np.nanpercentile(bv, 97.5)}
            if metric == "target_ref":
                nd = null_all if who == "all" else null_t[who]
                row.update(null_mean=nd.mean(), null_95=np.percentile(nd, 95),
                           p_perm=(1 + (nd >= v).sum()) / (1 + len(nd)))
            if who != "all":
                row.update(n_vox=nvox[who][0], n_vox_roi=nvox[who][1])
            rows.append(row)
        for k, lab in enumerate(labels):
            rows.append({**base, "metric": "split_half", "label": lab, "value": sh[k][0],
                         "n_items_split": sh[k][1], "n_vox": nvox[k][0], "n_vox_roi": nvox[k][1]})
        for i, li in enumerate(labels):
            for j, lj in enumerate(labels):
                mats.append({"roi": roi, "a": li, "b": lj, "spearman": S[i, j]})
        o = obs
        print(f"  {roi} [{time.time() - t0:.0f}s] target_ref={o[('target_ref', 'all')]:.3f} "
              f"ref_ref={o[('ref_ref', 'all')]:.3f} fraction={o[('fraction', 'all')]:.3f} "
              f"(null95 {np.percentile(null_all, 95):.3f})", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, sep="\t", index=False, float_format="%.5g")
    print(f"wrote {args.out}")
    if args.out_matrix:
        pd.DataFrame(mats).to_csv(args.out_matrix, sep="\t", index=False, float_format="%.5g")
        print(f"wrote {args.out_matrix}")


if __name__ == "__main__":
    main()
