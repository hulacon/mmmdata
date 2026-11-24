#!/usr/bin/env python3
"""Phase-pair similarity inside a cross-validated cue-shared retrieval subspace.

Question: is there a part of the retrieval pattern that the image-cued and the
word-cued retrieval of the same item share, and does encoding reinstate along
it? A plain average of the two retrievals keeps half of each cue-specific part,
so it cannot answer this. Instead, fit the directions in voxel space along which
the two retrievals of the same item co-vary, and score the benchmark's cells
inside that subspace and in its complement.

Per subject x ROI x fold (folds = retrieval sessions; an item's two TB
retrievals share a session, so both leave together):
  1. Patterns are cleaned and run-mean normalised exactly as in benchmark_6cell.
  2. Training rows = non-anchor TB retrievals of the other sessions' items.
     PCA (cap --pca-rank) on the stacked image + word training rows.
  3. Basis "shared": eigenvectors of the symmetrised cross-covariance
     Xi^T Xw + Xw^T Xi between item-matched image and word rows (in PCA
     space), ordered by eigenvalue (largest positive first).
     Basis "pca" (control): the top PCs themselves — high variance, not
     cross-cue agreement.
  4. For each k in --ks, the held-out items' cells (enc<->image, enc<->word
     by reCon, image<->word) are scored on the patterns projected onto the
     k-dim subspace ("sub") and onto its orthogonal complement ("comp"),
     with benchmark_6cell's run-matched baseline and within-run item null.
     Encoding patterns never enter a fit. The retrieval runs of a held-out
     session hold only held-out items, so every retrieval-side baseline pool
     is out of fold.
Correlations of projected patterns are computed from k-dim coordinates plus
the trial Gram matrix (Pearson over voxels; no V x N projections are formed).

Outputs (per subject): <out_root>/results/retrieval_modeling/<sub>/
  <sub>_cue_shared.tsv            cell level, folds pooled at the item level
  <sub>_cue_shared_items.tsv.gz   item level
Usage:
    python cue_shared_subspace.py --subject sub-## [--types D] [--ks 1 2 5 10 20 50]
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


b6 = _load_module("benchmark_6cell", HERE / "benchmark_6cell.py")
tb, ps = b6.tb, b6.ps
SEED = 20261124
CELLS = [("enc", "image"), ("enc", "word"), ("image", "word")]


def corr_from_coords(Z, u, V, dot_extra=None, mean_extra=None, sq_extra=None):
    """Pearson over voxels of projected patterns f = B z (B orthonormal, V x k).

    Z: (k, N) coordinates; u = B^T 1. With dot_extra / mean_extra / sq_extra
    (the full-pattern Gram, means and squared norms) the complement p - Bz is
    scored instead.
    """
    dot = Z.T @ Z
    mean = (u @ Z) / V
    sq = (Z * Z).sum(0)
    if dot_extra is not None:
        dot = dot_extra - dot
        mean = mean_extra - mean
        sq = sq_extra - sq
    cov = dot / V - np.outer(mean, mean)
    var = np.clip(sq / V - mean ** 2, 1e-20, None)
    return cov / np.sqrt(np.outer(var, var))


def fit_bases(P, tr_img, tr_word, pca_rank):
    """-> {basis: (V x r) orthonormal, columns ordered}, from training rows only."""
    Xi, Xw = P[:, tr_img].T, P[:, tr_word].T           # (n, V), item-matched rows
    X = np.vstack([Xi, Xw])
    mu = X.mean(0)
    Xi, Xw, X = Xi - mu, Xw - mu, X - mu
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    r = min(pca_rank, int((s > s[0] * 1e-8).sum()))
    W = Vt[:r].T                                         # V x r PCA basis
    A, Bm = Xi @ W, Xw @ W
    C = A.T @ Bm
    C = 0.5 * (C + C.T)
    ev, E = np.linalg.eigh(C)
    order = np.argsort(ev)[::-1]
    return {"shared": W @ E[:, order], "pca": W}, ev[order]


def score(R, trials, pairs, pools, n_perm, rng):
    pairs = pairs.copy()
    pairs["within"] = R[pairs.a.to_numpy(), pairs.b.to_numpy()]
    pairs["across"] = b6.across_baseline(R, trials, pairs, pools)
    items = pairs.groupby("mmmId").agg(within=("within", "mean"), across=("across", "mean"))
    null = b6.null_deltas(R, trials, pairs, pools, n_perm, rng)
    return items, null


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject", required=True)
    ap.add_argument("--arm", default="siloed", choices=list(b6.ARM_FILES))
    ap.add_argument("--types", nargs="+", default=["D"], choices=["B", "C", "D"])
    ap.add_argument("--rois", nargs="+", default=ps.PATTERN_ROI_NAMES)
    ap.add_argument("--ks", nargs="+", type=int, default=[1, 2, 5, 10, 20, 50])
    ap.add_argument("--pca-rank", type=int, default=200)
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--meanvol-frac", type=float, default=0.25)
    ap.add_argument("--beta-cap", type=float, default=100.0)
    ap.add_argument("--cache-root", default=None)
    ap.add_argument("--out-root", default=None)
    args = ap.parse_args()

    cfg = tb.load_config()
    bids_root = Path(cfg["bids_project_dir"])
    cache_root = Path(args.cache_root) if args.cache_root else bids_root / "derivatives" / b6.CACHE_TREE
    out_dir = ((Path(args.out_root) if args.out_root else cache_root)
               / "results" / "retrieval_modeling" / args.subject)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    rows, item_rows = [], []
    for t in args.types:
        trials, patterns, mvs = b6.load_arm(cache_root, args.subject, args.arm, t)
        pools = b6.run_pools(trials)
        ret = trials[trials.subgroup.isin(["image", "word"]) & ~trials.anchor & ~trials.fin]
        wide = ret.reset_index().pivot_table(index="mmmId", columns="subgroup", values="index",
                                             aggfunc="first").dropna().astype(int)
        item_ses = ret.groupby("mmmId").session.first().loc[wide.index]
        if ret.groupby("mmmId").session.nunique().max() != 1:
            sys.exit("ERROR: an item's two TB retrievals span sessions; session folds would leak")
        folds = sorted(item_ses.unique())
        all_pairs = {f"{X}<->{Y}": b6.build_pairs(trials, X, Y) for X, Y in CELLS}
        all_pairs = {c: p[~p.anchor & ~p.same_run & ~p.fin] for c, p in all_pairs.items()}
        print(f"[{args.arm} TYPE{t}] {len(wide)} items in {len(folds)} session folds", flush=True)
        for roi in args.rois:
            t0 = time.time()
            ok = b6.clean_voxels(patterns[roi], mvs.get(roi), args.meanvol_frac, args.beta_cap)
            P = b6.normalize(patterns[roi][ok], trials, "runmean")
            V = P.shape[0]
            G = P.T @ P
            mfull, sqfull = P.mean(0), np.diag(G).copy()
            acc = {}                                     # (basis,k,part,cell,reCon) -> items, nulls
            for f in folds:
                test = set(item_ses.index[item_ses == f])
                tr = wide.loc[~wide.index.isin(test)]
                bases, ev = fit_bases(P, tr["image"].to_numpy(), tr["word"].to_numpy(), args.pca_rank)
                for bname, B in bases.items():
                    for k in args.ks:
                        if k > B.shape[1]:
                            continue
                        Bk = B[:, :k]
                        Z, u = Bk.T @ P, Bk.sum(0)
                        Rs = {"sub": corr_from_coords(Z, u, V),
                              "comp": corr_from_coords(Z, u, V, G, mfull, sqfull)}
                        for part, R in Rs.items():
                            for cell, pr in all_pairs.items():
                                pt = pr[pr.mmmId.isin(test)]
                                strata = (pt.groupby("reCon", dropna=False) if pt.reCon.notna().any()
                                          else [(np.nan, pt)])
                                for rc, g in strata:
                                    if g.empty:
                                        continue
                                    items, null = score(R, trials, g, pools, args.n_perm, rng)
                                    key = (bname, k, part, cell, rc)
                                    a = acc.setdefault(key, ([], []))
                                    a[0].append(items)
                                    a[1].append((null, len(items)))
            for (bname, k, part, cell, rc), (its, nulls) in acc.items():
                items = pd.concat(its)
                delta = float((items.within - items.across).mean())
                n = sum(m for _, m in nulls)
                null = sum(nl * m for nl, m in nulls) / n      # item-weighted across folds
                meta = dict(subject=args.subject, arm=args.arm, beta=f"TYPE{t}", roi=roi,
                            n_voxels=V, basis=bname, k=k, part=part, cell=cell, reCon=rc)
                rows.append(meta | dict(n_items=len(items), within=float(items.within.mean()),
                                        across=float(items.across.mean()), delta=delta,
                                        null_sd=float(null.std()),
                                        p_perm=float((np.sum(null >= delta) + 1) / (args.n_perm + 1)),
                                        n_perm=args.n_perm, n_folds=len(its)))
                for mmm, it in items.iterrows():
                    item_rows.append(meta | dict(mmmId=int(mmm), within=it.within, across=it.across))
            print(f"  {roi:12s} V={V}  {time.time() - t0:.0f}s", flush=True)

    stem = out_dir / args.subject
    pd.DataFrame(rows).to_csv(f"{stem}_cue_shared.tsv", sep="\t", index=False, float_format="%.5g")
    pd.DataFrame(item_rows).to_csv(f"{stem}_cue_shared_items.tsv.gz", sep="\t", index=False,
                                   float_format="%.5g")
    print(f"\nwrote {stem}_cue_shared.tsv")


if __name__ == "__main__":
    main()
