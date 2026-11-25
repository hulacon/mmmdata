#!/usr/bin/env python3
"""Cross-run item reliability of TB single-trial betas, per ROI.

Reads one ROI pattern cache in the ``extract_roi_betas.py`` format
(``<sub>_arm-<arm>[_set-<set>]_desc-type{b,c,d}_roipatterns.npz``, or any npz
with the same arrays) and measures how much stimulus-specific structure the
betas carry, using only exposure pairs that sit in DIFFERENT runs. Pairs inside
one run share scanner noise, drift and (for TYPED) the ridge operator's
neighbour mixing, so they are never used. Runs are keyed on (session, task, run):
TB encoding and retrieval number their runs independently.

Three measures per ROI, each with a shuffle floor:

  voxel_r      Voxel split-half reliability. Each repeated item contributes one
               cross-run exposure pair (its earliest exposure, and its earliest
               exposure in another run). For each voxel, Pearson r across items
               between the two exposure vectors; summarised as the median and
               90th percentile over voxels. Null: the second exposure's item
               labels are permuted within (run a, run b) blocks; read the
               value against null_mean, which is not zero after session z.
  item_disc    Pattern item discriminability. For every cross-run exposure pair
               (x of item i in run a, y of item i in run b), the pattern
               correlation r(x, y) minus the mean r(x, y_j) over candidate items
               j whose exposure y_j is in the same run b (``--candidates
               session``: the same session, minus x's own run). Candidates
               exclude any
               item that was ever adjacent (lag 1) to item i in one of i's runs,
               so serial neighbour mixing cannot inflate the baseline. Averaged
               per item, then over items. Also reported: ident_pct, the mean
               percentile rank of the true match among the candidates (chance
               0.5). Null: a random candidate stands in for the true match.
  anchor_disc  Cross-session identity of the items presented in every session
               (``sharedId == 1``). Per session, the mean pattern of each anchor;
               for every pair of different sessions, same-anchor minus
               different-anchor pattern r. Null: anchor labels permuted within
               session.

Patterns are z-scored per voxel within session first (``--no-zscore`` skips
it). Voxels are kept when finite in every trial, meanvol >= --meanvol-frac x the
ROI median, and median |beta| <= --beta-cap (percent signal change; drops
edge voxels whose betas are unbounded).

Output: a long TSV, one row per (roi, metric), with the observed value, the
null mean / 2.5 / 97.5 / 95th percentiles and a one-sided permutation p.

Usage:
    python item_reliability.py --cache <path/to/cache.npz> --out <out.tsv>
    python item_reliability.py --cache <cache.npz> --rois EVC VTC --n-null 500 --out <out.tsv>
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

META_COLS = ["session", "run", "task", "mmmId", "sharedId", "enCon", "onset"]


# ── loading ──────────────────────────────────────────────────────────────────

def load_trials(d) -> pd.DataFrame:
    missing = [c for c in META_COLS if c not in d.files]
    if missing:
        sys.exit(f"cache lacks trial metadata {missing}; expected the "
                 "extract_roi_betas.py cache format")
    t = pd.DataFrame({c: d[c] for c in META_COLS})
    t["mmmId"] = t["mmmId"].astype(str)
    t["session"] = t["session"].astype(str)
    t["run_key"] = (t["session"] + "|" + t["task"].astype(str) + "|"
                    + t["run"].astype(str))
    # within-run serial position, by onset
    t["pos"] = t.groupby("run_key")["onset"].rank(method="first").astype(int) - 1
    t["anchor"] = t["sharedId"] == 1
    return t


def clean_voxels(P, mv, meanvol_frac, beta_cap):
    ok = np.isfinite(P).all(axis=1)
    if mv is not None and meanvol_frac > 0:
        ok &= mv >= meanvol_frac * np.nanmedian(mv)
    with np.errstate(invalid="ignore"):
        ok &= np.nanmedian(np.abs(P), axis=1) <= beta_cap
    return ok


def session_zscore(P, session):
    Z = np.empty_like(P)
    for ses in np.unique(session):
        c = session == ses
        b = P[:, c]
        sd = b.std(axis=1, keepdims=True)
        sd[sd == 0] = np.inf
        Z[:, c] = (b - b.mean(axis=1, keepdims=True)) / sd
    return Z


def colnorm(P):
    """Centre and unit-normalise each column over voxels (so dot = Pearson r)."""
    Z = P - P.mean(axis=0, keepdims=True)
    n = np.linalg.norm(Z, axis=0, keepdims=True)
    n[n == 0] = np.inf
    return Z / n


# ── pair structure (depends only on the design) ──────────────────────────────

def adjacency(t: pd.DataFrame) -> dict[str, set[str]]:
    """item -> items ever presented at lag 1 from it within a run."""
    adj: dict[str, set[str]] = {}
    for _, g in t.sort_values(["run_key", "pos"]).groupby("run_key"):
        ids = g["mmmId"].to_numpy()
        for a, b in zip(ids[:-1], ids[1:]):
            adj.setdefault(a, set()).add(b)
            adj.setdefault(b, set()).add(a)
    return adj


def cross_run_pairs(t: pd.DataFrame, scope: str = "any"):
    """-> list of (item, col_x, col_y) for all cross-run exposure pairs of
    non-anchor items, and one 'first' pair per item for voxel_r.

    scope: "any" (every cross-run pair), "within-session" (both exposures in
    one session, different runs) or "cross-session"."""
    reg = t[~t["anchor"]]
    allp, first = [], []
    for item, g in reg.groupby("mmmId"):
        if g["run_key"].nunique() < 2:
            continue
        g = g.sort_values(["session", "task", "run", "onset"])
        cols, rk = g.index.to_numpy(), g["run_key"].to_numpy()
        ss = g["session"].to_numpy()
        ok_pairs = []
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                if rk[i] == rk[j]:
                    continue
                same = ss[i] == ss[j]
                if scope == "within-session" and not same:
                    continue
                if scope == "cross-session" and same:
                    continue
                ok_pairs.append((item, cols[i], cols[j]))
        if ok_pairs:
            allp += ok_pairs
            first.append(ok_pairs[0])
    return allp, first


# ── measures ─────────────────────────────────────────────────────────────────

def voxel_r(Z, first, t, n_null, rng, block_on="run_key"):
    items = [p[0] for p in first]
    i1 = np.array([p[1] for p in first])
    i2 = np.array([p[2] for p in first])
    # permute only within (run of exposure 1, run of exposure 2) blocks, so
    # run-level offsets left by session z-scoring are the same in the null
    block = (t.loc[i1, block_on].to_numpy().astype(object) + "~"
             + t.loc[i2, block_on].to_numpy().astype(object))

    def rvox(a_idx, b_idx):
        a, b = Z[:, a_idx], Z[:, b_idx]
        a = a - a.mean(1, keepdims=True)
        b = b - b.mean(1, keepdims=True)
        den = np.sqrt((a * a).sum(1) * (b * b).sum(1))
        den[den == 0] = np.nan
        return (a * b).sum(1) / den

    r = rvox(i1, i2)
    obs = {"median": np.nanmedian(r), "p90": np.nanpercentile(r, 90),
           "mean": np.nanmean(r)}
    groups = [np.flatnonzero(block == b) for b in np.unique(block)]
    null = {k: np.empty(n_null) for k in obs}
    if not any(len(g) > 1 for g in groups):
        sys.exit("voxel_r null: every permutation block holds one item; "
                 "use --candidates session")
    for k in range(n_null):
        perm = np.arange(len(i2))
        for g in groups:
            perm[g] = g[rng.permutation(len(g))]
        rn = rvox(i1, i2[perm])
        null["median"][k] = np.nanmedian(rn)
        null["p90"][k] = np.nanpercentile(rn, 90)
        null["mean"][k] = np.nanmean(rn)
    return obs, null, len(items)


def item_disc(C, allp, t, adj, n_null, rng, pool_on="run_key"):
    """C: trial x trial pattern-correlation matrix (all trials)."""
    rk = t[pool_on].to_numpy()
    run = t["run_key"].to_numpy()
    mmm = t["mmmId"].to_numpy()
    anchor = t["anchor"].to_numpy()
    run_cols = {k: np.flatnonzero((rk == k) & ~anchor) for k in np.unique(rk)}

    per_item_d, per_item_pct = {}, {}
    cand_sets = []   # for the null: (row r(x, candidates), index of true)
    for item, cx, cy in allp:
        for x, y in ((cx, cy), (cy, cx)):           # both directions
            pool = run_cols[rk[y]]
            bad = adj.get(item, set()) | {item}
            # never compare x with a trial from its own run (shared noise)
            keep = pool[~np.isin(mmm[pool], list(bad)) & (run[pool] != run[x])]
            if len(keep) < 5:
                continue
            match = C[x, y]
            others = C[x, keep]
            per_item_d.setdefault(item, []).append(match - others.mean())
            per_item_pct.setdefault(item, []).append((others < match).mean())
            cand_sets.append(others)
    d = np.array([np.mean(v) for v in per_item_d.values()])
    pct = np.array([np.mean(v) for v in per_item_pct.values()])
    obs = {"item_disc": d.mean(), "ident_pct": pct.mean()}

    # null: a random candidate plays the match, the rest are the candidates
    owner = []
    for item, v in per_item_d.items():
        owner += [item] * len(v)
    owner = np.array(owner)
    _, inv = np.unique(owner, return_inverse=True)
    cnt = np.bincount(inv)
    n = np.array([len(o) for o in cand_sets])
    off = np.concatenate([[0], np.cumsum(n)[:-1]])
    flat = np.concatenate(cand_sets)
    rank = np.concatenate([np.argsort(np.argsort(o)) for o in cand_sets])
    S = np.array([o.sum() for o in cand_sets])
    null = {k: np.empty(n_null) for k in obs}
    for k in range(n_null):
        j = off + (rng.random(len(n)) * n).astype(int)
        m = flat[j]
        dn = m - (S - m) / (n - 1)
        pn = rank[j] / (n - 1)
        null["item_disc"][k] = (np.bincount(inv, dn) / cnt).mean()
        null["ident_pct"][k] = (np.bincount(inv, pn) / cnt).mean()
    return obs, null, len(d), len(cand_sets)


def anchor_disc(Z, t, n_null, rng):
    a = t[t["anchor"]]
    ids = np.sort(a["mmmId"].unique())
    sess = np.sort(a["session"].unique())
    if len(ids) < 2 or len(sess) < 2:
        return None
    M = np.full((len(sess), len(ids), Z.shape[0]), np.nan)
    cols = {}
    for si, s in enumerate(sess):
        for ai, i in enumerate(ids):
            c = a.index[(a["session"] == s) & (a["mmmId"] == i)].to_numpy()
            if len(c):
                cols[(si, ai)] = c
                M[si, ai] = Z[:, c].mean(1)

    def score(Mx):
        X = Mx - Mx.mean(axis=2, keepdims=True)
        X /= np.linalg.norm(X, axis=2, keepdims=True) + 1e-12
        same, diff = [], []
        for s1 in range(len(sess)):
            for s2 in range(s1 + 1, len(sess)):
                R = X[s1] @ X[s2].T                    # anchors x anchors
                same.append(np.nanmean(np.diag(R)))
                off = R[~np.eye(len(ids), dtype=bool)]
                diff.append(np.nanmean(off))
        return np.nanmean(same) - np.nanmean(diff)

    obs = {"anchor_disc": score(M)}
    null = {"anchor_disc": np.empty(n_null)}
    for k in range(n_null):
        Mp = np.stack([M[si][rng.permutation(len(ids))] for si in range(len(sess))])
        null["anchor_disc"][k] = score(Mp)
    return obs, null, len(ids), len(sess)


# ── driver ───────────────────────────────────────────────────────────────────

def summarise(roi, metric, val, nd, extra):
    nd = nd[np.isfinite(nd)]
    return {"roi": roi, "metric": metric, "value": float(val),
            "null_mean": float(nd.mean()),
            "null_q025": float(np.percentile(nd, 2.5)),
            "null_q975": float(np.percentile(nd, 97.5)),
            "null_q95": float(np.percentile(nd, 95)),
            "p_perm": float((1 + (nd >= val).sum()) / (1 + len(nd))),
            "n_null": len(nd), **extra}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache", required=True, type=Path)
    ap.add_argument("--rois", nargs="*", default=None,
                    help="ROI names (default: every roi_names entry)")
    ap.add_argument("--n-null", type=int, default=200)
    ap.add_argument("--meanvol-frac", type=float, default=0.25)
    ap.add_argument("--beta-cap", type=float, default=100.0)
    ap.add_argument("--no-zscore", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pairs", default="any",
                    choices=["any", "within-session", "cross-session"],
                    help="which cross-run exposure pairs to use")
    ap.add_argument("--candidates", default="run", choices=["run", "session"],
                    help="item_disc comparison pool: other items in the repeat's "
                         "run (default) or session; also sets the voxel_r "
                         "permutation block (run pair or session pair)")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)

    if not args.cache.is_file():
        sys.exit(f"cache not found: {args.cache}")
    d = np.load(args.cache, allow_pickle=True)
    t = load_trials(d)
    roi_names = [str(r) for r in d["roi_names"]]
    rois = args.rois or roi_names
    unknown = [r for r in rois if f"patterns_{r}" not in d.files]
    if unknown:
        sys.exit(f"ROIs not in cache: {unknown}; cache has {roi_names}")
    head = {k: str(d[k]) for k in ("subject", "arm", "beta_type") if k in d.files}
    head.update(pairs=args.pairs, candidates=args.candidates)

    adj = adjacency(t)
    allp, first = cross_run_pairs(t, args.pairs)
    if not first:
        sys.exit(f"no exposure pairs with --pairs {args.pairs}")
    key = "run_key" if args.candidates == "run" else "session"
    print(f"{head} | {len(t)} trials | {len(first)} items with cross-run "
          f"pairs | {len(allp)} cross-run pairs", flush=True)

    rows = []
    for roi in rois:
        t0 = time.time()
        rng = np.random.default_rng(args.seed)
        P = d[f"patterns_{roi}"].astype(np.float64)
        mv = d[f"meanvol_{roi}"] if f"meanvol_{roi}" in d.files else None
        ok = clean_voxels(P, mv, args.meanvol_frac, args.beta_cap)
        P = P[ok]
        if P.shape[0] < 10:
            print(f"  {roi}: {P.shape[0]} voxels after cleaning; skipped")
            continue
        Z = P if args.no_zscore else session_zscore(P, t["session"].to_numpy())
        base = {**head, "n_vox": int(ok.sum()), "n_vox_roi": int(len(ok))}

        obs, null, n_items = voxel_r(Z, first, t, args.n_null, rng, key)
        for k in obs:
            rows.append(summarise(roi, f"voxel_r_{k}", obs[k], null[k],
                                  {**base, "n_items": n_items}))
        Cn = colnorm(Z)
        C = Cn.T @ Cn
        obs, null, n_items, n_pairs = item_disc(C, allp, t, adj, args.n_null, rng, key)
        for k in obs:
            rows.append(summarise(roi, k, obs[k], null[k],
                                  {**base, "n_items": n_items, "n_pairs": n_pairs}))
        res = anchor_disc(Z, t, max(args.n_null, 1000), rng)
        if res:
            obs, null, n_anc, n_ses = res
            rows.append(summarise(roi, "anchor_disc", obs["anchor_disc"],
                                  null["anchor_disc"],
                                  {**base, "n_items": n_anc, "n_sessions": n_ses}))
        msg = " ".join(f"{r['metric']}={r['value']:.4f}(p={r['p_perm']:.3f})"
                       for r in rows if r["roi"] == roi)
        print(f"  {roi} [{base['n_vox']}/{base['n_vox_roi']} vox, "
              f"{time.time() - t0:.0f}s] {msg}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, sep="\t", index=False, float_format="%.5g")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
