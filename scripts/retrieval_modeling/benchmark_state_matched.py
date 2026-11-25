#!/usr/bin/env python3
"""State-matched across-item baselines for the TB phase-pair benchmark.

benchmark_6cell.py's baseline for a within-item pair (a, b) is the mean
similarity of a to the other items' phase-Y trials in run(b), averaged with
the mirror. Every TB trial ends in a 1-3 key press (encoding: association
rating; retrieval: vividness), so a within-item pair can share a response,
and the state behind it, that its across-item baseline does not. This script
recomputes the baseline with the replacement trial required to match the
trial it replaces:

  none        no matching: reproduces benchmark_6cell's baseline exactly
              (asserted on every run).
  key         same key (rating 1/2/3) as the replaced trial.
  key_recon   same key and, for retrieval trials, the same reCon (retrieval
              in the encoding session vs a later one).

A pair whose replaced trial has no response, or whose matched pool is empty
on either side, gets no baseline at that level and is dropped from it (counted
in n_pairs_dropped); nothing is imputed. Pairs, strata and item-level
aggregation are benchmark_6cell's: non-anchor, cross-run, TB trials
(fin=False). The null permutes item identity within the same matched pool.

Cells: enc<->image, enc<->word (by reCon) and image<->word.

Outputs (per subject, beside the 6-cell outputs):
  <sub>[_set-<set>]_state_matched.tsv            cell x level
  <sub>[_set-<set>]_state_matched_items.tsv.gz   item level (within, across per level)

Usage:
    python benchmark_state_matched.py --subject sub-## [--types D] [--roi-set networks]
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
bs = _load_module("benchmark_strata", HERE / "benchmark_strata.py")
tb = b6.tb

SEED = 20261124
KEYS = {6.0: 1, 7.0: 2, 8.0: 3}
LEVELS = ("none", "key", "key_recon")
CELLS = [("enc", "image"), ("enc", "word"), ("image", "word")]


def trial_state(trials: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """-> (key 1-3 or nan, reCon for retrieval trials else nan) per trial."""
    key = trials["resp"].map(KEYS).to_numpy(dtype=float)
    recon = np.where(trials["subgroup"].isin(["image", "word"]), trials["reCon"].to_numpy(dtype=float),
                     np.nan)
    return key, recon


def matched_candidates(trials, pairs, pools, key, recon, level):
    """Per pair: (pool on b's side for a, pool on a's side for b), each matched to
    the trial it replaces. None when the replaced trial has no state at this level."""
    item = trials["mmmId"].to_numpy()
    rk = trials["run_key"].to_numpy()
    out = []
    for a, b in zip(pairs.a.to_numpy(), pairs.b.to_numpy()):
        kb = pools[rk[b]]; kb = kb[item[kb] != item[a]]     # replaces b
        ka = pools[rk[a]]; ka = ka[item[ka] != item[b]]     # replaces a
        if level != "none":
            if np.isnan(key[a]) or np.isnan(key[b]):
                out.append(None); continue
            kb = kb[key[kb] == key[b]]
            ka = ka[key[ka] == key[a]]
            if level == "key_recon":
                if not np.isnan(recon[b]):
                    kb = kb[recon[kb] == recon[b]]
                if not np.isnan(recon[a]):
                    ka = ka[recon[ka] == recon[a]]
        out.append((kb, ka) if len(kb) and len(ka) else None)
    return out


def baseline(R, pairs, cands) -> np.ndarray:
    a, b = pairs.a.to_numpy(), pairs.b.to_numpy()
    return np.array([np.nan if c is None else 0.5 * (R[a[n], c[0]].mean() + R[c[1], b[n]].mean())
                     for n, c in enumerate(cands)])


def null_deltas(R, pairs, cands, n_perm, rng) -> np.ndarray:
    """benchmark_6cell's null (b's side replaced by a random pool member), drawn
    from the matched pool."""
    a = pairs.a.to_numpy()
    base = pairs["across"].to_numpy()
    pool = [c[0] for c in cands]
    lens = np.array([len(p) for p in pool])
    starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
    flat = np.concatenate(pool)
    w = 1.0 / pairs.groupby("mmmId")["a"].transform("size").to_numpy()
    w /= pairs["mmmId"].nunique()
    out = np.empty(n_perm)
    step = max(1, int(2e7 // max(1, len(pool))))
    for p0 in range(0, n_perm, step):
        draws = rng.random((min(step, n_perm - p0), len(pool)))
        pick = flat[starts[None, :] + (draws * lens[None, :]).astype(int)]
        out[p0:p0 + pick.shape[0]] = ((R[a[None, :], pick] - base[None, :]) * w[None, :]).sum(1)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject", required=True)
    ap.add_argument("--arm", default="siloed", choices=list(b6.ARM_FILES))
    ap.add_argument("--types", nargs="+", default=["D"], choices=["B", "C", "D"])
    ap.add_argument("--norms", nargs="+", default=["runmean"], choices=["raw", "runmean"])
    ap.add_argument("--roi-set", default="pattern6")
    ap.add_argument("--rois", nargs="+", default=None)
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
        t0 = time.time()
        rs = args.roi_set
        trials, _, _ = b6.load_arm(cache_root, args.subject, args.arm, t, rs, rois=[])
        trials["resp"] = bs.load_resp(cache_root, args.subject, args.arm, t, rs)
        key, recon = trial_state(trials)
        pools = b6.run_pools(trials)
        tbt = trials[~trials.anchor & ~trials.fin]
        print(f"[{args.arm} TYPE{t}] {len(trials)} trials; TB non-anchor keyed: "
              + ", ".join(f"{p} {np.isfinite(key[tbt.index[tbt.subgroup == p]]).mean():.1%}"
                          for p in b6.PHASES), flush=True)
        cell_pairs, cell_cands = {}, {}
        for X, Y in CELLS:
            p = b6.build_pairs(trials, X, Y)
            p = p[~p.anchor & ~p.same_run & ~p.fin].reset_index(drop=True)
            cell_pairs[(X, Y)] = p
            cell_cands[(X, Y)] = {lv: matched_candidates(trials, p, pools, key, recon, lv) for lv in LEVELS}
        for roi in (args.rois or b6.cache_roi_names(cache_root, args.subject, args.arm, t, rs)):
            _, patterns, mvs = b6.load_arm(cache_root, args.subject, args.arm, t, rs, rois=[roi])
            ok = b6.clean_voxels(patterns[roi], mvs.get(roi), args.meanvol_frac, args.beta_cap)
            for norm in args.norms:
                R = b6.corr_matrix(b6.normalize(patterns[roi][ok], trials, norm))
                meta = dict(subject=args.subject, arm=args.arm, beta=f"TYPE{t}", roi=roi, norm=norm,
                            n_voxels=int(ok.sum()))
                for (X, Y), pairs in cell_pairs.items():
                    pairs = pairs.copy()
                    pairs["within"] = R[pairs.a.to_numpy(), pairs.b.to_numpy()]
                    ref = b6.across_baseline(R, trials, pairs, pools)
                    per_item = None
                    for lv in LEVELS:
                        c = cell_cands[(X, Y)][lv]
                        pairs[f"across_{lv}"] = baseline(R, pairs, c)
                        if lv == "none":
                            assert np.allclose(pairs["across_none"], ref), "unmatched baseline != benchmark_6cell"
                        keep = pairs[f"across_{lv}"].notna().to_numpy()
                        strata = ["reCon"] if X == "enc" else []
                        for vals, g in (pairs[keep].groupby(strata) if strata else [((), pairs[keep])]):
                            gi = g.index.to_numpy()
                            g = g.assign(across=g[f"across_{lv}"])
                            items = g.groupby("mmmId").agg(within=("within", "mean"),
                                                           across=("across", "mean"))
                            delta = float((items.within - items.across).mean())
                            null = null_deltas(R, g, [c[i] for i in gi], args.n_perm, rng)
                            rc = float(vals[0] if isinstance(vals, tuple) else vals) if strata else np.nan
                            n_all = int((pairs.reCon == rc).sum()) if strata else len(pairs)
                            rows.append(meta | dict(
                                cell=f"{X}<->{Y}", level=lv, reCon=rc, n_items=len(items), n_pairs=len(g),
                                n_pairs_dropped=n_all - len(g), within=float(items.within.mean()),
                                across=float(items.across.mean()), delta=delta, null_sd=float(null.std()),
                                p_perm=float((np.sum(null >= delta) + 1) / (args.n_perm + 1))))
                        it = (pairs.assign(across=pairs[f"across_{lv}"]).groupby("mmmId")
                              .agg(within=("within", "mean"), across=("across", "mean"),
                                   reCon=("reCon", "first"), n_pairs=("across", "count")))
                        it = it.rename(columns={"across": f"across_{lv}", "n_pairs": f"n_pairs_{lv}"})
                        per_item = it if per_item is None else per_item.join(it.drop(columns=["within", "reCon"]))
                    for mmm, r in per_item.iterrows():
                        item_rows.append(meta | dict(cell=f"{X}<->{Y}", mmmId=int(mmm)) | r.to_dict())
            print(f"  {roi:12s} voxels {ok.sum()}/{len(ok)}  {time.time() - t0:.0f}s", flush=True)

    stem = out_dir / (args.subject + ("" if args.roi_set == "pattern6" else f"_set-{args.roi_set}"))
    pd.DataFrame(rows).to_csv(f"{stem}_state_matched.tsv", sep="\t", index=False, float_format="%.5g")
    pd.DataFrame(item_rows).to_csv(f"{stem}_state_matched_items.tsv.gz", sep="\t", index=False,
                                   float_format="%.5g")
    print(f"\nwrote {stem}_state_matched{{.tsv,_items.tsv.gz}}")


if __name__ == "__main__":
    main()
