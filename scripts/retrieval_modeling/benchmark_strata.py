#!/usr/bin/env python3
"""Stratified reads of the TB phase-pair benchmark: behaviour and repetition.

Companion to benchmark_6cell.py, whose pair, baseline and null definitions it
reuses unchanged (within-item vs run-matched across-item Pearson similarity,
item-level aggregation, item identity permuted within run). Only the grouping of
pairs differs: each stratum below gets its own delta, null and p. The 6-cell
outputs are not touched.

Strata (non-anchor, cross-run, TB trials only — fin=False):
  vividness  the retrieval trial's rating. Keys 6/7/8 are ratings 1/2/3
             (least / somewhat / most vivid), following the order of the
             task's key list. enc<->image and enc<->word: crossed with reCon,
             because ratings depend strongly on the retrieval delay.
             image<->word: by each side's rating separately (viv_image,
             viv_word), crossed with the item's reCon (both retrievals share it).
  afc        the item's later two-alternative recognition outcome
             (beh/ task-TB2AFC, one test per non-anchor item): hc_correct /
             lc_correct / incorrect. Crossed with reCon (image<->word: the
             item's reCon).
  enCon      the item's encoding condition (1 single, 2 repeated x3,
             3 triplet x3). Crossed with reCon for enc<->ret and image<->word; enc<->enc
             (tuning-exposed — read TYPEB beside TYPED) by enCon alone.
Anchors (the six super-repeat items, sharedId=1; anchor=True, cross-run):
  role       triplet context (pairId 1-3) vs single context (pairId 75-77),
             every cell, fin=False and fin=True kept apart.
  pairs      every anchor pair with its two sessions, written out per pair so
             similarity can be read against session and session gap.

Outputs (per subject, beside the 6-cell outputs):
  <sub>_strata.tsv            one row per cell x stratifier x level
  <sub>_strata_items.tsv.gz   item level, same labels
  <sub>_anchor_pairs.tsv.gz   per anchor pair: sessions, within, across
A stratum's n_items is reported as is; consumers decide what is too thin.

Usage:
    python benchmark_strata.py --subject sub-## [--types B D] [--norms runmean]
"""

from __future__ import annotations

import argparse
import importlib.util
import re
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
VIVIDNESS_KEYS = {6.0: 1, 7.0: 2, 8.0: 3}
TRIPLET_ANCHOR_PAIRIDS = {1, 2, 3}
SINGLE_ANCHOR_PAIRIDS = {75, 76, 77}


# ── behaviour ────────────────────────────────────────────────────────────

def load_afc(bids_root: Path, sub: str, trials: pd.DataFrame) -> pd.Series:
    """mmmId -> 2AFC outcome for non-anchor items (word is unique per item)."""
    files = sorted(bids_root.glob(f"{sub}/ses-*/beh/{sub}_ses-*_task-TB2AFC_*_beh.tsv"))
    if not files:
        sys.exit(f"ERROR: no TB2AFC beh files under {bids_root / sub}/ses-*/beh/")
    a = pd.concat([pd.read_csv(f, sep="\t") for f in files], ignore_index=True)
    ret = trials[trials.subgroup.isin(["image", "word"]) & ~trials.anchor & ~trials.fin]
    w2m = ret.drop_duplicates("word").set_index("word").mmmId
    if ret.groupby("word").mmmId.nunique().max() != 1:
        sys.exit("ERROR: a word maps to more than one non-anchor item; cannot key 2AFC on word")
    a = a[a.word.isin(w2m.index)].copy()
    dup = a.word.duplicated(keep=False)
    if dup.any():
        sys.exit(f"ERROR: {dup.sum()} 2AFC rows test a non-anchor word more than once")
    acc = a.trial_accuracy.astype(float)
    hc = a.resp.isin([1, 4])
    out = np.where(acc == 1, np.where(hc, "hc_correct", "lc_correct"), "incorrect")
    return pd.Series(out, index=a.word.map(w2m).to_numpy(), name="afc")


def annotate(trials: pd.DataFrame, afc: pd.Series) -> pd.DataFrame:
    t = trials.copy()
    t["viv"] = t["resp"].map(VIVIDNESS_KEYS)
    t.loc[t.subgroup == "enc", "viv"] = np.nan
    item_encon = t[t.subgroup == "enc"].groupby("mmmId").enCon.first()
    t["item_enCon"] = t.mmmId.map(item_encon)
    t["afc"] = t.mmmId.map(afc)
    t["ses_num"] = t.session.map(lambda s: int(re.sub(r"\D", "", s)))
    pid = t.pairId.astype(float)
    t["role"] = np.where(pid.isin(TRIPLET_ANCHOR_PAIRIDS), "triplet",
                         np.where(pid.isin(SINGLE_ANCHOR_PAIRIDS), "single", None))
    t.loc[~t.anchor, "role"] = None
    return t


def augment(pairs: pd.DataFrame, t: pd.DataFrame) -> pd.DataFrame:
    a, b = pairs.a.to_numpy(), pairs.b.to_numpy()
    p = pairs.copy()
    p["viv_a"], p["viv_b"] = t.viv.to_numpy()[a], t.viv.to_numpy()[b]
    p["enCon"] = t.item_enCon.to_numpy()[a]
    # an item's two retrievals share its reCon, so retrieval<->retrieval
    # cells can be crossed with the retrieval delay too
    p["item_reCon"] = np.where(np.isnan(t.reCon.to_numpy()[b]), t.reCon.to_numpy()[a],
                               t.reCon.to_numpy()[b])
    p["afc"] = t.afc.to_numpy()[a]
    p["role"] = t.role.to_numpy()[a]
    p["ses_a"], p["ses_b"] = t.ses_num.to_numpy()[a], t.ses_num.to_numpy()[b]
    return p


# ── strata ───────────────────────────────────────────────────────────────

def stratified(R, trials, pairs, pools, cell, stratifier, keys, n_perm, rng, meta, item_rows):
    rows = []
    for vals, g in pairs.groupby(keys, dropna=True):
        vals = vals if isinstance(vals, tuple) else (vals,)
        level = "|".join(f"{k}={v:g}" if isinstance(v, (int, float, np.floating)) else f"{k}={v}"
                         for k, v in zip(keys, vals))
        items = g.groupby("mmmId").agg(within=("within", "mean"), across=("across", "mean"),
                                       n_pairs=("a", "size"))
        delta = float((items.within - items.across).mean())
        null = b6.null_deltas(R, trials, g, pools, n_perm, rng)
        label = dict(cell=cell, stratifier=stratifier, level=level,
                     anchor=bool(g.anchor.iloc[0]), fin=bool(g.fin.iloc[0]))
        rows.append(meta | label | dict(
            n_items=len(items), n_pairs=int(len(g)), within=float(items.within.mean()),
            across=float(items.across.mean()), delta=delta, null_sd=float(null.std()),
            p_perm=float((np.sum(null >= delta) + 1) / (n_perm + 1)), n_perm=n_perm))
        for mmm, it in items.iterrows():
            item_rows.append(meta | label | dict(mmmId=int(mmm), within=it.within,
                                                  across=it.across, n_pairs=int(it.n_pairs)))
    return rows


def plan(X: str, Y: str):
    """(stratifier, keys) applied to the non-anchor, cross-run, TB pairs of a cell."""
    if X == "enc" and Y in ("image", "word"):
        return [("vividness", ["reCon", "viv_b"]), ("afc", ["reCon", "afc"]),
                ("enCon", ["reCon", "enCon"])]
    if (X, Y) == ("image", "word"):
        return [("viv_image", ["item_reCon", "viv_a"]), ("viv_word", ["item_reCon", "viv_b"]),
                ("afc", ["item_reCon", "afc"]), ("enCon", ["item_reCon", "enCon"])]
    if (X, Y) == ("enc", "enc"):
        return [("enCon", ["enCon"])]
    return []


# ── main ─────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject", required=True)
    ap.add_argument("--arm", default="siloed", choices=list(b6.ARM_FILES))
    ap.add_argument("--types", nargs="+", default=["B", "D"], choices=["B", "C", "D"])
    ap.add_argument("--norms", nargs="+", default=["runmean"], choices=["raw", "runmean"])
    ap.add_argument("--rois", nargs="+", default=ps.PATTERN_ROI_NAMES)
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

    rows, item_rows, anchor_rows = [], [], []
    for t in args.types:
        t0 = time.time()
        trials, patterns, mvs = b6.load_arm(cache_root, args.subject, args.arm, t)
        trials["resp"] = load_resp(cache_root, args.subject, args.arm, t)
        trials["word"] = load_field(cache_root, args.subject, args.arm, t, "word")
        trials["pairId"] = load_field(cache_root, args.subject, args.arm, t, "pairId").astype(float)
        afc = load_afc(bids_root, args.subject, trials)
        ta = annotate(trials, afc)
        pools = b6.run_pools(trials)
        print(f"[{args.arm} TYPE{t}] {len(trials)} trials; 2AFC outcomes "
              f"{afc.value_counts().to_dict()}", flush=True)
        for roi in args.rois:
            ok = b6.clean_voxels(patterns[roi], mvs.get(roi), args.meanvol_frac, args.beta_cap)
            for norm in args.norms:
                R = b6.corr_matrix(b6.normalize(patterns[roi][ok], trials, norm))
                meta = dict(subject=args.subject, arm=args.arm, beta=f"TYPE{t}", roi=roi,
                            norm=norm, n_voxels=int(ok.sum()), n_voxels_total=int(len(ok)))
                for X, Y in b6.CELLS:
                    pairs = b6.build_pairs(trials, X, Y)
                    if pairs.empty:
                        continue
                    pairs["within"] = R[pairs.a.to_numpy(), pairs.b.to_numpy()]
                    pairs["across"] = b6.across_baseline(R, trials, pairs, pools)
                    pairs = augment(pairs, ta)
                    cell = f"{X}<->{Y}"
                    core = pairs[~pairs.anchor & ~pairs.same_run & ~pairs.fin]
                    for name, keys in plan(X, Y):
                        rows += stratified(R, trials, core, pools, cell, name, keys,
                                           args.n_perm, rng, meta, item_rows)
                    anc = pairs[pairs.anchor & ~pairs.same_run]
                    for fin, g in anc.groupby("fin"):
                        rows += stratified(R, trials, g, pools, cell, "anchor_role", ["role"],
                                           args.n_perm, rng, meta, item_rows)
                    if not anc.empty:
                        anchor_rows.append(anc.assign(cell=cell, **meta)[
                            list(meta) + ["cell", "mmmId", "role", "fin", "ses_a", "ses_b",
                                          "within", "across"]])
            print(f"  {roi:12s} voxels {ok.sum()}/{len(ok)}  {time.time() - t0:.0f}s", flush=True)

    stem = out_dir / args.subject
    pd.DataFrame(rows).to_csv(f"{stem}_strata.tsv", sep="\t", index=False, float_format="%.5g")
    pd.DataFrame(item_rows).to_csv(f"{stem}_strata_items.tsv.gz", sep="\t", index=False,
                                   float_format="%.5g")
    pd.concat(anchor_rows).to_csv(f"{stem}_anchor_pairs.tsv.gz", sep="\t", index=False,
                                  float_format="%.5g")
    print(f"\nwrote {out_dir}/{args.subject}_{{strata.tsv,strata_items.tsv.gz,anchor_pairs.tsv.gz}}")


def _npz(cache_root, sub, arm, t, phase_dir):
    p = (b6.cache_dir(cache_root, sub, phase_dir)
         / f"{sub}_arm-{phase_dir}_desc-type{t.lower()}_roipatterns.npz")
    return np.load(p, allow_pickle=True)


def load_field(cache_root, sub, arm, t, field) -> np.ndarray:
    """A per-trial cache field, concatenated in load_arm's phase order."""
    return np.concatenate([_npz(cache_root, sub, arm, t, b6.ARM_FILES[arm][p])[field]
                           for p in b6.PHASES])


def load_resp(cache_root, sub, arm, t) -> np.ndarray:
    return pd.to_numeric(pd.Series(load_field(cache_root, sub, arm, t, "resp").astype(str)),
                         errors="coerce").to_numpy()


if __name__ == "__main__":
    main()
