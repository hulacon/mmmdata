#!/usr/bin/env python3
"""6-cell phase-pair pattern-similarity benchmark for the TB single-trial fits.

Within-item vs run-matched across-item similarity, per subject x ROI x beta
type x arm, for the six phase pairs over {enc, image, word}:
enc<->enc, enc<->image, enc<->word, image<->image, word<->word, image<->word.

Definitions (design record: mmmdata-agents docs/workbench/retrieval-modeling/,
DECIDED 2026-08-26; conventions verified on the fits 2026-09-10):

  pair       (a, b): trial a of item i in phase X, trial b of item i in phase Y,
             a != b. Stratified by `same_run`, by `fin` (either trial is a
             ses-30 FINretrieval trial — a months-delayed second retrieval,
             reported apart from the TB retrievals since the 2026-09-11
             TB+FIN refit; under fin=True `reCon` is still the item's TB
             condition), and, for enc<->ret cells, by the retrieval trial's
             `reCon` (1 = retrieved in the encoding session, 2 = in a later
             session). Encoding repeats are 3 presentations within ONE
             session over 1-3 runs, so enc<->enc has both `same_run` strata;
             image<->word is one pair per item, same session, different runs
             (fin=False) plus, for the 240 FIN items, one cross-session pair
             (fin=True). image<->image / word<->word non-anchor within-item
             pairs exist only for the 120 FIN items per cue (fin=True).
  within     r(a, b) — Pearson across the ROI's voxels.
  across     run-matched baseline: mean r(a, k) over trials k of phase Y in
             run(b) whose item differs from i and is not an anchor, averaged
             with the mirror (trials of phase X in run(a) against b).
  item level first: within_i / across_i average a item's pairs before any cell
             statistic; delta = mean over items of (within_i - across_i).
  null       item identity permuted within run: each pair's Y-side trial is
             replaced by a random trial of another item in the same run and
             phase (the same pool the baseline averages over), delta
             recomputed; p = P(delta_null >= delta) over --n-perm draws.
  anchors    the 6 super-repeat items (sharedId=1, 42x encoded, 14x retrieved
             per cue) are the ONLY within-item pairs of image<->image and
             word<->word; they are kept but reported as their own stratum
             everywhere and never enter an across baseline.
  modality   Settles-when 4: from the across-item structure of the retrieval
             trials, index = mean(r_ii, r_ww) - r_iw over different-item,
             different-run, same-session pairs; null = run->modality labels
             re-drawn within session (runs are cue-pure). Raw betas only:
             run-mean removal deletes the modality signal by construction.

Voxels: finite in every trial, meanvol >= --meanvol-frac x the ROI median
(edge/air voxels get unbounded percent-signal betas), median |beta| over
trials <= --beta-cap. Two normalizations are reported: raw, and runmean (the
per-run mean pattern of each phase subtracted).

Inputs: the ROI caches written by extract_roi_betas.py. Outputs (per subject):
  <out_root>/results/retrieval_modeling/<sub>/<sub>_6cell.tsv       cell level
  <out_root>/results/retrieval_modeling/<sub>/<sub>_6cell_items.tsv.gz item level
  <out_root>/results/retrieval_modeling/<sub>/<sub>_modality.tsv

Arms: `siloed` reads enc + ret-image + ret-word (the production TB+FIN
fits, DECIDED 2026-09-11); `siloed-tbonly` reads the retained TB-only
retrieval caches (`ret-*-tbonly`) with the same enc cache; `pooled` reads the
retained pooled caches (its fit tree was deleted 2026-09-11).

Usage:
    python benchmark_6cell.py --subject sub-## [--arms siloed siloed-tbonly pooled] [--types B C D]
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).resolve().parent.parent


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


tb = _load_module("glmsingle_tb", SCRIPTS / "glmsingle_tb.py")
ps = _load_module("pattern_similarity_shared", SCRIPTS / "pattern_similarity" / "shared.py")

CACHE_TREE = "pattern_similarity"
PHASES = ["enc", "image", "word"]
CELLS = [("enc", "enc"), ("enc", "image"), ("enc", "word"),
         ("image", "image"), ("word", "word"), ("image", "word")]
ARM_FILES = {
    "siloed": {"enc": "enc", "image": "ret-image", "word": "ret-word"},
    "siloed-tbonly": {"enc": "enc", "image": "ret-image-tbonly", "word": "ret-word-tbonly"},
}
ANCHORS = set(range(995, 1001))
SEED = 20260910


# ── loading ──────────────────────────────────────────────────────────────

def cache_dir(cache_root: Path, sub: str, arm_dir: str) -> Path:
    return cache_root / "cache" / "glmsingle_tb" / sub / arm_dir


def load_arm(cache_root: Path, sub: str, arm: str, t: str):
    """-> trials DataFrame (one row per column), {roi: (V, N)}, {roi: meanvol}."""
    files = ([(ARM_FILES[arm][p], p) for p in PHASES] if arm in ARM_FILES
             else [("pooled", None)])
    frames, pats, mvs = [], {r: [] for r in ps.PATTERN_ROI_NAMES}, {}
    for arm_dir, phase in files:
        p = cache_dir(cache_root, sub, arm_dir) / f"{sub}_arm-{arm_dir}_desc-type{t.lower()}_roipatterns.npz"
        if not p.exists():
            sys.exit(f"ERROR: cache missing: {p}\n  run extract_roi_betas.py --subject {sub} --arm {arm_dir}")
        d = np.load(p, allow_pickle=True)
        df = pd.DataFrame({
            "session": d["session"].astype(str), "run": d["run"].astype(int),
            "task": d["task"].astype(str), "subgroup": d["subgroup"].astype(str),
            "mmmId": pd.to_numeric(pd.Series(d["mmmId"].astype(str))).astype(int).to_numpy(),
            "sharedId": d["sharedId"].astype(float), "reCon": d["reCon"].astype(float),
            "enCon": d["enCon"].astype(float), "onset": d["onset"].astype(float),
        })
        if phase is not None:
            assert (df["subgroup"] == phase).all(), (p, df["subgroup"].unique())
        frames.append(df)
        for roi in ps.PATTERN_ROI_NAMES:
            pats[roi].append(d[f"patterns_{roi}"])
            if roi not in mvs and f"meanvol_{roi}" in d.files:
                mvs[roi] = d[f"meanvol_{roi}"]
    trials = pd.concat(frames, ignore_index=True)
    trials["run_key"] = (trials["session"] + "/" + trials["subgroup"] + "/"
                         + trials["run"].astype(str))
    trials["anchor"] = trials["mmmId"].isin(ANCHORS)
    trials["fin"] = trials["task"] == "FINretrieval"
    patterns = {roi: np.concatenate(pats[roi], axis=1) for roi in ps.PATTERN_ROI_NAMES}
    return trials, patterns, mvs


def clean_voxels(P: np.ndarray, mv, meanvol_frac: float, beta_cap: float):
    ok = np.isfinite(P).all(axis=1)
    if mv is not None:
        ok &= mv >= meanvol_frac * np.nanmedian(mv)
    with np.errstate(invalid="ignore"):
        ok &= np.nanmedian(np.abs(P), axis=1) <= beta_cap
    return ok


def normalize(P: np.ndarray, trials: pd.DataFrame, how: str) -> np.ndarray:
    P = P.astype(np.float64)
    if how == "runmean":
        for _, idx in trials.groupby("run_key").indices.items():
            P[:, idx] -= P[:, idx].mean(axis=1, keepdims=True)
    return P


def corr_matrix(P: np.ndarray) -> np.ndarray:
    Z = P - P.mean(axis=0, keepdims=True)
    Z /= (Z.std(axis=0, keepdims=True) + 1e-12)
    return (Z.T @ Z) / Z.shape[0]


# ── pairs ─────────────────────────────────────────────────────────────────

def build_pairs(trials: pd.DataFrame, X: str, Y: str) -> pd.DataFrame:
    """All within-item (a, b) pairs between phases X and Y, with strata."""
    tx = trials[trials.subgroup == X]
    ty = trials[trials.subgroup == Y]
    m = tx.reset_index().merge(ty.reset_index(), on="mmmId", suffixes=("_a", "_b"))
    m = m[m.index_a != m.index_b]
    if X == Y:
        m = m[m.index_a < m.index_b]
    m = m.rename(columns={"index_a": "a", "index_b": "b"})
    m["same_run"] = m.run_key_a == m.run_key_b
    m["fin"] = m.fin_a | m.fin_b
    m["anchor"] = m.anchor_a
    if X == "enc" and Y in ("image", "word"):
        m["reCon"] = m.reCon_b
    else:
        m["reCon"] = np.nan
    return m[["mmmId", "a", "b", "same_run", "fin", "anchor", "reCon"]].reset_index(drop=True)


def run_pools(trials: pd.DataFrame) -> dict:
    """run_key -> array of non-anchor trial indices in that run."""
    return {k: np.asarray(v) for k, v in
            trials[~trials.anchor].groupby("run_key").indices.items()}


def across_baseline(R, trials, pairs, pools) -> np.ndarray:
    """Symmetric run-matched baseline per pair (see module docstring)."""
    item = trials["mmmId"].to_numpy()
    rk = trials["run_key"].to_numpy()
    out = np.empty(len(pairs))
    a_arr, b_arr = pairs.a.to_numpy(), pairs.b.to_numpy()
    for n, (a, b) in enumerate(zip(a_arr, b_arr)):
        kb = pools[rk[b]]; kb = kb[item[kb] != item[a]]
        ka = pools[rk[a]]; ka = ka[item[ka] != item[b]]
        out[n] = 0.5 * (R[a, kb].mean() + R[ka, b].mean())
    return out


def null_deltas(R, trials, pairs, pools, n_perm: int, rng) -> np.ndarray:
    """Item-permuted-within-run null of the item-level mean delta."""
    item = trials["mmmId"].to_numpy()
    rk = trials["run_key"].to_numpy()
    a_arr, b_arr = pairs.a.to_numpy(), pairs.b.to_numpy()
    base = pairs["across"].to_numpy()
    cand = []
    for a, b in zip(a_arr, b_arr):
        kb = pools[rk[b]]
        cand.append(kb[item[kb] != item[a]])
    lens = np.array([len(c) for c in cand])
    starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
    cand_flat = np.concatenate(cand)
    # item-level averaging weights: each pair contributes 1/n_pairs(item)
    w = 1.0 / pairs.groupby("mmmId")["a"].transform("size").to_numpy()
    w /= pairs["mmmId"].nunique()
    out = np.empty(n_perm)
    step = max(1, int(2e7 // max(1, len(cand))))      # bound the (perm x pair) block
    for p0 in range(0, n_perm, step):
        draws = rng.random((min(step, n_perm - p0), len(cand)))
        pick = cand_flat[starts[None, :] + (draws * lens[None, :]).astype(int)]
        out[p0:p0 + pick.shape[0]] = ((R[a_arr[None, :], pick] - base[None, :]) * w[None, :]).sum(1)
    return out


def cell_rows(R, trials, X, Y, pools, n_perm, rng, meta: dict, item_rows: list):
    pairs = build_pairs(trials, X, Y)
    if pairs.empty:
        return []
    pairs["within"] = R[pairs.a.to_numpy(), pairs.b.to_numpy()]
    pairs["across"] = across_baseline(R, trials, pairs, pools)
    rows = []
    # strata: anchors apart; then same_run x fin; then reCon where defined
    strata = []
    for anchor in (False, True):
        sub = pairs[pairs.anchor == anchor]
        if sub.empty:
            continue
        keys = ["same_run", "fin"] + (["reCon"] if sub.reCon.notna().any() else [])
        for vals, g in sub.groupby(keys, dropna=False):
            vals = vals if isinstance(vals, tuple) else (vals,)
            strata.append((anchor, dict(zip(keys, vals)), g))
    for anchor, key, g in strata:
        items = g.groupby("mmmId").agg(within=("within", "mean"), across=("across", "mean"),
                                       n_pairs=("a", "size"))
        delta = float((items.within - items.across).mean())
        null = null_deltas(R, trials, g, pools, n_perm, rng)
        p = float((np.sum(null >= delta) + 1) / (n_perm + 1))
        label = dict(cell=f"{X}<->{Y}", anchor=anchor, same_run=bool(key["same_run"]),
                     fin=bool(key["fin"]), reCon=key.get("reCon", np.nan))
        rows.append(meta | label | dict(
            n_items=len(items), n_pairs=int(len(g)), within=float(items.within.mean()),
            across=float(items.across.mean()), delta=delta, null_sd=float(null.std()),
            p_perm=p, n_perm=n_perm))
        for mmm, it in items.iterrows():
            item_rows.append(meta | label | dict(mmmId=int(mmm), within=it.within,
                                                  across=it.across, n_pairs=int(it.n_pairs)))
    return rows


# ── modality (Settles-when 4) ─────────────────────────────────────────────

def modality_rows(R, trials, n_perm, rng, meta):
    ret = trials[trials.subgroup.isin(["image", "word"]) & ~trials.anchor]
    idx = ret.index.to_numpy()
    ses = ret.session.to_numpy(); rk = ret.run_key.to_numpy()
    item = ret.mmmId.to_numpy(); mod = ret.subgroup.to_numpy()
    Rr = R[np.ix_(idx, idx)]
    I, J = np.triu_indices(len(idx), 1)
    ok = (ses[I] == ses[J]) & (rk[I] != rk[J]) & (item[I] != item[J])
    I, J, r = I[ok], J[ok], Rr[I[ok], J[ok]]

    def index(mod_vec):
        same = mod_vec[I] == mod_vec[J]
        return float(r[same].mean() - r[~same].mean()), float(r[same].mean()), float(r[~same].mean())

    obs, r_same, r_diff = index(mod)
    # null: re-draw run->modality labels within session
    runs = ret.groupby(["session", "run_key"])["subgroup"].first().reset_index()
    run_ids = {k: n for n, k in enumerate(runs.run_key)}
    trial_run = np.array([run_ids[k] for k in rk])
    per_session = [g.index.to_numpy() for _, g in runs.groupby("session")]
    labels0 = runs.subgroup.to_numpy()
    null = np.empty(n_perm)
    for p in range(n_perm):
        labels = labels0.copy()
        for ix in per_session:
            labels[ix] = labels[rng.permutation(ix)]
        null[p] = index(labels[trial_run])[0]
    return [meta | dict(r_same_modality=r_same, r_diff_modality=r_diff, index=obs,
                        null_sd=float(null.std()),
                        p_perm=float((np.sum(null >= obs) + 1) / (n_perm + 1)),
                        n_pairs=int(len(r)), n_perm=n_perm)]


# ── main ─────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject", required=True)
    ap.add_argument("--arms", nargs="+", default=["siloed", "siloed-tbonly"],
                    choices=["siloed", "siloed-tbonly", "pooled"])
    ap.add_argument("--types", nargs="+", default=["B", "C", "D"], choices=["B", "C", "D"])
    ap.add_argument("--norms", nargs="+", default=["raw", "runmean"], choices=["raw", "runmean"])
    ap.add_argument("--rois", nargs="+", default=ps.PATTERN_ROI_NAMES)
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--meanvol-frac", type=float, default=0.25)
    ap.add_argument("--beta-cap", type=float, default=100.0)
    ap.add_argument("--cache-root", default=None)
    ap.add_argument("--out-root", default=None)
    args = ap.parse_args()

    cfg = tb.load_config()
    bids_root = Path(cfg["bids_project_dir"])
    cache_root = Path(args.cache_root) if args.cache_root else bids_root / "derivatives" / CACHE_TREE
    out_dir = (Path(args.out_root) if args.out_root else cache_root) / "results" / "retrieval_modeling" / args.subject
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    rows, item_rows, mod_rows = [], [], []
    for arm in args.arms:
        for t in args.types:
            t0 = time.time()
            trials, patterns, mvs = load_arm(cache_root, args.subject, arm, t)
            pools = run_pools(trials)
            print(f"[{arm} TYPE{t}] {len(trials)} trials, "
                  f"{trials.subgroup.value_counts().to_dict()}, {len(pools)} runs", flush=True)
            for roi in args.rois:
                ok = clean_voxels(patterns[roi], mvs.get(roi), args.meanvol_frac, args.beta_cap)
                for norm in args.norms:
                    P = normalize(patterns[roi][ok], trials, norm)
                    R = corr_matrix(P)
                    meta = dict(subject=args.subject, arm=arm, beta=f"TYPE{t}", roi=roi,
                                norm=norm, n_voxels=int(ok.sum()), n_voxels_total=int(len(ok)))
                    for X, Y in CELLS:
                        rows += cell_rows(R, trials, X, Y, pools, args.n_perm, rng, meta, item_rows)
                    if norm == "raw":
                        # runs are cue-pure, so run-mean removal deletes the
                        # modality signal by construction: read it on raw only
                        mod_rows += modality_rows(R, trials, args.n_perm, rng, meta)
                print(f"  {roi:12s} voxels {ok.sum()}/{len(ok)}  done {time.time() - t0:.0f}s", flush=True)

    cells = pd.DataFrame(rows)
    cells.to_csv(out_dir / f"{args.subject}_6cell.tsv", sep="\t", index=False, float_format="%.5g")
    pd.DataFrame(item_rows).to_csv(out_dir / f"{args.subject}_6cell_items.tsv.gz", sep="\t",
                                   index=False, float_format="%.5g")
    pd.DataFrame(mod_rows).to_csv(out_dir / f"{args.subject}_modality.tsv", sep="\t",
                                  index=False, float_format="%.5g")
    print(f"\nwrote {out_dir}")
    show = cells[(cells.norm == "runmean") & (~cells.anchor) & (~cells.same_run)]
    print(show.pivot_table(index=["arm", "beta", "roi"], columns=["cell", "fin", "reCon"],
                           values="delta", dropna=False).round(3).to_string())


if __name__ == "__main__":
    main()
