#!/usr/bin/env python3
"""Anchor benchmark: pattern similarity and rotation-by-transfer for the six
super-repeat items (design record Settles-when 9).

The anchors are encoded three times in every trial-based session and
retrieved once per cue in each, always in the encoding session. Across-item
comparisons stay INSIDE the anchor set (five foils). Three readouts per
subject x ROI, every phase pair among {enc, ret-word, ret-image}:

  averaged   one pattern per anchor per phase = mean over every instance
             (encoding: all exposures of all sessions; retrieval: all
             sessions). Same- vs other-anchor correlation (delta) and
             identification among the six (2AFC over the 30 ordered pairs,
             rank-1). Null = the exact 720 relabellings of one side.
             Ceiling = odd vs even sessions within one phase. CI = bootstrap
             over sessions.
  loso       leave-one-session-out: the encoding mean of the other sessions
             vs the held-out session's single retrieval trial (cross-run,
             session-matched -- the reCon-1 regime); plus the same-session
             encoding mean vs that retrieval, and word- vs image-cued
             retrieval of the same session. Mean over sessions; null = joint
             per-session relabellings; CI = bootstrap over sessions.
  transfer   the main pass's pooled orthogonal map (fitted on the non-anchor
             items only, rank k* read from the pass's class table) applied
             to the averaged anchor encodings: identification among the six
             with vs without the map. No map is fitted on the anchors.

Phase means are removed with the NON-anchor items' means of that phase (the
same centring the main pass applies), so six items are not centring
themselves. Voxels are cleaned by the benchmark rule as in fit_pair.

Usage:
    python anchor_benchmark.py --subject sub-## [--rois mPFC AngularGyrus ...]
        [--pass-root <out_root of the main pass>] [--out-root DIR] [--n-boot 2000]
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
import basis  # noqa: E402
import fit_pair as fp  # noqa: E402
import maps  # noqa: E402
import score  # noqa: E402

PHASES = ("enc", "ret-word", "ret-image")
PAIRS = (("enc", "ret-word"), ("enc", "ret-image"), ("ret-word", "ret-image"))
PERMS = np.array(list(itertools.permutations(range(6))))          # the exact null


def rowcorr(A, B):
    return score._row_corr_matrix(A, B)


def pair_stats(X, Y):
    """delta (same - other), 2AFC over ordered pairs, rank-1 fraction for
    X (6, k) vs Y (6, k) with matched rows."""
    C = rowcorr(X, Y)
    d = np.diag(C)
    off = C[~np.eye(len(C), dtype=bool)]
    wins = d[:, None] > C
    np.fill_diagonal(wins, False)
    n = len(C)
    return {"delta": float(d.mean() - off.mean()), "acc_2afc": float(wins.sum() / (n * (n - 1))),
            "acc_rank1": float((wins.sum(axis=1) == n - 1).mean()), "same": float(d.mean()),
            "other": float(off.mean())}


def exact_null(X, Y, stat: str):
    """The statistic under every relabelling of Y's rows (720 values)."""
    return np.array([pair_stats(X, Y[p])[stat] for p in PERMS])


def load_subject(args):
    cfg = fp.tb.load_config()
    output_dir = Path(cfg["output_dir"])
    cache_root = Path(args.cache_root) if args.cache_root else output_dir / fp.CACHE_TREE
    design_root = Path(args.design_root) if args.design_root else output_dir / fp.DESIGN_TREE
    out_root = Path(args.out_root) if args.out_root else output_dir / fp.DESIGN_TREE / "anchors"
    arm_map = {p: (a if p == "enc" else a.replace("-tbonly", args.arm_suffix)) for p, a in fp.PHASE_ARMS.items()}
    design = fp.load_design(design_root, args.subject)
    return cache_root, design, arm_map, out_root, output_dir


def session_patterns(arm: dict, phase: str):
    """Anchor patterns per (anchor, session): (anchors, sessions, A x S x V).
    Encoding = mean of that session's exposures; retrieval = the trial."""
    t = arm["trials"]
    a = t[t["anchor"].astype(bool)]
    anchors = np.array(sorted(a["mmmId"].astype(int).unique()))
    sessions = sorted(a["session"].unique())
    X = np.full((len(anchors), len(sessions), arm["P"].shape[1]), np.nan)
    for i, an in enumerate(anchors):
        for j, s in enumerate(sessions):
            idx = a.index[(a["mmmId"].astype(int) == an) & (a["session"] == s)]
            if len(idx):
                X[i, j] = arm["P"][idx].mean(axis=0)
    return anchors, sessions, X


def nonanchor_mean(arm: dict, phase: str, items: np.ndarray):
    """Phase mean over non-anchor items (exposure-averaged for encoding)."""
    t = arm["trials"]
    ids = t["mmmId"].astype(int).to_numpy()
    X, _ = basis.average_exposures(arm["P"], ids, items)
    return X.mean(axis=0), X


def pass_rank(pass_root: Path, subject: str, rung: str, roi: str, pair: str) -> int | None:
    """Median nested-CV rank of the forward orthogonal map in the main pass."""
    stem = f"{subject}_rung-{rung}_roi-{roi}_pair-{pair.replace(':', '')}_desc-typed"
    f = pass_root / "fits" / subject / f"{stem}_transformation_class.tsv"
    if not f.exists():
        return None
    tc = pd.read_csv(f, sep="\t", na_values=["n/a"], keep_default_na=False)
    sel = tc[(tc["class"] == "procrustes") & (tc["direction"] == "forward") & tc["rank_selected"].astype(bool)]
    return int(sel["rank"].median()) if len(sel) else None


def run_roi(roi: str, rung: str, args, cache_root, design, arm_map, rng) -> dict:
    arms = {p: fp.load_arm(cache_root, args.subject, p, args.beta_type, roi, design, arm_map) for p in PHASES}
    keep = np.ones(arms["enc"]["P"].shape[1], dtype=bool)
    for a in arms.values():
        keep &= basis.clean_voxels(a["P"].T, a["meanvol"], 0.25, 100.0)
    for a in arms.values():
        a["P"] = a["P"][:, keep]
    items, _, _ = fp.item_sets({p: a["trials"] for p, a in arms.items()})
    anchors, sessions, S = {}, {}, {}
    means, nonanchor = {}, {}
    for p in PHASES:
        anchors[p], sessions[p], S[p] = session_patterns(arms[p], p)
        means[p], nonanchor[p] = nonanchor_mean(arms[p], p, items)
    common = sorted(set.intersection(*[set(s) for s in sessions.values()]))
    assert all((anchors[p] == anchors["enc"]).all() for p in PHASES), "anchor sets differ across phases"
    for p in PHASES:
        cols = [sessions[p].index(s) for s in common]
        S[p] = S[p][:, cols, :] - means[p]                     # centred on the non-anchor phase mean
    n_s = len(common)
    meta = {"subject": args.subject, "rung": rung, "roi": roi, "n_vox": int(keep.sum()),
            "n_anchors": len(anchors["enc"]), "n_sessions": n_s}
    rows = {"averaged": [], "loso": [], "transfer": []}

    # ── averaged instances ────────────────────────────────────────────────
    bar = {p: np.nanmean(S[p], axis=1) for p in PHASES}
    for x, y in PAIRS:
        st = pair_stats(bar[x], bar[y])
        nd, na = exact_null(bar[x], bar[y], "delta"), exact_null(bar[x], bar[y], "acc_2afc")
        boots = {"delta": [], "acc_2afc": []}
        for _ in range(args.n_boot):
            idx = rng.integers(0, n_s, n_s)
            b = pair_stats(np.nanmean(S[x][:, idx], axis=1), np.nanmean(S[y][:, idx], axis=1))
            boots["delta"].append(b["delta"]); boots["acc_2afc"].append(b["acc_2afc"])
        rows["averaged"].append(dict(meta, pair=f"{x}:{y}", **st,
                                     delta_p_exact=score.null_p(st["delta"], nd),
                                     acc_p_exact=score.null_p(st["acc_2afc"], na),
                                     delta_ci_lo=np.quantile(boots["delta"], .025), delta_ci_hi=np.quantile(boots["delta"], .975),
                                     acc_ci_lo=np.quantile(boots["acc_2afc"], .025), acc_ci_hi=np.quantile(boots["acc_2afc"], .975)))
    for p in PHASES:                                              # split-half ceilings
        odd, even = np.nanmean(S[p][:, 0::2], axis=1), np.nanmean(S[p][:, 1::2], axis=1)
        st = pair_stats(odd, even)
        rows["averaged"].append(dict(meta, pair=f"{p}:{p} (odd:even sessions)", **st,
                                     delta_p_exact=score.null_p(st["delta"], exact_null(odd, even, "delta")),
                                     acc_p_exact=score.null_p(st["acc_2afc"], exact_null(odd, even, "acc_2afc")),
                                     delta_ci_lo=np.nan, delta_ci_hi=np.nan, acc_ci_lo=np.nan, acc_ci_hi=np.nan))

    # ── leave-one-session-out ─────────────────────────────────────────────
    def loso(kind, fx, fy):
        per = []
        nulls = np.zeros((args.n_null, n_s))
        for j in range(n_s):
            X, Y = fx(j), fy(j)
            per.append(pair_stats(X, Y))
            for d in range(args.n_null):
                nulls[d, j] = pair_stats(X, Y[PERMS[rng.integers(1, len(PERMS))]])["acc_2afc"]
        acc = np.array([q["acc_2afc"] for q in per]); delta = np.array([q["delta"] for q in per])
        m, lo, hi = score.fold_ci(acc, seed=args.seed)
        rows["loso"].append(dict(meta, comparison=kind, n_folds=n_s, acc_2afc=m, acc_ci_lo=lo, acc_ci_hi=hi,
                                 delta=float(delta.mean()), acc_rank1=float(np.mean([q["acc_rank1"] for q in per])),
                                 null_p=score.null_p(m, nulls.mean(axis=1)), null_mean=float(nulls.mean())))
    others = lambda p, j: np.nanmean(np.delete(S[p], j, axis=1), axis=1)
    for r in ("ret-word", "ret-image"):
        loso(f"enc(other sessions):{r}(held-out)", lambda j: others("enc", j), lambda j, r=r: S[r][:, j])
        loso(f"enc(same session):{r}(same session)", lambda j: S["enc"][:, j], lambda j, r=r: S[r][:, j])
    loso("ret-word:ret-image (same session)", lambda j: S["ret-word"][:, j], lambda j: S["ret-image"][:, j])

    # ── rotation by transfer ──────────────────────────────────────────────
    for r in ("ret-word", "ret-image"):
        pair = f"enc:{r}"
        E, R = nonanchor["enc"], nonanchor[r]
        nm = basis.BlockNormaliser().fit(E, R, None)
        En, Rn = nm.transform(E, "E"), nm.transform(R, "R")
        W, _ = basis.shared_basis(En, Rn, k_max=args.k_max)
        k = pass_rank(Path(args.pass_root), args.subject, rung, roi, pair) if args.pass_root else None
        k_src = "pass" if k else "default"
        k = min(k or args.default_rank, W.shape[1])
        Wk = W[:, :k]
        Q = maps.OrthogonalProcrustes().fit(En @ Wk, Rn @ Wk).Q
        Ea = (bar["enc"] + means["enc"] - nm.mean_e) / nm.scale @ Wk       # anchors in the pass's basis
        Ra = (bar[r] + means[r] - nm.mean_r) / nm.scale @ Wk
        ident, mapped = pair_stats(Ea, Ra), pair_stats(Ea @ Q, Ra)
        rows["transfer"].append(dict(meta, pair=pair, rank=k, rank_source=k_src,
                                     acc_identity=ident["acc_2afc"], acc_mapped=mapped["acc_2afc"],
                                     gain=mapped["acc_2afc"] - ident["acc_2afc"],
                                     delta_identity=ident["delta"], delta_mapped=mapped["delta"],
                                     acc_identity_p_exact=score.null_p(ident["acc_2afc"], exact_null(Ea, Ra, "acc_2afc")),
                                     acc_mapped_p_exact=score.null_p(mapped["acc_2afc"], exact_null(Ea @ Q, Ra, "acc_2afc"))))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject", required=True)
    ap.add_argument("--rois", nargs="*", default=None, help="default: every ladder ROI in the cache")
    ap.add_argument("--beta-type", default="D")
    ap.add_argument("--arm-suffix", default="-tbonly")
    ap.add_argument("--pass-root", default=None, help="main pass out_root (for the pooled map's rank)")
    ap.add_argument("--default-rank", type=int, default=50)
    ap.add_argument("--k-max", type=int, default=200)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-null", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260923)
    ap.add_argument("--cache-root", default=None)
    ap.add_argument("--design-root", default=None)
    ap.add_argument("--out-root", default=None, help="default <output_dir>/neural_rotation/anchors")
    args = ap.parse_args()
    t0 = time.time()
    cache_root, design, arm_map, out_root, _ = load_subject(args)
    probe = np.load(fp.cache_path(cache_root, args.subject, arm_map["enc"], args.beta_type), allow_pickle=True)
    rois = args.rois or [str(r) for r in probe["roi_names"]]
    rungs = dict(zip([str(r) for r in probe["roi_names"]], [str(r) for r in probe["roi_rungs"]])) if "roi_rungs" in probe.files else {}
    rng = np.random.default_rng(args.seed)
    out = {"averaged": [], "loso": [], "transfer": []}
    for roi in rois:
        rows = run_roi(roi, rungs.get(roi, "i"), args, cache_root, design, arm_map, rng)
        for k, v in rows.items():
            out[k] += v
        print(f"  {roi}: done ({time.time() - t0:.0f}s)", flush=True)
    out_root.mkdir(parents=True, exist_ok=True)
    for k, v in out.items():
        p = out_root / f"{args.subject}_desc-anchors_{k}.tsv"
        pd.DataFrame(v).to_csv(p, sep="\t", index=False, na_rep="n/a", float_format="%.6g")
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
