#!/usr/bin/env python3
"""A2 — across-subject alignment via the TB arm, evaluated on NAT ISC.

Pre-registered in mmmdata-agents docs/workbench/srm-stimulus-space/log.md
(2026-08-19, item 6 A2 + amendments): per-ROI orthogonal Procrustes fit on
TB patterns over the 1000 shared images; primary = pairwise NATencoding ISC
after alignment vs the anatomical baseline (MNI voxel identity, same
partition). The stimulus-space-mediated route (source brain -> EBind ->
target brain through each subject's own ridge maps) is reported alongside.
Inference: circular time-shift null within movie segment; movie-bootstrap
CI on the route-minus-anatomical difference.

Routes, per ordered pair (source X -> target Y) and ROI, on the joint
valid-voxel mask (identical MNI res-2 voxel indices across subjects,
verified in the log):
  anatomical  Y_hat = X_nat                      (voxel identity)
  procrustes  Y_hat = R^T X_nat,  R = argmin ||TB_X R - TB_Y||_F, R'R = I
  stimulus    Y_hat = E_Y(D_X(X_nat)),  D_X: voxels->EBind, E_Y: EBind->voxels
              (ridge both ways, alpha by RidgeCV on TB patterns)

Metric: per (voxel, movie) Pearson r over the movie's pair-common TRs;
median over voxels per movie; mean over the 60 movies. Delta vs anatomical
is computed movie-paired on the identical partition.

TB training arms: TYPED GLMsingle betas (primary) and rawTR trial patterns
(sensitivity, all subjects — motivated by the beta-reliability finding that
univariate TB beta reliability is at floor for sub-04/05).

Usage:
    python a2_procrustes.py --dry-run
    python a2_procrustes.py                          # betas / original
    python a2_procrustes.py --tb-source rawtr        # sensitivity arm
    python a2_procrustes.py --pipeline nordic        # variant sensitivity
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from a1_encoding import (  # noqa: E402
    N_MOVIES, RIDGE_ALPHAS, ROI_NAMES, SEED, SUBJECTS, PIPELINE,
    IMG_PARQUET, OUT_ROOT, beta_cache, build_feature_tables,
    load_image_features, load_nat_runs, load_tb_betas,
    mmmid_to_stimulus_id, movie_name_to_slug, rawtr_cache)

A2_DIR = OUT_ROOT / "a2"
N_PERMUTATIONS = 1000
N_BOOTSTRAP = 1000
ROUTES = ("anatomical", "procrustes", "stimulus")


# ── brain-side loading ────────────────────────────────────────────────────

def load_tb_rawtr(sub: str, pipeline: str
                  ) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Per-ROI (V x 1000) repeat-averaged TB rawTR trial patterns.

    Trial patterns are already Model-8 residualized and run-z-scored in the
    cache; no further session-z (unlike the betas, which get session-z per
    the GLMsingle convention).
    """
    files = sorted((rawtr_cache(pipeline) / sub).glob(
        f"{sub}_ses-*_task-TBencoding_run-*_desc-model8_roipatterns.npz"))
    pats: dict[str, list[np.ndarray]] = {roi: [] for roi in ROI_NAMES}
    ids = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        ids.append(d["mmmId"].astype(str))
        for roi in ROI_NAMES:
            pats[roi].append(d[f"patterns_{roi}"].astype(np.float64))
    mmm = np.concatenate(ids)
    uids = np.unique(mmm)
    out = {}
    for roi in ROI_NAMES:
        m = np.concatenate(pats[roi], axis=1)
        out[roi] = np.column_stack(
            [np.nanmean(m[:, mmm == i], axis=1) for i in uids])
    return out, uids


def load_tb_training(sub: str, tb_source: str, pipeline: str
                     ) -> tuple[dict[str, np.ndarray], np.ndarray]:
    if tb_source == "betas":
        return load_tb_betas(sub, pipeline)
    return load_tb_rawtr(sub, pipeline)


def nat_movie_segments(sub: str, pipeline: str) -> dict[str, dict]:
    """slug -> {roi: (V x n) repeat-averaged NAT rawTR time courses}."""
    runs = load_nat_runs(sub, pipeline, "rawtr")
    name_map = movie_name_to_slug()
    seg: dict[str, dict] = {}
    for run in runs:
        names, tr_idx = run["movie_name"], run["tr_index"]
        starts = np.flatnonzero(np.r_[True, tr_idx[1:] <= tr_idx[:-1]])
        bounds = np.r_[starts, len(tr_idx)]
        for s, e in zip(bounds[:-1], bounds[1:]):
            slug = name_map[names[s].strip().lower()]
            j = tr_idx[s:e]
            entry = seg.setdefault(slug, {"sum": {}, "cnt": {}})
            for roi in ROI_NAMES:
                block = run["patterns"][roi][:, s:e]
                acc = entry["sum"].setdefault(
                    roi, np.zeros((block.shape[0], 0)))
                if acc.shape[1] < j.max() + 1:
                    pad = np.zeros((block.shape[0], j.max() + 1 - acc.shape[1]))
                    entry["sum"][roi] = np.hstack([acc, pad])
                    entry["cnt"][roi] = np.hstack([
                        entry["cnt"].setdefault(
                            roi, np.zeros((block.shape[0], 0))), pad])
                finite = np.isfinite(block)
                entry["sum"][roi][:, j] += np.where(finite, block, 0.0)
                entry["cnt"][roi][:, j] += finite
    out = {}
    for slug, entry in seg.items():
        out[slug] = {}
        for roi in ROI_NAMES:
            cnt = entry["cnt"][roi]
            with np.errstate(invalid="ignore"):
                out[slug][roi] = np.where(
                    cnt > 0, entry["sum"][roi] / cnt, np.nan)
    return out


# ── core evaluation ───────────────────────────────────────────────────────

def _voxel_z(mat: np.ndarray) -> np.ndarray:
    """z-score each row (voxel) across columns (images)."""
    mu = mat.mean(axis=1, keepdims=True)
    sd = mat.std(axis=1, keepdims=True)
    return (mat - mu) / np.where(sd == 0, 1, sd)


def _norm_rows_per_movie(segs: list[np.ndarray]) -> list[np.ndarray]:
    """Center + unit-norm each voxel row within each movie.

    After this, per-(voxel, movie) Pearson r = elementwise-product row sum,
    and a circular shift of columns leaves the normalization intact.
    """
    out = []
    for m in segs:
        c = m - m.mean(axis=1, keepdims=True)
        n = np.linalg.norm(c, axis=1, keepdims=True)
        out.append(c / np.where(n == 0, 1, n))
    return out


def _lag_medians(src: list[np.ndarray], tgt: list[np.ndarray]
                 ) -> list[np.ndarray]:
    """Per movie: median-over-voxel r at EVERY circular lag of the source.

    Cross-correlation theorem: with rows centered and unit-normed within
    movie, r at source shift k is irfft(conj(rfft(a)) * rfft(b))[:, k].
    Element [movie][0] is the observed (unshifted) statistic; elements
    1..L-1 are the exact circular time-shift null support.
    """
    meds = []
    for a, b in zip(src, tgt):
        L = a.shape[1]
        c = np.fft.irfft(np.conj(np.fft.rfft(a, axis=1))
                         * np.fft.rfft(b, axis=1), n=L, axis=1)
        meds.append(np.median(c, axis=0))
    return meds


def evaluate_pair(src: str, tgt: str, roi: str,
                  tb: dict[str, dict[str, np.ndarray]],
                  nat: dict[str, dict[str, dict[str, np.ndarray]]],
                  R_cache: dict, feats: np.ndarray,
                  rng: np.random.Generator,
                  ranks: list[int] | None = None) -> list[dict]:
    """One ordered pair, one ROI. With ranks=None runs the pre-registered
    routes; with ranks=[k, ...] runs the low-rank Procrustes diagnostic
    (anatomical + rank-k partial Procrustes R_k = U_k V_k^T from the SVD of
    the TB cross-covariance; the full-rank route is k = V)."""
    from scipy.linalg import orthogonal_procrustes
    from sklearn.linear_model import RidgeCV

    slugs = sorted(set(nat[src]) & set(nat[tgt]))
    assert len(slugs) == N_MOVIES, f"{src}/{tgt}: {len(slugs)} movies"

    # pair-common movie lengths and joint valid-voxel mask
    tbx, tby = tb[src][roi], tb[tgt][roi]
    valid = np.isfinite(tbx).all(axis=1) & np.isfinite(tby).all(axis=1)
    segx, segy = [], []
    for slug in slugs:
        a, b = nat[src][slug][roi], nat[tgt][slug][roi]
        L = min(a.shape[1], b.shape[1])
        a, b = a[:, :L], b[:, :L]
        valid &= np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
        segx.append(a)
        segy.append(b)
    segx = [a[valid] for a in segx]
    segy = [b[valid] for b in segy]
    Xtr = _voxel_z(tbx[valid])                     # (Vv, 1000)
    Ytr = _voxel_z(tby[valid])

    # route maps
    key = frozenset((src, tgt))
    if key not in R_cache:
        R_cache[key] = {}

    if ranks is not None:
        # SVD of the TB cross-covariance M = Xtr Ytr^T once per unordered
        # pair; the reverse direction swaps U and V (M_yx = M_xy^T).
        if src not in R_cache[key]:
            U, _, Vt = np.linalg.svd(Xtr @ Ytr.T, full_matrices=False)
            R_cache[key][src] = (U, Vt)
            R_cache[key][tgt] = (Vt.T, U.T)
        U, Vt = R_cache[key][src]
        mapped = {"anatomical": segx}
        for k in ranks:
            kk = min(k, U.shape[1])
            Rk = U[:, :kk] @ Vt[:kk]
            mapped[f"procrustes_k{k}"] = [Rk.T @ a for a in segx]
        routes = tuple(mapped)
    else:
        if src not in R_cache[key]:
            R, _ = orthogonal_procrustes(Xtr.T, Ytr.T)  # x_row @ R ~ y_row
            R_cache[key][src] = R
            R_cache[key][tgt] = R.T
        R = R_cache[key][src]

        dec = RidgeCV(alphas=RIDGE_ALPHAS)
        dec.fit(Xtr.T, feats)                      # source voxels -> EBind
        enc = RidgeCV(alphas=RIDGE_ALPHAS)
        enc.fit(feats, Ytr.T)                      # EBind -> target voxels

        mapped = {
            "anatomical": segx,
            "procrustes": [R.T @ a for a in segx],
            "stimulus": [enc.predict(dec.predict(a.T)).T for a in segx],
        }
        routes = ROUTES

    tgt_n = _norm_rows_per_movie(segy)
    lag_meds = {route: _lag_medians(_norm_rows_per_movie(m), tgt_n)
                for route, m in mapped.items()}
    obs = {route: np.array([lm[0] for lm in lag_meds[route]])
           for route in routes}

    # circular time-shift null, offsets shared across routes (paired delta)
    lengths = np.array([a.shape[1] for a in segx])
    offsets = rng.integers(1, lengths, size=(N_PERMUTATIONS, len(lengths)))
    null = {route: np.stack(
        [lag_meds[route][i][offsets[:, i]] for i in range(len(lengths))],
        axis=1).mean(axis=1) for route in routes}

    # movie bootstrap on the paired per-movie difference vs anatomical
    boot_idx = rng.integers(0, N_MOVIES, size=(N_BOOTSTRAP, N_MOVIES))

    records = []
    for route in routes:
        isc = float(obs[route].mean())
        p_perm = float(
            (1 + (null[route] >= isc).sum()) / (1 + N_PERMUTATIONS))
        rec = {
            "source": src, "target": tgt, "roi": roi, "route": route,
            "isc": isc, "p_perm": p_perm,
            "n_voxels": int(valid.sum()), "n_movies": N_MOVIES,
        }
        if route != "anatomical":
            d = obs[route] - obs["anatomical"]
            d_null = null[route] - null["anatomical"]
            boots = d[boot_idx].mean(axis=1)
            rec.update({
                "delta_vs_anat": float(d.mean()),
                "p_delta": float(
                    (1 + (d_null >= d.mean()).sum()) / (1 + N_PERMUTATIONS)),
                "delta_ci95_lo": float(np.percentile(boots, 2.5)),
                "delta_ci95_hi": float(np.percentile(boots, 97.5)),
            })
        if route == "stimulus":
            rec["alpha_decode"] = float(np.atleast_1d(dec.alpha_)[0])
            rec["alpha_encode"] = float(np.atleast_1d(enc.alpha_)[0])
        records.append(rec)
        extra = ("" if route == "anatomical" else
                 f" delta={rec['delta_vs_anat']:+.4f} "
                 f"[{rec['delta_ci95_lo']:+.4f},{rec['delta_ci95_hi']:+.4f}] "
                 f"p_delta={rec['p_delta']:.4f}")
        print(f"  {src}->{tgt} {roi:<12} {route:<11} isc={isc:+.4f} "
              f"(p={p_perm:.4f}){extra}", flush=True)
    return records


# ── dry run ───────────────────────────────────────────────────────────────

def dry_run(pipeline: str) -> int:
    problems = 0

    def check(label, cond):
        nonlocal problems
        print(f"  [{'ok' if cond else 'MISSING'}] {label}")
        problems += not cond

    print(f"brain caches (pipeline={pipeline}):")
    for sub in SUBJECTS:
        beta = (beta_cache(pipeline) / sub /
                f"{sub}_task-TBencoding_desc-typed_roipatterns.npz")
        check(beta, beta.exists())
        n_tb = len(list((rawtr_cache(pipeline) / sub)
                        .glob("*task-TBencoding*npz")))
        check(f"{sub}: 42 TB rawTR caches (found {n_tb})", n_tb == 42)
        n_nat = len(list((rawtr_cache(pipeline) / sub)
                         .glob("*task-NATencoding*npz")))
        check(f"{sub}: 20 NAT rawTR caches (found {n_nat})", n_nat == 20)
    print("features:")
    check(IMG_PARQUET, IMG_PARQUET.exists())
    print("image-id alignment across subjects and arms:")
    ids = {}
    for sub in SUBJECTS:
        d = np.load(beta_cache(pipeline) / sub /
                    f"{sub}_task-TBencoding_desc-typed_roipatterns.npz",
                    allow_pickle=True)
        ids[sub] = np.unique(d["mmmId"].astype(str))
        check(f"{sub}: 1000 unique TB beta mmmIds (found {len(ids[sub])})",
              len(ids[sub]) == 1000)
    check("identical mmmId sets across subjects",
          all(np.array_equal(ids["sub-03"], ids[s]) for s in SUBJECTS[1:]))
    print("voxel-grid identity across subjects (anatomical baseline):")
    vox = {}
    for sub in SUBJECTS:
        f = next((rawtr_cache(pipeline) / sub).glob("*task-NATencoding*npz"))
        d = np.load(f, allow_pickle=True)
        vox[sub] = {roi: d[f"voxidx_{roi}"] for roi in ROI_NAMES}
    check("voxidx identical across subjects, all ROIs",
          all(np.array_equal(vox["sub-03"][roi], vox[s][roi])
              for s in SUBJECTS[1:] for roi in ROI_NAMES))
    print(f"\nDry run {'PASSED' if problems == 0 else f'FAILED ({problems})'}")
    return problems


# ── entry point ───────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tb-source", choices=["betas", "rawtr"],
                    default="betas",
                    help="TB training patterns: TYPED GLMsingle betas "
                    "(primary) or rawTR trial patterns (sensitivity)")
    ap.add_argument("--pipeline", choices=["original", "nordic"],
                    default=PIPELINE,
                    help="preprocessing variant, both arms (sensitivity)")
    ap.add_argument("--roi", choices=ROI_NAMES, help="one ROI (default: all)")
    ap.add_argument("--ranks", type=lambda s: [int(k) for k in s.split(",")],
                    help="low-rank Procrustes diagnostic: comma list of "
                    "ranks k (clamped to V per ROI); replaces the "
                    "pre-registered routes with anatomical + procrustes_k*")
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve all inputs and report, compute nothing")
    args = ap.parse_args()

    if args.dry_run:
        sys.exit(1 if dry_run(args.pipeline) else 0)

    build_feature_tables()

    # EBind image features, column-z-scored, rows aligned to the shared
    # sorted-mmmId order every TB loader uses
    id_map = mmmid_to_stimulus_id()
    img_wide = load_image_features()

    print(f"=== A2 Procrustes (tb_source={args.tb_source}, "
          f"pipeline={args.pipeline}) ===", flush=True)
    t0 = time.time()
    tb, nat = {}, {}
    ref_ids = None
    for sub in SUBJECTS:
        pats, ids = load_tb_training(sub, args.tb_source, args.pipeline)
        if ref_ids is None:
            ref_ids = ids
        assert np.array_equal(ids, ref_ids), f"{sub}: mmmId order differs"
        tb[sub] = pats
        nat[sub] = nat_movie_segments(sub, args.pipeline)
    X_img = img_wide.loc[[id_map[i] for i in ref_ids]].to_numpy(np.float64)
    feats = (X_img - X_img.mean(0)) / X_img.std(0)
    print(f"  loaded ({time.time() - t0:.0f}s)", flush=True)

    rois = [args.roi] if args.roi else ROI_NAMES
    rng = np.random.default_rng(SEED)
    R_cache: dict = {}
    records = []
    for roi in rois:
        R_cache.clear()                    # masks differ per ROI
        for src, tgt in permutations(SUBJECTS, 2):
            t1 = time.time()
            records += evaluate_pair(src, tgt, roi, tb, nat, R_cache,
                                     feats, rng, ranks=args.ranks)
            print(f"  ({time.time() - t1:.0f}s)", flush=True)

    result = pd.DataFrame.from_records(records)
    result["tb_source"] = args.tb_source
    result["pipeline"] = args.pipeline
    result["feature_space"] = "ebind"
    result["created_at"] = datetime.now(timezone.utc).isoformat()

    A2_DIR.mkdir(parents=True, exist_ok=True)
    tag = "" if args.tb_source == "betas" else f"_tb-{args.tb_source}"
    if args.pipeline != "original":
        tag += f"_pipe-{args.pipeline}"
    if args.ranks:
        tag += "_lowrank"
    if args.roi:
        tag += f"_roi-{args.roi}"
    out = A2_DIR / f"a2_summary{tag}.tsv"
    result.to_csv(out, sep="\t", index=False)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
