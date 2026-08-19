#!/usr/bin/env python3
"""A1 — within-subject cross-arm encoding (srm-stimulus-space pilot).

Pre-registered in mmmdata-agents docs/workbench/srm-stimulus-space/log.md
(2026-08-19, incl. same-day amendments): ridge encoding EBind -> voxels fit
on TBencoding GLMsingle betas (original variant), evaluated on NATencoding
Model-8-residualized TR patterns (original variant). Primary metric:
movie-identification normalized rank accuracy over the 60 movies
(chance 0.5); secondary: median voxelwise prediction r. Inference:
permutation of movie identity.

Inputs (all pre-existing; this script computes, never preprocesses):
  - TB betas / NAT rawTR per-ROI caches from scripts/pattern_similarity/
    (derivatives/pattern_similarity/cache/{glmsingle,rawtr}/original/)
  - EBind features consumed via `psytwill features` (Contract B surface):
    step 0 builds two parquet tables under
    derivatives/stimuli_features/psytwill/ from the per-set extractor CSVs
  - stimulus registry (stimuli/stimulus_registry/) for mmmId -> stimulus_id
    and movie_name -> slug (case-insensitive, incl. variants)

Timing model (pre-registration item 5): NAT rawTR cache column j of a movie
samples the response at movie-time 4.5 + 1.5*j s (the cache's +3 TR shift).
Primary: features HRF-convolved (SPM double-gamma, 0.5 s grid) and evaluated
at those times. Sensitivity (--lag-model fixed): raw features at 1.5*j s.

Usage:
    python a1_encoding.py --dry-run                  # resolve + report inputs
    python a1_encoding.py --build-features           # step 0 only
    python a1_encoding.py --sub sub-03               # one subject
    python a1_encoding.py                            # all three subjects
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import tomllib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# ── configuration (config TOMLs, not hard-coded paths) ───────────────────

_CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "config"


def _load_config() -> dict:
    cfg = {}
    for name in ("base.toml", "local.toml"):
        p = _CONFIG_DIR / name
        if p.exists():
            with open(p, "rb") as f:
                for section in tomllib.load(f).values():
                    if isinstance(section, dict):
                        cfg.update(section)
    return cfg


_CFG = _load_config()
BIDS_ROOT = Path(_CFG.get("bids_project_dir", "/gpfs/projects/hulacon/shared/mmmdata"))
DERIV_ROOT = BIDS_ROOT / "derivatives"

SUBJECTS = ["sub-03", "sub-04", "sub-05"]
ROI_NAMES = ["EVC", "EAC", "Hippocampus", "AG", "Precuneus", "mPFC"]
TR = 1.5
HRF_SHIFT_S = 4.5              # the rawTR cache's +3 TR alignment shift
FEATURE_DT = 0.5               # extractor frame grid (stated in sidecars)
N_MOVIES = 60
N_PERMUTATIONS = 1000
SEED = 20260819
RIDGE_ALPHAS = np.logspace(1, 6, 11)

PIPELINE = "original"          # pre-registration amendment (a): non-NORDIC
PS_CACHE = DERIV_ROOT / "pattern_similarity" / "cache"
NAT_CHUNK_S = 4.5              # NAT GLMsingle pseudo-trial duration


def beta_cache(pipeline: str) -> Path:
    return PS_CACHE / "glmsingle" / pipeline


def rawtr_cache(pipeline: str) -> Path:
    return PS_CACHE / "rawtr" / pipeline

FEAT_ROOT = DERIV_ROOT / "stimuli_features"
PSYTWILL_DIR = FEAT_ROOT / "psytwill"
IMG_PARQUET = PSYTWILL_DIR / "shared1000_ebind_features.parquet"
MOV_PARQUET = PSYTWILL_DIR / "movies_ebind_features.parquet"

REGISTRY = BIDS_ROOT / "stimuli" / "stimulus_registry"
OUT_ROOT = DERIV_ROOT / "srm_stimulus_space"
A1_DIR = OUT_ROOT / "a1"

FEATURE_SPACE = "ebind"        # pre-registration item 2 (primary)


# ── step 0: features via psytwill (Contract B consumption) ───────────────

def feature_inputs() -> tuple[list[Path], list[Path]]:
    img = [FEAT_ROOT / "shared1000" / "ebind.csv"]
    movies = pd.read_csv(REGISTRY / "movies.tsv", sep="\t")
    mov = [FEAT_ROOT / "movies" / slug / "ebind.csv"
           for slug in movies["stimulus_id"]]
    return img, mov


def build_feature_tables() -> None:
    """Aggregate extractor CSVs into the two psytwill feature parquets."""
    from psytwill.features import build_features

    img, mov = feature_inputs()
    PSYTWILL_DIR.mkdir(parents=True, exist_ok=True)
    for inputs, out in ((img, IMG_PARQUET), (mov, MOV_PARQUET)):
        if out.exists():
            print(f"[features] {out.name} exists — keeping (delete to rebuild)")
            continue
        t0 = time.time()
        summary = build_features(inputs, out)
        print(f"[features] {out.name}: {summary['rows']} rows, "
              f"{summary['n_stimuli']} stimuli ({time.time() - t0:.0f}s)")


def load_image_features() -> pd.DataFrame:
    """(stimulus_id x 1024) EBind image matrix from the psytwill table."""
    t = pd.read_parquet(IMG_PARQUET, columns=["stimulus_id", "feature", "value"])
    wide = t.pivot(index="stimulus_id", columns="feature", values="value")
    return wide[sorted(wide.columns)]


def load_movie_features() -> dict[str, np.ndarray]:
    """slug -> (n_frames x 1024) EBind frame matrix on the 0.5 s grid."""
    t = pd.read_parquet(
        MOV_PARQUET, columns=["stimulus_id", "time", "feature", "value"])
    out = {}
    for slug, g in t.groupby("stimulus_id"):
        wide = g.pivot(index="time", columns="feature", values="value")
        wide = wide[sorted(wide.columns)].sort_index()
        grid = np.asarray(wide.index, dtype=float)
        assert np.allclose(np.diff(grid), FEATURE_DT), (
            f"{slug}: frame grid is not uniform {FEATURE_DT}s")
        out[str(slug)] = wide.to_numpy(dtype=np.float64)
    return out


# ── registry joins ────────────────────────────────────────────────────────

def mmmid_to_stimulus_id() -> dict[str, str]:
    reg = pd.read_csv(REGISTRY / "shared1000.tsv", sep="\t")
    return dict(zip(reg["mmmId"].astype(str), reg["stimulus_id"]))


def movie_name_to_slug() -> dict[str, str]:
    """Case-insensitive events movie_name (+variants) -> registry slug."""
    reg = pd.read_csv(REGISTRY / "movies.tsv", sep="\t")
    lookup = {}
    for _, row in reg.iterrows():
        names = [row["movie_name"]]
        if isinstance(row.get("movie_name_variants"), str):
            names += [v for v in row["movie_name_variants"].split(";") if v]
        for name in names:
            lookup[name.strip().lower()] = row["stimulus_id"]
    return lookup


# ── HRF (pre-registration item 5: SPM canonical double-gamma) ────────────

def spm_hrf(dt: float = FEATURE_DT, duration: float = 32.0) -> np.ndarray:
    """SPM canonical double-gamma (peak 6 s, undershoot 16 s, ratio 1/6)."""
    from scipy.stats import gamma

    t = np.arange(0, duration + dt, dt)
    h = gamma.pdf(t, 6.0, scale=1.0) - gamma.pdf(t, 16.0, scale=1.0) / 6.0
    return h / h.sum()


def movie_design(frames: np.ndarray, tr_index: np.ndarray,
                 lag_model: str) -> np.ndarray:
    """Feature rows for one movie's cached TR/chunk columns.

    rawTR cache column j samples the response at movie-time 4.5 + 1.5*j s.
    hrf: predict via convolution evaluated at that acquisition time.
    fixed: raw features at the stimulus time the cache aligns to (1.5*j).
    chunkmean (NAT GLMsingle): betas are deconvolved per voxel (FITHRF),
    so chunk j is predicted by the mean raw features over its stimulus
    window [4.5*j, 4.5*(j+1)) — no HRF model, no shift.
    """
    n = frames.shape[0]
    grid = np.arange(n) * FEATURE_DT
    if lag_model == "chunkmean":
        rows = np.empty((len(tr_index), frames.shape[1]))
        for i, j in enumerate(tr_index):
            sel = (grid >= NAT_CHUNK_S * j) & (grid < NAT_CHUNK_S * (j + 1))
            rows[i] = frames[sel if sel.any() else [-1]].mean(axis=0)
        return rows
    if lag_model == "hrf":
        h = spm_hrf()
        conv = np.apply_along_axis(
            lambda c: np.convolve(c, h), 0, frames)  # (n + len(h) - 1, D)
        conv_t = np.arange(conv.shape[0]) * FEATURE_DT
        query = HRF_SHIFT_S + TR * tr_index
        src, src_t = conv, conv_t
    else:
        query = TR * tr_index
        src, src_t = frames, grid
    rows = np.empty((len(tr_index), frames.shape[1]))
    for d in range(frames.shape[1]):
        rows[:, d] = np.interp(query, src_t, src[:, d])
    return rows


# ── brain-side loading ────────────────────────────────────────────────────

def _session_zscore(pat: np.ndarray, session: np.ndarray) -> np.ndarray:
    """z-score each voxel's values within session (GLMsingle convention)."""
    z = np.full_like(pat, np.nan)
    for ses in np.unique(session):
        cols = session == ses
        block = pat[:, cols]
        mu = np.nanmean(block, axis=1, keepdims=True)
        sd = np.nanstd(block, axis=1, keepdims=True)
        sd[sd == 0] = np.nan
        z[:, cols] = (block - mu) / sd
    return z


def load_tb_betas(sub: str, pipeline: str
                  ) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Per-ROI (V x 1000) session-z-scored, repeat-averaged beta matrices."""
    f = (beta_cache(pipeline) / sub /
         f"{sub}_task-TBencoding_desc-typed_roipatterns.npz")
    d = np.load(f, allow_pickle=True)
    mmm = d["mmmId"].astype(str)
    session = d["session"].astype(str)
    ids = np.unique(mmm)
    out = {}
    for roi in ROI_NAMES:
        z = _session_zscore(d[f"patterns_{roi}"].astype(np.float64), session)
        out[roi] = np.column_stack(
            [np.nanmean(z[:, mmm == i], axis=1) for i in ids])
    return out, ids


def load_nat_runs(sub: str, pipeline: str, nat_source: str) -> list[dict]:
    """NATencoding response segments as run-shaped dicts.

    rawtr: one dict per run cache (Model-8 residualized, z-scored TRs).
    glmsingle: the single NAT chunk-beta cache, session-z-scored, split
    into per-(session, run) dicts with chunk_idx playing tr_index's role.
    """
    if nat_source == "rawtr":
        files = sorted((rawtr_cache(pipeline) / sub).glob(
            f"{sub}_ses-*_task-NATencoding_run-*_desc-model8_roipatterns.npz"))
        runs = []
        for f in files:
            d = np.load(f, allow_pickle=True)
            runs.append({
                "file": f.name,
                "movie_name": d["movie_name"].astype(str),
                "tr_index": d["tr_index"].astype(int),
                "patterns": {roi: d[f"patterns_{roi}"].astype(np.float64)
                             for roi in ROI_NAMES},
            })
        return runs

    f = (beta_cache(pipeline) / sub /
         f"{sub}_task-NATencoding_desc-typed_roipatterns.npz")
    d = np.load(f, allow_pickle=True)
    session = d["session"].astype(str)
    run = d["run"].astype(int)
    z = {roi: _session_zscore(d[f"patterns_{roi}"].astype(np.float64), session)
         for roi in ROI_NAMES}
    runs = []
    for ses in np.unique(session):
        for r in np.unique(run[session == ses]):
            cols = (session == ses) & (run == r)
            runs.append({
                "file": f"{f.name}[{ses},run-{r}]",
                "movie_name": d["movie_name"].astype(str)[cols],
                "tr_index": d["chunk_idx"].astype(int)[cols],
                "patterns": {roi: z[roi][:, cols] for roi in ROI_NAMES},
            })
    return runs


# ── the analysis ─────────────────────────────────────────────────────────

def fit_and_evaluate(sub: str, lag_model: str, nat_source: str = "rawtr",
                     pipeline: str = PIPELINE) -> pd.DataFrame:
    from sklearn.linear_model import RidgeCV

    rng = np.random.default_rng(SEED)
    img_wide = load_image_features()
    mov_feats = load_movie_features()
    id_map = mmmid_to_stimulus_id()
    name_map = movie_name_to_slug()

    betas, mmm_ids = load_tb_betas(sub, pipeline)
    stim_ids = [id_map[i] for i in mmm_ids]
    X_img = img_wide.loc[stim_ids].to_numpy(dtype=np.float64)
    assert X_img.shape[0] == len(mmm_ids), "feature/beta row mismatch"

    # z-score feature columns within each set (implementation detail:
    # image and convolved-movie features are scaled independently).
    Xz = (X_img - X_img.mean(0)) / X_img.std(0)

    # NAT design + actual patterns, averaged over repeat presentations
    runs = load_nat_runs(sub, pipeline, nat_source)
    seg = {}   # slug -> {"X": rows, "Y": {roi: sum}, "n": count per column}
    for run in runs:
        names, tr_idx = run["movie_name"], run["tr_index"]
        starts = np.flatnonzero(np.r_[True, tr_idx[1:] <= tr_idx[:-1]])
        bounds = np.r_[starts, len(tr_idx)]
        for s, e in zip(bounds[:-1], bounds[1:]):
            slug = name_map[names[s].strip().lower()]
            j = tr_idx[s:e]
            entry = seg.setdefault(slug, {"n_max": 0, "sum": {}, "cnt": {}})
            entry["n_max"] = max(entry["n_max"], int(j.max()) + 1)
            for roi in ROI_NAMES:
                block = run["patterns"][roi][:, s:e]     # (V, e-s)
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

    slugs = sorted(seg)
    assert len(slugs) == N_MOVIES, f"{sub}: {len(slugs)} movies, expected {N_MOVIES}"
    K = min(entry["n_max"] for entry in seg.values())

    records = []
    for roi in ROI_NAMES:
        Y = betas[roi].T                                   # (1000, V)
        valid = np.isfinite(Y).all(axis=0)
        Yv = Y[:, valid]
        model = RidgeCV(alphas=RIDGE_ALPHAS)
        model.fit(Xz, Yv)

        pred, actual = {}, {}
        for slug in slugs:
            entry = seg[slug]
            cnt = entry["cnt"][roi]
            with np.errstate(invalid="ignore"):
                Ya = np.where(cnt > 0, entry["sum"][roi] / cnt, np.nan)[valid]
            j = np.arange(entry["n_max"])
            Xm = movie_design(mov_feats[slug], j, lag_model)
            Xm = (Xm - Xm.mean(0)) / np.where(Xm.std(0) == 0, 1, Xm.std(0))
            pred[slug] = model.predict(Xm).T               # (Vv, n_max)
            actual[slug] = Ya

        # secondary: median voxelwise r over concatenated movie time
        P = np.hstack([pred[s] for s in slugs])
        A = np.hstack([actual[s] for s in slugs])
        ok = np.isfinite(A).all(axis=1)
        Pc = P[ok] - P[ok].mean(1, keepdims=True)
        Ac = A[ok] - A[ok].mean(1, keepdims=True)
        vox_r = (Pc * Ac).sum(1) / np.sqrt(
            (Pc ** 2).sum(1) * (Ac ** 2).sum(1))
        median_r = float(np.nanmedian(vox_r))

        # primary: identification rank accuracy on first-K-TR segments
        def flat(mat_dict):
            m = np.stack([mat_dict[s][ok][:, :K] for s in slugs])
            m = m.reshape(len(slugs), -1)
            return (m - m.mean(1, keepdims=True)) / m.std(1, keepdims=True)

        Pf, Af = flat(pred), flat(actual)
        corr = Pf @ Af.T / Pf.shape[1]                     # (60, 60)
        ranks = (corr >= np.diag(corr)[:, None]).sum(1)    # 1 = best
        rank_acc = float((len(slugs) - ranks).mean() / (len(slugs) - 1))

        null = np.empty(N_PERMUTATIONS)
        for p in range(N_PERMUTATIONS):
            perm = rng.permutation(len(slugs))
            r_p = (corr[perm] >= np.diag(corr[perm])[:, None]).sum(1)
            null[p] = (len(slugs) - r_p).mean() / (len(slugs) - 1)
        p_val = float((1 + (null >= rank_acc).sum()) / (1 + N_PERMUTATIONS))

        records.append({
            "subject": sub, "roi": roi, "lag_model": lag_model,
            "nat_source": nat_source,
            "feature_space": FEATURE_SPACE, "pipeline": pipeline,
            "rank_accuracy": rank_acc, "p_perm": p_val,
            "median_voxel_r": median_r, "n_voxels": int(valid.sum()),
            "n_voxels_eval": int(ok.sum()), "n_movies": len(slugs),
            "K_trs": int(K), "alpha": float(model.alpha_),
        })
        print(f"  {sub} {roi:<12} rank_acc={rank_acc:.3f} (p={p_val:.4f}) "
              f"median_r={median_r:+.4f} alpha={model.alpha_:.0f}")
    return pd.DataFrame.from_records(records)


# ── dry run ───────────────────────────────────────────────────────────────

def dry_run() -> int:
    problems = 0

    def check(label, cond):
        nonlocal problems
        print(f"  [{'ok' if cond else 'MISSING'}] {label}")
        problems += not cond

    print("features:")
    img, mov = feature_inputs()
    for p in img + mov[:3]:
        check(p, p.exists())
    check(f"...all {len(mov)} movie ebind.csv",
          all(p.exists() for p in mov))
    print("registry:")
    for f in ("shared1000.tsv", "movies.tsv"):
        check(REGISTRY / f, (REGISTRY / f).exists())
    for pipeline in ("original", "nordic"):
        print(f"brain caches (pipeline={pipeline}):")
        for sub in SUBJECTS:
            for task in ("TBencoding", "NATencoding"):
                beta = (beta_cache(pipeline) / sub /
                        f"{sub}_task-{task}_desc-typed_roipatterns.npz")
                check(beta, beta.exists())
            n = len(list((rawtr_cache(pipeline) / sub)
                         .glob("*task-NATencoding*npz")))
            check(f"{sub}: 20 NAT rawTR caches (found {n})", n == 20)
    print("movie-name mapping (rawTR names -> registry slugs):")
    name_map = movie_name_to_slug()
    unmapped = set()
    for sub in SUBJECTS:
        for f in (rawtr_cache("original") / sub).glob("*task-NATencoding*npz"):
            d = np.load(f, allow_pickle=True)
            for name in np.unique(d["movie_name"].astype(str)):
                if name.strip().lower() not in name_map:
                    unmapped.add(name)
    check(f"all cached movie names resolve ({len(unmapped)} unmapped: "
          f"{sorted(unmapped)[:4]})", not unmapped)
    print(f"\nDry run {'PASSED' if problems == 0 else f'FAILED ({problems})'}")
    return problems


# ── entry point ───────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sub", choices=SUBJECTS, help="one subject (default: all)")
    ap.add_argument("--lag-model", choices=["hrf", "fixed"], default="hrf",
                    help="hrf = pre-registered primary; fixed = sensitivity "
                    "(rawtr only; glmsingle always uses chunkmean)")
    ap.add_argument("--nat-source", choices=["rawtr", "glmsingle"],
                    default="rawtr",
                    help="NAT response representation: Model-8 rawTR "
                    "(primary) or GLMsingle 4.5s chunk betas (diagnostic)")
    ap.add_argument("--pipeline", choices=["original", "nordic"],
                    default=PIPELINE,
                    help="preprocessing variant, both arms (diagnostic)")
    ap.add_argument("--build-features", action="store_true",
                    help="only build the psytwill feature parquets")
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve all inputs and report, compute nothing")
    args = ap.parse_args()

    if args.dry_run:
        sys.exit(1 if dry_run() else 0)

    build_feature_tables()
    if args.build_features:
        return

    A1_DIR.mkdir(parents=True, exist_ok=True)
    desc = OUT_ROOT / "dataset_description.json"
    if not desc.exists():
        desc.write_text(json.dumps({
            "Name": "MMMData srm-stimulus-space pilot",
            "BIDSVersion": "1.9.0",
            "DatasetType": "derivative",
            "GeneratedBy": [{
                "Name": "mmmdata/scripts/srm_stimulus_space/",
                "Description": "Pre-registered TB<->NAT stimulus-space "
                "transfer pilot; charter in mmmdata-agents "
                "docs/workbench/srm-stimulus-space/",
            }],
            "SourceDatasets": [
                {"URL": "../pattern_similarity"},
                {"URL": "../stimuli_features"},
            ],
        }, indent=2))

    lag_model = ("chunkmean" if args.nat_source == "glmsingle"
                 else args.lag_model)
    subs = [args.sub] if args.sub else SUBJECTS
    frames = []
    for sub in subs:
        print(f"=== A1 encoding: {sub} (nat={args.nat_source}, "
              f"lag_model={lag_model}, pipeline={args.pipeline}) ===")
        t0 = time.time()
        frames.append(fit_and_evaluate(
            sub, lag_model, args.nat_source, args.pipeline))
        print(f"  ({time.time() - t0:.0f}s)")
    result = pd.concat(frames, ignore_index=True)
    result["created_at"] = datetime.now(timezone.utc).isoformat()

    tag = "" if lag_model == "hrf" else f"_lag-{lag_model}"
    if args.nat_source != "rawtr":
        tag += f"_nat-{args.nat_source}"
    if args.pipeline != "original":
        tag += f"_pipe-{args.pipeline}"
    out = A1_DIR / (f"a1_summary{tag}_{subs[0]}.tsv" if args.sub
                    else f"a1_summary{tag}.tsv")
    result.to_csv(out, sep="\t", index=False)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
