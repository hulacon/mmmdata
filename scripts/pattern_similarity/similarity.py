#!/usr/bin/env python3
"""Phase 6 — pattern-similarity computation (cache-only I/O).

Fills the 4-cell similarity design (within/across subject x same/different
item) for every (paradigm x pipeline x timeseries x ROI) combination, from
the Phase 3-5 caches under derivatives/pattern_similarity/cache/.

Core metric: Pearson r across mutually finite ROI voxels, per matched
column pair, averaged over the relevant trials/TRs/chunks (plan D3/D6).

Cells (plan Phase 6):
  - NAT within_same:  per repeated movie, all 45 session pairs,
    TR/chunk-matched (truncate to min shared length).
  - NAT within_diff:  10 seeded draws per same pair — contiguous window of
    another movie from the second member's run, same length (plan D4).
  - TB  within_same:  per triplet item, all cross-session presentation
    pairs (42 presentations -> 819 pairs).
  - TB  within_diff:  10 seeded draws per pair — position-matched trial in
    a contiguous 3-trial all-non-shared window from the second member's run.
  - across_same:      3 subject pairs; NAT all 10x10 session pairs, TB all
    42x42 presentation pairs over all 6 shared items (context_match
    recorded); same-session-number subset reported as session_scope=matched.
  - across_diff:      surrogates drawn from subject B's runs (position-
    matched 3-windows when the item is triplet-role in B; any non-shared
    trial when single-role).

Surrogate draws are seeded per (paradigm, cell, unit, item) — excluding
pipeline/timeseries — so the same surrogates are used in both pipelines
(paired contrasts; TB draws are also identical across timeseries).

Outputs:
  results/similarity_summary.tsv  — one row per unit x item x cell x combo
  results/similarity_cells.tsv    — collapsed over unit/item

Plan: docs/doc/pattern-similarity-plan.md, Phase 6.

Usage:
    python similarity.py                       # full grid
    python similarity.py --paradigm TB --timeseries glmsingle   # smoke test
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from shared import (  # noqa: E402
    CACHE_DIR, N_DIFF_DRAWS, NAT_RUNS, NAT_SESSIONS, PATTERN_ROI_NAMES,
    REPEATED_MOVIES, RESULT_COLS, RESULTS_DIR, SEED, SUBJECTS, TB_RUNS,
    TB_SESSIONS, shared_item_registry,
)

TIMESERIES = ["rawtr", "glmsingle"]
PIPELINES_ORDERED = ["original", "nordic"]
CONFOUND_MODEL = {"rawtr": "model8", "glmsingle": "glmdenoise"}
SUBJECT_PAIRS = list(itertools.combinations(SUBJECTS, 2))
N_TB_PRES = 42          # presentations per shared item per subject
TB_WINDOW = 3           # triplet window length for position-matched surrogates

# rng stream codes (paradigm/cell), used in seed keys
RNG_TB_WITHIN, RNG_TB_ACROSS, RNG_NAT_WITHIN, RNG_NAT_ACROSS = 1, 2, 3, 4


def _subnum(sub: str) -> int:
    return int(sub.split("-")[1])


# ── NaN-aware correlation kernels ────────────────────────────────────────

def matched_corr(A: np.ndarray, B: np.ndarray):
    """Column-matched Pearson r over mutually finite voxels.

    A, B: (V, n) with identical column meaning. Returns (r, n_vox), each
    (n,); r is NaN where <3 mutually finite voxels.
    """
    M = np.isfinite(A) & np.isfinite(B)
    A0 = np.where(M, A, 0.0).astype(np.float64)
    B0 = np.where(M, B, 0.0).astype(np.float64)
    n = M.sum(axis=0)
    sx, sy = A0.sum(0), B0.sum(0)
    sxx, syy, sxy = (A0 * A0).sum(0), (B0 * B0).sum(0), (A0 * B0).sum(0)
    cov = n * sxy - sx * sy
    var = (n * sxx - sx**2) * (n * syy - sy**2)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = cov / np.sqrt(var)
    r[n < 3] = np.nan
    return r, n


def pairwise_corr(A: np.ndarray, B: np.ndarray):
    """All-pairs Pearson r between columns of A (V, nA) and B (V, nB), each
    pair over its mutually finite voxels (pairwise-complete observations).

    Returns (R, N): (nA, nB) r matrix + mutually-finite voxel counts.
    """
    MA, MB = np.isfinite(A), np.isfinite(B)
    A0 = np.where(MA, A, 0.0).astype(np.float64)
    B0 = np.where(MB, B, 0.0).astype(np.float64)
    MAf, MBf = MA.astype(np.float64), MB.astype(np.float64)
    n = MAf.T @ MBf
    sx = A0.T @ MBf          # sum of a over the joint mask
    sy = MAf.T @ B0
    sxy = A0.T @ B0
    sxx = (A0 * A0).T @ MBf
    syy = MAf.T @ (B0 * B0)
    cov = n * sxy - sx * sy
    var = (n * sxx - sx**2) * (n * syy - sy**2)
    with np.errstate(invalid="ignore", divide="ignore"):
        R = cov / np.sqrt(var)
    R[n < 3] = np.nan
    return R, n


# ── cache loaders (unified per-stream structures) ────────────────────────

def _mmm_int(x) -> np.ndarray:
    """Normalize mmmId arrays ('995', '995.0', 995.0) to int."""
    return pd.to_numeric(pd.Series(np.asarray(x).astype(str))).astype(int).to_numpy()


def load_tb(ts: str, sub: str, pipeline: str) -> dict:
    """TB stream -> patterns {roi: (V, 2562)}, session/run/pos/mmmId arrays.

    pos = chronological within-run trial index (0..60); column order within
    each (session, run) is validated downstream by triplet contiguity.
    """
    if ts == "glmsingle":
        d = np.load(CACHE_DIR / "glmsingle" / pipeline / sub /
                    f"{sub}_task-TBencoding_desc-typed_roipatterns.npz",
                    allow_pickle=True)
        session = d["session"].astype(str)
        run = d["run"].astype(int)
        mmm = _mmm_int(d["mmmId"])
        pos = np.zeros(len(session), dtype=int)
        seen: dict = {}
        for g, key in enumerate(zip(session, run)):
            pos[g] = seen[key] = seen.get(key, -1) + 1
        patterns = {roi: d[f"patterns_{roi}"] for roi in PATTERN_ROI_NAMES}
        voxidx = {roi: d[f"voxidx_{roi}"] for roi in PATTERN_ROI_NAMES}
    else:
        parts, ses_l, run_l, mmm_l = {roi: [] for roi in PATTERN_ROI_NAMES}, [], [], []
        voxidx = None
        for ses in TB_SESSIONS:
            for r in TB_RUNS:
                d = np.load(CACHE_DIR / "rawtr" / pipeline / sub /
                            f"{sub}_{ses}_task-TBencoding_run-{r:02d}"
                            f"_desc-model8_roipatterns.npz", allow_pickle=True)
                n = d["mmmId"].shape[0]
                ses_l += [ses] * n
                run_l += [r] * n
                mmm_l.append(_mmm_int(d["mmmId"]))
                for roi in PATTERN_ROI_NAMES:
                    parts[roi].append(d[f"patterns_{roi}"])
                if voxidx is None:
                    voxidx = {roi: d[f"voxidx_{roi}"] for roi in PATTERN_ROI_NAMES}
        session = np.array(ses_l)
        run = np.array(run_l)
        mmm = np.concatenate(mmm_l)
        pos = np.concatenate([np.arange(np.sum((session == s) & (run == r)))
                              for s in TB_SESSIONS for r in TB_RUNS])
        patterns = {roi: np.concatenate(parts[roi], axis=1)
                    for roi in PATTERN_ROI_NAMES}
    return {"patterns": patterns, "session": session, "run": run,
            "pos": pos, "mmmId": mmm, "voxidx": voxidx}


def load_nat(ts: str, sub: str, pipeline: str) -> dict:
    """NAT stream -> patterns {roi: (V, N)}, session/run/movie/uidx arrays.

    uidx = tr_index (rawtr) or chunk_idx (glmsingle): the within-movie
    position used to column-match presentations across sessions/subjects.
    """
    if ts == "glmsingle":
        d = np.load(CACHE_DIR / "glmsingle" / pipeline / sub /
                    f"{sub}_task-NATencoding_desc-typed_roipatterns.npz",
                    allow_pickle=True)
        session = d["session"].astype(str)
        run = d["run"].astype(int)
        movie = d["movie_name"].astype(str)
        uidx = d["chunk_idx"].astype(int)
        patterns = {roi: d[f"patterns_{roi}"] for roi in PATTERN_ROI_NAMES}
        voxidx = {roi: d[f"voxidx_{roi}"] for roi in PATTERN_ROI_NAMES}
    else:
        parts, ses_l, run_l, mov_l, uidx_l = (
            {roi: [] for roi in PATTERN_ROI_NAMES}, [], [], [], [])
        voxidx = None
        for ses in NAT_SESSIONS:
            for r in NAT_RUNS:
                d = np.load(CACHE_DIR / "rawtr" / pipeline / sub /
                            f"{sub}_{ses}_task-NATencoding_run-{r:02d}"
                            f"_desc-model8_roipatterns.npz", allow_pickle=True)
                n = d["tr_index"].shape[0]
                ses_l += [ses] * n
                run_l += [r] * n
                mov_l.append(d["movie_name"].astype(str))
                uidx_l.append(d["tr_index"].astype(int))
                for roi in PATTERN_ROI_NAMES:
                    parts[roi].append(d[f"patterns_{roi}"])
                if voxidx is None:
                    voxidx = {roi: d[f"voxidx_{roi}"] for roi in PATTERN_ROI_NAMES}
        session = np.array(ses_l)
        run = np.array(run_l)
        movie = np.concatenate(mov_l)
        uidx = np.concatenate(uidx_l)
        patterns = {roi: np.concatenate(parts[roi], axis=1)
                    for roi in PATTERN_ROI_NAMES}
    return {"patterns": patterns, "session": session, "run": run,
            "movie": movie, "uidx": uidx, "voxidx": voxidx}


# ── TB run structure (positions, surrogate windows) ──────────────────────

def tb_structure(data: dict, registry: pd.DataFrame) -> dict:
    """Per-run column maps + triplet positions + surrogate draw pools.

    Returns dict with:
      runs:        {(ses, run): global cols ordered by pos}
      tpos:        (N,) triplet position 0-2 (-1 elsewhere)
      win_starts:  {(ses, run): pos values starting an all-non-shared
                    contiguous TB_WINDOW-trial window}
      nonshared:   {(ses, run): pos values of non-shared trials}
    Validates triplet contiguity in every run (also proves that cache
    column order is chronological within runs).
    """
    shared6 = registry["mmmId"].to_numpy()
    triplet = registry.loc[registry["role"] == "triplet", "mmmId"].to_numpy()
    N = len(data["mmmId"])
    runs: dict = {}
    for g, key in enumerate(zip(data["session"], data["run"])):
        runs.setdefault(key, []).append(g)
    tpos = np.full(N, -1, dtype=int)
    win_starts, nonshared = {}, {}
    for key in sorted(runs):
        cols = np.asarray(runs[key])
        cols = cols[np.argsort(data["pos"][cols], kind="stable")]
        runs[key] = cols
        ids = data["mmmId"][cols]
        ti = np.flatnonzero(np.isin(ids, triplet))
        # 0-3 triplet occurrences per run (3/session, unevenly distributed);
        # each occurrence is a contiguous block of all 3 triplet items
        assert len(ti) % 3 == 0, f"{key}: {len(ti)} triplet trials"
        for b in range(0, len(ti), 3):
            blk = ti[b:b + 3]
            assert blk[2] - blk[0] == 2 and set(ids[blk]) == set(triplet), (
                f"{key}: triplet block not contiguous/complete "
                f"(positions {blk}, ids {ids[blk]})"
            )
            tpos[cols[blk]] = np.arange(3)
        ok = ~np.isin(ids, shared6)
        nonshared[key] = np.flatnonzero(ok)
        w = np.flatnonzero(ok[:-2] & ok[1:-1] & ok[2:])
        assert len(w) >= N_DIFF_DRAWS, f"{key}: only {len(w)} surrogate windows"
        win_starts[key] = w
    return {"runs": runs, "tpos": tpos, "win_starts": win_starts,
            "nonshared": nonshared}


def tb_draw_surrogates(rng, struct_b: dict, data_b: dict, gj: np.ndarray,
                       position_matched: bool) -> np.ndarray:
    """Surrogate global columns (len(gj), N_DIFF_DRAWS) in gj's own runs.

    position_matched: draw an all-non-shared contiguous 3-window and take
    the trial at gj's triplet position; otherwise draw any non-shared trial.
    Grouped by run in sorted order so draws are reproducible.
    """
    surr = np.empty((len(gj), N_DIFF_DRAWS), dtype=int)
    keys = [(data_b["session"][g], data_b["run"][g]) for g in gj]
    for key in sorted(set(keys)):
        sel = np.flatnonzero([k == key for k in keys])
        if position_matched:
            pool = struct_b["win_starts"][key]
            draws = rng.integers(0, len(pool), (len(sel), N_DIFF_DRAWS))
            pos = pool[draws] + struct_b["tpos"][gj[sel]][:, None]
        else:
            pool = struct_b["nonshared"][key]
            draws = rng.integers(0, len(pool), (len(sel), N_DIFF_DRAWS))
            pos = pool[draws]
        surr[sel] = struct_b["runs"][key][pos]
    return surr


# ── TB cells ─────────────────────────────────────────────────────────────

def tb_within_rows(sub, data, registry, struct, pipeline, ts, rows):
    triplet = (registry[registry["role"] == "triplet"]
               .sort_values("position")["mmmId"].to_numpy())
    ses = data["session"]
    for m in triplet:
        idx = np.flatnonzero(data["mmmId"] == m)
        assert len(idx) == N_TB_PRES, f"{sub} item {m}: {len(idx)} presentations"
        iu, ju = np.triu_indices(len(idx), 1)
        cross = ses[idx[iu]] != ses[idx[ju]]
        pi, pj = iu[cross], ju[cross]
        assert len(pi) == 819, f"{sub} item {m}: {len(pi)} cross-session pairs"

        rng = np.random.default_rng([SEED, RNG_TB_WITHIN, _subnum(sub), int(m)])
        surr = tb_draw_surrogates(rng, struct, data, idx[pj],
                                  position_matched=True)
        for roi in PATTERN_ROI_NAMES:
            P = data["patterns"][roi]
            R, Nv = pairwise_corr(P[:, idx], P)
            base = dict(paradigm="TB", pipeline=pipeline, timeseries=ts,
                        roi=roi, unit=sub, item=str(m),
                        confound_model=CONFOUND_MODEL[ts],
                        session_scope="all", context_match="na")
            rows.append(base | dict(
                cell="within_same", r=np.nanmean(R[pi, idx[pj]]),
                n_pairs=len(pi), n_voxels=Nv[pi, idx[pj]].mean(), n_draws=0))
            rows.append(base | dict(
                cell="within_diff", r=np.nanmean(R[pi[:, None], surr]),
                n_pairs=len(pi), n_voxels=Nv[pi[:, None], surr].mean(),
                n_draws=N_DIFF_DRAWS))


def tb_across_rows(pair, datas, registries, structs, pipeline, ts, rows):
    sub_a, sub_b = pair
    da, db = datas[sub_a], datas[sub_b]
    ra, rb = registries[sub_a], registries[sub_b]
    for roi in PATTERN_ROI_NAMES:
        assert np.array_equal(da["voxidx"][roi], db["voxidx"][roi]), (
            f"{pair} {roi}: voxel sets differ across subjects")
    role_a = ra.set_index("mmmId")["role"]
    role_b = rb.set_index("mmmId")["role"]
    unit = f"{sub_a}+{sub_b}"

    for m in sorted(ra["mmmId"]):
        ia = np.flatnonzero(da["mmmId"] == m)
        ib = np.flatnonzero(db["mmmId"] == m)
        assert len(ia) == len(ib) == N_TB_PRES
        context = ("triplet-triplet" if role_a[m] == role_b[m] == "triplet"
                   else "single-single" if role_a[m] == role_b[m] == "single"
                   else "mixed")
        matched = da["session"][ia][:, None] == db["session"][ib][None, :]
        I = np.repeat(np.arange(len(ia)), len(ib))
        J = np.tile(np.arange(len(ib)), len(ia))

        rng = np.random.default_rng(
            [SEED, RNG_TB_ACROSS, _subnum(sub_a), _subnum(sub_b), int(m)])
        surr = tb_draw_surrogates(rng, structs[sub_b], db, ib[J],
                                  position_matched=(role_b[m] == "triplet"))
        for roi in PATTERN_ROI_NAMES:
            R, Nv = pairwise_corr(da["patterns"][roi][:, ia],
                                  db["patterns"][roi])
            r_same, n_same = R[:, ib], Nv[:, ib]
            r_diff = R[I[:, None], surr]
            n_diff = Nv[I[:, None], surr]
            m_flat = matched.ravel()
            base = dict(paradigm="TB", pipeline=pipeline, timeseries=ts,
                        roi=roi, unit=unit, item=str(m),
                        confound_model=CONFOUND_MODEL[ts],
                        context_match=context)
            for scope, sel in (("all", slice(None)),
                               ("matched", m_flat)):
                n_pairs = matched.size if scope == "all" else int(m_flat.sum())
                rows.append(base | dict(
                    cell="across_same", session_scope=scope,
                    r=np.nanmean(r_same.ravel()[sel]), n_pairs=n_pairs,
                    n_voxels=n_same.ravel()[sel].mean(), n_draws=0))
                rows.append(base | dict(
                    cell="across_diff", session_scope=scope,
                    r=np.nanmean(r_diff[sel]), n_pairs=n_pairs,
                    n_voxels=n_diff[sel].mean(), n_draws=N_DIFF_DRAWS))


# ── NAT cells ────────────────────────────────────────────────────────────

def nat_maps(data: dict):
    """{(ses, movie): cols sorted by uidx}, {(ses, movie): run},
    {(ses, run): [movies]}."""
    cols_map, run_of, run_movies = {}, {}, {}
    df = pd.DataFrame({"g": np.arange(len(data["uidx"])),
                       "ses": data["session"], "run": data["run"],
                       "movie": data["movie"], "uidx": data["uidx"]})
    for (ses, movie), grp in df.groupby(["ses", "movie"], sort=True):
        runs = grp["run"].unique()
        assert len(runs) == 1, f"{ses}/{movie}: spans runs {runs}"
        cols_map[(ses, movie)] = grp.sort_values("uidx")["g"].to_numpy()
        run_of[(ses, movie)] = int(runs[0])
        run_movies.setdefault((ses, int(runs[0])), []).append(movie)
    for key in run_movies:
        run_movies[key] = sorted(run_movies[key])
    return cols_map, run_of, run_movies


def nat_draw_windows(rng, cols_map, run_movies, ses, run, exclude_movie, L):
    """N_DIFF_DRAWS contiguous length-L column windows from other movies in
    (ses, run). Returns list of global-column arrays."""
    cands = [cols_map[(ses, mv)] for mv in run_movies[(ses, run)]
             if mv != exclude_movie and len(cols_map[(ses, mv)]) >= L]
    assert cands, f"{ses} run-{run}: no surrogate movie with >= {L} columns"
    wins = []
    for _ in range(N_DIFF_DRAWS):
        c = cands[rng.integers(len(cands))]
        s = rng.integers(0, len(c) - L + 1)
        wins.append(c[s:s + L])
    return wins


def _nat_pair_cells(da, db, cma, cmb, run_of_b, run_movies_b, ses_pairs, rng):
    """Same/diff stats per ROI over a list of (movie, ses_a, ses_b) pairs.

    Returns {roi: dict(r_same=[], n_same=[], r_diff=[], n_diff=[])} with one
    entry per session pair (diff entries averaged over draws).
    """
    acc = {roi: {"r_same": [], "n_same": [], "r_diff": [], "n_diff": []}
           for roi in PATTERN_ROI_NAMES}
    for movie, s1, s2 in ses_pairs:
        c1, c2 = cma[(s1, movie)], cmb[(s2, movie)]
        L = min(len(c1), len(c2))
        wins = nat_draw_windows(rng, cmb, run_movies_b, s2,
                                run_of_b[(s2, movie)], movie, L)
        for roi in PATTERN_ROI_NAMES:
            A = da["patterns"][roi][:, c1[:L]]
            B = db["patterns"][roi][:, c2[:L]]
            r, n = matched_corr(A, B)
            acc[roi]["r_same"].append(np.nanmean(r))
            acc[roi]["n_same"].append(n.mean())
            r_d, n_d = [], []
            for w in wins:
                rw, nw = matched_corr(A, db["patterns"][roi][:, w])
                r_d.append(np.nanmean(rw))
                n_d.append(nw.mean())
            acc[roi]["r_diff"].append(np.mean(r_d))
            acc[roi]["n_diff"].append(np.mean(n_d))
    return acc


def nat_within_rows(sub, data, pipeline, ts, rows):
    cols_map, run_of, run_movies = nat_maps(data)
    for mi, movie in enumerate(REPEATED_MOVIES):
        sess = sorted(s for (s, mv) in cols_map if mv == movie)
        assert len(sess) == len(NAT_SESSIONS), (
            f"{sub}/{movie}: {len(sess)} sessions")
        pairs = [(movie, s1, s2) for s1, s2 in itertools.combinations(sess, 2)]
        assert len(pairs) == 45
        rng = np.random.default_rng([SEED, RNG_NAT_WITHIN, _subnum(sub), mi])
        acc = _nat_pair_cells(data, data, cols_map, cols_map, run_of,
                              run_movies, pairs, rng)
        for roi in PATTERN_ROI_NAMES:
            a = acc[roi]
            base = dict(paradigm="NAT", pipeline=pipeline, timeseries=ts,
                        roi=roi, unit=sub, item=movie,
                        confound_model=CONFOUND_MODEL[ts],
                        session_scope="all", context_match="na")
            rows.append(base | dict(
                cell="within_same", r=np.nanmean(a["r_same"]),
                n_pairs=len(pairs), n_voxels=np.mean(a["n_same"]), n_draws=0))
            rows.append(base | dict(
                cell="within_diff", r=np.nanmean(a["r_diff"]),
                n_pairs=len(pairs), n_voxels=np.mean(a["n_diff"]),
                n_draws=N_DIFF_DRAWS))


def nat_across_rows(pair, datas, pipeline, ts, rows):
    sub_a, sub_b = pair
    da, db = datas[sub_a], datas[sub_b]
    for roi in PATTERN_ROI_NAMES:
        assert np.array_equal(da["voxidx"][roi], db["voxidx"][roi]), (
            f"{pair} {roi}: voxel sets differ across subjects")
    cma, run_of_a, _ = nat_maps(da)
    cmb, run_of_b, run_movies_b = nat_maps(db)
    unit = f"{sub_a}+{sub_b}"
    for mi, movie in enumerate(REPEATED_MOVIES):
        sess_a = sorted(s for (s, mv) in cma if mv == movie)
        sess_b = sorted(s for (s, mv) in cmb if mv == movie)
        pairs = [(movie, s1, s2)
                 for s1 in sess_a for s2 in sess_b]
        assert len(pairs) == 100
        matched = np.array([s1 == s2 for _, s1, s2 in pairs])
        rng = np.random.default_rng(
            [SEED, RNG_NAT_ACROSS, _subnum(sub_a), _subnum(sub_b), mi])
        acc = _nat_pair_cells(da, db, cma, cmb, run_of_b, run_movies_b,
                              pairs, rng)
        for roi in PATTERN_ROI_NAMES:
            a = {k: np.asarray(v) for k, v in acc[roi].items()}
            base = dict(paradigm="NAT", pipeline=pipeline, timeseries=ts,
                        roi=roi, unit=unit, item=movie,
                        confound_model=CONFOUND_MODEL[ts], context_match="na")
            for scope, sel in (("all", slice(None)), ("matched", matched)):
                n_pairs = len(pairs) if scope == "all" else int(matched.sum())
                rows.append(base | dict(
                    cell="across_same", session_scope=scope,
                    r=np.nanmean(a["r_same"][sel]), n_pairs=n_pairs,
                    n_voxels=a["n_same"][sel].mean(), n_draws=0))
                rows.append(base | dict(
                    cell="across_diff", session_scope=scope,
                    r=np.nanmean(a["r_diff"][sel]), n_pairs=n_pairs,
                    n_voxels=a["n_diff"][sel].mean(), n_draws=N_DIFF_DRAWS))


# ── driver ───────────────────────────────────────────────────────────────

def run_stream(paradigm, pipeline, ts, registries, rows):
    t0 = time.time()
    load = load_tb if paradigm == "TB" else load_nat
    datas = {sub: load(ts, sub, pipeline) for sub in SUBJECTS}
    print(f"[{paradigm}/{pipeline}/{ts}] loaded "
          f"({time.time() - t0:.0f}s)", flush=True)
    if paradigm == "TB":
        structs = {sub: tb_structure(datas[sub], registries[sub])
                   for sub in SUBJECTS}
        for sub in SUBJECTS:
            tb_within_rows(sub, datas[sub], registries[sub], structs[sub],
                           pipeline, ts, rows)
        for pair in SUBJECT_PAIRS:
            tb_across_rows(pair, datas, registries, structs, pipeline, ts, rows)
    else:
        for sub in SUBJECTS:
            nat_within_rows(sub, datas[sub], pipeline, ts, rows)
        for pair in SUBJECT_PAIRS:
            nat_across_rows(pair, datas, pipeline, ts, rows)
    print(f"[{paradigm}/{pipeline}/{ts}] done ({time.time() - t0:.0f}s)",
          flush=True)


def final_checks(df: pd.DataFrame):
    """Plan Phase 6 'Verify': closed-form pair counts, cell coverage, sanity."""
    expect = {("TB", "within_same"): 819, ("TB", "within_diff"): 819,
              ("NAT", "within_same"): 45, ("NAT", "within_diff"): 45}
    for (par, cell), n in expect.items():
        got = df.loc[(df.paradigm == par) & (df.cell == cell), "n_pairs"]
        assert (got == n).all(), f"{par}/{cell}: n_pairs {got.unique()} != {n}"
    for par, n_all, n_m in (("TB", 1764, 126), ("NAT", 100, 10)):
        sel = df[(df.paradigm == par) & df.cell.str.startswith("across")]
        assert (sel.loc[sel.session_scope == "all", "n_pairs"] == n_all).all()
        assert (sel.loc[sel.session_scope == "matched", "n_pairs"] == n_m).all()
    combos = df[df.session_scope == "all"].groupby(
        ["paradigm", "pipeline", "timeseries", "roi"])["cell"].nunique()
    assert (combos == 4).all(), "missing cells in some combo"
    print(f"Pair-count + coverage checks passed "
          f"({len(combos)} combos x 4 cells).")
    evc = df[(df.roi == "EVC") & (df.session_scope == "all")]
    for (par, pipe, ts), grp in evc.groupby(
            ["paradigm", "pipeline", "timeseries"]):
        same = grp.loc[grp.cell == "within_same", "r"].mean()
        diff = grp.loc[grp.cell == "within_diff", "r"].mean()
        flag = "OK" if same > diff else "UNEXPECTED"
        print(f"  sanity EVC {par}/{pipe}/{ts}: within_same={same:.4f} "
              f"> within_diff={diff:.4f}? [{flag}]")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paradigm", nargs="+", choices=["TB", "NAT"],
                    default=["TB", "NAT"])
    ap.add_argument("--pipeline", nargs="+", choices=PIPELINES_ORDERED,
                    default=PIPELINES_ORDERED)
    ap.add_argument("--timeseries", nargs="+", choices=TIMESERIES,
                    default=TIMESERIES)
    ap.add_argument("--no-checks", action="store_true",
                    help="skip closed-form count checks (partial-grid runs)")
    args = ap.parse_args()

    registries = {sub: shared_item_registry(sub) for sub in SUBJECTS}
    rows: list[dict] = []
    for paradigm in args.paradigm:
        for pipeline in args.pipeline:
            for ts in args.timeseries:
                run_stream(paradigm, pipeline, ts, registries, rows)

    df = pd.DataFrame(rows)[RESULT_COLS]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    full = (len(args.paradigm), len(args.pipeline), len(args.timeseries)) == (2, 2, 2)
    suffix = "" if full else "_partial"
    p = RESULTS_DIR / f"similarity_summary{suffix}.tsv"
    df.to_csv(p, sep="\t", index=False, float_format="%.6f")
    print(f"\nWrote {p} ({len(df)} rows)")

    cells = (df.groupby(["paradigm", "pipeline", "timeseries", "roi", "cell",
                         "session_scope"], sort=True)
             .agg(r=("r", "mean"), n_units=("r", "size"),
                  n_voxels=("n_voxels", "mean"))
             .reset_index())
    pc = RESULTS_DIR / f"similarity_cells{suffix}.tsv"
    cells.to_csv(pc, sep="\t", index=False, float_format="%.6f")
    print(f"Wrote {pc} ({len(cells)} rows)")

    if not args.no_checks:
        final_checks(df)
    print("\nDone.")


if __name__ == "__main__":
    main()
