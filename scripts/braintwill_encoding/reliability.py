#!/usr/bin/env python3
"""Read-time column masks for the encoding harness: per-column response reliability.

The harness scores every column (each column's α and band weights are its own, so a mask
changes no score); a consumer reads scores through a mask. This driver writes the
per-column reliability tables and default masks (encoding-harness settle 3, DECIDED
2026-10-12).

  films   repeat reliability over the repeated films' showings in the alignment sessions
          (``films.py`` role ``dropped_repeat``). These showings are in neither the 46-film
          pool nor the held-out sessions, so the mask is independent of every training row,
          and **ses-23/26 never enter**: a held-out-session row is an error, not a filter.

  tb      repeat reliability over TBencoding image–word pairs (encoding-harness settle 3, B).
          Every trial plays the pair's word over the image, so the betas are pair-evoked and
          a repeat replays the same word: auditory cortex is covered. Repeats only — items
          with ``enCon == 2``, including the super repeats among them; triplets
          (``enCon == 3``) are left out because TYPED leaks ~28 % of each neighbour into a
          beta and a triplet repeats its neighbours in the same order, which would inflate
          their reliability.

Method (films). Each showing's window is put on the first showing's volume grid
(``films.paired_slices``: nearest whole volume), z-scored per column within the showing,
and the films are concatenated, so a column's series is ``(n_showings, T)`` with zero mean
and unit variance per film segment. The statistic is the mean correlation over showing
pairs, computed without forming the pairs:

    r̄ = (|Σ_i z_i|² − Σ_i |z_i|²) / (2 · n_pairs · T)

and the Spearman–Brown reliability of the n-showing mean is ``n·r̄ / (1 + (n−1)·r̄)``.
Null: each showing's segment of each film is shifted circularly by an independent offset
of at least ``MIN_SHIFT_TR`` volumes from either end (keeps each series' autocorrelation,
breaks stimulus locking); ``N_NULL`` draws give a per-column empirical p,
``(1 + #{null ≥ r̄}) / (N_NULL + 1)``. Default mask = Benjamini–Hochberg q < ``Q`` within
subject, over the columns finite in every showing.

Method (tb). Betas are the tb-enc register's: TYPED ``betasmd`` on the grayordinates,
z-scored per column within each session (``harness.zscore_sessions``). The statistic is the
item-weighted mean within-item variance over the repeated items' trials, σ²_noise; with the
pooled variance σ²_total of the same trials it gives NSD's
``ncsnr = sqrt(max(0, σ²_total − σ²_noise)) / sqrt(σ²_noise)``. Null: item labels permuted
among the included trials within each session (group sizes kept), ``N_NULL`` draws, per-column
p = ``(1 + #{null σ²_noise ≤ observed}) / (N_NULL + 1)``. Default mask = BH q < ``Q`` **and**
the label-free beta floor M3 used (finite in every trial, median |beta| ≤ 100 on the raw betas).

Output: ``<derivatives>/encoding_harness/reliability/register-<films|tb-enc>/sub-##.npz``
(``r_mean, spearman_brown, p, q, mask, valid``) + ``sub-##.json`` (method, showings, seed,
code version).

  ceiling the time-series registers' noise ceiling and its floor (tb-nat-encoding settle 3,
          DECIDED 2026-10-15): one estimator for ``films`` and ``tb-enc-ts``. NC per column =
          the mean over repeat pairs and identical-input volumes of y_i · y_j, y in the
          register's own standardisation, so fraction-of-ceiling = held-out R² / NC.
          Identical-input volumes: every delay span of the FIR window, the union
          ``[s − max(DELAYS)·TR, s − (min(DELAYS) − 1)·TR)``, lies inside one presentation of
          the repeated stimulus. **films**: the repeat reliability's own showings, z-scored
          per showing; NC is exactly its r̄. **tb-enc-ts**: TBencoding triplets (``enCon == 3``,
          one sequence = ``(pairId − 1) // 3`` within a session): pairs of *intact*
          presentations (3 consecutive slots, no rest inside, one run) with *the same item
          order*, 5 volumes each; y z-scored per run after the register's drop. SE by
          bootstrap: tb over sequences; films over ``BLOCK_TR``-volume blocks within each
          film (2 films cannot be resampled). Floor: ``NC ≥ CEILING_MIN`` and
          ``NC ≥ CEILING_SE_K · SE``. Output ``reliability/ceiling/register-<name>/sub-##.npz``
          (``nc, nc_se, floor, valid``) + ``.json``.

Usage:
    python reliability.py films --sub ##
    python reliability.py tb --sub ##
    python reliability.py ceiling --register {films,tb-enc-ts} --sub ##
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
_FS = _REPO / "scripts" / "functional_space"
for _p in (_REPO / "src" / "python", _FS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import encoding as fs_enc  # noqa: E402
import films as fm  # noqa: E402
import grayordinates as go  # noqa: E402

ROLE = "dropped_repeat"
N_NULL = 1000
MIN_SHIFT_TR = 20  # 30 s at TR 1.5: well past the response's autocorrelation
Q = 0.05
SEED = 0
BETA_CAP = 100.0  # functional_space/score_route.py's M3 floor (label-free part)
CEILING_MIN = 0.10  # ceiling floor, absolute part (bounds R²/NC's noise gain at 10x)
CEILING_SE_K = 3.0  # ceiling floor, precision part: NC >= k * its bootstrap SE
N_BOOT = 1000
BLOCK_TR = 10  # films ceiling SE: block length for the within-film block bootstrap


def repeat_rows(windows, sub: str) -> dict[str, dict[str, object]]:
    """Each repeated film's showings for one subject: ``{film: {showing: row}}``."""
    rows = windows[(windows["sub"] == sub) & (windows["role"] == ROLE)]
    leaked = rows[rows["ses"].isin(fm.HELDOUT_SESSIONS)]
    if len(leaked):
        raise RuntimeError(f"sub-{sub}: {len(leaked)} {ROLE} rows come from held-out sessions "
                           f"{sorted(set(leaked['ses']))}; the films mask must never read them")
    if rows.empty:
        raise RuntimeError(f"sub-{sub}: no {ROLE} rows")
    out = {}
    for film, g in rows.groupby("stimulus_id", sort=True):
        if g["showing"].duplicated().any():
            raise RuntimeError(f"sub-{sub} {film}: a showing number repeats")
        out[film] = {str(r.showing): r for r in g.itertuples(index=False)}
    return out


def _zscore(y: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return (y - y.mean(0)) / y.std(0)


def film_stack(showings: dict, cleaned: Path, cache: dict) -> np.ndarray:
    """``(n_showings, T, n_columns)``, every showing on the first showing's grid, z-scored."""
    ref = sorted(showings, key=int)[0]
    slices = fm.paired_slices(showings, ref)
    out = [_zscore(fm.film_series(showings[s], cleaned, cache)[slices[s]].astype(np.float32))
           for s in sorted(showings, key=int)]
    return np.stack(out)


def mean_pair_corr(z: np.ndarray) -> np.ndarray:
    """Mean correlation over showing pairs of ``(n, T, C)`` z-scored series."""
    n, t, _ = z.shape
    total = z.sum(axis=0)
    return ((total ** 2).sum(axis=0) - (z ** 2).sum(axis=(0, 1))) / (n * (n - 1) * t)


def shifted(z: np.ndarray, bounds: list[tuple[int, int]], rng: np.random.Generator) -> np.ndarray:
    """Each showing's segment of each film rolled by an independent offset."""
    out = np.empty_like(z)
    for lo, hi in bounds:
        t = hi - lo
        for i in range(z.shape[0]):
            k = int(rng.integers(MIN_SHIFT_TR, t - MIN_SHIFT_TR + 1))
            out[i, lo:hi] = np.roll(z[i, lo:hi], k, axis=0)
    return out


def bh(p: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg q-values."""
    order = np.argsort(p)
    ranked = p[order] * p.size / np.arange(1, p.size + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.minimum(q, 1.0)
    return out


def _git_head() -> dict:
    def run(*a):
        return subprocess.run(["git", "-C", str(_REPO), *a], capture_output=True, text=True).stdout.strip()
    return {"head": run("rev-parse", "--short", "HEAD"),
            "dirty": bool(run("status", "--porcelain", "--", "scripts/braintwill_encoding"))}


def cmd_films(args) -> None:
    t0 = time.time()
    paths = fs_enc.Paths()
    cleaned = go.tree_root(paths.derivatives)
    films = repeat_rows(fm.load_windows(paths.derivatives), args.sub)
    n_show = {f: len(s) for f, s in films.items()}
    if len(set(n_show.values())) != 1:
        raise RuntimeError(f"films have different showing counts {n_show}")
    cache, stacks, bounds, at = {}, [], [], 0
    for film, showings in films.items():
        z = film_stack(showings, cleaned, cache)
        stacks.append(z)
        bounds.append((at, at + z.shape[1]))
        at += z.shape[1]
        if z.shape[1] <= 2 * MIN_SHIFT_TR:
            raise RuntimeError(f"{film}: {z.shape[1]} volumes is too short for the shift null")
    z = np.concatenate(stacks, axis=1)
    valid = np.isfinite(z).all(axis=(0, 1))
    zv = z[:, :, valid]
    r = mean_pair_corr(zv)
    rng = np.random.default_rng(SEED)
    exceed = np.zeros(r.size, np.int64)
    for _ in range(N_NULL):
        exceed += mean_pair_corr(shifted(zv, bounds, rng)) >= r
    p = (1 + exceed) / (N_NULL + 1)
    q = bh(p)
    n = z.shape[0]
    sb = n * r / (1 + (n - 1) * r)

    def full(v, fill=np.nan):
        out = np.full(valid.size, fill, dtype=np.float32 if fill is not False else bool)
        out[valid] = v
        return out

    mask = full(q < Q, False)
    dest = paths.derivatives / "encoding_harness" / "reliability" / "register-films"
    dest.mkdir(parents=True, exist_ok=True)
    np.savez(dest / f"sub-{args.sub}.npz", r_mean=full(r), spearman_brown=full(sb), p=full(p), q=full(q),
             mask=mask, valid=valid)
    side = {
        "register": "films", "sub": args.sub, "source_role": ROLE,
        "heldout_sessions_excluded": list(fm.HELDOUT_SESSIONS),
        "films": {f: {"showings": sorted(s, key=int), "sessions": [s[k].ses for k in sorted(s, key=int)],
                      "volumes": int(b[1] - b[0])} for (f, s), b in zip(films.items(), bounds)},
        "statistic": "mean inter-showing correlation over concatenated per-showing z-scored film segments",
        "grid": "each showing on the first showing's volume grid (films.paired_slices, nearest volume)",
        "null": {"kind": "independent circular shift per showing per film", "n": N_NULL,
                 "min_shift_tr": MIN_SHIFT_TR, "seed": SEED},
        "mask": f"Benjamini-Hochberg q < {Q} within subject over valid columns",
        "n_columns": int(valid.size), "n_valid": int(valid.sum()), "n_mask": int(mask.sum()),
        "r_mean_q50_q95_q99": [round(float(v), 4) for v in np.quantile(r, [0.5, 0.95, 0.99])],
        "spearman_brown_q50_q95": [round(float(v), 4) for v in np.quantile(sb, [0.5, 0.95])],
        "mmmdata": _git_head(), "seconds": round(time.time() - t0, 1),
        "created": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    }
    (dest / f"sub-{args.sub}.json").write_text(json.dumps(side, indent=2) + "\n")
    print(json.dumps({k: side[k] for k in ("sub", "n_valid", "n_mask", "r_mean_q50_q95_q99",
                                           "spearman_brown_q50_q95", "seconds")}))


def item_noise(y: np.ndarray, labels: np.ndarray, n_items: int) -> np.ndarray:
    """Item-weighted mean within-item variance (ddof 1) per column; ``labels`` index items 0..n-1."""
    from scipy import sparse

    m = sparse.csr_matrix((np.ones(labels.size), (labels, np.arange(labels.size))), shape=(n_items, labels.size))
    n = np.asarray(m.sum(axis=1)).ravel()[:, None]
    s = m @ y
    ss = m @ (y * y)
    return ((ss - s * s / n) / (n - 1)).mean(axis=0)


def permute_within(labels: np.ndarray, blocks: list[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    out = labels.copy()
    for idx in blocks:
        out[idx] = labels[idx][rng.permutation(idx.size)]
    return out


def cmd_tb(args) -> None:
    import harness as h
    from core.config import load_config

    t0 = time.time()
    paths = h.Paths()
    trials = h.tb_trials(paths.derivatives, Path(load_config()["paths"]["bids_project_dir"]), args.sub)
    table = pd.read_csv(go.grayordinates_path(paths.cleaned), sep="\t")
    raw = h.tb_betas(paths.derivatives, args.sub, trials, table)
    y_all = h.zscore_sessions(trials, raw)
    super_ids = h.super_repeats(trials)
    keep = (trials["enCon"] == 2).to_numpy()
    counts = trials[keep].groupby("stimulus_id").size()
    if counts.min() < 2:
        raise RuntimeError(f"sub-{args.sub}: a repeated item has {counts.min()} trial")
    with np.errstate(invalid="ignore"):
        floor = np.isfinite(raw).all(axis=0) & (np.nanmedian(np.abs(raw), axis=0) <= BETA_CAP)
    valid = np.isfinite(y_all[keep]).all(axis=0)
    y = y_all[keep][:, valid].astype(np.float64)
    del raw, y_all
    sub_t = trials[keep].reset_index(drop=True)
    items = sorted(set(sub_t["stimulus_id"]))
    labels = sub_t["stimulus_id"].map({s: i for i, s in enumerate(items)}).to_numpy()
    blocks = list(sub_t.groupby("session", sort=False).indices.values())
    noise = item_noise(y, labels, len(items))
    total = y.var(axis=0, ddof=1)
    ncsnr = np.sqrt(np.maximum(0.0, total - noise)) / np.sqrt(noise)
    rng = np.random.default_rng(SEED)
    exceed = np.zeros(noise.size, np.int64)
    for _ in range(N_NULL):
        exceed += item_noise(y, permute_within(labels, blocks, rng), len(items)) <= noise
    p = (1 + exceed) / (N_NULL + 1)
    q = bh(p)

    def full(v, fill=np.nan):
        out = np.full(valid.size, fill, dtype=np.float32 if fill is not False else bool)
        out[valid] = v
        return out

    mask = full(q < Q, False) & floor
    dest = paths.derivatives / "encoding_harness" / "reliability" / "register-tb-enc"
    dest.mkdir(parents=True, exist_ok=True)
    np.savez(dest / f"sub-{args.sub}.npz", ncsnr=full(ncsnr), noise_var=full(noise), p=full(p), q=full(q),
             floor=floor, mask=mask, valid=valid)
    side = {
        "register": "tb-enc", "sub": args.sub, "source": "TBencoding trials with enCon == 2 (repeated pairs, incl. "
        "super repeats among them); triplets excluded",
        "n_items": len(items), "n_trials": int(keep.sum()),
        "super_repeats_included": sorted(super_ids & set(items)),
        "statistic": "item-weighted mean within-item variance of per-session z-scored betas; ncsnr as NSD",
        "null": {"kind": "item labels permuted within session", "n": N_NULL, "seed": SEED},
        "mask": f"BH q < {Q} AND finite in every trial AND median |raw beta| <= {BETA_CAP}",
        "n_columns": int(valid.size), "n_valid": int(valid.sum()), "n_floor": int(floor.sum()),
        "n_mask": int(mask.sum()),
        "ncsnr_q50_q95_q99": [round(float(v), 4) for v in np.quantile(ncsnr, [0.5, 0.95, 0.99])],
        "mmmdata": _git_head(), "seconds": round(time.time() - t0, 1),
        "created": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    }
    (dest / f"sub-{args.sub}.json").write_text(json.dumps(side, indent=2) + "\n")
    print(json.dumps({k: side[k] for k in ("sub", "n_items", "n_trials", "n_valid", "n_floor", "n_mask",
                                           "ncsnr_q50_q95_q99", "seconds")}))


# ---------------------------------------------------------------------------
# ceiling: one estimator for the time-series registers (tb-nat-encoding settle 3)
# ---------------------------------------------------------------------------

def triplet_presentations(events: pd.DataFrame, tr: float) -> pd.DataFrame:
    """One run's triplet presentations: runs of consecutive slots of one sequence; a new one
    starts at the sequence's first member (pairId ≡ 1 mod 3; the design opens every
    presentation on it, A-B-C and A-C-B alike) or at a repeated item.

    Slots are the ``image`` and ``rest`` rows in onset order; a sequence is
    ``(pairId − 1) // 3`` among the ``enCon == 3`` slots. Columns: ``seq``, ``k0`` (the first
    slot's nominal onset in volumes; an off-grid onset is an error), ``items`` (mmmId
    tuple), ``n_slots``, ``soa`` (nominal slot spacing, NaN for one slot), ``lag`` (max of
    ``onset_actual − onset`` over the slots).
    """
    slots = events[events["trial_type"].isin(["image", "rest"])].sort_values("onset").reset_index(drop=True)
    tri = (slots["trial_type"] == "image") & (pd.to_numeric(slots["enCon"], errors="coerce") == 3)
    seq = pd.Series(pd.NA, index=slots.index, dtype="Int64")
    seq[tri] = ((pd.to_numeric(slots.loc[tri, "pairId"]).astype(int) - 1) // 3).to_numpy()
    item = pd.to_numeric(slots["mmmId"], errors="coerce")
    first = (pd.to_numeric(slots["pairId"], errors="coerce") - 1) % 3 == 0  # every presentation opens on it
    group, g_id, seen, prev = np.full(len(slots), -1), -1, set(), pd.NA
    for i in range(len(slots)):  # a new presentation at a new sequence, its first member, or a repeat
        if pd.isna(seq[i]):
            prev = pd.NA
            continue
        if pd.isna(prev) or seq[i] != prev or first[i] or item[i] in seen:
            g_id, seen = g_id + 1, set()
        seen.add(item[i])
        group[i], prev = g_id, seq[i]
    out = []
    for _, g in slots[group >= 0].groupby(group[group >= 0]):
        on = g["onset"].to_numpy(float)
        k0 = on[0] / tr
        if abs(k0 - round(k0)) > 1e-6:
            raise RuntimeError(f"a triplet slot starts off the volume grid ({on[0]} s, TR {tr})")
        soa = np.diff(on)
        if soa.size and np.ptp(soa) > 1e-6:
            raise RuntimeError(f"uneven slot spacing inside a presentation at {on[0]} s: {soa}")
        out.append({"seq": int(seq[g.index[0]]), "k0": int(round(k0)),
                    "items": tuple(pd.to_numeric(g["mmmId"]).astype(int)), "n_slots": len(g),
                    "soa": float(soa[0]) if soa.size else np.nan,
                    "lag": float((g["onset_actual"] - g["onset"]).max())})
    return pd.DataFrame(out, columns=["seq", "k0", "items", "n_slots", "soa", "lag"])


def identical_volumes(k0: int, n_slots: int, soa: float, tr: float, delays) -> np.ndarray:
    """Volumes whose whole FIR window ``[s − max(d)·TR, s − (min(d) − 1)·TR)`` lies inside
    the presentation ``[k0·TR, k0·TR + n_slots·soa)``."""
    end = k0 * tr + n_slots * soa
    first = k0 + max(delays)
    last = int(np.floor((end + (min(delays) - 1) * tr) / tr + 1e-6))
    return np.arange(first, last + 1)


def ceiling_pairs(pres: pd.DataFrame, tr: float, delays, n_vol: dict, drop: int) -> pd.DataFrame:
    """Identical-input volume pairs over intact, same-order presentations of each sequence.

    ``pres`` holds every run's presentations with ``run`` and ``ses`` columns; a sequence
    is keyed on (ses, seq). Intact = 3 slots. Rows: ``unit`` (sequence key), ``pair``,
    ``run_i, k_i, run_j, k_j`` (volume indices in run time); volumes before ``drop`` or past the run are
    left out (an error if that empties a presentation)."""
    rows, pair = [], 0
    intact = pres[pres["n_slots"] == 3]
    for (ses, seq), g in intact.groupby(["ses", "seq"], sort=True):
        for _, h in g.groupby("items", sort=False):
            p = h.to_dict("records")
            for a in range(len(p)):
                for b in range(a + 1, len(p)):
                    va = identical_volumes(p[a]["k0"], 3, p[a]["soa"], tr, delays)
                    vb = identical_volumes(p[b]["k0"], 3, p[b]["soa"], tr, delays)
                    if va.size != vb.size or va.size == 0:
                        raise RuntimeError(f"ses-{ses} seq {seq}: presentations give {va.size} vs {vb.size} volumes")
                    ok = ((va >= drop) & (va < n_vol[p[a]["run"]]) & (vb >= drop) & (vb < n_vol[p[b]["run"]]))
                    if not ok.all():
                        raise RuntimeError(f"ses-{ses} seq {seq}: an identical-input volume falls outside the kept run")
                    rows += [{"unit": f"{ses}:{seq}", "pair": pair, "run_i": p[a]["run"], "k_i": int(i),
                              "run_j": p[b]["run"], "k_j": int(j), "lag_gap": abs(p[a]["lag"] - p[b]["lag"])}
                             for i, j in zip(va, vb)]
                    pair += 1
    return pd.DataFrame(rows, columns=["unit", "pair", "run_i", "k_i", "run_j", "k_j", "lag_gap"])


def pair_products(z: np.ndarray) -> np.ndarray:
    """``(T, C)``: per volume, the mean over showing pairs of z_i · z_j (its mean over T is
    ``mean_pair_corr``)."""
    n = z.shape[0]
    return (np.square(z.sum(axis=0)) - np.square(z).sum(axis=0)) / (n * (n - 1))


def unit_bootstrap(sums: np.ndarray, counts: np.ndarray, rng: np.random.Generator, n_boot: int = N_BOOT,
                   strata: np.ndarray | None = None) -> np.ndarray:
    """SE of ``Σ sums / Σ counts`` resampling units (rows) with replacement, within strata."""
    strata = np.zeros(len(counts), int) if strata is None else np.asarray(strata)
    w = np.zeros((n_boot, len(counts)))
    for s_ in np.unique(strata):
        idx = np.flatnonzero(strata == s_)
        draw = rng.integers(0, idx.size, (n_boot, idx.size))
        np.add.at(w, (np.repeat(np.arange(n_boot), idx.size), idx[draw].ravel()), 1.0)
    est = (w @ sums) / (w @ counts)[:, None]
    return est.std(axis=0, ddof=1)


def ceiling_floor(nc: np.ndarray, se: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return (nc >= CEILING_MIN) & (nc >= CEILING_SE_K * se)


def _ceiling_films(args, paths) -> tuple[dict, dict]:
    cleaned = go.tree_root(paths.derivatives)
    films = repeat_rows(fm.load_windows(paths.derivatives), args.sub)
    cache, contrib, blocks, strata, meta, at = {}, [], [], [], {}, 0
    for f_i, (film, showings) in enumerate(films.items()):
        z = film_stack(showings, cleaned, cache)  # (n, T, C), z per showing
        n, t, _ = z.shape
        contrib.append(pair_products(z))
        starts = list(range(at, at + t, BLOCK_TR))
        blocks += starts
        strata += [f_i] * len(starts)
        at += t
        meta[film] = {"showings": sorted(showings, key=int), "volumes": t, "pairs": n * (n - 1) // 2}
    c = np.concatenate(contrib)
    valid = np.isfinite(c).all(axis=0)
    cv = c[:, valid].astype(np.float64)
    edges = np.r_[blocks, len(cv)]
    sums = np.add.reduceat(cv, edges[:-1], axis=0)
    counts = np.diff(edges).astype(float)
    nc = cv.mean(axis=0)
    se = unit_bootstrap(sums, counts, np.random.default_rng(SEED), strata=np.array(strata))
    side = {"source_role": ROLE, "heldout_sessions_excluded": list(fm.HELDOUT_SESSIONS), "films": meta,
            "standardisation": "z per showing (each showing on the first showing's grid)",
            "identical_input": "whole film windows (every volume of a showing)",
            "se": {"kind": "block bootstrap within film", "block_tr": BLOCK_TR, "n_blocks": len(counts)}}
    stored = paths.derivatives / "encoding_harness" / "reliability" / "register-films" / f"sub-{args.sub}.npz"
    if stored.exists():
        r = np.load(stored)["r_mean"][valid]
        side["max_abs_diff_vs_films_r_mean"] = float(np.nanmax(np.abs(r - nc)))
    return {"nc": nc, "se": se, "valid": valid}, side


def _ceiling_tb_ts(args, paths) -> tuple[dict, dict]:
    import harness as h
    from core.config import load_config

    bids = Path(load_config()["paths"]["bids_project_dir"])
    manifest = pd.read_csv(Path(paths.cleaned) / "manifest.tsv", sep="\t", dtype=str)
    runs = h.tb_ts_runs(manifest, args.sub)
    tr = float(runs["RepetitionTime"].astype(float).unique().item())
    n_vol = dict(zip(runs["stem"], runs["n_vol"].astype(int)))
    pres = []
    for r in runs.itertuples(index=False):
        ev = bids / f"sub-{args.sub}" / f"ses-{r.ses}" / "func" / f"{r.stem}_events.tsv"
        if not ev.exists():
            raise FileNotFoundError(ev)
        pres.append(triplet_presentations(pd.read_csv(ev, sep="\t"), tr).assign(run=r.stem, ses=r.ses))
    pres = pd.concat(pres, ignore_index=True)
    pairs = ceiling_pairs(pres, tr, fs_enc.DELAYS_TR, n_vol, h.TB_TS_DROP)
    used = sorted(set(pairs["run_i"]) | set(pairs["run_j"]))
    y = {}
    for r in runs[runs["stem"].isin(used)].itertuples(index=False):
        d = go.load_run(Path(paths.cleaned) / r.path, paths.cleaned).data[h.TB_TS_DROP:]
        y[r.stem] = _zscore(d).astype(np.float32)  # the register's standardisation (tb_ts_design)
    n_cols = next(iter(y.values())).shape[1]
    units = sorted(set(pairs["unit"]))
    sums = np.zeros((len(units), n_cols))
    counts = np.zeros(len(units))
    for u_i, (u, g) in enumerate(pairs.groupby("unit", sort=True)):
        a = np.concatenate([y[run][[k - h.TB_TS_DROP]] for run, k in zip(g["run_i"], g["k_i"])])
        b = np.concatenate([y[run][[k - h.TB_TS_DROP]] for run, k in zip(g["run_j"], g["k_j"])])
        sums[u_i] = (a.astype(np.float64) * b).sum(axis=0)
        counts[u_i] = len(g)
    valid = np.isfinite(sums).all(axis=0)
    nc = sums[:, valid].sum(axis=0) / counts.sum()
    se = unit_bootstrap(sums[:, valid], counts, np.random.default_rng(SEED))
    n_pres = pres.groupby(["ses", "seq"]).size()
    side = {"standardisation": f"z per run after dropping {h.TB_TS_DROP} volumes (tb-enc-ts)",
            "identical_input": "intact (3-slot, no rest, one run), same-order triplet presentation pairs; "
                               f"volumes k0+{max(fs_enc.DELAYS_TR)}.. whose FIR window lies inside the triplet",
            "n_runs": len(runs), "n_runs_used": len(used), "n_sequences": int(len(n_pres)),
            "n_presentations": int(len(pres)), "n_intact": int((pres["n_slots"] == 3).sum()),
            "n_units": len(units), "n_pairs": int(pairs["pair"].nunique()), "n_pair_volumes": int(len(pairs)),
            "volumes_per_pair": sorted(int(v) for v in set(pairs.groupby("pair").size())),
            "same_run_pair_volumes": int((pairs["run_i"] == pairs["run_j"]).sum()),
            "onset_lag_gap_q50_max_s": [round(float(pairs["lag_gap"].median()), 3),
                                        round(float(pairs["lag_gap"].max()), 3)],
            "se": {"kind": "bootstrap over sequences", "n_units": len(units)}}
    return {"nc": nc, "se": se, "valid": valid}, side


def cmd_ceiling(args) -> None:
    import harness as h

    t0 = time.time()
    paths = h.Paths()
    est, side = (_ceiling_films if args.register == "films" else _ceiling_tb_ts)(args, paths)
    valid = est["valid"]

    def full(v):
        out = np.full(valid.size, np.nan, np.float32)
        out[valid] = v
        return out

    nc, se = full(est["nc"]), full(est["se"])
    floor = ceiling_floor(nc, se) & valid
    dest = paths.derivatives / "encoding_harness" / "reliability" / "ceiling" / f"register-{args.register}"
    dest.mkdir(parents=True, exist_ok=True)
    np.savez(dest / f"sub-{args.sub}.npz", nc=nc, nc_se=se, floor=floor, valid=valid)
    v = est["nc"]
    side = {"register": args.register, "sub": args.sub,
            "statistic": "mean over repeat pairs and identical-input volumes of y_i * y_j",
            **side, "n_boot": N_BOOT, "seed": SEED,
            "floor": f"NC >= {CEILING_MIN} and NC >= {CEILING_SE_K} x bootstrap SE",
            "n_columns": int(valid.size), "n_valid": int(valid.sum()), "n_floor": int(floor.sum()),
            "n_min_only": int((nc[valid] >= CEILING_MIN).sum()),
            "nc_q50_q95_q99": [round(float(q), 4) for q in np.quantile(v, [0.5, 0.95, 0.99])],
            "se_q50": round(float(np.median(est["se"])), 4),
            "mmmdata": _git_head(), "seconds": round(time.time() - t0, 1),
            "created": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")}
    (dest / f"sub-{args.sub}.json").write_text(json.dumps(side, indent=2) + "\n")
    print(json.dumps({k: side[k] for k in ("register", "sub", "n_valid", "n_floor", "n_min_only", "nc_q50_q95_q99",
                                           "se_q50", "seconds") if k in side}))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="verb", required=True)
    for name in ("films", "tb"):
        sub.add_parser(name).add_argument("--sub", required=True)
    c = sub.add_parser("ceiling")
    c.add_argument("--register", required=True, choices=("films", "tb-enc-ts"))
    c.add_argument("--sub", required=True)
    args = ap.parse_args()
    {"films": cmd_films, "tb": cmd_tb, "ceiling": cmd_ceiling}[args.verb](args)


if __name__ == "__main__":
    main()
