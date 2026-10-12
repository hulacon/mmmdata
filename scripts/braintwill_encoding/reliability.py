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

Usage:
    python reliability.py films --sub ##
    python reliability.py tb --sub ##
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="verb", required=True)
    for name in ("films", "tb"):
        sub.add_parser(name).add_argument("--sub", required=True)
    args = ap.parse_args()
    {"films": cmd_films, "tb": cmd_tb}[args.verb](args)


if __name__ == "__main__":
    main()
