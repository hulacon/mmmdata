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

Output: ``<derivatives>/encoding_harness/reliability/register-films/sub-##.npz``
(``r_mean, spearman_brown, p, q, mask, valid``) + ``sub-##.json`` (method, showings, seed,
code version).

Usage:
    python reliability.py films --sub ##
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="verb", required=True)
    f = sub.add_parser("films")
    f.add_argument("--sub", required=True)
    args = ap.parse_args()
    {"films": cmd_films}[args.verb](args)


if __name__ == "__main__":
    main()
