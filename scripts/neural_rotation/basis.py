"""Cleaning, exposure averaging, reliability preselection, block
normalisation and the shared basis for the neural-rotation operator fits.

Conventions (every module in this directory): patterns are ``(n_items,
n_vox)`` float64, item order is the design table's ``mmmId`` order for the
items passed in, and NaN never survives ``clean_voxels``. Voxel cleaning is
the settled 6-cell benchmark's rule (imported, not copied).

Shapes are stated on every public function.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parent.parent


def robust_svd(a: np.ndarray, full_matrices: bool = False):
    """numpy's gesdd, falling back to LAPACK gesvd when gesdd does not
    converge (it can, on ill-conditioned full-rank cross-products; a
    permutation-null draw killed a cluster cell this way on 2026-09-22).
    Same (U, s, Vt) where both succeed."""
    try:
        return np.linalg.svd(a, full_matrices=full_matrices)
    except np.linalg.LinAlgError:
        from scipy import linalg
        return linalg.svd(a, full_matrices=full_matrices, lapack_driver="gesvd")


def _load_module(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def clean_voxels(P: np.ndarray, meanvol, meanvol_frac: float = 0.25,
                 beta_cap: float = 100.0) -> np.ndarray:
    """Boolean keep-vector over voxels, the 6-cell benchmark's rule.

    P: (V, N_trials) betas; meanvol: (V,) or None. Keeps voxels finite in
    every trial, with meanvol >= meanvol_frac x the ROI median, and median
    |beta| over trials <= beta_cap. Delegates to
    retrieval_modeling/benchmark_6cell.py:clean_voxels (single source).
    """
    bench = _load_module("benchmark_6cell",
                         _SCRIPTS / "retrieval_modeling" / "benchmark_6cell.py")
    return bench.clean_voxels(P, meanvol, meanvol_frac, beta_cap)


# ── items from trials ───────────────────────────────────────────────────────

def average_exposures(P_trials: np.ndarray, item_ids: np.ndarray,
                      items: np.ndarray):
    """Mean pattern per item over its trials.

    P_trials: (N_trials, V); item_ids: (N_trials,) the item of each trial;
    items: (n_items,) the items wanted, in the order the rows come back.
    Returns (X (n_items, V), n_exposures (n_items,) int). An item with no
    trial raises: the caller's fold bookkeeping is wrong.
    """
    items = np.asarray(items)
    X = np.zeros((len(items), P_trials.shape[1]), dtype=np.float64)
    n = np.zeros(len(items), dtype=int)
    pos = {it: i for i, it in enumerate(items)}
    for row, it in zip(P_trials, item_ids):
        i = pos.get(it)
        if i is None:
            continue
        X[i] += row
        n[i] += 1
    if (n == 0).any():
        missing = items[n == 0][:5]
        raise ValueError(f"{int((n == 0).sum())} items have no trial (e.g. {missing.tolist()})")
    return X / n[:, None], n


def exposure_halves(P_trials: np.ndarray, item_ids: np.ndarray,
                    exposure: np.ndarray, items: np.ndarray):
    """For three-exposure items: (A (n, V) exposure 1, B (n, V) mean of
    exposures 2 and 3, kept_items (n,)). Items lacking all three exposures
    are dropped (single-exposure items have no halves).

    exposure: (N_trials,) 1-based exposure index of each trial.
    """
    items = np.asarray(items)
    A, B, kept = [], [], []
    by_item = {}
    for i, (it, e) in enumerate(zip(item_ids, exposure)):
        by_item.setdefault(it, {})[int(e)] = i
    for it in items:
        d = by_item.get(it, {})
        if {1, 2, 3} <= set(d):
            A.append(P_trials[d[1]])
            B.append((P_trials[d[2]] + P_trials[d[3]]) / 2)
            kept.append(it)
    if not kept:
        raise ValueError("no item with three exposures among the items given")
    return np.asarray(A), np.asarray(B), np.asarray(kept)


# ── reliability and preselection ────────────────────────────────────────────

def _corr_columns(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pearson r per column between A and B (both (n, V)), across rows."""
    A = A - A.mean(axis=0)
    B = B - B.mean(axis=0)
    num = (A * B).sum(axis=0)
    den = np.sqrt((A * A).sum(axis=0) * (B * B).sum(axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = num / den
    return np.where(np.isfinite(r), r, 0.0)


def split_half_reliability(P_trials: np.ndarray, item_ids: np.ndarray,
                           exposure: np.ndarray, items: np.ndarray) -> np.ndarray:
    """Per-voxel encoding reliability (V,): over the three-exposure items
    among ``items``, the mean over the three exposure pairings of the
    correlation across items between exposure a and exposure b.
    """
    items = np.asarray(items)
    by_item = {}
    for i, (it, e) in enumerate(zip(item_ids, exposure)):
        by_item.setdefault(it, {})[int(e)] = i
    idx = [[by_item[it][e] for e in (1, 2, 3)] for it in items
           if {1, 2, 3} <= set(by_item.get(it, {}))]
    if len(idx) < 3:
        raise ValueError("split-half reliability needs >= 3 three-exposure items")
    idx = np.asarray(idx)
    r = np.zeros(P_trials.shape[1])
    for a, b in ((0, 1), (0, 2), (1, 2)):
        r += _corr_columns(P_trials[idx[:, a]], P_trials[idx[:, b]])
    return r / 3


def preselect(r: np.ndarray, frac: float, block_ids: np.ndarray | None = None) -> np.ndarray:
    """Boolean keep-vector (V,): the top ``frac`` of voxels by ``r`` within
    each block (one block when block_ids is None). frac = 1.0 keeps all.
    At least one voxel per block is always kept."""
    V = len(r)
    if block_ids is None:
        block_ids = np.zeros(V, dtype=int)
    keep = np.zeros(V, dtype=bool)
    for b in np.unique(block_ids):
        idx = np.flatnonzero(block_ids == b)
        n = max(1, int(round(frac * len(idx))))
        order = idx[np.argsort(-r[idx], kind="stable")]
        keep[order[:n]] = True
    return keep


# ── normalisation ───────────────────────────────────────────────────────────

@dataclass
class BlockNormaliser:
    """Per-phase mean removal + per-block scaling, fitted on training items.

    fit(E, R, block_ids): E, R (n_train, V); block_ids (V,) int.
      - translation: mean_R - mean_E over training items (reported)
      - each block is divided by its Frobenius norm / sqrt(n_vox_block),
        computed on the stacked, centred training patterns of BOTH phases,
        so the same units apply to E and R and blocks weigh equally.
    transform(X, phase): (n, V) -> (n, V) with that phase's mean removed.
    """
    mean_e: np.ndarray = field(default=None)
    mean_r: np.ndarray = field(default=None)
    scale: np.ndarray = field(default=None)
    block_ids: np.ndarray = field(default=None)

    def fit(self, E: np.ndarray, R: np.ndarray, block_ids: np.ndarray | None = None):
        V = E.shape[1]
        self.block_ids = np.zeros(V, dtype=int) if block_ids is None else np.asarray(block_ids)
        self.mean_e, self.mean_r = E.mean(axis=0), R.mean(axis=0)
        stacked = np.vstack([E - self.mean_e, R - self.mean_r])
        self.scale = np.ones(V)
        for b in np.unique(self.block_ids):
            cols = self.block_ids == b
            fro = np.sqrt((stacked[:, cols] ** 2).sum())
            s = fro / np.sqrt(cols.sum())
            self.scale[cols] = s if s > 0 else 1.0
        return self

    def transform(self, X: np.ndarray, phase: str) -> np.ndarray:
        mean = self.mean_e if phase == "E" else self.mean_r
        return (X - mean) / self.scale

    @property
    def translation(self) -> np.ndarray:
        """(V,) mean_R - mean_E in the scaled units."""
        return (self.mean_r - self.mean_e) / self.scale

    @property
    def translation_norm(self) -> float:
        return float(np.linalg.norm(self.translation) / np.sqrt(len(self.translation)))


# ── shared basis ────────────────────────────────────────────────────────────

RANK_GRID = (10, 25, 50, 100, 200, "full")


def shared_basis(E: np.ndarray, R: np.ndarray, k_max: int | None = None):
    """PCA basis of the stacked, normalised training patterns.

    E, R: (n_train, V). Returns (W (V, k_max), explained (k_max,) fraction
    of variance per component). k_max defaults to min(V, 2 n_train - 1).
    Projection is X @ W.
    """
    S = np.vstack([E, R])
    S = S - S.mean(axis=0)
    kmax_possible = min(S.shape[1], S.shape[0] - 1)
    k_max = kmax_possible if k_max is None else min(k_max, kmax_possible)
    _, s, Vt = robust_svd(S)
    var = s ** 2
    explained = var / var.sum()
    return Vt[:k_max].T.copy(), explained[:k_max]


def resolve_ranks(grid, k_max: int) -> list:
    """The rank grid as integers <= k_max, deduplicated, 'full' -> k_max."""
    out = []
    for k in grid:
        kk = k_max if k == "full" else int(k)
        if kk <= k_max and kk not in out:
            out.append(kk)
    return sorted(out)


def inner_folds(n: int, n_folds: int, rng: np.random.Generator) -> list:
    """List of (train_idx, test_idx) over n items, items shuffled once."""
    perm = rng.permutation(n)
    parts = np.array_split(perm, n_folds)
    return [(np.concatenate([p for j, p in enumerate(parts) if j != i]), parts[i])
            for i in range(n_folds)]
