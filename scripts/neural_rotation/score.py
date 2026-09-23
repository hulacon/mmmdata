"""Gate, ceiling, null and CIs for the neural-rotation operator fits.

  identify         run-matched held-out item identification (the gate)
  encoding_ceiling the only ceiling: enc -> enc identity, exposure 1 vs the
                   mean of exposures 2 and 3 on three-exposure test items
                   (there is NO retrieval-side reliability figure -- DECIDED)
  permute_within_run / null_p
                   item identity permuted within run on the training side
  fold_ci          bootstrap over folds
  rdm_correlation  geometry preservation (Settles-when 2)

Shapes: patterns (n_items, k); run_ids (n_items,) any hashable per item.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def _row_corr_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """C[i, j] = Pearson corr(A_i, B_j) across the k dimensions."""
    A = A - A.mean(axis=1, keepdims=True)
    B = B - B.mean(axis=1, keepdims=True)
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return A @ B.T


def identify(R_hat: np.ndarray, R_true: np.ndarray, run_ids) -> dict:
    """Run-matched identification of held-out items.

    For each test item i the candidates are the test items retrieved in the
    same run. acc_2afc = mean over ordered pairs (i, j != i, same run) of
    [corr(r_hat_i, r_i) > corr(r_hat_i, r_j)]; acc_rank = fraction of items
    whose true pattern ranks first among its run's candidates. Items alone
    in their run contribute nothing. Returns both plus n_items, n_pairs.
    """
    run_ids = np.asarray(run_ids)
    C = _row_corr_matrix(R_hat, R_true)
    d = np.diag(C)
    same = run_ids[:, None] == run_ids[None, :]
    np.fill_diagonal(same, False)
    wins = (d[:, None] > C) & same
    n_pairs = int(same.sum())
    n_cand = same.sum(axis=1)
    has = n_cand > 0
    acc_2afc = float(wins.sum() / n_pairs) if n_pairs else float("nan")
    first = (wins.sum(axis=1) == n_cand) & has
    acc_rank = float(first.sum() / has.sum()) if has.any() else float("nan")
    return {"acc_2afc": acc_2afc, "acc_rank": acc_rank,
            "n_items": int(has.sum()), "n_pairs": n_pairs}


def encoding_ceiling(A: np.ndarray, B: np.ndarray, run_ids_A) -> dict:
    """Identity-map identification of exposure-1 patterns (A) from the mean
    of exposures 2+3 (B), run-matched on exposure 1's run. Same basis and
    test items as the cell it accompanies."""
    return identify(B, A, run_ids_A)


def gain_fraction(gain: float, ceiling: float, chance: float = 0.5) -> float:
    """gain as a fraction of the encoding ceiling's excess over chance."""
    excess = ceiling - chance
    return float(gain / excess) if excess > 0 else float("nan")


def permute_within_run(rng: np.random.Generator, run_ids) -> np.ndarray:
    """Permutation index (n,) that shuffles items within each run."""
    run_ids = np.asarray(run_ids)
    perm = np.arange(len(run_ids))
    for r in np.unique(run_ids):
        idx = np.flatnonzero(run_ids == r)
        perm[idx] = rng.permutation(idx)
    return perm


def null_p(observed: float, null_values) -> float:
    """P(null >= observed), with the +1 correction."""
    null_values = np.asarray(null_values, dtype=float)
    null_values = null_values[np.isfinite(null_values)]
    if null_values.size == 0 or not np.isfinite(observed):
        return float("nan")
    return float((1 + (null_values >= observed).sum()) / (1 + null_values.size))


def fold_ci(values, n_boot: int = 10000, seed: int = 0, alpha: float = 0.05):
    """Bootstrap percentile CI of the mean over folds: (mean, lo, hi)."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    if v.size == 1:
        return float(v[0]), float(v[0]), float(v[0])
    rng = np.random.default_rng(seed)
    draws = rng.choice(v, size=(n_boot, v.size), replace=True).mean(axis=1)
    return float(v.mean()), float(np.quantile(draws, alpha / 2)), float(np.quantile(draws, 1 - alpha / 2))


def item_ci(x, y, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05):
    """OLS slope of y on x with a bootstrap-over-items CI: (slope, lo, hi, n)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)
    if n < 3 or np.ptp(x) == 0:
        return float("nan"), float("nan"), float("nan"), n
    slope = float(np.polyfit(x, y, 1)[0])
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if np.ptp(x[idx]) == 0:
            continue
        boots.append(np.polyfit(x[idx], y[idx], 1)[0])
    if not boots:
        return slope, float("nan"), float("nan"), n
    return slope, float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2)), n


def rdm_correlation(E: np.ndarray, R: np.ndarray) -> float:
    """Spearman correlation between the upper triangles of the two item
    RDMs (1 - Pearson across dimensions) of E and R (same items, same order)."""
    if len(E) < 4:
        return float("nan")
    iu = np.triu_indices(len(E), k=1)
    de = 1 - _row_corr_matrix(E, E)[iu]
    dr = 1 - _row_corr_matrix(R, R)[iu]
    rho = stats.spearmanr(de, dr).statistic
    return float(rho) if np.isfinite(rho) else float("nan")
