"""Split-half ROI reliability: the charter's bake-off metric.

Settles-when 2 of glm-strategy: an estimator wins by mean Dice overlap of
top-N-voxel masks built from odd vs even runs, per subject per category.
Pure numpy so it can score any engine's maps, including ones written by
code that never imported nilearn.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def top_n_mask(stat: np.ndarray, n: int, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Boolean mask of the ``n`` highest values of ``stat`` (within ``mask``).

    NaNs never qualify. If fewer than ``n`` finite voxels exist, all of them
    are selected and no error is raised — the Dice that follows will say so.
    """
    stat = np.asarray(stat, dtype=float)
    if n <= 0:
        raise ValueError("n must be positive")
    valid = np.isfinite(stat)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    flat = np.where(valid.ravel(), stat.ravel(), -np.inf)
    k = min(n, int(valid.sum()))
    out = np.zeros(flat.shape, dtype=bool)
    if k:
        idx = np.argpartition(flat, -k)[-k:]
        out[idx] = True
    return out.reshape(stat.shape)


def dice(a: np.ndarray, b: np.ndarray) -> float:
    """Dice coefficient of two boolean masks; 0 when both are empty."""
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    denom = a.sum() + b.sum()
    if denom == 0:
        return 0.0
    return float(2.0 * np.logical_and(a, b).sum() / denom)


def split_half_dice(
    run_maps: Sequence[np.ndarray], n: int, mask: Optional[np.ndarray] = None
) -> float:
    """Dice between top-N masks of the odd-run mean and the even-run mean.

    ``run_maps`` are per-run statistic maps in run order (1-based odd/even,
    so ``run_maps[0]`` is run 1, odd). Fewer than two runs cannot be split
    and raise.
    """
    if len(run_maps) < 2:
        raise ValueError("split_half_dice needs at least two runs")
    odd = np.mean([m for i, m in enumerate(run_maps) if i % 2 == 0], axis=0)
    even = np.mean([m for i, m in enumerate(run_maps) if i % 2 == 1], axis=0)
    return dice(top_n_mask(odd, n, mask), top_n_mask(even, n, mask))
