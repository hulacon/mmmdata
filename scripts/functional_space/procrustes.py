"""Piecewise orthogonal Procrustes with shrinkage toward identity, and template averaging.

The primary transform of the functional-space study (pre-registration §7). For
one piece, with source ``X`` and target ``Y`` (samples x columns, the same
columns in both, since both are in the grayordinate space):

    R = argmin_{R orthogonal} ||X R - Y||^2 + lam_eff * ||R - I||^2

For orthogonal ``R``, ``||R - I||^2 = 2p - 2 tr(R)``, so this is the Procrustes
problem on ``M = X'Y + lam_eff * I``: with ``M = U S V'``, ``R = U V'``.
``lam = 0`` is plain Procrustes; ``lam = inf`` gives ``R = I``, the anatomical
baseline (vertex/voxel identity). ``lam`` is scaled by the piece's mean
singular value of ``X'Y`` (``lam_eff = lam * mean(S)``), so one grid means the
same in every piece (§8).

Columns that are invalid (NaN) in either input are left out of the fit and come
out NaN after ``apply``; nothing is imputed.

Template averaging follows fmralign's ``GroupAlignment``: start from the
subjects' anatomical mean, then alternately fit every subject to the template
and replace the template with the mean of the aligned data.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: λ grid (§8, proposed): {0} ∪ logspace(-2, 2, 6) ∪ {∞}, in units of the piece's mean singular value.
LAMBDA_GRID = (0.0, *np.logspace(-2, 2, 6).tolist(), float("inf"))


def shrunk_procrustes(x: np.ndarray, y: np.ndarray, lam: float = 0.0) -> np.ndarray:
    """Orthogonal ``R`` (p x p) minimising ``||x R - y||^2 + lam_eff ||R - I||^2``."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"source {x.shape} and target {y.shape} differ in shape")
    if lam < 0:
        raise ValueError(f"lam must be >= 0, got {lam}")
    return procrustes_from_cross(x.T @ y, lam)


def procrustes_from_cross(m: np.ndarray, lam: float = 0.0) -> np.ndarray:
    """``R`` from the cross-product ``m = x'y`` alone; storing ``m`` gives ``R`` for any ``lam`` later."""
    if lam < 0:
        raise ValueError(f"lam must be >= 0, got {lam}")
    m = np.asarray(m, dtype=np.float64)
    p = m.shape[0]
    if np.isinf(lam):
        return np.eye(p)
    if lam > 0:
        sv = np.linalg.svd(m, compute_uv=False)
        m = m + lam * sv.mean() * np.eye(p)
    u, _, vt = np.linalg.svd(m, full_matrices=False)
    return u @ vt


@dataclass
class PiecewiseTransform:
    """Per-piece orthogonal maps over a shared column space."""

    n_columns: int
    pieces: dict[object, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)  # label -> (cols, R)
    lam: dict[object, float] = field(default_factory=dict)

    def apply(self, x: np.ndarray) -> np.ndarray:
        """Map source data into the target space; columns outside every fitted piece are NaN."""
        x = np.asarray(x)
        if x.shape[1] != self.n_columns:
            raise ValueError(f"data has {x.shape[1]} columns, transform {self.n_columns}")
        out = np.full(x.shape, np.nan, dtype=np.result_type(x.dtype, np.float32))
        for cols, r in self.pieces.values():
            out[:, cols] = x[:, cols] @ r
        return out

    def inverse(self) -> "PiecewiseTransform":
        """The transpose map (target -> source); exact because every ``R`` is orthogonal."""
        return PiecewiseTransform(self.n_columns, {k: (c, r.T) for k, (c, r) in self.pieces.items()},
                                  dict(self.lam))


def _valid_columns(*arrays: np.ndarray) -> np.ndarray:
    ok = np.ones(arrays[0].shape[1], dtype=bool)
    for a in arrays:
        ok &= np.isfinite(a).all(axis=0)
    return ok


def cross_products(x: np.ndarray, y: np.ndarray, labels: np.ndarray) -> dict[object, tuple[np.ndarray, np.ndarray]]:
    """Per piece label, ``(cols, x[:, cols]' y[:, cols])`` over the columns valid in both.

    ``labels`` gives each column's piece; a label < 0, NaN or ``""`` marks a
    column that belongs to no piece.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    labels = np.asarray(labels)
    if x.shape != y.shape or labels.shape != (x.shape[1],):
        raise ValueError(f"shapes differ: x {x.shape}, y {y.shape}, labels {labels.shape}")
    valid = _valid_columns(x, y)
    out = {}
    for lab in _piece_labels(labels):
        cols = np.flatnonzero((labels == lab) & valid)
        if cols.size:
            out[lab] = (cols, x[:, cols].astype(np.float64).T @ y[:, cols].astype(np.float64))
    return out


def transform_from_cross(cross: dict[object, tuple[np.ndarray, np.ndarray]], n_columns: int, lam=0.0
                         ) -> PiecewiseTransform:
    """The piecewise transform for ``lam`` (a scalar or ``{label: lam}``) from stored cross-products."""
    tf = PiecewiseTransform(n_columns)
    for lab, (cols, m) in cross.items():
        piece_lam = lam[lab] if isinstance(lam, dict) else lam
        tf.pieces[lab] = (cols, procrustes_from_cross(m, piece_lam))
        tf.lam[lab] = float(piece_lam)
    return tf


def fit_piecewise(x: np.ndarray, y: np.ndarray, labels: np.ndarray, lam=0.0) -> PiecewiseTransform:
    """Fit one ``R`` per piece label (see ``cross_products`` for ``labels``)."""
    return transform_from_cross(cross_products(x, y, labels), np.asarray(x).shape[1], lam)


def _piece_labels(labels: np.ndarray) -> list:
    if labels.dtype.kind in "fc":
        keep = np.isfinite(labels) & (labels >= 0)
    elif labels.dtype.kind in "iu":
        keep = labels >= 0
    else:
        keep = np.array([lab not in ("", None) for lab in labels])
    return list(np.unique(labels[keep]))


def template_average(data: list[np.ndarray], labels: np.ndarray, lam=0.0, n_iter: int = 3
                     ) -> tuple[np.ndarray, list[PiecewiseTransform]]:
    """Iterative Procrustes template over subjects that share samples.

    Returns the template (samples x columns) and each subject's transform into
    it. Iteration 0 is the anatomical mean; each later iteration fits every
    subject to the current template and averages the aligned data.
    """
    if len(data) < 2:
        raise ValueError("a template needs at least two subjects")
    shape = data[0].shape
    if any(d.shape != shape for d in data):
        raise ValueError(f"subjects differ in shape: {[d.shape for d in data]}")
    template = np.nanmean(np.stack(data), axis=0)
    transforms: list[PiecewiseTransform] = []
    for _ in range(n_iter):
        transforms = [fit_piecewise(d, template, labels, lam) for d in data]
        template = np.nanmean(np.stack([t.apply(d) for t, d in zip(transforms, data)]), axis=0)
    return template, transforms
