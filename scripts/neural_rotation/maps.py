"""The five map classes of the neural-rotation pilot, one interface.

Every class fits ``R ~ f(E)`` on training items and predicts ``R_hat`` for
new ``E``, in the k-dimensional shared basis of basis.py unless noted:

  (a) Identity, ScaledIdentity
  (b) OrthogonalProcrustes   -- Q in O(k); a det = -1 solution is kept and
                                the proper-rotation solution fitted beside it
  (c) Ridge                  -- alpha by nested CV
  (d) ReducedRank            -- ridge then rank-r truncation (Izenman);
                                r by nested CV
  (e) SemanticWarp           -- feature bottleneck: ridge E -> F, ridge
                                F -> R, composed; the oracle F -> R from the
                                true features is a bound, not a map from E

Rank k of the Procrustes map is the basis dimension, so the rank sweep of
the design record is a basis sweep and there is no separate "partial
Procrustes": the metric is defined in the shared basis, and the lesson of
the cross-subject pilot (rank-k alignment recovers where full-rank voxel
Procrustes overfits) is capacity control, which the basis dimension is.

Shapes: E, R, F are (n_items, k) / (n_items, d_features) float64.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:                 # sibling modules import by name
    sys.path.insert(0, _HERE)
from basis import inner_folds, robust_svd  # noqa: E402


def _score_2afc_all(R_hat: np.ndarray, R: np.ndarray) -> float:
    """Mean over ordered item pairs (i, j != i) of [corr(r_hat_i, r_i) >
    corr(r_hat_i, r_j)] -- the inner-CV scorer (no run matching inside)."""
    A = R_hat - R_hat.mean(axis=1, keepdims=True)
    B = R - R.mean(axis=1, keepdims=True)
    A /= np.linalg.norm(A, axis=1, keepdims=True) + 1e-12
    B /= np.linalg.norm(B, axis=1, keepdims=True) + 1e-12
    C = A @ B.T                        # C[i, j] = corr(r_hat_i, r_j)
    d = np.diag(C)[:, None]
    n = C.shape[0]
    wins = (d > C).sum(axis=1)         # counts j != i automatically (d > d is False)
    return float(wins.sum() / (n * (n - 1)))


def _ridge_solve(X: np.ndarray, Y: np.ndarray, alpha: float) -> np.ndarray:
    """B (p, q) minimising ||X B - Y||² + alpha ||B||², no intercept."""
    p = X.shape[1]
    return np.linalg.solve(X.T @ X + alpha * np.eye(p), X.T @ Y)


def _nested_pick(candidates, fit_predict, E, R, n_folds, rng):
    """argmax over candidates of the mean inner-fold 2AFC score."""
    candidates = list(candidates)
    if len(candidates) == 1:
        return candidates[0], {candidates[0]: float("nan")}
    folds = inner_folds(len(E), n_folds, rng)
    best, best_score, curve = None, -np.inf, {}
    for c in candidates:
        scores = []
        for tr, te in folds:
            R_hat = fit_predict(c, E[tr], R[tr], E[te])
            scores.append(_score_2afc_all(R_hat, R[te]))
        curve[c] = float(np.mean(scores))
        if curve[c] > best_score:
            best, best_score = c, curve[c]
    return best, curve


class _Pinnable:
    """pinned() -> a fresh instance with the fitted hyperparameters fixed, so
    a refit (nested rank search, permutation null) skips the inner search."""

    def pinned(self):
        return type(self)()


@dataclass
class Identity(_Pinnable):
    name: str = "identity"
    n_params: int = 0

    def fit(self, E, R, **kw):
        return self

    def predict(self, E):
        return E


@dataclass
class ScaledIdentity(_Pinnable):
    name: str = "scaled_identity"
    c: float = 1.0
    n_params: int = 1

    def fit(self, E, R, **kw):
        self.c = float((E * R).sum() / max((E * E).sum(), 1e-12))
        return self

    def predict(self, E):
        return self.c * E


@dataclass
class OrthogonalProcrustes(_Pinnable):
    """Q = argmin ||E Q - R||_F over O(k): U S Vt = svd(E.T R), Q = U Vt.

    ``proper=True`` forces det(Q) = +1 (flips the last singular direction
    when the unconstrained solution is a reflection). ``det_sign`` records
    the unconstrained solution's determinant so the table can tag it.
    """
    proper: bool = False
    name: str = "procrustes"
    Q: np.ndarray = field(default=None)
    det_sign: int = 1
    n_params: int = 0

    def fit(self, E, R, **kw):
        U, _, Vt = robust_svd(E.T @ R)
        Q = U @ Vt
        self.det_sign = int(np.sign(np.linalg.det(Q)))
        if self.proper and self.det_sign < 0:
            U = U.copy()
            U[:, -1] *= -1
            Q = U @ Vt
        self.Q = Q
        k = Q.shape[0]
        self.n_params = k * (k - 1) // 2
        if self.proper:
            self.name = "procrustes_proper"
        return self

    def predict(self, E):
        return E @ self.Q

    def pinned(self):
        return OrthogonalProcrustes(proper=self.proper)


@dataclass
class Ridge(_Pinnable):
    alphas: tuple = tuple(np.logspace(-2, 4, 7))
    n_inner: int = 5
    seed: int = 0
    name: str = "ridge"
    B: np.ndarray = field(default=None)
    alpha: float = None
    curve: dict = field(default_factory=dict)
    n_params: int = 0

    def fit(self, E, R, **kw):
        rng = np.random.default_rng(self.seed)
        self.alpha, self.curve = _nested_pick(
            self.alphas, lambda a, Et, Rt, Ev: Ev @ _ridge_solve(Et, Rt, a),
            E, R, self.n_inner, rng)
        self.B = _ridge_solve(E, R, self.alpha)
        self.n_params = self.B.size
        return self

    def predict(self, E):
        return E @ self.B

    def pinned(self):
        return Ridge(alphas=(self.alpha,), seed=self.seed)


@dataclass
class ReducedRank(_Pinnable):
    """Ridge fit then rank-r truncation of the fitted values' column space
    (Izenman): B_r = B P_r P_r^T with P_r the top-r right singular vectors
    of E B."""
    ranks: tuple = (5, 10, 25, 50)
    alpha: float = 1.0
    n_inner: int = 5
    seed: int = 0
    name: str = "reduced_rank"
    B: np.ndarray = field(default=None)
    r: int = None
    curve: dict = field(default_factory=dict)
    n_params: int = 0

    @staticmethod
    def _fit_rank(E, R, alpha, r):
        B = _ridge_solve(E, R, alpha)
        _, _, Vt = robust_svd(E @ B)
        P = Vt[:r].T
        return B @ P @ P.T

    def fit(self, E, R, **kw):
        rng = np.random.default_rng(self.seed)
        k = E.shape[1]
        ranks = [r for r in self.ranks if r <= k] or [k]
        self.r, self.curve = _nested_pick(
            ranks, lambda r, Et, Rt, Ev: Ev @ self._fit_rank(Et, Rt, self.alpha, r),
            E, R, self.n_inner, rng)
        self.B = self._fit_rank(E, R, self.alpha, self.r)
        self.n_params = self.r * (E.shape[1] + R.shape[1])
        return self

    def predict(self, E):
        return E @ self.B

    def pinned(self):
        return ReducedRank(ranks=(self.r,), alpha=self.alpha, seed=self.seed)


@dataclass
class SemanticWarp(_Pinnable):
    """Feature bottleneck: E -> F_hat (ridge), F -> R_hat (ridge), composed.

    fit(E, R, F=...) needs the training items' features F (n, d),
    standardised on training items here. predict(E) goes through F_hat;
    predict_oracle(F_test) uses the true held-out features and is an upper
    bound on what the feature space can say about R, not a map from E.
    """
    alphas: tuple = tuple(np.logspace(-2, 4, 7))
    alphas_fr: tuple = None            # defaults to alphas
    n_inner: int = 5
    seed: int = 0
    name: str = "semantic_warp"
    B_ef: np.ndarray = field(default=None)
    B_fr: np.ndarray = field(default=None)
    f_mean: np.ndarray = field(default=None)
    f_std: np.ndarray = field(default=None)
    alpha_ef: float = None
    alpha_fr: float = None
    n_params: int = 0

    def _std(self, F):
        return (F - self.f_mean) / self.f_std

    def fit(self, E, R, F=None, **kw):
        if F is None:
            raise ValueError("SemanticWarp.fit needs F=(n_train, d) features")
        rng = np.random.default_rng(self.seed)
        self.f_mean, self.f_std = F.mean(axis=0), F.std(axis=0) + 1e-8
        Fs = self._std(F)
        self.alpha_ef, _ = _nested_pick(
            self.alphas, lambda a, Et, Ft, Ev: Ev @ _ridge_solve(Et, Ft, a),
            E, Fs, self.n_inner, rng)
        self.alpha_fr, _ = _nested_pick(
            self.alphas_fr or self.alphas, lambda a, Ft, Rt, Fv: Fv @ _ridge_solve(Ft, Rt, a),
            Fs, R, self.n_inner, rng)
        self.B_ef = _ridge_solve(E, Fs, self.alpha_ef)
        self.B_fr = _ridge_solve(Fs, R, self.alpha_fr)
        self.n_params = self.B_ef.size + self.B_fr.size
        return self

    def predict(self, E):
        return E @ self.B_ef @ self.B_fr

    def predict_oracle(self, F):
        return self._std(F) @ self.B_fr

    def pinned(self):
        return SemanticWarp(alphas=(self.alpha_ef,), alphas_fr=(self.alpha_fr,), seed=self.seed)


CLASS_ORDER = ("identity", "scaled_identity", "procrustes", "procrustes_proper",
               "ridge", "reduced_rank", "semantic_warp", "semantic_oracle")


def make_classes(seed: int = 0, with_features: bool = True) -> list:
    """The class objects the pilot compares, fresh (unfitted)."""
    out = [Identity(), ScaledIdentity(), OrthogonalProcrustes(),
           Ridge(seed=seed), ReducedRank(seed=seed)]
    if with_features:
        out.append(SemanticWarp(seed=seed))
    return out
