"""Exact per-voxel operators of a GLMsingle fit, and the trial-to-trial structure they imply.

GLMsingle (Prince et al., 2022) estimates single-trial betas in stages. Type B fits each voxel's
library HRF by OLS. Type C adds the GLMdenoise noise regressors. Type D adds fractional ridge
regression (fracridge) at a per-voxel fraction chosen by cross-validation, then scales and offsets the
result to match the unregularized betas (``wantautoscale``). Given what the fit stored — HRF index,
number of noise PCs, fraction, scale — each type is linear in the voxel's data. Its operator can
therefore be rebuilt from the fit's own outputs without refitting:

    beta_hat = A y,   R = A S,   V = A Sigma A^T

R says how much of every trial's response each trial's estimate carries. V is the estimation
covariance under a noise model Sigma. Both are properties of the design and the stored choices; GLMsingle
does not write them.

**Run structure.** A run's trial columns have support only in that run's time points, and nuisance is
projected per run, so the design's SVD is the union of per-run SVDs. A voxel's ridge penalty is shared
across runs. It is solved from the voxel's fraction with fracridge's definition,
``||beta_lambda|| / ||beta_OLS|| = frac`` in the SVD basis, using the voxel's type-C betas as beta_OLS.

**What is reported, all exact for the stored choices.**
- :func:`leakage`: mean ``R[i, i+l]`` / mean ``R[i, i]`` over within-run trial pairs ``l`` apart.
- :func:`lag_sums`: run-centred lag sums of ``R R^T`` (signal) and ``A Sigma A^T`` (noise: AR(1) at
  each ``rhos`` value, or any stationary autocorrelation passed in ``acfs``). :func:`null_profile`
  combines them into the pattern similarity a null with lag-independent item signal plus temporal
  noise predicts at each lag. A consumer comparing trials by lag can cite it as the estimator's own
  contribution.
- :func:`noise_lag_basis`: the OLS noise lag sums for every autocorrelation at once, as one row per
  autocorrelation lag. A per-voxel noise model (one ACF per voxel) is then a matrix product.

**Not modelled.** The cross-validation that chose the fraction, HRF and PC count. Those are treated as
fixed, so procedure-level properties of the selection are out of scope. Spatially correlated noise does
not change the per-voxel lag sums, but it does change pooled pattern similarity; :func:`null_profile`
ignores it.

The GLMsingle helpers used here (HRF library, polynomial projection, design convolution) are imported
lazily from the installed ``glmsingle`` package, so the operator matches the fit exactly.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

DEFAULT_LAGS = tuple(range(7))


def hrf_library(stimdur: float, tr: float) -> np.ndarray:
    """GLMsingle's 20-kernel HRF library for this stimulus duration and TR, time × 20, unit max."""
    from glmsingle.hrf.gethrf import getcanonicalhrflibrary
    from glmsingle.hrf.normalisemax import normalisemax

    return normalisemax(getcanonicalhrflibrary(stimdur, tr).T, 0)


def nuisance_projectors(
    n_times: Sequence[int],
    maxpolydeg: Sequence[int],
    extra: Sequence[np.ndarray | None] | None = None,
) -> list[np.ndarray]:
    """``I - P`` per run for GLMsingle's polynomials plus any extra regressors (e.g. noise PCs)."""
    from glmsingle.ols.make_poly_matrix import make_polynomial_matrix, make_projection_matrix

    out = []
    for p, n in enumerate(n_times):
        X = make_polynomial_matrix(int(n), int(maxpolydeg[p]))
        if extra is not None and extra[p] is not None and np.size(extra[p]):
            X = np.c_[X, extra[p]]
        out.append(make_projection_matrix(X))
    return out


def trial_order(design: np.ndarray) -> np.ndarray:
    """Trial columns present in one run's single-trial design, in onset order.

    Refuses a column with more than one onset: a single-trial design has exactly one per trial.
    """
    cols = np.flatnonzero(design.any(axis=0))
    if np.any(design[:, cols].sum(axis=0) != 1):
        raise ValueError("a single-trial design column has more than one onset")
    return cols[np.argsort(design[:, cols].argmax(axis=0))]


@dataclass(frozen=True)
class RunBasis:
    """One run's nuisance-projected design in SVD form: ``P X = U diag(s) Vt``.

    ``trials`` are the run's trial columns in onset order; ``Vt``'s columns follow that order.
    """

    trials: np.ndarray
    s: np.ndarray
    U: np.ndarray
    Vt: np.ndarray


def run_bases(designs: Sequence[np.ndarray], hrf: np.ndarray, projectors: Sequence[np.ndarray],
              tr: float) -> list[RunBasis]:
    """Per-run SVD bases of the design GLMsingle fitted for one HRF kernel."""
    from glmsingle.design.convolve_design import convolve_design

    if len(designs) != len(projectors):
        raise ValueError("one nuisance projector per run")
    seen: set[int] = set()
    out = []
    for d, P in zip(designs, projectors):
        order = trial_order(d)
        if seen.intersection(order.tolist()):
            raise ValueError("a trial column appears in more than one run")
        seen.update(order.tolist())
        X = convolve_design(d[:, order], hrf, {"n_times": d.shape[0], "tr": tr})
        U, s, Vt = np.linalg.svd(P @ X, full_matrices=False)
        out.append(RunBasis(trials=order, s=s, U=U, Vt=Vt))
    return out


def _coefficients(bases: Sequence[RunBasis], betas: np.ndarray) -> np.ndarray:
    return np.concatenate([betas[:, b.trials] @ b.Vt.T for b in bases], axis=1)


def solve_lambda(bases: Sequence[RunBasis], beta_ols: np.ndarray, frac: np.ndarray,
                 iters: int = 60) -> np.ndarray:
    """Per-voxel ridge penalty whose solution has ``frac`` of the OLS solution's norm.

    ``beta_ols`` is voxels × trials (any per-voxel scale; the ratio is scale-free). A fraction of 1
    gives 0. Bisection in log lambda between 1e-8 and 1e8 times the largest squared singular value.
    """
    beta_ols = np.atleast_2d(np.asarray(beta_ols, dtype=float))
    frac = np.broadcast_to(np.asarray(frac, dtype=float), (beta_ols.shape[0],))
    s2 = np.concatenate([b.s**2 for b in bases])
    c2 = _coefficients(bases, beta_ols) ** 2
    norm2 = c2.sum(axis=1)
    lo = np.full(frac.shape, np.log(s2.max() * 1e-8))
    hi = np.full(frac.shape, np.log(s2.max() * 1e8))
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        g = s2 / (s2 + np.exp(mid)[:, None])
        f = np.sqrt(np.sum(g**2 * c2, axis=1) / norm2)
        above = f > frac
        lo = np.where(above, mid, lo)
        hi = np.where(above, hi, mid)
    return np.where(frac >= 1.0, 0.0, np.exp(0.5 * (lo + hi)))


def ridge_betas(bases: Sequence[RunBasis], beta_ols: np.ndarray, lam: np.ndarray) -> np.ndarray:
    """The ridge solution at ``lam`` per voxel, from the OLS betas (voxels × trials, same columns)."""
    beta_ols = np.atleast_2d(np.asarray(beta_ols, dtype=float))
    lam = np.broadcast_to(np.asarray(lam, dtype=float), (beta_ols.shape[0],))
    out = np.zeros_like(beta_ols)
    for b in bases:
        c = beta_ols[:, b.trials] @ b.Vt.T
        g = b.s**2 / (b.s**2 + lam[:, None])
        out[:, b.trials] = (g * c) @ b.Vt
    return out


def _gains(s: np.ndarray, lam: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """``(g, d)`` with ``R = Vt^T diag(g) Vt`` and ``A = Vt^T diag(d) U^T``; OLS when ``lam`` is None."""
    if lam is None:
        return np.ones((1, s.size)), (1.0 / s)[None, :]
    lam = np.atleast_1d(np.asarray(lam, dtype=float))[:, None]
    return s**2 / (s**2 + lam), s / (s**2 + lam)


def leakage(bases: Sequence[RunBasis], lam: np.ndarray | None, lags: Sequence[int] = (1, 2)) -> np.ndarray:
    """mean R[i, i+l] / mean R[i, i] over within-run pairs, per voxel (columns follow ``lags``)."""
    num = {l: 0.0 for l in lags}
    n_pairs = {l: 0 for l in lags}
    diag, n_trials = 0.0, 0
    for b in bases:
        g, _ = _gains(b.s, lam)
        n = b.Vt.shape[1]
        diag = diag + g.sum(axis=1)  # rows of Vt are orthonormal
        n_trials += n
        for l in lags:
            if l < n:
                num[l] = num[l] + g @ np.sum(b.Vt[:, : n - l] * b.Vt[:, l:], axis=1)
                n_pairs[l] += n - l
    return np.column_stack([(num[l] / n_pairs[l]) / (diag / n_trials) for l in lags])


def ar1_covariance(n: int, rho: float) -> np.ndarray:
    """Unit-variance AR(1) covariance over ``n`` consecutive samples."""
    if rho == 0:
        return np.eye(n)
    i = np.arange(n)
    return rho ** np.abs(i[:, None] - i[None, :])


@dataclass(frozen=True)
class LagSums:
    """Run-centred lag sums: ``signal[v, l]`` of R R^T, ``noise[key][v, l]`` of A Sigma A^T (unit variance).

    ``key`` is an AR(1) coefficient (float) or the name of an autocorrelation passed to :func:`lag_sums`.
    Lag 0 is the diagonal. ``n_pairs[l]`` and ``n_trials`` turn sums into means.
    """

    lags: tuple[int, ...]
    signal: np.ndarray
    noise: dict[float | str, np.ndarray]
    n_pairs: dict[int, int]
    n_trials: int


def acf_covariance(n: int, acf: np.ndarray) -> np.ndarray:
    """Unit-variance stationary covariance over ``n`` consecutive samples from an autocorrelation
    ``acf[0..]`` (``acf[0]`` must be 1, at least ``n`` lags)."""
    acf = np.asarray(acf, dtype=float)
    if acf.ndim != 1 or acf.size < n:
        raise ValueError(f"autocorrelation needs at least {n} lags, got shape {acf.shape}")
    if not np.isclose(acf[0], 1.0):
        raise ValueError(f"autocorrelation must be 1 at lag 0, got {acf[0]}")
    i = np.arange(n)
    return acf[np.abs(i[:, None] - i[None, :])]


def lag_sums(bases: Sequence[RunBasis], lam: np.ndarray | None = None,
             rhos: Sequence[float] = (0.0,), lags: Sequence[int] = DEFAULT_LAGS,
             acfs: Mapping[str, np.ndarray] | None = None) -> LagSums:
    """Lag sums of the run-centred signal and noise covariances of the estimate.

    Centring is within run over trials, as a consumer that removes each run's mean pattern would do.
    ``lam`` None gives OLS; otherwise one penalty per voxel. Noise is AR(1) at each of ``rhos``, keyed
    by the coefficient, plus one stationary noise model per entry of ``acfs`` (name → autocorrelation,
    as :func:`acf_covariance` takes it), keyed by the name.
    """
    lags = tuple(lags)
    acfs = dict(acfs or {})
    if set(acfs) & set(rhos):
        raise ValueError("an autocorrelation name collides with an AR(1) coefficient key")
    nv = 1 if lam is None else np.atleast_1d(lam).size
    sig = np.zeros((nv, len(lags)))
    keys = list(rhos) + list(acfs)
    noi = {k: np.zeros((nv, len(lags))) for k in keys}
    n_pairs = {l: 0 for l in lags}
    n_trials = 0
    for b in bases:
        g, d = _gains(b.s, lam)
        ut = b.Vt - b.Vt.mean(axis=1, keepdims=True)
        n = ut.shape[1]
        n_trials += n
        nt = b.U.shape[0]
        M = {rho: b.U.T @ ar1_covariance(nt, rho) @ b.U for rho in rhos}
        M.update({name: b.U.T @ acf_covariance(nt, a) @ b.U for name, a in acfs.items()})
        for li, l in enumerate(lags):
            if l >= n:
                continue
            n_pairs[l] += n - l
            W = ut[:, : n - l] @ ut[:, l:].T
            sig[:, li] += g**2 @ np.diag(W)
            for k in keys:
                noi[k][:, li] += np.sum((d @ (M[k] * W)) * d, axis=1)
    return LagSums(lags=lags, signal=sig, noise=noi, n_pairs=n_pairs, n_trials=n_trials)


def noise_lag_basis(bases: Sequence[RunBasis], lags: Sequence[int] = DEFAULT_LAGS) -> np.ndarray:
    """OLS noise lag sums for every stationary autocorrelation at once: ``F[k, l]``.

    For a unit-variance noise model with autocorrelation ``a`` (``a[0] = 1``), the OLS noise lag sum
    at lag ``l`` is ``sum_k a[k] F[k, l]``, so ``acfs @ F`` gives per-voxel lag sums for one ACF per
    voxel, exactly, without one ``U^T Sigma U`` per voxel. ``F`` has one row per autocorrelation lag,
    up to the longest run. Matches ``lag_sums(bases, None, acfs=...)`` for any one ACF.
    """
    lags = tuple(lags)
    K = max(b.U.shape[0] for b in bases)
    F = np.zeros((K, len(lags)))
    for b in bases:
        _, d = _gains(b.s, None)
        dd = d[0]
        ut = b.Vt - b.Vt.mean(axis=1, keepdims=True)
        n = ut.shape[1]
        Ws = {li: ut[:, : n - l] @ ut[:, l:].T for li, l in enumerate(lags) if l < n}
        U = b.U
        for k in range(U.shape[0]):
            # U^T T_k U, with T_k the symmetric shift (ones where |i - j| = k)
            Mk = U.T @ U if k == 0 else U[:-k].T @ U[k:] + U[k:].T @ U[:-k]
            Mk = dd[:, None] * Mk * dd[None, :]
            for li, W in Ws.items():
                F[k, li] += np.sum(Mk * W)
    return F


def null_profile(sums: LagSums, rho: float | str, tau2: np.ndarray, sigma2: np.ndarray,
                 weight: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Predicted pooled pattern r at each lag > 0, and the part of it due to noise alone.

    A ratio of expectations over voxels: ``sum_v w (tau2 S_l + sigma2 N_l) / n_pairs_l`` over the same
    at lag 0 per trial. ``weight`` (e.g. an autoscale factor squared) defaults to 1.
    """
    w = np.ones_like(np.asarray(tau2, dtype=float)) if weight is None else np.asarray(weight, dtype=float)
    S, N = sums.signal, sums.noise[rho]
    i0 = sums.lags.index(0)
    den = np.sum(w * (tau2 * S[:, i0] + sigma2 * N[:, i0])) / sums.n_trials
    out, noise_only = [], []
    for li, l in enumerate(sums.lags):
        if l == 0:
            continue
        out.append(np.sum(w * (tau2 * S[:, li] + sigma2 * N[:, li])) / sums.n_pairs[l] / den)
        noise_only.append(np.sum(w * sigma2 * N[:, li]) / sums.n_pairs[l] / den)
    return np.array(out), np.array(noise_only)


def run_center(betas: np.ndarray, runs: np.ndarray) -> np.ndarray:
    """Subtract each run's mean over trials (voxels × trials in, same out)."""
    out = np.array(betas, dtype=float, copy=True)
    for r in np.unique(runs):
        m = runs == r
        out[:, m] -= out[:, m].mean(axis=1, keepdims=True)
    return out


def signal_and_noise_variance(centred: np.ndarray, pairs: np.ndarray, unit_noise_diag: np.ndarray,
                              signal_diag: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-voxel item-signal variance tau2 and noise variance sigma2 from unregularized betas.

    ``pairs`` (k × 2) index trials showing the same item in different runs, so their noise is
    independent and their covariance estimates tau2. sigma2 is the remaining variance over the
    operator's unit-noise diagonal. Repetition effects between exposures bias tau2 low.
    """
    tau2 = np.mean(centred[:, pairs[:, 0]] * centred[:, pairs[:, 1]], axis=1)
    total = np.mean(centred**2, axis=1)
    sigma2 = np.clip((total - tau2 * signal_diag) / unit_noise_diag, 0, None)
    return tau2, sigma2
