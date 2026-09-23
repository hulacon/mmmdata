"""The rotation metric (design record Settles-when 7), four numbers in one
shared basis, from an orthogonal map Q in O(k) and the projections it was
fitted on:

  plane_decomposition  real Schur form of Q: 2x2 rotation blocks with angles
                       theta_j in [0, pi] and 1x1 blocks for fixed (+1) or
                       reflected (-1) axes; each with its variance weight
                       (share of E's variance in that plane)
  mean_plane_angle     sum_j w_j theta_j          -- the headline magnitude
  geodesic_distance    ||log Q||_F / sqrt(2) = sqrt(sum_j theta_j^2) on the
                       proper part
  principal_angles     Bjorck-Golub between the top-m PCs of E and R
  per_item_plane_angle signed angle between e_i and r_i inside a plane
  age_slopes           per-plane OLS slope of that angle on item age
  anchor_regression    similarity ~ lag + absolute time (RoPE relative-time
                       readout)

Shapes: Q (k, k); E, R (n_items, k); plane basis P (k, 2).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import linalg

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from basis import robust_svd  # noqa: E402
from score import item_ci  # noqa: E402


def plane_decomposition(Q: np.ndarray, E: np.ndarray, tol: float = 1e-8) -> list:
    """Invariant planes and axes of Q, each with its variance weight in E.

    Returns a list of dicts sorted by weight (descending):
      kind    'plane' | 'fixed' | 'reflected'
      theta   angle in [0, pi] ('fixed' -> 0, 'reflected' -> pi)
      P       (k, 2) orthonormal plane basis ((k, 1) for an axis)
      weight  share of E's (centred) total variance inside the block,
              normalised so the weights sum to 1 over all blocks
    """
    T, Z = linalg.schur(Q, output="real")
    k = Q.shape[0]
    Ec = E - E.mean(axis=0)
    total = float((Ec ** 2).sum()) or 1.0
    blocks, i = [], 0
    while i < k:
        if i < k - 1 and abs(T[i + 1, i]) > tol:
            theta = float(abs(np.arctan2(T[i + 1, i], T[i, i])))
            P = Z[:, i:i + 2]
            kind = "plane"
            i += 2
        else:
            P = Z[:, i:i + 1]
            kind = "fixed" if T[i, i] > 0 else "reflected"
            theta = 0.0 if kind == "fixed" else float(np.pi)
            i += 1
        w = float(((Ec @ P) ** 2).sum() / total)
        blocks.append({"kind": kind, "theta": theta, "P": P, "weight": w})
    blocks.sort(key=lambda b: -b["weight"])
    return blocks


def mean_plane_angle(blocks: list) -> float:
    """Variance-weighted mean angle over every block, in radians."""
    w = np.array([b["weight"] for b in blocks])
    th = np.array([b["theta"] for b in blocks])
    return float((w * th).sum() / max(w.sum(), 1e-12))


def geodesic_distance(blocks: list) -> tuple[float, bool]:
    """(sqrt(sum theta_j^2) over the rotation planes, reflected_axis_present).
    A det = -1 map has a reflected axis, which has no logarithm in SO(k);
    the distance is reported on the proper part and the flag set."""
    planes = [b["theta"] for b in blocks if b["kind"] == "plane"]
    reflected = any(b["kind"] == "reflected" for b in blocks)
    return float(np.sqrt(np.sum(np.square(planes)))), reflected


def principal_angles(E: np.ndarray, R: np.ndarray, m: int = 10) -> dict:
    """Bjorck-Golub principal angles between the top-m PC subspaces of E and R.

    Returns cosines (m,), angles_deg (m,), mean_cos, and
    share_enc_var_in_ret = ||E U_R U_R^T||_F^2 / ||E||_F^2 (E centred).
    """
    Ec, Rc = E - E.mean(axis=0), R - R.mean(axis=0)
    m = min(m, Ec.shape[1], len(Ec) - 1)
    _, _, VtE = robust_svd(Ec)
    _, _, VtR = robust_svd(Rc)
    UE, UR = VtE[:m].T, VtR[:m].T
    cos = np.clip(robust_svd(UE.T @ UR)[1], 0, 1)
    share = float(((Ec @ UR) ** 2).sum() / max((Ec ** 2).sum(), 1e-12))
    return {"m": m, "cosines": cos, "angles_deg": np.degrees(np.arccos(cos)),
            "mean_cos": float(cos.mean()), "share_enc_var_in_ret": share}


def per_item_plane_angle(E: np.ndarray, R: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Signed angle (n,) in (-pi, pi] from e_i to r_i inside the plane P
    (k, 2): atan2 of the 2x2 determinant and the dot product of the
    projected 2-vectors. NaN where either projection is ~0."""
    a, b = E @ P, R @ P
    det = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    dot = (a * b).sum(axis=1)
    ang = np.arctan2(det, dot)
    small = (np.linalg.norm(a, axis=1) < 1e-10) | (np.linalg.norm(b, axis=1) < 1e-10)
    ang[small] = np.nan
    return ang


def unwrap_to_reference(angles: np.ndarray, reference: float) -> np.ndarray:
    """Shift each angle by 2 pi so it lies within pi of ``reference``
    (a plane's fitted theta), so the slope on age is not broken by the
    branch cut when the typical angle sits near +-pi."""
    return reference + np.angle(np.exp(1j * (angles - reference)))


def age_slopes(angles: np.ndarray, age_days: np.ndarray, mask: np.ndarray,
               n_boot: int = 2000, seed: int = 0) -> dict:
    """OLS slope (deg/day) of per-item angle on age over ``mask`` items with
    a bootstrap-over-items CI. Returns slope, ci_lo, ci_hi, n_items,
    age_dependent (CI excludes 0)."""
    x, y = np.asarray(age_days, float)[mask], np.degrees(np.asarray(angles, float))[mask]
    slope, lo, hi, n = item_ci(x, y, n_boot=n_boot, seed=seed)
    dep = bool(np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0))
    return {"slope_per_day": slope, "ci_lo": lo, "ci_hi": hi, "n_items": n,
            "age_dependent": dep}


def content_position_split(spectrum_rows: list) -> dict:
    """From per-plane rows with 'weight' and 'age_dependent': the variance
    share in age-dependent ('position') vs age-invariant ('content') planes,
    normalised over the planes given."""
    w = np.array([r["weight"] for r in spectrum_rows], float)
    dep = np.array([bool(r["age_dependent"]) for r in spectrum_rows])
    tot = w.sum() or 1.0
    return {"position_share": float(w[dep].sum() / tot),
            "content_share": float(w[~dep].sum() / tot)}


def anchor_regression(y, lag_days, abs_time_days, n_boot: int = 2000, seed: int = 0) -> dict:
    """y ~ 1 + lag_days + abs_time_days by OLS, bootstrap CIs over pairs.
    RoPE's defining property is dependence on relative position only, so
    the absolute-time coefficient is the readout. Returns both coefficients
    with CIs and n."""
    y, l, t = (np.asarray(v, float) for v in (y, lag_days, abs_time_days))
    ok = np.isfinite(y) & np.isfinite(l) & np.isfinite(t)
    y, l, t = y[ok], l[ok], t[ok]
    n = len(y)
    out = {"n": n}
    if n < 4:
        for name in ("lag", "abs_time"):
            out.update({f"{name}_coef": np.nan, f"{name}_ci_lo": np.nan, f"{name}_ci_hi": np.nan})
        return out
    X = np.column_stack([np.ones(n), l, t])

    def fit(idx):
        b, *_ = np.linalg.lstsq(X[idx], y[idx], rcond=None)
        return b[1:]

    coef = fit(np.arange(n))
    rng = np.random.default_rng(seed)
    boots = np.array([fit(rng.integers(0, n, n)) for _ in range(n_boot)])
    for j, name in enumerate(("lag", "abs_time")):
        out[f"{name}_coef"] = float(coef[j])
        out[f"{name}_ci_lo"] = float(np.quantile(boots[:, j], 0.025))
        out[f"{name}_ci_hi"] = float(np.quantile(boots[:, j], 0.975))
    return out
