"""GLMsingle operator rebuild: every quantity against an independent computation.

The ridge solve is checked against the ``fracridge`` package itself; leakage and lag sums against
explicit dense matrices built in the test. Synthetic designs only.
"""

import numpy as np
import pytest

pytest.importorskip("glmsingle")
fracridge = pytest.importorskip("fracridge").fracridge

from neuroimaging.glm import glmsingle_operator as go  # noqa: E402

TR, STIMDUR = 1.5, 3.0
N_RUNS, N_TIMES, PER_RUN = 3, 70, 9


def _setup(seed=0):
    rng = np.random.default_rng(seed)
    k = N_RUNS * PER_RUN
    designs = []
    for p in range(N_RUNS):
        d = np.zeros((N_TIMES, k))
        cols = np.arange(p * PER_RUN, (p + 1) * PER_RUN)
        onsets = 4 + 3 * np.arange(PER_RUN)
        perm = rng.permutation(PER_RUN)  # column order differs from onset order
        d[onsets[perm], cols] = 1
        designs.append(d)
    hrf = go.hrf_library(STIMDUR, TR)[:, 6]
    extra = [rng.normal(size=(N_TIMES, 1)), None, rng.normal(size=(N_TIMES, 2))]
    proj = go.nuisance_projectors([N_TIMES] * N_RUNS, [2] * N_RUNS, extra)
    return rng, designs, hrf, proj


def _full_design(designs, hrf, proj):
    from glmsingle.design.convolve_design import convolve_design
    return [P @ convolve_design(d, hrf, {"n_times": d.shape[0], "tr": TR}) for d, P in zip(designs, proj)]


def test_ridge_solve_reproduces_fracridge():
    rng, designs, hrf, proj = _setup()
    bases = go.run_bases(designs, hrf, proj, TR)
    Xs = _full_design(designs, hrf, proj)
    X = np.concatenate(Xs)
    y = np.concatenate([P @ rng.normal(size=(N_TIMES, 6)) for P in proj])
    beta_ols = np.linalg.lstsq(X, y, rcond=None)[0].T
    norm = np.linalg.norm(beta_ols, axis=1)
    for frac in (0.6, 0.2, 0.05):
        ours = go.ridge_betas(bases, beta_ols, go.solve_lambda(bases, beta_ols, frac))
        np.testing.assert_allclose(np.linalg.norm(ours, axis=1) / norm, frac, atol=1e-6)
        # fracridge interpolates its penalty on a grid, so it hits the fraction only approximately;
        # its solution must still lie on the same ridge path, at the fraction it actually achieved.
        ref = fracridge(X, y, frac)[0].T
        achieved = np.linalg.norm(ref, axis=1) / norm
        assert np.all(np.abs(achieved - frac) < 0.02)
        on_path = go.ridge_betas(bases, beta_ols, go.solve_lambda(bases, beta_ols, achieved))
        np.testing.assert_allclose(on_path, ref, rtol=1e-5, atol=1e-8)


def test_fraction_one_is_ols():
    rng, designs, hrf, proj = _setup(1)
    bases = go.run_bases(designs, hrf, proj, TR)
    b = rng.normal(size=(3, N_RUNS * PER_RUN))
    lam = go.solve_lambda(bases, b, 1.0)
    assert np.all(lam == 0)
    np.testing.assert_allclose(go.ridge_betas(bases, b, lam), b, atol=1e-10)


def _direct_per_run(designs, hrf, proj, lam):
    """Per run, in onset order: X_perp, A (on raw y), R."""
    from glmsingle.design.convolve_design import convolve_design
    out = []
    for d, P in zip(designs, proj):
        order = go.trial_order(d)
        Xp = P @ convolve_design(d[:, order], hrf, {"n_times": N_TIMES, "tr": TR})
        G = Xp.T @ Xp
        A = np.linalg.solve(G + lam * np.eye(G.shape[0]), Xp.T) @ P
        out.append((Xp, A, A @ (P @ convolve_design(d[:, order], hrf, {"n_times": N_TIMES, "tr": TR}))))
    return out


@pytest.mark.parametrize("lam", [None, 0.5, 20.0])
def test_leakage_matches_dense_R(lam):
    _, designs, hrf, proj = _setup(2)
    bases = go.run_bases(designs, hrf, proj, TR)
    per = _direct_per_run(designs, hrf, proj, 0.0 if lam is None else lam)
    diag = np.mean(np.concatenate([np.diag(R) for _, _, R in per]))
    for li, l in enumerate((1, 2)):
        off = np.mean(np.concatenate([np.diag(R, k=l) for _, _, R in per]))
        got = go.leakage(bases, None if lam is None else np.array([lam]), lags=(1, 2))[0, li]
        assert got == pytest.approx(off / diag, abs=1e-8)


def test_ols_leaks_nothing():
    _, designs, hrf, proj = _setup(3)
    bases = go.run_bases(designs, hrf, proj, TR)
    np.testing.assert_allclose(go.leakage(bases, None), 0.0, atol=1e-10)


@pytest.mark.parametrize("lam", [None, 3.0])
@pytest.mark.parametrize("rho", [0.0, 0.5])
def test_lag_sums_match_dense_centred_covariances(lam, rho):
    _, designs, hrf, proj = _setup(4)
    bases = go.run_bases(designs, hrf, proj, TR)
    sums = go.lag_sums(bases, None if lam is None else np.array([lam]), rhos=(rho,), lags=(0, 1, 2, 3))
    per = _direct_per_run(designs, hrf, proj, 0.0 if lam is None else lam)
    Sigma = go.ar1_covariance(N_TIMES, rho)
    for li, l in enumerate((0, 1, 2, 3)):
        sig = noi = 0.0
        for _, A, R in per:
            n = R.shape[0]
            C = np.eye(n) - 1.0 / n
            sig += np.trace(C @ R @ R.T @ C, offset=l)
            noi += np.trace(C @ A @ Sigma @ A.T @ C, offset=l)
        assert sums.signal[0, li] == pytest.approx(sig, rel=1e-8, abs=1e-10)
        assert sums.noise[rho][0, li] == pytest.approx(noi, rel=1e-8, abs=1e-10)
    assert sums.n_trials == N_RUNS * PER_RUN
    assert sums.n_pairs[1] == N_RUNS * (PER_RUN - 1)


def test_null_profile_with_no_signal_is_the_noise_correlation():
    _, designs, hrf, proj = _setup(5)
    bases = go.run_bases(designs, hrf, proj, TR)
    sums = go.lag_sums(bases, None, rhos=(0.3,), lags=(0, 1, 2))
    r, noise_only = go.null_profile(sums, 0.3, tau2=np.zeros(1), sigma2=np.ones(1))
    np.testing.assert_allclose(r, noise_only)
    N = sums.noise[0.3][0]
    np.testing.assert_allclose(r, [N[1] / sums.n_pairs[1] / (N[0] / sums.n_trials),
                                   N[2] / sums.n_pairs[2] / (N[0] / sums.n_trials)])


def test_signal_and_noise_variance_recover_known_values():
    rng = np.random.default_rng(6)
    n_items, v = 400, 2000
    tau2, sigma2 = 0.3, 2.0
    item = rng.normal(0, np.sqrt(tau2), size=(v, n_items))
    a = item + rng.normal(0, np.sqrt(sigma2), size=(v, n_items))
    b = item + rng.normal(0, np.sqrt(sigma2), size=(v, n_items))
    centred = np.concatenate([a, b], axis=1)
    pairs = np.column_stack([np.arange(n_items), n_items + np.arange(n_items)])
    t_hat, s_hat = go.signal_and_noise_variance(centred, pairs, unit_noise_diag=np.ones(v),
                                                signal_diag=np.ones(v))
    assert np.median(t_hat) == pytest.approx(tau2, abs=0.03)
    assert np.median(s_hat) == pytest.approx(sigma2, abs=0.05)


def test_trial_order_and_run_bases_refuse_malformed_designs():
    _, designs, hrf, proj = _setup(7)
    bad = designs[0].copy()
    bad[10, np.flatnonzero(bad.any(axis=0))[0]] = 1
    with pytest.raises(ValueError, match="more than one onset"):
        go.trial_order(bad)
    with pytest.raises(ValueError, match="more than one run"):
        go.run_bases([designs[0], designs[0]], hrf, proj[:2], TR)
