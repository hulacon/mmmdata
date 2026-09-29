#!/usr/bin/env python3
"""
h1_power_sim.py — simulation-only power of the functional-space H1 decision
rule (pre-registration §9 / §11.8). Uses no data and computes no score.

The rule being simulated:

* per target × network, a one-sided film-level sign-flip test on the
  per-film gains (combined − anatomical, draw-averaged), enumerating all
  2**n_films sign patterns, so p is exact;
* Holm step-down across the networks within each target, at ``--alpha``;
* a network counts if it survives in >= ``--min-targets`` targets;
* go if >= 1 network counts.

Generative model for the per-film gains (unit film-to-film SD, so the effect
size ``d`` is mean gain / SD):

    g[t, n, f] = d * effect[n] + sqrt(rho) * z[t, f] + sqrt(1 - rho) * e[t, n, f]

``z`` is shared by all networks of one target on one film (networks are scored
on the same films), ``e`` is independent. Targets are independent. Gains are
Gaussian, i.e. symmetric, which is the sign-flip test's assumption under H0.

Output: one TSV row per (rho, n_effect_networks, d) with

* ``p_go``: P(go) from any network;
* ``p_go_true``: P(go) via a network that carries an effect;
* ``power_target``: P(an effect network survives Holm in one target);
* ``mc_se``: Monte Carlo SE of ``p_go``.

Usage:
    python h1_power_sim.py --out <dir>/h1_power.tsv
    python h1_power_sim.py --out <dir>/h1_power.tsv --nsim 500   # quick
"""

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np


def sign_matrix(n_films):
    """All 2**n sign patterns as rows; row 0 is all +1 (the observed pattern)."""
    pats = np.array(list(itertools.product([1.0, -1.0], repeat=n_films)))
    assert np.all(pats[0] == 1.0)
    return pats


def signflip_p(gains, signs, chunk=4096):
    """Exact one-sided sign-flip p for each row of ``gains`` (M × n_films).

    p = share of sign patterns whose mean >= the observed mean. The observed
    value is taken from the same product (row 0 of ``signs``), so the identity
    pattern ties with itself exactly and p >= 1 / 2**n.
    """
    out = np.empty(gains.shape[0])
    for s in range(0, gains.shape[0], chunk):
        flipped = gains[s:s + chunk] @ signs.T          # (m, 2**n)
        obs = flipped[:, :1]
        out[s:s + chunk] = (flipped >= obs).mean(axis=1)
    return out


def holm_reject(p, alpha):
    """Holm step-down along the last axis; returns a boolean reject array."""
    k = p.shape[-1]
    order = np.argsort(p, axis=-1)
    p_sorted = np.take_along_axis(p, order, axis=-1)
    thresh = alpha / (k - np.arange(k))
    passed = np.cumprod(p_sorted <= thresh, axis=-1).astype(bool)
    reject = np.empty_like(passed)
    np.put_along_axis(reject, order, passed, axis=-1)
    return reject


def simulate_cell(rng, signs, d, n_eff, rho, nsim, n_targets, n_networks,
                  n_films, alpha, min_targets):
    effect = np.zeros(n_networks)
    effect[:n_eff] = d
    z = rng.standard_normal((nsim, n_targets, 1, n_films))
    e = rng.standard_normal((nsim, n_targets, n_networks, n_films))
    g = effect[None, None, :, None] + np.sqrt(rho) * z + np.sqrt(1 - rho) * e
    p = signflip_p(g.reshape(-1, n_films), signs).reshape(
        nsim, n_targets, n_networks)
    rej = holm_reject(p, alpha)                          # (nsim, T, N)
    counts = rej.sum(axis=1) >= min_targets              # (nsim, N)
    go = counts.any(axis=1)
    go_true = counts[:, :n_eff].any(axis=1) if n_eff else np.zeros(nsim, bool)
    power_target = rej[:, :, :n_eff].mean() if n_eff else np.nan
    return go.mean(), go_true.mean(), power_target


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True, type=Path, help="output TSV")
    ap.add_argument("--nsim", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260928)
    ap.add_argument("--n-films", type=int, default=12)
    ap.add_argument("--n-targets", type=int, default=3)
    ap.add_argument("--n-networks", type=int, default=7)
    ap.add_argument("--min-targets", type=int, default=2)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--rho", type=float, nargs="+", default=[0.0, 0.5])
    ap.add_argument("--d-max", type=float, default=2.0)
    ap.add_argument("--d-step", type=float, default=0.1)
    args = ap.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    signs = sign_matrix(args.n_films)
    d_grid = np.round(np.arange(0, args.d_max + 1e-9, args.d_step), 3)

    rows = []
    for rho in args.rho:
        for n_eff in range(0, args.n_networks + 1):
            for d in (d_grid if n_eff else [0.0]):
                p_go, p_go_true, pw = simulate_cell(
                    rng, signs, d, n_eff, rho, args.nsim, args.n_targets,
                    args.n_networks, args.n_films, args.alpha,
                    args.min_targets)
                se = np.sqrt(p_go * (1 - p_go) / args.nsim)
                rows.append((rho, n_eff, d, p_go, p_go_true, pw, se))
                print(f"rho={rho} n_eff={n_eff} d={d:.2f} p_go={p_go:.3f} "
                      f"power_target={pw:.3f}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write("rho\tn_effect_networks\td\tp_go\tp_go_true\t"
                 "power_target\tmc_se\n")
        for r in rows:
            fh.write("\t".join(f"{v:.4f}" if isinstance(v, float) else str(v)
                               for v in r) + "\n")
    print(f"wrote {args.out} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
