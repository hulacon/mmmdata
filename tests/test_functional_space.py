"""functional-space route tooling: film windows, overlap partitions, shrunk Procrustes.

Synthetic inputs only; nothing here reads the dataset.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

FS = Path(__file__).resolve().parents[1] / "scripts" / "functional_space"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"fs_{name}", FS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses resolve their module through sys.modules
    spec.loader.exec_module(mod)
    return mod


pr = _load("procrustes")


def _rotation(p: int, seed: int) -> np.ndarray:
    q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((p, p)))
    return q


# ---------------------------------------------------------------------------
# shrunk Procrustes
# ---------------------------------------------------------------------------

class TestShrunkProcrustes:
    def setup_method(self):
        rng = np.random.default_rng(0)
        self.q = _rotation(8, 1)
        self.x = rng.standard_normal((200, 8))
        self.y = self.x @ self.q + 0.01 * rng.standard_normal((200, 8))

    def test_zero_lambda_recovers_rotation(self):
        r = pr.shrunk_procrustes(self.x, self.y, 0.0)
        assert np.allclose(r, self.q, atol=1e-2)

    def test_infinite_lambda_is_identity(self):
        assert np.array_equal(pr.shrunk_procrustes(self.x, self.y, np.inf), np.eye(8))

    @pytest.mark.parametrize("lam", pr.LAMBDA_GRID)
    def test_always_orthogonal(self, lam):
        r = pr.shrunk_procrustes(self.x, self.y, lam)
        assert np.allclose(r.T @ r, np.eye(8), atol=1e-10)

    def test_shrinkage_moves_toward_identity(self):
        traces = [np.trace(pr.shrunk_procrustes(self.x, self.y, lam)) for lam in pr.LAMBDA_GRID]
        assert all(b >= a - 1e-9 for a, b in zip(traces, traces[1:]))
        assert traces[-1] == pytest.approx(8.0)

    def test_minimises_penalised_objective(self):
        """R beats random orthogonal maps and the identity on the stated objective."""
        lam = 0.5
        m = self.x.T @ self.y
        lam_eff = lam * np.linalg.svd(m, compute_uv=False).mean()

        def loss(r):
            return np.sum((self.x @ r - self.y) ** 2) + lam_eff * np.sum((r - np.eye(8)) ** 2)

        best = loss(pr.shrunk_procrustes(self.x, self.y, lam))
        rivals = [np.eye(8), self.q] + [_rotation(8, s) for s in range(10, 20)]
        assert all(best <= loss(r) + 1e-8 for r in rivals)

    def test_lambda_scale_invariant(self):
        """λ is in units of the piece's mean singular value: rescaling the data changes nothing."""
        a = pr.shrunk_procrustes(self.x, self.y, 0.3)
        b = pr.shrunk_procrustes(10 * self.x, 10 * self.y, 0.3)
        assert np.allclose(a, b)

    def test_rejects_shape_mismatch_and_negative_lambda(self):
        with pytest.raises(ValueError):
            pr.shrunk_procrustes(self.x, self.y[:, :7])
        with pytest.raises(ValueError):
            pr.shrunk_procrustes(self.x, self.y, -1.0)


class TestPiecewise:
    def setup_method(self):
        rng = np.random.default_rng(2)
        self.labels = np.repeat([0, 1, 2], 5)
        self.q = [_rotation(5, s) for s in (3, 4, 5)]
        self.x = rng.standard_normal((150, 15))
        self.y = np.empty_like(self.x)
        for k in range(3):
            c = self.labels == k
            self.y[:, c] = self.x[:, c] @ self.q[k]

    def test_block_diagonal_recovery(self):
        tf = pr.fit_piecewise(self.x, self.y, self.labels)
        assert np.allclose(tf.apply(self.x), self.y, atol=1e-8)

    def test_inverse_is_transpose(self):
        tf = pr.fit_piecewise(self.x, self.y, self.labels)
        assert np.allclose(tf.inverse().apply(self.y), self.x, atol=1e-8)

    def test_invalid_columns_stay_nan(self):
        x = self.x.copy()
        x[:, 3] = np.nan
        tf = pr.fit_piecewise(x, self.y, self.labels)
        out = tf.apply(x)
        assert np.isnan(out[:, 3]).all()
        assert np.isfinite(np.delete(out, 3, axis=1)).all()

    def test_unlabelled_columns_are_nan(self):
        labels = self.labels.copy()
        labels[:5] = -1
        out = pr.fit_piecewise(self.x, self.y, labels).apply(self.x)
        assert np.isnan(out[:, :5]).all() and np.isfinite(out[:, 5:]).all()

    def test_per_piece_lambda(self):
        tf = pr.fit_piecewise(self.x, self.y, self.labels, lam={0: 0.0, 1: np.inf, 2: 0.0})
        assert np.array_equal(tf.pieces[1][1], np.eye(5))
        assert tf.lam == {0: 0.0, 1: float("inf"), 2: 0.0}


class TestTemplateAverage:
    def test_aligned_subjects_agree_better_than_anatomical(self):
        rng = np.random.default_rng(6)
        labels = np.repeat([0, 1], 6)
        shared = rng.standard_normal((300, 12))
        subs = []
        for s in (7, 8):
            d = shared.copy()
            for k in range(2):
                c = labels == k
                d[:, c] = d[:, c] @ _rotation(6, s * 10 + k)
            subs.append(d + 0.1 * rng.standard_normal(d.shape))
        before = np.corrcoef(subs[0].ravel(), subs[1].ravel())[0, 1]
        _, tfs = pr.template_average(subs, labels, lam=0.0)
        a, b = (t.apply(d) for t, d in zip(tfs, subs))
        after = np.corrcoef(a.ravel(), b.ravel())[0, 1]
        assert after > 0.9 > before

    def test_infinite_lambda_is_anatomical_mean(self):
        rng = np.random.default_rng(9)
        subs = [rng.standard_normal((40, 6)) for _ in range(2)]
        tpl, _ = pr.template_average(subs, np.zeros(6, int), lam=np.inf)
        assert np.allclose(tpl, (subs[0] + subs[1]) / 2)


# ---------------------------------------------------------------------------
# film windows
# ---------------------------------------------------------------------------

films = _load("films")


class TestFilmWindows:
    def test_window_is_shifted_and_buffered(self):
        # film at 12 s playing 30 s; TR 1.5: volumes wholly inside [22.5, 46.5)
        start, n = films.film_window(12.0, 30.0, 1.5, 1000, shift=4.5, buffer=6.0)
        assert (start, n) == (15, 16)

    def test_buffer_longer_than_film_gives_empty_window(self):
        assert films.film_window(0.0, 4.0, 1.5, 100, shift=4.5, buffer=6.0)[1] == 0

    def test_roles(self):
        df = pd.DataFrame({
            "sub": ["01"] * 5,
            "ses": ["10", "10", "11", "90", "90"],
            "stimulus_id": ["a", "rep", "rep", "b", "rep"],
        })
        roles = films.assign_roles(df, heldout=("90",)).tolist()
        assert roles == ["alignment", "dropped_repeat", "dropped_repeat", "heldout", "heldout_repeat"]


# ---------------------------------------------------------------------------
# overlap partitions
# ---------------------------------------------------------------------------

parts = _load("partitions")
POOL = [f"film-{i:02d}" for i in range(46)]
SUBS = ("01", "02", "03")


class TestPartitions:
    @pytest.fixture(scope="class")
    def table(self):
        return parts.build_partitions(POOL, SUBS, draws_zero=4, draws_other=3)

    def test_invariants_hold(self, table):
        assert parts.check_partitions(table, POOL, SUBS) == []

    @pytest.mark.parametrize("pct,want", [(0, 16), (25, 20), (50, 24), (100, 31)])
    def test_tuning_counts_match_preregistration(self, pct, want):
        assert parts.expected_tuning(pct, 46) == want

    def test_draws_are_nested(self):
        """Raising the draw count leaves the earlier draws unchanged."""
        small = parts.build_partitions(POOL, SUBS, draws_zero=2, draws_other=2)
        big = parts.build_partitions(POOL, SUBS, draws_zero=5, draws_other=4)
        key = ["scenario", "pct", "draw", "target", "subject", "use"]
        keep = big[big["draw"] < 2].sort_values(key + ["stimulus_id"], ignore_index=True)
        assert keep.equals(small.sort_values(key + ["stimulus_id"], ignore_index=True))

    def test_check_catches_a_leak(self, table):
        bad = table.copy()
        row = bad[(bad["scenario"] == "primary") & (bad["pct"] == 0) & (bad["use"] == "tuning")].index[0]
        job = bad.loc[row, ["scenario", "pct", "draw", "target", "subject"]]
        own = bad[(bad["scenario"] == job["scenario"]) & (bad["pct"] == job["pct"]) & (bad["draw"] == job["draw"])
                  & (bad["target"] == job["target"]) & (bad["subject"] == job["subject"])
                  & (bad["use"] != "tuning")]
        bad.loc[row, "stimulus_id"] = own["stimulus_id"].iat[0]
        assert any("tuning film" in f for f in parts.check_partitions(bad, POOL, SUBS))
