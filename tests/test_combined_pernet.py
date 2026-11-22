"""functional-space combined_pernet: the decomposed M2b objective, κ selection, CV halves, piece networks.

Synthetic inputs only; nothing here reads the dataset.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

FS = Path(__file__).resolve().parents[1] / "scripts" / "functional_space"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"fs_{name}", FS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


cp = _load("combined_pernet")
sc = _load("scoring")


def _films(rng, lengths, p, shared=0.6):
    base = [rng.standard_normal((n, p)) for n in lengths]
    a = [sc.zscore_columns(x + (1 - shared) * rng.standard_normal(x.shape)) for x in base]
    b = [sc.zscore_columns(x + (1 - shared) * rng.standard_normal(x.shape)) for x in base]
    return a, b


class TestDecomposedObjective:
    def setup_method(self):
        self.rng = np.random.default_rng(0)
        self.lengths = [43, 31, 57, 20]
        self.p = 24
        self.a, self.b = _films(self.rng, self.lengths, self.p)
        self.bounds = cp._bounds(self.lengths)
        self.seg, self.seg_film = cp.segment_rows(self.lengths)

    def _sums(self, cols_split):
        ya, yb = np.concatenate(self.a), np.concatenate(self.b)
        tot = None
        for cols in cols_split:
            vals = [cp.pattern_sums(ya[:, cols], cp.zscore_films(yb[:, cols], self.bounds), self.seg),
                    cp.pattern_sums(yb[:, cols], cp.zscore_films(ya[:, cols], self.bounds), self.seg)]
            s = {k: np.stack([v[i] for v in vals])[None, None] for i, k in enumerate(("C", "st", "sst", "sp", "ssp"))}
            s["n"] = np.array([[[vals[0][5], vals[1][5]]]], dtype=float)
            tot = s if tot is None else {k: tot[k] + s[k] for k in s}
        return tot

    def _direct(self, films_idx=None):
        keep = range(len(self.lengths)) if films_idx is None else films_idx
        tf = {f"f{k}": self.a[k] for k in keep}
        pf = {f"f{k}": self.b[k] for k in keep}
        r1 = sc.m2b(tf, pf, {"n": np.arange(self.p)})["rank_acc"].mean()
        r2 = sc.m2b(pf, tf, {"n": np.arange(self.p)})["rank_acc"].mean()
        return (r1 + r2) / 2

    def test_segments_drop_remainders(self):
        assert len(self.seg) == sum(n // 10 for n in self.lengths)
        assert np.array_equal(np.bincount(self.seg_film), [n // 10 for n in self.lengths])

    def test_pieces_sum_to_the_whole(self):
        split = [np.arange(0, 10), np.arange(10, 17), np.arange(17, self.p)]
        got = cp.rank_accuracy(self._sums(split))[0, 0]
        assert got == pytest.approx(self._direct(), abs=1e-9)

    def test_subset_ranks_within_subset(self):
        sums = self._sums([np.arange(self.p)])
        idx = np.flatnonzero(np.isin(self.seg_film, [0, 2]))
        assert cp.rank_accuracy(sums, idx)[0, 0] == pytest.approx(self._direct([0, 2]), abs=1e-9)

    def test_nonfinite_column_dropped_like_m2b(self):
        self.b[1][:, 3] = 0.0  # constant in one film -> NaN after z-scoring
        split = [np.arange(0, 12), np.arange(12, self.p)]
        got = cp.rank_accuracy(self._sums(split))[0, 0]
        tf = {f"f{k}": self.a[k] for k in range(4)}
        pf = {f"f{k}": self.b[k] for k in range(4)}
        r1 = sc.m2b(tf, pf, {"n": np.arange(self.p)})["rank_acc"].mean()
        r2 = sc.m2b(pf, tf, {"n": np.arange(self.p)})["rank_acc"].mean()
        assert got == pytest.approx((r1 + r2) / 2, abs=1e-9)


class TestSelection:
    def test_kappa_one_is_global(self):
        obj = np.random.default_rng(1).random((7, 40))
        assert np.all(cp.choose(obj, 1.0) == obj.mean(0).argmax())

    def test_kappa_zero_is_per_network(self):
        obj = np.random.default_rng(2).random((7, 40))
        assert np.array_equal(cp.choose(obj, 0.0), obj.argmax(1))

    def test_ties_go_to_larger_kappa(self):
        obj = np.ones((3, 5))
        k, score = cp.cv_kappa([obj, obj])
        assert k == 1.0 and len(set(score.values())) == 1

    def test_cv_prefers_per_network_when_halves_agree(self):
        rng = np.random.default_rng(3)
        best = rng.integers(0, 40, 7)
        obj = rng.random((7, 40)) * 0.1
        obj[np.arange(7), best] = 1.0
        k, _ = cp.cv_kappa([obj, obj + 0.01 * rng.random((7, 40))])
        assert np.array_equal(cp.choose(obj, k), best)  # the chosen κ keeps every network's own best cell
        assert not np.array_equal(cp.choose(obj, 1.0), best)

    def test_film_halves(self):
        f = cp.film_halves(16, 7)
        assert f.sum() == 8 and np.array_equal(f, cp.film_halves(16, 7))
        assert cp.film_halves(15, 7).sum() == 7


class TestPieceNetworks:
    def test_majority_and_straddle(self):
        import stimulus_route as sr

        nets = np.array(["Vis"] * 6 + ["SomMot"] * 2 + ["", ""] + ["THAL"] * 3, dtype=object)
        common = sr.PieceColumns({"p0": np.arange(0, 8), "p1": np.arange(8, 13)}, {})
        out, share = cp.piece_networks(common, nets)
        assert out == {"p0": "Vis", "p1": ""}
        assert share == pytest.approx(2 / 8)
