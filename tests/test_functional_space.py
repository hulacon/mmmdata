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


# ---------------------------------------------------------------------------
# pieces and CHA targets
# ---------------------------------------------------------------------------

pcs = _load("pieces")


def _toy_grayordinates(n_cortex=6, n_sub=2, n_hipp=6):
    rows = []
    for hemi in ("L", "R"):
        rows += [{"piece": "cortex", "structure": f"CORTEX_{hemi}", "hemi": hemi, "vertex": v} for v in range(n_cortex)]
    for hemi in ("L", "R"):
        rows += [{"piece": "hippocampus", "structure": f"HIPPOCAMPUS_{hemi}", "hemi": hemi, "vertex": v}
                 for v in range(n_hipp)]
    rows += [{"piece": "subcortex", "structure": s, "hemi": s[-1], "vertex": -1}
             for s in ("THALAMUS_L", "THALAMUS_L", "PUTAMEN_R")[: n_sub + 1]]
    return pd.DataFrame(rows)


class TestPieces:
    def test_piece_labels(self):
        g = _toy_grayordinates()
        parcels = {"L": np.array([0, 1, 1, 2, 2, 2]), "R": np.array([3, 3, 0, 4, 4, 4])}
        names = {1: "p1", 2: "p2", 3: "p3", 4: "p4"}
        x = np.arange(6, dtype=float)  # unfold long axis
        lab = pcs.piece_labels(g, parcels, names, x)
        assert list(lab[:6]) == ["", "p1", "p1", "p2", "p2", "p2"]
        assert list(lab[6:12]) == ["p3", "p3", "", "p4", "p4", "p4"]
        assert list(lab[12:18]) == [f"HIPPOCAMPUS_L_ax{t}" for t in (1, 1, 2, 2, 3, 3)]
        assert list(lab[-3:]) == ["THALAMUS_L", "THALAMUS_L", "PUTAMEN_R"]

    def test_cortex_tiles_are_nearest_centre_and_skip_the_wall(self):
        rng = np.random.default_rng(0)
        xyz = rng.standard_normal((50, 3))
        xyz /= np.linalg.norm(xyz, axis=1, keepdims=True)
        wall = np.zeros(50, bool)
        wall[[1, 30]] = True
        tiles = pcs.cortex_tiles({"L": xyz, "R": xyz}, {"L": wall, "R": wall}, n_centres=5)
        centres, tile = tiles["L"]
        assert list(centres) == [0, 2, 3, 4]  # vertex 1 is a wall centre, dropped
        assert (tile[wall] == -1).all()
        d = np.linalg.norm(xyz[:, None] - xyz[centres][None], axis=2)
        assert np.array_equal(tile[~wall], d.argmin(axis=1)[~wall])

    def test_target_assignment_covers_every_non_wall_grayordinate(self):
        g = _toy_grayordinates()
        tile = np.array([0, 0, 1, 1, -1, 1])
        assign, names = pcs.target_assignment(g, {"L": (np.array([0, 2]), tile), "R": (np.array([0, 2]), tile)})
        assert names[:4] == ["cortex_L_v0", "cortex_L_v2", "cortex_R_v0", "cortex_R_v2"]
        assert list(assign[:12]) == [0, 0, 1, 1, -1, 1, 2, 2, 3, 3, -1, 3]
        assert set(names[4:]) == {"HIPPOCAMPUS_L", "HIPPOCAMPUS_R", "THALAMUS_L", "PUTAMEN_R"}
        assert (assign[12:] >= 4).all()


cha = _load("cha")


class _ToyGeometry:
    """Nested target levels over pieces of 4 columns: 1 target per piece, per pair, per column."""

    def __init__(self, n_pieces: int = 30):
        n = 4 * n_pieces
        self.labels = np.repeat(np.arange(n_pieces), 4)
        self.levels = {
            "ico3": (np.arange(n) // 4, [f"t{i}" for i in range(n // 4)]),
            "ico4": (np.arange(n) // 2, [f"t{i}" for i in range(n // 2)]),
            "ico5": (np.arange(n), [f"t{i}" for i in range(n)]),
        }
        self.n = n


def _toy_subjects(geo, rotation, seed=0, t=2000, noise=0.3):
    """Shared sparse (non-global) connectivity, independent rest time courses, a
    per-piece orthogonal map per subject, and a shared stimulus response in the
    same per-subject frame."""
    rng = np.random.default_rng(seed)
    mix = np.zeros((2 * geo.n, geo.n))
    for i in range(mix.shape[0]):
        mix[i, rng.choice(geo.n, 5, replace=False)] = rng.standard_normal(5)
    stim = rng.standard_normal((300, mix.shape[0])) @ mix
    q, rest, resp = {}, {}, {}
    for s in ("A", "B", "T"):
        q[s] = np.zeros((geo.n, geo.n))
        for j in range(geo.n // 4):
            q[s][4 * j: 4 * j + 4, 4 * j: 4 * j + 4] = rotation(rng)
        lat = rng.standard_normal((t, mix.shape[0])) @ mix
        rest[s] = (lat @ q[s] + noise * rng.standard_normal((t, geo.n))).astype(np.float32)
        resp[s] = stim @ q[s] + noise * rng.standard_normal(stim.shape)
    return q, rest, resp


def _small_rotation(angle):
    from scipy.linalg import expm

    def draw(rng):
        a = rng.standard_normal((4, 4))
        a = (a - a.T) / 2
        return expm(angle * a / np.linalg.norm(a, 2))
    return draw


def _r(a, b):
    a = (a - a.mean(0)) / a.std(0)
    b = (b - b.mean(0)) / b.std(0)
    return float((a * b).mean())


class TestCha:
    def test_target_signals_are_member_means(self):
        rng = np.random.default_rng(1)
        x = rng.standard_normal((30, 6)).astype(np.float32)
        x[:, 5] = np.nan
        s = cha.target_signals(x, np.array([0, 0, 1, 1, 1, 1]), 2)
        assert np.allclose(s[:, 0], x[:, :2].mean(1), atol=1e-6)
        assert np.allclose(s[:, 1], x[:, 2:5].mean(1), atol=1e-6)  # the NaN member is left out

    def test_target_without_members_is_an_error(self):
        with pytest.raises(ValueError):
            cha.target_signals(np.ones((5, 3), np.float32), np.array([0, 0, 0]), 2)

    def test_profiles_are_correlations(self):
        rng = np.random.default_rng(2)
        x = rng.standard_normal((200, 5)).astype(np.float32)
        s = rng.standard_normal((200, 3)).astype(np.float32)
        p = cha.connectivity_profiles(x, s)
        r = np.corrcoef(np.hstack([s, x]).T)[:3, 3:]
        z = (r - r.mean(0)) / r.std(0)
        assert np.allclose(p, z, atol=1e-4)

    def test_core_recovers_any_rotation_from_common_frame_targets(self):
        """Oracle targets (each subject's data in the true common frame): profile +
        Procrustes map one subject's stimulus response onto another's almost exactly."""
        geo = _ToyGeometry()
        q, rest, resp = _toy_subjects(geo, lambda rng: _rotation(4, int(rng.integers(1 << 30))))
        assign, names = geo.levels["ico5"]
        prof = {s: cha.connectivity_profiles(
            rest[s], cha.target_signals((rest[s] @ q[s].T).astype(np.float32), assign, len(names))) for s in rest}
        tf = {s: pr.fit_piecewise(prof[s], prof["A"], geo.labels) for s in rest}
        mapped = tf["T"].inverse().apply(tf["B"].apply(resp["B"]))
        assert _r(mapped, resp["T"]) > 0.95 > 0.5 > _r(resp["B"], resp["T"])

    def test_bootstraps_from_anatomy_and_beats_it(self):
        """Anatomy roughly right (small in-piece rotations): the full densifying fit
        maps the template subjects' responses onto the target better than identity."""
        geo = _ToyGeometry()
        q, rest, resp = _toy_subjects(geo, _small_rotation(0.6), seed=1)
        cross, diag, _ = cha.fit_target(geo, rest, "T", log=lambda *a: None)
        tf = {s: pr.transform_from_cross(cross[s], geo.n, 0.0) for s in rest}
        for s in ("A", "B"):
            anat = _r(resp[s], resp["T"])
            aligned = _r(tf["T"].inverse().apply(tf[s].apply(resp[s])), resp["T"])
            assert aligned > anat + 0.08
        assert diag["ico5"]["sub-T"]["aligned"] > diag["ico5"]["sub-T"]["anatomical"]
