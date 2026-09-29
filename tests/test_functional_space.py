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


# ---------------------------------------------------------------------------
# encoders
# ---------------------------------------------------------------------------

enc = _load("encoding")


def _frames(length_s, n_feat=3, seed=0, bands=None):
    t = np.arange(0, length_s, enc.FRAME_S)
    x = np.random.default_rng(seed).standard_normal((t.size, n_feat)).astype(np.float32)
    return enc.Features("toy", t, x, bands or {"b": slice(0, n_feat)})


class TestEncoderDesign:
    def test_binned_is_the_mean_of_frames_in_span(self):
        f = _frames(30.0)
        out = enc.binned(f, np.array([0.0, 1.5, 3.0]), 1.5)
        assert np.allclose(out[1], f.x[3:6].mean(0))

    def test_span_outside_the_stimulus_is_an_error(self):
        f = _frames(30.0)
        with pytest.raises(ValueError):
            enc.binned(f, np.array([-1.5]), 1.5)
        with pytest.raises(ValueError):
            enc.binned(f, np.array([27.0]), 1.5, end=28.0)

    @pytest.mark.parametrize("onset,play", [(12.7, 366.0), (390.9, 171.08), (15.48, 238.0), (3.3, 30.2)])
    def test_every_delay_of_every_window_volume_lands_on_the_film(self, onset, play):
        """The decided window (shift 4.5 s, buffer 6 s) supports delays of 3-7 TRs with nothing invented."""
        tr = 1.5
        start, n = films.film_window(onset, play, tr, 10_000)
        f = _frames(play + 2.0)  # a file a little longer than what was shown
        x, bands = enc.delayed(f, enc.film_volume_starts(onset, start, n, tr), tr, end=play)
        assert x.shape == (n, 3 * len(enc.DELAYS_TR)) and np.isfinite(x).all()

    def test_a_shorter_delay_would_leave_the_film(self):
        tr, onset, play = 1.5, 12.7, 60.0
        start, n = films.film_window(onset, play, tr, 10_000)
        with pytest.raises(ValueError):
            enc.delayed(_frames(play), enc.film_volume_starts(onset, start, n, tr), tr, delays=(2, 3), end=play)

    def test_band_layout(self):
        f = _frames(40.0, n_feat=5, bands={"a": slice(0, 2), "b": slice(2, 5)})
        x, bands = enc.delayed(f, enc.probe_volume_starts(f, 1.5), 1.5, delays=(3, 4))
        assert bands == {"a": slice(0, 4), "b": slice(4, 10)}
        assert np.allclose(x[:, 2:4], enc.binned(f, enc.probe_volume_starts(f, 1.5) - 6.0, 1.5)[:, :2])

    def test_probe_grid_keeps_only_fully_covered_volumes(self):
        f = _frames(60.0)
        starts = enc.probe_volume_starts(f, 1.5)
        enc.delayed(f, starts, 1.5)  # no error
        assert starts[0] == pytest.approx(7 * 1.5)

    def test_grouped_splits_keep_films_whole(self):
        groups = np.repeat([f"film{i}" for i in range(7)], 4)
        splits = enc.grouped_splits(groups, 5)
        tested = np.concatenate([te for _, te in splits])
        assert sorted(tested) == list(range(28))
        for tr_idx, te_idx in splits:
            assert not set(groups[tr_idx]) & set(groups[te_idx])


class TestEncoderFit:
    @pytest.fixture(autouse=True)
    def _need_himalaya(self):
        pytest.importorskip("himalaya")

    def _data(self, bands, n=600, seed=0):
        rng = np.random.default_rng(seed)
        p = max(s.stop for s in bands.values())
        x = rng.standard_normal((n, p)).astype(np.float32)
        w = rng.standard_normal((p, 40))
        y = (x @ w + 0.5 * rng.standard_normal((n, 40))).astype(np.float32)
        y[:, 3] = np.nan
        groups = np.repeat(np.arange(10), n // 10)
        return x, y, groups

    @pytest.mark.parametrize("bands", [{"a": slice(0, 20)}, {"a": slice(0, 8), "b": slice(8, 20)}])
    def test_recovers_a_planted_linear_map(self, bands):
        x, y, groups = self._data(bands)
        e = enc.Encoder(bands, n_iter=5, backend="numpy").fit(x[:500], y[:500], groups[:500])
        pred = e.predict(x[500:])
        assert np.isnan(pred[:, 3]).all()
        ok = np.isfinite(y[500:]).all(0)
        r = [np.corrcoef(pred[:, j], y[500:, j])[0, 1] for j in np.flatnonzero(ok)]
        assert np.median(r) > 0.9
        assert 0.0 <= e.diagnostics_["alpha_at_grid_edge"] <= 1.0
        assert e.diagnostics_["alpha_at_grid_edge"] == pytest.approx(
            e.diagnostics_["alpha_at_low_edge"] + e.diagnostics_["alpha_at_high_edge"])


# ---------------------------------------------------------------------------
# stimulus route (Gram-space template and entry)
# ---------------------------------------------------------------------------

sr = _load("stimulus_route")


def _accumulate(pcols, pairs, target, data, n_chunks=3):
    acc = sr.GramAccumulator(pcols, pairs, target)
    for idx in np.array_split(np.arange(next(iter(data.values())).shape[0]), n_chunks):
        acc.add({s: x[idx] for s, x in data.items()})
    return acc


class TestStimulusRoute:
    def _subjects(self, rows=400, n_pieces=3, p=5, seed=0):
        rng = np.random.default_rng(seed)
        labels = np.repeat(np.arange(n_pieces), p)
        shared = rng.standard_normal((rows, labels.size))
        data = {}
        for k, s in enumerate(("A", "B")):
            d = shared.copy()
            for j in range(n_pieces):
                c = labels == j
                d[:, c] = d[:, c] @ _rotation(p, 100 * k + j)
            data[s] = d + 0.2 * rng.standard_normal(d.shape)
        return labels, data

    def test_gram_template_reproduces_template_average(self):
        labels, data = self._subjects()
        valid = {s: np.ones(labels.size, bool) for s in ("A", "B", "T")}
        pcols = sr.piece_columns(labels, valid, ["A", "B"], "T")
        acc = _accumulate(pcols, [("A", "A"), ("A", "B"), ("B", "B")], "T", data)
        rot, cross = sr.gram_template(acc.g, ["A", "B"], list(pcols.template))
        tpl, tfs = pr.template_average([data["A"], data["B"]], labels, lam=0.0, n_iter=sr.TEMPLATE_ITERATIONS)
        for s, tf in zip(("A", "B"), tfs):
            for lab, (cols, r) in tf.pieces.items():
                assert np.allclose(rot[s][lab], r, atol=1e-8)
                assert np.allclose(cross[s][lab], data[s][:, cols].T @ tpl[:, cols], atol=1e-6)

    def test_chunking_does_not_change_the_grams(self):
        labels, data = self._subjects(seed=1)
        valid = {s: np.ones(labels.size, bool) for s in ("A", "B", "T")}
        pcols = sr.piece_columns(labels, valid, ["A", "B"], "T")
        one = _accumulate(pcols, [("A", "B")], "T", data, n_chunks=1).g
        many = _accumulate(pcols, [("A", "B")], "T", data, n_chunks=7).g
        for lab in one[("A", "B")]:
            assert np.allclose(one[("A", "B")][lab], many[("A", "B")][lab], atol=1e-8)

    def test_target_cross_over_its_own_columns(self):
        labels, data = self._subjects(seed=2)
        rng = np.random.default_rng(3)
        valid = {"A": np.ones(labels.size, bool), "B": np.ones(labels.size, bool),
                 "T": rng.random(labels.size) > 0.3}
        valid["B"][0] = False  # a column the template leaves out
        pcols = sr.piece_columns(labels, valid, ["A", "B"], "T")
        assert 0 not in pcols.template[0]
        tpl_acc = _accumulate(pcols, [("A", "A"), ("A", "B"), ("B", "B")], "T", data)
        rot, _ = sr.gram_template(tpl_acc.g, ["A", "B"], list(pcols.template))
        t_rows = {s: rng.standard_normal((120, labels.size)) for s in ("A", "B", "T")}
        tgt_acc = _accumulate(pcols, [("T", "A"), ("T", "B")], "T", t_rows)
        got = sr.target_cross(tgt_acc.g, "T", ["A", "B"], rot, pcols)
        for lab, pos in pcols.target.items():
            cols = pcols.template[lab]
            template = sum(t_rows[s][:, cols] @ rot[s][lab] for s in ("A", "B")) / 2
            want = t_rows["T"][:, cols[pos]].T @ template[:, pos]
            assert np.allclose(got[lab], want, atol=1e-8)

    def test_a_target_on_disjoint_rows_enters_the_template(self):
        """Template on rows A and B share; the target brings rows of its own. Held-out
        responses of a template subject map onto the target's far better than identity."""
        rng = np.random.default_rng(4)
        labels = np.repeat(np.arange(4), 6)
        g = labels.size
        q = {s: np.zeros((g, g)) for s in ("A", "B", "T")}
        for k, s in enumerate(q):
            for j in range(4):
                c = np.flatnonzero(labels == j)
                q[s][np.ix_(c, c)] = _rotation(6, 1000 * k + j)
        signal = lambda n: rng.standard_normal((n, g))  # noqa: E731
        tpl_rows, tgt_rows, test = signal(500), signal(300), signal(200)
        valid = {s: np.ones(g, bool) for s in q}
        pcols = sr.piece_columns(labels, valid, ["A", "B"], "T")
        noisy = lambda x, s: x @ q[s] + 0.3 * rng.standard_normal(x.shape)  # noqa: E731
        tpl_acc = _accumulate(pcols, [("A", "A"), ("A", "B"), ("B", "B")], "T",
                              {s: noisy(tpl_rows, s) for s in ("A", "B")})
        rot, cross = sr.gram_template(tpl_acc.g, ["A", "B"], list(pcols.template))
        tgt_acc = _accumulate(pcols, [("T", "A"), ("T", "B")], "T", {s: noisy(tgt_rows, s) for s in q})
        cross["T"] = sr.target_cross(tgt_acc.g, "T", ["A", "B"], rot, pcols)
        tf = {s: pr.transform_from_cross(sr.as_cross(cross[s], pcols.template), g, 0.0) for s in q}
        mapped = tf["T"].inverse().apply(tf["A"].apply(test @ q["A"]))
        assert _r(mapped, test @ q["T"]) > 0.9 > 0.3 > _r(test @ q["A"], test @ q["T"])
        diag = sr.alignment_diagnostics(cross["T"])
        assert diag["captured"] > 1.0 and diag["tr_over_p"][str(float("inf"))] == 1.0

    def test_probe_batches_keep_clips_whole(self):
        sizes = {"c1": 3, "c2": 5, "c3": 2, "c4": 4}
        design = lambda sid: np.full((sizes[sid], 2), float(sid[1:]))  # noqa: E731
        batches = list(sr.probe_batches(design, list(sizes), chunk=6))
        assert [b[0].shape[0] for b in batches] == [8, 6]
        for x, groups in batches:
            for sid in set(groups):
                assert (x[groups == sid] == float(sid[1:])).all()

    def test_center_blocks(self):
        x = np.arange(12, dtype=float).reshape(6, 2)
        out = sr.center_blocks(x, np.array(["a"] * 3 + ["b"] * 3))
        assert np.allclose(out[:3].mean(0), 0) and np.allclose(out[3:].mean(0), 0)
        assert np.allclose(out.std(0), x[:3].std(0))  # centred, not rescaled
