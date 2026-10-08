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

    def test_job_list_order_and_shared_subset(self, table):
        jobs = parts.job_list(table)
        assert len(jobs) == len(table.drop_duplicates(["scenario", "pct", "draw", "target"]))
        assert jobs.equals(parts.job_list(table.sample(frac=1, random_state=0)))  # order is input-independent
        shared = parts.job_list(table, shared=True)
        assert (shared["scenario"] == "primary").all() and (shared["s"] > 0).all()
        # the shared list is a contiguous block of the full one, in the same order
        start = jobs.index[(jobs["scenario"] == "primary") & (jobs["s"] > 0)][0]
        assert jobs.iloc[start:start + len(shared)].reset_index(drop=True).equals(shared)

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
        cross, diag, _, _ = cha.fit_target(geo, rest, "T", log=lambda *a: None)
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

    def test_undefined_frames_drop_out_of_the_mean(self):
        """A NaN frame (extractor left it undefined) is left out of its column only."""
        f = _frames(30.0)
        f.x[4, 0] = np.nan
        out = enc.binned(f, np.array([0.0, 1.5, 3.0]), 1.5)
        assert out[1, 0] == pytest.approx(f.x[[3, 5], 0].mean())
        assert out[1, 1] == pytest.approx(f.x[3:6, 1].mean())
        assert np.isfinite(out).all()

    def test_all_finite_frames_bin_as_before(self):
        f = _frames(30.0)
        starts = np.arange(0.0, 27.0, 1.5)
        csum = np.vstack([np.zeros((1, 3)), np.cumsum(f.x, axis=0, dtype=np.float64)])
        lo, hi = (starts / enc.FRAME_S).astype(int), (starts / enc.FRAME_S).astype(int) + 3
        assert np.array_equal(enc.binned(f, starts, 1.5), ((csum[hi] - csum[lo]) / 3.0).astype(np.float32))

    def test_span_with_no_defined_frame_is_an_error(self):
        f = _frames(30.0)
        f.x[3:6, 2] = np.nan
        with pytest.raises(ValueError, match="no defined frame"):
            enc.binned(f, np.array([1.5]), 1.5)

    def test_projection_features_stack_bands_per_stimulus(self):
        import pandas as pd

        t = np.arange(0.0, 3.0, 0.5)
        v = pd.DataFrame({"stimulus_id": ["s"] * 6, "time": t, "V_000": 1.0, "V_001": 2.0})
        a = pd.DataFrame({"stimulus_id": ["s"] * 6, "time": t[::-1], "A_000": np.r_[np.nan, np.arange(5.0)]})
        (f,) = enc.projection_features({"V": v, "A": a})
        assert f.bands == {"V": slice(0, 2), "A": slice(2, 3)}
        assert np.allclose(f.times, t) and np.isnan(f.x[-1, 2]) and f.x[0, 2] == 4.0

    def test_frames_a_later_band_omits_are_undefined(self):
        """`space project` drops a frame its block cannot place; the release keeps it as NaN."""
        import pandas as pd

        v = pd.DataFrame({"stimulus_id": ["s"] * 4, "time": [0.0, 0.5, 1.0, 1.5], "V_000": 1.0})
        a = pd.DataFrame({"stimulus_id": ["s"] * 3, "time": [0.0, 0.5, 1.0], "A_000": 2.0})
        (f,) = enc.projection_features({"V": v, "A": a})
        assert np.array_equal(np.isnan(f.x[:, 1]), [False, False, False, True])

    def test_projection_features_refuse_frames_off_the_first_band(self):
        import pandas as pd

        v = pd.DataFrame({"stimulus_id": ["s"] * 3, "time": [0.0, 0.5, 1.0], "V_000": 1.0})
        a = pd.DataFrame({"stimulus_id": ["s"] * 2, "time": [1.0, 1.5], "A_000": 1.0})
        with pytest.raises(ValueError, match="frames differ"):
            enc.projection_features({"V": v, "A": a})
        w = pd.DataFrame({"stimulus_id": ["s"] * 2, "time": [0.25, 0.75], "V_000": 1.0})
        with pytest.raises(ValueError, match="bin starts"):
            enc.projection_features({"V": w})

    def test_string_times_sort_as_numbers(self):
        """`psytwill space project` writes time as a string; "10.0" must not sort before "2.0"."""
        import pandas as pd

        t = [str(x) for x in np.arange(0.0, 12.0, 0.5)]
        v = pd.DataFrame({"stimulus_id": "s", "time": t[::-1], "V_000": np.arange(24.0)[::-1]})
        (f,) = enc.projection_features({"V": v})
        assert np.array_equal(f.times, np.arange(0.0, 12.0, 0.5)) and np.array_equal(f.x[:, 0], np.arange(24.0))

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

    def test_lag_sweep_peaks_at_zero_and_the_null_collapses(self):
        bands = {"a": slice(0, 20)}
        x, y, groups = self._data(bands, n=1000)
        y = enc._zscore_film_windows(y, groups)  # as the real path does, so 1 + score ~ CV R^2
        q99 = {}
        for k in (-2, 0, 2, "half"):
            d = enc.Encoder(bands, n_iter=5, backend="numpy").fit(
                x, enc.roll_within_windows(y, groups, k), groups).diagnostics_
            q99[k] = d["cv_1_plus_score_q50_q95_q99"][2]
        assert q99[0] > 0.9
        assert max(q99[-2], q99[2], q99["half"]) < 0.2


class TestRollWithinWindows:
    def test_sign_and_window_confinement(self):
        y = np.arange(10, dtype=float)[:, None]
        groups = np.array(["a"] * 6 + ["b"] * 4)
        out = enc.roll_within_windows(y, groups, 1)
        # row t takes row t + 1 of its own window, wrapping inside the window
        assert out[:, 0].tolist() == [1, 2, 3, 4, 5, 0, 7, 8, 9, 6]
        assert enc.roll_within_windows(y, groups, -1)[:, 0].tolist() == [5, 0, 1, 2, 3, 4, 9, 6, 7, 8]

    def test_half_rolls_each_window_by_half_its_length(self):
        y = np.arange(10, dtype=float)[:, None]
        groups = np.array(["a"] * 6 + ["b"] * 4)
        assert enc.roll_within_windows(y, groups, "half")[:, 0].tolist() == [3, 4, 5, 0, 1, 2, 8, 9, 6, 7]


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


# ---------------------------------------------------------------------------
# pairing shared films across subjects
# ---------------------------------------------------------------------------

fm = _load("films")


def _win(start, n, onset, tr=1.5):
    return pd.Series({"start": start, "n": n, "onset": onset, "repetition_time": tr})


class TestPairedSlices:
    def test_nearest_film_time_within_half_a_tr(self):
        tr = 1.5
        rows = {"a": _win(10, 20, 3.0), "b": _win(12, 21, 3.2), "c": _win(8, 19, 0.3)}
        sl = fm.paired_slices(rows, "a")
        lengths = {s.stop - s.start for s in sl.values()}
        assert len(lengths) == 1 and lengths.pop() > 0
        for s, r in rows.items():
            t = (r.start + np.arange(sl[s].start, sl[s].stop)) * tr - r.onset
            t_ref = (rows["a"].start + np.arange(sl["a"].start, sl["a"].stop)) * tr - rows["a"].onset
            assert np.all(np.abs(t - t_ref) <= tr / 2 + 1e-9)
            assert 0 <= sl[s].start and sl[s].stop <= r.n

    def test_whole_volume_offset_is_shifted_not_index_paired(self):
        # b's grid starts exactly one volume later in film time: b's volume 0 is a's volume 1
        rows = {"a": _win(10, 20, 0.0), "b": _win(11, 20, 0.0)}
        sl = fm.paired_slices(rows, "a")
        assert sl == {"a": slice(1, 20), "b": slice(0, 19)}

    def test_no_overlap_is_an_error(self):
        with pytest.raises(ValueError):
            fm.paired_slices({"a": _win(0, 5, 0.0), "b": _win(20, 5, 0.0)}, "a")


# ---------------------------------------------------------------------------
# response route (shared films, measured rows)
# ---------------------------------------------------------------------------

rr = _load("response_route")


def _parts(uses: dict[str, dict[str, str]], pct=50, draw=0, target="03", scenario="primary"):
    return pd.DataFrame([{"scenario": scenario, "pct": pct, "s": 0, "draw": draw, "target": target,
                          "subject": sub, "stimulus_id": sid, "use": use}
                         for sub, m in uses.items() for sid, use in m.items()])


def _windows(subs, films, n=10):
    return pd.DataFrame([{"sub": s, "stimulus_id": f, "role": "alignment", "start": 5 + 20 * i, "n": n,
                          "onset": 3.0 + 0.4 * int(s), "repetition_time": 1.5}
                         for s in subs for i, f in enumerate(films)])


class TestResponseRoute:
    def test_shared_films_are_the_three_way_align_shared_set(self):
        uses = {s: {"f1": "align_shared", "f2": "align_shared", f"u{s}": "align_unique", "t": "tuning"}
                for s in ("03", "04", "05")}
        films, rows = rr.shared_films(_parts(uses), _windows(["03", "04", "05"], ["f1", "f2", "u03", "u04", "u05"]),
                                      "primary", 50, 0, "03")
        assert films == ["f1", "f2"] and sorted(rows) == ["03", "04", "05"]
        assert all(list(r.index) == ["f1", "f2"] for r in rows.values())

    def test_undefined_when_the_target_shares_nothing(self):
        # secondary: template subjects share, the target does not
        uses = {"04": {"f1": "align_shared"}, "05": {"f1": "align_shared"}, "03": {"u": "align_unique"}}
        assert rr.shared_films(_parts(uses), _windows(["03", "04", "05"], ["f1", "u"]), "primary", 50, 0, "03") \
            == ([], {})

    def test_paired_block_equal_rows_zscored(self):
        subs, films = ["03", "04", "05"], ["f1", "f2"]
        w = _windows(subs, films)
        rows = {s: w[w["sub"] == s].set_index("stimulus_id") for s in subs}
        rng = np.random.default_rng(0)
        series = {s: {f: rng.standard_normal((10, 4)) for f in films} for s in subs}
        series["04"]["f1"][:, 2] = 1.0  # constant column -> invalid (NaN), not filled
        blk = rr.paired_block(series, rows, subs, "04", films)
        assert len({b.shape for b in blk.values()}) == 1
        assert np.isnan(blk["04"][:, 2]).any() and np.isfinite(blk["03"]).all()
        assert np.allclose(np.nanmean(blk["03"][: blk["03"].shape[0] // 2], 0), 0, atol=1e-9)

    def test_target_rows_map_into_the_template_block(self):
        subs, films = ["03", "04", "05"], ["f1", "f2"]
        w = _windows(subs, films)
        w.loc[w["sub"] == "03", "start"] += 1  # the target's grid sits a volume later
        rows = {s: w[w["sub"] == s].set_index("stimulus_id") for s in subs}
        rng = np.random.default_rng(1)
        series = {s: {f: rng.standard_normal((10, 3)) for f in films} for s in subs}
        tpl_subs = ["04", "05"]
        idx = rr.target_row_map(rows, tpl_subs, "03", "04", films)
        tpl = rr.paired_block(series, rows, tpl_subs, "04", films)
        tgt = rr.paired_block(series, rows, subs, "04", films)
        assert idx.shape[0] == tgt["04"].shape[0]
        # the reference subject's raw series agree up to the per-slice z-score, so compare ranks per film
        raw_tpl = np.concatenate([series["04"][f][fm.paired_slices({s: rows[s].loc[f] for s in tpl_subs}, "04")["04"]]
                                  for f in films])
        raw_tgt = np.concatenate([series["04"][f][fm.paired_slices({s: rows[s].loc[f] for s in subs}, "04")["04"]]
                                  for f in films])
        assert np.array_equal(raw_tpl[idx], raw_tgt)


# ---------------------------------------------------------------------------
# SRM comparator and PCA control
# ---------------------------------------------------------------------------

srm = _load("srm_route")


class TestSRMRoute:
    def _planted(self, n=400, p=30, k=4, seed=0):
        rng = np.random.default_rng(seed)
        s = rng.standard_normal((n, k))
        ws = [np.linalg.qr(rng.standard_normal((p, k)))[0] for _ in range(3)]
        xs = [s @ w.T + 0.05 * rng.standard_normal((n, p)) for w in ws]
        return xs, ws

    def test_srm_recovers_the_shared_response_through_the_target_entry(self):
        pytest.importorskip("brainiak")
        xs, _ = self._planted()
        tgt_rows = np.arange(20, 380)  # the target block covers a sub-range of the template rows
        w_tpl, w_t, k_eff = srm.fit_piece(xs[:2], xs[2][tgt_rows], tgt_rows, k=4)
        assert k_eff == 4
        # template subject 0 carried into the target's space predicts the target's data
        pred = xs[0][tgt_rows] @ w_tpl[0] @ w_t.T
        r = np.corrcoef(pred.ravel(), xs[2][tgt_rows].ravel())[0, 1]
        assert r > 0.95
        assert np.allclose(w_t.T @ w_t, np.eye(4), atol=1e-6)

    def test_k_is_capped_at_the_piece_size(self):
        pytest.importorskip("brainiak")
        xs, _ = self._planted(p=6, k=3)
        _, w_t, k_eff = srm.fit_piece(xs[:2], xs[2], np.arange(xs[2].shape[0]), k=100)
        assert k_eff == 6 and w_t.shape == (6, 6)

    def test_pca_control_is_one_orthonormal_basis(self):
        xs, _ = self._planted()
        v, k_eff = srm.pca_piece(xs[:2], 4)
        assert k_eff == 4 and np.allclose(v.T @ v, np.eye(4), atol=1e-8)

    def test_all_k_wrapper_slices_the_targets_pca_rows(self):
        pytest.importorskip("brainiak")
        xs, _ = self._planted(p=12, k=3)
        pos = np.array([0, 2, 3, 5, 7, 8, 9, 11])  # the target lacks four of the template's columns
        out = srm.fit_piece_all_k(xs[:2], xs[2][:, pos], np.arange(xs[2].shape[0]), pos, [3, 50])
        for k, (ws, w_t, v, v_t, ke) in out.items():
            assert ke == min(k, pos.size) and w_t.shape == (pos.size, ke) and np.array_equal(v_t, v[pos])


# ---------------------------------------------------------------------------
# combined model
# ---------------------------------------------------------------------------

cb = _load("combined")


class TestCombined:
    def _planted(self, n_pieces=3, p=5, rows=(300, 200), seed=0, noise=0.3):
        """Two blocks of rows for subjects A, B, T sharing one per-piece rotation each."""
        rng = np.random.default_rng(seed)
        labels = np.repeat(np.array([f"p{j}" for j in range(n_pieces)]), p)
        g = labels.size
        q = {s: np.zeros((g, g)) for s in ("A", "B", "T")}
        for k, s in enumerate(q):
            for j in range(n_pieces):
                c = np.flatnonzero(labels == f"p{j}")
                q[s][np.ix_(c, c)] = _rotation(p, 100 * k + j)
        blocks = []
        for n in rows:
            base = rng.standard_normal((n, g))
            blocks.append({s: base @ q[s] + noise * rng.standard_normal((n, g)) for s in q})
        return labels, q, blocks

    def _block(self, name, labels, data, valid=None):
        valid = valid or {s: np.ones(labels.size, bool) for s in ("A", "B", "T")}
        pcols = sr.piece_columns(labels, valid, ["A", "B"], "T")
        tpl = _accumulate(pcols, [("A", "A"), ("A", "B"), ("B", "B")], "T", {s: data[s] for s in ("A", "B")}).g
        tgt = _accumulate(pcols, [("T", "A"), ("T", "B")], "T", data).g
        return cb.Block(name, tpl, tgt, pcols)

    def test_simplex_grid(self):
        assert len(cb.simplex(2)) == 5 and len(cb.simplex(3)) == 15
        assert all(np.isclose(sum(w), 1) for w in cb.simplex(3))
        assert (1.0, 0.0, 0.0) in cb.simplex(3) and (0.5, 0.25, 0.25) in cb.simplex(3)

    def test_energy_scale_of_a_zscored_block_is_one_over_rows(self):
        labels, _, blocks = self._planted()
        z = {s: (x - x.mean(0)) / x.std(0) for s, x in blocks[0].items()}
        assert np.isclose(cb.energy_scale(self._block("b", labels, z), ["A", "B"]), 1 / 300)

    def test_restricted_grams_equal_grams_on_the_common_columns(self):
        labels, _, blocks = self._planted(seed=1)
        rng = np.random.default_rng(2)
        v1 = {s: rng.random(labels.size) > 0.15 for s in ("A", "B", "T")}
        v2 = {s: rng.random(labels.size) > 0.15 for s in ("A", "B", "T")}
        b1, b2 = self._block("one", labels, blocks[0], v1), self._block("two", labels, blocks[1], v2)
        common = cb.common_columns([b1, b2])
        both = {s: v1[s] & v2[s] for s in v1}
        direct = self._block("one", labels, blocks[0], both)
        got = cb.restrict(b1, common)
        assert set(common.template) == set(direct.pcols.template)
        for lab, cols in common.template.items():
            assert np.array_equal(cols, direct.pcols.template[lab])
            for pair in got.tpl:
                assert np.allclose(got.tpl[pair][lab], direct.tpl[pair][lab])
            if lab in common.target:
                assert np.array_equal(common.target[lab], direct.pcols.target[lab])
                for pair in got.tgt:
                    assert np.allclose(got.tgt[pair][lab], direct.tgt[pair][lab])

    def test_stacked_grams_equal_procrustes_on_the_stacked_rows(self):
        """Weighted Gram sums == template averaging on rows stacked with sqrt(w c) scaling."""
        labels, _, blocks = self._planted(seed=3)
        bl = [self._block(f"b{i}", labels, d) for i, d in enumerate(blocks)]
        coefs = [0.25 * cb.energy_scale(bl[0], ["A", "B"]), 0.75 * cb.energy_scale(bl[1], ["A", "B"])]
        stacked = {s: np.vstack([np.sqrt(c) * d[s] for c, d in zip(coefs, blocks)]) for s in ("A", "B", "T")}
        tpl, tfs = pr.template_average([stacked["A"], stacked["B"]], labels, lam=0.0, n_iter=sr.TEMPLATE_ITERATIONS)
        for lab in bl[0].pcols.template:
            out = cb.fit_piece([{pair: b.tpl[pair][lab] for pair in b.tpl} for b in bl],
                               [{pair: b.tgt[pair][lab] for pair in b.tgt} for b in bl], coefs, ["A", "B"], "T",
                               bl[0].pcols.target[lab])
            cols = bl[0].pcols.template[lab]
            for s in ("A", "B", "T"):
                assert np.allclose(out[s], stacked[s][:, cols].T @ tpl[:, cols], atol=1e-6)

    def test_rotation_matches_procrustes_from_cross(self):
        m = np.random.default_rng(4).standard_normal((6, 6))
        for lam in pr.LAMBDA_GRID:
            assert np.allclose(cb._rotation(m, lam, cb._mean_sv(m)), pr.procrustes_from_cross(m, lam))

    def test_tuning_prefers_alignment_when_anatomy_is_wrong_and_identity_is_anatomical(self):
        labels, q, blocks = self._planted(seed=5)
        bl = [self._block(f"b{i}", labels, d) for i, d in enumerate(blocks)]
        coefs = [cb.energy_scale(b, ["A", "B"]) for b in bl]
        rng = np.random.default_rng(6)
        test = rng.standard_normal((150, labels.size))
        y = {s: test @ q[s] + 0.3 * rng.standard_normal(test.shape) for s in ("A", "B")}
        weights = cb.simplex(2)
        results = {}
        for lab, cols in bl[0].pcols.template.items():
            results[lab] = cb.tune_piece([{pair: b.tpl[pair][lab] for pair in b.tpl} for b in bl], coefs, weights,
                                         ["A", "B"], {s: y[s][:, cols] for s in y})
        table = cb.tuning_table(results, weights, ["b0", "b1"])
        inf = table[np.isinf(table["lam"])]
        assert np.allclose(inf["objective"], _r(y["A"], y["B"]), atol=1e-9)  # identity = anatomical
        best = cb.select(table, ["b0", "b1"])
        assert best["b0+b1"]["objective"] > 0.8 > 0.2 > inf["objective"].iat[0]
        assert best["b0+b1"]["objective"] >= max(best["b0"]["objective"], best["b1"]["objective"])

    def test_a_nonfinite_tuning_column_leaves_the_piece_out(self):
        labels, _, blocks = self._planted()
        b = self._block("b", labels, blocks[0])
        lab, cols = next(iter(b.pcols.template.items()))
        y = {s: np.random.default_rng(7).standard_normal((50, cols.size)) for s in ("A", "B")}
        y["B"][:, 0] = np.nan
        assert cb.tune_piece([{pair: b.tpl[pair][lab] for pair in b.tpl}], [1.0], [(1.0,)], ["A", "B"], y) is None

    def test_select_uses_closed_faces(self):
        rows = [{"w_x": wx, "w_y": 1 - wx, "lam": 0.0, "objective": o}
                for wx, o in ((0.0, 0.5), (0.25, 0.1), (0.5, 0.2), (0.75, 0.3), (1.0, 0.4))]
        sel = cb.select(pd.DataFrame(rows), ["x", "y"])
        assert sel["x+y"]["weights"] == {"x": 0.0, "y": 1.0}  # the full model may land on a vertex
        assert sel["x"]["objective"] == 0.4 and sel["y"]["objective"] == 0.5

    def test_grams_round_trip_with_columns(self, tmp_path):
        labels, _, blocks = self._planted()
        rng = np.random.default_rng(8)
        valid = {s: rng.random(labels.size) > 0.2 for s in ("A", "B", "T")}
        b = self._block("b", labels, blocks[0], valid)
        sr.save_grams(b.tpl, tmp_path / "grams_template_x.npz", b.pcols)
        sr.save_grams(b.tgt, tmp_path / "grams_target_x.npz", b.pcols)
        got = cb.load_block("b", tmp_path, ("x",), ["A", "B"], "T")
        for lab in b.pcols.template:
            assert np.array_equal(got.pcols.template[lab], b.pcols.template[lab])
            assert np.allclose(got.tpl[("A", "B")][lab], b.tpl[("A", "B")][lab], atol=1e-4)

    def test_columns_from_crosses_fallback(self, tmp_path):
        labels, _, blocks = self._planted()
        rng = np.random.default_rng(9)
        valid = {s: rng.random(labels.size) > 0.2 for s in ("A", "B", "T")}
        b = self._block("b", labels, blocks[0], valid)
        tcols = {lab: b.pcols.template[lab][pos] for lab, pos in b.pcols.target.items()}
        for s in ("A", "B", "T"):
            cols = tcols if s == "T" else b.pcols.template
            cha.save_cross({lab: (c, np.eye(c.size)) for lab, c in cols.items()}, tmp_path / f"cross_sub-{s}.npz")
        pc_ = cb.columns_from_crosses(tmp_path, ["A", "B"], "T")
        for lab in b.pcols.template:
            assert np.array_equal(pc_.template[lab], b.pcols.template[lab])
        for lab in b.pcols.target:
            assert np.array_equal(pc_.target[lab], b.pcols.target[lab])


# ---------------------------------------------------------------------------
# scoring (synthetic only)
# ---------------------------------------------------------------------------

sc = _load("scoring")


class TestScoring:
    def test_segments_drop_the_remainder(self):
        assert sc.segment_slices(25, 10) == [slice(0, 10), slice(10, 20)]
        assert sc.segment_slices(9, 10) == []

    def test_identification_perfect_random_and_ties(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal((40, 50))
        rank, top1 = sc.identification(x, x + 0.01 * rng.standard_normal(x.shape))
        assert np.allclose(rank, 1) and top1.all()
        rank, _ = sc.identification(x, rng.standard_normal(x.shape))
        assert 0.35 < rank.mean() < 0.65
        same = np.tile(rng.standard_normal(5), (3, 1))  # every pattern identical: all ties
        rank, _ = sc.identification(same + np.arange(3)[:, None] * 0, same)
        assert np.allclose(rank, 0.5)

    def test_m2b_ranks_matched_films_and_zscores_per_film(self):
        rng = np.random.default_rng(1)
        nets = {"n0": np.arange(20), "n1": np.arange(20, 40)}
        films = {f: rng.standard_normal((35, 40)) for f in ("a", "b", "c")}
        tpl = {f: 5.0 + 3.0 * x + 0.5 * rng.standard_normal(x.shape) for f, x in films.items()}  # offset + scale
        df = sc.m2b(films, tpl, nets)
        assert set(df["network"]) == {"n0", "n1"} and df["n_segments"].iat[0] == 9  # 3 per film, remainder 5 dropped
        assert df["rank_acc"].mean() > 0.95
        shuffled = sc.m2b(films, {f: rng.standard_normal(x.shape) for f, x in films.items()}, nets)
        assert 0.3 < shuffled["rank_acc"].mean() < 0.7
        assert set(sc.per_film(df)["film"]) == {"a", "b", "c"}

    def test_m2b_leaves_out_nonfinite_columns(self):
        rng = np.random.default_rng(2)
        films = {f: rng.standard_normal((20, 10)) for f in ("a", "b")}
        tpl = {f: x.copy() for f, x in films.items()}
        tpl["b"][:, 3] = np.nan
        df = sc.m2b(films, tpl, {"n": np.arange(10)})
        assert (df["n_columns"] == 9).all()

    def test_project_with_identity_is_the_template_mean(self):
        rng = np.random.default_rng(3)
        n = 12
        ident = pr.PiecewiseTransform(n, {0: (np.arange(n), np.eye(n))})
        data = {s: rng.standard_normal((5, n)) for s in ("A", "B")}
        got = sc.project(data, {"A": ident, "B": ident, "T": ident}, "T")
        assert np.allclose(got, (data["A"] + data["B"]) / 2)

    def test_carry_square_matches_inverse_apply(self):
        rng = np.random.default_rng(31)
        n = 10
        labels = np.repeat([0, 1], 5)
        tf = {s: pr.PiecewiseTransform(n, {lab: (np.flatnonzero(labels == lab), _rotation(5, 10 * i + lab))
                                           for lab in (0, 1)}) for i, s in enumerate("AT")}
        x = rng.standard_normal((7, n))
        assert np.allclose(sc.carry(x, tf["A"], tf["T"]), tf["T"].inverse().apply(tf["A"].apply(x)), atol=1e-5)

    def test_carry_nonsquare_target_and_srm_bases(self):
        """The target's columns may be a subset of the template's; SRM bases are columns x k."""
        rng = np.random.default_rng(32)
        n = 8
        x = rng.standard_normal((6, n))
        r_s = _rotation(5, 1)                     # template subject: 5 columns -> 5 template columns
        r_t = _rotation(5, 2)[[0, 2, 3]]          # target: 3 of those columns, rows orthonormal
        src = pr.PiecewiseTransform(n, {"p": (np.arange(5), r_s)})
        tgt = pr.PiecewiseTransform(n, {"p": (np.array([0, 2, 3]), r_t)})
        got = sc.carry(x, src, tgt)
        assert np.allclose(got[:, [0, 2, 3]], x[:, :5] @ r_s @ r_t.T, atol=1e-5)
        assert np.isnan(got[:, [1, 4, 5, 6, 7]]).all()
        w_s, w_t = np.linalg.qr(rng.standard_normal((5, 2)))[0], np.linalg.qr(rng.standard_normal((3, 2)))[0]
        got = sc.carry(x, pr.PiecewiseTransform(n, {"p": (np.arange(5), w_s)}),
                       pr.PiecewiseTransform(n, {"p": (np.array([0, 2, 3]), w_t)}))
        assert np.allclose(got[:, [0, 2, 3]], x[:, :5] @ w_s @ w_t.T, atol=1e-5)

    def test_carry_target_on_a_column_subset_matches_target_cross(self):
        """As stimulus_route.target_cross: a target valid on a subset of the template's columns enters on those
        template coordinates, so R_T is square over its own columns."""
        rng = np.random.default_rng(33)
        n, p, keep = 7, 5, np.array([0, 1, 3, 4])
        r_s = _rotation(p, 3)
        r_t = _rotation(keep.size, 4)
        src = pr.PiecewiseTransform(n, {"p": (np.arange(p), r_s)})
        tgt = pr.PiecewiseTransform(n, {"p": (keep, r_t)})
        x = rng.standard_normal((6, n))
        got = sc.carry(x, src, tgt)
        assert np.allclose(got[:, keep], (x[:, :p] @ r_s)[:, keep] @ r_t.T, atol=1e-5)
        assert np.isnan(got[:, [2, 5, 6]]).all()
        with pytest.raises(ValueError):
            sc.carry(x, src, pr.PiecewiseTransform(n, {"p": (np.array([0, 1, 5, 6]), r_t)}))

    def test_carry_nan_column_spoils_only_its_piece(self):
        n = 6
        tf = pr.PiecewiseTransform(n, {"a": (np.arange(3), np.eye(3)), "b": (np.arange(3, 6), np.eye(3))})
        x = np.ones((4, n))
        x[:, 1] = np.nan
        got = sc.carry(x, tf, tf)
        assert np.isnan(got[:, :3]).all() and np.isfinite(got[:, 3:]).all()

    def test_m1_angle_is_scored_within_hemisphere(self):
        """Each hemisphere maps the contralateral hemifield; a good prediction must score near 1."""
        rng = np.random.default_rng(4)
        n = 400
        hemi = np.repeat(["L", "R"], n // 2)
        ang = np.where(hemi == "L", rng.uniform(100, 260, n), rng.uniform(-80, 80, n))
        ecc = rng.uniform(1, 8, n)
        t = sc.cartesian(ang, ecc)
        p = sc.cartesian(ang + rng.normal(0, 5, n), ecc)
        got = sc.m1_angle(t, p, hemi, np.ones(n, bool), {"vis": np.arange(n)})
        assert got["vis"] > 0.9

    def test_m1_map_pearson_per_network(self):
        rng = np.random.default_rng(5)
        m = rng.standard_normal(30)
        got = sc.m1_map(m, 2 * m + 1, {"a": np.arange(15), "b": np.arange(15, 30)})
        assert np.isclose(got["a"], 1) and np.isclose(got["b"], 1)

    def _cells(self, value_fn, targets=("03", "04", "05"), networks=("n0", "n1", "n2"), films=range(12)):
        return pd.DataFrame([{"target": t, "network": n, "film": f"f{f:02d}", "rank_acc": value_fn(t, n, f)}
                             for t in targets for n in networks for f in films])

    def test_reference_is_the_stronger_baseline_per_target_and_network(self):
        mni = self._cells(lambda t, n, f: 0.6 if n == "n0" else 0.5)
        fs6 = self._cells(lambda t, n, f: 0.55)
        ref = sc.reference_scores({"mni": mni, "fsaverage6": fs6})
        chosen = ref.groupby("network")["baseline"].first().to_dict()
        assert chosen == {"n0": "mni", "n1": "fsaverage6", "n2": "fsaverage6"}

    def test_decision_rule(self):
        rng = np.random.default_rng(6)
        route = self._cells(lambda t, n, f: 0.6 + (0.2 if n == "n1" and t != "05" else 0.0)
                            + 0.01 * rng.standard_normal())
        ref = self._cells(lambda t, n, f: 0.6)
        res = sc.decide(sc.film_gains(route, ref))
        assert res["go"] and res["networks_counted"] == ["n1"]
        assert res["p"]["03"]["n1"] == 1 / 2 ** 12  # every film positive: the exact floor
        worse = sc.decide(sc.film_gains(self._cells(lambda t, n, f: 0.5), ref))
        assert not worse["go"]

    def test_gains_need_matching_cells(self):
        route = self._cells(lambda t, n, f: 0.6)
        with pytest.raises(ValueError):
            sc.film_gains(route, route.iloc[:-1])

    def test_selftest_small(self):
        res = sc.selftest(n_columns=700, piece=50, n_films=3, film_trs=40, n_items=60, noise=1.0,
                          log=lambda *a: None)
        assert res["passed"], res["scores"]


# ---------------------------------------------------------------------------
# scoring driver
# ---------------------------------------------------------------------------

scr = _load("score_route")


class TestScoreRoute:
    def test_face_names(self):
        three = ["cha", "stimulus", "response"]
        assert scr.face_name("cha+stimulus+response", three, "ebind") == "combined"
        assert scr.face_name("cha+response", three, "ebind") == "combined-minus-stimulus"
        assert scr.face_name("stimulus", three, "ebind") == "stimulus-ebind"
        assert scr.face_name("cha", ["cha", "stimulus"], "ebind") == "cha"
        assert scr.face_name("cha+stimulus", ["cha", "stimulus"], "ebind") == "combined"

    def test_mni_network_labels(self):
        vox = pd.DataFrame({"schaefer7n": ["7Networks_LH_Vis_1", "7Networks_RH_Default_PFCm_3", np.nan, ""]})
        assert list(scr.mni_network_labels(vox)) == ["Vis", "Default", "", ""]

    def test_shared_valid_is_the_intersection(self):
        tgt = {"a": np.ones((3, 5)), "b": np.ones((3, 5))}
        tgt["b"][1, 0] = np.nan
        m1 = {k: v.copy() for k, v in tgt.items()}
        m1["a"][:, 2] = np.nan
        m2 = {k: np.ones((3, 5)) for k in tgt}
        m2["b"][:, 4] = np.inf
        ok = scr.shared_valid(tgt, {"m1": m1, "m2": m2})
        assert list(ok) == [False, True, False, True, False]

    def test_srm_objective_prefers_the_true_bases(self):
        rng = np.random.default_rng(40)
        shared = rng.standard_normal((200, 4))
        w = {s: np.linalg.qr(rng.standard_normal((12, 4)))[0] for s in ("A", "B")}
        y = {s: shared @ w[s].T + 0.1 * rng.standard_normal((200, 12)) for s in ("A", "B")}
        cols = np.arange(12)
        good, n = scr.srm_objective({s: {"p": (cols, w[s])} for s in w}, ["A", "B"], y)
        wrong = {s: {"p": (cols, np.linalg.qr(rng.standard_normal((12, 4)))[0])} for s in w}
        bad, _ = scr.srm_objective(wrong, ["A", "B"], y)
        assert n == 12 and good > 0.9 and good > bad

    def test_srm_objective_leaves_out_nonfinite_pieces(self):
        y = {"A": np.ones((5, 4)), "B": np.ones((5, 4))}
        y["A"][0, 3] = np.nan
        w = {s: {"p": (np.arange(2), np.eye(2)), "q": (np.arange(2, 4), np.eye(2))} for s in y}
        _, n = scr.srm_objective(w, ["A", "B"], y)
        assert n == 0  # 'q' left out (NaN); 'p' constant columns give NaN r and drop too

    def test_project_all_identity_and_transform(self):
        rng = np.random.default_rng(41)
        n = 6
        data = {s: {"f": rng.standard_normal((4, n)).astype(np.float32)} for s in ("A", "B", "T")}
        ident = pr.PiecewiseTransform(n, {"p": (np.arange(n), np.eye(n))})
        out = scr.project_all({"anat": None, "id": {s: ident for s in data}}, data, ["A", "B"], "T")
        assert np.allclose(out["anat"]["f"], out["id"]["f"], atol=1e-6)

    def test_identification_foil_pool(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal((30, 40))
        noisy = x + 0.5 * rng.standard_normal(x.shape)
        full_rank, full_top1 = sc.identification(x, noisy)
        same_rank, same_top1 = sc.identification(x, noisy, np.ones(30, bool))
        assert np.array_equal(full_rank, same_rank) and np.array_equal(full_top1, same_top1)
        tpl = noisy.copy()
        tpl[1] = tpl[0] + 0.01 * rng.standard_normal(40)  # row 1 is a near-copy of row 0: a privileged foil
        pool = np.ones(30, bool)
        pool[1] = False
        r_all, _ = sc.identification(x, tpl)
        r_pool, _ = sc.identification(x, tpl, pool)
        assert r_pool[0] >= r_all[0]
        with pytest.raises(ValueError):
            sc.identification(x, noisy, np.zeros(30, bool))

    def test_score_maps_and_items_identity_recovers_shared_maps(self):
        rng = np.random.default_rng(43)
        n = 60
        networks = np.array(["Vis"] * 30 + ["SomMot"] * 30, dtype=object)
        hemi = np.tile(np.repeat(["L", "R"], 15), 2)
        keys = [("floc", "a"), ("floc", "b"), ("motor", "handVsFootDerived"), ("prf", "x"), ("prf", "y")]
        base = rng.standard_normal((len(keys), n))
        base[3:] = np.stack(sc.cartesian(np.where(hemi == "L", 180.0, 0.0) + rng.uniform(-60, 60, n),
                                         rng.uniform(1, 7, n)))
        maps = {s: (base + 0.05 * rng.standard_normal(base.shape)).astype(np.float32) for s in ("A", "B", "T")}
        maps["T"][0, 5] = np.nan
        keep = {s: np.ones(n, bool) for s in maps}
        m1 = scr.score_maps({"anatomical": None}, maps, keys, keep, networks, hemi, ["A", "B"], "T")
        med = m1[(m1["contrast"] == "median")].set_index(["network", "component"])["r"]
        assert (med > 0.95).all()
        ang = m1[m1["component"] == "prf"].set_index("network")["r"]
        assert (ang > 0.9).all()
        assert m1.loc[(m1["contrast"] == "a") & (m1["network"] == "Vis"), "n_columns"].iat[0] == 29
        assert set(m1.loc[m1["read"], "component"]) == {"floc", "motor", "prf"}  # Vis reads all three
        shared = rng.standard_normal((20, n))
        items = {s: (shared + 0.1 * rng.standard_normal((20, n))).astype(np.float32) for s in maps}
        floor = {s: np.ones(n, bool) for s in maps}
        floor["B"][:10] = False
        m3, cols = scr.score_items({"anatomical": None}, items, floor, np.arange(20) % 2 == 0, networks,
                                   ["A", "B"], "T")
        assert cols == {"SomMot": 30, "Vis": 20}
        assert set(m3["foils"]) == {"all", "no_triplet"} and (m3["rank_acc"] > 0.95).all()

    def test_mc_error_uses_per_draw_gains(self):
        rows = []
        for d in range(4):
            for model, v in (("combined", 0.6 + 0.01 * d), ("anatomical", 0.5)):
                rows += [{"target": "03", "draw": d, "model": model, "network": "Vis", "film": f, "rank_acc": v}
                         for f in ("a", "b")]
        level = pd.DataFrame(rows)
        ref = scr.draw_average(level, "anatomical").assign(baseline="fsaverage6")
        se = scr.mc_error(level, "combined", ref)
        want = np.std([0.1, 0.11, 0.12, 0.13], ddof=1) / 2
        assert np.isclose(se["median"], want) and se["n_cells"] == 2
        mni = ref.assign(baseline="mni")
        assert np.isclose(scr.mc_error(level, "combined", mni)["median"], want)

    def test_robustness_signs_per_rejected_cell(self):
        m3 = pd.DataFrame([{"target": t, "draw": d, "model": m, "network": "Vis", "foils": "all",
                            "rank_acc": v} for t in ("03", "04") for d in range(2)
                           for m, v in (("combined", 0.7), ("anatomical", 0.6))])
        m1 = pd.DataFrame([{"target": t, "draw": d, "model": m, "network": "Vis", "component": c,
                            "contrast": "median" if c != "prf" else "angle", "read": True,
                            "r": v - (0.2 if (c == "prf" and t == "04" and m == "combined") else 0)}
                           for t in ("03", "04") for d in range(2) for c in ("floc", "prf")
                           for m, v in (("combined", 0.5), ("anatomical", 0.4))])
        res = {"reject": {"03": {"Vis": True}, "04": {"Vis": True}}, "min_targets": 2}
        out = scr.robustness(res, m3, m1, "combined")
        agree = {c["target"]: c["agrees"] for c in out["cells"]}
        assert agree == {"03": True, "04": False}
        assert out["robust_networks"] == [] and not out["robust"] and not out["all_cells_agree"]
        # Per network: one agreeing target is enough when the go criterion needs only one.
        out = scr.robustness({**res, "min_targets": 1}, m3, m1, "combined")
        assert out["robust_networks"] == ["Vis"] and out["robust"] and not out["all_cells_agree"]


# ---------------------------------------------------------------------------
# secondary families and the smoothed-anatomical control
# ---------------------------------------------------------------------------

fam = _load("families")


class TestFamilies:
    # A 5-vertex path 0-1-2-3-4 plus an isolated column 5 (stands in for subcortex).
    EDGES = np.array([[0, 1], [1, 2], [2, 3], [3, 4]])

    def test_smoother_preserves_constants_and_missing(self):
        sm = fam.Smoother(self.EDGES, 6)
        x = np.full((2, 6), 3.0)
        x[0, 2] = np.nan
        out = sm.run(x, [0, 1, 5])
        assert np.array_equal(out[0], x.astype(np.float32), equal_nan=True)
        for k in (1, 5):
            assert np.isnan(out[k][0, 2])  # missing stays missing
            assert np.allclose(out[k][np.isfinite(out[k])], 3.0)  # missing entries are not averaged in as 0

    def test_smoother_spreads_on_mesh_only(self):
        sm = fam.Smoother(self.EDGES, 6)
        x = np.zeros((1, 6))
        x[0, 2] = 1.0
        x[0, 5] = 7.0
        y = sm.run(x, [1])[1]
        # One step: x <- (x + mean of neighbours) / 2.
        assert np.allclose(y[0, :5], [0.0, 0.25, 0.5, 0.25, 0.0])
        assert y[0, 5] == 7.0  # off the mesh: untouched

    def test_neighbour_r_and_matching(self):
        rng = np.random.default_rng(0)
        sm = fam.Smoother(self.EDGES, 6)
        x = rng.standard_normal((400, 6))
        r = {k: fam.neighbour_r({"f": y}, self.EDGES) for k, y in sm.run(x, [0, 2, 8]).items()}
        assert abs(r[0]) < 0.15 and r[0] < r[2] < r[8]  # smoothing raises neighbour correlation
        assert fam.match_steps(r[2] + 1e-3, r) == 2
        assert fam.match_steps(-1.0, r) == 0

    def test_item_floor_per_label(self):
        base = np.array([True, True, True, True, False])
        meanvol = np.array([100.0, 10.0, 100.0, 100.0, 100.0])
        labels = np.array(["a", "a", "a", "", "a"], dtype=object)
        ok = scr.item_floor(base, meanvol, labels)
        # Label a's median meanvol is 100, so the 10 falls under 0.25 x 100; unlabelled columns keep the base.
        assert list(ok) == [True, False, True, True, False]

    def test_project_all_callable_model(self):
        data = {s: {"f": np.arange(6, dtype=float).reshape(1, 6) * (i + 1)} for i, s in enumerate(["A", "B", "T"])}
        out = scr.project_all({"double": lambda x: 2 * x}, data, ["A", "B"], "T")
        assert np.allclose(out["double"]["f"], 2 * 1.5 * np.arange(6))


summ = _load("summarize")


class TestSummarize:
    def _frame(self, film: bool):
        rows = []
        for t in ("03", "04"):
            for d in range(4):
                for f in (range(3) if film else [None]):
                    for m, v in (("anatomical", 0.6), ("cha", 0.7), ("combined", 0.75)):
                        r = {"scenario": "primary", "pct": 0, "target": t, "draw": d, "network": "Vis", "model": m,
                             "rank_acc": v + (0.01 * (f or 0))}
                        if film:
                            r["film"] = f"f{f}"
                        rows.append(r)
        return pd.DataFrame(rows)

    def test_point_estimates_and_gains(self):
        for film in (True, False):
            out = summ.summarize_metric(self._frame(film), "rank_acc", ["scenario", "pct", "target"], ["network"],
                                        "film" if film else None, n_boot=200, seed=0)
            row = out[(out["target"] == "all") & (out["model"] == "combined")].iloc[0]
            base = 0.75 + (0.01 if film else 0.0)
            assert np.isclose(row["mean"], base)
            assert np.isclose(row["gain_anatomical"], 0.15) and np.isclose(row["gain_cha"], 0.05)
            # A constant gain has a degenerate interval at the gain.
            assert np.isclose(row["gain_anatomical_lo"], 0.15) and np.isclose(row["gain_anatomical_hi"], 0.15)
            assert row["ci_lo"] <= row["mean"] <= row["ci_hi"]
            assert set(out["target"]) == {"03", "04", "all"}
