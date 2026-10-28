"""neuroimaging.nsd: trial mapping, surface cuts, depth subsets and pattern reliability (synthetic)."""

import numpy as np
import pandas as pd
import pytest

from neuroimaging import nsd


def _shared1000(n=nsd.N_SHARED):
    nsd_ids = np.arange(n) * 7 + 3  # ascending, as in shared1000.csv
    return pd.DataFrame({"fileNames": [f"shared{i + 1:04d}_nsd{v:05d}.png" for i, v in enumerate(nsd_ids)],
                         "mmmId": np.arange(1, n + 1), "nsdId": nsd_ids})


class TestSharedTrials:
    def test_maps_trials_to_mmmdata_images(self):
        table = _shared1000()
        subjectim = np.concatenate([table.nsdId.to_numpy(), np.arange(10_000) + 90_000])
        order = np.full(2 * nsd.TRIALS_PER_SESSION, nsd.N_SHARED + 5)
        order[[0, 10, nsd.TRIALS_PER_SESSION + 1]] = [1, 2, 1]
        t = nsd.shared_trials(order, subjectim, table, n_sessions=2)
        assert t[["session", "trial", "shared_idx"]].values.tolist() == [[1, 1, 1], [1, 11, 2], [2, 2, 1]]
        assert t.mmm_id.tolist() == [1, 2, 1]
        assert t.stimulus_id.tolist() == ["shared0001_nsd00003", "shared0002_nsd00010", "shared0001_nsd00003"]
        assert t.rep.tolist() == [1, 1, 2]

    def test_only_requested_sessions(self):
        table = _shared1000()
        order = np.ones(3 * nsd.TRIALS_PER_SESSION, int)
        assert nsd.shared_trials(order, table.nsdId.to_numpy(), table, n_sessions=2).session.max() == 2

    def test_unknown_image_is_an_error(self):
        table = _shared1000()
        subjectim = table.nsdId.to_numpy().copy()
        subjectim[0] = 1
        with pytest.raises(ValueError, match="not in shared1000"):
            nsd.shared_trials(np.ones(nsd.TRIALS_PER_SESSION, int), subjectim, table, n_sessions=1)


class TestSurface:
    def test_cut_takes_the_leading_vertices_of_each_hemisphere(self):
        n = nsd.ICO_VERTICES[5]
        lh, rh = np.arange(n + 4), -np.arange(n + 4)
        out = nsd.cut_hemispheres(lh, rh, 5)
        assert out.shape == (2 * n,) and out[n - 1] == n - 1 and out[n] == 0 and out[-1] == -(n - 1)

    def test_cut_refuses_a_short_hemisphere(self):
        with pytest.raises(ValueError, match="ico6"):
            nsd.cut_hemispheres(np.zeros(10), np.zeros(10), 6)


class TestDepthSubsets:
    def test_white_matter_edge_and_deep(self):
        wm = np.zeros((9, 9, 9), bool)
        wm[1:8, 1:8, 1:8] = True
        d = nsd.depth_subsets(wm, np.zeros_like(wm), np.zeros_like(wm))
        assert d["white_matter_deep"].sum() == 3 ** 3  # 7-wide block eroded twice
        assert d["white_matter_edge"].sum() == 7 ** 3 - 5 ** 3
        assert not (d["white_matter_deep"] & d["white_matter_edge"]).any()

    def test_hippocampal_gaps_from_neighbouring_gray(self):
        hpc = np.zeros((12, 5, 5), bool)
        hpc[1:8, 2, 2] = True
        gray = np.zeros_like(hpc)
        gray[0, 2, 2] = True  # one neighbour at the x=0 end
        gray[1, 2, 2] = True  # inside the hippocampus: never counts as a neighbour
        d = nsd.depth_subsets(np.zeros_like(hpc), hpc, gray)
        xs = lambda k: np.flatnonzero(d[k].any(axis=(1, 2))).tolist()  # noqa: E731
        assert xs("hippocampus_gray_adjacent") == [1]
        assert xs("hippocampus_gray_gap1") == [2, 3, 4, 5, 6, 7]
        assert xs("hippocampus_gray_gap2") == [3, 4, 5, 6, 7]


def _trials(n_images, n_sessions, rng):
    """3 repeats per image, each in a random session (repeats may share one)."""
    rows = [{"stimulus_id": f"img{i:03d}", "rep": r, "session": int(rng.integers(n_sessions))}
            for i in range(n_images) for r in (1, 2, 3)]
    return pd.DataFrame(rows)


class TestPatternReliability:
    def test_image_signal_is_detected(self):
        rng = np.random.default_rng(0)
        t = _trials(120, 8, rng)
        sig = {s: rng.normal(size=60) for s in t.stimulus_id.unique()}
        x = np.stack([sig[s] for s in t.stimulus_id]) + rng.normal(size=(len(t), 60))
        r = nsd.pattern_reliability(x, t, n_perm=200, rng=rng)
        assert r["pattern_discrim"] > 0.2 and r["pattern_discrim_p"] < 0.01
        assert r["ident_top1"] > 3 * r["ident_chance"]
        assert r["rdm_rho"] > 0.1

    def test_shared_session_structure_alone_is_not_image_signal(self):
        """Trials of one session share a pattern; no image signal. Session-matched baselines
        must leave nothing, where a free comparison would not."""
        rng = np.random.default_rng(1)
        t = _trials(150, 6, rng)
        ses = {s: 3 * rng.normal(size=60) for s in t.session.unique()}
        x = np.stack([ses[s] for s in t.session]) + rng.normal(size=(len(t), 60))
        r = nsd.pattern_reliability(x, t, n_perm=200, rng=rng)
        assert abs(r["pattern_discrim"]) < 0.02
        assert r["pattern_discrim_p"] > 0.01
        assert r["ident_top1"] < 2 * r["ident_chance"]

    def test_units_with_nan_are_dropped(self):
        rng = np.random.default_rng(2)
        t = _trials(30, 4, rng)
        x = rng.normal(size=(len(t), 20))
        x[0, 3] = np.nan
        assert nsd.pattern_reliability(x, t, n_perm=10, rng=rng)["n_units"] == 19


def test_sample_units_draws_within_the_roi():
    roi = np.zeros(100, bool)
    roi[10:40] = True
    s = nsd.sample_units(roi, 12, np.random.default_rng(0))
    assert s.sum() == 12 and not (s & ~roi).any()
    assert nsd.sample_units(roi, 99, np.random.default_rng(0)).sum() == 30


def test_beta_paths_by_space(tmp_path):
    p = nsd.NSDPaths(tmp_path / "rel", tmp_path / "betas")
    assert p.beta_file(1, "func1pt8mm", 3).name == "betas_session03.nii.gz"
    assert p.beta_file(2, "fsaverage", 40, "rh").name == "rh.betas_session40.mgh"
    assert "subj02" in str(p.beta_file(2, "fsaverage", 40, "rh"))
    with pytest.raises(ValueError):
        p.beta_file(1, "fsaverage", 1)
    with pytest.raises(ValueError):
        p.beta_file(1, "MNI", 1)
