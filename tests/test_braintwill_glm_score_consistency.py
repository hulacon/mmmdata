"""score_consistency.py: Cochran's Q, the scale statistic, the halves and the null draw, on toy inputs."""

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("nilearn")

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "braintwill_glm"
sys.path.insert(0, str(SCRIPTS))
try:
    import score_consistency as sc  # noqa: E402
except ImportError as exc:  # braintwill not importable: adapter refuses loudly
    pytest.skip(f"score_consistency not importable: {exc}", allow_module_level=True)


def test_cochran_matches_hand_computation():
    effect = np.array([[1.0, 0.0], [3.0, 0.0], [2.0, 5.0]])
    variance = np.array([[1.0, 1.0], [1.0, 1.0], [4.0, np.nan]])   # voxel 1, run 2 masked
    fe, fe_var, q, df = sc.cochran(effect, variance)
    w = np.array([1.0, 1.0, 0.25])
    fe0 = (w * effect[:, 0]).sum() / w.sum()
    assert fe[0] == pytest.approx(fe0)
    assert fe_var[0] == pytest.approx(1 / w.sum())
    assert q[0] == pytest.approx((w * (effect[:, 0] - fe0) ** 2).sum())
    assert df.tolist() == [2, 1]
    assert fe[1] == pytest.approx(0.0) and q[1] == pytest.approx(0.0)   # masked run ignored


def test_one_valid_run_is_excluded_not_zero():
    effect = np.array([[1.0], [2.0]])
    variance = np.array([[1.0], [np.nan]])
    _, _, q, df = sc.cochran(effect, variance)
    assert np.isnan(q[0]) and df[0] == 0
    s = sc.summarise_q(q, df)
    assert s == {"voxels": 0, "excluded": 1}


@pytest.mark.parametrize("runs", [2, 6, 25])
def test_rho_estimates_variance_scale_at_any_run_count(runs):
    rng = np.random.default_rng(0)
    n = 40_000
    sd = rng.uniform(0.5, 2.0, size=(runs, n))
    effect = 3.0 + sd * rng.standard_normal((runs, n))
    _, _, q, df = sc.cochran(effect, sd ** 2)
    calibrated = sc.summarise_q(q, df)
    assert calibrated["rho"] == pytest.approx(1.0, abs=0.03)
    assert calibrated["frac_p05"] == pytest.approx(0.05, abs=0.006)
    _, _, q, df = sc.cochran(effect, sd ** 2 / 1.3)          # variances 1.3x too small
    assert sc.summarise_q(q, df)["rho"] == pytest.approx(1.3, abs=0.04)


def test_halves_follow_sorted_position_not_label_order():
    all_runs = ["s_ses-01_run-01", "s_ses-01_run-02", "s_ses-02_run-01", "s_ses-02_run-02"]
    labels = ["s_ses-02_run-01", "s_ses-01_run-01", "s_ses-02_run-02"]   # one run missing, shuffled
    odd, even = sc.halves(labels, all_runs)
    assert odd.tolist() == [True, True, False]
    assert even.tolist() == [False, False, True]


def test_held_out_selects_on_odd_and_scores_on_even():
    rng = np.random.default_rng(1)
    runs, n = 6, 5_000
    truth = np.zeros(n)
    truth[:500] = 4.0
    effect = truth + rng.standard_normal((runs, n))
    variance = np.ones((runs, n))
    odd = np.arange(runs) % 2 == 0
    out = sc.held_out(effect, variance, odd, ~odd, (100, 500))
    assert out["odd_runs"] == 3 and out["even_runs"] == 3
    assert out["top_n"]["100"]["voxels"] == 100
    assert len(out["abs_z_deciles"]) == 10
    assert sum(d["voxels"] for d in out["abs_z_deciles"]) == n
    assert out["top_n"]["500"]["rho"] == pytest.approx(1.0, abs=0.2)   # homogeneous truth: no excess


def test_null_replicates_draw_without_replacement_and_cap_at_pool():
    rng = np.random.default_rng(2)
    pool, draws, n = 5, 3, 2_000
    effect = rng.standard_normal((pool, draws, n)).astype(np.float32)
    variance = np.ones((pool, draws, n), np.float32)
    out = sc.null_replicates(effect, variance, 9, 20, np.random.default_rng(3))
    assert out["runs"] == 5 and out["requested_runs"] == 9
    assert len(out["rho_all"]) == 20
    assert 0.8 < out["rho"]["p50"] < 1.2


def test_load_null_stacks_runs_and_draws(tmp_path):
    for i in range(2):
        np.savez(tmp_path / f"r{i}.npz", **{
            f"block_AvsB_d{d:02d}_{stat}": np.full(4, i * 10 + d, np.float32)
            for d in range(3) for stat in ("effect", "variance")
        }, event_AvsB_d00_effect=np.zeros(4, np.float32), event_AvsB_d00_variance=np.ones(4, np.float32))
    effect, variance = sc.load_null([tmp_path / "r0.npz", tmp_path / "r1.npz"], "block_AvsB")
    assert effect.shape == (2, 3, 4)
    assert effect[1, 2, 0] == 12
    with pytest.raises(SystemExit):
        sc.load_null([tmp_path / "r0.npz"], "block_AvsBaseline")
