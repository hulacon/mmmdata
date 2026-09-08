"""The frozen harness: cells enumerate, the spec freezes, scores behave."""

import json

import numpy as np
import pytest

from neuroimaging.glm import harness


def test_factorial_enumerates_27_cells_per_model_plus_standalone_arms():
    cells = harness.factorial_cells()
    fact = [c for c in cells if not c.standalone]
    assert len(fact) == 3 * 3 * 3 * 3
    standalone = [c for c in cells if c.standalone]
    assert {(c.model, c.hrf) for c in standalone} == {("floc", "glmsingle"), ("tbrepetition", "glmsingle-betas")}
    assert len({c.id for c in cells}) == len(cells)
    c = harness.Cell.parse("model-floc_hrf-spm_conf-motion6_engine-nilearn-ar1")
    assert c == harness.Cell("floc", "spm", "motion6", "nilearn-ar1") and c.id.endswith("engine-nilearn-ar1")
    with pytest.raises(ValueError, match="expected model"):
        harness.Cell.parse("floc")


def test_freeze_once_and_refuse_drift(tmp_path):
    path = harness.freeze(tmp_path)
    spec = json.loads(path.read_text())
    assert spec["n_sets"]["floc"] == list(harness.N_SETS["floc"]) and "sha256" in spec
    assert harness.freeze(tmp_path) == path  # idempotent
    spec["z_threshold"] = 2.3
    path.write_text(json.dumps(spec))
    with pytest.raises(RuntimeError, match="edited after freezing"):
        harness.check_frozen(tmp_path)
    spec["sha256"] = harness.spec_digest({k: v for k, v in spec.items() if k != "sha256"})
    path.write_text(json.dumps(spec))  # internally consistent, but not what the code says
    with pytest.raises(RuntimeError, match="differs from the frozen.*z_threshold"):
        harness.check_frozen(tmp_path)
    with pytest.raises(FileNotFoundError, match="plan"):
        harness.check_frozen(tmp_path / "nowhere")


def test_split_runs_alternates():
    assert harness.split_runs(6) == ([0, 2, 4], [1, 3, 5])
    assert harness.split_runs(2) == ([0], [1])
    with pytest.raises(ValueError):
        harness.split_runs(1)


def test_scores_are_one_for_identical_halves_and_low_for_noise():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(10, 10, 10))
    mask = np.ones(a.shape, dtype=bool)
    mask[0] = False
    same = harness.score_halves(a, a, mask, (50, 100))
    assert same["r"] == pytest.approx(1.0) and same["dice@50"] == 1.0 and same["dice@100"] == 1.0
    assert same["n_mask"] == 900 and same["n_valid"] == 900
    b = rng.normal(size=a.shape)
    noise = harness.score_halves(a, b, mask, (50,))
    assert abs(noise["r"]) < 0.2 and noise["dice@50"] < 0.3
    thr = harness.score_halves(a + 5, a + 5, mask, (50,), z_half1=a + 5, z_half2=a + 5)
    assert thr[f"dice@z{harness.Z_THRESHOLD}"] == 1.0 and thr[f"n@z{harness.Z_THRESHOLD}"][0] > 800


def test_collect_reads_every_cell(tmp_path):
    cell = harness.Cell("motor", "spm", "motion6", "nilearn-ols")
    d = harness.cell_dir(tmp_path, "aa", cell)
    harness.write_scores(d / "scores.json", {"cell": cell.id, "subject": "aa",
                                             "contrasts": {"handVsRest": {"r": 0.5, "dice@500": 0.4, "n_mask": 9}}})
    rows = harness.collect_scores(tmp_path)
    assert {(r["metric"], r["value"]) for r in rows} == {("r", 0.5), ("dice@500", 0.4)}
    assert rows[0]["engine"] == "nilearn-ols" and rows[0]["model"] == "motor"
