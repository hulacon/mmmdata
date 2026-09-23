"""glm_bakeoff.py end to end on the synthetic motor tree: plan, fit, refuse, collect."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

nib = pytest.importorskip("nibabel")
pytest.importorskip("nilearn")

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import glm_bakeoff  # noqa: E402
from neuroimaging.glm import harness  # noqa: E402
from test_glm_cli import SPACE, _seed_run  # noqa: E402


@pytest.fixture
def tree(tmp_path):
    _seed_run(tmp_path, "aa", "30", "01", 0, acompcor=False)
    _seed_run(tmp_path, "aa", "30", "02", 1, acompcor=False)
    return tmp_path


def test_plan_freezes_harness_and_writes_units(tree, capsys):
    rc = glm_bakeoff.main(["--bids-root", str(tree), "plan", "--subjects", "sub-aa", "--models", "motor"])
    assert rc == 0
    base = tree / "derivatives" / "glm_bakeoff"
    assert (base / "harness.json").exists()
    units = (base / "units.txt").read_text().splitlines()
    n = len(harness.HRF_LEVELS) * len(harness.CONFOUND_LEVELS) * len(harness.ENGINE_LEVELS)
    assert len(units) == n and units[0].startswith("sub-aa model-motor_hrf-spm_conf-motion6_engine-")
    assert f"{n} array units" in capsys.readouterr().out


def test_plan_can_slice_the_factorial_to_one_track(tree, capsys):
    """Track A plans a slice: one HRF x confound cell, its own units file."""
    rc = glm_bakeoff.main([
        "--bids-root", str(tree), "--units", "trackA.txt", "plan", "--subjects", "sub-aa",
        "--models", "motor", "--hrfs", "spm", "--confounds", "acompcor",
        "--engines", "film-pervoxel", "film-tukey", "film-smoothed", "--no-standalone",
    ])
    assert rc == 0
    base = tree / "derivatives" / "glm_bakeoff"
    units = (base / "trackA.txt").read_text().splitlines()
    assert len(units) == 3
    assert {u.split("engine-")[1] for u in units} == {"film-pervoxel", "film-tukey", "film-smoothed"}
    assert all("hrf-spm_conf-acompcor" in u for u in units)
    with pytest.raises(SystemExit, match="filters select no cell"):
        glm_bakeoff.main(["--bids-root", str(tree), "plan", "--subjects", "sub-aa",
                          "--models", "motor", "--engines", "nope", "--no-standalone"])


def test_fit_writes_half_maps_and_scores_then_skips_without_force(tree, capsys):
    glm_bakeoff.main(["--bids-root", str(tree), "plan", "--subjects", "aa", "--models", "motor"])
    cell = "model-motor_hrf-spm_conf-motion6_engine-nilearn-ols"
    rc = glm_bakeoff.main(["--bids-root", str(tree), "fit", "--subject", "sub-aa", "--cell", cell])
    assert rc == 0
    d = tree / "derivatives" / "glm_bakeoff" / "sub-aa" / cell
    rec = json.loads((d / "scores.json").read_text())
    s = rec["contrasts"]["handVsRest"]
    assert set(s) >= {"r", "dice@500", "dice@4000", f"dice@z{glm_bakeoff.harness.Z_THRESHOLD}", "n_mask"}
    assert rec["halves"] == [["sub-aa_ses-30_task-motor_run-01"], ["sub-aa_ses-30_task-motor_run-02"]]
    assert rec["config"]["confounds"] == list(rec["config"]["confounds"]) and rec["config"]["acompcor_n"] == 0
    for h in (1, 2):
        z = nib.load(str(d / f"sub-aa_task-motor_space-{SPACE}_half-{h}_contrast-handVsRest_stat-z_statmap.nii.gz"))
        assert z.get_fdata()[0:2, 0:2, 0:2].mean() > 3.0
    assert s["r"] > 0.5  # the planted cluster is in both halves
    # by unit number, and the no-force skip
    assert glm_bakeoff.main(["--bids-root", str(tree), "fit", "--unit", "1"]) == 0
    assert "pass --force" in capsys.readouterr().out


def test_missing_confound_columns_and_missing_hrfindex_are_loud(tree):
    glm_bakeoff.main(["--bids-root", str(tree), "plan", "--subjects", "aa", "--models", "motor"])
    with pytest.raises(KeyError, match="a_comp_cor"):
        glm_bakeoff.main(["--bids-root", str(tree), "fit", "--subject", "aa",
                          "--cell", "model-motor_hrf-spm_conf-acompcor_engine-nilearn-ols"])
    with pytest.raises(SystemExit, match="prep --subject sub-aa"):
        glm_bakeoff.main(["--bids-root", str(tree), "fit", "--subject", "aa",
                          "--cell", "model-motor_hrf-voxelwise_conf-motion6_engine-nilearn-ols"])


def test_spmderiv_cell_and_collect(tree, capsys):
    glm_bakeoff.main(["--bids-root", str(tree), "plan", "--subjects", "aa", "--models", "motor"])
    cell = "model-motor_hrf-spmderiv_conf-motion6_engine-nilearn-ar1"
    assert glm_bakeoff.main(["--bids-root", str(tree), "fit", "--subject", "aa", "--cell", cell]) == 0
    assert glm_bakeoff.main(["--bids-root", str(tree), "collect"]) == 0
    tsv = tree / "derivatives" / "glm_bakeoff" / "scores.tsv"
    lines = tsv.read_text().splitlines()
    assert lines[0].split("\t")[:4] == ["subject", "model", "hrf", "confounds"]
    assert any("spmderiv" in ln and "\tr\t" in ln for ln in lines[1:])
    assert "spmderiv" in capsys.readouterr().out


def test_keep_per_run_writes_run_maps_into_a_separate_out_base(tree):
    """--out-base gets its own frozen harness; --keep-per-run keeps the FE inputs.

    With one run per half (the motor tree) each half map IS its run map, so
    the recombination check is exact equality.
    """
    base = tree / "derivatives" / "glm_bakeoff_pass2"
    pre = ["--bids-root", str(tree), "--out-base", str(base)]
    assert glm_bakeoff.main(pre + ["plan", "--subjects", "aa", "--models", "motor"]) == 0
    assert (base / "harness.json").exists() and not (tree / "derivatives" / "glm_bakeoff").exists()
    cell = "model-motor_hrf-spm_conf-motion6_engine-nilearn-ols"
    assert glm_bakeoff.main(pre + ["fit", "--subject", "aa", "--cell", cell, "--keep-per-run"]) == 0
    d = base / "sub-aa" / cell
    per_run = d / "per-run"
    maps = sorted(p.name for p in per_run.glob("*_statmap.nii.gz"))
    n_contrasts = len(json.loads((d / "scores.json").read_text())["contrasts"])
    assert len(maps) == 2 * n_contrasts * 3  # runs x contrasts x (effect, variance, t)
    run1_effect = f"sub-aa_ses-30_task-motor_run-01_space-{SPACE}_contrast-handVsRest_stat-effect_statmap.nii.gz"
    assert run1_effect in maps
    idx = json.loads((per_run / "per-run.json").read_text())
    assert [(r["run"], r["half"]) for r in idx["runs"]] == [("01", 1), ("02", 2)]
    assert idx["runs"][0]["contrasts"]["handVsRest"]["dof"] > 0
    rec = json.loads((d / "scores.json").read_text())
    assert rec["config"]["per_run_maps"] == "per-run"
    half1 = nib.load(str(d / f"sub-aa_task-motor_space-{SPACE}_half-1_contrast-handVsRest_stat-effect_statmap.nii.gz"))
    run1 = nib.load(str(per_run / run1_effect))
    np.testing.assert_allclose(half1.get_fdata(), run1.get_fdata())
    assert half1.get_data_dtype() == np.float32 and run1.get_data_dtype() == np.float32  # not the mask's uint8
    # the GLMsingle arm has no per-run estimates to keep
    with pytest.raises(SystemExit, match="keep-per-run"):
        glm_bakeoff.main(pre + ["fit", "--subject", "aa", "--cell", "model-motor_hrf-glmsingle_conf-na_engine-na",
                                "--keep-per-run"])
