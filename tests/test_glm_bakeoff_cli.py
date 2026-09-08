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
from test_glm_cli import SPACE, _seed_run  # noqa: E402


@pytest.fixture
def tree(tmp_path):
    _seed_run(tmp_path, "aa", "30", "01", 0)
    _seed_run(tmp_path, "aa", "30", "02", 1)
    return tmp_path


def test_plan_freezes_harness_and_writes_units(tree, capsys):
    rc = glm_bakeoff.main(["--bids-root", str(tree), "plan", "--subjects", "sub-aa", "--models", "motor"])
    assert rc == 0
    base = tree / "derivatives" / "glm_bakeoff"
    assert (base / "harness.json").exists()
    units = (base / "units.txt").read_text().splitlines()
    assert len(units) == 27 and units[0].startswith("sub-aa model-motor_hrf-spm_conf-motion6_engine-")
    assert "27 array units" in capsys.readouterr().out


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
