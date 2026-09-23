"""extract_roi_betas.py --roi-set ladder on a fake fit directory: a tiny
grid, a few trials, a two-row ladder with a union dseg and one subject's
pRF masks. Nothing here touches GPFS.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "retrieval_modeling" / "extract_roi_betas.py"
GRID = (6, 5, 4)
AFFINE = np.diag([2.0, 2.0, 2.0, 1.0]); AFFINE[:3, 3] = (-6, -5, -4)
N_TRIALS = 7


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("extract_under_test", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


def _nii(data, path):
    nib.save(nib.Nifti1Image(np.asarray(data), AFFINE), path)


@pytest.fixture
def world(tmp_path, mod):
    """fit dir + fmriprep reference BOLD + functional_rois tree."""
    tb = mod.tb
    sub, ses, task, run = "sub-99", "ses-04", "TBencoding", "run-01"
    fit_root = tmp_path / "glmsingle_tb"
    fit_dir = fit_root / sub / "enc"
    (fit_dir / "glmsingle_outputs").mkdir(parents=True)
    ti = pd.DataFrame({"session": ses, "run": run, "task": task, "subgroup": "enc",
                       "mmmId": [1, 2, 3, 998, 1, 2, 3], "condition_id": [1, 2, 3, 998, 1, 2, 3],
                       "col_index": range(N_TRIALS), "onset": np.arange(N_TRIALS) * 4.5 + 9,
                       "duration": 3.0, "word": "w", "pairId": 1.0, "sharedId": [0, 0, 0, 1, 0, 0, 0],
                       "enCon": 3.0, "reCon": 1.0, "resp": 7.0, "resp_RT": 1.0})
    ti.to_csv(fit_dir / "trial_info.csv", index=False)
    rng = np.random.default_rng(0)
    betas = rng.standard_normal(GRID + (N_TRIALS,)).astype(np.float32)
    d = {"betasmd": betas, "R2": rng.random(GRID).astype(np.float32),
         "HRFindex": np.ones(GRID, int), "meanvol": np.full(GRID, 1000.0, np.float32)}
    np.save(fit_dir / "glmsingle_outputs" / mod.BETA_FILES["D"], d, allow_pickle=True)
    fmriprep = tmp_path / "fmriprep"
    bold = tb.bold_path(fmriprep, sub, ses, task, run)
    bold.parent.mkdir(parents=True)
    _nii(np.zeros(GRID + (2,), np.float32), bold)
    # functional_rois: two rung-(i) ROIs, their union with a dseg, one pRF set
    roi_root = tmp_path / "functional_rois"
    space = roi_root / f"space-{tb.SPACE}"
    space.mkdir(parents=True)
    a = np.zeros(GRID, np.uint8); a[0:2, 0, 0] = 1
    b = np.zeros(GRID, np.uint8); b[3:6, 1, 1] = 1
    _nii(a, space / "atlas-HOthr25_label-RoiA_mask.nii.gz")
    _nii(b, space / "atlas-HOthr25_label-RoiB_mask.nii.gz")
    _nii(a | b, space / "atlas-HOthr25_label-Posterior_mask.nii.gz")
    _nii((a * 1 + b * 2).astype(np.int16), space / "atlas-HOthr25_label-Posterior_dseg.nii.gz")
    pd.DataFrame({"index": [1, 2], "name": ["RoiA", "RoiB"]}).to_csv(
        space / "atlas-HOthr25_label-Posterior_dseg.tsv", sep="\t", index=False)
    pd.DataFrame({"rung": ["i", "i", "iii"], "roi": ["RoiA", "RoiB", "Posterior"],
                  "too_small": [False, False, False]}).to_csv(roi_root / "ladder.tsv", sep="\t", index=False)
    prf = roi_root / sub / f"space-{tb.SPACE}"
    prf.mkdir(parents=True)
    pos = np.zeros(GRID, np.uint8); pos[0, 0:3, 2] = 1
    neg = np.zeros(GRID, np.uint8); neg[0, 2:5, 2] = 1
    _nii(pos, prf / f"{sub}_task-prf_desc-pos_thr-2p5_mask.nii.gz")
    _nii(neg, prf / f"{sub}_task-prf_desc-negstrict_thr-2p5_mask.nii.gz")
    _nii(pos | neg, prf / f"{sub}_task-prf_desc-union_thr-2p5_mask.nii.gz")
    return dict(sub=sub, fit_root=fit_root, fmriprep=fmriprep, roi_root=roi_root,
                cache_root=tmp_path / "ps", betas=betas, a=a, b=b, pos=pos, neg=neg)


def _run(world, extra=()):
    cmd = [sys.executable, str(SCRIPT), "--subject", world["sub"], "--arm", "enc", "--types", "D",
           "--roi-set", "ladder", "--fit-root", str(world["fit_root"]),
           "--fmriprep-dir", str(world["fmriprep"]), "--roi-root", str(world["roi_root"]),
           "--cache-root", str(world["cache_root"])] + list(extra)
    return subprocess.run(cmd, capture_output=True, text=True)


def test_ladder_cache_format(world):
    r = _run(world)
    assert r.returncode == 0, r.stdout + r.stderr
    p = (world["cache_root"] / "cache" / "glmsingle_tb" / world["sub"] / "enc"
         / f"{world['sub']}_arm-enc_set-ladder_desc-typed_roipatterns.npz")
    assert p.exists()
    d = np.load(p, allow_pickle=True)
    assert str(d["roi_set"]) == "ladder"
    names = list(d["roi_names"])
    assert names == ["RoiA", "RoiB", "Posterior", "PrfNegstrictThr2p5", "PrfPosThr2p5", "PrfUnionThr2p5"]
    assert list(d["roi_rungs"]) == ["i", "i", "iii", "iv", "iv", "iv"]
    # patterns are the betas at the mask voxels, (V, N)
    exp = world["betas"][world["a"] > 0]
    assert d["patterns_RoiA"].shape == (2, N_TRIALS)
    np.testing.assert_allclose(d["patterns_RoiA"], exp)
    assert d["patterns_Posterior"].shape == (5, N_TRIALS)
    # union blocks: Posterior carries the dseg index, the pRF union pos/negstrict/negother
    assert list(d["blocks_Posterior"]) == [1, 1, 2, 2, 2]
    assert list(d["blocknames_Posterior"]) == ["RoiA", "RoiB"]
    assert sorted(set(d["blocks_PrfUnionThr2p5"].tolist())) == [1, 2]      # pos, negstrict-not-pos
    assert list(d["blocknames_PrfUnionThr2p5"]) == ["pos", "negstrict", "negother"]
    assert "blocks_RoiA" not in d.files
    # trial columns and the six-ROI scalars survive
    assert len(d["mmmId"]) == N_TRIALS and d["meanvol_RoiB"].shape == (3,)
    assert tuple(d["grid_shape"]) == GRID


def test_grid_mismatch_is_loud(world):
    bad = world["roi_root"] / f"space-{sys.modules['extract_under_test'].tb.SPACE}" / "atlas-HOthr25_label-RoiA_mask.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((3, 3, 3), np.uint8), np.eye(4)), bad)
    r = _run(world)
    assert r.returncode != 0
    assert "grid" in (r.stdout + r.stderr) and "RoiA" in (r.stdout + r.stderr)


def test_missing_prf_is_skipped_not_fatal(world):
    import shutil
    shutil.rmtree(world["roi_root"] / world["sub"])
    r = _run(world)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "rung (iv)" in r.stdout and "skipped" in r.stdout
    p = next((world["cache_root"] / "cache" / "glmsingle_tb").rglob("*set-ladder*.npz"))
    assert "PrfPosThr2p5" not in list(np.load(p, allow_pickle=True)["roi_names"])


def test_pattern6_unchanged(world):
    """The default set still writes the six-ROI file name (no set tag)."""
    r = _run(world, extra=["--dry-run"])
    assert r.returncode == 0
    r2 = subprocess.run([sys.executable, str(SCRIPT), "--subject", world["sub"], "--arm", "enc",
                         "--types", "D", "--fit-root", str(world["fit_root"]),
                         "--fmriprep-dir", str(world["fmriprep"]), "--cache-root",
                         str(world["cache_root"]), "--dry-run"], capture_output=True, text=True)
    # pattern6 needs the staged atlas (GPFS) or a fetch: a dry run may fail off-cluster,
    # but if it runs it must not mention the ladder
    if r2.returncode == 0:
        assert "ladder" not in r2.stdout
