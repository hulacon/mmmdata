"""Staged Harvard-Oxford loader in scripts/pattern_similarity/shared.py.

Fixture atlases are tiny synthetic volumes on a fake grid; nothing here
touches GPFS or the network.
"""

import importlib.util
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

SHARED = Path(__file__).resolve().parent.parent / "scripts" / "pattern_similarity" / "shared.py"


@pytest.fixture(scope="module")
def ps():
    spec = importlib.util.spec_from_file_location("ps_shared_under_test", SHARED)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


CORT = {24: "Intracalcarine Cortex", 45: "Heschl's Gyrus (includes H1 and H2)",
        21: "Angular Gyrus", 31: "Precuneous Cortex", 25: "Frontal Medial Cortex"}
SUB = {9: "Left Hippocampus", 19: "Right Hippocampus"}
AFFINE = np.diag([2.0, 2.0, 2.0, 1.0]); AFFINE[:3, 3] = (-6, -6, -6)


def _write(anat: Path, atlas: str, labels: dict, data: np.ndarray):
    stem = f"tpl-MNI152NLin2009cAsym_atlas-{atlas}_res-2_desc-th25_dseg"
    nib.save(nib.Nifti1Image(data.astype(np.uint8), AFFINE), anat / f"{stem}.nii.gz")
    with open(anat / f"{stem}.tsv", "w") as fh:
        fh.write("index\tname\n")
        for i in range(1, max(labels) + 1):
            fh.write(f"{i}\t{labels.get(i, f'label{i}')}\n")


@pytest.fixture
def atlases_dir(tmp_path):
    anat = tmp_path / "tpl-MNI152NLin2009cAsym" / "anat"
    anat.mkdir(parents=True)
    cort = np.zeros((6, 6, 6), dtype=int)
    for k, v in enumerate(CORT):            # one 2-voxel slab per cortical ROI, spanning x<0 and x>=0
        cort[2:4, k, 0] = v
    sub = np.zeros((6, 6, 6), dtype=int)
    sub[1, 1, 1] = 9; sub[4, 1, 1] = 19; sub[4, 2, 1] = 19
    _write(anat, "HOCPA", CORT, cort)
    _write(anat, "HOSPA", SUB, sub)
    return tmp_path


def test_staged_masks_and_affine(ps, atlases_dir):
    masks, affine = ps.load_bilateral_roi_masks(source="staged", atlases_dir=atlases_dir)
    assert set(masks) == set(ps.PATTERN_ROI_NAMES)
    assert np.allclose(affine, AFFINE)
    assert all(m.shape == (6, 6, 6) and m.dtype == bool for m in masks.values())
    assert {r: int(m.sum()) for r, m in masks.items()} == {
        "EVC": 2, "EAC": 2, "Hippocampus": 3, "AG": 2, "Precuneus": 2, "mPFC": 2}


def test_split_hemi_uses_world_x(ps, atlases_dir):
    masks, _ = ps.load_bilateral_roi_masks(split_hemi=True, source="staged", atlases_dir=atlases_dir)
    assert masks[("EVC", "L")].sum() == 1 and masks[("EVC", "R")].sum() == 1
    assert masks[("Hippocampus", "L")].sum() == 1 and masks[("Hippocampus", "R")].sum() == 2


def test_auto_prefers_staged(ps, atlases_dir, capsys):
    masks, _ = ps.load_bilateral_roi_masks(source="auto", atlases_dir=atlases_dir)
    assert masks["EVC"].shape == (6, 6, 6)
    assert "WARNING" not in capsys.readouterr().out


def test_staged_missing_names_path(ps, tmp_path):
    with pytest.raises(FileNotFoundError, match="atlas-HOCPA"):
        ps.load_bilateral_roi_masks(source="staged", atlases_dir=tmp_path)


def test_label_swap_is_loud(ps, atlases_dir):
    tsv = next((atlases_dir / "tpl-MNI152NLin2009cAsym" / "anat").glob("*HOCPA*.tsv"))
    text = tsv.read_text().replace("Intracalcarine Cortex", "Somewhere Else")
    tsv.write_text(text)
    with pytest.raises(ValueError, match="label 24"):
        ps.load_bilateral_roi_masks(source="staged", atlases_dir=atlases_dir)


def test_bad_source_rejected(ps, atlases_dir):
    with pytest.raises(ValueError, match="source must be"):
        ps.load_bilateral_roi_masks(source="cached", atlases_dir=atlases_dir)
