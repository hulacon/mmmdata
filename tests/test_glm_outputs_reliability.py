"""Output naming carries Contract A keys; split-half Dice behaves."""

import json

import numpy as np
import pytest

from neuroimaging.glm.outputs import (
    STATS,
    ensure_dataset_description,
    output_dir,
    statmap_name,
    write_run_metadata,
)
from neuroimaging.glm.reliability import dice, split_half_dice, top_n_mask


def test_statmap_name_full_and_pooled_forms():
    assert statmap_name("03", "floc", "MNI152NLin2009cAsym_res-2", "faceVsObject", "z",
                        session="03", run="01") == (
        "sub-03_ses-03_task-floc_run-01_space-MNI152NLin2009cAsym_res-2_"
        "contrast-faceVsObject_stat-z_statmap.nii.gz"
    )
    # Fixed effects over sessions: no ses-, no run-.
    assert statmap_name("sub-03", "floc", "MNI152NLin2009cAsym_res-2", "faceVsObject", "effect") == (
        "sub-03_task-floc_space-MNI152NLin2009cAsym_res-2_contrast-faceVsObject_stat-effect_statmap.nii.gz"
    )


def test_statmap_name_rejects_unknown_stat():
    with pytest.raises(ValueError, match="stat must be one of"):
        statmap_name("03", "floc", "s", "c", "pvalue")
    assert set(STATS) == {"effect", "variance", "t", "z"}


def test_output_dir_layout(tmp_path):
    assert output_dir(tmp_path, "glm_localizer", "sub-03", "ses-30") == (
        tmp_path / "glm_localizer" / "sub-03" / "ses-30" / "func"
    )
    assert output_dir(tmp_path, "glm_localizer", "03") == tmp_path / "glm_localizer" / "sub-03" / "func"


def test_dataset_description_written_once_with_source(tmp_path):
    base = tmp_path / "glm_localizer"
    dd = ensure_dataset_description(base, tmp_path / "fmriprep", "motor", "nilearn")
    desc = json.loads(dd.read_text())
    assert desc["DatasetType"] == "derivative"
    assert desc["SourceDatasets"] == [{"URL": str(tmp_path / "fmriprep")}]
    assert any(g["Name"] == "nilearn" for g in desc["GeneratedBy"])
    dd.write_text("{}")  # a second call must not clobber
    ensure_dataset_description(base, tmp_path / "other", "motor", "nilearn")
    assert dd.read_text() == "{}"


def test_run_metadata_serializes_paths(tmp_path):
    p = write_run_metadata(tmp_path / "x" / "meta.json", {"path": tmp_path, "n": 1})
    assert json.loads(p.read_text())["path"] == str(tmp_path)


def test_top_n_mask_selects_highest_within_mask_and_ignores_nan():
    stat = np.array([[1.0, 5.0, np.nan], [4.0, 2.0, 3.0]])
    m = top_n_mask(stat, 2)
    assert m.tolist() == [[False, True, False], [True, False, False]]
    within = top_n_mask(stat, 2, mask=np.array([[True, False, True], [True, True, True]]))
    assert within.tolist() == [[False, False, False], [True, False, True]]
    assert top_n_mask(stat, 10).sum() == 5  # only finite voxels
    with pytest.raises(ValueError):
        top_n_mask(stat, 0)


def test_dice_identities():
    a = np.array([1, 1, 0, 0], dtype=bool)
    b = np.array([1, 0, 1, 0], dtype=bool)
    assert dice(a, a) == 1.0
    assert dice(a, ~a) == 0.0
    assert dice(a, b) == 0.5
    assert dice(np.zeros(3, bool), np.zeros(3, bool)) == 0.0


def test_split_half_dice_is_one_for_identical_runs_and_low_for_noise():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(8, 8, 8))
    assert split_half_dice([base, base, base, base], n=20) == 1.0
    noisy = [rng.normal(size=(8, 8, 8)) for _ in range(4)]
    assert split_half_dice(noisy, n=20) < 0.5
    with pytest.raises(ValueError):
        split_half_dice([base], n=20)
