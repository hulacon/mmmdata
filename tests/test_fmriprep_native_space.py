"""Native (``func``) output space in fMRIPrep discovery and path builders.

fMRIPrep writes ``--output-spaces func`` files with NO ``space-`` entity
(``*_desc-preproc_bold.nii.gz``); template spaces carry ``space-<label>``.
The default label must never pick up the native files and vice versa.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from neuroimaging import fmriprep_layout as fl
from neuroimaging import io as nio
from neuroimaging.constants import DEFAULT_SPACE, NATIVE_SPACE

MNI = DEFAULT_SPACE


def _seed(root: Path, *, native: bool, template: bool) -> Path:
    func = root / "derivatives" / "fmriprep" / "sub-aa" / "ses-30" / "func"
    func.mkdir(parents=True)
    prefix = "sub-aa_ses-30_task-motor_run-01"
    (func / f"{prefix}_desc-confounds_timeseries.tsv").touch()
    (func / f"{prefix}_desc-confounds_timeseries.json").touch()
    if native:
        for tail in ("desc-preproc_bold.nii.gz", "desc-brain_mask.nii.gz",
                     "desc-hmc_boldref.nii.gz", "desc-coreg_boldref.nii.gz"):
            (func / f"{prefix}_{tail}").touch()
    if template:
        for tail in ("desc-preproc_bold.nii.gz", "desc-brain_mask.nii.gz", "boldref.nii.gz"):
            (func / f"{prefix}_space-{MNI}_{tail}").touch()
    return func


def test_space_part_is_empty_only_for_native():
    assert fl.space_part(NATIVE_SPACE) == ""
    assert fl.space_part(MNI) == f"_space-{MNI}"


def test_layout_builders_drop_entity_for_native(tmp_path):
    bold = fl.bold_path(tmp_path, "aa", "30", "motor", "01", space=NATIVE_SPACE)
    mask = fl.mask_path(tmp_path, "aa", "30", "motor", "01", space=NATIVE_SPACE)
    assert bold.name == "sub-aa_ses-30_task-motor_run-01_desc-preproc_bold.nii.gz"
    assert mask.name == "sub-aa_ses-30_task-motor_run-01_desc-brain_mask.nii.gz"
    assert "space-" in fl.bold_path(tmp_path, "aa", "30", "motor", "01").name


def test_native_resolves_spaceless_files_and_coreg_boldref(tmp_path):
    func = _seed(tmp_path, native=True, template=True)
    (run,) = nio.find_fmriprep_runs(space=NATIVE_SPACE, bids_root=tmp_path)
    assert run.space == NATIVE_SPACE
    assert run.bold == func / "sub-aa_ses-30_task-motor_run-01_desc-preproc_bold.nii.gz"
    assert run.mask == func / "sub-aa_ses-30_task-motor_run-01_desc-brain_mask.nii.gz"
    assert run.boldref == func / "sub-aa_ses-30_task-motor_run-01_desc-coreg_boldref.nii.gz"
    assert run.confounds is not None


def test_template_space_still_resolves_its_own_files(tmp_path):
    func = _seed(tmp_path, native=True, template=True)
    (run,) = nio.find_fmriprep_runs(bids_root=tmp_path)  # DEFAULT_SPACE
    assert run.bold == func / f"sub-aa_ses-30_task-motor_run-01_space-{MNI}_desc-preproc_bold.nii.gz"
    assert run.boldref == func / f"sub-aa_ses-30_task-motor_run-01_space-{MNI}_boldref.nii.gz"


def test_each_space_is_blind_to_the_other(tmp_path):
    _seed(tmp_path, native=True, template=False)
    (run,) = nio.find_fmriprep_runs(bids_root=tmp_path)
    assert run.bold is None and run.mask is None and run.boldref is None
    (run,) = nio.find_fmriprep_runs(space=NATIVE_SPACE, bids_root=tmp_path)
    assert run.bold is not None

    other = tmp_path / "other"
    _seed(other, native=False, template=True)
    (run,) = nio.find_fmriprep_runs(space=NATIVE_SPACE, bids_root=other)
    assert run.bold is None and run.boldref is None


@pytest.mark.parametrize("space", [NATIVE_SPACE, MNI])
def test_t1w_is_not_conflated_with_native(tmp_path, space):
    _seed(tmp_path, native=True, template=True)
    (run,) = nio.find_fmriprep_runs(space="T1w", bids_root=tmp_path)
    assert run.bold is None
