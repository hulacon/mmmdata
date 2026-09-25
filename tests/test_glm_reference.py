"""The frozen reference spec: it resolves every GlmConfig field, refuses edits, and may only grow regimes."""

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from neuroimaging.constants import MOTION_6
from neuroimaging.glm import reference
from neuroimaging.glm.config import GlmConfig
from neuroimaging.glm.reference import (
    CONSUMER_FIELDS,
    REGIME_FIELDS,
    load_reference_spec,
    reference_config,
    spec_digest,
)
from neuroimaging.io import FmriprepRun, check_native_pool, mask_intersection


def test_spec_matches_its_digest():
    spec = load_reference_spec()
    assert spec["sha256"] == spec_digest(spec)


def test_every_glmconfig_field_is_decided_by_the_spec():
    # A new GlmConfig field fails here until the spec says what the reference does with it.
    spec = load_reference_spec()
    fields = {f.name for f in dataclasses.fields(GlmConfig)}
    decided = set(spec["config"]) | set(REGIME_FIELDS) | set(CONSUMER_FIELDS)
    assert fields == decided


def test_reference_regime_is_pinned():
    regime = load_reference_spec()["confound_regimes"]["reference"]
    assert regime["status"] == "frozen"
    assert regime["confounds"] == list(MOTION_6) and regime["acompcor_n"] == 6


def test_reference_config_is_ols_spm_unsmoothed_with_spikes():
    cfg = reference_config()
    assert (cfg.noise_model, cfg.hrf_model, cfg.smoothing_fwhm) == ("ols", "spm", None)
    assert cfg.include_non_steady_state and cfg.include_cosine and cfg.drift_model is None
    assert cfg.confounds == tuple(MOTION_6) and cfg.acompcor_n == 6
    assert reference_config(calibrated=True).noise_model == "ar1"
    assert reference_config(output_tree="elsewhere").output_tree == "elsewhere"


def test_unknown_regime_is_named():
    with pytest.raises(KeyError, match="reference"):
        reference_config("nope")


def _copy(tmp_path, mutate):
    spec = json.loads(reference.SPEC_PATH.read_text())
    mutate(spec)
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))
    return path


def test_edited_spec_is_refused(tmp_path):
    path = _copy(tmp_path, lambda s: s["config"].update(smoothing_fwhm=5.0))
    with pytest.raises(RuntimeError, match="edited"):
        load_reference_spec(path)


def test_new_regime_is_an_extension_not_an_edit(tmp_path):
    path = _copy(tmp_path, lambda s: s["confound_regimes"].update(
        extra={"status": "provisional", "confounds": [], "acompcor_n": 0}))
    assert "extra" in load_reference_spec(path)["confound_regimes"]


def _mask_run(tmp_path, name, arr, affine=np.eye(4)):
    nib = pytest.importorskip("nibabel")
    path = tmp_path / f"{name}_mask.nii.gz"
    nib.Nifti1Image(arr.astype(np.uint8), affine).to_filename(str(path))
    return FmriprepRun(subject="aa", session="01", task="t", run=name, space="x", variant="fmriprep", mask=path)


def test_mask_intersection_keeps_voxels_inside_every_run(tmp_path):
    a = np.ones((3, 3, 3)); b = np.ones((3, 3, 3)); b[0, 0, 0] = 0
    _, inter = mask_intersection([_mask_run(tmp_path, "01", a), _mask_run(tmp_path, "02", b)])
    assert inter.sum() == 26 and not inter[0, 0, 0]


def test_mask_intersection_refuses_mixed_grids(tmp_path):
    a = _mask_run(tmp_path, "01", np.ones((3, 3, 3)))
    b = _mask_run(tmp_path, "02", np.ones((3, 3, 3)), affine=np.diag([2.0, 2.0, 2.0, 1.0]))
    with pytest.raises(ValueError, match="grid differs"):
        mask_intersection([a, b])


def _native_run(tmp_path, ses, run, translation, shape=(10, 10, 10), zoom=2.0):
    """A func-space run with a brain mask and a boldref->T1w ITK coreg transform."""
    nib = pytest.importorskip("nibabel")
    d = tmp_path / f"ses-{ses}"
    d.mkdir(exist_ok=True)
    prefix = f"sub-aa_ses-{ses}_task-t_run-{run}"
    mask = d / f"{prefix}_desc-brain_mask.nii.gz"
    nib.Nifti1Image(np.ones(shape, dtype=np.uint8), np.diag([zoom, zoom, zoom, 1.0])).to_filename(str(mask))
    conf = d / f"{prefix}_desc-confounds_timeseries.tsv"
    conf.write_text("x\n0\n")
    t = " ".join(str(v) for v in translation)
    (d / f"{prefix}_from-boldref_to-T1w_mode-image_desc-coreg_xfm.txt").write_text(
        "#Insight Transform File V1.0\n#Transform 0\nTransform: AffineTransform_float_3_3\n"
        f"Parameters: 1 0 0 0 1 0 0 0 1 {t}\nFixedParameters: 0 0 0\n")
    return FmriprepRun(subject="aa", session=ses, task="t", run=run, space="func", variant="fmriprep",
                       mask=mask, confounds=conf)


def test_native_pool_within_a_session_passes_under_half_a_voxel(tmp_path):
    runs = [_native_run(tmp_path, "02", "01", (0, 0, 0)), _native_run(tmp_path, "02", "02", (0.3, 0, 0))]
    assert check_native_pool(runs) == pytest.approx(0.3)
    _, inter = mask_intersection(runs)
    assert inter.all()


def test_native_pool_refuses_head_movement_between_runs(tmp_path):
    # Same grid, same session, but the second run's anatomy sits 1.5 mm away (> 1 mm = half a 2 mm voxel).
    runs = [_native_run(tmp_path, "02", "01", (0, 0, 0)), _native_run(tmp_path, "02", "02", (0, 1.5, 0))]
    with pytest.raises(ValueError, match="head moved"):
        mask_intersection(runs)


def test_native_pool_refuses_sessions_even_on_an_identical_grid(tmp_path):
    runs = [_native_run(tmp_path, "02", "01", (0, 0, 0)), _native_run(tmp_path, "03", "01", (0, 0, 0))]
    with pytest.raises(ValueError, match="span sessions"):
        mask_intersection(runs)


def test_template_space_pools_across_sessions_without_the_native_check(tmp_path):
    a = _native_run(tmp_path, "02", "01", (0, 0, 0))
    b = _native_run(tmp_path, "03", "01", (0, 9, 0))
    a, b = (dataclasses.replace(r, space="T1w") for r in (a, b))
    _, inter = mask_intersection([a, b])
    assert inter.all()
