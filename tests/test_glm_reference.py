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
from neuroimaging.io import FmriprepRun, mask_intersection


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
