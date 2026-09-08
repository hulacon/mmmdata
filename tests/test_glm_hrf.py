"""GLMsingle library kernels resolve into nilearn designs; HRFindex grids are checked."""

import numpy as np
import pandas as pd
import pytest

nib = pytest.importorskip("nibabel")
pytest.importorskip("nilearn")
pytest.importorskip("glmsingle")

from neuroimaging.constants import MOTION_6  # noqa: E402
from neuroimaging.glm.config import GlmConfig  # noqa: E402
from neuroimaging.glm.design import build_design_matrix  # noqa: E402
from neuroimaging.glm.hrf import (  # noqa: E402
    LIBRARY_SIZE,
    glmsingle_library,
    hrfindex_to_image,
    library_kernel,
    load_hrfindex,
    resolve_hrf_model,
)
from neuroimaging.glm.models import load_model  # noqa: E402


def test_library_has_twenty_peak_normalised_kernels():
    lib = glmsingle_library()
    assert lib.shape == (LIBRARY_SIZE, 501)
    assert np.allclose(np.abs(lib).max(axis=1), 1.0)


def test_kernel_callable_matches_nilearn_signature_and_support():
    k = library_kernel(0)
    samples = k(1.5, 50)
    assert samples.ndim == 1 and samples.max() == pytest.approx(1.0, abs=1e-6)
    assert samples[0] == 0.0 and samples[-1] == pytest.approx(0.0, abs=1e-3)
    with pytest.raises(ValueError, match="0..19"):
        library_kernel(LIBRARY_SIZE)


def test_resolve_passes_nilearn_names_and_refuses_typos():
    assert resolve_hrf_model("spm") == "spm"
    assert resolve_hrf_model("spm + derivative") == "spm + derivative"
    assert callable(resolve_hrf_model("glmsingle:7"))
    with pytest.raises(ValueError, match="Unknown hrf_model"):
        resolve_hrf_model("spm+derivative")


def _motor_events():
    rows, t = [], 0.0
    for _ in range(2):
        for c in ("hand", "foot", "mouth", "saccade", "rest"):
            rows.append({"onset": t, "duration": 20.0, "trial_type": c})
            t += 20.0
    return pd.DataFrame(rows)


def test_library_design_has_the_same_columns_as_spm_and_a_similar_regressor():
    model = load_model("motor")
    conf = pd.DataFrame(np.zeros((150, 6)), columns=MOTION_6)
    spm = build_design_matrix(_motor_events(), conf, 1.5, 150, model, GlmConfig(hrf_model="spm"))
    # kernel 10 peaks near SPM's (4.3 s vs ~5 s); 0 and 19 are the fastest and slowest
    lib = build_design_matrix(_motor_events(), conf, 1.5, 150, model, GlmConfig(hrf_model="glmsingle:10"))
    assert list(spm.columns) == list(lib.columns)
    assert np.corrcoef(spm["hand"], lib["hand"])[0, 1] > 0.95
    fast = build_design_matrix(_motor_events(), conf, 1.5, 150, model, GlmConfig(hrf_model="glmsingle:0"))
    slow = build_design_matrix(_motor_events(), conf, 1.5, 150, model, GlmConfig(hrf_model="glmsingle:19"))
    assert np.argmax(fast["hand"][:40]) < np.argmax(spm["hand"][:40]) < np.argmax(slow["hand"][:40])


def test_hrfindex_image_refuses_wrong_grid_and_values(tmp_path):
    ref = nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.float32), np.eye(4))
    img = hrfindex_to_image(np.zeros((4, 4, 4), dtype=int), ref)
    assert img.shape == (4, 4, 4)
    with pytest.raises(ValueError, match="does not match"):
        hrfindex_to_image(np.zeros((4, 4, 5), dtype=int), ref)
    with pytest.raises(ValueError, match="0..19"):
        hrfindex_to_image(np.full((4, 4, 4), 20), ref)
    path = tmp_path / "idx.nii.gz"
    img.to_filename(str(path))
    assert load_hrfindex(path, reference=ref).shape == (4, 4, 4)
    other = nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.float32), np.diag([2, 2, 2, 1]))
    with pytest.raises(ValueError, match="different grid"):
        load_hrfindex(path, reference=other)
