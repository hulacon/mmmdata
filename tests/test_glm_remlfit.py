"""The 3dREMLfit wrapper: matrix format offline; numerical equivalence when AFNI is present."""

import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

nib = pytest.importorskip("nibabel")
pytest.importorskip("nilearn")

from neuroimaging.constants import MOTION_6  # noqa: E402
from neuroimaging.glm.config import GlmConfig  # noqa: E402
from neuroimaging.glm.design import build_design_matrix, contrast_vectors  # noqa: E402
from neuroimaging.glm.estimators import (  # noqa: E402
    ENGINES,
    NilearnEstimator,
    RemlfitEstimator,
    get_estimator,
    gltsym_expression,
    t_to_z,
    write_afni_matrix,
)
from neuroimaging.glm.models import load_model  # noqa: E402

TR, N, SHAPE = 1.5, 200, (6, 6, 6)
ACTIVE = (slice(0, 2), slice(0, 2), slice(0, 2))
AFNI_DIR = Path("/packages/afni/24.1.22")


def _afni_available() -> bool:
    if shutil.which("3dREMLfit"):
        return True
    if AFNI_DIR.exists():
        os.environ["PATH"] = f"{AFNI_DIR}{os.pathsep}{os.environ.get('PATH', '')}"
        return shutil.which("3dREMLfit") is not None
    return False


def _events():
    rows, t = [], 0.0
    for _ in range(3):
        for c in ("hand", "foot", "mouth", "saccade", "rest"):
            rows.append({"onset": t, "duration": 20.0, "trial_type": c})
            t += 20.0
    return pd.DataFrame(rows)


def _run(seed, ar=0.0):
    rng = np.random.default_rng(seed)
    model = load_model("motor")
    conf = pd.DataFrame(rng.normal(scale=0.1, size=(N, 6)), columns=MOTION_6)
    dm = build_design_matrix(_events(), conf, TR, N, model, GlmConfig(smoothing_fwhm=None))
    noise = rng.normal(size=SHAPE + (N,))
    for t in range(1, N):
        noise[..., t] += ar * noise[..., t - 1]
    data = 100.0 + noise
    data[ACTIVE] += 2.0 * dm["hand"].to_numpy()[None, None, None, :]
    img = nib.Nifti1Image(data.astype(np.float32), np.eye(4))
    mask = nib.Nifti1Image(np.ones(SHAPE, dtype=np.uint8), np.eye(4))
    return img, mask, dm, contrast_vectors(model, list(dm.columns))


def test_matrix_header_names_columns_and_stimuli(tmp_path):
    _, _, dm, vecs = _run(0)
    labels = write_afni_matrix(tmp_path / "x.xmat.1D", dm, TR, ["hand", "rest"])
    text = (tmp_path / "x.xmat.1D").read_text()
    assert text.startswith("# <matrix") and text.rstrip().endswith("# </matrix>")
    assert f'ColumnLabels = "{" ; ".join(labels)}"' in text
    assert 'StimLabels = "hand ; rest"' in text and f'RowTR = "{TR}"' in text
    body = [ln for ln in text.splitlines() if not ln.startswith("#")]
    assert len(body) == N and len(body[0].split()) == dm.shape[1]
    assert gltsym_expression(vecs["handVsRest"], labels) == "SYM: +1*hand -1*rest"


def test_t_to_z_is_signed_monotone_and_finite():
    z = t_to_z(np.array([-5.0, 0.0, 2.0, 500.0]), 100)
    assert z[0] < 0 < z[2] < z[3] and z[1] == 0 and np.isfinite(z).all()


def test_engine_table_and_noise_model_guards():
    assert set(ENGINES) == {"nilearn-ols", "nilearn-ar1", "remlfit-arma11"}
    img, mask, dm, vecs = _run(1)
    with pytest.raises(ValueError, match="arma11"):
        NilearnEstimator().fit_run(img, dm, vecs, t_r=TR, mask=mask, cfg=GlmConfig(noise_model="arma11"))
    assert get_estimator("remlfit").name == "remlfit"


@pytest.mark.skipif(not _afni_available(), reason="3dREMLfit not on PATH (module load afni/24.1.22)")
def test_remlfit_ols_matches_nilearn_ols_to_numerical_precision():
    img, mask, dm, vecs = _run(2)
    cfg = GlmConfig(smoothing_fwhm=None, noise_model="ols")
    ref = NilearnEstimator().fit_run(img, dm, vecs, t_r=TR, mask=mask, cfg=cfg)["handVsRest"]
    got = RemlfitEstimator(n_threads=1).fit_run(img, dm, vecs, t_r=TR, mask=mask, cfg=cfg)["handVsRest"]
    assert np.abs(got.effect.get_fdata() - ref.effect.get_fdata()).max() < 1e-3
    assert np.abs(got.stat.get_fdata() - ref.stat.get_fdata()).max() < 1e-3
    assert np.abs(got.variance.get_fdata() - ref.variance.get_fdata()).max() < 1e-4
    assert got.dof == ref.dof


@pytest.mark.skipif(not _afni_available(), reason="3dREMLfit not on PATH (module load afni/24.1.22)")
def test_remlfit_arma11_recovers_the_effect_under_autocorrelated_noise():
    img, mask, dm, vecs = _run(3, ar=0.4)
    cfg = GlmConfig(smoothing_fwhm=None, noise_model="arma11")
    est = RemlfitEstimator(n_threads=1).fit_run(img, dm, vecs, t_r=TR, mask=mask, cfg=cfg)
    assert set(est) == set(vecs)
    hand = est["handVsRest"]
    assert hand.effect.get_fdata()[ACTIVE].mean() > 1.0
    assert hand.z.get_fdata()[ACTIVE].min() > 3.0
    assert abs(hand.z.get_fdata()[3:, 3:, 3:].mean()) < 1.5
    assert np.isfinite(hand.variance.get_fdata()).all()
