"""The FSL FILM wrapper: design/contrast formats offline; numerics when FSL is present.

Track A of the glm-strategy pass-2 design (2026-09-11): FILM is the engine
that pools the noise model spatially, and ``--noest`` is the contract arm
that ties it to the pass-1 nilearn table.
"""

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
    FILM_NOISE_MODELS,
    FilmEstimator,
    NilearnEstimator,
    get_estimator,
    write_fsl_contrasts,
    write_fsl_design,
)
from neuroimaging.glm.models import load_model  # noqa: E402

TR, N, SHAPE = 1.5, 200, (6, 6, 6)
ACTIVE = (slice(0, 2), slice(0, 2), slice(0, 2))
FSL_BIN = Path("/packages/fsl/6.0.7.9/fsl/bin")


def _fsl_available() -> bool:
    if shutil.which("film_gls"):
        return True
    if (FSL_BIN / "film_gls").exists():
        os.environ["PATH"] = f"{FSL_BIN}{os.pathsep}{os.environ.get('PATH', '')}"
        return shutil.which("film_gls") is not None
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


def test_design_and_contrast_files_are_in_fsl_format(tmp_path):
    _, _, dm, vecs = _run(0)
    write_fsl_design(tmp_path / "design.mat", dm)
    text = (tmp_path / "design.mat").read_text()
    assert f"/NumWaves\t{dm.shape[1]}" in text and f"/NumPoints\t{N}" in text
    body = [ln for ln in text.splitlines() if ln and not ln.startswith("/")]
    assert len(body) == N and len(body[0].split()) == dm.shape[1]
    # the intercept's peak-to-peak range is 0; PPheights must not be
    heights = [float(x) for x in text.split("/PPheights")[1].splitlines()[0].split()]
    assert len(heights) == dm.shape[1] and min(heights) > 0

    names = write_fsl_contrasts(tmp_path / "design.con", vecs, dm.shape[1])
    con = (tmp_path / "design.con").read_text()
    assert names == list(vecs) and f"/NumContrasts\t{len(vecs)}" in con
    assert f"/ContrastName1\t{names[0]}" in con
    rows = [ln for ln in con.splitlines() if ln and not ln.startswith("/")]
    assert len(rows) == len(vecs)
    np.testing.assert_allclose([float(x) for x in rows[0].split()], vecs[names[0]])
    with pytest.raises(ValueError, match="weights for a"):
        write_fsl_contrasts(tmp_path / "bad.con", {"c": np.ones(3)}, dm.shape[1])


def test_flags_define_each_noise_model():
    est = FilmEstimator(susan_mask_size=5)
    assert est.flags("ols") == ["--noest"]
    assert est.flags("ar1") == ["--ar"]
    assert est.flags("tukey") == []  # FILM's default taper
    assert est.flags("tukey-smoothed") == ["--sa", "--ms=5"]
    assert set(FILM_NOISE_MODELS) == {"ols", "ar1", "tukey", "tukey-smoothed"}
    with pytest.raises(ValueError, match="film takes noise_model"):
        est.flags("arma11")
    # --sa pools only the Tukey path; a pooled AR(1) is not a thing FILM does
    with pytest.raises(ValueError, match="does not pool the --ar path"):
        est.flags("ar1-smoothed")
    assert get_estimator("film").name == "film"


def test_engine_table_carries_the_three_track_a_levels():
    assert ENGINES["film-pervoxel"] == ("film", "ar1")
    assert ENGINES["film-tukey"] == ("film", "tukey")
    assert ENGINES["film-smoothed"] == ("film", "tukey-smoothed")


@pytest.mark.skipif(not _fsl_available(), reason="film_gls not on PATH (module load fsl/6.0.7.9)")
def test_film_ols_matches_nilearn_ols_to_numerical_precision():
    """The contract arm: with the autocorrelation off, FILM is the pass-1 OLS."""
    img, mask, dm, vecs = _run(2)
    ref = NilearnEstimator().fit_run(
        img, dm, vecs, t_r=TR, mask=mask, cfg=GlmConfig(smoothing_fwhm=None, noise_model="ols")
    )["handVsRest"]
    got = FilmEstimator().fit_run(
        img, dm, vecs, t_r=TR, mask=mask, cfg=GlmConfig(smoothing_fwhm=None, noise_model="ols")
    )["handVsRest"]
    assert np.abs(got.effect.get_fdata() - ref.effect.get_fdata()).max() < 1e-3
    assert np.abs(got.stat.get_fdata() - ref.stat.get_fdata()).max() < 1e-3
    assert np.abs(got.variance.get_fdata() - ref.variance.get_fdata()).max() < 1e-4


@pytest.mark.skipif(not _fsl_available(), reason="film_gls not on PATH (module load fsl/6.0.7.9)")
@pytest.mark.parametrize("noise_model", ["ar1", "tukey", "tukey-smoothed"])
def test_film_recovers_the_effect_and_inflates_variance_under_autocorrelated_noise(noise_model):
    img, mask, dm, vecs = _run(3, ar=0.4)
    cfg = GlmConfig(smoothing_fwhm=None, noise_model=noise_model)
    ols = FilmEstimator().fit_run(
        img, dm, vecs, t_r=TR, mask=mask, cfg=GlmConfig(smoothing_fwhm=None, noise_model="ols")
    )["handVsRest"]
    got = FilmEstimator().fit_run(img, dm, vecs, t_r=TR, mask=mask, cfg=cfg)
    assert set(got) == set(vecs)
    hand = got["handVsRest"]
    assert hand.effect.get_fdata()[ACTIVE].mean() > 1.0
    assert hand.z.get_fdata()[ACTIVE].min() > 3.0
    assert np.isfinite(hand.variance.get_fdata()).all() and hand.dof > 0
    # whitening changes the variance, not (much) the effect
    assert np.abs(hand.effect.get_fdata() - ols.effect.get_fdata()).max() < 0.6
    ratio = hand.variance.get_fdata() / ols.variance.get_fdata()
    assert 1.2 < float(np.median(ratio)) < 4.0


@pytest.mark.skipif(not _fsl_available(), reason="film_gls not on PATH (module load fsl/6.0.7.9)")
def test_smoothing_pools_the_variance_estimate_across_voxels():
    """`--sa` is the pooling: same Tukey taper, less variance-map scatter.

    This is the Track A contrast, and it is the Tukey pair rather than the
    AR(1) pair because FILM pools only this path.
    """
    img, mask, dm, vecs = _run(4, ar=0.4)
    fit = lambda nm: FilmEstimator().fit_run(  # noqa: E731
        img, dm, vecs, t_r=TR, mask=mask, cfg=GlmConfig(smoothing_fwhm=None, noise_model=nm)
    )["handVsRest"]
    per, pooled = fit("tukey"), fit("tukey-smoothed")
    assert pooled.variance.get_fdata().std() < per.variance.get_fdata().std()
    # the pooled estimate is the same object, smoothed: means stay close
    assert abs(pooled.variance.get_fdata().mean() / per.variance.get_fdata().mean() - 1) < 0.2


@pytest.mark.skipif(not _fsl_available(), reason="film_gls not on PATH (module load fsl/6.0.7.9)")
def test_out_of_mask_voxels_carry_no_estimate():
    img, mask_img, dm, vecs = _run(5)
    arr = np.ones(SHAPE, dtype=np.uint8)
    arr[4:, :, :] = 0
    mask = nib.Nifti1Image(arr, np.eye(4))
    got = FilmEstimator().fit_run(
        img, dm, vecs, t_r=TR, mask=mask, cfg=GlmConfig(smoothing_fwhm=None, noise_model="ar1")
    )["handVsRest"]
    assert (got.effect.get_fdata()[4:] == 0).all()
    assert np.isnan(got.variance.get_fdata()[4:]).all()
    assert np.isfinite(got.effect.get_fdata()[:4]).all()
