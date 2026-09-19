"""prf_analyzeprf_assemble.py: analyzePRF chunk outputs -> released maps.

Data-free: synthetic chunks with known pixel parameters. The unit conversion
must agree with fit_prf.to_visual (the Python fit's), and chunk coverage of
the mask must be asserted, not silently NaN-filled.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import fit_prf  # noqa: E402
import prf_analyzeprf_assemble as asm  # noqa: E402

RES, FOV = fit_prf.APERTURE_RES, fit_prf.FOV_DEG


def _analyzeprf_outputs(row, col, sigma, gain, expt):
    """What analyzePRF.m computes from its 1-based [row col sigma gain expt]."""
    c = (1 + RES) / 2.0
    ang = np.degrees(np.arctan2(c - row, col - c)) % 360.0
    ecc = np.hypot(c - row, col - c)
    rfsize = np.abs(sigma) / np.sqrt(expt)
    return ang, ecc, rfsize


def _chunk(vxs, row, col, sigma, gain, expt, r2):
    ang, ecc, rfsize = _analyzeprf_outputs(row, col, sigma, gain, expt)
    return {"vxs": np.asarray(vxs, dtype=float), "ang": ang, "ecc": ecc,
            "expt": expt, "rfsize": rfsize, "R2": r2, "gain": gain,
            "numiters": np.full(len(vxs), 40.0),
            "params": np.column_stack([row, col, sigma, gain, expt])}


def test_conversion_matches_fit_prf_to_visual():
    rng = np.random.default_rng(1)
    n = 50
    row = rng.uniform(10, 90, n); col = rng.uniform(10, 90, n)
    sigma = rng.uniform(0.5, 20, n); expt = rng.uniform(0.05, 1.0, n)
    gain = rng.uniform(0.5, 5, n); r2 = rng.uniform(0, 80, n)
    chunk = _chunk(np.arange(1, n + 1), row, col, sigma, gain, expt, r2)
    res, numiters = asm.chunks_to_results([chunk], n, FOV, RES)
    # fit_prf's pixel coordinates are 0-based (x0 = column, y0 = row)
    ang, ecc, size, sigma_deg = fit_prf.to_visual(col - 1, row - 1, sigma, expt, res=RES, fov=FOV)
    dang = (res["angle"] - ang + 180) % 360 - 180
    assert np.abs(dang).max() < 1e-9
    assert np.allclose(res["eccentricity"], ecc)
    assert np.allclose(res["size"], size)
    assert np.allclose(res["sigma"], sigma_deg)
    assert np.allclose(res["exponent"], expt)
    assert np.allclose(res["gain"], gain)
    assert np.allclose(res["R2"], r2)
    assert res["R2"].dtype == np.float32
    assert np.all(numiters == 40)


def test_chunks_stitch_in_any_order():
    n = 30
    a = _chunk(np.arange(11, 31), *[np.full(20, v) for v in (40.0, 60.0, 3.0, 1.0, 0.5)], np.arange(20.0))
    b = _chunk(np.arange(1, 11), *[np.full(10, v) for v in (50.0, 50.0, 2.0, 1.0, 0.5)], np.arange(10.0) + 100)
    res, _ = asm.chunks_to_results([a, b], n, FOV, RES)
    assert np.allclose(res["R2"][:10], np.arange(10) + 100)
    assert np.allclose(res["R2"][10:], np.arange(20))
    assert not np.isnan(res["angle"]).any()


def test_missing_chunk_is_an_error_not_nan():
    a = _chunk(np.arange(1, 11), *[np.full(10, v) for v in (50.0, 50.0, 2.0, 1.0, 0.5)], np.zeros(10))
    with pytest.raises(SystemExit) as e:
        asm.chunks_to_results([a], 25, FOV, RES)
    assert "15 voxels never fitted" in str(e.value)


def test_duplicate_voxels_are_an_error():
    a = _chunk(np.arange(1, 11), *[np.full(10, v) for v in (50.0, 50.0, 2.0, 1.0, 0.5)], np.zeros(10))
    with pytest.raises(SystemExit) as e:
        asm.chunks_to_results([a, a], 10, FOV, RES)
    assert "fitted twice" in str(e.value)


def test_export_paths_are_bids_like_and_shared_by_polarity():
    from prf_analyzeprf_export import export_paths
    p = export_paths("/w/sub-03", "03", "T1w")
    assert p["mat"].name == "sub-03_task-prf_space-T1w_desc-pooled_analyzeprfinput.mat"
    assert p["mask"].name == "sub-03_task-prf_space-T1w_desc-fit_mask.nii.gz"
    assert "prf" not in p["json"].name.replace("task-prf", "").replace("analyzeprf", "")
