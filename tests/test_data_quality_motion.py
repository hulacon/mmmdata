"""data-quality T1.7: FD recomputation, NSS handling, aliased stop bands, breathing peak, filtered FD."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neuroimaging import data_quality_motion as dqm

TR = 1.5


def _confounds(params: np.ndarray, nss: tuple[int, ...] = ()) -> pd.DataFrame:
    df = pd.DataFrame(params, columns=list(dqm.MOTION_COLUMNS))
    df["framewise_displacement"] = dqm.framewise_displacement(params)
    for i, v in enumerate(nss):
        col = np.zeros(len(df))
        col[v] = 1
        df[f"non_steady_state_outlier{i:02d}"] = col
    return df


def _params(n: int = 200, freq: float = 0.0, amp: float = 0.0, seed: int = 0) -> np.ndarray:
    """Slow random-walk head motion plus an optional sinusoid on trans_y (pseudomotion)."""
    rng = np.random.default_rng(seed)
    p = np.cumsum(rng.normal(scale=[0.01] * 3 + [1e-4] * 3, size=(n, 6)), axis=0)
    p[:, 1] += amp * np.sin(2 * np.pi * freq * TR * np.arange(n))
    return p


# ---------------------------------------------------------------------------
# FD
# ---------------------------------------------------------------------------

def test_fd_is_power_formula_with_rotations_as_arc_length():
    p = np.zeros((3, 6))
    p[1] = [0.1, -0.2, 0.0, 0.001, 0.0, -0.002]   # |d trans| = 0.3, |d rot| = 0.003 rad -> 0.15 mm
    p[2] = p[1]
    fd = dqm.framewise_displacement(p)
    assert np.isnan(fd[0])
    assert fd[1] == pytest.approx(0.3 + 50 * 0.003)
    assert fd[2] == 0


def test_motion_row_refuses_an_fd_that_is_not_fmriprep_formula():
    df = _confounds(_params())
    df["framewise_displacement"] *= 1.1
    with pytest.raises(ValueError, match="differs from fMRIPrep"):
        dqm.motion_row(df, TR, dqm.Band(0.2, 0.3, "run", 0.25))


def test_leading_nss_frames_and_the_frame_out_of_them_are_not_counted():
    p = _params(n=100)
    p[:2] += 5.0                                  # two wild leading volumes
    row = dqm.motion_row(_confounds(p, nss=(0, 1)), TR, dqm.Band(0.2, 0.3, "run", 0.25))
    assert row["n_nss"] == 2
    assert row["fd_n_frames"] == 100 - 3          # vols 0, 1 flagged; vol 2 is displaced out of vol 1
    assert row["fd_max"] < 1.0                    # the 5 mm jump never reaches the summary


def test_a_non_steady_state_volume_after_the_start_is_refused():
    with pytest.raises(ValueError, match="after volume"):
        dqm.motion_row(_confounds(_params(), nss=(0, 50)), TR, dqm.Band(0.2, 0.3, "run", 0.25))


# ---------------------------------------------------------------------------
# Stop band at TR 1.5 (fs 0.667 Hz, Nyquist 0.333 Hz)
# ---------------------------------------------------------------------------

def test_band_below_nyquist_is_a_plain_bandstop():
    kind, lo, hi = dqm.aliased_stopband(0.2, 0.3, TR)
    assert kind == "bandstop" and (lo, hi) == pytest.approx((0.2, 0.3))


def test_band_straddling_nyquist_becomes_a_lowpass_at_its_lowest_alias():
    # 0.38 Hz aliases to 0.667 - 0.38 = 0.287; the band's own low edge 0.26 is lower
    kind, lo, hi = dqm.aliased_stopband(0.26, 0.38, TR)
    assert kind == "lowpass" and lo == pytest.approx(0.26) and hi == pytest.approx(1 / 3)
    kind, lo, _ = dqm.aliased_stopband(0.32, 0.44, TR)   # 0.44 -> 0.227 is now the lowest
    assert kind == "lowpass" and lo == pytest.approx(2 / 3 - 0.44)


def test_band_above_nyquist_folds_to_a_bandstop_below_it():
    kind, lo, hi = dqm.aliased_stopband(0.40, 0.50, TR)
    assert kind == "bandstop" and (lo, hi) == pytest.approx((2 / 3 - 0.50, 2 / 3 - 0.40))


def test_band_reaching_the_sampling_rate_is_refused():
    with pytest.raises(ValueError, match="sampling rate"):
        dqm.aliased_stopband(0.3, 0.7, TR)


# ---------------------------------------------------------------------------
# Breathing peak and filtered FD
# ---------------------------------------------------------------------------

def test_breathing_peak_finds_a_planted_rate():
    fs = 125.0
    t = np.arange(0, 300, 1 / fs)
    rng = np.random.default_rng(3)
    trace = np.sin(2 * np.pi * 0.27 * t) + 0.3 * rng.normal(size=t.size) + 0.5 * np.sin(2 * np.pi * 0.02 * t)
    assert dqm.breathing_peak(trace, fs) == pytest.approx(0.27, abs=1 / 60)


@pytest.mark.parametrize("resp_hz", [0.25, 0.40])      # below Nyquist, and aliased to 0.267
def test_filtered_fd_removes_planted_pseudomotion_and_keeps_real_motion(resp_hz):
    clean = _params(n=300, seed=5)
    noisy = _params(n=300, freq=resp_hz, amp=0.08, seed=5)
    band = dqm.Band(resp_hz - dqm.HALF_WIDTH_HZ, resp_hz + dqm.HALF_WIDTH_HZ, "run", resp_hz)
    row_noisy = dqm.motion_row(_confounds(noisy), TR, band)
    row_clean = dqm.motion_row(_confounds(clean), TR, band)
    assert row_noisy["fd_mean"] > 2 * row_clean["fd_mean"]            # the sinusoid inflates raw FD
    assert row_noisy["fdf_mean"] == pytest.approx(row_clean["fdf_mean"], rel=0.15)


def test_band_falls_back_run_then_subject_then_sample():
    peaks = pd.DataFrame({"sub": ["03"] * 5 + ["04"] * 5,
                          "peak_hz": [0.15, 0.16, 0.17, 0.16, 0.15, 0.28, 0.30, 0.31, 0.29, 0.30]})
    per_sub, sample = dqm.fallback_bands(peaks)
    own = dqm.band_for(("03", "01", "rest", ""), {("03", "01", "rest", ""): 0.2}, per_sub, sample)
    assert own.source == "run" and own.peak_hz == 0.2
    sub = dqm.band_for(("04", "02", "rest", ""), {}, per_sub, sample)
    assert sub.source == "subject" and sub.lo > 0.2                   # sub-04's faster breathing
    other = dqm.band_for(("09", "01", "rest", ""), {}, per_sub, sample)
    assert other.source == "sample" and other.lo < sub.lo
    edge = dqm.band_for(("03", "02", "rest", ""), {("03", "02", "rest", ""): np.nan}, per_sub, sample)
    assert edge.source == "subject"                                   # an edge maximum is no peak


def test_fallback_band_is_the_median_peak_not_an_envelope():
    # an envelope around 0.10 .. 0.35 would fold to a ~0.04 Hz low-pass; the median band stays a notch
    peaks = pd.DataFrame({"sub": ["03"] * 5, "peak_hz": [0.10, 0.25, 0.26, 0.27, 0.35]})
    per_sub, _ = dqm.fallback_bands(peaks)
    assert (per_sub["03"].lo, per_sub["03"].hi) == pytest.approx((0.20, 0.32))
    assert dqm.aliased_stopband(per_sub["03"].lo, per_sub["03"].hi, TR)[0] == "bandstop"


def test_breathing_peak_on_the_search_edge_is_nan():
    fs = 125.0
    t = np.arange(0, 300, 1 / fs)
    drift = np.cumsum(np.random.default_rng(4).normal(size=t.size))  # 1/f^2: rises toward 0 Hz
    assert np.isnan(dqm.breathing_peak(drift, fs))
