"""Tier 1 of the data-quality collection, motion row (registry T1.7): FD, raw and respiration-filtered.

One row per run, with no regime (motion does not depend on what is regressed
out). The driver is ``scripts/data_quality/tier1.py motion``; it writes
``tier1_motion.tsv`` at the tree root, which tier 2 reads for QC-FC. Design
record: mmmdata-agents ``docs/archive/workbench/data-quality/`` (log 2026-09-29).

Definitions:

* **FD** is Power's framewise displacement from fMRIPrep's six rigid-body
  parameters: ``sum |d trans| + RADIUS_MM * sum |d rot|`` with rotations in
  radians, the same formula fMRIPrep writes as ``framewise_displacement``.
  :func:`framewise_displacement` recomputes it so that raw and filtered FD come
  out of one code path; :func:`motion_row` checks the raw one against
  fMRIPrep's column and refuses to continue if they differ.
* **Frames counted.** Frame ``t`` has an FD only if neither ``t`` nor ``t - 1``
  is a non-steady-state volume, so a displacement out of a flagged volume is
  not read as motion. The first frame has none.
* **Filtered FD** (Fair et al. 2020): the six parameters are band-stop filtered
  at the breathing frequency before differencing, which removes respiratory
  pseudomotion (the chest moves the B0 field, so the realignment reports head
  motion that did not happen). At TR 1.5 s the Nyquist frequency is 0.333 Hz,
  inside the adult breathing range, so the band is **folded to its alias**
  first (:func:`aliased_stopband`). A band that straddles Nyquist becomes a
  low-pass at its lowest aliased edge.
* **Which band.** Where the run's own ``recording-respiratory`` is usable
  (duckbrain ``physio_reality.tsv``) and has a breathing peak, the band is
  that peak +- ``HALF_WIDTH_HZ`` (``band_source = run``). Otherwise it is the
  subject's median peak +- ``HALF_WIDTH_HZ`` (``subject``), or the sample's
  median for a subject with no peak at all (``sample``). A median, not an
  envelope of the subject's peaks: an envelope wide enough to cover them
  folds into a low-pass near 0.04 Hz that removes real motion too. Every row
  records its band, its source and why the run's own peak was or was not used
  (``resp_status``).
* **Breathing peak.** Welch PSD of the in-scan stretch of the recording
  (``0 <= t < n_vol * TR`` after ``StartTime``), decimated to about 12.5 Hz,
  60 s segments (one segment on a trace shorter than that, at least
  ``MIN_TRACE_S``). The peak is the maximum inside ``RESP_SEARCH_HZ`` and must
  be interior to it: a maximum on the band's edge means the spectrum is still
  rising (slow drift), not that breathing was found, and the run falls back.
* **Filter.** Order-2 Butterworth, applied forwards and backwards (zero
  phase), over the volumes after the leading non-steady-state ones. A
  non-steady-state volume anywhere but the start is refused.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

TABLE_NAME = "tier1_motion"
SCHEMA_VERSION = "1.0"

MOTION_COLUMNS = ("trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z")
#: Head radius (mm) turning radians into arc length; fMRIPrep's default.
RADIUS_MM = 50.0
FD_THRESHOLDS = (0.2, 0.3, 0.5)
#: Largest |recomputed - fMRIPrep| FD (mm) accepted; fMRIPrep writes 6-7 significant digits.
FD_TOLERANCE_MM = 1e-3

#: Where a breathing peak is looked for: 6-36 breaths per minute.
RESP_SEARCH_HZ = (0.1, 0.6)
#: Half-width of a run's stop band around its peak (Fair 2020's ABCD band, 0.31-0.43 Hz, is +-0.06).
HALF_WIDTH_HZ = 0.06
RESP_TARGET_FS = 12.5
WELCH_SEGMENT_S = 60.0
#: Shortest in-scan trace a peak is read from; the 40-volume math runs give 60 s, i.e. one Welch segment.
MIN_TRACE_S = 30.0
FILTER_ORDER = 2


# ---------------------------------------------------------------------------
# FD
# ---------------------------------------------------------------------------

def framewise_displacement(params: np.ndarray, radius: float = RADIUS_MM) -> np.ndarray:
    """``(N, 6)`` trans (mm) + rot (rad) -> ``(N,)`` Power FD; the first frame is NaN."""
    params = np.asarray(params, dtype=float)
    d = np.abs(np.diff(params, axis=0))
    fd = d[:, :3].sum(axis=1) + radius * d[:, 3:].sum(axis=1)
    return np.r_[np.nan, fd]


def leading_nss(nss: np.ndarray) -> int:
    """Count of leading non-steady-state volumes; refuses any flagged volume after the first steady one."""
    nss = np.asarray(nss, dtype=bool)
    n = int(np.argmin(nss)) if not nss.all() else len(nss)
    if nss[n:].any():
        raise ValueError(f"non-steady-state volumes after volume {n}: {list(np.flatnonzero(nss[n:]) + n)[:5]}")
    return n


def fd_summary(fd: np.ndarray, prefix: str) -> dict:
    """Mean / median / max and fraction over each threshold, over the finite frames."""
    x = fd[np.isfinite(fd)]
    out = {f"{prefix}_n_frames": int(len(x))}
    if len(x) == 0:
        out.update({f"{prefix}_{k}": np.nan for k in ("mean", "median", "max")})
        out.update({f"{prefix}_frac_gt_{t}": np.nan for t in FD_THRESHOLDS})
        return out
    out.update({f"{prefix}_mean": float(x.mean()), f"{prefix}_median": float(np.median(x)),
                f"{prefix}_max": float(x.max())})
    out.update({f"{prefix}_frac_gt_{t}": float((x > t).mean()) for t in FD_THRESHOLDS})
    return out


# ---------------------------------------------------------------------------
# Respiration band
# ---------------------------------------------------------------------------

def alias(f: float, fs: float) -> float:
    """The frequency ``f`` appears at when sampled at ``fs`` (folded into ``[0, fs/2]``)."""
    r = f % fs
    return min(r, fs - r)


def aliased_stopband(lo: float, hi: float, tr: float) -> tuple[str, float, float]:
    """``(kind, lo, hi)`` of the filter that removes ``[lo, hi]`` Hz from a series sampled every ``tr`` s.

    ``kind`` is ``bandstop`` (remove ``[lo, hi]``) or ``lowpass`` (keep below
    ``lo``; ``hi`` is Nyquist). A band crossing Nyquist folds onto
    ``[min(lo, fs - hi), nyquist]``, i.e. a low-pass. Bands wider than the
    sampling rate are refused: they would alias onto everything.
    """
    fs = 1.0 / tr
    nyq = fs / 2
    if not 0 < lo < hi:
        raise ValueError(f"stop band needs 0 < lo < hi, got [{lo}, {hi}]")
    if hi >= fs:
        raise ValueError(f"stop band [{lo:.3f}, {hi:.3f}] Hz reaches the sampling rate {fs:.3f} Hz")
    if lo < nyq <= hi:
        return "lowpass", min(lo, fs - hi), nyq
    a, b = sorted((alias(lo, fs), alias(hi, fs)))
    if b >= nyq - 1e-9:
        return "lowpass", a, nyq
    if a <= 0:
        raise ValueError(f"stop band [{lo:.3f}, {hi:.3f}] Hz aliases onto 0 Hz at TR {tr}")
    return "bandstop", a, b


def breathing_peak(signal: np.ndarray, fs: float) -> float:
    """Frequency (Hz) of the largest Welch PSD peak inside :data:`RESP_SEARCH_HZ`; NaN if it is on the edge."""
    from scipy import signal as ss

    x = np.asarray(signal, dtype=float)
    if not np.isfinite(x).all():
        raise ValueError("respiratory trace has non-finite samples")
    x = x - x.mean()
    q = max(int(fs // RESP_TARGET_FS), 1)
    if q > 1:
        x = ss.decimate(x, q)
    f_s = fs / q
    nper = min(int(WELCH_SEGMENT_S * f_s), len(x))
    freqs, power = ss.welch(x, fs=f_s, nperseg=nper)
    band = (freqs >= RESP_SEARCH_HZ[0]) & (freqs <= RESP_SEARCH_HZ[1])
    if band.sum() < 3:
        raise ValueError(f"trace too short to resolve {RESP_SEARCH_HZ} Hz")
    k = int(np.argmax(power[band]))
    if k in (0, band.sum() - 1):
        return float("nan")  # edge maximum: no breathing peak inside the band
    return float(freqs[band][k])


def in_scan_trace(physio_tsv: Path, n_vol: int, tr: float) -> tuple[np.ndarray, float]:
    """The respiratory column over ``0 <= t < n_vol * tr`` (BIDS physio: headerless TSV + JSON sidecar)."""
    meta = json.loads(Path(str(physio_tsv).replace(".tsv.gz", ".json")).read_text())
    fs = float(meta["SamplingFrequency"])
    start = float(meta.get("StartTime", 0.0))
    columns = list(meta.get("Columns", ["respiratory"]))
    data = pd.read_csv(physio_tsv, sep="\t", header=None, names=columns)
    x = data["respiratory"].to_numpy(float)
    t = start + np.arange(len(x)) / fs
    keep = (t >= 0) & (t < n_vol * tr)
    if keep.sum() < MIN_TRACE_S * fs:
        raise ValueError(f"{physio_tsv}: {keep.sum() / fs:.0f} s of in-scan trace, need {MIN_TRACE_S:.0f} s")
    return x[keep], fs


@dataclasses.dataclass(frozen=True)
class Band:
    lo: float  # un-aliased breathing band, Hz
    hi: float
    source: str  # run | subject | sample
    peak_hz: float = np.nan  # the run's own peak (source == run only)


def fallback_bands(peaks: pd.DataFrame) -> tuple[dict[str, Band], Band]:
    """Per-subject and sample-wide bands: median run peak +- ``HALF_WIDTH_HZ`` (columns ``sub``, ``peak_hz``)."""
    def band(p: pd.Series, source: str) -> Band:
        mid = float(np.median(p.to_numpy(float)))
        return Band(mid - HALF_WIDTH_HZ, mid + HALF_WIDTH_HZ, source)

    peaks = peaks[np.isfinite(peaks["peak_hz"].astype(float))] if not peaks.empty else peaks
    if peaks.empty:
        raise ValueError("no breathing peak in any usable respiratory recording; there is no band to fall back on")
    per_sub = {sub: band(g["peak_hz"], "subject") for sub, g in peaks.groupby("sub")}
    return per_sub, band(peaks["peak_hz"], "sample")


def filter_motion(params: np.ndarray, tr: float, band: Band) -> tuple[np.ndarray, dict]:
    """Zero-phase Butterworth over the six parameters; returns the filtered array and the filter used."""
    from scipy import signal as ss

    kind, lo, hi = aliased_stopband(band.lo, band.hi, tr)
    nyq = 0.5 / tr
    if kind == "lowpass":
        b, a = ss.butter(FILTER_ORDER, lo / nyq, btype="lowpass")
    else:
        b, a = ss.butter(FILTER_ORDER, [lo / nyq, hi / nyq], btype="bandstop")
    out = ss.filtfilt(b, a, np.asarray(params, dtype=float), axis=0)
    return out, {"filter_kind": kind, "filter_lo_hz": lo, "filter_hi_hz": hi}


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------

def motion_row(confounds: pd.DataFrame, tr: float, band: Band) -> dict:
    """Every T1.7 column for one run from its fMRIPrep confounds table."""
    missing = [c for c in (*MOTION_COLUMNS, "framewise_displacement") if c not in confounds]
    if missing:
        raise KeyError(f"confounds table lacks {missing}")
    params = confounds[list(MOTION_COLUMNS)].to_numpy(float)
    if not np.isfinite(params).all():
        raise ValueError("motion parameters have non-finite values")
    nss_cols = [c for c in confounds.columns if c.startswith("non_steady_state_outlier")]
    nss = confounds[nss_cols].to_numpy(float).sum(axis=1) > 0 if nss_cols else np.zeros(len(confounds), bool)
    n_lead = leading_nss(nss)

    raw = framewise_displacement(params)
    theirs = confounds["framewise_displacement"].to_numpy(float)
    both = np.isfinite(raw) & np.isfinite(theirs)
    worst = float(np.max(np.abs(raw[both] - theirs[both]))) if both.any() else 0.0
    if worst > FD_TOLERANCE_MM:
        raise ValueError(f"recomputed FD differs from fMRIPrep's by up to {worst:.4g} mm; the formula is not theirs")

    counted = np.ones(len(raw), bool)
    counted[: n_lead + 1] = False  # NSS volumes and the frame displaced out of the last one
    filt = np.full(len(raw), np.nan)
    fparams, used = filter_motion(params[n_lead:], tr, band)
    filt[n_lead:] = framewise_displacement(fparams)

    row = {
        "n_vol": len(confounds), "n_nss": int(nss.sum()),
        "n_motion_outlier": sum(c.startswith("motion_outlier") for c in confounds.columns),
        "fd_check_max_abs_diff": worst,
        **fd_summary(np.where(counted, raw, np.nan), "fd"),
        **fd_summary(np.where(counted, filt, np.nan), "fdf"),
        "band_source": band.source, "resp_peak_hz": band.peak_hz,
        "band_lo_hz": band.lo, "band_hi_hz": band.hi, **used,
    }
    return row


def physio_index(physio_reality_tsv: Path) -> pd.DataFrame:
    """duckbrain's usability table, respiratory rows only (``relpath, sub, ses, task, run, verdict``)."""
    path = Path(physio_reality_tsv)
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; duckbrain's physio usability table is an input to T1.7")
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    return df[df["recording"] == "respiratory"].reset_index(drop=True)


def band_for(key: tuple, run_peaks: dict[tuple, float], per_sub: dict[str, Band], sample: Band) -> Band:
    """The run's own band if it has a usable peak, else its subject's, else the sample's."""
    p = run_peaks.get(key, np.nan)
    if np.isfinite(p):
        return Band(max(p - HALF_WIDTH_HZ, 1e-3), p + HALF_WIDTH_HZ, "run", p)
    return per_sub.get(key[0], sample)


def run_key(sub: str, ses: str, task: str, run: Optional[str]) -> tuple:
    return (sub, ses, task, run or "")
