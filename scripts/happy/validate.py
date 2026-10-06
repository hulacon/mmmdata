#!/usr/bin/env python
"""Score happy's BOLD-derived cardiac waveform against the pulse recording.

For each run, reads:
  - the raw BOLD, its sidecar (TR, SliceTiming) and header;
  - the run's ``recording-pulse`` physio (BIDS tsv.gz + json; StartTime is the
    time of the first sample relative to volume 0);
  - happy's ``pure``-mode output (cardiac estimated from BOLD alone):
    ``<root>_desc-stdrescardfromfmri_timeseries.{tsv.gz,json}`` and
    ``<root>_desc-runinfo.json``;
  - with ``--fmriprep``: the run's confounds TSV and native-space brain mask.

Both waveforms go through the same path: band-pass, peak detection, cardiac
phase by linear interpolation between peaks (RETROICOR, Glover et al. 2000),
then 2nd-order Fourier regressors sin/cos(phi), sin/cos(2 phi).

Waveform agreement (regressors sampled once per volume at its median slice time):
  hr_pulse, hr_happy      mean heart rate (bpm) from peak counts over the run
  hr_happy_spectral       happy's own estimate (runinfo, for cross-checking)
  dhr                     hr_happy - hr_pulse
  r_zero                  mean Pearson r over the 4 regressor pairs, zero lag
  r_span                  mean multiple R of each pulse regressor on all 4
                          happy regressors. A constant pulse-transit delay
                          is a constant phase rotation within each sin/cos
                          pair, so r_span is blind to it; r_zero is not.
  lag_best, r_best        lag (s, happy shifted) maximising r_zero in +-1 s
  beat_offset_ms          median offset, pulse beat -> nearest happy beat
  beat_jitter_ms          robust SD of that offset about its median
  phase_jitter_rad        the same jitter as cardiac phase (2 pi jitter / IBI)
  happy_corr_raw2pleth    happy's self-reported score from a `ref` run, if any

BOLD variance (``--fmriprep``): the raw BOLD (not slice-time corrected, so each
slice gets regressors at its own acquisition times) is fitted voxelwise in the
brain mask with a nuisance model (intercept, DCT drift at 1/128 Hz, the 24
fMRIPrep motion columns) plus the 4 cardiac regressors from one source. The
gain in R^2 over the nuisance model is dR2. Four sources:
  pulse   regressors from the recording (peaks)
  happy   regressors from happy's waveform, by the same peak picking
  hphase  regressors from happy's own cardiac phase (``instphase_unwrapped``,
          slice resolution): no peak picking
  null    the pulse regressors evaluated on a time-reversed axis: same beat
          statistics, wrong alignment, so its dR2 is what 4 cardiac-like
          regressors buy by chance
Timepoints where any regressor or confound is undefined (before the first beat,
after the last, the first row of the motion derivatives) are dropped from all
fits of that slice, never filled.
  dr2_<src>_mean           mean dR2 over mask voxels
  dr2_<src>_top            mean dR2 over the top 5 % of voxels ranked by the mean
                           dR2 of pulse, happy and hphase (selection symmetric
                           in the recording and the BOLD-derived sources)
  recovery_<src>_{mean,top}  (src - null) / (pulse - null) on those means: the
                           share of the pulse's above-chance cardiac variance
                           that src recovers (src = happy, hphase)
  dr2_map_r_<src>          spatial correlation of the pulse and src dR2 maps
  r_span_hphase            r_span with happy's own phase in place of its peaks

Usage:
  validate.py --bids <bids_root> --happy <out_dir>/pure --runs runs.txt --out table.tsv
              [--ref <out_dir>/ref] [--fmriprep <fmriprep_dir>] [--index N]
  validate.py --collect <rows_dir> --out table.tsv

``--index N`` scores only line N (1-based) of runs.txt, for SLURM arrays; point
``--out`` at a per-run file and ``--collect`` the directory afterwards.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import signal

WORK_FS = 50.0           # common rate for filtering and peak detection (Hz)
BAND = (0.6, 3.0)        # cardiac band (Hz): 36-180 bpm
MIN_IBI = 1.0 / BAND[1]  # shortest inter-beat interval accepted (s)
LAGS = np.arange(-1.0, 1.0001, 0.02)
DRIFT_CUTOFF = 128.0     # s; DCT high-pass period
TOP_FRACTION = 0.05
PhaseFn = Callable[[np.ndarray], np.ndarray]
MOTION = [f"{k}_{a}{s}" for k in ("trans", "rot") for a in "xyz"
          for s in ("", "_derivative1", "_power2", "_derivative1_power2")]


def read_bids_physio(tsv: Path, column: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (time, values) for one column of a BIDS physio/timeseries file."""
    meta = json.loads(tsv.with_name(tsv.name.replace(".tsv.gz", ".json")).read_text())
    cols = meta["Columns"]
    if column not in cols:
        raise KeyError(f"{column!r} not in {tsv.name} columns {cols}")
    data = pd.read_csv(tsv, sep="\t", header=None, names=cols, compression="gzip")
    y = data[column].to_numpy(dtype=float)
    t = float(meta["StartTime"]) + np.arange(len(y)) / float(meta["SamplingFrequency"])
    return t, y


def to_work_rate(t: np.ndarray, y: np.ndarray, t0: float, t1: float) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate onto a WORK_FS grid over [t0, t1]; refuse if not covered."""
    if t[0] > t0 + 0.5 or t[-1] < t1 - 0.5:
        raise ValueError(f"waveform covers {t[0]:.2f}..{t[-1]:.2f} s, need {t0:.2f}..{t1:.2f}")
    grid = np.arange(t0, t1, 1.0 / WORK_FS)
    good = np.isfinite(y)
    return grid, np.interp(grid, t[good], y[good])


def bandpass(y: np.ndarray) -> np.ndarray:
    sos = signal.butter(2, BAND, btype="bandpass", fs=WORK_FS, output="sos")
    return signal.sosfiltfilt(sos, y - y.mean())


def peaks(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Beat times: maxima of the band-passed waveform."""
    z = (y - y.mean()) / y.std()
    idx, _ = signal.find_peaks(z, distance=int(MIN_IBI * WORK_FS), prominence=0.5)
    return t[idx]


def mean_hr(beats: np.ndarray) -> float:
    return 60.0 * (len(beats) - 1) / (beats[-1] - beats[0])


def phase_at(beats: np.ndarray, tq: np.ndarray) -> np.ndarray:
    """RETROICOR cardiac phase in [0, 2 pi); NaN outside the first/last beat."""
    k = np.searchsorted(beats, tq, side="right") - 1
    ok = (k >= 0) & (k < len(beats) - 1)
    ph = np.full(tq.shape, np.nan)
    kk = k[ok]
    ph[ok] = 2 * np.pi * (tq[ok] - beats[kk]) / (beats[kk + 1] - beats[kk])
    return ph


def regressors(phase: np.ndarray) -> np.ndarray:
    return np.column_stack([np.sin(phase), np.cos(phase), np.sin(2 * phase), np.cos(2 * phase)])


def r_zero(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.all(np.isfinite(a), 1) & np.all(np.isfinite(b), 1)
    return float(np.mean([np.corrcoef(a[ok, j], b[ok, j])[0, 1] for j in range(a.shape[1])]))


def r_span(ref: np.ndarray, est: np.ndarray) -> float:
    ok = np.all(np.isfinite(ref), 1) & np.all(np.isfinite(est), 1)
    X = np.column_stack([np.ones(ok.sum()), est[ok]])
    out = []
    for j in range(ref.shape[1]):
        y = ref[ok, j]
        resid = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
        out.append(np.sqrt(max(0.0, 1 - resid.var() / y.var())))
    return float(np.mean(out))


def dct_basis(n: int, tr: float) -> np.ndarray:
    """Cosine drift regressors with period >= DRIFT_CUTOFF (SPM convention)."""
    k = int(np.floor(2 * n * tr / DRIFT_CUTOFF))
    t = np.arange(n)
    return np.column_stack([np.cos(np.pi * (t + 0.5) * j / n) for j in range(1, k + 1)]) if k else np.empty((n, 0))


def bold_variance(bold: Path, slice_times: np.ndarray, tr: float, phases: dict[str, PhaseFn],
                  confounds: Path, mask: Path) -> dict:
    """Voxelwise dR2 of each source's cardiac regressors over a nuisance model.

    ``phases`` maps source name -> phase function of time; it must hold
    "pulse" and "null", and every other entry is scored against them.
    """
    img = nib.load(bold)
    m = np.asarray(nib.load(mask).dataobj) > 0
    if m.shape != img.shape[:3]:
        raise ValueError(f"mask {m.shape} does not match BOLD {img.shape[:3]}")
    data = img.get_fdata(dtype=np.float32)  # one read of the 4-D gz (never slab-read)
    n = data.shape[3]
    conf = pd.read_csv(confounds, sep="\t")[MOTION].to_numpy(dtype=float)
    if len(conf) != n:
        raise ValueError(f"confounds have {len(conf)} rows, BOLD has {n} volumes")
    nuis = np.column_stack([dct_basis(n, tr), conf])

    dr2 = {k: [] for k in phases}
    for z in range(data.shape[2]):
        if not m[:, :, z].any():
            continue
        t = np.arange(n) * tr + slice_times[z]
        reg = {k: regressors(f(t)) for k, f in phases.items()}
        ok = np.all(np.isfinite(nuis), 1)
        for r in reg.values():
            ok &= np.all(np.isfinite(r), 1)
        Y = data[:, :, z, :][m[:, :, z]].T[ok].astype(np.float64)  # (time, voxels)
        N = nuis[ok]
        N = np.column_stack([np.ones(len(N)), N[:, N.std(0) > 0]])  # one intercept, no constant columns
        Q, _ = np.linalg.qr(N)
        Yr = Y - Q @ (Q.T @ Y)
        tss = ((Y - Y.mean(0)) ** 2).sum(0)
        for key, r in reg.items():
            S = r[ok] - Q @ (Q.T @ r[ok])
            Qs, _ = np.linalg.qr(S)
            gain = ((Qs.T @ Yr) ** 2).sum(0)
            with np.errstate(divide="ignore", invalid="ignore"):
                dr2[key].append(np.where(tss > 0, gain / tss, np.nan))
    maps = {k: np.concatenate(v) for k, v in dr2.items()}
    good = np.all([np.isfinite(v) for v in maps.values()], 0)
    maps = {k: v[good] for k, v in maps.items()}
    candidates = [k for k in maps if k not in ("pulse", "null")]
    # Voxel selection symmetric in the recording and every BOLD-derived source.
    rank = np.mean([maps[k] for k in ["pulse", *candidates]], 0)
    top = rank >= np.quantile(rank, 1 - TOP_FRACTION)

    row = {"mask_voxels": int(good.sum())}
    for k, v in maps.items():
        row[f"dr2_{k}_mean"] = float(v.mean())
        row[f"dr2_{k}_top"] = float(v[top].mean())
    for k in candidates:
        for sel in ("mean", "top"):
            p, h, o = (row[f"dr2_{s}_{sel}"] for s in ("pulse", k, "null"))
            row[f"recovery_{k}_{sel}"] = (h - o) / (p - o) if p > o else np.nan
        row[f"dr2_map_r_{k}"] = float(np.corrcoef(maps["pulse"], maps[k])[0, 1])
    return row


def score_run(bids: Path, rel: str, happy_dir: Path, ref_dir: Path | None, column: str,
              fmriprep: Path | None) -> dict:
    bold = bids / rel
    base = bold.name.removesuffix("_bold.nii.gz")
    subses = Path(rel).parent.parent
    side = json.loads(bold.with_name(base + "_bold.json").read_text())
    tr = float(side["RepetitionTime"])
    slice_times = np.asarray(side["SliceTiming"], dtype=float)
    nvol = nib.load(bold).shape[3]
    t_vol = np.arange(nvol) * tr + float(np.median(slice_times))
    t0, t1 = 0.0, nvol * tr

    tp, yp = read_bids_physio(bold.with_name(base + "_recording-pulse_physio.tsv.gz"), "cardiac")
    gp, wp = to_work_rate(tp, yp, t0, t1)
    beats_p = peaks(gp, bandpass(wp))

    root = happy_dir / subses / base
    th, yh = read_bids_physio(Path(f"{root}_desc-stdrescardfromfmri_timeseries.tsv.gz"), column)
    gh, wh = to_work_rate(th, yh, t0, t1)
    beats_h = peaks(gh, bandpass(wh))
    info = json.loads(Path(f"{root}_desc-runinfo.json").read_text())
    # happy's own cardiac phase (Hilbert phase of the filtered fundamental, at
    # slice resolution): no beat picking. Interpolated unwrapped, then wrapped.
    ts, inst = read_bids_physio(Path(f"{root}_desc-slicerescardfromfmri_timeseries.tsv.gz"),
                                "instphase_unwrapped")
    good = np.isfinite(inst)
    ts, inst = ts[good], inst[good]

    def happy_phase(tq: np.ndarray) -> np.ndarray:
        ph = np.interp(tq, ts, inst, left=np.nan, right=np.nan)
        return np.mod(ph, 2 * np.pi)

    def pulse_phase(tq: np.ndarray) -> np.ndarray:
        return phase_at(beats_p, tq)

    def happy_beat_phase(tq: np.ndarray) -> np.ndarray:
        return phase_at(beats_h, tq)

    reg_p = regressors(phase_at(beats_p, t_vol))
    reg_h = regressors(phase_at(beats_h, t_vol))
    lag_r = [r_zero(reg_p, regressors(phase_at(beats_h + lag, t_vol))) for lag in LAGS]
    best = int(np.nanargmax(lag_r))

    row = {
        "run": base,
        "nvol": nvol,
        "beats_pulse": len(beats_p),
        "beats_happy": len(beats_h),
        "hr_pulse": mean_hr(beats_p),
        "hr_happy": mean_hr(beats_h),
        "hr_happy_spectral": info.get("cardiacbpm_dlfiltered", info.get("cardiacbpm_bold")),
        "r_zero": r_zero(reg_p, reg_h),
        "r_span": r_span(reg_p, reg_h),
        "r_span_hphase": r_span(reg_p, regressors(happy_phase(t_vol))),
        "lag_best": float(LAGS[best]),
        "r_best": float(lag_r[best]),
    }
    row["dhr"] = row["hr_happy"] - row["hr_pulse"]
    # Beat timing: offset of each pulse beat to the nearest happy beat. The
    # median is the (expected) pulse-transit delay; the robust SD around it is
    # the beat-timing jitter that sets how well phase regressors can agree.
    off = beats_h[np.abs(beats_h[None, :] - beats_p[:, None]).argmin(1)] - beats_p
    med = float(np.median(off))
    jitter = 1.4826 * float(np.median(np.abs(off - med)))
    row["beat_offset_ms"] = 1000 * med
    row["beat_jitter_ms"] = 1000 * jitter
    row["phase_jitter_rad"] = 2 * np.pi * jitter / float(np.median(np.diff(beats_p)))
    if ref_dir is not None:
        ref_info = Path(f"{ref_dir / subses / base}_desc-runinfo.json")
        if ref_info.exists():
            ri = json.loads(ref_info.read_text())
            row["happy_corr_raw2pleth"] = ri.get("corrcoeff_raw2pleth")
            row["happy_delay_raw2pleth"] = ri.get("delay_raw2pleth")
            row["happy_corr_filt2pleth"] = ri.get("corrcoeff_filt2pleth")
            row["happy_bpm_pleth"] = ri.get("cardiacbpm_pleth")
    if fmriprep is not None:
        func = fmriprep / subses / "func"
        t_end = nvol * tr
        phases = {
            "pulse": pulse_phase,
            "happy": happy_beat_phase,
            "hphase": happy_phase,
            "null": lambda tq: pulse_phase(t_end - tq),
        }
        row.update(bold_variance(
            bold, slice_times, tr, phases,
            func / f"{base}_desc-confounds_timeseries.tsv",
            func / f"{base}_desc-brain_mask.nii.gz",
        ))
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bids", type=Path)
    ap.add_argument("--happy", type=Path, help="happy pure-mode output dir")
    ap.add_argument("--ref", type=Path, help="happy ref-mode output dir (optional)")
    ap.add_argument("--fmriprep", type=Path, help="fMRIPrep derivatives dir; enables BOLD variance")
    ap.add_argument("--runs", type=Path)
    ap.add_argument("--index", type=int, help="score only this (1-based) line of --runs")
    ap.add_argument("--collect", type=Path, help="concatenate per-run .tsv files in this dir")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--column", default="cardiacfromfmri_dlfiltered_25.0Hz",
                    help="happy waveform column to score")
    args = ap.parse_args()

    if args.collect is not None:
        table = pd.concat([pd.read_csv(f, sep="\t") for f in sorted(args.collect.glob("*.tsv"))],
                          ignore_index=True)
    else:
        if not (args.bids and args.happy and args.runs):
            ap.error("--bids, --happy and --runs are required unless --collect")
        lines = args.runs.read_text().split()
        if args.index is not None:
            lines = [lines[args.index - 1]]
        rows = []
        for rel in lines:
            try:
                rows.append(score_run(args.bids, rel, args.happy, args.ref, args.column, args.fmriprep))
            except (OSError, KeyError, ValueError) as err:
                rows.append({"run": Path(rel).name.removesuffix("_bold.nii.gz"), "error": str(err)})
        table = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, sep="\t", index=False, float_format="%.5g")
    with pd.option_context("display.width", 200, "display.max_columns", 40):
        print(table)


if __name__ == "__main__":
    main()
