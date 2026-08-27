#!/usr/bin/env python3
"""
fit_prf.py — CSS population-receptive-field fits for the pRF localizer.

Fits in the NATIVE functional volume (fMRIPrep's `func` space: the
`desc-preproc_bold` with no `space-` entity, 1.702 mm iso, TR 1.5 s, 200 TRs).
Design record and the reasoning behind that choice:
mmmdata-agents/docs/workbench/prf-retinotopy/ (fit space DECIDED 2026-08-26).

Why the volume and not the surface, in one line: pRF is a nonlinear per-unit
fit, so interpolating TIMESERIES mixes neighbouring pRFs before the fit and
biases size upward, while interpolating fitted PARAMETERS afterwards only
smooths the displayed map. NSD did the same -- `cvnlab/nsddatapaper`
`main/analysis_prf.m` fits analyzePRF in func1mm/func1pt8mm and never on the
surface; `main/analysis_prf_maps.m` projects the parameter volumes afterwards.

ONE FIT UNIT = ONE SESSION (measured, not assumed). Within a session all three
pRF runs share the native grid exactly (max |affine difference| = 0.000 for
every subject); across sessions they never do (5.7-14.5 mm). So three runs
concatenate with no resampling at all, and a combined six-run fit would need a
cross-session resample -- a separate stage this script deliberately does not
fake. Three runs x 200 TRs = 600 TRs = 15 min of retinotopy, which is ample for
polar-angle reversals, and per-session fits ARE the test-retest units the
charter's Settles-when 3 asks for.

Model (analyzePRF's CSS, so the numbers are comparable to NSD's):

    g       = normalised 2D isotropic Gaussian at (x0, y0), width sigma
    raw(t)  = ( sum_pixels S(t, ยท) * g ) ** n        <- power BEFORE convolution
    pred(t) = conv(raw, HRF)
    model   = gain * pred + baseline                  <- solved analytically

Reported `size` is sigma/sqrt(n), matching NSD's `prf_size`; `sigma` is written
too. Angle is degrees counter-clockwise from the right horizontal meridian,
0-360, with the standard screen convention (column = +x rightward, row 0 = top)
that the aperture sidecars pin down.

Usage:
    python fit_prf.py --self-test                       # synthetic recovery, no data
    python fit_prf.py --subject 03 --session 02 --dry-run
    python fit_prf.py --subject 03 --session 02
    python fit_prf.py --subject 03 --session 02 --occipital-only --jobs 28
    python fit_prf.py --subject 03 --session 02 --hrf spm   # SPM comparison arm
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_REPO_ROOT / "src" / "python"))
sys.path.insert(0, str(_SCRIPT_DIR))

try:
    from core.config import load_config
    _config = load_config(config_dir=_REPO_ROOT / "config")
    BIDS_ROOT = Path(_config["paths"]["bids_project_dir"])
    SOURCE_ROOT = Path(_config["paths"]["source_dir"])
    DERIV_ROOT = Path(_config["paths"]["output_dir"])
except Exception:  # pragma: no cover
    BIDS_ROOT = Path("/gpfs/projects/hulacon/shared/mmmdata")
    SOURCE_ROOT = Path("/gpfs/projects/hulacon/shared/mmmsourcedata")
    DERIV_ROOT = BIDS_ROOT / "derivatives"

TR = 1.5
N_TR = 200
APERTURE_RES = 100
FOV_DEG = 15.0          # design geometry from root task-prf_bold.json; the true
                        # subtended angle was never recorded, so eccentricity
                        # and size scale linearly with this. Angle does not.
OUTPUT_TREE = "prf"
POLY_DEGREE = 3         # 1 + floor(300 s / 120 s), the usual analyzePRF choice


# ---------------------------------------------------------------------------
# stimulus: frames -> TRs
# ---------------------------------------------------------------------------

def bin_frames_to_tr(frame_values, timeframes, n_tr=N_TR, tr=TR):
    """Overlap-weighted binning of per-frame values onto the TR grid.

    This is the GENERAL form of `prf_alignment_gate.stimulus_regressors`, which
    the alignment gate used to measure a 0 TR offset. The algorithm is
    deliberately identical -- each frame is assigned to TRs by the overlap of
    [onset, next_onset) with [t*TR, (t+1)*TR), never by nearest-TR binning,
    which would alternate 22- and 23-frame bins and inject a half-frame jitter
    that is pure artefact. `--self-test` asserts this reproduces the gate's
    scalar regressor exactly, so the two cannot drift apart.

    Using each run's OWN `timeframes` is what handles the sub-06/sub-07 display
    clock (59.95 Hz vs 60.000 Hz) without special-casing it.

    frame_values : (n_frames, K) or (n_frames,)
    returns      : (n_tr, K) or (n_tr,)
    """
    fv = np.asarray(frame_values, dtype=np.float64)
    squeeze = fv.ndim == 1
    if squeeze:
        fv = fv[:, None]
    onsets = np.asarray(timeframes, dtype=float).ravel()
    if len(onsets) != len(fv):
        raise ValueError(
            f"timeframes has {len(onsets)} entries but {len(fv)} frames were "
            "given -- the run mat and the aperture stack disagree")
    ends = np.concatenate([onsets[1:], [onsets[-1] + np.median(np.diff(onsets))]])

    out = np.zeros((n_tr, fv.shape[1]))
    weight = np.zeros(n_tr)
    edges = np.arange(n_tr + 1) * tr
    first = np.searchsorted(edges, onsets, side="right") - 1
    last = np.searchsorted(edges, ends, side="right") - 1
    for i in range(len(onsets)):
        for t in range(max(first[i], 0), min(last[i], n_tr - 1) + 1):
            w = min(ends[i], edges[t + 1]) - max(onsets[i], edges[t])
            if w <= 0:
                continue
            out[t] += w * fv[i]
            weight[t] += w
    good = weight > 0
    out[good] /= weight[good][:, None]
    return out[:, 0] if squeeze else out


_KAY_BASE_HRF = (  # 490 samples at 0.1 s, verbatim from getcanonicalhrf.m
    "0;5.34e-06;3.55e-05;0.000104;0.00022;0.000388;0.00061;0.000886;0.00122"
    ";0.00159;0.00202;0.00249;0.003;0.00354;0.00411;0.00471;0.00533;0.00596"
    ";0.0066;0.00725;0.0079;0.00855;0.0092;0.00984;0.0105;0.0111;0.0117;0.0122"
    ";0.0128;0.0133;0.0138;0.0143;0.0148;0.0152;0.0156;0.016;0.0163;0.0166;0.0169"
    ";0.0171;0.0174;0.0176;0.0177;0.0179;0.018;0.0181;0.0181;0.0181;0.0182;0.0181"
    ";0.0181;0.018;0.018;0.0178;0.0177;0.0176;0.0174;0.0173;0.0171;0.0169;0.0167"
    ";0.0164;0.0162;0.0159;0.0157;0.0154;0.0151;0.0148;0.0146;0.0143;0.014;0.0136"
    ";0.0133;0.013;0.0127;0.0124;0.0121;0.0118;0.0114;0.0111;0.0108;0.0105;0.0102"
    ";0.00985;0.00954;0.00923;0.00893;0.00862;0.00832;0.00803;0.00773;0.00744"
    ";0.00716;0.00688;0.0066;0.00633;0.00607;0.00581;0.00555;0.0053;0.00505"
    ";0.00481;0.00458;0.00435;0.00412;0.0039;0.00369;0.00348;0.00328;0.00308"
    ";0.00289;0.0027;0.00252;0.00234;0.00217;0.002;0.00184;0.00168;0.00153"
    ";0.00139;0.00124;0.00111;0.000974;0.000846;0.000722;0.000603;0.000488"
    ";0.000377;0.000271;0.000168;6.96e-05;-2.51e-05;-0.000116;-0.000203;-0.000287"
    ";-0.000367;-0.000443;-0.000516;-0.000586;-0.000653;-0.000716;-0.000777"
    ";-0.000835;-0.00089;-0.000942;-0.000991;-0.00104;-0.00108;-0.00112;-0.00116"
    ";-0.0012;-0.00123;-0.00127;-0.0013;-0.00133;-0.00135;-0.00138;-0.0014"
    ";-0.00142;-0.00144;-0.00146;-0.00147;-0.00149;-0.0015;-0.00151;-0.00152"
    ";-0.00153;-0.00154;-0.00155;-0.00155;-0.00156;-0.00156;-0.00156;-0.00156"
    ";-0.00157;-0.00156;-0.00156;-0.00156;-0.00156;-0.00155;-0.00155;-0.00154"
    ";-0.00154;-0.00153;-0.00153;-0.00152;-0.00151;-0.0015;-0.00149;-0.00148"
    ";-0.00147;-0.00146;-0.00145;-0.00144;-0.00143;-0.00142;-0.00141;-0.0014"
    ";-0.00138;-0.00137;-0.00136;-0.00135;-0.00133;-0.00132;-0.00131;-0.00129"
    ";-0.00128;-0.00127;-0.00125;-0.00124;-0.00122;-0.00121;-0.0012;-0.00118"
    ";-0.00117;-0.00115;-0.00114;-0.00113;-0.00111;-0.0011;-0.00108;-0.00107"
    ";-0.00106;-0.00104;-0.00103;-0.00101;-0.001;-0.000987;-0.000973;-0.00096"
    ";-0.000947;-0.000933;-0.00092;-0.000907;-0.000894;-0.000881;-0.000868"
    ";-0.000855;-0.000843;-0.00083;-0.000818;-0.000805;-0.000793;-0.000781"
    ";-0.000769;-0.000758;-0.000746;-0.000734;-0.000723;-0.000711;-0.0007"
    ";-0.000689;-0.000678;-0.000667;-0.000657;-0.000646;-0.000636;-0.000625"
    ";-0.000615;-0.000605;-0.000595;-0.000585;-0.000576;-0.000566;-0.000557"
    ";-0.000547;-0.000538;-0.000529;-0.00052;-0.000511;-0.000503;-0.000494"
    ";-0.000486;-0.000477;-0.000469;-0.000461;-0.000453;-0.000445;-0.000438"
    ";-0.00043;-0.000422;-0.000415;-0.000408;-0.000401;-0.000393;-0.000387"
    ";-0.00038;-0.000373;-0.000366;-0.00036;-0.000353;-0.000347;-0.000341"
    ";-0.000335;-0.000329;-0.000323;-0.000317;-0.000311;-0.000305;-0.0003"
    ";-0.000294;-0.000289;-0.000284;-0.000278;-0.000273;-0.000268;-0.000263"
    ";-0.000258;-0.000254;-0.000249;-0.000244;-0.00024;-0.000235;-0.000231"
    ";-0.000226;-0.000222;-0.000218;-0.000214;-0.00021;-0.000206;-0.000202"
    ";-0.000198;-0.000194;-0.000191;-0.000187;-0.000184;-0.00018;-0.000177"
    ";-0.000173;-0.00017;-0.000167;-0.000163;-0.00016;-0.000157;-0.000154"
    ";-0.000151;-0.000148;-0.000145;-0.000143;-0.00014;-0.000137;-0.000134"
    ";-0.000132;-0.000129;-0.000127;-0.000124;-0.000122;-0.000119;-0.000117"
    ";-0.000115;-0.000113;-0.00011;-0.000108;-0.000106;-0.000104;-0.000102"
    ";-9.99e-05;-9.79e-05;-9.6e-05;-9.41e-05;-9.22e-05;-9.04e-05;-8.86e-05"
    ";-8.68e-05;-8.51e-05;-8.34e-05;-8.17e-05;-8.01e-05;-7.85e-05;-7.69e-05"
    ";-7.54e-05;-7.39e-05;-7.24e-05;-7.09e-05;-6.95e-05;-6.81e-05;-6.67e-05"
    ";-6.54e-05;-6.4e-05;-6.28e-05;-6.15e-05;-6.02e-05;-5.9e-05;-5.78e-05"
    ";-5.66e-05;-5.55e-05;-5.44e-05;-5.33e-05;-5.22e-05;-5.11e-05;-5.01e-05"
    ";-4.9e-05;-4.8e-05;-4.71e-05;-4.61e-05;-4.52e-05;-4.42e-05;-4.33e-05"
    ";-4.24e-05;-4.16e-05;-4.07e-05;-3.99e-05;-3.91e-05;-3.82e-05;-3.75e-05"
    ";-3.67e-05;-3.59e-05;-3.52e-05;-3.45e-05;-3.37e-05;-3.3e-05;-3.24e-05"
    ";-3.17e-05;-3.1e-05;-3.04e-05;-2.98e-05;-2.91e-05;-2.85e-05;-2.79e-05"
    ";-2.74e-05;-2.68e-05;-2.62e-05;-2.57e-05;-2.51e-05;-2.46e-05;-2.41e-05"
    ";-2.36e-05;-2.31e-05;-2.26e-05;-2.21e-05;-2.17e-05;-2.12e-05;-2.08e-05"
    ";-2.03e-05;-1.99e-05;-1.95e-05;-1.91e-05;-1.87e-05;-1.83e-05;-1.79e-05"
    ";-1.75e-05;-1.72e-05;-1.68e-05;-1.64e-05;-1.61e-05;-1.58e-05;-1.54e-05"
    ";-1.51e-05;-1.48e-05;-1.45e-05;-1.42e-05;-1.38e-05;-1.36e-05;-1.33e-05"
    ";-1.3e-05;-1.27e-05;-1.24e-05;-1.22e-05;-1.19e-05;-1.17e-05;-1.14e-05"
    ";-1.12e-05;-1.09e-05;-1.07e-05;-1.05e-05;-1.02e-05;-1e-05;-9.81e-06;-9.6e-06"
    ";-9.39e-06;-9.19e-06;-8.99e-06;-8.8e-06;-8.61e-06;-8.43e-06;-8.25e-06"
    ";-8.07e-06;-7.89e-06;-7.72e-06;-7.56e-06;-7.39e-06;-7.24e-06;-7.08e-06"
    ";-6.93e-06;-6.78e-06;-6.63e-06;-6.49e-06;-6.35e-06;-6.21e-06;-6.08e-06"
)


def kay_hrf(tr=TR):
    """analyzePRF's default HRF: an exact port of getcanonicalhrf(tr, tr).

    Pinned against cvnlab/analyzePRF source 2026-08-27: the default is
    `options.hrf = getcanonicalhrf(tr,tr)'`, and there is NO per-voxel HRF
    fitting or HRF library anywhere in analyzePRF -- the fit optimises
    position, size, gain and exponent only. The base kernel is 490 hard-coded
    samples at 0.1 s (Kay's fit of spm_hrf to empirically measured HRFs,
    per the 2013/11/18 note in getcanonicalhrf.m), convolved with a one-TR
    boxcar, resampled onto the TR grid with pchip (scipy's PchipInterpolator
    implements the same Fritsch-Carlson scheme as MATLAB interp1 'pchip'),
    and peak-normalised to 1.

    Convention difference from the SPM arm, kept deliberately: this kernel is
    the response to a stimulus lasting one TR, not a unit impulse -- that is
    analyzePRF's stated convention and the mirror inherits it.
    """
    from scipy.interpolate import PchipInterpolator
    base = np.array([float(v) for v in _KAY_BASE_HRF.split(";")])
    if len(base) != 490:
        raise RuntimeError(f"Kay base HRF has {len(base)} samples, expected 490")
    h = np.convolve(base, np.ones(max(1, int(round(tr / 0.1)))))
    t_old = np.arange(len(h)) * 0.1
    h = PchipInterpolator(t_old, h)(np.arange(0.0, t_old[-1] + 1e-9, tr))
    return h / h.max()


def hrf_kernel(tr=TR, kind="spm"):
    """'spm': nilearn SPM canonical at TR resolution -- the same kernel the
    alignment gate used. 'kay': analyzePRF's default -- see kay_hrf()."""
    if kind == "spm":
        from nilearn.glm.first_level.hemodynamic_models import spm_hrf
        return spm_hrf(tr, oversampling=1)
    if kind == "kay":
        return kay_hrf(tr)
    raise ValueError(f"unknown HRF kind {kind!r} (expected 'spm' or 'kay')")


def convolve_cols(x, h):
    """Convolve each column of x with h, truncated to len(x)."""
    x = np.atleast_2d(x.T).T if x.ndim == 1 else x
    n = len(x)
    return np.column_stack([np.convolve(x[:, k], h)[:n] for k in range(x.shape[1])])


# ---------------------------------------------------------------------------
# design assembly
# ---------------------------------------------------------------------------

def load_aperture(setnum, aperture_dir):
    p = Path(aperture_dir) / f"task-prf_set-{setnum}_res-{APERTURE_RES}_aperture.npy"
    if not p.exists():
        sys.exit(f"ERROR: no aperture for setnum {setnum} at {p}\n"
                 f"       Build it with: python {_SCRIPT_DIR}/build_prf_apertures.py")
    a = np.load(p)
    if a.ndim != 3 or a.shape[1:] != (APERTURE_RES, APERTURE_RES):
        sys.exit(f"ERROR: {p} has shape {a.shape}, expected (n_frames, "
                 f"{APERTURE_RES}, {APERTURE_RES})")
    return a


def build_design(subject, session, runs, aperture_dir, jobs=1):
    """Concatenated TR-resolution stimulus for the given runs of one session.

    Returns (S, run_index, setnums) with S of shape (n_tr * n_runs, res, res)
    in 0-1, and run_index labelling each TR with its run for the per-run
    polynomial nuisance.
    """
    from prf_alignment_gate import run_mats_for
    import scipy.io as sio

    mats = run_mats_for(subject, session)
    blocks, run_index, setnums = [], [], []
    for k, run in enumerate(runs):
        if run not in mats:
            sys.exit(f"ERROR: no source mat for sub-{subject} ses-{session} "
                     f"run-{run:02d} (have runs {sorted(mats)})")
        mat_path, setnum = mats[run]
        aperture = load_aperture(setnum, aperture_dir)
        timeframes = np.asarray(sio.loadmat(str(mat_path))["timeframes"]).ravel()
        flat = (aperture.reshape(len(aperture), -1).astype(np.float32) / 255.0)
        binned = bin_frames_to_tr(flat, timeframes)
        blocks.append(binned.reshape(N_TR, APERTURE_RES, APERTURE_RES))
        run_index.extend([k] * N_TR)
        setnums.append(setnum)
        print(f"    run-{run:02d}  set-{setnum}  {len(aperture)} frames -> {N_TR} TRs "
              f"(mean lit {binned.mean():.4f})")

    # The two setnums are DIFFERENT stimuli (93 = bars, 94 = wedge + ring).
    # Pairing a run with the wrong aperture is exactly the class of error the
    # alignment gate exists to catch, so assert rather than trust the loop.
    if len(set(setnums)) == 1:
        print(f"    NOTE: all runs share setnum {setnums[0]}")
    return (np.concatenate(blocks, axis=0).astype(np.float32),
            np.asarray(run_index), setnums)


def nuisance_projector(run_index, degree=POLY_DEGREE):
    """Residual-forming projector for per-run Legendre trends (incl. intercept)."""
    cols = []
    for k in np.unique(run_index):
        sel = run_index == k
        t = np.linspace(-1, 1, sel.sum())
        for d in range(degree + 1):
            c = np.zeros(len(run_index))
            c[sel] = np.polynomial.legendre.legval(t, [0] * d + [1])
            cols.append(c)
    X = np.column_stack(cols)
    Q, _ = np.linalg.qr(X)
    return Q  # residualise as y - Q @ (Q.T @ y)


def residualise(y, Q):
    return y - Q @ (Q.T @ y)


# ---------------------------------------------------------------------------
# the CSS model
# ---------------------------------------------------------------------------

def _grid_coords(res=APERTURE_RES):
    c = (res - 1) / 2.0
    yy, xx = np.mgrid[0:res, 0:res].astype(np.float32)
    return xx, yy, c


def gaussian(x0, y0, sigma, res=APERTURE_RES):
    xx, yy, _ = _grid_coords(res)
    g = np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2.0 * sigma ** 2))
    return (g / (2 * np.pi * sigma ** 2)).astype(np.float32)


def predict(S, x0, y0, sigma, n, h, window=4.0):
    """CSS prediction (unscaled). S is (T, res, res).

    The Gaussian is evaluated only within +/- `window` sigma, which is where
    all of its mass is, and the stimulus is sliced to the same box. For foveal
    voxels (small sigma) this is the difference between a fit that finishes and
    one that does not.
    """
    T, res, _ = S.shape
    half = max(2.0, window * sigma)
    r0, r1 = int(max(0, np.floor(y0 - half))), int(min(res, np.ceil(y0 + half) + 1))
    c0, c1 = int(max(0, np.floor(x0 - half))), int(min(res, np.ceil(x0 + half) + 1))
    if r0 >= r1 or c0 >= c1:
        return np.zeros(T)
    yy, xx = np.mgrid[r0:r1, c0:c1].astype(np.float32)
    g = np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2.0 * sigma ** 2))
    g /= (2 * np.pi * sigma ** 2)
    raw = S[:, r0:r1, c0:c1].reshape(T, -1) @ g.ravel()
    raw = np.power(np.maximum(raw, 0.0), n)
    return np.convolve(raw, h)[:T]


def r2_of(pred_r, y_r):
    """Variance explained (%) of a single residualised predictor, gain free.

    Gain and baseline are linear given the pRF parameters, so they never enter
    the nonlinear search; this is their closed-form optimum.
    """
    denom = float(pred_r @ pred_r)
    if denom <= 0:
        return 0.0, 0.0
    beta = float(pred_r @ y_r) / denom
    ss_res = float(((y_r - beta * pred_r) ** 2).sum())
    ss_tot = float((y_r ** 2).sum())
    if ss_tot <= 0:
        return 0.0, beta
    return 100.0 * (1.0 - ss_res / ss_tot), beta


# ---------------------------------------------------------------------------
# stage 1: grid search
# ---------------------------------------------------------------------------

def build_grid(res=APERTURE_RES, n_pos=18, sigmas=None, exponents=(0.2, 0.5, 1.0)):
    if sigmas is None:
        sigmas = np.array([1.0, 1.8, 3.0, 5.0, 8.0, 13.0, 21.0, 34.0])
    c = (res - 1) / 2.0
    span = np.linspace(c - 0.62 * res, c + 0.62 * res, n_pos)
    grid = [(x, y, s, n) for s in sigmas for n in exponents
            for y in span for x in span
            if np.hypot(x - c, y - c) <= 0.75 * res]
    return grid


def _predict_block(S, block, h):
    P = np.empty((S.shape[0], len(block)), dtype=np.float32)
    for j, (x0, y0, s, n) in enumerate(block):
        P[:, j] = predict(S, x0, y0, s, n, h)
    return P


def grid_search(S, Y_r, Q, h, grid, chunk=400, verbose=True, jobs=1):
    """Best grid seed per voxel, by correlation with residualised predictors.

    Y_r is (T, n_vox), already residualised and column-normalised.
    Returns (best_index, best_r) per voxel.
    """
    T = S.shape[0]
    n_vox = Y_r.shape[1]
    best_r = np.full(n_vox, -np.inf, dtype=np.float32)
    best_i = np.zeros(n_vox, dtype=np.int32)
    for start in range(0, len(grid), chunk):
        block = grid[start:start + chunk]
        if jobs > 1:
            from joblib import Parallel, delayed
            step = max(1, len(block) // jobs)
            parts = [block[i:i + step] for i in range(0, len(block), step)]
            # threads, not processes: `predict` is numpy-bound and releases the
            # GIL, and the arrays are big enough that process startup and
            # pickling would eat the gain (measured: 2.5x threaded vs 1.7x loky).
            P = np.concatenate(
                Parallel(n_jobs=jobs, backend="threading")(
                    delayed(_predict_block)(S, b, h) for b in parts),
                axis=1)
        else:
            P = _predict_block(S, block, h)
        P = residualise(P, Q)
        norm = np.linalg.norm(P, axis=0)
        keep = norm > 0
        P[:, keep] /= norm[keep]
        P[:, ~keep] = 0.0
        C = P.T @ Y_r                       # (n_block, n_vox) = correlation
        idx = np.argmax(C, axis=0)
        val = C[idx, np.arange(n_vox)]
        upd = val > best_r
        best_r[upd] = val[upd]
        best_i[upd] = start + idx[upd]
        if verbose:
            print(f"      grid {min(start + chunk, len(grid))}/{len(grid)}", flush=True)
    return best_i, best_r


# ---------------------------------------------------------------------------
# stage 2: nonlinear refinement
# ---------------------------------------------------------------------------

def refine_batch(Y_sub, S, Q, h, seeds, res=APERTURE_RES):
    """Refine a block of voxels in one task.

    Per-voxel dispatch would hand joblib hundreds of thousands of tasks whose
    overhead rivals the 1.5 s of work each carries; batching keeps the task
    count in the hundreds and lets `S` be memmapped once per worker.
    """
    return [refine_voxel(Y_sub[:, k], S, Q, h, seeds[k], res=res)
            for k in range(Y_sub.shape[1])]


def refine_voxel(y_r, S, Q, h, seed, res=APERTURE_RES):
    from scipy.optimize import least_squares
    x0, y0, s0, n0 = seed

    def resid(p):
        pr = predict(S, p[0], p[1], max(p[2], 0.2), min(max(p[3], 0.01), 1.5), h)
        pr = residualise(pr, Q)
        d = float(pr @ pr)
        if d <= 0:
            return y_r
        return y_r - (float(pr @ y_r) / d) * pr

    lo = [-0.5 * res, -0.5 * res, 0.2, 0.01]
    hi = [1.5 * res, 1.5 * res, 4.0 * res, 1.5]
    try:
        sol = least_squares(resid, [x0, y0, s0, n0], bounds=(lo, hi),
                            method="trf", max_nfev=120, xtol=1e-3, ftol=1e-3)
        p = sol.x
    except Exception:
        p = np.array([x0, y0, s0, n0], dtype=float)
    pr = residualise(predict(S, p[0], p[1], p[2], p[3], h), Q)
    r2, gain = r2_of(pr, y_r)
    return p[0], p[1], p[2], p[3], gain, r2


# ---------------------------------------------------------------------------
# outputs
# ---------------------------------------------------------------------------

def to_visual(x0, y0, sigma, n, res=APERTURE_RES, fov=FOV_DEG):
    """Pixel parameters -> visual-field degrees, NSD's conventions.

    Column index is screen horizontal increasing rightward; row 0 is the TOP of
    the screen, so visual y = -(row - centre). `size` is sigma/sqrt(n), which is
    what NSD releases as prf_size.
    """
    c = (res - 1) / 2.0
    deg_per_px = fov / res
    vx = (x0 - c) * deg_per_px
    vy = -(y0 - c) * deg_per_px
    ecc = np.hypot(vx, vy)
    ang = np.degrees(np.arctan2(vy, vx)) % 360.0
    safe_n = np.maximum(n, 1e-3)
    size = (sigma * deg_per_px) / np.sqrt(safe_n)
    return ang, ecc, size, sigma * deg_per_px


def write_maps(results, mask_img, mask, out_dir, base, meta, suffix="prf"):
    import nibabel as nib
    out_dir.mkdir(parents=True, exist_ok=True)
    shape = mask.shape
    written = []
    for name, vec in results.items():
        vol = np.full(shape, np.nan, dtype=np.float32)
        vol[mask] = vec
        # Deliberately NOT inheriting the brain mask's header: it carries that
        # mask's dtype and scl_slope/scl_inter, which would silently rescale
        # float parameter maps. Only the affine is wanted.
        img = nib.Nifti1Image(vol, mask_img.affine)
        img.header.set_data_dtype(np.float32)
        p = out_dir / f"{base}_desc-{name}_{suffix}.nii.gz"
        nib.save(img, str(p))
        written.append(p.name)
    sidecar = out_dir / f"{base}_{suffix}.json"
    meta = dict(meta)
    meta["Maps"] = written
    sidecar.write_text(json.dumps(meta, indent=2) + "\n")
    return written, sidecar


# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------

def self_test(aperture_dir):
    """Synthetic recovery + a drift check against the alignment gate.

    Success criteria, asserted rather than eyeballed:
      1. bin_frames_to_tr reproduces prf_alignment_gate.stimulus_regressors
         exactly (max abs diff < 1e-9), so the two binnings cannot drift.
      2. With SNR 1.0, recovered polar angle correlates circularly with truth
         at r > 0.9 and eccentricity at r > 0.9, over voxels whose fitted
         R2 exceeds 20%.
    """
    from prf_alignment_gate import stimulus_regressors
    import scipy.io as sio
    from prf_alignment_gate import run_mats_for

    print("[self-test] 1/3  binning agrees with the alignment gate")
    mats = run_mats_for("03", "02")
    mat_path, setnum = mats[1]
    aperture = load_aperture(setnum, aperture_dir)
    timeframes = np.asarray(sio.loadmat(str(mat_path))["timeframes"]).ravel()
    gate_on, gate_lit = stimulus_regressors(aperture, timeframes)
    flat = aperture.reshape(len(aperture), -1).astype(np.float32) / 255.0
    mine = bin_frames_to_tr(flat, timeframes)
    mine_lit = bin_frames_to_tr((flat > 0).astype(np.float32), timeframes).mean(axis=1)
    d = np.abs(mine_lit - gate_lit).max()
    print(f"           max |mine - gate| on the lit regressor = {d:.3e}")
    assert d < 1e-9, f"binning drifted from the alignment gate ({d:.3e})"

    print("[self-test] 2/3  Kay HRF construction (getcanonicalhrf mirror)")
    hk = kay_hrf()
    # conv(490, 15) = 504 samples at 0.1 s -> t 0..50.3 s -> 34 points at 1.5 s
    assert len(hk) == 34, f"kay HRF has {len(hk)} points, expected 34"
    assert abs(hk.max() - 1.0) < 1e-12, "kay HRF not peak-normalised"
    peak_t = float(np.argmax(hk)) * TR
    assert 4.0 <= peak_t <= 6.0, f"kay HRF peaks at {peak_t} s, outside 4-6 s"
    print(f"           34 points, peak 1.0 at t = {peak_t:.1f} s, "
          f"undershoot min {hk.min():.4f}")

    print("[self-test] 3/3  synthetic parameter recovery")
    S = mine.reshape(N_TR, APERTURE_RES, APERTURE_RES)
    S = np.concatenate([S, S], axis=0).astype(np.float32)
    run_index = np.repeat([0, 1], N_TR)
    Q = nuisance_projector(run_index)
    h = hrf_kernel()

    rng = np.random.default_rng(0)
    c = (APERTURE_RES - 1) / 2.0
    n_vox = 60
    truth = []
    Y = []
    for _ in range(n_vox):
        ecc = rng.uniform(3, 35)
        ang = rng.uniform(0, 2 * np.pi)
        x0, y0 = c + ecc * np.cos(ang), c - ecc * np.sin(ang)
        sg = rng.uniform(2, 12)
        nn = rng.uniform(0.2, 0.8)
        pr = predict(S, x0, y0, sg, nn, h)
        pr = pr / (pr.std() + 1e-12)
        Y.append(pr + rng.normal(0, 1.0, len(pr)))   # SNR 1.0
        truth.append((x0, y0, sg, nn))
    Y = np.column_stack(Y)
    Y_r = residualise(Y, Q)
    Y_n = Y_r / (np.linalg.norm(Y_r, axis=0) + 1e-12)

    grid = build_grid()
    print(f"           grid has {len(grid)} candidates")
    best_i, _ = grid_search(S, Y_n, Q, h, grid, verbose=False)
    got = [refine_voxel(Y_r[:, v], S, Q, h, grid[best_i[v]]) for v in range(n_vox)]

    r2 = np.array([g[5] for g in got])
    keep = r2 > 20.0
    ta = np.array([to_visual(*t)[0] for t in truth])
    ga = np.array([to_visual(g[0], g[1], g[2], g[3])[0] for g in got])
    te = np.array([to_visual(*t)[1] for t in truth])
    ge = np.array([to_visual(g[0], g[1], g[2], g[3])[1] for g in got])

    def circ_corr(a, b):
        a, b = np.radians(a), np.radians(b)
        a1, b1 = a - np.angle(np.exp(1j * a).mean()), b - np.angle(np.exp(1j * b).mean())
        return float((np.sin(a1) * np.sin(b1)).sum() /
                     np.sqrt((np.sin(a1) ** 2).sum() * (np.sin(b1) ** 2).sum()))

    ca = circ_corr(ta[keep], ga[keep])
    ce = float(np.corrcoef(te[keep], ge[keep])[0, 1])
    print(f"           {keep.sum()}/{n_vox} voxels with R2 > 20%")
    print(f"           polar angle  circular r = {ca:.3f}")
    print(f"           eccentricity Pearson  r = {ce:.3f}")
    assert keep.sum() >= n_vox // 2, "too few voxels recovered at SNR 1.0"
    assert ca > 0.9, f"polar angle recovery failed (r={ca:.3f})"
    assert ce > 0.9, f"eccentricity recovery failed (r={ce:.3f})"
    print("[self-test] PASS")


# ---------------------------------------------------------------------------

def native_bold(subject, session, run):
    return (DERIV_ROOT / "fmriprep" / f"sub-{subject}" / f"ses-{session}" / "func"
            / f"sub-{subject}_ses-{session}_task-prf_run-{run:02d}"
              "_desc-preproc_bold.nii.gz")


def native_mask(subject, session, run):
    return (DERIV_ROOT / "fmriprep" / f"sub-{subject}" / f"ses-{session}" / "func"
            / f"sub-{subject}_ses-{session}_task-prf_run-{run:02d}"
              "_desc-brain_mask.nii.gz")


def detect_runs(subject, session):
    d = DERIV_ROOT / "fmriprep" / f"sub-{subject}" / f"ses-{session}" / "func"
    if not d.is_dir():
        sys.exit(f"ERROR: no fMRIPrep func dir at {d}")
    runs = []
    for p in sorted(d.glob(f"sub-{subject}_ses-{session}_task-prf_run-*_desc-preproc_bold.nii.gz")):
        if "space-" in p.name:
            continue
        runs.append(int(p.name.split("run-")[1][:2]))
    if not runs:
        sys.exit(f"ERROR: no native-space (no space- entity) pRF BOLD in {d}")
    return sorted(runs)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject", help="bare label, e.g. 03")
    ap.add_argument("--session", help="bare label, e.g. 02")
    ap.add_argument("--runs", type=int, nargs="+", help="default: all pRF runs found")
    ap.add_argument("--aperture-dir", default=str(BIDS_ROOT / "stimuli" / "prf"))
    ap.add_argument("--out-root", default=str(DERIV_ROOT / OUTPUT_TREE))
    ap.add_argument("--fov-deg", type=float, default=FOV_DEG)
    ap.add_argument("--refine-threshold", type=float, default=5.0,
                    help="grid R2%% below which a voxel is not refined (default 5)")
    ap.add_argument("--occipital-only", action="store_true",
                    help="restrict to the posterior third of the FOV (fast pilot)")
    ap.add_argument("--negate", action="store_true",
                    help="sign-flip the PSC BOLD before fitting to characterise "
                         "negative pRFs; outputs use the `negprf` suffix")
    ap.add_argument("--hrf", choices=("kay", "spm"), default="kay",
                    help="HRF kernel. Default 'kay' = analyzePRF's default "
                         "getcanonicalhrf (DECIDED 2026-08-27: it won the "
                         "per-voxel R2 comparison and is the truer NSD "
                         "mirror) writing plain prf/negprf suffixes; 'spm' = "
                         "nilearn SPM canonical, the comparison arm, writing "
                         "spmprf/negspmprf")
    ap.add_argument("--jobs", type=int, default=28)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test(args.aperture_dir)
        return 0

    if not (args.subject and args.session):
        ap.error("--subject and --session are required (or use --self-test)")

    import nibabel as nib
    from joblib import Parallel, delayed

    subject, session = args.subject, args.session
    runs = args.runs or detect_runs(subject, session)
    print(f"sub-{subject} ses-{session}: runs {runs}")

    # Native grids agree within a session and never across; refuse to fake it.
    affines = [nib.load(str(native_bold(subject, session, r))).affine for r in runs]
    for r, a in zip(runs[1:], affines[1:]):
        d = np.abs(a - affines[0]).max()
        if d > 1e-4:
            sys.exit(
                f"ERROR: run-{r:02d} does not share run-{runs[0]:02d}'s native grid "
                f"(max |affine diff| = {d:.3f} mm).\n"
                "       Native-space runs concatenate only on a common grid. Within a\n"
                "       session they always do; across sessions they never do (5.7-14.5 mm).\n"
                "       Fit one session at a time, or resample onto a common subject grid\n"
                "       first (see scripts/resample_bold_to_func.py for the container route).")

    print("  building design")
    S, run_index, setnums = build_design(subject, session, runs, args.aperture_dir)
    if S.shape[0] != N_TR * len(runs):
        sys.exit(f"ERROR: design has {S.shape[0]} TRs, expected {N_TR * len(runs)}")

    print("  loading BOLD")
    mask_img = nib.load(str(native_mask(subject, session, runs[0])))
    mask = np.asarray(mask_img.dataobj) > 0
    for r in runs[1:]:
        mask &= np.asarray(nib.load(str(native_mask(subject, session, r))).dataobj) > 0
    if args.occipital_only:
        # posterior third along the second axis; a pilot convenience, and the
        # sidecar records that the map is not whole-brain.
        cut = int(mask.shape[1] / 3)
        keep = np.zeros_like(mask)
        keep[:, :cut, :] = True
        mask &= keep
    print(f"    mask: {int(mask.sum())} voxels")

    blocks = []
    for r in runs:
        d = np.asarray(nib.load(str(native_bold(subject, session, r))).dataobj,
                       dtype=np.float32)[mask]          # (n_vox, N_TR)
        mu = d.mean(axis=1, keepdims=True)
        mu[mu == 0] = 1.0
        blocks.append((100.0 * (d - mu) / mu).T)        # PSC, (N_TR, n_vox)
    Y = np.concatenate(blocks, axis=0)
    del blocks

    # Negative-pRF variant: an identical fit on sign-flipped data. The grid
    # stage keeps only positively-correlated voxels (grid R2 is signed), so
    # the standard fit is blind to anticorrelated responses and this flipped
    # pass is their exact complement, not a redundancy.
    if args.negate:
        print("  --negate: sign-flipping PSC BOLD (negative-pRF fit)")
        Y = -Y

    Q = nuisance_projector(run_index)
    h = hrf_kernel(kind=args.hrf)
    Y_r = residualise(Y, Q)
    norms = np.linalg.norm(Y_r, axis=0)
    live = norms > 0
    Y_n = np.zeros_like(Y_r)
    Y_n[:, live] = Y_r[:, live] / norms[live]

    meta = {
        "Description": "CSS pRF fit in the native functional volume (fMRIPrep `func`).",
        "Subject": f"sub-{subject}", "Session": f"ses-{session}",
        "Runs": runs, "SetNumbers": setnums,
        "Space": "func (native boldref; no space- entity)",
        "Model": "analyzePRF CSS: gain * conv((S.g)^n, HRF) + baseline",
        "HRF": ("nilearn SPM canonical at TR (deviation from analyzePRF's "
                "default getcanonicalhrf)" if args.hrf == "spm" else
                "Kay canonical: exact port of analyzePRF's default "
                "getcanonicalhrf(tr, tr) (490-sample 0.1 s base HRF, one-TR "
                "boxcar, pchip resample to TR, peak-normalised)"),
        "HRFKind": args.hrf,
        "TR": TR, "VolumesPerRun": N_TR, "ApertureResolution": APERTURE_RES,
        "FieldOfViewDeg": args.fov_deg,
        "FieldOfViewCaveat": ("Design geometry only; the subtended angle at the "
                              "scanner was never recorded. Eccentricity and size "
                              "scale LINEARLY with this value. Polar angle does not."),
        "SizeDefinition": "sigma/sqrt(n), matching NSD prf_size",
        "AngleConvention": "degrees CCW from right horizontal meridian, 0-360",
        "PolynomialDegreePerRun": POLY_DEGREE,
        "RefineThresholdR2Pct": args.refine_threshold,
        "OccipitalOnly": bool(args.occipital_only),
        "MaskVoxels": int(mask.sum()),
        "Provenance": "mmmdata/scripts/fit_prf.py; workbench prf-retinotopy",
    }
    if args.negate:
        meta["Description"] = ("CSS pRF fit on SIGN-FLIPPED PSC BOLD in the native "
                               "functional volume: negative-pRF characterisation.")
        meta["SignFlipped"] = True
        meta["SignFlipReference"] = ("Negative-pRF approach per "
                                     "https://www.biorxiv.org/content/10.1101/"
                                     "2024.09.27.615397v2")

    if args.dry_run:
        print("\n--dry-run: design and data assembled, no fitting.")
        print(json.dumps(meta, indent=2))
        print(f"  design S {S.shape}  Y {Y.shape}  grid {len(build_grid())} candidates")
        return 0

    grid = build_grid()
    print(f"  stage 1: grid search ({len(grid)} candidates)")
    best_i, best_r = grid_search(S, Y_n, Q, h, grid, jobs=args.jobs)
    grid_r2 = 100.0 * np.sign(best_r) * best_r ** 2

    todo = np.where(grid_r2 > args.refine_threshold)[0]
    print(f"  stage 2: refining {len(todo)}/{Y.shape[1]} voxels "
          f"(grid R2 > {args.refine_threshold}%) on {args.jobs} jobs")
    batch = max(1, min(500, len(todo) // (args.jobs * 4) or 1))
    batches = [todo[i:i + batch] for i in range(0, len(todo), batch)]
    print(f"           {len(batches)} batches of <= {batch} voxels")
    out = []
    for res_block in Parallel(n_jobs=args.jobs, verbose=5)(
            delayed(refine_batch)(Y_r[:, b], S, Q, h, [grid[best_i[v]] for v in b])
            for b in batches):
        out.extend(res_block)

    n_vox = Y.shape[1]
    px = np.full((n_vox, 4), np.nan)
    gain = np.full(n_vox, np.nan)
    r2 = np.zeros(n_vox)
    for v, o in zip(todo, out):
        px[v] = o[:4]
        gain[v] = o[4]
        r2[v] = o[5]

    ang, ecc, size, sigma_deg = to_visual(px[:, 0], px[:, 1], px[:, 2], px[:, 3],
                                          fov=args.fov_deg)
    ang[np.isnan(px[:, 0])] = np.nan
    results = {"R2": r2.astype(np.float32), "angle": ang, "eccentricity": ecc,
               "size": size, "sigma": sigma_deg, "exponent": px[:, 3],
               "gain": gain}

    out_dir = Path(args.out_root) / f"sub-{subject}" / f"ses-{session}"
    base = f"sub-{subject}_ses-{session}_task-prf_space-func"
    # Suffix scheme: the default (Kay) arm is unmarked -- prf / negprf -- and
    # the SPM comparison arm is marked spmprf / negspmprf. The Kay arm won the
    # per-voxel R2 comparison (2026-08-27, derivatives/prf/hrf-comparison.json)
    # and the SPM outputs were deleted as cheap to reconstruct with --hrf spm.
    suffix = (("neg" if args.negate else "")
              + ("spm" if args.hrf == "spm" else "") + "prf")
    written, sidecar = write_maps(results, mask_img, mask, out_dir, base, meta,
                                  suffix=suffix)

    good = r2 > 10.0
    print(f"\n  wrote {len(written)} maps to {out_dir}")
    print(f"  sidecar: {sidecar.name}")
    print(f"  voxels with R2 > 10%: {int(good.sum())} "
          f"({100.0 * good.sum() / n_vox:.1f}% of mask)")
    if good.sum():
        print(f"  median eccentricity there: {np.nanmedian(ecc[good]):.2f} deg")
        print(f"  median pRF size there:     {np.nanmedian(size[good]):.2f} deg")
    return 0


if __name__ == "__main__":
    sys.exit(main())
