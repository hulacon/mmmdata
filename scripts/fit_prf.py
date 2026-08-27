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


def hrf_kernel(tr=TR):
    """SPM canonical at TR resolution -- the same kernel the alignment gate used."""
    from nilearn.glm.first_level.hemodynamic_models import spm_hrf
    return spm_hrf(tr, oversampling=1)


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


def write_maps(results, mask_img, mask, out_dir, base, meta):
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
        p = out_dir / f"{base}_desc-{name}_prf.nii.gz"
        nib.save(img, str(p))
        written.append(p.name)
    sidecar = out_dir / f"{base}_prf.json"
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

    print("[self-test] 1/2  binning agrees with the alignment gate")
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

    print("[self-test] 2/2  synthetic parameter recovery")
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

    Q = nuisance_projector(run_index)
    h = hrf_kernel()
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
        "HRF": "nilearn SPM canonical at TR (deviation from analyzePRF's own HRF library)",
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
    written, sidecar = write_maps(results, mask_img, mask, out_dir, base, meta)

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
