#!/usr/bin/env python3
"""
Alignment gate for the pRF localizer: does the aperture stack join to the TRs?

A constant offset between the stimulus frame sequence and the acquired volumes
produces pRF maps that look like textbook retinotopy -- smooth, contiguous,
plausible polar-angle progressions -- while being rotated or displaced, and
nothing downstream of a successful-looking fit reveals it. This runs before any
fit, and it is cheap.

Two independent tests, neither of which presupposes the answer:

  offset   A stimulus-present regressor is built from the run's OWN `timeframes`
           (so the display-clock regime is handled per run rather than assumed),
           convolved with an HRF, and correlated with every voxel at candidate
           shifts. The shift that maximises the response is the measured offset.
           It should be 0 TR.

  drive    Two runs of the SAME setnum in the same session saw a byte-identical
           aperture sequence, so their stimulus-driven responses should agree.
           Voxelwise correlation between them is a model-free split-half
           reliability map -- it needs no HRF, no offset and no design matrix.
           A noise ceiling comes from correlating runs of DIFFERENT setnums,
           which share everything except the aperture sequence.

Both are reported with the MNI centre of mass and extent of the surviving
cluster, so "it drives occipital cortex" is a measurement rather than an
impression.

Usage:
    python prf_alignment_gate.py --subject 03 --session 02 \
        --aperture-dir DIR --out-dir DIR
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_REPO_ROOT / "src" / "python"))
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

SPACE = "MNI152NLin2009cAsym_res-2"   # anatomically interpretable without an
                                      # atlas fetch; the gate is about timing,
                                      # not about border precision
TR = 1.5
N_TR = 200


# ---------------------------------------------------------------------------

def run_mats_for(subject, session):
    """Map run index -> (path, setnum) from the source behavioural mats."""
    import re
    d = SOURCE_ROOT / f"sub-{subject}" / f"ses-{session}" / "behavioral"
    out = {}
    for p in sorted(d.glob("*_exp9[34].mat")):
        m = re.search(r"_run(\d+)_exp(\d+)\.mat$", p.name)
        out[int(m.group(1))] = (p, int(m.group(2)))
    if not out:
        raise FileNotFoundError(f"no pRF run mats in {d}")
    return out


def stimulus_regressors(aperture, timeframes, n_tr=N_TR, tr=TR):
    """Per-TR stimulus regressors on the run's OWN frame clock.

    `timeframes` gives each stimulus frame's onset relative to the first frame,
    and the first frame is within ~25 ms of the scanner trigger (measured, all
    30 runs), so stimulus time and volume time share an origin. Each frame is
    assigned to TRs by the overlap of [onset, next_onset) with [t*TR, (t+1)*TR)
    -- not by nearest-TR binning, which would alternate 22- and 23-frame bins
    and inject a half-frame jitter that is pure artefact.

    Returns (stim_on, lit) each length n_tr:
      stim_on  fraction of the TR during which the aperture was non-blank
      lit      mean fraction of the aperture's pixels passing the carrier
    """
    onsets = np.asarray(timeframes, dtype=float)
    ends = np.concatenate([onsets[1:], [onsets[-1] + np.median(np.diff(onsets))]])
    frame_nonblank = (aperture.reshape(len(aperture), -1).max(axis=1) > 0).astype(float)
    frame_lit = (aperture.reshape(len(aperture), -1) > 0).mean(axis=1)

    stim_on = np.zeros(n_tr)
    lit = np.zeros(n_tr)
    weight = np.zeros(n_tr)
    edges = np.arange(n_tr + 1) * tr
    first = np.searchsorted(edges, onsets, side="right") - 1
    last = np.searchsorted(edges, ends, side="right") - 1
    for i in range(len(onsets)):
        for t in range(max(first[i], 0), min(last[i], n_tr - 1) + 1):
            w = min(ends[i], edges[t + 1]) - max(onsets[i], edges[t])
            if w <= 0:
                continue
            stim_on[t] += w * frame_nonblank[i]
            lit[t] += w * frame_lit[i]
            weight[t] += w
    good = weight > 0
    stim_on[good] /= weight[good]
    lit[good] /= weight[good]
    return stim_on, lit


def sector_regressors(aperture, timeframes, n_polar=8, n_ecc=2,
                      n_tr=N_TR, tr=TR):
    """Per-TR lit fraction within each visual-field sector.

    The ON/OFF regressor of `stimulus_regressors` is on ~75% of the time and
    becomes very smooth once convolved, so shifting it by a TR barely changes
    the fit and the offset it measures is only good to about +/-1 TR. Sectors
    fix that: a sweeping bar enters each sector at a different moment, so the
    regressors carry sharp, mutually offset transients and the fit degrades
    quickly once the stimulus is shifted off true.

    Returns (n_tr, n_polar * n_ecc).
    """
    n_frames, res, _ = aperture.shape
    yy, xx = np.mgrid[0:res, 0:res]
    c = (res - 1) / 2.0
    rad = np.sqrt((xx - c) ** 2 + (yy - c) ** 2) / c          # 0..1 (corners >1)
    ang = (np.degrees(np.arctan2(-(yy - c), xx - c)) + 360.0) % 360.0
    labels = np.full((res, res), -1, dtype=int)
    inside = rad <= 1.0
    ei = np.clip((rad * n_ecc).astype(int), 0, n_ecc - 1)
    pi = np.clip((ang / (360.0 / n_polar)).astype(int), 0, n_polar - 1)
    labels[inside] = (ei * n_polar + pi)[inside]

    n_sec = n_polar * n_ecc
    flat = aperture.reshape(n_frames, -1) > 0
    lab_flat = labels.ravel()
    frame_sec = np.zeros((n_frames, n_sec), dtype=np.float32)
    for k in range(n_sec):
        sel = lab_flat == k
        if sel.any():
            frame_sec[:, k] = flat[:, sel].mean(axis=1)

    onsets = np.asarray(timeframes, dtype=float)
    ends = np.concatenate([onsets[1:], [onsets[-1] + np.median(np.diff(onsets))]])
    edges = np.arange(n_tr + 1) * tr
    out = np.zeros((n_tr, n_sec))
    weight = np.zeros(n_tr)
    first = np.searchsorted(edges, onsets, side="right") - 1
    last = np.searchsorted(edges, ends, side="right") - 1
    for i in range(n_frames):
        for t in range(max(first[i], 0), min(last[i], n_tr - 1) + 1):
            w = min(ends[i], edges[t + 1]) - max(onsets[i], edges[t])
            if w <= 0:
                continue
            out[t] += w * frame_sec[i]
            weight[t] += w
    good = weight > 0
    out[good] /= weight[good][:, None]
    return out


def convolve_hrf(x, tr=TR):
    from nilearn.glm.first_level.hemodynamic_models import spm_hrf
    h = spm_hrf(tr, oversampling=1)
    x = np.asarray(x)
    if x.ndim == 1:
        return np.convolve(x, h)[:len(x)]
    return np.column_stack([np.convolve(x[:, k], h)[:len(x)]
                            for k in range(x.shape[1])])


def mask_path(subject, session, run, space=SPACE):
    return (DERIV_ROOT / "fmriprep" / f"sub-{subject}" / f"ses-{session}" / "func"
            / f"sub-{subject}_ses-{session}_task-prf_run-{run:02d}"
              f"_space-{space}_desc-brain_mask.nii.gz")


def common_mask(subject, session, runs, space=SPACE):
    """Intersection of the per-run brain masks.

    fMRIPrep writes one brain mask per run and they do NOT agree voxel for
    voxel (they differ by a few hundred voxels here), so every cross-run
    comparison has to be made on the intersection or it compares misaligned
    vectors -- or, if the counts happen to differ, raises.
    """
    import nibabel as nb
    img = None
    acc = None
    for run in runs:
        m = nb.load(str(mask_path(subject, session, run, space)))
        d = np.asarray(m.dataobj) > 0
        acc = d if acc is None else (acc & d)
        img = m
    return acc, img


def load_bold(subject, session, run, space=SPACE):
    import nibabel as nb
    f = (DERIV_ROOT / "fmriprep" / f"sub-{subject}" / f"ses-{session}" / "func"
         / f"sub-{subject}_ses-{session}_task-prf_run-{run:02d}"
           f"_space-{space}_desc-preproc_bold.nii.gz")
    m = (DERIV_ROOT / "fmriprep" / f"sub-{subject}" / f"ses-{session}" / "func"
         / f"sub-{subject}_ses-{session}_task-prf_run-{run:02d}"
           f"_space-{space}_desc-brain_mask.nii.gz")
    for p in (f, m):
        if not p.exists():
            raise FileNotFoundError(f"missing preprocessed input: {p}")
    img = nb.load(str(f))
    return img, nb.load(str(m))


def clean(data2d, confounds_tsv):
    """Detrend and project out motion + fMRIPrep cosine drift terms."""
    import pandas as pd
    df = pd.read_csv(confounds_tsv, sep="\t")
    cols = [c for c in df.columns
            if c.startswith("cosine")
            or c in ("trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z")]
    X = df[cols].to_numpy(dtype=float)
    X = np.nan_to_num(X)
    X = np.column_stack([X, np.ones(len(X))])
    beta, *_ = np.linalg.lstsq(X, data2d, rcond=None)
    resid = data2d - X @ beta
    sd = resid.std(axis=0)
    sd[sd == 0] = 1.0
    return resid / sd


def cluster_report(stat, mask, mask_img, threshold, label):
    """MNI centre of mass and extent of the largest suprathreshold cluster."""
    from scipy import ndimage
    vol = np.zeros(mask_img.shape, dtype=float)
    vol[mask] = stat
    supra = vol >= threshold
    lab, n = ndimage.label(supra)
    if n == 0:
        print(f"  {label}: no voxel reaches r >= {threshold:.3f}")
        return None
    sizes = ndimage.sum(supra, lab, range(1, n + 1))
    big = int(np.argmax(sizes)) + 1
    idx = np.argwhere(lab == big)
    w = vol[lab == big]
    com_vox = (idx * w[:, None]).sum(0) / w.sum()
    mni = nb_affine_apply(mask_img.affine, com_vox)
    peak_vox = np.unravel_index(np.argmax(vol), vol.shape)
    peak_mni = nb_affine_apply(mask_img.affine, np.array(peak_vox, dtype=float))
    print(f"  {label}: largest cluster {int(sizes[big-1])} voxels at r >= "
          f"{threshold:.3f} ({int(supra.sum())} suprathreshold total, "
          f"{n} clusters)")
    print(f"    cluster centre of mass  MNI  "
          f"({mni[0]:+6.1f}, {mni[1]:+6.1f}, {mni[2]:+6.1f})")
    print(f"    peak r = {vol.max():.3f} at MNI "
          f"({peak_mni[0]:+6.1f}, {peak_mni[1]:+6.1f}, {peak_mni[2]:+6.1f})")
    return {"n_voxels": int(sizes[big - 1]), "n_supra": int(supra.sum()),
            "n_clusters": int(n), "com_mni": [float(v) for v in mni],
            "peak_r": float(vol.max()),
            "peak_mni": [float(v) for v in peak_mni]}


def nb_affine_apply(affine, ijk):
    return (affine[:3, :3] @ np.asarray(ijk, dtype=float)) + affine[:3, 3]


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject", required=True, help="bare label, e.g. 03")
    ap.add_argument("--session", required=True, help="bare label, e.g. 02")
    ap.add_argument("--aperture-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--resolution", type=int, default=100)
    ap.add_argument("--n-select", type=int, default=1000,
                    help="voxels kept for test 3, ranked by split-half r")
    ap.add_argument("--max-shift", type=int, default=10,
                    help="candidate offsets scanned, in TRs (default +/-10)")
    args = ap.parse_args()

    import scipy.io as sio
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ap_dir = Path(args.aperture_dir)
    runs = run_mats_for(args.subject, args.session)
    print(f"sub-{args.subject} ses-{args.session}: runs "
          f"{ {r: s for r, (_, s) in runs.items()} }\n")

    results = {"subject": args.subject, "session": args.session, "runs": {}}

    # One shared mask and one load per run: the per-run masks disagree, and
    # decompressing these volumes twice is the whole runtime.
    mask, mask_img = common_mask(args.subject, args.session, sorted(runs))
    print(f"common brain mask: {int(mask.sum())} voxels "
          f"(intersection of {len(runs)} per-run masks)\n")
    results["n_voxels"] = int(mask.sum())

    cleaned = {}
    for run in sorted(runs):
        img, _m = load_bold(args.subject, args.session, run)
        d = np.asarray(img.dataobj)[mask].T.astype(np.float32)
        conf = (DERIV_ROOT / "fmriprep" / f"sub-{args.subject}"
                / f"ses-{args.session}" / "func"
                / f"sub-{args.subject}_ses-{args.session}_task-prf"
                  f"_run-{run:02d}_desc-confounds_timeseries.tsv")
        cleaned[run] = clean(d, conf)
        print(f"  loaded run-{run:02d}: {cleaned[run].shape}")
        del d, img
    print()

    # ---- test 1: offset ----------------------------------------------------
    print("=== TEST 1: measured frame/TR offset ===")
    shifts = np.arange(-args.max_shift, args.max_shift + 1)
    for run, (mat_path, setnum) in sorted(runs.items()):
        aperture = np.load(ap_dir / f"task-prf_set-{setnum}_res-{args.resolution}"
                                    f"_aperture.npy")
        timeframes = np.asarray(sio.loadmat(str(mat_path))["timeframes"]).ravel()
        stim_on, _lit = stimulus_regressors(aperture, timeframes)
        pred = convolve_hrf(stim_on)
        data = cleaned[run]

        curve = []
        for sh in shifts:
            p = np.roll(pred, sh)
            valid = np.ones(len(p), dtype=bool)
            if sh > 0:
                valid[:sh] = False
            elif sh < 0:
                valid[sh:] = False
            pv = p[valid]
            pv = (pv - pv.mean()) / pv.std()
            dv = data[valid]
            dv = dv - dv.mean(0)
            sd = dv.std(0)
            dv = dv / np.where(sd == 0, 1, sd)
            curve.append(np.percentile((pv[:, None] * dv).mean(0), 99.9))
        curve = np.array(curve)
        best = int(shifts[np.argmax(curve)])
        r0 = float(curve[shifts == 0][0])
        print(f"  run-{run:02d} (set {setnum}): best offset {best:+d} TR "
              f"({best * TR:+.2f} s), r99.9 = {curve.max():.3f}; "
              f"at 0 TR r = {r0:.3f}; "
              f"next-best {sorted(curve)[-2]:.3f}")
        results["runs"][f"run-{run:02d}"] = {
            "setnum": setnum, "best_offset_tr": best,
            "r999_at_best": float(curve.max()), "r999_at_zero": r0,
            "curve": [float(v) for v in curve]}
    results["shifts"] = [int(s) for s in shifts]

    # ---- test 2: split-half over identical apertures ------------------------
    print("\n=== TEST 2: split-half reliability (same setnum, same session) ===")
    by_set = {}
    for run, (_p, setnum) in runs.items():
        by_set.setdefault(setnum, []).append(run)
    pair = next(((sn, sorted(rr)[:2]) for sn, rr in sorted(by_set.items())
                 if len(rr) >= 2), None)
    if pair is None:
        print("  no setnum has two runs in this session; skipped")
    else:
        setnum, (ra, rb) = pair
        a, b = cleaned[ra], cleaned[rb]
        r_same = (a * b).mean(0)
        print(f"  setnum {setnum}, run-{ra:02d} vs run-{rb:02d} "
              f"(byte-identical apertures): max r {r_same.max():.3f}, "
              f"99.9th pct {np.percentile(r_same, 99.9):.3f}, "
              f"median {np.median(r_same):.3f}")
        rep = cluster_report(r_same, mask, mask_img,
                             float(np.percentile(r_same, 99.5)), "split-half")
        results["split_half"] = {
            "setnum": setnum, "runs": [ra, rb], "max_r": float(r_same.max()),
            "p999": float(np.percentile(r_same, 99.9)),
            "median_r": float(np.median(r_same)), "cluster": rep}
        np.save(out_dir / "r_split_half.npy", r_same)

        other = sorted(r for r, (_p, sn) in runs.items() if sn != setnum)
        if other:
            run_o = other[0]
            r_diff = (a * cleaned[run_o]).mean(0)
            print(f"  NOISE FLOOR run-{ra:02d} vs run-{run_o:02d} "
                  f"(different setnum, so a different aperture sequence): "
                  f"max r {r_diff.max():.3f}, "
                  f"99.9th pct {np.percentile(r_diff, 99.9):.3f}, "
                  f"median {np.median(r_diff):.3f}")
            results["noise_floor"] = {
                "run": run_o, "max_r": float(r_diff.max()),
                "p999": float(np.percentile(r_diff, 99.9)),
                "median_r": float(np.median(r_diff))}
            np.save(out_dir / "r_diff_setnum.npy", r_diff)
        np.save(out_dir / "mask.npy", mask)

    # ---- test 3: position-resolved offset, on reliability-selected voxels ---
    print("\n=== TEST 3: sector-regressor offset (sharp) ===")
    if "split_half" not in results:
        print("  needs a split-half pair for voxel selection; skipped")
    else:
        # Selection is model-free -- split-half reliability never touches the
        # stimulus timing -- so it cannot bias which offset wins.
        r_sel = np.load(out_dir / "r_split_half.npy")
        k = min(args.n_select, r_sel.size)
        sel = np.argsort(r_sel)[-k:]
        print(f"  {k} voxels selected by split-half r "
              f"(min {r_sel[sel].min():.3f}, max {r_sel[sel].max():.3f})")
        results["test3_selection"] = {
            "n": int(k), "min_r": float(r_sel[sel].min()),
            "max_r": float(r_sel[sel].max())}
        for run, (mat_path, setnum) in sorted(runs.items()):
            aperture = np.load(ap_dir / f"task-prf_set-{setnum}"
                                        f"_res-{args.resolution}_aperture.npy")
            timeframes = np.asarray(sio.loadmat(str(mat_path))["timeframes"]).ravel()
            S = convolve_hrf(sector_regressors(aperture, timeframes))
            Y = cleaned[run][:, sel]
            Y = Y - Y.mean(0)
            sst = (Y ** 2).sum(0)
            curve = []
            for sh in shifts:
                X = np.roll(S, sh, axis=0)
                valid = np.ones(len(X), dtype=bool)
                if sh > 0:
                    valid[:sh] = False
                elif sh < 0:
                    valid[sh:] = False
                Xv = X[valid]
                Xv = np.column_stack([Xv - Xv.mean(0), np.ones(len(Xv))])
                Yv = Y[valid]
                Yv = Yv - Yv.mean(0)
                beta, *_ = np.linalg.lstsq(Xv, Yv, rcond=None)
                sse = ((Yv - Xv @ beta) ** 2).sum(0)
                r2 = 1.0 - sse / np.maximum((Yv ** 2).sum(0), 1e-12)
                curve.append(float(np.median(r2)))
            curve = np.array(curve)
            best = int(shifts[np.argmax(curve)])
            r2_0 = float(curve[shifts == 0][0])
            drop = curve.max() - max(
                curve[shifts == best - 1][0] if best - 1 >= shifts[0] else -1,
                curve[shifts == best + 1][0] if best + 1 <= shifts[-1] else -1)
            print(f"  run-{run:02d} (set {setnum}): best offset {best:+d} TR, "
                  f"median R2 {curve.max():.4f}; at 0 TR {r2_0:.4f}; "
                  f"drop to nearest neighbour {drop:.4f}")
            results["runs"][f"run-{run:02d}"].update({
                "test3_best_offset_tr": best,
                "test3_r2_at_best": float(curve.max()),
                "test3_r2_at_zero": r2_0,
                "test3_curve": [float(v) for v in curve]})

    with open(out_dir / f"sub-{args.subject}_ses-{args.session}_gate.json", "w") as fh:
        json.dump(results, fh, indent=2)
        fh.write("\n")
    print(f"\nwrote {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
