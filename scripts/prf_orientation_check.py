#!/usr/bin/env python3
"""Determine the displayed orientation of the reconstructed pRF aperture.

The aperture stack is read out of a MATLAB v7.3 workspace with h5py, which
returns dimensions in reverse order -- so the frame h5py hands back is the
transpose of the one MATLAB drew. Get that wrong and polar angle is mirrored,
V1/V2/V3 borders land in the wrong place, and the maps still look like
textbook retinotopy. So measure it rather than reason about it.

WHAT DOES NOT WORK, and why it is worth saying: scoring each candidate
orientation by the fit of a free-weight sector GLM. A 90-degree rotation
merely permutes the sector regressors, and R-squared is invariant to permuting
design-matrix columns, so all eight candidates score identically. Any test
that lets the model re-weight space cannot see orientation.

WHAT WORKS: anchor the stimulus to anatomy the reconstruction cannot influence.
Two facts about visual cortex settle both axes:

  horizontal   Each hemisphere represents the CONTRALATERAL visual field, so
               preferred horizontal position must correlate NEGATIVELY with a
               voxel's MNI x.
  vertical     Dorsal V1 (above the calcarine) represents the LOWER visual
               field, so preferred vertical position must correlate
               NEGATIVELY with MNI z.

Sector weights are fitted once, in the raw frame. A candidate orientation is
then just a rigid map applied to the resulting preferred position, so the eight
candidates are scored without refitting -- and the fit itself is identical
across them, which is precisely the degeneracy above, now harmless.

Usage:
    python prf_orientation_check.py --subject 03 --session 02 \
        --aperture-dir DIR --gate-dir DIR
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))
from prf_alignment_gate import (  # noqa: E402
    DERIV_ROOT, N_TR, TR, clean, common_mask, convolve_hrf, load_bold,
    run_mats_for,
)

TRANSFORMS = {
    "identity":       lambda a: a,
    "rot90":          lambda a: np.rot90(a, 1, axes=(0, 1)),
    "rot180":         lambda a: np.rot90(a, 2, axes=(0, 1)),
    "rot270":         lambda a: np.rot90(a, 3, axes=(0, 1)),
    "transpose":      lambda a: a.T,
    "fliplr":         lambda a: a[:, ::-1],
    "flipud":         lambda a: a[::-1, :],
    "anti_transpose": lambda a: np.rot90(a.T, 2, axes=(0, 1)),
}


def sector_geometry(res, n_polar=8, n_ecc=2):
    """Label image plus each sector's centroid in screen coords (+x right, +y up)."""
    yy, xx = np.mgrid[0:res, 0:res]
    c = (res - 1) / 2.0
    rad = np.sqrt((xx - c) ** 2 + (yy - c) ** 2) / c
    ang = (np.degrees(np.arctan2(-(yy - c), xx - c)) + 360.0) % 360.0
    lab = np.full((res, res), -1, dtype=int)
    inside = rad <= 1.0
    ei = np.clip((rad * n_ecc).astype(int), 0, n_ecc - 1)
    pi = np.clip((ang / (360.0 / n_polar)).astype(int), 0, n_polar - 1)
    lab[inside] = (ei * n_polar + pi)[inside]
    return lab, n_polar * n_ecc, c


def sector_positions(lab, n_sec, c, transform):
    """Screen position of each raw-frame sector under a candidate orientation.

    Measured by transforming the label image itself, so the mapping is read off
    the same operation that would be applied to the aperture -- no hand-derived
    sign conventions to get backwards.
    """
    t = transform(lab)
    pos = np.zeros((n_sec, 2))
    for k in range(n_sec):
        ys, xs = np.nonzero(t == k)
        if len(ys):
            pos[k] = [xs.mean() - c, -(ys.mean() - c)]   # +x right, +y up
    return pos


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject", required=True)
    ap.add_argument("--session", required=True)
    ap.add_argument("--aperture-dir", required=True)
    ap.add_argument("--gate-dir", required=True)
    ap.add_argument("--resolution", type=int, default=100)
    ap.add_argument("--n-select", type=int, default=2000)
    ap.add_argument("--out-json")
    args = ap.parse_args()

    import scipy.io as sio
    ap_dir = Path(args.aperture_dir)
    runs = run_mats_for(args.subject, args.session)

    r_sel = np.load(Path(args.gate_dir) / "r_split_half.npy")
    sel = np.argsort(r_sel)[-args.n_select:]
    mask, mask_img = common_mask(args.subject, args.session, sorted(runs))

    # MNI coordinate of every selected voxel -- the anatomical anchor
    ijk = np.argwhere(mask)[sel].astype(float)
    mni = (mask_img.affine[:3, :3] @ ijk.T).T + mask_img.affine[:3, 3]
    print(f"{len(sel)} voxels, split-half r {r_sel[sel].min():.3f}-"
          f"{r_sel[sel].max():.3f}")
    print(f"  MNI x {mni[:,0].min():+.0f}..{mni[:,0].max():+.0f}, "
          f"z {mni[:,2].min():+.0f}..{mni[:,2].max():+.0f}\n")

    lab, n_sec, c = sector_geometry(args.resolution)
    lab_flat = lab.ravel()

    # ---- fit sector weights once, in the raw frame --------------------------
    betas = []
    for run, (mat_path, setnum) in sorted(runs.items()):
        img, _m = load_bold(args.subject, args.session, run)
        d = np.asarray(img.dataobj)[mask].T.astype(np.float32)
        conf = (DERIV_ROOT / "fmriprep" / f"sub-{args.subject}"
                / f"ses-{args.session}" / "func"
                / f"sub-{args.subject}_ses-{args.session}_task-prf"
                  f"_run-{run:02d}_desc-confounds_timeseries.tsv")
        Y = clean(d, conf)[:, sel]
        del d, img

        aperture = np.load(ap_dir / f"task-prf_set-{setnum}"
                                    f"_res-{args.resolution}_aperture.npy")
        tf = np.asarray(sio.loadmat(str(mat_path))["timeframes"]).ravel()
        onsets = tf
        ends = np.concatenate([onsets[1:], [onsets[-1] + np.median(np.diff(onsets))]])
        edges = np.arange(N_TR + 1) * TR
        W = np.zeros((len(onsets), N_TR))
        first = np.searchsorted(edges, onsets, side="right") - 1
        last = np.searchsorted(edges, ends, side="right") - 1
        for i in range(len(onsets)):
            for t in range(max(first[i], 0), min(last[i], N_TR - 1) + 1):
                w = min(ends[i], edges[t + 1]) - max(onsets[i], edges[t])
                if w > 0:
                    W[i, t] = w
        wsum = np.where(W.sum(0) == 0, 1.0, W.sum(0))

        flat = aperture.reshape(len(aperture), -1) > 0
        frame_sec = np.zeros((len(aperture), n_sec), dtype=np.float32)
        for k in range(n_sec):
            s = lab_flat == k
            if s.any():
                frame_sec[:, k] = flat[:, s].mean(axis=1)
        X = convolve_hrf((W.T @ frame_sec) / wsum[:, None])
        Xd = np.column_stack([X - X.mean(0), np.ones(len(X))])
        Yd = Y - Y.mean(0)
        b, *_ = np.linalg.lstsq(Xd, Yd, rcond=None)
        betas.append(b[:n_sec])
        print(f"  fitted run-{run:02d} (set {setnum})")
        del Y
    B = np.mean(betas, axis=0)                     # (n_sec, n_voxels)
    print()

    # preferred position: positive sector weights only, so a voxel is placed by
    # what drives it rather than by what suppresses it
    Bp = np.clip(B, 0, None)
    tot = Bp.sum(0)
    keep = tot > 0
    print(f"  {int(keep.sum())}/{len(sel)} voxels have positive sector weight\n")

    # ---- score each candidate orientation against anatomy -------------------
    rows = []
    for name, fn in TRANSFORMS.items():
        pos = sector_positions(lab, n_sec, c, fn)
        pref = (Bp[:, keep].T @ pos) / tot[keep][:, None]     # (V, 2)
        rx = np.corrcoef(pref[:, 0], mni[keep, 0])[0, 1]      # want NEGATIVE
        rz = np.corrcoef(pref[:, 1], mni[keep, 2])[0, 1]      # want NEGATIVE
        rows.append((name, rx, rz, -(rx + rz)))
    rows.sort(key=lambda r: -r[3])

    print(f"{'transform':>16} {'r(pref_x, MNI x)':>18} {'r(pref_y, MNI z)':>18} "
          f"{'score':>8}  verdict")
    print("-" * 78)
    for name, rx, rz, sc in rows:
        ok = "contralateral + dorsal=lower" if (rx < 0 and rz < 0) else ""
        print(f"{name:>16} {rx:>18.3f} {rz:>18.3f} {sc:>8.3f}  {ok}")
    win, wx, wz, wsc = rows[0]
    print(f"\nWINNER: {win}   r(x)={wx:.3f}  r(z)={wz:.3f}")
    print(f"runner-up: {rows[1][0]} (score {rows[1][3]:.3f}, "
          f"margin {wsc - rows[1][3]:.3f})")
    if not (wx < 0 and wz < 0):
        print("\n*** NO candidate satisfies both anatomical constraints -- "
              "do not use this result ***")

    if args.out_json:
        with open(args.out_json, "w") as fh:
            json.dump({"winner": win,
                       "candidates": [{"transform": n, "r_x_vs_mni_x": rx,
                                       "r_y_vs_mni_z": rz, "score": sc}
                                      for n, rx, rz, sc in rows]}, fh, indent=2)
            fh.write("\n")
        print(f"wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
