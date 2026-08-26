#!/usr/bin/env python3
"""Confirm the pRF aperture orientation from the GENERATING parameters alone.

Independent of `prf_orientation_check.py`, which answers the same question from
BOLD. This one needs no imaging data at all: the experiment workspace stores the
bar positions used to synthesise the masks, so the stored masks can be checked
directly against the geometry that produced them.

The chain:

  * `barsweepdirs` = [0, 45, 90, 135] deg is the bar's DIRECTION OF MOTION in
    standard polar convention. Verified, not assumed: `xposes`/`yposes` hold one
    sweep's trajectory, and it runs x:+0.40 -> -0.40 while y:-0.40 -> +0.40,
    i.e. motion vector (-1,+1), atan2 = 135 deg -- the last of the four.
  * setnum 93's 1680 masks are 4 sweeps x 420 steps. Their centroids move
    cardinal / diagonal / cardinal / diagonal, matching that order, so sweep 1
    is the 0 deg sweep (pure x motion) and sweep 3 the 90 deg sweep (pure y).
  * A sweep that moves purely in x must move along whichever array axis is the
    screen's horizontal. That is the whole test.

Two hypotheses are scored:
  NO TRANSPOSE   h5py column = +x, h5py row = -y
  TRANSPOSE      h5py row    = +x, h5py column = -y   (h5py reverses MATLAB
                 v7.3 dimensions, so this is the expected answer)
"""

import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_REPO_ROOT / "src" / "python"))
try:
    from core.config import load_config
    _cfg = load_config(config_dir=_REPO_ROOT / "config")
    SOURCE_ROOT = Path(_cfg["paths"]["source_dir"])
except Exception:  # pragma: no cover
    SOURCE_ROOT = Path("/gpfs/projects/hulacon/shared/mmmsourcedata")

WORKSPACE = (SOURCE_ROOT / "shared" / "experiment_code" / "localizer" / "prf"
             / "workspace_retinotopyCaltsmash.mat")
SWEEPS = [(901, 1320, 0.0), (1321, 1740, 45.0),
          (1741, 2160, 90.0), (2161, 2580, 135.0)]


def centroids(block):
    """Weighted centroid of each frame, in the h5py-read axes (row, col)."""
    n = len(block)
    rr = np.full(n, np.nan)
    cc = np.full(n, np.nan)
    for k in range(n):
        m = block[k].astype(np.float32)
        if m.sum() > 0:
            ys, xs = np.nonzero(m)
            w = m[ys, xs]
            rr[k] = (ys * w).sum() / w.sum()
            cc[k] = (xs * w).sum() / w.sum()
    return rr, cc


def main():
    import h5py
    with h5py.File(WORKSPACE, "r") as f:
        xp = np.array(f["xposes"]).ravel()
        yp = np.array(f["yposes"]).ravel()
        res = int(np.array(f["res"]).ravel()[0])
        dirs = np.degrees(np.array(f["barsweepdirs"]).ravel())

        ang = np.degrees(np.arctan2(yp[-1] - yp[0], xp[-1] - xp[0])) % 360
        print(f"xposes/yposes trajectory: motion angle {ang:.1f} deg "
              f"-> matches barsweepdirs entry {dirs[np.argmin(np.abs(dirs - ang))]:.0f}")
        print(f"barsweepdirs = {dirs}, res = {res}\n")

        # signed progression along the sweep axis, recovered from the 135 deg case
        theta = np.radians(135.0)
        rho = xp / np.cos(theta)
        print(f"progression rho: {rho[0]:+.3f} -> {rho[-1]:+.3f} "
              f"(monotonic {'yes' if np.all(np.diff(rho) >= -1e-9) else 'NO'})\n")

        print(f"{'sweep':>22} {'dir':>5} {'moves along':>13} "
              f"{'r(row, x_pred)':>15} {'r(col, x_pred)':>15}")
        print("-" * 76)
        score = {"no_transpose": [], "transpose": []}
        for lo, hi, deg in SWEEPS:
            blk = f["maskimages"][lo - 1:hi]
            rr, cc = centroids(blk)
            ok = ~np.isnan(rr)
            n = ok.sum()
            r = np.radians(deg)
            x_pred = (rho[:len(rr)] * np.cos(r))[ok]
            y_pred = (rho[:len(rr)] * np.sin(r))[ok]

            def cor(a, b):
                return 0.0 if np.std(b) < 1e-9 else float(np.corrcoef(a, b)[0, 1])

            r_row_x, r_col_x = cor(rr[ok], x_pred), cor(cc[ok], x_pred)
            moves = ("row" if abs(np.std(rr[ok])) > 5 * abs(np.std(cc[ok]))
                     else "col" if abs(np.std(cc[ok])) > 5 * abs(np.std(rr[ok]))
                     else "both (diag)")
            print(f"{f'{lo}-{hi}':>22} {deg:>4.0f}d {moves:>13} "
                  f"{r_row_x:>15.3f} {r_col_x:>15.3f}")
            # cardinal sweeps are the only discriminating ones
            if deg in (0.0, 90.0):
                score["no_transpose"].append(cor(cc[ok], x_pred)
                                             if deg == 0.0 else cor(-rr[ok], y_pred))
                score["transpose"].append(cor(rr[ok], x_pred)
                                          if deg == 0.0 else cor(-cc[ok], y_pred))

    print("\nDiscriminating (cardinal) sweeps only -- correlation with the")
    print("generating position under each hypothesis; +1 is a perfect match:\n")
    for name, vals in score.items():
        print(f"  {name:>14}:  0deg/x {vals[0]:+.3f}   90deg/y {vals[1]:+.3f}   "
              f"mean {np.mean(vals):+.3f}")
    win = max(score, key=lambda k: np.mean(score[k]))
    print(f"\nVERDICT: {win.upper().replace('_', ' ')}")
    if win == "transpose":
        print("  h5py row = +x (screen horizontal, increasing rightward)")
        print("  h5py col = -y (screen vertical, so after transposing, row 0 = top)")
        print("  => each frame must be transposed on read; matches the BOLD result")
        print("     in prf_orientation_check.py (r = -0.669 contralateral).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
