#!/usr/bin/env python3
"""
compare_prf_hrf.py — per-voxel comparison of the two pRF HRF arms.

The fits differ in exactly one component: the HRF kernel (`fit_prf.py --hrf`).
  kay  -> suffixes prf / negprf        (analyzePRF default getcanonicalhrf;
                                        the default arm, DECIDED 2026-08-27)
  spm  -> suffixes spmprf / negspmprf  (nilearn SPM canonical, comparison arm;
                                        its outputs were deleted 2026-08-27 as
                                        cheap to reconstruct with --hrf spm)
The 2026-08-27 comparison that settled the default ran under the older naming
(spm unmarked, kay marked) and is recorded in
derivatives/prf/hrf-comparison.json.
Everything else — stimulus, mask, nuisance projection, grid, refinement — is
identical, so a per-voxel R² difference is attributable to the kernel alone.

One caveat the numbers must carry: the refinement set is NOT shared. A voxel is
refined only if its grid R² clears the threshold, and grid R² depends on the
HRF, so each arm refines its own voxel set and unrefined voxels hold R² = 0.
Comparisons are therefore reported on the UNION of supra-threshold voxels
(default R² > 10% in either arm) — a voxel one kernel finds and the other
misses is a real difference, not missing data — with the intersection reported
alongside.

Usage:
    python compare_prf_hrf.py                       # all sessions, both polarities
    python compare_prf_hrf.py --subject 03 --session 02
    python compare_prf_hrf.py --json out.json
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
    DERIV_ROOT = Path(_config["paths"]["output_dir"])
except Exception:  # pragma: no cover
    DERIV_ROOT = Path("/gpfs/projects/hulacon/shared/mmmdata/derivatives")

PRF_ROOT = DERIV_ROOT / "prf"
ARMS = {"pos": ("spmprf", "prf"), "neg": ("negspmprf", "negprf")}
MAPS = ("R2", "angle", "eccentricity", "size")


def load_maps(sub, ses, suffix):
    import nibabel as nib
    d = PRF_ROOT / f"sub-{sub}" / f"ses-{ses}"
    base = f"sub-{sub}_ses-{ses}_task-prf_space-func"
    out = {}
    for name in MAPS:
        p = d / f"{base}_desc-{name}_{suffix}.nii.gz"
        if not p.exists():
            return None  # arm not (yet) fitted for this session
        out[name] = np.asarray(nib.load(str(p)).dataobj, dtype=np.float64)
    return out


def circ_corr(a_deg, b_deg):
    a, b = np.radians(a_deg), np.radians(b_deg)
    a1 = a - np.angle(np.exp(1j * a).mean())
    b1 = b - np.angle(np.exp(1j * b).mean())
    denom = np.sqrt((np.sin(a1) ** 2).sum() * (np.sin(b1) ** 2).sum())
    if denom == 0:
        return float("nan")
    return float((np.sin(a1) * np.sin(b1)).sum() / denom)


def compare_session(sub, ses, polarity, thresh):
    spm_sfx, kay_sfx = ARMS[polarity]
    spm = load_maps(sub, ses, spm_sfx)
    kay = load_maps(sub, ses, kay_sfx)
    if spm is None or kay is None:
        return None

    r2s, r2k = spm["R2"], kay["R2"]
    # R² maps are 0 where the grid gate rejected the voxel and NaN outside the
    # brain mask; treat NaN as 0 so union logic is well-defined.
    r2s = np.nan_to_num(r2s)
    r2k = np.nan_to_num(r2k)
    union = (r2s > thresh) | (r2k > thresh)
    inter = (r2s > thresh) & (r2k > thresh)
    n_u, n_i = int(union.sum()), int(inter.sum())
    if n_u == 0:
        return None

    d = r2k[union] - r2s[union]
    res = {
        "subject": sub, "session": ses, "polarity": polarity,
        "threshold_r2_pct": thresh,
        "n_supra_spm": int((r2s > thresh).sum()),
        "n_supra_kay": int((r2k > thresh).sum()),
        "n_union": n_u, "n_intersection": n_i,
        "median_delta_r2": float(np.median(d)),
        "mean_delta_r2": float(np.mean(d)),
        "frac_kay_wins": float(np.mean(d > 0)),
        "r2_pearson_union": float(np.corrcoef(r2s[union], r2k[union])[0, 1]),
    }
    if n_i >= 10:
        res.update({
            "angle_circ_r_inter": circ_corr(spm["angle"][inter], kay["angle"][inter]),
            "ecc_pearson_inter": float(np.corrcoef(
                spm["eccentricity"][inter], kay["eccentricity"][inter])[0, 1]),
            "size_pearson_inter": float(np.corrcoef(
                spm["size"][inter], kay["size"][inter])[0, 1]),
            "median_ecc_spm": float(np.median(spm["eccentricity"][inter])),
            "median_ecc_kay": float(np.median(kay["eccentricity"][inter])),
            "median_size_spm": float(np.median(spm["size"][inter])),
            "median_size_kay": float(np.median(kay["size"][inter])),
        })
    return res


def sessions_present():
    for sub_dir in sorted(PRF_ROOT.glob("sub-*")):
        for ses_dir in sorted(sub_dir.glob("ses-*")):
            yield sub_dir.name[4:], ses_dir.name[4:]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject", help="bare label; default: every fitted subject")
    ap.add_argument("--session", help="bare label; default: every fitted session")
    ap.add_argument("--threshold", type=float, default=10.0,
                    help="R2%% defining supra-threshold voxels (default 10)")
    ap.add_argument("--json", help="write the full result list to this path")
    args = ap.parse_args()

    rows = []
    for sub, ses in sessions_present():
        if args.subject and sub != args.subject:
            continue
        if args.session and ses != args.session:
            continue
        for polarity in ARMS:
            r = compare_session(sub, ses, polarity, args.threshold)
            if r is not None:
                rows.append(r)

    if not rows:
        sys.exit(f"ERROR: no session has both HRF arms fitted under {PRF_ROOT} "
                 f"(looked for suffix pairs {list(ARMS.values())})")

    hdr = (f"{'sub':>4} {'ses':>4} {'pol':>4} {'n_spm':>7} {'n_kay':>7} "
           f"{'union':>7} {'medΔR2':>8} {'kay>spm':>8} {'ang_r':>6} {'ecc_r':>6}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['subject']:>4} {r['session']:>4} {r['polarity']:>4} "
              f"{r['n_supra_spm']:>7} {r['n_supra_kay']:>7} {r['n_union']:>7} "
              f"{r['median_delta_r2']:>8.2f} {r['frac_kay_wins']:>8.2f} "
              f"{r.get('angle_circ_r_inter', float('nan')):>6.3f} "
              f"{r.get('ecc_pearson_inter', float('nan')):>6.3f}")

    for polarity in ARMS:
        sel = [r for r in rows if r["polarity"] == polarity]
        if not sel:
            continue
        med = float(np.median([r["median_delta_r2"] for r in sel]))
        wins = float(np.mean([r["frac_kay_wins"] for r in sel]))
        ns = sum(r["n_supra_spm"] for r in sel)
        nk = sum(r["n_supra_kay"] for r in sel)
        print(f"\n[{polarity}] {len(sel)} sessions: median per-session ΔR² "
              f"(kay−spm) = {med:+.2f} pp on union voxels; mean frac kay>spm = "
              f"{wins:.2f}; supra-threshold totals spm {ns} vs kay {nk}")

    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2) + "\n")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
