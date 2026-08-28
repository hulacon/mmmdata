#!/usr/bin/env python3
"""
prf_test_retest.py — session-to-session reliability of projected pRF maps.

Stage 2 of the projection -> test-retest leg (workbench prf-retinotopy,
HANDOFF 2026-08-27; closes the charter's Settles-when 3). Compares the two
pRF sessions of each subject on the shared fsnative surface — the surface is
the same mesh across sessions, so vertices correspond exactly and no
resampling is involved.

Criterion, PRE-REGISTERED 2026-08-27 before any number existed:
  include vertices with R2 > 10% in BOTH sessions; report
    - polar angle : circular correlation (Fisher-Lee, the same estimator as
                    fit_prf.py's self-test)
    - eccentricity: Pearson r
    - size        : Pearson r
per subject x hemisphere x polarity.

Prints the table and writes it as JSON beside the maps
(derivatives/prf/test-retest.json). The numbers themselves are findings and
belong in the workbench log, not in this script.

Usage:
    python prf_test_retest.py                # all subjects found on disk
    python prf_test_retest.py --subjects 03 04
"""

import argparse
import json
import sys
from collections import defaultdict
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

R2_THRESHOLD = 10.0
HEMIS = ("L", "R")
POLARITIES = ("prf", "negprf")


def circ_corr(a_deg, b_deg):
    """Fisher-Lee circular correlation, matching fit_prf.py's self-test."""
    a, b = np.radians(a_deg), np.radians(b_deg)
    a1 = a - np.angle(np.exp(1j * a).mean())
    b1 = b - np.angle(np.exp(1j * b).mean())
    return float((np.sin(a1) * np.sin(b1)).sum() /
                 np.sqrt((np.sin(a1) ** 2).sum() * (np.sin(b1) ** 2).sum()))


def load_shape(subject, session, hemi, param, polarity):
    import nibabel as nib
    p = (DERIV_ROOT / "prf" / f"sub-{subject}" / f"ses-{session}"
         / f"sub-{subject}_ses-{session}_task-prf_space-fsnative"
           f"_hemi-{hemi}_desc-{param}_{polarity}.shape.gii")
    if not p.exists():
        sys.exit(f"ERROR: missing projected map {p}\n"
                 "       Run project_prf_fsnative.py first.")
    return nib.load(str(p)).darrays[0].data


def discover_subjects():
    subs = defaultdict(set)
    for sc in sorted((DERIV_ROOT / "prf").glob(
            "sub-*/ses-*/sub-*_ses-*_task-prf_space-fsnative_prf.json")):
        subs[sc.name.split("_")[0].split("-")[1]].add(
            sc.name.split("_")[1].split("-")[1])
    pairs = {s: sorted(ss) for s, ss in subs.items() if len(ss) == 2}
    if not pairs:
        sys.exit(f"ERROR: no subject with two projected pRF sessions under "
                 f"{DERIV_ROOT / 'prf'}")
    for s, ss in sorted(subs.items()):
        if len(ss) != 2:
            print(f"  NOTE: sub-{s} has {len(ss)} projected session(s) "
                  f"({sorted(ss)}); skipped — test-retest needs exactly 2")
    return pairs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subjects", nargs="+", help="bare labels; default: all found")
    args = ap.parse_args()

    pairs = discover_subjects()
    if args.subjects:
        missing = [s for s in args.subjects if s not in pairs]
        if missing:
            sys.exit(f"ERROR: no projected session pair for subjects {missing} "
                     f"(have {sorted(pairs)})")
        pairs = {s: pairs[s] for s in args.subjects}

    results = []
    hdr = (f"{'subject':>8} {'pol':>7} {'hemi':>4} {'n_vert':>7} "
           f"{'angle_circ_r':>12} {'ecc_r':>7} {'size_r':>7}")
    print(f"\nInclusion: R2 > {R2_THRESHOLD}% in BOTH sessions (pre-registered)")
    print(hdr)
    print("-" * len(hdr))
    for sub, (ses_a, ses_b) in sorted(pairs.items()):
        for pol in POLARITIES:
            for hemi in HEMIS:
                r2a = load_shape(sub, ses_a, hemi, "R2", pol)
                r2b = load_shape(sub, ses_b, hemi, "R2", pol)
                keep = ((r2a > R2_THRESHOLD) & (r2b > R2_THRESHOLD))
                for param in ("angle", "eccentricity", "size"):
                    va = load_shape(sub, ses_a, hemi, param, pol)
                    vb = load_shape(sub, ses_b, hemi, param, pol)
                    keep &= np.isfinite(va) & np.isfinite(vb)
                n = int(keep.sum())
                row = {"subject": f"sub-{sub}", "sessions": [ses_a, ses_b],
                       "polarity": pol, "hemi": hemi, "n_vertices": n}
                if n < 10:
                    row.update(angle_circ_r=None, ecc_r=None, size_r=None)
                    print(f"{row['subject']:>8} {pol:>7} {hemi:>4} {n:>7} "
                          f"{'—':>12} {'—':>7} {'—':>7}   (too few vertices)")
                else:
                    ang = circ_corr(
                        load_shape(sub, ses_a, hemi, "angle", pol)[keep],
                        load_shape(sub, ses_b, hemi, "angle", pol)[keep])
                    ecc = float(np.corrcoef(
                        load_shape(sub, ses_a, hemi, "eccentricity", pol)[keep],
                        load_shape(sub, ses_b, hemi, "eccentricity", pol)[keep])[0, 1])
                    size = float(np.corrcoef(
                        load_shape(sub, ses_a, hemi, "size", pol)[keep],
                        load_shape(sub, ses_b, hemi, "size", pol)[keep])[0, 1])
                    row.update(angle_circ_r=round(ang, 4), ecc_r=round(ecc, 4),
                               size_r=round(size, 4))
                    print(f"{row['subject']:>8} {pol:>7} {hemi:>4} {n:>7} "
                          f"{ang:>12.3f} {ecc:>7.3f} {size:>7.3f}")
                results.append(row)

    out = DERIV_ROOT / "prf" / "test-retest.json"
    out.write_text(json.dumps({
        "Description": ("Session-to-session reliability of projected pRF "
                        "parameters on the shared fsnative surface."),
        "Criterion": (f"vertices with R2 > {R2_THRESHOLD}% in both sessions "
                      "(pre-registered 2026-08-27); angle = Fisher-Lee "
                      "circular correlation, eccentricity/size = Pearson r"),
        "Provenance": "mmmdata/scripts/prf_test_retest.py; workbench prf-retinotopy",
        "Results": results,
    }, indent=2) + "\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
