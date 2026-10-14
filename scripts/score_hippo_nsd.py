#!/usr/bin/env python3
"""Score hippocampal segmentations (HippUnfold 1.5.2/2.2.0, HSF, FreeSurfer) against NSD's manual MTL labels.

Per subject: the T1-space -> HRT2 rigid transform is recovered, every arm is put on the HRT2 grid (primary),
NSD's 0.8 mm T1 grid (cross-check) and eight phases of a 1.8 x 1.8 x 2 mm grid, and scored for both scoring
variants (``primary``, ``alt``). ``--group`` concatenates the subject tables. Library and rules:
src/python/neuroimaging/hippo_nsd.py.

NSD data stay outside this dataset: ``--ppdata`` is NSD's ``nsddata/ppdata``; ``--eval-root`` holds the arm
trees under ``derivatives/`` and receives ``scores/``.

Needs ANTs on PATH, loaded before the venv:
    module load ants/2.5.2

Usage:
    python score_hippo_nsd.py --sub ## --ppdata <dir> --eval-root <dir> [--jobs 8]
    python score_hippo_nsd.py --group --eval-root <dir>
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src" / "python") not in sys.path:  # idempotent: tests import this module repeatedly
    sys.path.insert(0, str(REPO_ROOT / "src" / "python"))

from neuroimaging import data_quality as dq  # noqa: E402
from neuroimaging import hippo_nsd as hn  # noqa: E402


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    who = ap.add_mutually_exclusive_group(required=True)
    who.add_argument("--sub", help="bare two-digit subject label (NSD subj0N -> 0N)")
    who.add_argument("--group", action="store_true", help="concatenate every subject's tables")
    ap.add_argument("--eval-root", type=Path, required=True, help="holds derivatives/<arm trees>; scores/ is written here")
    ap.add_argument("--ppdata", type=Path, help="NSD nsddata/ppdata (required with --sub)")
    ap.add_argument("--xfm-dir", type=Path, help="validated sub-##_from-T1w_to-HRT2_mode-image_xfm.mat files to "
                    "reuse (default: <eval-root>/xfm when present; otherwise each subject is registered)")
    ap.add_argument("--jobs", type=int, default=1, help="registrations/resamplings in parallel")
    ap.add_argument("--arms", nargs="+", choices=[a.name for a in hn.ARMS],
                    help="score only these arms (default: all)")
    args = ap.parse_args(argv)

    out = args.eval_root / "scores"
    if args.group:
        tables = hn.collect(out)
        for name, df in tables.items():
            dst = out / f"group_desc-{name}_stats.tsv"
            df.to_csv(dst, sep="\t", index=False, float_format="%.6g")
            print(f"{dst}: {len(df)} rows, {df['subject'].nunique()} subjects")
        return
    if args.ppdata is None:
        ap.error("--sub needs --ppdata")
    provenance = {"code_version": dq.code_version(REPO_ROOT)}
    t0 = time.time()
    arms = [a for a in hn.ARMS if not args.arms or a.name in args.arms]
    xfm_dir = args.xfm_dir or args.eval_root / "xfm"
    hn.build_subject(out, args.eval_root / "derivatives", args.ppdata, args.sub, provenance, arms=arms,
                     n_jobs=args.jobs, log=lambda m: print(m, flush=True),
                     xfm_from=xfm_dir / f"sub-{args.sub}_from-T1w_to-HRT2_mode-image_xfm.mat")
    print(f"sub-{args.sub}: {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
