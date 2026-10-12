#!/usr/bin/env python3
"""Candidate BOLD->anatomy alignments for one subject: how far each would move every run (transforms only).

Estimates arm B (unbiased EPI template + one bbregister) and arm C (NSD-style
session affines to a reference session + BSplineSyN to the T2w) from the
subject's fMRIPrep T1w boldrefs, and writes one row per run with the
displacement from the current per-run BBR. Library and method:
src/python/neuroimaging/alignment_candidates.py.

Needs ANTs and FreeSurfer on PATH, loaded before the venv:
    module load ants/2.5.2 freesurfer/8.2.0

Usage:
    python stage0_deltas.py --sub ## --reference-session ## --out <dir> --jobs 8 [--arms B,C] [--check-run <entity_prefix>]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src" / "python") not in sys.path:  # idempotent: tests import this module repeatedly
    sys.path.insert(0, str(REPO_ROOT / "src" / "python"))

from core.config import load_config  # noqa: E402
from neuroimaging import alignment_candidates as ac  # noqa: E402
from neuroimaging import data_quality as dq  # noqa: E402
from neuroimaging import data_quality_voxelmaps as dqv  # noqa: E402
from neuroimaging.constants import DEFAULT_VARIANT  # noqa: E402
from neuroimaging.io import find_fmriprep_runs  # noqa: E402


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sub", required=True, help="bare label, e.g. 04")
    ap.add_argument("--reference-session", required=True, help="arm C's reference session, bare label")
    ap.add_argument("--out", required=True, type=Path, help="output root (one sub-## directory per subject)")
    ap.add_argument("--t2w-acq", default="SPC", help="acq label of the whole-brain T2w (default %(default)s)")
    ap.add_argument("--jobs", type=int, default=1, help="registrations in parallel")
    ap.add_argument("--arms", default="B,C", help="comma-separated subset of B,C (default %(default)s)")
    ap.add_argument("--check-run", help="entity prefix of one run to bbregister as a convention check")
    ap.add_argument("--bids-root", help="override config paths.bids_project_dir")
    ap.add_argument("--variant", default=DEFAULT_VARIANT, help="fMRIPrep tree name (default %(default)s)")
    args = ap.parse_args(argv)

    bids_root = Path(args.bids_root or load_config()["paths"]["bids_project_dir"])
    fmriprep_tree = bids_root / "derivatives" / args.variant
    runs = find_fmriprep_runs(subject=args.sub, variant=args.variant, space=dqv.VOLUME_SPACE, bids_root=bids_root,
                              allow_mixed_designs=True)
    runs = [r for r in runs if r.boldref is not None and r.mask is not None]
    if not runs:
        sys.exit(f"sub-{args.sub}: no {dqv.VOLUME_SPACE} runs with boldref and mask under {fmriprep_tree}")
    t0 = time.time()
    ac.build_cell(args.out, fmriprep_tree, bids_root, args.sub, runs, args.reference_session,
                  {"fmriprep_version": dq.pipeline_version(fmriprep_tree), "code_version": dq.code_version(REPO_ROOT)},
                  t2w_acq=args.t2w_acq, n_jobs=args.jobs, check_run=args.check_run,
                  arms=tuple(a.strip() for a in args.arms.split(",") if a.strip()),
                  log=lambda m: print(m, flush=True))
    print(f"sub-{args.sub}: {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
