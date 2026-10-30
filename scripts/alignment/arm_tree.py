#!/usr/bin/env python3
"""An fMRIPrep derivatives tree for one alignment arm: production, with only the runs' coreg xfm replaced.

Symlinks the subject's production anatomy and the chosen sessions, and
replaces each chosen-task run's ``from-boldref_to-T1w`` xfm with the arm's,
composed from the stage 0 transforms (``stage0_deltas.py``). Arm ``A`` replaces
nothing: a control tree that must reproduce production exactly. Pass the tree
to ``fmriprep_t1w_backfill.sbatch`` as ``DERIV_TREE``: fMRIPrep then reuses
every fit (head motion, fieldmaps, anatomy) and resamples once with the arm's
placement. Library: src/python/neuroimaging/alignment_candidates.py
(``build_arm_tree``).

Writes beside the tree (not in it)::

    <tree>/../sub-##_arm-<arm>_desc-armtree.tsv    one row per run: displacement from A, stage 0's value
    <tree>/../sub-##_arm-<arm>_desc-armtree.json   provenance

Usage:
    python arm_tree.py --sub ## --arm B --stage0 <stage0 out root> --sessions 04 05 --task TBencoding --out <tree>
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src" / "python") not in sys.path:  # idempotent: tests import this module repeatedly
    sys.path.insert(0, str(REPO_ROOT / "src" / "python"))

from core.config import load_config  # noqa: E402
from neuroimaging import alignment_candidates as ac  # noqa: E402
from neuroimaging import data_quality as dq  # noqa: E402
from neuroimaging.constants import DEFAULT_VARIANT  # noqa: E402


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sub", required=True, help="bare label, e.g. 04")
    ap.add_argument("--arm", required=True, choices=("A", "B"), help="B, or A for a control tree")
    ap.add_argument("--stage0", required=True, type=Path, help="stage0_deltas.py output root holding arm B")
    ap.add_argument("--sessions", required=True, nargs="+", help="bare labels")
    ap.add_argument("--task", required=True, action="append", help="task whose runs get the arm's xfm (repeatable)")
    ap.add_argument("--out", required=True, type=Path, help="the arm tree (an fMRIPrep derivatives root)")
    ap.add_argument("--bids-root", help="override config paths.bids_project_dir")
    ap.add_argument("--variant", default=DEFAULT_VARIANT, help="production fMRIPrep tree name (default %(default)s)")
    args = ap.parse_args(argv)

    fmriprep_tree = Path(args.bids_root or load_config()["paths"]["bids_project_dir"]) / "derivatives" / args.variant
    table = ac.build_arm_tree(fmriprep_tree, args.stage0, args.sub, args.arm, args.sessions, args.task, args.out,
                              log=lambda m: print(m, flush=True))
    stem = args.out.parent / f"sub-{args.sub}_arm-{args.arm}_desc-armtree"
    table.to_csv(f"{stem}.tsv", sep="\t", index=False, float_format="%.4f")
    Path(f"{stem}.json").write_text(json.dumps({
        "subject": args.sub, "arm": args.arm, "sessions": args.sessions, "tasks": args.task, "tree": str(args.out),
        "production": str(fmriprep_tree), "stage0": str(args.stage0), "n_runs": len(table),
        "tolerance_mm": ac.ARM_TREE_TOL_MM, "fmriprep_version": dq.pipeline_version(fmriprep_tree),
        "code_version": dq.code_version(REPO_ROOT),
        "created": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")}, indent=2) + "\n")


if __name__ == "__main__":
    main()
