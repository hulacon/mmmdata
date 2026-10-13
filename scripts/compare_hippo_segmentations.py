#!/usr/bin/env python3
"""Compare hippocampal segmentations (HippUnfold 1.5.2/2.2.0, HSF, FreeSurfer aseg) in fMRIPrep T1w space.

Per subject: every arm resampled once into a 0.3 mm box per hemisphere and onto
the BOLD grid; volumes, pairwise Dice, voxel-assignment agreement, a T2w
registration check, maps and a coronal montage. ``--group`` concatenates the
subject tables. Library and method: src/python/neuroimaging/hippo_compare.py.

Needs ANTs on PATH, loaded before the venv:
    module load ants/2.5.2

Usage:
    python compare_hippo_segmentations.py --sub ## [--out <dir>] [--jobs 8]
    python compare_hippo_segmentations.py --group [--out <dir>]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src" / "python") not in sys.path:  # idempotent: tests import this module repeatedly
    sys.path.insert(0, str(REPO_ROOT / "src" / "python"))

from core.config import load_config  # noqa: E402
from neuroimaging import data_quality as dq  # noqa: E402
from neuroimaging import hippo_compare as hc  # noqa: E402
from neuroimaging.constants import DEFAULT_VARIANT  # noqa: E402

OUT_TREE = "hippunfold_eval"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    who = ap.add_mutually_exclusive_group(required=True)
    who.add_argument("--sub", help="bare subject label, e.g. 03")
    who.add_argument("--group", action="store_true", help="concatenate every subject's tables under --out")
    ap.add_argument("--out", type=Path, help=f"output tree (default <derivatives>/{OUT_TREE})")
    ap.add_argument("--jobs", type=int, default=1, help="registrations/resamplings in parallel")
    ap.add_argument("--variant", default=DEFAULT_VARIANT, help="fMRIPrep tree name (default %(default)s)")
    args = ap.parse_args(argv)

    paths = load_config()["paths"]
    bids_root, deriv = Path(paths["bids_project_dir"]), Path(paths["output_dir"])
    out = args.out or deriv / OUT_TREE
    provenance = {"code_version": dq.code_version(REPO_ROOT)}

    if args.group:
        tables = hc.collect(out)
        for name, df in tables.items():
            dst = out / f"group_desc-{name}_stats.tsv"
            df.to_csv(dst, sep="\t", index=False, float_format="%.6g")
            print(f"{dst}: {len(df)} rows, {df['subject'].nunique()} subjects")
        return

    hc.ensure_dataset_description(out, provenance)
    t0 = time.time()
    hc.build_subject(out, bids_root, deriv, args.sub, provenance, fmriprep_variant=args.variant, n_jobs=args.jobs,
                     log=lambda m: print(m, flush=True))
    print(f"sub-{args.sub}: {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
