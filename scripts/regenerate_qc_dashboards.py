#!/usr/bin/env python
"""Regenerate QC dashboards against current derivatives.

Writes one HTML per (subject, modality) combination to
``derivatives/qc_review/``, plus an all-subjects HTML per modality.
Motion traces come from the NORDIC fMRIPrep variant (the one the
Layer 2 streams consume). Decisions are read from the canonical path
``derivatives/preprocessing_qc/`` so the dashboard surfaces the stubs
written by ``generate_qc_stubs.py`` and any later human decisions.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_PYTHON = Path(__file__).resolve().parent.parent / "src" / "python"
if str(REPO_PYTHON) not in sys.path:
    sys.path.insert(0, str(REPO_PYTHON))

from neuroimaging.constants import DEFAULT_BIDS_ROOT, DERIVATIVES_DIRS  # noqa: E402
from neuroimaging.qc_dashboard import generate_dashboard  # noqa: E402


SUBJECTS = ("sub-03", "sub-04", "sub-05")
MODALITIES = ("bold", "T1w", "T2w")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--bids-root", type=Path, default=DEFAULT_BIDS_ROOT,
    )
    ap.add_argument(
        "--out-dir", type=Path, default=None,
        help="Output directory (default: {bids_root}/derivatives/qc_review)",
    )
    ap.add_argument(
        "--variant", choices=["fmriprep", "fmriprep_nordic"],
        default="fmriprep_nordic",
        help="fMRIPrep variant for motion traces (default: %(default)s)",
    )
    ap.add_argument(
        "--subject", default=None,
        help="Regenerate only this subject (e.g. sub-03). Default: all.",
    )
    ap.add_argument(
        "--modality", default=None, choices=list(MODALITIES),
        help="Regenerate only this modality. Default: all.",
    )
    args = ap.parse_args()

    mriqc_dir = args.bids_root / DERIVATIVES_DIRS["mriqc"]
    fmriprep_dir = args.bids_root / DERIVATIVES_DIRS[args.variant]
    decisions_dir = args.bids_root / DERIVATIVES_DIRS["preprocessing_qc"]
    out_dir = args.out_dir or (args.bids_root / "derivatives" / "qc_review")
    out_dir.mkdir(parents=True, exist_ok=True)

    subjects = (args.subject,) if args.subject else SUBJECTS
    modalities = (args.modality,) if args.modality else MODALITIES

    targets: list[tuple[str | None, str]] = []
    for mod in modalities:
        targets.append((None, mod))                    # all-subjects combined
        for sub in subjects:
            targets.append((sub, mod))

    print(f"Regenerating {len(targets)} dashboards -> {out_dir}")
    print(f"  MRIQC:     {mriqc_dir}")
    print(f"  fMRIPrep:  {fmriprep_dir}")
    print(f"  decisions: {decisions_dir}")
    print()

    for sub, mod in targets:
        subject_id = sub.replace("sub-", "") if sub else None
        label = sub if sub else "all"
        save_path = out_dir / f"qc_dashboard_{label}_{mod}.html"
        html_path = generate_dashboard(
            mriqc_dir=mriqc_dir,
            fmriprep_dir=fmriprep_dir,
            decisions_dir=decisions_dir,
            subject=subject_id,
            modality=mod,
            save_path=save_path,
            bids_root=args.bids_root,
        )
        size_mb = save_path.stat().st_size / (1024 * 1024)
        print(f"  {label:<8s} {mod:<4s}  {size_mb:.1f} MB  {html_path}")

    print()
    print(f"Done. Open e.g. {out_dir}/qc_dashboard_all_bold.html")
    return 0


if __name__ == "__main__":
    sys.exit(main())
