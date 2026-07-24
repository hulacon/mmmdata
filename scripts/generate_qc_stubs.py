#!/usr/bin/env python
"""Auto-stub preprocessing_qc decisions for every raw BOLD run.

Writes one decision JSON per run at
``{bids_root}/derivatives/preprocessing_qc/sub-XX/{run_key}_decision.json``
(the canonical path read by the pipeline harness and written by the
interactive QC dashboard), so the pipeline gate is never blocked by a
missing file.

Every stub is recorded as ``pending`` with ``automated=True``: this script
is not a reviewer and must never produce something that reads as a human
judgement. Its FD-based opinion is stored as a *recommendation* — ``keep``,
or ``investigate`` above the motion threshold — which the dashboard shows
as an advisory note next to a PENDING badge.

Idempotent. Runs that already have a decision JSON are left untouched
so previous human-entered decisions are never overwritten.

Usage
-----
    python scripts/generate_qc_stubs.py --dry-run        # show what would change
    python scripts/generate_qc_stubs.py                  # write stubs
    python scripts/generate_qc_stubs.py --subject sub-03 # one subject
    python scripts/generate_qc_stubs.py --investigate-threshold 0.6
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Make the mmmdata src/python importable regardless of invocation cwd
REPO_PYTHON = Path(__file__).resolve().parent.parent / "src" / "python"
if str(REPO_PYTHON) not in sys.path:
    sys.path.insert(0, str(REPO_PYTHON))

from neuroimaging.constants import DEFAULT_BIDS_ROOT, DERIVATIVES_DIRS  # noqa: E402
from neuroimaging.qc import qc_settings  # noqa: E402
from neuroimaging.qc_dashboard import save_decision  # noqa: E402


AUTO_REVIEWER = "auto-stub"


def _compute_fd_metrics(
    confounds_tsv: Path, fd_threshold: float
) -> dict[str, float | None]:
    """Return mean_fd, max_fd, pct_high_motion from a confounds TSV."""
    if not confounds_tsv.exists():
        return {"mean_fd": None, "max_fd": None, "pct_high_motion": None}
    df = pd.read_csv(
        confounds_tsv, sep="\t",
        usecols=lambda c: c == "framewise_displacement",
    )
    if "framewise_displacement" not in df.columns:
        return {"mean_fd": None, "max_fd": None, "pct_high_motion": None}
    fd = pd.to_numeric(df["framewise_displacement"], errors="coerce")
    valid = fd.notna().sum()
    if valid == 0:
        return {"mean_fd": None, "max_fd": None, "pct_high_motion": None}
    n_high = int((fd > fd_threshold).sum())
    return {
        "mean_fd": float(fd.mean()),
        "max_fd": float(fd.max()),
        "pct_high_motion": round(100 * n_high / int(valid), 1),
    }


def _confounds_tsv_for(
    bids_root: Path, variant_key: str, bold_path: Path
) -> Path:
    """Map a raw BOLD path to its fMRIPrep confounds TSV."""
    name = bold_path.name.removesuffix(".nii.gz")  # e.g. sub-03_ses-04_task-TBencoding_run-01_bold
    stem = name.removesuffix("_bold")
    sub = bold_path.parent.parent.parent.name  # sub-XX
    ses = bold_path.parent.parent.name         # ses-YY
    return (
        bids_root / DERIVATIVES_DIRS[variant_key] / sub / ses / "func"
        / f"{stem}_desc-confounds_timeseries.tsv"
    )


def _parse_bids_entities(run_key: str) -> tuple[str, str, str, str | None]:
    """Extract (subject, session, task, run) from a run key.

    run_key example: sub-03_ses-04_task-TBencoding_run-01_bold
    """
    parts = dict(
        p.split("-", 1) for p in run_key.split("_") if "-" in p
    )
    return (
        parts["sub"],
        parts["ses"],
        parts["task"],
        parts.get("run"),
    )


def iter_bold_runs(bids_root: Path, subject_filter: str | None = None):
    """Yield raw BOLD paths across all sub-*/ses-*/func/ directories."""
    sub_pattern = subject_filter if subject_filter else "sub-*"
    for sub_dir in sorted(bids_root.glob(sub_pattern)):
        if not sub_dir.is_dir():
            continue
        for ses_dir in sorted(sub_dir.glob("ses-*")):
            func_dir = ses_dir / "func"
            if not func_dir.exists():
                continue
            yield from sorted(func_dir.glob("*_bold.nii.gz"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--bids-root", type=Path, default=DEFAULT_BIDS_ROOT,
        help="BIDS dataset root (default: %(default)s)",
    )
    ap.add_argument(
        "--subject", default=None,
        help="Process only this subject (e.g. sub-03). Default: all.",
    )
    ap.add_argument(
        "--variant", choices=["fmriprep", "fmriprep_nordic"],
        default="fmriprep_nordic",
        help="Which fMRIPrep variant's confounds to use for FD (default: %(default)s)",
    )
    ap.add_argument(
        "--fd-threshold", type=float, default=None,
        help=(
            "FD threshold (mm) for pct_high_motion "
            "(default: [qc].fd_threshold from config/base.toml)"
        ),
    )
    ap.add_argument(
        "--investigate-threshold", type=float, default=None,
        help=(
            "Mean FD threshold above which the stub recommends "
            "'investigate' instead of 'keep' (default: "
            "[qc].investigate_threshold from config/base.toml)"
        ),
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Report what would be written without modifying disk.",
    )
    args = ap.parse_args()

    settings = qc_settings()
    fd_threshold = (
        args.fd_threshold if args.fd_threshold is not None
        else settings["fd_threshold"]
    )
    investigate_threshold = (
        args.investigate_threshold if args.investigate_threshold is not None
        else settings["investigate_threshold"]
    )

    decisions_root = args.bids_root / DERIVATIVES_DIRS["preprocessing_qc"]

    n_total = 0
    n_existing = 0
    n_rec_keep = 0
    n_rec_investigate = 0
    n_missing_fd = 0

    for bold in iter_bold_runs(args.bids_root, args.subject):
        n_total += 1
        run_key = bold.name.removesuffix(".nii.gz")
        subject, session, task, run = _parse_bids_entities(run_key)
        json_path = decisions_root / f"sub-{subject}" / f"{run_key}_decision.json"

        if json_path.exists():
            n_existing += 1
            continue

        confounds_tsv = _confounds_tsv_for(args.bids_root, args.variant, bold)
        fd = _compute_fd_metrics(confounds_tsv, fd_threshold)
        if fd["mean_fd"] is None:
            n_missing_fd += 1
            reason_parts = ["auto-stub: FD unavailable (no framewise_displacement)"]
            recommendation = "keep"
        else:
            reason_parts = [
                f"auto-stub from {args.variant} confounds",
                f"mean_fd={fd['mean_fd']:.3f}mm",
                f"max_fd={fd['max_fd']:.3f}mm",
                f"pct_high_motion={fd['pct_high_motion']}% (>{fd_threshold}mm)",
            ]
            if fd["mean_fd"] > investigate_threshold:
                recommendation = "investigate"
                reason_parts.append(
                    f"mean_fd exceeds {investigate_threshold}mm — suggest review"
                )
            else:
                recommendation = "keep"

        reason_parts.append("awaiting human sign-off")
        reason = "; ".join(reason_parts)

        if args.dry_run:
            print(f"[DRY] pending (rec: {recommendation:<11s}) {run_key}  | {reason}")
        else:
            save_decision(
                decisions_dir=decisions_root,
                subject=subject,
                session=session,
                task=task,
                run=run,
                decision="pending",
                reason=reason,
                reviewer=AUTO_REVIEWER,
                suffix="bold",
                automated=True,
                recommendation=recommendation,
            )

        if recommendation == "investigate":
            n_rec_investigate += 1
        else:
            n_rec_keep += 1

    print()
    print("=" * 60)
    print("QC stub generation summary")
    print("=" * 60)
    print(f"Mode:                    {'DRY RUN' if args.dry_run else 'WRITTEN'}")
    print(f"Variant for FD:          {args.variant}")
    print(f"FD threshold:            {fd_threshold} mm")
    print(f"Investigate threshold:   {investigate_threshold} mm (mean_fd)")
    if args.subject:
        print(f"Subject filter:          {args.subject}")
    print()
    print(f"Total BOLD runs:         {n_total}")
    print(f"  already had decision:  {n_existing}")
    print(f"  stubbed pending:       {n_rec_keep + n_rec_investigate}")
    print(f"    recommend 'keep':        {n_rec_keep}")
    print(f"    recommend 'investigate': {n_rec_investigate}")
    print(f"  FD unavailable:        {n_missing_fd}")
    print()
    print("All stubs are 'pending' — none counts as a review.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
