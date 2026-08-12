#!/usr/bin/env python3
"""Submit dcm2bids DICOM-to-BIDS conversion jobs to SLURM.

Python counterpart to submit_dcm2bids.sh: assembles the sbatch invocation
for dcm2bids.sbatch (--export of SUBJECT/SESSION/REPO_ROOT plus a per-job
name) so the command assembly lives in one place. Importable, too:
`build_sbatch_command()` returns the argv list, `submit()` runs (or, with
dry_run=True, just previews) a single submission.

Usage:
    python scripts/submit_dcm2bids.py sub-03 ses-06        # Single session
    python scripts/submit_dcm2bids.py sub-03 all           # All sessions
    python scripts/submit_dcm2bids.py sub-03 ses-04 ses-30 # Multiple sessions
    python scripts/submit_dcm2bids.py sub-03 ses-06 --dry-run
"""

import argparse
import subprocess
import sys
from pathlib import Path


def _find_repo_root() -> Path:
    """Find the mmmdata repo root (parent of this scripts/ dir)."""
    return Path(__file__).resolve().parent.parent


def build_sbatch_command(
    subject: str,
    session: str,
    repo_root: Path | None = None,
) -> list[str]:
    """Assemble the sbatch command for one subject/session pair.

    Mirrors submit_dcm2bids.sh: dcm2bids.sbatch requires SUBJECT, SESSION,
    and REPO_ROOT in its environment, so all three are exported.
    """
    root = (repo_root or _find_repo_root()).resolve()
    return [
        "sbatch",
        f"--export=ALL,SUBJECT={subject},SESSION={session},REPO_ROOT={root}",
        f"--job-name=dcm2bids_{subject}_{session}",
        str(root / "scripts" / "dcm2bids.sbatch"),
    ]


def submit(
    subject: str,
    session: str,
    *,
    dry_run: bool = False,
    repo_root: Path | None = None,
) -> int:
    """Submit (or, with dry_run=True, preview) one dcm2bids SLURM job.

    Prints the assembled sbatch command; returns sbatch's exit code
    (0 for a dry run).
    """
    root = (repo_root or _find_repo_root()).resolve()
    sbatch_file = root / "scripts" / "dcm2bids.sbatch"
    if not sbatch_file.exists():
        print(f"ERROR: sbatch file not found: {sbatch_file}", file=sys.stderr)
        return 1

    cmd = build_sbatch_command(subject, session, root)
    print(f"Command: {' '.join(cmd)}")
    if dry_run:
        print("  (dry-run: not submitted)")
        return 0

    # Ensure logs directory exists (dcm2bids.sbatch writes logs there)
    (root / "logs").mkdir(exist_ok=True)

    result = subprocess.run(cmd, cwd=str(root / "scripts"))
    return result.returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Submit dcm2bids conversion jobs to SLURM.",
    )
    parser.add_argument(
        "subject",
        help="Subject ID (e.g. sub-03)",
    )
    parser.add_argument(
        "sessions", nargs="+",
        help="Session ID(s) (e.g. ses-06), or 'all' for all sessions",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the assembled sbatch command(s) without submitting",
    )
    args = parser.parse_args(argv)

    errors = 0
    for session in args.sessions:
        print(f"Submitting: {args.subject} / {session}")
        rc = submit(args.subject, session, dry_run=args.dry_run)
        if rc != 0:
            print(
                f"  [!] {args.subject}/{session}: submission failed (exit {rc})",
                file=sys.stderr,
            )
            errors += 1

    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
