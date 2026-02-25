#!/usr/bin/env python3
"""Run dcm2bids config generation and DICOM-to-BIDS conversion.

Two-stage script:
  1. Generate dcm2bids config JSON via the CLI module (always runs)
  2. Run dcm2bids conversion (unless --generate-only)

Must be run from the mmmdata repo root (or the SLURM sbatch script handles
this automatically via load_config.sh).

Usage:
    python -m scripts.run_dcm2bids --subject sub-03 --session ses-06
    python scripts/run_dcm2bids.py --subject sub-03 --session all --generate-only
    python scripts/run_dcm2bids.py --subject sub-03 --session ses-06 --dry-run
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _find_python() -> str:
    """Return the Python interpreter path."""
    return sys.executable


def _find_repo_root() -> Path:
    """Find the mmmdata repo root (parent of this scripts/ dir)."""
    return Path(__file__).resolve().parent.parent


def _load_config(repo_root: Path) -> dict:
    """Load config by invoking the config loader as subprocess."""
    code = (
        "import sys, json; "
        "sys.path.insert(0, str(sys.argv[1])); "
        "from core.config import load_config; "
        "print(json.dumps(load_config()))"
    )
    result = subprocess.run(
        [_find_python(), "-c", code, str(repo_root / "src" / "python")],
        capture_output=True, text=True, timeout=10,
    )
    if result.returncode != 0:
        print(f"ERROR loading config: {result.stderr}", file=sys.stderr)
        sys.exit(1)
    return json.loads(result.stdout)


def generate_configs(
    python: str,
    repo_root: Path,
    subject: str,
    session: str,
    config_dir: Path,
    *,
    dry_run: bool = False,
    force: bool = False,
) -> subprocess.CompletedProcess:
    """Run config generation via the dcm2bids_config CLI module."""
    cmd = [
        python, "-m", "src.python.dcm2bids_config.cli",
        "--subject", subject,
        "--session", session,
        "--config-dir", str(config_dir),
    ]
    if dry_run:
        cmd.append("--dry-run")
    if force:
        cmd.append("--force")

    return subprocess.run(cmd, cwd=str(repo_root))


def run_dcm2bids(
    subject: str,
    session: str,
    bids_root: Path,
    config_dir: Path,
    dicom_dir: Path,
    container: Path,
    *,
    force: bool = False,
) -> int:
    """Run dcm2bids for a single subject/session via Singularity.

    Returns the dcm2bids exit code.
    """
    config_path = config_dir / subject / f"{session}_conf.json"
    if not config_path.exists():
        print(f"  [!] Config not found: {config_path}")
        return 1

    if not container.exists():
        print(f"  [!] Singularity container not found: {container}")
        return 1

    cmd = [
        "singularity", "exec",
        "--cleanenv",
        "-B", f"{bids_root}:{bids_root}",
        "-B", f"{config_dir}:{config_dir}",
        str(container),
        "dcm2bids",
        "-d", str(dicom_dir),
        "-p", subject.replace("sub-", ""),
        "-s", session.replace("ses-", ""),
        "-c", str(config_path),
        "-o", str(bids_root),
    ]
    if force:
        cmd.append("--force_dcm2bids")

    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate dcm2bids configs and run DICOM-to-BIDS conversion.",
    )
    parser.add_argument(
        "--subject", required=True,
        help="Subject ID (e.g. sub-03)",
    )
    parser.add_argument(
        "--session", required=True,
        help="Session ID (e.g. ses-06) or 'all' for all sessions",
    )
    parser.add_argument(
        "--generate-only", action="store_true",
        help="Generate config files only; do not run dcm2bids",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview config generation without writing files",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing config files and force dcm2bids reconversion",
    )
    parser.add_argument(
        "--config-dir",
        help="Override config output directory",
    )
    args = parser.parse_args(argv)

    repo_root = _find_repo_root()
    python = _find_python()
    cfg = _load_config(repo_root)

    bids_root = Path(cfg["paths"]["bids_project_dir"])
    code_root = Path(cfg["paths"]["code_root"])
    singularity_dir = Path(cfg["paths"]["singularity_dir"])
    container = singularity_dir / "dcm2bids-3.2.0.sif"

    if args.config_dir:
        config_dir = Path(args.config_dir)
    else:
        config_dir = code_root / "config" / "dcm2bids_overrides"

    # Stage 1: Generate configs
    print("=" * 60)
    print("Stage 1: Config generation")
    print("=" * 60)

    result = generate_configs(
        python, repo_root,
        args.subject, args.session, config_dir,
        dry_run=args.dry_run, force=args.force,
    )

    if result.returncode != 0:
        print("\nConfig generation failed")
        return 1

    if args.dry_run or args.generate_only:
        return 0

    # Stage 2: Run dcm2bids
    print()
    print("=" * 60)
    print("Stage 2: dcm2bids conversion")
    print("=" * 60)

    # Determine which sessions to convert
    if args.session == "all":
        # Find all generated config files for this subject
        config_subject_dir = config_dir / args.subject
        if not config_subject_dir.exists():
            print(f"  No config dir found: {config_subject_dir}")
            return 1
        config_files = sorted(config_subject_dir.glob("ses-*_conf.json"))
        sessions = [f.stem.replace("_conf", "") for f in config_files]
    else:
        sessions = [args.session]

    if not sessions:
        print("  No sessions to convert")
        return 0

    errors = 0
    for session in sessions:
        dicom_dir = bids_root / "sourcedata" / args.subject / session / "dicom"
        if not dicom_dir.exists():
            print(f"  [!] {args.subject}/{session}: DICOM dir not found: {dicom_dir}")
            errors += 1
            continue

        rc = run_dcm2bids(
            args.subject, session, bids_root, config_dir, dicom_dir,
            container, force=args.force,
        )
        if rc != 0:
            print(f"  [!] {args.subject}/{session}: dcm2bids failed (exit {rc})")
            errors += 1
        else:
            print(f"  [+] {args.subject}/{session}: conversion complete")

    print()
    print("=" * 60)
    if errors:
        print(f"Completed with {errors} error(s)")
    else:
        print(f"All {len(sessions)} session(s) converted successfully")
    print("=" * 60)

    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
