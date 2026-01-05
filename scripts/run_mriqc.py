#!/usr/bin/env python3
"""
run_mriqc.py - Run MRIQC on MMMData BIDS dataset

This script runs MRIQC quality control on all subjects and sessions
using Singularity/Apptainer containerization.

Usage:
    python run_mriqc.py [--analysis-level participant|group] [--subjects sub-01 sub-02]

Arguments:
    --analysis-level    : participant (default) or group
    --subjects          : Specific subjects to process (optional, default: all)
    --nprocs            : Number of parallel processes (default: 4)
    --mem-gb            : Memory limit in GB (default: 16)
    --fs-license        : Path to FreeSurfer license file (optional)
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src' / 'python'))

from core.config import load_config


def run_mriqc(
    bids_dir,
    output_dir,
    analysis_level='participant',
    subjects=None,
    nprocs=4,
    mem_gb=16,
    fs_license=None,
    mriqc_version='24.0.0'
):
    """
    Run MRIQC using Singularity

    Parameters
    ----------
    bids_dir : str or Path
        Path to BIDS dataset
    output_dir : str or Path
        Path to output directory (will be created if doesn't exist)
    analysis_level : str
        'participant' or 'group'
    subjects : list of str, optional
        List of subject IDs to process (e.g., ['sub-01', 'sub-02'])
        If None, processes all subjects
    nprocs : int
        Number of parallel processes
    mem_gb : int
        Memory limit in GB
    fs_license : str or Path, optional
        Path to FreeSurfer license file
    mriqc_version : str
        MRIQC version to use
    """

    bids_dir = Path(bids_dir)
    output_dir = Path(output_dir)
    work_dir = output_dir / 'work'
    singularity_dir = bids_dir / 'singularity_images'
    mriqc_image = singularity_dir / f'mriqc-{mriqc_version}.simg'

    # Create directories
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    singularity_dir.mkdir(parents=True, exist_ok=True)

    # Download MRIQC image if needed
    if not mriqc_image.exists():
        print(f"MRIQC Singularity image not found. Downloading version {mriqc_version}...")
        print("This may take several minutes...")

        pull_cmd = [
            'singularity', 'pull',
            str(mriqc_image),
            f'docker://nipreps/mriqc:{mriqc_version}'
        ]

        try:
            subprocess.run(pull_cmd, check=True)
            print("Download complete!")
        except subprocess.CalledProcessError as e:
            print(f"ERROR: Failed to download MRIQC image")
            print(f"Command: {' '.join(pull_cmd)}")
            sys.exit(1)

    # Build singularity command
    cmd = [
        'singularity', 'run', '--cleanenv',
        '-B', f'{bids_dir}:{bids_dir}:ro',
        '-B', f'{output_dir}:{output_dir}',
        '-B', f'{work_dir}:{work_dir}',
        str(mriqc_image),
        str(bids_dir),
        str(output_dir),
        analysis_level,
        '--work-dir', str(work_dir),
        '--nprocs', str(nprocs),
        '--mem', str(mem_gb),
        '--verbose-reports',
        '--no-sub'
    ]

    # Add subject filter if specified
    if subjects and analysis_level == 'participant':
        for subj in subjects:
            # Remove 'sub-' prefix if present for consistency
            subj_id = subj.replace('sub-', '')
            cmd.extend(['--participant-label', subj_id])

    # Add FreeSurfer license if provided
    if fs_license:
        fs_license = Path(fs_license)
        if fs_license.exists():
            cmd.extend(['--fs-license-file', str(fs_license)])
        else:
            print(f"WARNING: FreeSurfer license not found at {fs_license}")

    # Print summary
    print("=" * 60)
    print(f"Running MRIQC {mriqc_version}")
    print("=" * 60)
    print(f"BIDS Directory: {bids_dir}")
    print(f"Output Directory: {output_dir}")
    print(f"Analysis Level: {analysis_level}")
    if subjects:
        print(f"Subjects: {', '.join(subjects)}")
    else:
        print("Subjects: ALL")
    print(f"Processes: {nprocs}")
    print(f"Memory: {mem_gb} GB")
    print("=" * 60)
    print()

    # Run MRIQC
    try:
        subprocess.run(cmd, check=True)

        print()
        print("=" * 60)
        print("MRIQC completed successfully!")
        print("=" * 60)
        print(f"Results are in: {output_dir}")
        print()
        if analysis_level == 'participant':
            print("Next steps:")
            print(f"  - Review individual reports in: {output_dir}")
            print("  - Run group analysis: python run_mriqc.py --analysis-level group")
        else:
            print(f"Review group report: {output_dir}/group_*.html")
        print("=" * 60)

    except subprocess.CalledProcessError as e:
        print()
        print("=" * 60)
        print("ERROR: MRIQC failed")
        print("=" * 60)
        print("Check the logs above for error messages")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Run MRIQC on MMMData BIDS dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--analysis-level',
        choices=['participant', 'group'],
        default='participant',
        help='Analysis level (default: participant)'
    )

    parser.add_argument(
        '--subjects',
        nargs='+',
        help='Specific subjects to process (e.g., sub-01 sub-02). Default: all subjects'
    )

    parser.add_argument(
        '--nprocs',
        type=int,
        default=4,
        help='Number of parallel processes (default: 4)'
    )

    parser.add_argument(
        '--mem-gb',
        type=int,
        default=16,
        help='Memory limit in GB (default: 16)'
    )

    parser.add_argument(
        '--fs-license',
        help='Path to FreeSurfer license file (optional)'
    )

    parser.add_argument(
        '--mriqc-version',
        default='24.0.0',
        help='MRIQC version to use (default: 24.0.0)'
    )

    parser.add_argument(
        '--bids-dir',
        help='Override BIDS directory from config'
    )

    parser.add_argument(
        '--output-dir',
        help='Override output directory (default: <bids_dir>/derivatives/mriqc)'
    )

    args = parser.parse_args()

    # Load config
    config = load_config()

    # Get BIDS directory
    bids_dir = args.bids_dir or config.get('bids_project_dir')
    if not bids_dir:
        print("ERROR: BIDS directory not specified and not found in config")
        sys.exit(1)

    bids_dir = Path(bids_dir)

    # Get output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = bids_dir / 'derivatives' / 'mriqc'

    # Run MRIQC
    run_mriqc(
        bids_dir=bids_dir,
        output_dir=output_dir,
        analysis_level=args.analysis_level,
        subjects=args.subjects,
        nprocs=args.nprocs,
        mem_gb=args.mem_gb,
        fs_license=args.fs_license,
        mriqc_version=args.mriqc_version
    )


if __name__ == '__main__':
    main()
