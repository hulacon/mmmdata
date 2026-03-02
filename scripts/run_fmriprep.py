#!/usr/bin/env python3
"""
run_fmriprep.py - Run fMRIPrep on MMMData BIDS dataset

This script runs fMRIPrep preprocessing on all subjects and sessions
using Singularity/Apptainer containerization.

Usage:
    python run_fmriprep.py [--subjects sub-03 sub-04] [--output-spaces ...]

Arguments:
    --subjects          : Specific subjects to process (optional, default: all)
    --nprocs            : Number of parallel processes (default: 8)
    --mem-gb            : Memory limit in GB (default: 32)
    --output-spaces     : fMRIPrep output spaces (default: MNI152NLin2009cAsym:res-2 fsaverage6)
    --fs-license        : Path to FreeSurfer license file (auto-detected if omitted)
    --fs-subjects-dir   : Reuse existing FreeSurfer recon-all output (optional)
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src' / 'python'))

from core.config import load_config

DEFAULT_OUTPUT_SPACES = ['MNI152NLin2009cAsym:res-2', 'fsaverage6']
DEFAULT_FMRIPREP_VERSION = '24.1.1'


def find_fs_license(bids_dir):
    """Auto-detect FreeSurfer license file in the BIDS dataset."""
    candidates = [
        bids_dir / 'licenses' / 'fz_unlabeled_license.txt',
        bids_dir / 'license.txt',
        bids_dir / '.license',
        Path.home() / '.freesurfer_license.txt',
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def run_fmriprep(
    bids_dir,
    output_dir,
    subjects=None,
    nprocs=8,
    mem_gb=32,
    output_spaces=None,
    fs_license=None,
    fs_subjects_dir=None,
    fmriprep_version=DEFAULT_FMRIPREP_VERSION,
    singularity_dir=None,
    work_dir=None,
    anat_only=False,
    session=None,
    derivatives=None,
):
    """
    Run fMRIPrep using Singularity.

    Parameters
    ----------
    bids_dir : str or Path
        Path to BIDS dataset.
    output_dir : str or Path
        Path to output directory (will be created if doesn't exist).
    subjects : list of str, optional
        List of subject IDs to process (e.g., ['sub-03', 'sub-04']).
        If None, processes all subjects.
    nprocs : int
        Number of parallel processes.
    mem_gb : int
        Memory limit in GB.
    output_spaces : list of str, optional
        fMRIPrep output spaces. Default: MNI152NLin2009cAsym:res-2 fsaverage6.
    fs_license : str or Path, optional
        Path to FreeSurfer license file. Auto-detected if not provided.
    fs_subjects_dir : str or Path, optional
        Path to existing FreeSurfer subjects directory for reuse.
    fmriprep_version : str
        fMRIPrep version to use.
    singularity_dir : str or Path, optional
        Directory containing Singularity images.
    anat_only : bool
        If True, only run anatomical preprocessing (no functional).
    session : str, optional
        Process only this session (e.g., 'ses-01' or '01'). Creates a
        BIDS filter file to restrict functional processing to one session.
        When session is specified with a single subject, the work directory
        is isolated per-session to prevent race conditions between concurrent jobs.
    derivatives : str or Path, optional
        Path to pre-computed derivatives (e.g., fmriprep anat output).
        Passed to fMRIPrep's --derivatives flag to skip reprocessing.
    """
    bids_dir = Path(bids_dir)
    output_dir = Path(output_dir)
    if work_dir is None:
        work_dir = output_dir / 'work'
    else:
        work_dir = Path(work_dir) / 'fmriprep'

    # Isolate work dir per-session to prevent race conditions when multiple
    # session jobs run concurrently for the same subject
    if session and subjects and len(subjects) == 1:
        ses_id = session.replace('ses-', '')
        subj_id = subjects[0].replace('sub-', '')
        work_dir = work_dir / f'sub-{subj_id}_ses-{ses_id}'

    if singularity_dir is None:
        singularity_dir = bids_dir / 'singularity_images'
    else:
        singularity_dir = Path(singularity_dir)

    fmriprep_image = singularity_dir / f'fmriprep-{fmriprep_version}.simg'
    if output_spaces is None:
        output_spaces = DEFAULT_OUTPUT_SPACES

    # Create directories
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    # Validate container image
    if not fmriprep_image.exists():
        print(f"ERROR: fMRIPrep Singularity image not found: {fmriprep_image}")
        print(f"Download it with:")
        print(f"  singularity pull {fmriprep_image} docker://nipreps/fmriprep:{fmriprep_version}")
        sys.exit(1)

    # Auto-detect FreeSurfer license if not provided
    if fs_license is None:
        fs_license = find_fs_license(bids_dir)
    else:
        fs_license = Path(fs_license)

    if fs_license is None or not fs_license.exists():
        print("ERROR: FreeSurfer license file not found.")
        print("Provide one with --fs-license or place it in <bids_dir>/licenses/")
        sys.exit(1)

    # Build singularity bind mounts
    binds = [
        '-B', f'{bids_dir}:{bids_dir}:ro',
        '-B', f'{output_dir}:{output_dir}',
        '-B', f'{work_dir}:{work_dir}',
        '-B', f'{fs_license.parent}:{fs_license.parent}:ro',
    ]

    if fs_subjects_dir:
        fs_subjects_dir = Path(fs_subjects_dir)
        if fs_subjects_dir.exists():
            binds.extend(['-B', f'{fs_subjects_dir}:{fs_subjects_dir}'])
        else:
            print(f"WARNING: FreeSurfer subjects dir not found: {fs_subjects_dir}")
            fs_subjects_dir = None

    # Bind-mount pre-computed derivatives directory if needed
    if derivatives:
        derivatives = Path(derivatives)
        if derivatives.exists():
            deriv_str = str(derivatives)
            if not deriv_str.startswith(str(output_dir)):
                binds.extend(['-B', f'{deriv_str}:{deriv_str}:ro'])
        else:
            print(f"WARNING: Derivatives dir not found: {derivatives}, skipping")
            derivatives = None

    # Build fMRIPrep command
    cmd = [
        'singularity', 'run', '--cleanenv',
        *binds,
        str(fmriprep_image),
        str(bids_dir),
        str(output_dir),
        'participant',
        '--work-dir', str(work_dir),
        '--output-spaces', *output_spaces,
        '--fs-license-file', str(fs_license),
        '--nprocs', str(nprocs),
        '--mem-mb', str(mem_gb * 1024),
        '--skip-bids-validation',
        '--notrack',
    ]

    # Add subject filter if specified
    if subjects:
        for subj in subjects:
            subj_id = subj.replace('sub-', '')
            cmd.extend(['--participant-label', subj_id])

    # Add anat-only flag
    if anat_only:
        cmd.append('--anat-only')

    # Add session filter via bids-filter-file
    if session:
        ses_id = session.replace('ses-', '')
        filter_file = work_dir / f'bids_filter_ses-{ses_id}.json'
        bids_filter = {
            "bold": {"session": ses_id},
            "sbref": {"session": ses_id},
            "fmap": {"session": ses_id},
        }
        filter_file.write_text(json.dumps(bids_filter, indent=2))
        cmd.extend(['--bids-filter-file', str(filter_file)])

    # Add pre-computed derivatives (e.g., anat from Stage 1)
    if derivatives:
        cmd.extend(['--derivatives', str(derivatives)])

    # Add FreeSurfer subjects dir if reusing
    if fs_subjects_dir:
        cmd.extend(['--fs-subjects-dir', str(fs_subjects_dir)])

    # Print summary
    print("=" * 60)
    print(f"Running fMRIPrep {fmriprep_version}")
    print("=" * 60)
    print(f"BIDS Directory:    {bids_dir}")
    print(f"Output Directory:  {output_dir}")
    print(f"Work Directory:    {work_dir}")
    print(f"Output Spaces:     {' '.join(output_spaces)}")
    print(f"FS License:        {fs_license}")
    if fs_subjects_dir:
        print(f"FS Subjects Dir:   {fs_subjects_dir}")
    if derivatives:
        print(f"Derivatives:       {derivatives}")
    if anat_only:
        print(f"Mode:              anat-only")
    elif session:
        ses_id = session.replace('ses-', '')
        print(f"Mode:              func (ses-{ses_id} only)")
    else:
        print(f"Mode:              full (anat + func)")
    if subjects:
        print(f"Subjects:          {', '.join(subjects)}")
    else:
        print("Subjects:          ALL")
    print(f"Processes:         {nprocs}")
    print(f"Memory:            {mem_gb} GB")
    print(f"Container:         {fmriprep_image}")
    print("=" * 60)
    print()

    # Run fMRIPrep
    try:
        subprocess.run(cmd, check=True)

        print()
        print("=" * 60)
        print("fMRIPrep completed successfully!")
        print("=" * 60)
        print(f"Results are in: {output_dir}")
        print()
        print("Next steps:")
        print(f"  - Review HTML reports in: {output_dir}")
        print(f"  - Check confound files in: {output_dir}/sub-*/ses-*/func/")
        print("=" * 60)

    except subprocess.CalledProcessError:
        print()
        print("=" * 60)
        print("ERROR: fMRIPrep failed")
        print("=" * 60)
        print("Check the logs above for error messages")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Run fMRIPrep on MMMData BIDS dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--subjects',
        nargs='+',
        help='Specific subjects to process (e.g., sub-03 sub-04). Default: all subjects'
    )

    parser.add_argument(
        '--nprocs',
        type=int,
        default=8,
        help='Number of parallel processes (default: 8)'
    )

    parser.add_argument(
        '--mem-gb',
        type=int,
        default=32,
        help='Memory limit in GB (default: 32)'
    )

    parser.add_argument(
        '--output-spaces',
        nargs='+',
        default=DEFAULT_OUTPUT_SPACES,
        help=f'Output spaces (default: {" ".join(DEFAULT_OUTPUT_SPACES)})'
    )

    parser.add_argument(
        '--fs-license',
        help='Path to FreeSurfer license file (auto-detected if omitted)'
    )

    parser.add_argument(
        '--fs-subjects-dir',
        help='Reuse existing FreeSurfer recon-all output from this directory'
    )

    parser.add_argument(
        '--fmriprep-version',
        default=DEFAULT_FMRIPREP_VERSION,
        help=f'fMRIPrep version to use (default: {DEFAULT_FMRIPREP_VERSION})'
    )

    parser.add_argument(
        '--anat-only',
        action='store_true',
        help='Only run anatomical preprocessing (no functional)'
    )

    parser.add_argument(
        '--session',
        help='Process only this session (e.g., ses-01 or 01)'
    )

    parser.add_argument(
        '--derivatives',
        help='Path to pre-computed derivatives (e.g., fmriprep anat output) to skip reprocessing'
    )

    parser.add_argument(
        '--bids-dir',
        help='Override BIDS directory from config'
    )

    parser.add_argument(
        '--output-dir',
        help='Override output directory (default: <derivatives>/fmriprep)'
    )

    args = parser.parse_args()

    # Load config
    config = load_config()
    paths = config.get('paths', {})

    # Get BIDS directory
    bids_dir = args.bids_dir or paths.get('bids_project_dir')
    if not bids_dir:
        print("ERROR: BIDS directory not specified and not found in config")
        sys.exit(1)

    bids_dir = Path(bids_dir)

    # Get output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(paths.get('output_dir', str(bids_dir / 'derivatives'))) / 'fmriprep'

    # Get singularity directory from config
    singularity_dir = paths.get('singularity_dir')

    # Get work directory from config (must be outside BIDS tree)
    work_dir = paths.get('work_dir')

    # Run fMRIPrep
    run_fmriprep(
        bids_dir=bids_dir,
        output_dir=output_dir,
        subjects=args.subjects,
        nprocs=args.nprocs,
        mem_gb=args.mem_gb,
        output_spaces=args.output_spaces,
        fs_license=args.fs_license,
        fs_subjects_dir=args.fs_subjects_dir,
        fmriprep_version=args.fmriprep_version,
        singularity_dir=singularity_dir,
        work_dir=work_dir,
        anat_only=args.anat_only,
        session=args.session,
        derivatives=args.derivatives,
    )


if __name__ == '__main__':
    main()
