#!/usr/bin/env python3
"""
resample_bold_to_func.py — Post-hoc native-space (func) BOLD resampling

Produces HMC + SDC-corrected BOLD timeseries in each run's original
acquisition grid, without spatial normalization. Replicates what
fMRIPrep's ``--output-spaces func`` does, using the cached transforms
from an existing fMRIPrep run.

STC is intentionally skipped — per-voxel HRF fitting in GLMsingle
absorbs slice-timing delays, and the interpolation artifacts from STC
may reduce tSNR for single-trial estimation.

Must run inside the fMRIPrep Singularity container (needs sdcflows,
nitransforms, fmriprep.interfaces.resampling).

Usage (via container):
    singularity exec -B /gpfs:/gpfs fmriprep.simg python3 \\
        resample_bold_to_func.py \\
        --subject sub-03 --session ses-04 \\
        --bids-dir /path/to/mmmdata \\
        [--fmriprep-dir derivatives/fmriprep_nordic] \\
        [--nordic-dir derivatives/nordic/bids_input] \\
        [--dry-run]
"""

import argparse
import json
import glob
import os
import sys
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np


def find_runs(fmriprep_func_dir):
    """Find all BOLD runs by locating HMC transform files."""
    pattern = os.path.join(fmriprep_func_dir, '*_from-orig_to-boldref_*_desc-hmc_xfm.txt')
    xfm_files = sorted(glob.glob(pattern))
    runs = []
    for xfm_file in xfm_files:
        basename = os.path.basename(xfm_file)
        prefix = basename.split('_from-orig_to-boldref')[0]
        runs.append(prefix)
    return runs


def get_fmap_id(fmriprep_func_dir, prefix):
    """Extract fieldmap ID from the boldref-to-B0map transform filename."""
    pattern = os.path.join(fmriprep_func_dir, f'{prefix}_from-boldref_to-B0map*_xfm.txt')
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f'No boldref-to-B0map transform found for {prefix}')
    basename = os.path.basename(matches[0])
    fmap_id = basename.split('B0map')[1].split('_mode')[0]
    return f'B0map{fmap_id}', matches[0]


def find_fmap_files(fmriprep_fmap_dir, fmap_id):
    """Find coefficient and EPI reference files for a given fieldmap ID."""
    coeff_pattern = os.path.join(fmriprep_fmap_dir, f'*fmapid-{fmap_id}_desc-coeff_fieldmap.nii.gz')
    epi_pattern = os.path.join(fmriprep_fmap_dir, f'*fmapid-{fmap_id}_desc-epi_fieldmap.nii.gz')

    coeff_files = glob.glob(coeff_pattern)
    epi_files = glob.glob(epi_pattern)

    if not coeff_files:
        raise FileNotFoundError(f'No coefficient file for {fmap_id} in {fmriprep_fmap_dir}')
    if not epi_files:
        raise FileNotFoundError(f'No EPI reference for {fmap_id} in {fmriprep_fmap_dir}')

    return coeff_files[0], epi_files[0]


def resample_run(prefix, nordic_func_dir, fmriprep_func_dir, fmriprep_fmap_dir,
                 output_dir, workdir):
    """Resample a single BOLD run to native (func) space."""
    from fmriprep.interfaces.resampling import ResampleSeries, ReconstructFieldmap

    bold_file = os.path.join(nordic_func_dir, f'{prefix}_bold.nii.gz')
    bold_json = os.path.join(nordic_func_dir, f'{prefix}_bold.json')
    boldref_file = os.path.join(fmriprep_func_dir, f'{prefix}_desc-hmc_boldref.nii.gz')
    hmc_file = os.path.join(fmriprep_func_dir,
                            f'{prefix}_from-orig_to-boldref_mode-image_desc-hmc_xfm.txt')

    for f in [bold_file, bold_json, boldref_file, hmc_file]:
        if not os.path.exists(f):
            raise FileNotFoundError(f)

    with open(bold_json) as f:
        meta = json.load(f)

    fmap_id, b0_xfm_file = get_fmap_id(fmriprep_func_dir, prefix)
    coeff_file, epi_ref_file = find_fmap_files(fmriprep_fmap_dir, fmap_id)

    run_workdir = os.path.join(workdir, prefix)
    os.makedirs(run_workdir, exist_ok=True)

    # Reconstruct fieldmap in boldref space
    recon = ReconstructFieldmap()
    recon.inputs.in_coeffs = [coeff_file]
    recon.inputs.fmap_ref_file = epi_ref_file
    recon.inputs.target_ref_file = boldref_file
    recon.inputs.transforms = [b0_xfm_file]
    recon.inputs.inverse = [False]
    recon.base_dir = run_workdir
    fmap_file = recon.run().outputs.out_file

    # Resample: HMC + SDC in one interpolation step, no STC
    rs = ResampleSeries()
    rs.inputs.in_file = bold_file
    rs.inputs.ref_file = boldref_file
    rs.inputs.transforms = [hmc_file]
    rs.inputs.fieldmap = fmap_file
    rs.inputs.pe_dir = meta['PhaseEncodingDirection']
    rs.inputs.ro_time = meta['TotalReadoutTime']
    rs.inputs.jacobian = True
    rs.base_dir = run_workdir
    result_file = rs.run().outputs.out_file

    import shutil
    out_bold = os.path.join(output_dir, f'{prefix}_desc-preproc_bold.nii.gz')
    shutil.move(result_file, out_bold)

    out_json = os.path.join(output_dir, f'{prefix}_desc-preproc_bold.json')
    sidecar = {
        'SkullStripped': False,
        'SliceTimingCorrected': False,
        'RepetitionTime': meta['RepetitionTime'],
        'TaskName': meta.get('TaskName', ''),
        'Sources': [f'bids:raw:{prefix}_bold.nii.gz'],
        'Description': 'HMC + SDC corrected BOLD in native acquisition grid (no STC, no spatial normalization). Post-hoc resampled using cached fMRIPrep transforms.',
    }
    if 'StartTime' in meta:
        sidecar['StartTime'] = meta['StartTime']
    with open(out_json, 'w') as f:
        json.dump(sidecar, f, indent=2)
        f.write('\n')

    return out_bold


def main():
    parser = argparse.ArgumentParser(
        description='Post-hoc native-space BOLD resampling using cached fMRIPrep transforms',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--subject', required=True, help='Subject ID (e.g., sub-03)')
    parser.add_argument('--session', required=True, help='Session ID (e.g., ses-04)')
    parser.add_argument('--bids-dir', required=True, help='BIDS dataset root')
    parser.add_argument('--fmriprep-dir', default='derivatives/fmriprep_nordic',
                        help='fMRIPrep output dir relative to bids-dir (default: derivatives/fmriprep_nordic)')
    parser.add_argument('--nordic-dir', default='derivatives/nordic/bids_input',
                        help='NORDIC BIDS input dir relative to bids-dir (default: derivatives/nordic/bids_input)')
    parser.add_argument('--work-dir', default=None,
                        help='Working directory for intermediates (default: tempdir, cleaned on exit)')
    parser.add_argument('--dry-run', action='store_true',
                        help='List runs that would be processed without running')
    parser.add_argument('--skip-existing', action='store_true', default=True,
                        help='Skip runs that already have func-space output (default: True)')
    parser.add_argument('--no-skip-existing', action='store_false', dest='skip_existing',
                        help='Reprocess runs even if output exists')
    args = parser.parse_args()

    bids_dir = Path(args.bids_dir)
    sub = args.subject if args.subject.startswith('sub-') else f'sub-{args.subject}'
    ses = args.session if args.session.startswith('ses-') else f'ses-{args.session}'

    fmriprep_base = bids_dir / args.fmriprep_dir / sub / ses
    fmriprep_func = fmriprep_base / 'func'
    fmriprep_fmap = fmriprep_base / 'fmap'
    nordic_func = bids_dir / args.nordic_dir / sub / ses / 'func'
    output_dir = fmriprep_func

    for d in [fmriprep_func, fmriprep_fmap, nordic_func]:
        if not d.exists():
            print(f'ERROR: directory not found: {d}')
            sys.exit(1)

    runs = find_runs(str(fmriprep_func))
    if not runs:
        print(f'No BOLD runs found in {fmriprep_func}')
        sys.exit(1)

    if args.skip_existing:
        existing = set()
        for r in runs:
            out_file = output_dir / f'{r}_desc-preproc_bold.nii.gz'
            if out_file.exists():
                existing.add(r)
        runs_to_process = [r for r in runs if r not in existing]
    else:
        existing = set()
        runs_to_process = runs

    print('=' * 60)
    print(f'resample_bold_to_func.py')
    print('=' * 60)
    print(f'Subject:       {sub}')
    print(f'Session:       {ses}')
    print(f'fMRIPrep dir:  {fmriprep_base}')
    print(f'NORDIC input:  {nordic_func}')
    print(f'Output dir:    {output_dir}')
    print(f'Total runs:    {len(runs)}')
    print(f'Already done:  {len(existing)}')
    print(f'To process:    {len(runs_to_process)}')
    print(f'STC:           SKIPPED')
    print('=' * 60)

    if args.dry_run:
        for r in runs_to_process:
            print(f'  [DRY RUN] {r}')
        return

    if not runs_to_process:
        print('Nothing to do.')
        return

    use_tempdir = args.work_dir is None
    if use_tempdir:
        tmpdir = tempfile.mkdtemp(prefix='resample_func_')
        workdir = tmpdir
    else:
        workdir = args.work_dir
        os.makedirs(workdir, exist_ok=True)
        tmpdir = None

    try:
        for i, prefix in enumerate(runs_to_process, 1):
            print(f'\n[{i}/{len(runs_to_process)}] {prefix}')
            try:
                out = resample_run(
                    prefix,
                    str(nordic_func),
                    str(fmriprep_func),
                    str(fmriprep_fmap),
                    str(output_dir),
                    workdir,
                )
                size_mb = os.path.getsize(out) / 1e6
                print(f'  -> {os.path.basename(out)} ({size_mb:.0f} MB)')
            except Exception as e:
                print(f'  ERROR: {e}')
                continue
    finally:
        if tmpdir and os.path.exists(tmpdir):
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)

    print('\n' + '=' * 60)
    print('Done.')
    print('=' * 60)


if __name__ == '__main__':
    main()
