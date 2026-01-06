#!/usr/bin/env python3
"""
Create a BIDS file inventory spreadsheet across all subjects.
Each row represents one instance of a file type, with count indicating the sequential instance number.

This script can be used to generate an inventory of BIDS-compliant files across subjects,
making it easy to see which files are present or missing for each subject.
"""

import argparse
import re
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List


def parse_bids_filename(filepath: str) -> Dict:
    """
    Parse BIDS filename into components.

    Extracts BIDS entities from a filename following the BIDS naming convention.

    Args:
        filepath: Path to a BIDS-compliant file

    Returns:
        Dictionary containing extracted BIDS entities:
            - full_path: Complete file path
            - extension: File extension(s) (.nii.gz, .json, etc.)
            - subject: Subject ID (if present)
            - session: Session identifier (if present)
            - task: Task name (if present)
            - run: Run number (if present)
            - direction: Phase encoding direction (if present)
            - acquisition: Acquisition parameters (if present)
            - datatype: BIDS datatype (anat, func, dwi, fmap, or unknown)
            - suffix: BIDS suffix (T1w, bold, dwi, etc., if present)

    Example:
        >>> parse_bids_filename('sub-01/ses-baseline/anat/sub-01_ses-baseline_T1w.nii.gz')
        {'full_path': '...', 'subject': '01', 'session': 'baseline', 'datatype': 'anat', 'suffix': 'T1w', ...}
    """
    path = Path(filepath)
    filename = path.name

    # Extract BIDS entities using regex
    entities = {}
    entities['full_path'] = str(path)
    entities['extension'] = ''.join(path.suffixes)  # Get all suffixes (.nii.gz, .json, etc)

    # Extract subject
    sub_match = re.search(r'sub-(\w+)', filename)
    if sub_match:
        entities['subject'] = sub_match.group(1)

    # Extract session (alphanumeric to support ses-baseline, ses-pre, etc.)
    ses_match = re.search(r'ses-([\w]+)', filename)
    if ses_match:
        entities['session'] = ses_match.group(1)

    # Extract task (supports hyphens in task names like task-rest-eyes-open)
    task_match = re.search(r'task-([\w-]+)', filename)
    if task_match:
        entities['task'] = task_match.group(1)

    # Extract run (alphanumeric to support both numeric and named runs)
    run_match = re.search(r'run-([\w]+)', filename)
    if run_match:
        entities['run'] = run_match.group(1)

    # Extract direction (value before next underscore or dot)
    dir_match = re.search(r'dir-([\w]+)(?:_|\.)', filename)
    if dir_match:
        entities['direction'] = dir_match.group(1)

    # Extract acquisition (value before next underscore or dot)
    acq_match = re.search(r'acq-([\w]+)(?:_|\.)', filename)
    if acq_match:
        entities['acquisition'] = acq_match.group(1)

    # Extract datatype from path
    parts = path.parts
    if 'anat' in parts:
        entities['datatype'] = 'anat'
    elif 'func' in parts:
        entities['datatype'] = 'func'
    elif 'dwi' in parts:
        entities['datatype'] = 'dwi'
    elif 'fmap' in parts:
        entities['datatype'] = 'fmap'
    else:
        entities['datatype'] = 'unknown'

    # Extract suffix (T1w, T2w, bold, dwi, epi, etc)
    suffix_match = re.search(r'_([a-zA-Z0-9]+)\.(nii\.gz|json|tsv|bval|bvec)$', filename)
    if suffix_match:
        entities['suffix'] = suffix_match.group(1)

    return entities


def create_shorthand_label(entities: Dict, ext: str) -> str:
    """
    Create a shorthand label for the file type (without session/run).

    Args:
        entities: Dictionary of BIDS entities from parse_bids_filename()
        ext: File extension (e.g., '.nii.gz', '.json')

    Returns:
        String combining datatype, suffix, task, direction, acquisition, and extension
        separated by underscores (e.g., 'anat_T1w_nii' or 'func_bold_task-rest_nii')

    Example:
        >>> entities = {'datatype': 'func', 'suffix': 'bold', 'task': 'rest'}
        >>> create_shorthand_label(entities, '.nii.gz')
        'func_bold_task-rest_nii'
    """
    parts = []

    # Add datatype prefix
    datatype = entities.get('datatype', 'unknown')
    parts.append(datatype)

    # Add suffix (modality)
    suffix = entities.get('suffix', '')
    if suffix:
        parts.append(suffix)

    # Add task if present
    task = entities.get('task')
    if task:
        parts.append(f"task-{task}")

    # Add direction if present
    direction = entities.get('direction')
    if direction:
        parts.append(f"dir-{direction}")

    # Add acquisition if present
    acq = entities.get('acquisition')
    if acq:
        parts.append(f"acq-{acq}")

    # Add extension indicator
    if ext == '.nii.gz':
        parts.append('nii')
    elif ext == '.json':
        parts.append('json')
    elif ext == '.tsv':
        parts.append('tsv')
    elif ext == '.bval':
        parts.append('bval')
    elif ext == '.bvec':
        parts.append('bvec')

    return '_'.join(parts)


def find_bids_files(bids_root: Path, subjects: List[str]) -> Dict:
    """
    Find all BIDS files for given subjects.

    Args:
        bids_root: Path to the BIDS dataset root directory
        subjects: List of subject IDs (without 'sub-' prefix)

    Returns:
        Dictionary mapping subject IDs to lists of file paths found for each subject

    Example:
        >>> find_bids_files(Path('/data/bids'), ['01', '02'])
        {'01': ['/data/bids/sub-01/anat/sub-01_T1w.nii.gz', ...], '02': [...]}
    """
    files_by_subject = defaultdict(list)

    for subject in subjects:
        subject_dir = bids_root / f"sub-{subject}"
        if not subject_dir.exists():
            continue

        # Find all BIDS-compliant files
        for ext in ['*.nii.gz', '*.json', '*.tsv', '*.bval', '*.bvec']:
            for filepath in subject_dir.rglob(ext):
                files_by_subject[subject].append(str(filepath))

    return files_by_subject


def create_inventory(bids_root: Path, subjects: List[str], output_path: Path):
    """
    Create the BIDS file inventory TSV.

    Args:
        bids_root: Path to the BIDS dataset root directory
        subjects: List of subject IDs (without 'sub-' prefix)
        output_path: Path where the TSV inventory file will be written

    Raises:
        FileNotFoundError: If bids_root doesn't exist
        PermissionError: If unable to write to output_path
        ValueError: If subjects list is empty

    The output TSV contains columns for:
        - shorthand_label: Descriptive label for the file type
        - count: Sequential instance number for this file type
        - file_format: File extension
        - datatype: BIDS datatype (anat, func, etc.)
        - task: Task name (if applicable)
        - sub-XX: One column per subject with file paths or 'DNE' (Does Not Exist)
        - notes: Empty column for manual annotations
    """
    # Validation
    if not bids_root.exists():
        raise FileNotFoundError(f"BIDS root directory not found: {bids_root}")

    if not subjects:
        raise ValueError("Subject list is empty. Please provide at least one subject ID.")

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Find all files
    files_by_subject = find_bids_files(bids_root, subjects)

    # Index: file_type_key -> (session, run) -> subject -> filepath
    # file_type_key combines: shorthand_label + extension
    file_type_index = defaultdict(lambda: defaultdict(lambda: {subj: None for subj in subjects}))

    for subject in subjects:
        if subject not in files_by_subject:
            continue

        for filepath in files_by_subject[subject]:
            entities = parse_bids_filename(filepath)
            ext = entities['extension']

            # Create file type key (without session/run)
            label = create_shorthand_label(entities, ext)

            # Get session and run for this instance
            session = entities.get('session', 'none')
            run = entities.get('run', 'none')

            # Store in index
            file_type_index[label][(session, run)][subject] = filepath

    # Collect all rows
    rows = []

    for label in sorted(file_type_index.keys()):
        instances = file_type_index[label]

        # Sort instances by session, then run
        def sort_key(item):
            session, run = item[0]
            # Convert to integers for proper sorting, handle 'none' cases
            session_num = int(session) if session != 'none' else 0
            run_num = int(run) if run != 'none' else 0
            return (session_num, run_num)

        sorted_instances = sorted(instances.items(), key=sort_key)

        # Assign counts and create rows
        for count, ((session, run), subject_paths) in enumerate(sorted_instances, start=1):
            # Get first available file to extract metadata
            first_path = None
            for subj in subjects:
                if subject_paths[subj]:
                    first_path = subject_paths[subj]
                    break

            if not first_path:
                continue

            entities = parse_bids_filename(first_path)
            ext = entities['extension']
            datatype = entities.get('datatype', '')
            task = entities.get('task', 'n/a')

            # Build row data
            row_data = {
                'label': label,
                'count': count,
                'format': ext,
                'datatype': datatype,
                'task': task,
            }

            # Add paths for each subject dynamically
            for subj in subjects:
                subject_key = f'sub{subj}'
                row_data[subject_key] = subject_paths[subj] if subject_paths[subj] else 'DNE'

            rows.append(row_data)

    # Check if any files were found
    if not rows:
        print(f"WARNING: No BIDS files found for subjects {subjects} in {bids_root}")
        print("Creating empty inventory file with headers only.")

    # Write TSV
    try:
        with open(output_path, 'w') as f:
            # Write header dynamically based on subjects
            header_cols = ['shorthand_label', 'count', 'file_format', 'datatype', 'task']
            header_cols.extend([f'sub-{subj}' for subj in subjects])
            header_cols.append('notes')
            f.write('\t'.join(header_cols) + '\n')

            # Write rows
            for row in rows:
                row_values = [
                    str(row['label']),
                    str(row['count']),
                    str(row['format']),
                    str(row['datatype']),
                    str(row['task'])
                ]
                # Add subject paths in order
                for subj in subjects:
                    subject_key = f'sub{subj}'
                    row_values.append(str(row[subject_key]))
                # Add empty notes column
                row_values.append('')
                f.write('\t'.join(row_values) + '\n')

        print(f"✓ Inventory created: {output_path}")
        print(f"✓ Total file entries: {len(rows)}")
        print(f"✓ Subjects included: {', '.join(subjects)}")
    except PermissionError as e:
        raise PermissionError(f"Unable to write to {output_path}: {e}")


def auto_discover_subjects(bids_root: Path) -> List[str]:
    """
    Auto-discover subjects in a BIDS dataset.

    Args:
        bids_root: Path to the BIDS dataset root directory

    Returns:
        List of subject IDs (without 'sub-' prefix) sorted alphabetically
    """
    subjects = []
    for item in bids_root.iterdir():
        if item.is_dir() and item.name.startswith('sub-'):
            subject_id = item.name.replace('sub-', '')
            subjects.append(subject_id)
    return sorted(subjects)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Create a BIDS file inventory spreadsheet across subjects.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-discover subjects
  %(prog)s /data/bids_dataset /output/inventory.tsv

  # Specify subjects explicitly
  %(prog)s /data/bids_dataset /output/inventory.tsv --subjects 01 02 03

  # Use default paths (for this project)
  %(prog)s
        """
    )

    parser.add_argument(
        'bids_root',
        type=Path,
        nargs='?',
        default=Path('/projects/hulacon/shared/mmmdata'),
        help='Path to BIDS dataset root directory (default: %(default)s)'
    )

    parser.add_argument(
        'output_path',
        type=Path,
        nargs='?',
        default=Path('/projects/hulacon/shared/mmmdata/inventory/bids_file_inventory.tsv'),
        help='Path to output TSV file (default: %(default)s)'
    )

    parser.add_argument(
        '--subjects',
        type=str,
        nargs='+',
        help='Subject IDs without "sub-" prefix (e.g., 01 02 03). If not provided, auto-discovers all subjects.'
    )

    args = parser.parse_args()

    # Auto-discover subjects if not provided
    if args.subjects is None:
        print(f"Auto-discovering subjects in {args.bids_root}...")
        args.subjects = auto_discover_subjects(args.bids_root)
        if not args.subjects:
            print(f"ERROR: No subjects found in {args.bids_root}")
            sys.exit(1)
        print(f"Found {len(args.subjects)} subjects: {', '.join(args.subjects)}")

    try:
        create_inventory(args.bids_root, args.subjects, args.output_path)
    except (FileNotFoundError, ValueError, PermissionError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
