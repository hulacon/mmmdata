"""BIDS dataset utilities using pybids."""

from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd

try:
    from bids import BIDSLayout
except ImportError:
    raise ImportError(
        "pybids is required for this module. "
        "Install it with: pip install pybids"
    )

from .config import load_config


def summarize_bids_dataset(
    bids_dir: Optional[str | Path] = None,
    config: Optional[Dict[str, Any]] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Summarize the contents of a BIDS dataset using pybids.
    
    If bids_dir is not provided, loads it from the configuration file.
    
    Parameters
    ----------
    bids_dir : str or Path, optional
        Path to the BIDS dataset directory. If None, loads from config.
    config : dict, optional
        Pre-loaded configuration dictionary. If None, loads from config files.
    verbose : bool, default=True
        If True, prints summary information to stdout.
    
    Returns
    -------
    dict
        Dictionary containing dataset summary with keys:
        - 'dataset_path': Path to the dataset
        - 'n_subjects': Number of subjects
        - 'subjects': List of subject IDs
        - 'n_sessions': Number of sessions (total across all subjects)
        - 'sessions': List of unique session IDs
        - 'datatypes': List of datatypes (e.g., 'anat', 'func')
        - 'modalities': List of modalities (e.g., 'T1w', 'bold')
        - 'tasks': List of task names (for func data)
        - 'layout': BIDSLayout object for further querying
    
    Examples
    --------
    >>> # Use config file
    >>> summary = summarize_bids_dataset()
    >>> print(f"Found {summary['n_subjects']} subjects")
    
    >>> # Specify dataset path directly
    >>> summary = summarize_bids_dataset('/path/to/bids/dataset')
    >>> layout = summary['layout']
    >>> files = layout.get(subject='01', suffix='T1w')
    """
    # Load configuration if needed
    if bids_dir is None:
        if config is None:
            config = load_config()
        # Support both old flat config and new nested config structure
        if 'paths' in config:
            bids_dir = config['paths'].get('bids_project_dir')
        else:
            bids_dir = config.get('bids_project_dir')

        if bids_dir is None:
            raise ValueError(
                "bids_project_dir not found in configuration and no bids_dir provided"
            )

    bids_dir = Path(bids_dir)

    if not bids_dir.exists():
        raise FileNotFoundError(f"BIDS directory not found: {bids_dir}")

    # Initialize BIDSLayout
    if verbose:
        print(f"Indexing BIDS dataset: {bids_dir}")

    # Note: validate=False is used for performance reasons and to handle
    # datasets that may not be 100% BIDS-compliant but are still usable.
    # For strict BIDS validation, use the BIDS validator tool separately.
    # See: https://bids-standard.github.io/bids-validator/
    layout = BIDSLayout(bids_dir, validate=False)
    
    # Gather summary information
    subjects = layout.get_subjects()
    sessions = layout.get_sessions()
    datatypes = layout.get_datatypes()
    
    # Get modalities (suffixes)
    modalities = layout.get_suffixes()
    
    # Get tasks (if any functional data exists)
    tasks = layout.get_tasks()
    
    summary = {
        'dataset_path': str(bids_dir),
        'n_subjects': len(subjects),
        'subjects': subjects,
        'n_sessions': len(sessions) if sessions else 0,
        'sessions': sessions if sessions else [],
        'datatypes': datatypes,
        'modalities': modalities,
        'tasks': tasks,
        'layout': layout
    }
    
    # Print summary if verbose
    if verbose:
        print("\n" + "="*60)
        print("BIDS Dataset Summary")
        print("="*60)
        print(f"Dataset path: {bids_dir}")
        print(f"\nSubjects: {len(subjects)}")
        if len(subjects) <= 10:
            print(f"  Subject IDs: {', '.join(subjects)}")
        else:
            print(f"  Subject IDs: {', '.join(subjects[:5])} ... {', '.join(subjects[-5:])}")
        
        if sessions:
            print(f"\nSessions: {len(sessions)}")
            print(f"  Session IDs: {', '.join(sessions)}")
        
        print(f"\nDatatypes: {', '.join(datatypes)}")
        print(f"Modalities: {', '.join(modalities)}")
        
        if tasks:
            print(f"Tasks: {', '.join(tasks)}")
        
        # Count files by datatype
        print("\nFile counts by datatype:")
        for datatype in datatypes:
            files = layout.get(datatype=datatype)
            print(f"  {datatype}: {len(files)} files")
        
        print("="*60 + "\n")
    
    return summary


def get_subject_summary(
    subject_id: str,
    layout: Optional[BIDSLayout] = None,
    bids_dir: Optional[str | Path] = None
) -> pd.DataFrame:
    """
    Get a detailed summary of all files for a specific subject.
    
    Parameters
    ----------
    subject_id : str
        Subject ID (without 'sub-' prefix)
    layout : BIDSLayout, optional
        Pre-initialized BIDSLayout object. If None, creates one from bids_dir.
    bids_dir : str or Path, optional
        Path to BIDS dataset. Required if layout is None.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with one row per file, containing BIDS entities and file paths.
    """
    if layout is None:
        if bids_dir is None:
            config = load_config()
            bids_dir = config['bids_project_dir']
        layout = BIDSLayout(bids_dir, validate=False)
    
    # Get all files for this subject
    files = layout.get(subject=subject_id, return_type='object')
    
    # Extract entities and paths
    data = []
    for f in files:
        entities = f.get_entities()
        entities['path'] = f.path
        data.append(entities)
    
    return pd.DataFrame(data)
