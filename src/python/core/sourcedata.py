"""Utilities for inspecting the raw source data tree (mmmsourcedata).

The raw source tree is kept outside the BIDS dataset so the BIDS tree can
be shared without exposing PII. Its layout is
``sub-XX/ses-YY/{audio,behavioral,dicom,eyetracking,other}``, plus
``shared/`` (cross-subject material) and ``archive/`` (pilots, retired
conversion staging).

These functions are read-only: they scan what has been collected in the
source tree and compare the source-to-BIDS file inventory against the
files that actually exist in the BIDS dataset.
"""

import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

# Per-session modality subdirectories in the source tree
MODALITIES = ["audio", "behavioral", "dicom", "eyetracking", "other"]

# conversion_type values in the inventory that never produce a BIDS file
NON_CONVERTED_TYPES = ("timing_input", "no_conversion", "supplementary")


def list_source_subjects(source_dir: str | Path) -> List[str]:
    """List subject directories present in the source data tree.

    Parameters
    ----------
    source_dir : str or Path
        Root of the raw source tree (e.g. ``.../mmmsourcedata``).

    Returns
    -------
    list of str
        Sorted directory names starting with ``sub-`` (e.g. ``['sub-03',
        'sub-06']``). Empty if ``source_dir`` does not exist.
    """
    source_dir = Path(source_dir)
    if not source_dir.is_dir():
        return []
    return sorted(
        d.name for d in source_dir.iterdir()
        if d.is_dir() and d.name.startswith("sub-")
    )


def scan_sourcedata(
    source_dir: str | Path,
    subject_id: str,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Scan the raw source tree for one subject's collected data.

    Walks ``source_dir/sub-<subject_id>/ses-*`` and tallies what exists in
    each per-session modality subdirectory. DICOM directories are counted
    by series (one subdirectory per series); all other modalities are
    counted by file, with a breakdown by file extension. Useful for
    checking what has been collected for not-yet-BIDSified subjects.

    Parameters
    ----------
    source_dir : str or Path
        Root of the raw source tree (e.g. ``.../mmmsourcedata``).
    subject_id : str
        Subject ID without the ``sub-`` prefix, e.g. ``'06'``.
    session_id : str, optional
        Session ID without the ``ses-`` prefix, e.g. ``'01'``. If given,
        only that session is scanned; otherwise all sessions.

    Returns
    -------
    dict
        Dictionary with keys:

        - ``'subject_id'``: the subject ID as given
        - ``'path'``: path to the subject directory (str)
        - ``'session_count'``: number of sessions scanned
        - ``'totals'``: per-modality totals across sessions (zero
          entries omitted)
        - ``'sessions'``: per-session dict mapping modality to
          ``{'series_count': n}`` (dicom) or
          ``{'file_count': n, 'extensions': {...}}`` (everything else)

    Raises
    ------
    FileNotFoundError
        If no ``sub-<subject_id>`` directory exists under ``source_dir``.
    """
    source_dir = Path(source_dir)
    subject_dir = source_dir / f"sub-{subject_id}"
    if not subject_dir.is_dir():
        raise FileNotFoundError(
            f"No source data for sub-{subject_id} at {subject_dir}"
        )

    session_dirs = sorted(
        d for d in subject_dir.iterdir()
        if d.is_dir() and d.name.startswith("ses-")
    )
    if session_id:
        session_dirs = [d for d in session_dirs if d.name == f"ses-{session_id}"]

    sessions: Dict[str, Any] = {}
    totals: Dict[str, int] = dict.fromkeys(MODALITIES, 0)
    for ses_dir in session_dirs:
        ses_info: Dict[str, Any] = {}
        for modality in MODALITIES:
            mod_dir = ses_dir / modality
            if not mod_dir.is_dir():
                continue
            if modality == "dicom":
                # DICOM sessions hold one directory per series
                series = [d.name for d in mod_dir.iterdir() if d.is_dir()]
                ses_info[modality] = {"series_count": len(series)}
                totals[modality] += len(series)
            else:
                files = [f for f in mod_dir.rglob("*") if f.is_file()]
                ses_info[modality] = {
                    "file_count": len(files),
                    "extensions": count_by_extension(f.name for f in files),
                }
                totals[modality] += len(files)
        sessions[ses_dir.name] = ses_info

    return {
        "subject_id": subject_id,
        "path": str(subject_dir),
        "session_count": len(sessions),
        "totals": {k: v for k, v in totals.items() if v},
        "sessions": sessions,
    }


def count_by_extension(filenames: Iterable[str]) -> Dict[str, int]:
    """Count filenames by (lower-cased) extension.

    Parameters
    ----------
    filenames : iterable of str
        File names (or paths) to tally.

    Returns
    -------
    dict
        Mapping of extension (e.g. ``'.csv'``, or ``'(no extension)'``)
        to count, ordered by descending count.
    """
    counts: Dict[str, int] = {}
    for name in filenames:
        ext = Path(name).suffix.lower() or "(no extension)"
        counts[ext] = counts.get(ext, 0) + 1
    return dict(sorted(counts.items(), key=lambda x: -x[1]))


def diff_source_vs_bids(
    inventory_csv: str | Path,
    bids_root: str | Path,
    subject_id: Optional[str] = None,
    max_pending: int = 50,
) -> Dict[str, Any]:
    """Compare the source-to-BIDS file inventory against files on disk.

    Reads the file inventory CSV and, for each row that maps a source
    file to a BIDS destination, checks whether that destination exists
    under ``bids_root``. Rows whose ``conversion_type`` never produces a
    BIDS file (``timing_input``, ``no_conversion``, ``supplementary``)
    are skipped.

    Parameters
    ----------
    inventory_csv : str or Path
        Path to ``file_inventory.csv`` (must have ``bids_destination``
        and ``conversion_type`` columns).
    bids_root : str or Path
        Root of the BIDS dataset that destinations are relative to.
    subject_id : str, optional
        Subject ID without the ``sub-`` prefix. If given, only rows whose
        destination mentions ``sub-<subject_id>`` are considered;
        otherwise all subjects.
    max_pending : int, default=50
        Maximum number of pending files to list in ``'pending_files'``
        (caps output size; counts are always complete).

    Returns
    -------
    dict
        Dictionary with keys:

        - ``'total_mapped'``: rows with a BIDS destination considered
        - ``'converted_count'`` / ``'pending_count'``: how many exist
          on disk vs. not
        - ``'by_type'``: per-conversion-type converted/pending counts
        - ``'pending_files'``: up to ``max_pending`` pending entries,
          each ``{'bids_destination', 'conversion_type'}``

    Raises
    ------
    FileNotFoundError
        If ``inventory_csv`` does not exist.
    """
    inventory_csv = Path(inventory_csv)
    bids_root = Path(bids_root)
    if not inventory_csv.exists():
        raise FileNotFoundError(f"Inventory file not found: {inventory_csv}")

    converted: List[Dict[str, str]] = []
    pending: List[Dict[str, str]] = []
    by_type: Dict[str, Dict[str, int]] = {}

    with open(inventory_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dest = row.get("bids_destination", "").strip()
            conv_type = row.get("conversion_type", "unknown").strip()

            # Skip rows without a BIDS destination or that don't need conversion
            if not dest or conv_type in NON_CONVERTED_TYPES:
                continue

            # Filter by subject if requested
            if subject_id:
                if f"sub-{subject_id}" not in dest:
                    continue

            exists = (bids_root / dest).exists()

            entry = {"bids_destination": dest, "conversion_type": conv_type}
            if exists:
                converted.append(entry)
            else:
                pending.append(entry)

            if conv_type not in by_type:
                by_type[conv_type] = {"converted": 0, "pending": 0}
            by_type[conv_type]["converted" if exists else "pending"] += 1

    return {
        "total_mapped": len(converted) + len(pending),
        "converted_count": len(converted),
        "pending_count": len(pending),
        "by_type": by_type,
        "pending_files": pending[:max_pending],
    }
