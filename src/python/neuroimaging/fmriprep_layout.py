"""Path builders for fMRIPrep derivative filenames.

Centralizes knowledge of fMRIPrep's output naming convention so callers
(scripts, MCP tools) can construct BOLD / brain-mask / confounds paths
from a derivatives directory plus BIDS entities, without hard-coding
filename patterns.

Complements ``neuroimaging.io``: that module *discovers* completed runs
by globbing the filesystem; this one *constructs* the expected paths
deterministically, whether or not the files exist on disk. Callers are
responsible for existence checks.

Usage::

    from neuroimaging import fmriprep_layout

    bold = fmriprep_layout.bold_path(fmriprep_dir, "03", "13", "TBencoding")
    mask = fmriprep_layout.mask_path(fmriprep_dir, "03", "13", "TBencoding")
    conf = fmriprep_layout.confounds_path(fmriprep_dir, "03", "13", "TBencoding")
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from .constants import DEFAULT_SPACE

__all__ = [
    "DEFAULT_SPACE",
    "run_prefix",
    "func_dir",
    "bold_path",
    "mask_path",
    "confounds_path",
]

PathLike = Union[str, Path]


def run_prefix(
    subject: str,
    session: str,
    task: str,
    run: Optional[str] = "01",
) -> str:
    """BIDS filename prefix for one functional run.

    Parameters
    ----------
    subject, session : str
        Zero-padded IDs (e.g., "03", "13") without the ``sub-``/``ses-``
        prefixes.
    task : str
        BIDS task label (e.g., "TBencoding").
    run : str, optional
        Zero-padded run index. Pass ``None`` for single-run tasks whose
        filenames omit the ``run-`` entity.

    Returns
    -------
    str
        e.g., ``"sub-03_ses-13_task-TBencoding_run-01"``.
    """
    run_part = f"_run-{run}" if run else ""
    return f"sub-{subject}_ses-{session}_task-{task}{run_part}"


def func_dir(fmriprep_dir: PathLike, subject: str, session: str) -> Path:
    """Functional derivatives directory for one subject/session."""
    return Path(fmriprep_dir) / f"sub-{subject}" / f"ses-{session}" / "func"


def bold_path(
    fmriprep_dir: PathLike,
    subject: str,
    session: str,
    task: str,
    run: Optional[str] = "01",
    space: str = DEFAULT_SPACE,
) -> Path:
    """Path to the preprocessed BOLD NIfTI (``*_desc-preproc_bold.nii.gz``).

    Parameters
    ----------
    fmriprep_dir : path-like
        Root of the fMRIPrep derivatives tree (e.g.,
        ``.../derivatives/fmriprep``).
    subject, session, task, run : str
        BIDS entities; see :func:`run_prefix`.
    space : str
        Volumetric template and resolution string as it appears in the
        filename (e.g., ``"MNI152NLin2009cAsym_res-2"``).
    """
    return func_dir(fmriprep_dir, subject, session) / (
        f"{run_prefix(subject, session, task, run)}"
        f"_space-{space}_desc-preproc_bold.nii.gz"
    )


def mask_path(
    fmriprep_dir: PathLike,
    subject: str,
    session: str,
    task: str,
    run: Optional[str] = "01",
    space: str = DEFAULT_SPACE,
) -> Path:
    """Path to the brain mask (``*_desc-brain_mask.nii.gz``) matching
    :func:`bold_path` in the same space."""
    return func_dir(fmriprep_dir, subject, session) / (
        f"{run_prefix(subject, session, task, run)}"
        f"_space-{space}_desc-brain_mask.nii.gz"
    )


def confounds_path(
    fmriprep_dir: PathLike,
    subject: str,
    session: str,
    task: str,
    run: Optional[str] = "01",
) -> Path:
    """Path to the confounds TSV (``*_desc-confounds_timeseries.tsv``).

    Confounds files carry no ``space-`` entity.
    """
    return func_dir(fmriprep_dir, subject, session) / (
        f"{run_prefix(subject, session, task, run)}"
        f"_desc-confounds_timeseries.tsv"
    )
