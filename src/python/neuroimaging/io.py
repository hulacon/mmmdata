"""File discovery and loading for MMMData preprocessed neuroimaging data.

All filesystem interaction for fMRIPrep outputs is isolated here. Analysis
modules receive FmriprepRun objects or DataFrames and never touch the
filesystem directly.

Typical usage::

    from neuroimaging.io import find_fmriprep_runs, load_confounds

    runs = find_fmriprep_runs(subject="03", session="04", variant="fmriprep_nordic")
    for run in runs:
        confounds = load_confounds(run, columns=["framewise_displacement"])
"""

from __future__ import annotations

import dataclasses
import re
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

import pandas as pd

from .constants import (
    ACOMPCOR_6,
    COSINE_PREFIX,
    DEFAULT_BIDS_ROOT,
    DEFAULT_SPACE,
    DEFAULT_VARIANT,
    DERIVATIVES_DIRS,
    EVENTFILES_DIR,
    FMRIPREP_VARIANTS,
    MOTION_24,
)


# ---------------------------------------------------------------------------
# BIDS root resolution (same pattern as behavioral/io.py)
# ---------------------------------------------------------------------------

def _resolve_bids_root(bids_root: Optional[Path] = None) -> Path:
    """Return BIDS root from argument, config, or fallback constant."""
    if bids_root is not None:
        return Path(bids_root)
    try:
        code_root = Path(__file__).resolve().parents[3]
        if str(code_root) not in sys.path:
            sys.path.insert(0, str(code_root / "src" / "python"))
        from core.config import load_config
        config = load_config()
        return Path(config["paths"]["bids_project_dir"])
    except Exception:
        return DEFAULT_BIDS_ROOT


# ---------------------------------------------------------------------------
# FmriprepRun dataclass
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class FmriprepRun:
    """All paths for a single preprocessed BOLD run.

    Produced by ``find_fmriprep_runs()``. Fields that do not exist on disk
    are ``None``. The ``confounds`` field is the most reliable indicator of
    a completed fMRIPrep run; other fields may be missing independently.

    Attributes
    ----------
    subject : str
        Zero-padded subject ID (e.g., "03").
    session : str
        Zero-padded session ID (e.g., "04").
    task : str
        BIDS task label (e.g., "TBencoding").
    run : str | None
        Zero-padded run index (e.g., "01"), or None for single-run tasks.
    variant : str
        Either "fmriprep" or "fmriprep_nordic".
    space : str
        Volumetric template and resolution (e.g., "MNI152NLin2009cAsym_res-2").
    bold : Path | None
        Preprocessed BOLD NIfTI (*_desc-preproc_bold.nii.gz).
    mask : Path | None
        Brain mask in the same space as BOLD.
    boldref : Path | None
        BOLD reference image in the same space.
    confounds : Path | None
        Confounds timeseries TSV.
    confounds_json : Path | None
        Sidecar JSON for confounds.
    events : Path | None
        Events TSV (from derivatives/bids_validation/eventfiles or raw BIDS).
    surface_L, surface_R : Path | None
        fsaverage6 surface GIfTIs for each hemisphere.
    """

    subject: str
    session: str
    task: str
    run: Optional[str]
    variant: str
    space: str
    bold: Optional[Path] = None
    mask: Optional[Path] = None
    boldref: Optional[Path] = None
    confounds: Optional[Path] = None
    confounds_json: Optional[Path] = None
    events: Optional[Path] = None
    surface_L: Optional[Path] = None
    surface_R: Optional[Path] = None

    @property
    def run_part(self) -> str:
        """BIDS filename fragment for run entity ('' or '_run-XX')."""
        return f"_run-{self.run}" if self.run else ""

    @property
    def entity_prefix(self) -> str:
        """BIDS filename prefix up to and including run entity."""
        return (
            f"sub-{self.subject}_ses-{self.session}"
            f"_task-{self.task}{self.run_part}"
        )


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------

_CONFOUNDS_RE = re.compile(
    r"sub-(?P<subject>[^_]+)"
    r"_ses-(?P<session>[^_]+)"
    r"_task-(?P<task>[^_]+)"
    r"(?:_run-(?P<run>[^_]+))?"
    r"_desc-confounds_timeseries\.tsv$"
)


def _parse_confounds_name(path: Path) -> Optional[dict[str, Optional[str]]]:
    """Extract subject/session/task/run from a confounds TSV filename."""
    m = _CONFOUNDS_RE.match(path.name)
    if not m:
        return None
    return {
        "subject": m.group("subject"),
        "session": m.group("session"),
        "task": m.group("task"),
        "run": m.group("run"),
    }


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def find_fmriprep_runs(
    subject: Optional[str] = None,
    session: Optional[str] = None,
    task: Optional[str] = None,
    run: Optional[str] = None,
    variant: str = DEFAULT_VARIANT,
    space: str = DEFAULT_SPACE,
    bids_root: Optional[Path] = None,
) -> list[FmriprepRun]:
    """Discover preprocessed BOLD runs matching filters.

    Globs ``derivatives/{variant}/sub-*/ses-*/func/`` for confounds TSVs
    (the most reliable fMRIPrep completion indicator), then resolves
    sibling files for each.

    Parameters
    ----------
    subject, session, task, run : str, optional
        BIDS entity filters. If None, all are matched. Subject/session/run
        should be zero-padded strings (e.g., "03", not "3").
    variant : str
        Either "fmriprep" or "fmriprep_nordic".
    space : str
        Volumetric template and resolution (e.g., "MNI152NLin2009cAsym_res-2").
    bids_root : Path, optional
        BIDS root. If None, resolved via config.

    Returns
    -------
    list[FmriprepRun]
        Sorted by (subject, session, task, run). Missing optional files
        are None. Runs with no confounds TSV are not included.
    """
    if variant not in FMRIPREP_VARIANTS:
        raise ValueError(
            f"variant must be one of {FMRIPREP_VARIANTS}, got {variant!r}"
        )

    bids_root = _resolve_bids_root(bids_root)
    variant_dir = bids_root / DERIVATIVES_DIRS[variant]
    if not variant_dir.exists():
        return []

    sub_glob = f"sub-{subject}" if subject else "sub-*"
    ses_glob = f"ses-{session}" if session else "ses-*"

    confounds_paths = sorted(
        variant_dir.glob(f"{sub_glob}/{ses_glob}/func/*_desc-confounds_timeseries.tsv")
    )

    runs: list[FmriprepRun] = []
    for conf_path in confounds_paths:
        parsed = _parse_confounds_name(conf_path)
        if parsed is None:
            continue
        # Apply task/run filters
        if task is not None and parsed["task"] != task:
            continue
        if run is not None and parsed["run"] != run:
            continue

        runs.append(
            _build_fmriprep_run(
                conf_path.parent,
                parsed,
                variant=variant,
                space=space,
                bids_root=bids_root,
            )
        )

    return runs


def _build_fmriprep_run(
    func_dir: Path,
    parsed: dict[str, Optional[str]],
    variant: str,
    space: str,
    bids_root: Path,
) -> FmriprepRun:
    """Construct an FmriprepRun by probing for sibling files."""
    subject = parsed["subject"]
    session = parsed["session"]
    task = parsed["task"]
    run = parsed["run"]

    run_part = f"_run-{run}" if run else ""
    prefix = f"sub-{subject}_ses-{session}_task-{task}{run_part}"

    def _path(suffix: str) -> Optional[Path]:
        p = func_dir / f"{prefix}_{suffix}"
        return p if p.exists() else None

    bold = _path(f"space-{space}_desc-preproc_bold.nii.gz")
    mask = _path(f"space-{space}_desc-brain_mask.nii.gz")
    boldref = _path(f"space-{space}_boldref.nii.gz")
    confounds = _path("desc-confounds_timeseries.tsv")
    confounds_json = _path("desc-confounds_timeseries.json")
    surface_L = _path("hemi-L_space-fsaverage6_bold.func.gii")
    surface_R = _path("hemi-R_space-fsaverage6_bold.func.gii")

    events = find_events_file(
        subject=subject,
        session=session,
        task=task,
        run=run,
        bids_root=bids_root,
    )

    return FmriprepRun(
        subject=subject,
        session=session,
        task=task,
        run=run,
        variant=variant,
        space=space,
        bold=bold,
        mask=mask,
        boldref=boldref,
        confounds=confounds,
        confounds_json=confounds_json,
        events=events,
        surface_L=surface_L,
        surface_R=surface_R,
    )


def find_events_file(
    subject: str,
    session: str,
    task: str,
    run: Optional[str] = None,
    bids_root: Optional[Path] = None,
) -> Optional[Path]:
    """Find events TSV for a run.

    Checks ``derivatives/bids_validation/eventfiles/`` first (canonical
    validated events), then raw BIDS ``sub-*/ses-*/func/``.

    Returns None if no events file exists (expected for resting-state runs).
    """
    bids_root = _resolve_bids_root(bids_root)
    run_part = f"_run-{run}" if run else ""
    basename = f"sub-{subject}_ses-{session}_task-{task}{run_part}_events.tsv"

    # Canonical location
    canonical = (
        bids_root / EVENTFILES_DIR / f"sub-{subject}" / f"ses-{session}" / basename
    )
    if canonical.exists():
        return canonical

    # Raw BIDS fallback
    raw = bids_root / f"sub-{subject}" / f"ses-{session}" / "func" / basename
    if raw.exists():
        return raw

    return None


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_confounds(
    run: FmriprepRun,
    columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Load confounds TSV for a run.

    Parameters
    ----------
    run : FmriprepRun
        Run with a non-None ``confounds`` field.
    columns : sequence of str, optional
        If provided, load only these columns. Missing columns raise KeyError.

    Returns
    -------
    pd.DataFrame
        One row per volume. fMRIPrep's ``n/a`` values are parsed to NaN.

    Raises
    ------
    FileNotFoundError
        If ``run.confounds`` is None or the file does not exist.
    """
    if run.confounds is None or not run.confounds.exists():
        raise FileNotFoundError(
            f"No confounds file for {run.entity_prefix} ({run.variant})"
        )

    df = pd.read_csv(run.confounds, sep="\t", na_values=["n/a", "N/A", ""])

    if columns is not None:
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise KeyError(
                f"Columns not in confounds TSV: {missing}. "
                f"Available: {list(df.columns)[:10]}..."
            )
        df = df[list(columns)].copy()

    return df


def select_confound_columns(
    confounds_df: pd.DataFrame,
    motion: bool = True,
    acompcor: bool = True,
    cosine: bool = True,
) -> list[str]:
    """Return list of columns for a standard 24HMP + 6aCompCor + cosine selection.

    Useful helper when callers want to pick columns without enumerating them.
    Cosine column count varies per run; this matches all columns starting
    with ``cosine``.
    """
    cols: list[str] = []
    if motion:
        cols.extend(MOTION_24)
    if acompcor:
        cols.extend(ACOMPCOR_6)
    if cosine:
        cols.extend(
            c for c in confounds_df.columns if c.startswith(COSINE_PREFIX)
        )
    # Verify all selected cols exist
    missing = [c for c in cols if c not in confounds_df.columns]
    if missing:
        raise KeyError(
            f"Expected confound columns not in DataFrame: {missing}"
        )
    return cols


def load_bold(run: FmriprepRun) -> Any:
    """Lazy-load BOLD NIfTI via nibabel.

    Returns
    -------
    nibabel.Nifti1Image

    Raises
    ------
    FileNotFoundError
        If ``run.bold`` is None or does not exist.
    """
    if run.bold is None or not run.bold.exists():
        raise FileNotFoundError(
            f"No BOLD file for {run.entity_prefix} ({run.variant})"
        )
    import nibabel as nib
    return nib.load(str(run.bold))


def load_mask(run: FmriprepRun) -> Any:
    """Lazy-load brain mask NIfTI via nibabel."""
    if run.mask is None or not run.mask.exists():
        raise FileNotFoundError(
            f"No mask file for {run.entity_prefix} ({run.variant})"
        )
    import nibabel as nib
    return nib.load(str(run.mask))
