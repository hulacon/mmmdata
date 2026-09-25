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

from .fmriprep_layout import space_part
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
    NATIVE_SPACE,
    MixedLocalizerDesignError,
    check_single_design,
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
        Volumetric template and resolution (e.g., "MNI152NLin2009cAsym_res-2"),
        or ``NATIVE_SPACE`` ("func") for the run's own grid.
    bold : Path | None
        Preprocessed BOLD NIfTI (*_desc-preproc_bold.nii.gz).
    mask : Path | None
        Brain mask in the same space as BOLD.
    boldref : Path | None
        BOLD reference image in the same space. For ``NATIVE_SPACE`` this
        is the coregistered reference (``*_desc-coreg_boldref.nii.gz``),
        the one fMRIPrep aligned to T1w, not the pre-coregistration
        ``desc-hmc`` one.
    confounds : Path | None
        Confounds timeseries TSV.
    confounds_json : Path | None
        Sidecar JSON for confounds.
    events : Path | None
        Events TSV. Resolved from the main BIDS tree; the legacy
        derivatives/bids_validation/eventfiles tree was deleted 2026-08-20.
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
    allow_mixed_designs: bool = False,
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
        Volumetric template and resolution (e.g., "MNI152NLin2009cAsym_res-2"),
        or ``NATIVE_SPACE`` ("func") for the space-less native-grid files.
        Native grids differ across sessions and move with the head between
        runs; mask_intersection refuses a native pool that does not share one
        anatomy (check_native_pool). Pool across sessions in T1w.
    bids_root : Path, optional
        BIDS root. If None, resolved via config.
    allow_mixed_designs : bool
        Some localizer task labels (``LOCALIZER_DESIGNS`` in constants) cover
        two different protocols acquired in different sessions by different
        cohorts. When ``task`` names one of them and the matched runs span
        both, this raises ``MixedLocalizerDesignError`` unless set to True.
        Pass True only when pooling is intended and the analysis accounts
        for it; the usual fix is a ``session`` filter.

    Returns
    -------
    list[FmriprepRun]
        Sorted by (subject, session, task, run). Missing optional files
        are None. Runs with no confounds TSV are not included.

    Raises
    ------
    MixedLocalizerDesignError
        See ``allow_mixed_designs``. Only when ``task`` is given; an
        unfiltered sweep is not a design selection.
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

    if task is not None and not allow_mixed_designs:
        try:
            check_single_design(task, (r.session for r in runs))
        except MixedLocalizerDesignError as exc:
            by_subject = sorted({(r.subject, r.session) for r in runs})
            who = ", ".join(f"sub-{s} ses-{ss}" for s, ss in by_subject)
            raise MixedLocalizerDesignError(
                f"{exc} Matched: {who}. Add a session= filter, or pass "
                "allow_mixed_designs=True to pool deliberately."
            ) from None

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

    def _path(tail: str) -> Optional[Path]:
        p = func_dir / f"{prefix}{tail}"
        return p if p.exists() else None

    # "" for NATIVE_SPACE (fMRIPrep writes those files with no space entity),
    # "_space-<label>" otherwise. See fmriprep_layout.space_part.
    sp = space_part(space)
    bold = _path(f"{sp}_desc-preproc_bold.nii.gz")
    mask = _path(f"{sp}_desc-brain_mask.nii.gz")
    # Native has no bare boldref; the coregistered one is the T1w-aligned reference.
    boldref = _path(f"{sp}_boldref.nii.gz") if sp else _path("_desc-coreg_boldref.nii.gz")
    confounds = _path("_desc-confounds_timeseries.tsv")
    confounds_json = _path("_desc-confounds_timeseries.json")
    surface_L = _path("_hemi-L_space-fsaverage6_bold.func.gii")
    surface_R = _path("_hemi-R_space-fsaverage6_bold.func.gii")

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

    Checks the legacy ``derivatives/bids_validation/eventfiles/`` tree first,
    then raw BIDS ``sub-*/ses-*/func/``. That legacy tree was deleted
    2026-08-20, so in practice every lookup resolves to the main tree — which
    is the intended target. The first check is retained only so a restored
    tree would still win.

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


#: Native-space runs pooled voxelwise must map to the same anatomy to within
#: this fraction of the smallest voxel dimension (see check_native_pool).
NATIVE_POOL_TOL_VOX = 0.5


def coreg_xfm_path(run: FmriprepRun) -> Path:
    """fMRIPrep's ``<run>_from-boldref_to-T1w_mode-image_desc-coreg_xfm.txt``, beside the confounds."""
    return Path(run.confounds).parent / f"{run.entity_prefix}_from-boldref_to-T1w_mode-image_desc-coreg_xfm.txt"


def read_itk_affine(path: Path) -> tuple[Any, Any, Any]:
    """(matrix 3x3, translation 3, center 3) of a single-transform ITK affine text file.

    ITK maps a physical point x (LPS) as ``A (x - c) + c + t``.
    """
    import numpy as np

    params = fixed = None
    for line in Path(path).read_text().splitlines():
        if line.startswith("Parameters:"):
            params = np.array(line.split(":", 1)[1].split(), dtype=float)
        elif line.startswith("FixedParameters:"):
            fixed = np.array(line.split(":", 1)[1].split(), dtype=float)
    if params is None or params.size != 12:
        raise ValueError(f"{path}: expected one 3-D affine (12 parameters)")
    return params[:9].reshape(3, 3), params[9:], (fixed if fixed is not None else np.zeros(3))


def check_native_pool(runs: Sequence[FmriprepRun], mask: Any = None) -> float:
    """Refuse native-space (``func``) runs that do not share one anatomy.

    fMRIPrep's native space is each run's own boldref grid. Runs from
    different sessions can share a header affine (same prescription) while the
    head sits millimetres elsewhere, so the grid check in
    :func:`mask_intersection` cannot see it; what decides alignment is each
    run's boldref-to-T1w coregistration. Two conditions, both errors:

    - every run comes from one session (across sessions: use ``T1w``);
    - within the session, every run's coregistration maps the brain to within
      ``NATIVE_POOL_TOL_VOX`` of a voxel of the first run's (head moved
      between runs otherwise).

    Returns the largest displacement found, in mm, over the voxels of
    ``mask`` (default: the first run's brain mask).
    """
    import numpy as np

    sessions = sorted({r.session for r in runs})
    if len(sessions) > 1:
        raise ValueError(
            f"native-space runs span sessions {', '.join('ses-' + s for s in sessions)}; each session's "
            "native grid sits on a different head position. Pool in T1w (or a template) instead, "
            "or fit one session at a time"
        )
    if len(runs) < 2:
        return 0.0
    ref = mask if mask is not None else load_mask(runs[0])
    ijk = np.argwhere(np.asarray(ref.dataobj).astype(bool))[::7]  # a subsample is plenty for a max over a rigid map
    ras = ijk @ ref.affine[:3, :3].T + ref.affine[:3, 3]
    lps = ras * np.array([-1.0, -1.0, 1.0])

    def apply(xfm):
        a, t, c = xfm
        return (lps - c) @ a.T + c + t

    y0 = apply(read_itk_affine(coreg_xfm_path(runs[0])))
    tol = NATIVE_POOL_TOL_VOX * float(min(ref.header.get_zooms()[:3]))
    worst = 0.0
    for r in runs[1:]:
        d = float(np.linalg.norm(apply(read_itk_affine(coreg_xfm_path(r))) - y0, axis=1).max())
        worst = max(worst, d)
        if d > tol:
            raise ValueError(
                f"{r.entity_prefix} is {d:.2f} mm from {runs[0].entity_prefix} in native space "
                f"(tolerance {tol:.2f} mm, half a voxel): the head moved between runs. "
                "Pool in T1w (or a template) instead"
            )
    return worst


def mask_intersection(runs: Sequence[FmriprepRun]) -> tuple[Any, Any]:
    """The voxels inside every run's brain mask, as (image, boolean array).

    Raises ValueError when two runs' masks sit on different grids: runs pooled
    into one map must share a space and resolution. Native-space (``func``)
    runs must also share one anatomy, which a matching grid does not prove
    (:func:`check_native_pool`).
    """
    import nibabel as nib
    import numpy as np

    if runs and runs[0].space == NATIVE_SPACE:
        check_native_pool(runs)
    first = load_mask(runs[0])
    inter = np.asarray(first.dataobj).astype(bool)
    for r in runs[1:]:
        m = load_mask(r)
        if m.shape != first.shape or not np.allclose(m.affine, first.affine, atol=1e-3):
            raise ValueError(
                f"{r.entity_prefix} mask grid differs from {runs[0].entity_prefix}; "
                "runs pooled into one map must share a space and resolution"
            )
        inter &= np.asarray(m.dataobj).astype(bool)
    return nib.Nifti1Image(inter.astype(np.uint8), first.affine), inter
