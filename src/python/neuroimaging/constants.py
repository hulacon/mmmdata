"""Constants for neuroimaging data access in MMMData.

Path templates, confound column groups, and analysis stream mappings.
Parallel to behavioral/constants.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional


# ---------------------------------------------------------------------------
# Default BIDS root (fallback if config unavailable)
# ---------------------------------------------------------------------------

DEFAULT_BIDS_ROOT = Path("/gpfs/projects/hulacon/shared/mmmdata")


# ---------------------------------------------------------------------------
# fMRIPrep variant configuration
# ---------------------------------------------------------------------------

FMRIPREP_VARIANTS: tuple[str, ...] = ("fmriprep", "fmriprep_nordic")
# NORDIC arm punted from the re-preprocessing campaign (D14, 2026-08-21);
# the raw 25.2.5 arm is the campaign's only product and the default. Until
# the campaign's coverage gate passes, derivatives/fmriprep may be PARTIAL —
# check coverage before subject-level analyses.
DEFAULT_VARIANT: str = "fmriprep"
DEFAULT_SPACE: str = "MNI152NLin2009cAsym_res-2"

# Derivatives directories (relative to bids_root)
DERIVATIVES_DIRS: dict[str, str] = {
    "fmriprep": "derivatives/fmriprep",
    "fmriprep_nordic": "derivatives/fmriprep_nordic",
    "nordic": "derivatives/nordic",
    "mriqc": "derivatives/mriqc",
    "preprocessing_qc": "derivatives/preprocessing_qc",
    "ready": "derivatives/ready",
}

# Events files — legacy transitional location, DELETED 2026-08-20. Kept only
# so find_events_file() still prefers it if the tree is ever restored; in
# practice every lookup now falls through to the main BIDS tree, which is the
# decided target. Do not build new code against this.
EVENTFILES_DIR: str = "derivatives/bids_validation/eventfiles"


# ---------------------------------------------------------------------------
# Confound column groups (verified against fMRIPrep v24.1.1 output)
# ---------------------------------------------------------------------------

# 6 basic motion parameters
MOTION_6: list[str] = [
    "trans_x", "trans_y", "trans_z",
    "rot_x", "rot_y", "rot_z",
]

# Friston 24: 6 params + derivatives + quadratics + derivative quadratics
MOTION_24: list[str] = [
    col
    for base in MOTION_6
    for col in (
        base,
        f"{base}_derivative1",
        f"{base}_power2",
        f"{base}_derivative1_power2",
    )
]

# Anatomical CompCor — top 6 components (combined WM+CSF mask)
ACOMPCOR_6: list[str] = [f"a_comp_cor_{i:02d}" for i in range(6)]

# Cosine columns vary per run (depend on run length); match by prefix
COSINE_PREFIX: str = "cosine"

# QC-relevant columns
FD_COLUMN: str = "framewise_displacement"
DVARS_COLUMN: str = "std_dvars"


# ---------------------------------------------------------------------------
# QC threshold defaults
# ---------------------------------------------------------------------------
# Fallbacks used only when config/base.toml cannot be read. The live values
# come from the ``[qc]`` section there — see ``neuroimaging.qc.qc_settings``.
# Rationale and citations for each: docs/doc/qc-guidance.md.

DEFAULT_QC_SETTINGS: dict[str, float] = {
    "fd_threshold": 0.5,
    "investigate_threshold": 0.5,
    "iqr_multiplier": 1.5,
}


# ---------------------------------------------------------------------------
# Task-to-stream mapping
# ---------------------------------------------------------------------------

TASK_STREAM_MAP: dict[str, str] = {
    # Trial-based sessions
    "TBencoding": "glmsingle",
    "TBretrieval": "glmsingle",
    "TBmath": "glmsingle",
    "TBresting": "connectivity",
    # Naturalistic sessions
    "NATencoding": "naturalistic",
    "NATretrieval": "naturalistic",
    "NATmath": "naturalistic",
    "NATresting": "naturalistic",
    # Localizers — block-design go to GLMsingle stream
    "floc": "glmsingle",
    "motor": "glmsingle",
    "auditory": "glmsingle",
    "tone": "glmsingle",
    # pRF localizer goes to naturalistic stream
    "prf": "naturalistic",
    # Baseline resting
    "INITresting": "connectivity",
}


# ---------------------------------------------------------------------------
# Subject and session ranges
# ---------------------------------------------------------------------------

SUBJECT_IDS: tuple[str, ...] = ("03", "04", "05")

# All imaging sessions (excluding ses-29 which is behavioral-only)
ALL_SESSIONS: tuple[str, ...] = tuple(
    f"{s:02d}" for s in list(range(1, 29)) + [30]
)

TB_SESSIONS: tuple[str, ...] = tuple(f"{s:02d}" for s in range(4, 19))
NAT_SESSIONS: tuple[str, ...] = tuple(f"{s:02d}" for s in range(19, 29))

# Sessions in which functional localizers were acquired. There are two
# groups, acquired by different cohorts: the dedicated localizer sessions
# (ses-02/03) and the final session (ses-30). This used to be ("02", "03"),
# which silently excluded every ses-30 localizer run.
LOCALIZER_SESSION_GROUPS: dict[str, tuple[str, ...]] = {
    "ses-02/03": ("02", "03"),
    "ses-30": ("30",),
}
LOCALIZER_SESSIONS: tuple[str, ...] = tuple(
    s for group in LOCALIZER_SESSION_GROUPS.values() for s in group
)

# Task labels that cover two different protocols, one per session group,
# perfectly confounded with subject (no subject ran both). The root
# task-*_bold.json sidecars carry both instruction texts labelled by session;
# nothing else in BIDS marks the split. Selecting on one of these tasks across
# session groups pools two designs: `motor` cued a silent lip movement in one
# and spoken nonsense syllables in the other, so a shared `mouth` regressor
# means different things. Record and source map: mmmdata-agents
# docs/results/task-instructions-provenance.md.
LOCALIZER_DESIGNS: dict[str, dict[str, tuple[str, ...]]] = {
    "motor": dict(LOCALIZER_SESSION_GROUPS),
    "auditory": dict(LOCALIZER_SESSION_GROUPS),
}


class MixedLocalizerDesignError(ValueError):
    """A selection spans more than one design of a split localizer task."""


def localizer_design(task: str, session: str) -> Optional[str]:
    """Design label for ``(task, session)``, or None if ``task`` has one design.

    Raises ValueError if ``task`` is a split task and ``session`` is not one
    where it was acquired — a filter typo, not a third design.
    """
    designs = LOCALIZER_DESIGNS.get(task)
    if designs is None:
        return None
    for label, sessions in designs.items():
        if session in sessions:
            return label
    known = sorted(s for group in designs.values() for s in group)
    raise ValueError(
        f"task-{task} was not acquired in ses-{session}; "
        f"known sessions: {', '.join(f'ses-{s}' for s in known)}"
    )


def check_single_design(task: str, sessions: Iterable[str]) -> None:
    """Raise MixedLocalizerDesignError if ``sessions`` span >1 design of ``task``.

    No-op for tasks with a single design. Callers that mean to pool must say
    so explicitly rather than catching this.
    """
    if task not in LOCALIZER_DESIGNS:
        return
    by_design: dict[str, set[str]] = {}
    for s in sessions:
        by_design.setdefault(localizer_design(task, s), set()).add(s)
    if len(by_design) <= 1:
        return
    parts = ", ".join(
        f"{label} ({', '.join(f'ses-{s}' for s in sorted(ss))})"
        for label, ss in sorted(by_design.items())
    )
    raise MixedLocalizerDesignError(
        f"task-{task} selection spans {len(by_design)} designs: {parts}. "
        "These are different protocols under one label and cannot be pooled "
        "without saying so; filter by session, or opt in explicitly."
    )
