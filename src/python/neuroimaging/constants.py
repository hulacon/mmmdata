"""Constants for neuroimaging data access in MMMData.

Path templates, confound column groups, and analysis stream mappings.
Parallel to behavioral/constants.py.
"""

from __future__ import annotations

from pathlib import Path


# ---------------------------------------------------------------------------
# Default BIDS root (fallback if config unavailable)
# ---------------------------------------------------------------------------

DEFAULT_BIDS_ROOT = Path("/gpfs/projects/hulacon/shared/mmmdata")


# ---------------------------------------------------------------------------
# fMRIPrep variant configuration
# ---------------------------------------------------------------------------

FMRIPREP_VARIANTS: tuple[str, ...] = ("fmriprep", "fmriprep_nordic")
DEFAULT_VARIANT: str = "fmriprep_nordic"
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

# Events files — canonical validated copies live here
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
LOCALIZER_SESSIONS: tuple[str, ...] = ("02", "03")
