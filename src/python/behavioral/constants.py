"""Constants, enums, and mappings for MMMData behavioral analysis."""

from __future__ import annotations

from enum import IntEnum
from pathlib import Path


# ---------------------------------------------------------------------------
# Subject and session ranges
# ---------------------------------------------------------------------------

SUBJECT_IDS: tuple[str, ...] = ("03", "04", "05")

# Trial-based sessions where 2AFC recognition exists
TB_SESSIONS: tuple[str, ...] = tuple(f"{s:02d}" for s in range(4, 19))  # ses-04..ses-18

# Encoding events exist only through ses-17 (no encoding in ses-18)
TB_ENCODING_SESSIONS: tuple[str, ...] = tuple(f"{s:02d}" for s in range(4, 18))

FINAL_SESSION: str = "30"


# ---------------------------------------------------------------------------
# Default BIDS root (fallback if config unavailable)
# ---------------------------------------------------------------------------

DEFAULT_BIDS_ROOT = Path("/gpfs/projects/hulacon/shared/mmmdata")


# ---------------------------------------------------------------------------
# BIDS path templates (relative to bids_root)
# Use .format(sub=..., ses=..., run=...) or glob with * for run
# ---------------------------------------------------------------------------

TB2AFC_TEMPLATE = "sub-{sub}/ses-{ses}/beh/sub-{sub}_ses-{ses}_task-TB2AFC_run-01_beh.tsv"
FIN2AFC_TEMPLATE = "sub-{sub}/ses-30/beh/sub-{sub}_ses-30_task-FIN2AFC_beh.tsv"
FINTIMELINE_TEMPLATE = "sub-{sub}/ses-30/beh/sub-{sub}_ses-30_task-FINtimeline_beh.tsv"

# Events files live in derivatives/bids_validation/eventfiles/
EVENTFILES_PREFIX = "derivatives/bids_validation/eventfiles"
TBENCODING_TEMPLATE = (
    f"{EVENTFILES_PREFIX}/sub-{{sub}}/ses-{{ses}}/"
    "sub-{sub}_ses-{ses}_task-TBencoding_run-{run}_events.tsv"
)
TBRETRIEVAL_TEMPLATE = (
    f"{EVENTFILES_PREFIX}/sub-{{sub}}/ses-{{ses}}/"
    "sub-{sub}_ses-{ses}_task-TBretrieval_run-{run}_events.tsv"
)

# Glob-friendly versions (run replaced with *)
TBENCODING_GLOB = (
    f"{EVENTFILES_PREFIX}/sub-{{sub}}/ses-{{ses}}/"
    "sub-{sub}_ses-{ses}_task-TBencoding_run-*_events.tsv"
)
TBRETRIEVAL_GLOB = (
    f"{EVENTFILES_PREFIX}/sub-{{sub}}/ses-{{ses}}/"
    "sub-{sub}_ses-{ses}_task-TBretrieval_run-*_events.tsv"
)


# ---------------------------------------------------------------------------
# Encoding condition enum
# ---------------------------------------------------------------------------

class EnCon(IntEnum):
    """Encoding condition: how stimuli were presented during encoding."""
    SINGLE = 1    # Seen once
    REPEATS = 2   # Same pair repeated 3x within session
    TRIPLETS = 3  # Three pairs, each repeated 3x


ENCON_LABELS: dict[int, str] = {1: "single", 2: "repeats", 3: "triplets"}


# ---------------------------------------------------------------------------
# Retrieval condition enum
# ---------------------------------------------------------------------------

class ReCon(IntEnum):
    """Retrieval condition: when retrieval occurs relative to encoding."""
    WITHIN = 1   # Tested within the same session as encoding
    ACROSS = 2   # Tested in a later session


RECON_LABELS: dict[int, str] = {1: "within", 2: "across"}


# ---------------------------------------------------------------------------
# Scanner button box mapping
# ---------------------------------------------------------------------------

# In-scanner tasks record button box numbers 6, 7, 8 for a 3-point scale.
# Subtract SCANNER_RESP_OFFSET to get semantic rating: 6->1, 7->2, 8->3.
SCANNER_RESP_OFFSET: int = 5


# ---------------------------------------------------------------------------
# 2AFC response scheme
#   resp 1 = sure image1 is target
#   resp 2 = maybe image1 is target
#   resp 3 = maybe image2 is target
#   resp 4 = sure image2 is target
# recog = position chosen: 1 (resp in {1,2}) or 2 (resp in {3,4})
# confidence: sure (resp in {1,4}), maybe (resp in {2,3})
# ---------------------------------------------------------------------------

RESP_POSITION_MAP: dict[int, int] = {1: 1, 2: 1, 3: 2, 4: 2}
RESP_CONFIDENCE_MAP: dict[int, str] = {1: "sure", 2: "maybe", 3: "maybe", 4: "sure"}


# ---------------------------------------------------------------------------
# Session order mapping for regression / plotting x-axes
# ses-04 -> 0, ses-05 -> 1, ..., ses-18 -> 14
# ---------------------------------------------------------------------------

SESSION_ORDER: dict[str, int] = {f"{s:02d}": s - 4 for s in range(4, 19)}


# ---------------------------------------------------------------------------
# Column name lists per task (for validation)
# ---------------------------------------------------------------------------

TB2AFC_COLUMNS: list[str] = [
    "onset", "duration", "subject_id", "session_num", "run_num",
    "trial_type", "modality", "word", "image1", "image2",
    "correct_resp", "resp", "resp_RT", "trial_accuracy",
    "enCon", "reCon", "cueId", "pairId", "recog", "trial_id",
]

ENCODING_COLUMNS: list[str] = [
    "onset", "duration", "onset_actual", "duration_actual",
    "subject", "session", "run", "trial_type", "modality", "word",
    "pairId", "mmmId", "nsdId", "itmno", "sharedId",
    "voiceId", "voice", "enCon", "reCon", "resp", "resp_RT", "trial_id",
]

RETRIEVAL_COLUMNS: list[str] = ENCODING_COLUMNS + ["cueId"]

FIN2AFC_COLUMNS: list[str] = [
    "onset", "duration", "trial_type", "modality", "word",
    "image1", "image2", "correct_resp", "resp", "resp_RT",
    "accuracy", "trial_accuracy", "enCon", "reCon",
    "cueId", "pairId", "recog", "trial_id",
]

FINTIMELINE_COLUMNS: list[str] = [
    "onset", "duration", "trial_type", "word", "timeline_resp",
    "timeline_RT", "enCon", "reCon", "pairId", "trial_accuracy", "trial_id",
]

# Normalized column names after io.py loading
# TB2AFC: subject_id -> subject, session_num -> session
COLUMN_RENAMES: dict[str, str] = {
    "subject_id": "subject",
    "session_num": "session",
    "run_num": "run",
}


# ---------------------------------------------------------------------------
# Output derivatives
# ---------------------------------------------------------------------------

DERIVATIVES_DIR: str = "behavioral_analysis"
