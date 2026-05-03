"""Shared constants, path helpers, and event readers for the NORDIC benchmark.

Minimal foundation for tb_glmsingle_fit.py / tb_eval.py / nat_eval.py.
ROI mask loading is deferred to the eval scripts (parallel to
isc_confounds/extract_roi_timeseries.py:load_roi_masks).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

BIDS_ROOT = Path("/gpfs/projects/hulacon/shared/mmmdata")
DERIV_ROOT = BIDS_ROOT / "derivatives"

SUBJECTS = ["sub-03", "sub-04", "sub-05"]
TB_SESSIONS = [f"ses-{i:02d}" for i in range(4, 18)]   # ses-04..ses-17 (14)
NAT_SESSIONS = [f"ses-{i:02d}" for i in range(19, 29)]  # ses-19..ses-28 (10)
TB_RUNS = [1, 2, 3]
NAT_RUNS = [1, 2]

TR = 1.5
STIMDUR = 3.0
N_VOLS_TB = 210
MNI_SPACE = "MNI152NLin2009cAsym_res-2"

PIPELINES = {
    "original": DERIV_ROOT / "fmriprep",
    "nordic":   DERIV_ROOT / "fmriprep_nordic",
}

BENCHMARK_OUT = DERIV_ROOT / "nordic" / "benchmark"

# Unified output schema for benchmark_summary.tsv
SUMMARY_COLS = [
    "subject", "session", "stream", "roi", "pipeline",
    "within_r", "between_r", "discriminability",
    "n_within", "n_between",
]


# ── path helpers ─────────────────────────────────────────────────────────

def bold_path(sub: str, ses: str, run: int, task: str, pipeline: str) -> Path:
    """Path to fMRIPrep preproc BOLD in MNI res-2."""
    base = PIPELINES[pipeline] / sub / ses / "func"
    stem = f"{sub}_{ses}_task-{task}_run-{run:02d}"
    return base / f"{stem}_space-{MNI_SPACE}_desc-preproc_bold.nii.gz"


def mask_path(sub: str, ses: str, run: int, task: str, pipeline: str) -> Path:
    """Path to fMRIPrep brain mask in MNI res-2."""
    base = PIPELINES[pipeline] / sub / ses / "func"
    stem = f"{sub}_{ses}_task-{task}_run-{run:02d}"
    return base / f"{stem}_space-{MNI_SPACE}_desc-brain_mask.nii.gz"


def events_path(sub: str, ses: str, run: int, task: str) -> Path:
    """Path to raw BIDS events.tsv (same for both pipelines)."""
    base = BIDS_ROOT / sub / ses / "func"
    return base / f"{sub}_{ses}_task-{task}_run-{run:02d}_events.tsv"


# ── TB events reader ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class TBTrial:
    """One TBencoding image trial."""
    onset_s: float       # seconds
    onset_tr: int        # TR index = round(onset_s / TR)
    mmm_id: str          # NSD shared1000 stimulus id
    run: int
    session: str
    subject: str


def read_tb_trials(sub: str, ses: str, run: int) -> list[TBTrial]:
    """Read TBencoding events.tsv → list of image trials in chronological order.

    Filters out rest/fixation trials; keeps only `trial_type==image` rows.
    Per the 2026-05-03 audit, all such trials have valid onset, mmmId, and
    duration=3.0; no scrubbing needed.
    """
    path = events_path(sub, ses, run, "TBencoding")
    trials: list[TBTrial] = []
    with open(path) as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            if row["trial_type"] != "image":
                continue
            onset = float(row["onset"])
            trials.append(TBTrial(
                onset_s=onset,
                onset_tr=round(onset / TR),
                mmm_id=row["mmmId"],
                run=run,
                session=ses,
                subject=sub,
            ))
    return trials


def list_tb_runs(sub: str, pipeline: str) -> list[tuple[str, int]]:
    """Return [(session, run), ...] for all TBencoding runs of a subject in a pipeline."""
    runs = []
    for ses in TB_SESSIONS:
        for r in TB_RUNS:
            if bold_path(sub, ses, r, "TBencoding", pipeline).exists():
                runs.append((ses, r))
    return runs
