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


# ── Harvard-Oxford ROI masks (parallel to isc_confounds/extract_roi_timeseries.py) ──

# Cortical atlas labels (cort-maxprob-thr25-2mm)
CORTICAL_ROIS = {
    "V1":  [24],      # Intracalcarine Cortex
    "EAC": [45],      # Heschl's Gyrus
    "MT+": [13],      # Middle Temporal Gyrus, temporooccipital part
    "IFG": [5, 6],    # pars triangularis + opercularis
}

# Subcortical atlas labels (sub-maxprob-thr25-2mm)
SUBCORTICAL_ROIS = {
    "Hippocampus": {"L": [9], "R": [19]},
}

ROI_KEYS = [
    ("V1", "L"), ("V1", "R"),
    ("EAC", "L"), ("EAC", "R"),
    ("MT+", "L"), ("MT+", "R"),
    ("IFG", "L"), ("IFG", "R"),
    ("Hippocampus", "L"), ("Hippocampus", "R"),
]


def load_roi_masks():
    """Build volumetric Harvard-Oxford ROI masks. Cortical ROIs split L/R by x<0.

    Returns:
        masks: dict {(roi_name, hemi): bool ndarray (X, Y, Z)}
        affine: 4x4 atlas affine (MNI 2mm)
    """
    import nibabel as nib
    import numpy as np
    from nilearn.datasets import fetch_atlas_harvard_oxford

    cort = fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
    cort_img = cort.maps if hasattr(cort.maps, "affine") else nib.load(cort.maps)
    cort_data = np.asarray(cort_img.dataobj).astype(int)
    affine = cort_img.affine

    i_coords = np.arange(cort_data.shape[0])
    x_coords = affine[0, 0] * i_coords + affine[0, 3]
    left = x_coords < 0
    right = ~left

    masks = {}
    for roi, labels in CORTICAL_ROIS.items():
        m = np.isin(cort_data, labels)
        masks[(roi, "L")] = m & left[:, None, None]
        masks[(roi, "R")] = m & right[:, None, None]

    sub = fetch_atlas_harvard_oxford("sub-maxprob-thr25-2mm")
    sub_img = sub.maps if hasattr(sub.maps, "affine") else nib.load(sub.maps)
    sub_data = np.asarray(sub_img.dataobj).astype(int)
    for roi, hemi_labels in SUBCORTICAL_ROIS.items():
        for hemi, labels in hemi_labels.items():
            masks[(roi, hemi)] = np.isin(sub_data, labels)

    return masks, affine


def resample_masks_to_bold(masks, atlas_affine, bold_img):
    """Resample atlas-space masks to BOLD voxel space (no-op if already aligned)."""
    import nibabel as nib
    import numpy as np
    from nilearn.image import resample_to_img

    bold_shape = bold_img.shape[:3]
    atlas_shape = next(iter(masks.values())).shape
    if atlas_shape == bold_shape and np.allclose(atlas_affine, bold_img.affine):
        return masks
    out = {}
    for key, m in masks.items():
        m_img = nib.Nifti1Image(m.astype(np.float32), atlas_affine)
        out[key] = np.asarray(resample_to_img(m_img, bold_img, interpolation="nearest").dataobj) > 0.5
    return out
