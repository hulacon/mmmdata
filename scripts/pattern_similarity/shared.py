"""Shared constants and helpers for the pattern-similarity analysis.

Comparison grid: {fmriprep, fmriprep_nordic} x {rawTR-model8, GLMsingle}
x {TBencoding, NATencoding} x 6 bilateral Harvard-Oxford ROIs x 4 cells
(within/across subject x same/different item).

Plan: docs/doc/pattern-similarity-plan.md. Builds on
scripts/nordic_benchmark/shared.py (imported below as `nb`).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

# ── import nordic_benchmark/shared.py without sys.path/name collisions ───

_NB_SHARED = Path(__file__).resolve().parent.parent / "nordic_benchmark" / "shared.py"
_spec = importlib.util.spec_from_file_location("nordic_benchmark_shared", _NB_SHARED)
nb = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = nb  # dataclasses looks the module up in sys.modules
_spec.loader.exec_module(nb)

BIDS_ROOT = nb.BIDS_ROOT
DERIV_ROOT = nb.DERIV_ROOT
SUBJECTS = nb.SUBJECTS
TB_SESSIONS = nb.TB_SESSIONS
NAT_SESSIONS = nb.NAT_SESSIONS
TB_RUNS = nb.TB_RUNS
NAT_RUNS = nb.NAT_RUNS
TR = nb.TR
MNI_SPACE = nb.MNI_SPACE
PIPELINES = nb.PIPELINES
GLMSINGLE_DIRS = nb.GLMSINGLE_DIRS

bold_path = nb.bold_path
mask_path = nb.mask_path
events_path = nb.events_path
confounds_path = nb.confounds_path
read_tb_trials = nb.read_tb_trials
load_trial_metadata = nb.load_trial_metadata
glmsingle_outputs_dir = nb.glmsingle_outputs_dir
resample_masks_to_bold = nb.resample_masks_to_bold

# ── analysis constants ───────────────────────────────────────────────────

# NAT GLMsingle fits (produced by scripts/glmsingle_natencoding.py, Phase 2)
GLMSINGLE_NAT_DIRS = {
    "original": DERIV_ROOT / "glmsingle_nat",
    "nordic":   DERIV_ROOT / "glmsingle_nat_nordic",
}

PS_ROOT = DERIV_ROOT / "pattern_similarity"
CACHE_DIR = PS_ROOT / "cache"
RESULTS_DIR = PS_ROOT / "results"
QC_DIR = PS_ROOT / "qc"
FIGURES_DIR = PS_ROOT / "figures"

HRF_SHIFT_TRS = 3          # +4.5 s hemodynamic shift, both paradigms (plan D2)
SEED = 20260810            # RNG seed for different-item draws (plan D4)
N_DIFF_DRAWS = 10          # surrogate draws per same-item pair (plan D4)
NAT_CHUNK_S = 4.5          # NAT GLMsingle pseudo-trial duration (plan D7)
NAT_CHUNK_TRS = 3          # 4.5 s / 1.5 s TR — exact 3-TR stride

REPEATED_MOVIES = ["The Bench", "From Dad To Son"]  # condition==3, 1x/session

# Tidy result schema for results/similarity_summary.tsv
RESULT_COLS = [
    "paradigm", "pipeline", "timeseries", "roi", "cell", "unit", "item",
    "r", "n_pairs", "n_voxels", "n_draws", "confound_model",
    "session_scope", "context_match",
]

# ── bilateral Harvard-Oxford ROIs (plan D1) ──────────────────────────────

# label value -> required substring of the atlas label name (self-check)
PATTERN_CORTICAL_ROIS = {
    "EVC":       {24: "Intracalcarine"},
    "EAC":       {45: "Heschl"},
    "AG":        {21: "Angular Gyrus"},
    "Precuneus": {31: "Precuneous"},
    "mPFC":      {25: "Frontal Medial"},
}
PATTERN_SUBCORTICAL_ROIS = {
    "Hippocampus": {9: "Left Hippocampus", 19: "Right Hippocampus"},
}
PATTERN_ROI_NAMES = ["EVC", "EAC", "Hippocampus", "AG", "Precuneus", "mPFC"]


def load_bilateral_roi_masks(split_hemi: bool = False):
    """Build the 6 bilateral Harvard-Oxford ROI masks (maxprob-thr25, MNI 2mm).

    Label values are asserted against the atlas label names at load time so a
    nilearn atlas-version change cannot silently swap regions.

    Returns:
        masks: {roi_name: bool ndarray} — or {(roi_name, hemi): ...} when
               split_hemi=True (midline mPFC/Precuneus split by x<0 like the
               nordic benchmark's cortical ROIs; interpret with care)
        affine: 4x4 atlas affine
    """
    import nibabel as nib
    import numpy as np
    from nilearn.datasets import fetch_atlas_harvard_oxford

    def _load(atlas_name):
        atlas = fetch_atlas_harvard_oxford(atlas_name)
        img = atlas.maps if hasattr(atlas.maps, "affine") else nib.load(atlas.maps)
        labels = list(atlas.labels)
        return np.asarray(img.dataobj).astype(int), img.affine, labels

    cort_data, affine, cort_labels = _load("cort-maxprob-thr25-2mm")
    sub_data, sub_affine, sub_labels = _load("sub-maxprob-thr25-2mm")
    assert np.allclose(affine, sub_affine), "cortical/subcortical atlas grids differ"

    def _checked_mask(data, labels, spec):
        values = []
        for value, expect in spec.items():
            actual = labels[value]
            if expect not in actual:
                raise ValueError(
                    f"Harvard-Oxford label {value} is {actual!r}, expected ~{expect!r}"
                )
            values.append(value)
        return np.isin(data, values)

    masks = {}
    for roi, spec in PATTERN_CORTICAL_ROIS.items():
        masks[roi] = _checked_mask(cort_data, cort_labels, spec)
    for roi, spec in PATTERN_SUBCORTICAL_ROIS.items():
        masks[roi] = _checked_mask(sub_data, sub_labels, spec)

    if split_hemi:
        i_coords = np.arange(cort_data.shape[0])
        x_coords = affine[0, 0] * i_coords + affine[0, 3]
        left = (x_coords < 0)[:, None, None]
        masks = {
            (roi, hemi): m & (left if hemi == "L" else ~left)
            for roi, m in masks.items()
            for hemi in ("L", "R")
        }
    return masks, affine


# ── shared-item registry (TBencoding) ────────────────────────────────────

def shared_item_registry(sub: str):
    """Per-subject registry of the 6 shared TB items, discovered from events.

    Every subject sees the same 6-item pool (mmmId 995-1000); a per-subject
    design shuffle assigns 3 to the contiguous triplet (pairId 1-3, enCon==3)
    and 3 to super-repeat singles (pairId 75-77, enCon==2). See plan
    "Shared-item structure".

    Returns a DataFrame with columns:
        subject, pairId, mmmId, word, role ("triplet"|"single"),
        position (1-3 within-triplet order; 0 for singles), n_presentations
    Raises if the mapping is not constant across sessions/runs.
    """
    import pandas as pd

    frames = []
    for ses in TB_SESSIONS:
        for run in TB_RUNS:
            df = pd.read_csv(events_path(sub, ses, run, "TBencoding"), sep="\t")
            sel = df[(df["trial_type"] == "image") & (df["sharedId"] == 1)]
            frames.append(sel[["pairId", "mmmId", "word", "enCon"]])
    allrows = pd.concat(frames, ignore_index=True)

    per_pair = allrows.groupby("pairId")[["mmmId", "word", "enCon"]].nunique()
    unstable = per_pair[(per_pair != 1).any(axis=1)]
    if len(unstable):
        raise ValueError(f"{sub}: shared-item mapping varies across sessions:\n{unstable}")

    reg = (
        allrows.groupby("pairId")
        .agg(mmmId=("mmmId", "first"), word=("word", "first"),
             enCon=("enCon", "first"), n_presentations=("mmmId", "size"))
        .reset_index()
    )
    for col in ("pairId", "mmmId", "enCon"):  # float via NaN-padded events dtype
        reg[col] = reg[col].astype(int)
    reg["subject"] = sub
    reg["role"] = reg["enCon"].map({3: "triplet", 2: "single"})
    reg["position"] = reg["pairId"].where(reg["role"] == "triplet", 0)
    return reg[["subject", "pairId", "mmmId", "word", "role", "position",
                "n_presentations"]]
