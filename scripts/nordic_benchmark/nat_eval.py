#!/usr/bin/env python3
"""NAT benchmark evaluation: within-movie vs between-movie cross-subject ISC.

For each NAT session (ses-19..ses-28), for each pipeline (original, nordic):
  1. Load all 6 NATencoding BOLDs (3 subjects × 2 runs)
  2. Per ROI per subject, extract mean ROI timeseries
  3. Segment by movie using events.tsv → per-movie z-scored timeseries per
     (subject, movie)
  4. Within-movie pairs: corr(subj_A_movie_X, subj_B_movie_X) over all
     unordered subject pairs and movies (24 pairs/session: 8 movies × 3 pairs)
  5. Between-movie pairs: corr(subj_A_movie_X, subj_B_movie_Y) for X != Y,
     all subject permutations (168 pairs/session)
  6. Truncate to min length for between-movie pairs (movies have different
     durations).

Confound regression (optional): per-run OLS residualization of the ROI mean
timeseries before movie segmentation. Supports the no-regression baseline
("none") and Model 8 ("8" = 6 HM + WM + CSF + GS + cosine), the model selected
by the ISC confound comparison (mmmdata-agents docs/nordic-and-isc-benchmarks.md
§4). The regression reuses
Nastase & Hasson's extract_confounds utilities and the shared model_meta.json,
matching how scripts/isc_confounds/isc_benchmark.py applies Model 8.

Output:
  derivatives/nordic/benchmark/nat/pair_correlations.tsv
    columns: subject (="all"), session, stream=NAT, roi, pipeline,
             within_r, between_r, discriminability, n_within, n_between,
             confound_model
  derivatives/nordic/benchmark/nat/confound_comparison.tsv
    mean discriminability per roi × pipeline × confound_model (tidy summary)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from itertools import combinations, product
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from shared import (  # noqa: E402
    BENCHMARK_OUT, NAT_RUNS, NAT_SESSIONS, ROI_KEYS, SUBJECTS, SUMMARY_COLS, TR,
    bold_path, confounds_path, events_path, load_roi_masks, mask_path,
    resample_masks_to_bold,
)

# ── Nastase & Hasson confound utilities (Model 8 support) ────────────────
# natsort is not installed in the analysis env; install a minimal shim before
# importing extract_confounds (mirrors scripts/isc_confounds/isc_benchmark.py).
import re as _re
import types as _types


def _natsorted(seq, **_kwargs):
    def _key(s):
        return [int(c) if c.isdigit() else c.lower()
                for c in _re.split(r"(\d+)", str(s))]
    return sorted(seq, key=_key)


if "natsort" not in sys.modules:
    _natsort_shim = _types.ModuleType("natsort")
    _natsort_shim.natsorted = _natsorted
    sys.modules["natsort"] = _natsort_shim

_SCRIPT_DIR = Path(__file__).resolve().parent
_ISC_CONFOUNDS_CODE = _SCRIPT_DIR.parent.parent.parent / "isc-confounds"
_MODEL_META_PATH = _SCRIPT_DIR.parent / "isc_confounds" / "model_meta.json"
sys.path.insert(0, str(_ISC_CONFOUNDS_CODE))
from extract_confounds import extract_confounds, load_confounds  # noqa: E402

with open(_MODEL_META_PATH) as _fh:
    MODEL_META = json.load(_fh)

# CLI value → (output label, model_meta key). None key = no regression.
CONFOUND_CHOICES = {
    "none": ("none", None),
    "8": ("model-08", "8"),
}

# Output columns = unified schema + the confound_model dimension.
NAT_COLS = SUMMARY_COLS + ["confound_model"]


def regress_confounds(ts: np.ndarray, confound_matrix) -> np.ndarray:
    """OLS-residualize a 1D timeseries against a confound design (intercept added)."""
    if confound_matrix is None or confound_matrix.shape[1] == 0:
        return ts
    X = np.nan_to_num(np.asarray(confound_matrix, dtype=np.float64), nan=0.0)
    X = np.column_stack([np.ones(X.shape[0]), X])
    betas = np.linalg.lstsq(X, ts, rcond=None)[0]
    return ts - X @ betas


def confound_matrix_for_run(confounds_p: Path, model_id):
    """Build the (T × k) confound design for a run, or None for the baseline."""
    if model_id is None:
        return None
    df, meta = load_confounds(str(confounds_p))
    return extract_confounds(df, meta, MODEL_META[model_id]).values


def extract_movie_timeseries_for_run(
    bold_p: Path, brain_mask_p: Path, events_p: Path,
    roi_masks: dict, atlas_affine, confounds_p: Path | None = None,
    model_id=None,
) -> dict[str, dict[tuple[str, str], np.ndarray]]:
    """For one run, extract per-movie per-ROI z-scored timeseries.

    If model_id is not None, the ROI mean timeseries are OLS-residualized
    against the run's confound design (from confounds_p) at the full-run level
    before movie segmentation and per-movie z-scoring.

    Returns: {movie_name: {(roi, hemi): 1D z-scored timeseries}}
    """
    bold_img = nib.load(str(bold_p))
    brain_mask = np.asarray(nib.load(str(brain_mask_p)).dataobj) > 0
    masks_resampled = resample_masks_to_bold(roi_masks, atlas_affine, bold_img)
    bold = bold_img.get_fdata(dtype=np.float32)  # (X, Y, Z, T)

    # Per-ROI mean timeseries (intersect with brain mask)
    roi_ts = {}
    for k in ROI_KEYS:
        m = masks_resampled[k] & brain_mask
        if m.sum() == 0:
            roi_ts[k] = None
        else:
            roi_ts[k] = bold[m].mean(axis=0)  # (T,)

    # Confound regression at the full-run level (before segmentation)
    if model_id is not None:
        cmat = confound_matrix_for_run(confounds_p, model_id)
        for k, ts in roi_ts.items():
            if ts is not None:
                roi_ts[k] = regress_confounds(
                    ts.astype(np.float64), cmat
                ).astype(np.float32)

    # Movie segments
    events = pd.read_csv(events_p, sep="\t")
    movies = events[events["trial_type"] == "movie"]
    out = {}
    for _, row in movies.iterrows():
        name = row["movie_name"]
        onset_tr = int(np.round(row["onset"] / TR))
        dur_tr = int(np.round(row["duration"] / TR))
        movie_block = {}
        for k, ts in roi_ts.items():
            if ts is None:
                movie_block[k] = None
                continue
            end = min(onset_tr + dur_tr, len(ts))
            seg = ts[onset_tr:end].copy()
            if seg.std() > 0:
                seg = (seg - seg.mean()) / seg.std()
            movie_block[k] = seg
        out[name] = movie_block
    return out


def build_session_block(
    ses: str, pipeline: str, roi_masks: dict, atlas_affine, model_id=None,
) -> dict[tuple[str, str], dict[tuple[str, str], np.ndarray]]:
    """Aggregate across 3 subjects × 2 runs into {(subject, movie): {roi: ts}}.

    Each (subject, movie) entity is unique within session. If model_id is not
    None, ROI timeseries are confound-regressed per run before segmentation.
    """
    out: dict[tuple[str, str], dict[tuple[str, str], np.ndarray]] = {}
    for sub in SUBJECTS:
        for run in NAT_RUNS:
            bold_p = bold_path(sub, ses, run, "NATencoding", pipeline)
            mask_p = mask_path(sub, ses, run, "NATencoding", pipeline)
            evt_p = events_path(sub, ses, run, "NATencoding")
            conf_p = confounds_path(sub, ses, run, "NATencoding", pipeline)
            needed = [bold_p, mask_p, evt_p]
            if model_id is not None:
                needed.append(conf_p)
            if not all(p.exists() for p in needed):
                print(f"  MISSING: {sub} {ses} run-{run:02d} {pipeline}")
                continue
            run_movies = extract_movie_timeseries_for_run(
                bold_p, mask_p, evt_p, roi_masks, atlas_affine,
                confounds_p=conf_p, model_id=model_id,
            )
            for movie_name, roi_block in run_movies.items():
                key = (sub, movie_name)
                # If a movie spans run boundaries (shouldn't for NAT), the
                # second extraction would overwrite; guard anyway.
                if key in out:
                    continue
                out[key] = roi_block
    return out


def session_pair_stats(
    block: dict[tuple[str, str], dict[tuple[str, str], np.ndarray]],
    roi_key: tuple[str, str],
) -> tuple[float, float, int, int]:
    """For one session × one ROI, compute mean within-movie and between-movie r."""
    # Build entity list with valid timeseries for this ROI
    entities = []
    for (sub, mov), roi_block in block.items():
        ts = roi_block.get(roi_key)
        if ts is None or len(ts) < 4 or not np.isfinite(ts).all():
            continue
        entities.append((sub, mov, ts))
    if len(entities) < 2:
        return (np.nan, np.nan, 0, 0)

    within_rs, between_rs = [], []
    for (sa, ma, ta), (sb, mb, tb) in combinations(entities, 2):
        if sa == sb:
            continue  # different subjects required
        L = min(len(ta), len(tb))
        if L < 4:
            continue
        x = ta[:L]; y = tb[:L]
        if x.std() == 0 or y.std() == 0:
            continue
        r = float(np.corrcoef(x, y)[0, 1])
        if ma == mb:
            within_rs.append(r)
        else:
            between_rs.append(r)
    within_r = float(np.mean(within_rs)) if within_rs else np.nan
    between_r = float(np.mean(between_rs)) if between_rs else np.nan
    return (within_r, between_r, len(within_rs), len(between_rs))


def evaluate(pipelines: list[str], confound_models: list[tuple[str, object]],
             output_root: Path) -> Path:
    print("Loading Harvard-Oxford ROI masks...")
    masks_atlas, atlas_affine = load_roi_masks()

    rows = []
    for model_label, model_id in confound_models:
        for pipeline in pipelines:
            for ses in NAT_SESSIONS:
                t0 = time.time()
                print(f"\n[confound={model_label}] [{pipeline}] {ses}: "
                      "loading and segmenting movies...")
                block = build_session_block(
                    ses, pipeline, masks_atlas, atlas_affine, model_id=model_id)
                n_entities = len(block)
                if n_entities == 0:
                    print(f"  no entities for {ses} / {pipeline}, skipping")
                    continue
                for roi_name, hemi in ROI_KEYS:
                    within_r, between_r, n_w, n_b = session_pair_stats(block, (roi_name, hemi))
                    disc = within_r - between_r if np.isfinite(within_r) and np.isfinite(between_r) else np.nan
                    rows.append({
                        "subject": "all",
                        "session": ses,
                        "stream": "NAT",
                        "roi": f"{roi_name}_{hemi}",
                        "pipeline": pipeline,
                        "within_r": within_r,
                        "between_r": between_r,
                        "discriminability": disc,
                        "n_within": n_w,
                        "n_between": n_b,
                        "confound_model": model_label,
                    })
                print(f"  {n_entities} entities, {time.time()-t0:.0f}s")

    out = output_root / "nat" / "pair_correlations.tsv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=NAT_COLS, delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {out} ({len(rows)} rows)")

    df = pd.DataFrame(rows)
    if not df.empty:
        # Tidy per-ROI × pipeline × confound_model discriminability summary
        summary = (df.groupby(["roi", "pipeline", "confound_model"])
                     ["discriminability"].mean().round(5).reset_index())
        comp_out = output_root / "nat" / "confound_comparison.tsv"
        summary.to_csv(comp_out, sep="\t", index=False)
        print(f"Wrote {comp_out} ({len(summary)} rows)")

        print("\nMean discriminability per ROI × pipeline × confound_model:")
        piv = df.groupby(["roi", "pipeline", "confound_model"])["discriminability"].mean().unstack([1, 2]).round(4)
        print(piv)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pipeline", choices=["original", "nordic", "both"], default="both")
    ap.add_argument("--confound-model", choices=["none", "8", "both"], default="both",
                    help="Confound regression: 'none' (raw), '8' (Model 8: "
                         "6HM + WM/CSF + GS + cosine), or 'both' (default).")
    ap.add_argument("--output-root", type=Path, default=BENCHMARK_OUT)
    args = ap.parse_args()

    pipes = ["original", "nordic"] if args.pipeline == "both" else [args.pipeline]
    if args.confound_model == "both":
        cmods = [CONFOUND_CHOICES["none"], CONFOUND_CHOICES["8"]]
    else:
        cmods = [CONFOUND_CHOICES[args.confound_model]]
    evaluate(pipes, cmods, args.output_root)
    print("\nDone.")


if __name__ == "__main__":
    main()
