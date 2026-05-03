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

Output:
  derivatives/nordic/benchmark/nat/pair_correlations.tsv
    columns: subject (="all"), session, stream=NAT, roi, pipeline,
             within_r, between_r, discriminability, n_within, n_between
"""

from __future__ import annotations

import argparse
import csv
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
    bold_path, events_path, load_roi_masks, mask_path, resample_masks_to_bold,
)


def extract_movie_timeseries_for_run(
    bold_p: Path, brain_mask_p: Path, events_p: Path,
    roi_masks: dict, atlas_affine,
) -> dict[str, dict[tuple[str, str], np.ndarray]]:
    """For one run, extract per-movie per-ROI z-scored timeseries.

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
    ses: str, pipeline: str, roi_masks: dict, atlas_affine,
) -> dict[tuple[str, str], dict[tuple[str, str], np.ndarray]]:
    """Aggregate across 3 subjects × 2 runs into {(subject, movie): {roi: ts}}.

    Each (subject, movie) entity is unique within session.
    """
    out: dict[tuple[str, str], dict[tuple[str, str], np.ndarray]] = {}
    for sub in SUBJECTS:
        for run in NAT_RUNS:
            bold_p = bold_path(sub, ses, run, "NATencoding", pipeline)
            mask_p = mask_path(sub, ses, run, "NATencoding", pipeline)
            evt_p = events_path(sub, ses, run, "NATencoding")
            if not (bold_p.exists() and mask_p.exists() and evt_p.exists()):
                print(f"  MISSING: {sub} {ses} run-{run:02d} {pipeline}")
                continue
            run_movies = extract_movie_timeseries_for_run(
                bold_p, mask_p, evt_p, roi_masks, atlas_affine,
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


def evaluate(pipelines: list[str], output_root: Path) -> Path:
    print("Loading Harvard-Oxford ROI masks...")
    masks_atlas, atlas_affine = load_roi_masks()

    rows = []
    for pipeline in pipelines:
        for ses in NAT_SESSIONS:
            t0 = time.time()
            print(f"\n[{pipeline}] {ses}: loading and segmenting movies...")
            block = build_session_block(ses, pipeline, masks_atlas, atlas_affine)
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
                })
            print(f"  {n_entities} entities, {time.time()-t0:.0f}s")

    out = output_root / "nat" / "pair_correlations.tsv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=SUMMARY_COLS, delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {out} ({len(rows)} rows)")

    df = pd.DataFrame(rows)
    if not df.empty:
        print("\nMean discriminability per ROI × pipeline:")
        piv = df.groupby(["roi", "pipeline"])["discriminability"].mean().unstack().round(4)
        print(piv)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pipeline", choices=["original", "nordic", "both"], default="both")
    ap.add_argument("--output-root", type=Path, default=BENCHMARK_OUT)
    args = ap.parse_args()

    pipes = ["original", "nordic"] if args.pipeline == "both" else [args.pipeline]
    evaluate(pipes, args.output_root)
    print("\nDone.")


if __name__ == "__main__":
    main()
