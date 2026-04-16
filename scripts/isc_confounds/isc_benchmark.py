#!/usr/bin/env python3
"""
isc_benchmark.py — Benchmark MMMData ISC against Pixar (ds000228) adults.

Computes ISC distributions for two confound models (0=baseline, 8=6HM+WM/CSF/GS)
across two conditions:
  - full-run:    Pixar full 168 TRs, all C(33,3) = 5,456 three-subject combos
  - tr-matched:  Center-cropped to match mean MMMData clip duration in seconds

MMMData ISC is computed per movie clip (extracted via events TSVs), then averaged
to produce per-ROI point estimates for overlay on the benchmark distributions.

Input:  derivatives/qc/isc_benchmark/{mmmdata,pixar}/ (.npz + confounds)
Output: derivatives/qc/isc_benchmark/results/

Requires: neuroconda3 environment
"""

import argparse
import json
import re
import sys
import types
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

# ── natsort shim ─────────────────────────────────────────────────────────
def _natsorted(seq, **kwargs):
    def _key(s):
        return [int(c) if c.isdigit() else c.lower()
                for c in re.split(r'(\d+)', str(s))]
    return sorted(seq, key=_key)

natsort_shim = types.ModuleType("natsort")
natsort_shim.natsorted = _natsorted
sys.modules["natsort"] = natsort_shim

# ── Nastase confound utilities ───────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
ISC_CONFOUNDS_DIR = SCRIPT_DIR.parent.parent.parent / "isc-confounds"
sys.path.insert(0, str(ISC_CONFOUNDS_DIR))
from extract_confounds import load_confounds, extract_confounds

with open(SCRIPT_DIR / "model_meta.json") as _f:
    model_meta = json.load(_f)

# ── paths ────────────────────────────────────────────────────────────────
BIDS_ROOT = Path("/gpfs/projects/hulacon/shared/mmmdata")
BENCHMARK_DIR = BIDS_ROOT / "derivatives" / "qc" / "isc_benchmark"

MMMDATA_DIR = BENCHMARK_DIR / "mmmdata"
PIXAR_DIR = BENCHMARK_DIR / "pixar"

MMMDATA_SUBJECTS = ["sub-03", "sub-04", "sub-05"]
MMMDATA_SESSIONS = [f"ses-{i}" for i in range(19, 29)]
MMMDATA_RUNS = ["run-01", "run-02"]
MMMDATA_TR = 1.5

PIXAR_ADULTS = [f"sub-pixar{i:03d}" for i in range(123, 156)]
PIXAR_TR = 2.0

ROI_KEYS = ["V1_L", "V1_R", "EAC_L", "EAC_R", "MT+_L", "MT+_R",
            "IFG_L", "IFG_R", "Hippocampus_L", "Hippocampus_R"]

MODELS = ["0", "8"]

REPEATED_MOVIES = {"the bench", "from dad to son"}


# ── confound regression ─────────────────────────────────────────────────

def regress_confounds(ts, confound_matrix):
    """Remove confounds from a 1D timeseries via OLS. Returns residuals."""
    if confound_matrix is None or confound_matrix.shape[1] == 0:
        return ts
    X = np.nan_to_num(confound_matrix.copy(), nan=0.0)
    X = np.column_stack([np.ones(X.shape[0]), X])
    betas = np.linalg.lstsq(X, ts, rcond=None)[0]
    return ts - X @ betas


def get_confound_matrix(confounds_df, confounds_meta, model_id):
    """Extract confound matrix for a model, or None for model 0."""
    spec = model_meta[model_id]
    if not spec["confounds"] and not any(
        k in spec for k in ["a_comp_cor", "t_comp_cor", "c_comp_cor", "w_comp_cor"]
    ):
        return None
    return extract_confounds(confounds_df, confounds_meta, spec).values


# ── data loading ─────────────────────────────────────────────────────────

def load_pixar_data():
    """Load all Pixar adult ROI timeseries and confounds.

    Returns:
        ts_by_sub: {subject: {roi_key: 1D array}}
        conf_by_sub: {subject: (confounds_df, confounds_meta)}
    """
    ts_by_sub = {}
    conf_by_sub = {}

    for sub in PIXAR_ADULTS:
        npz_path = PIXAR_DIR / sub / f"{sub}_task-pixar_roi-timeseries.npz"
        conf_path = PIXAR_DIR / sub / f"{sub}_task-pixar_confounds.tsv"
        if not npz_path.exists():
            print(f"  WARNING: missing {sub}, skipping")
            continue

        data = np.load(str(npz_path))
        ts_by_sub[sub] = {k: data[k].astype(np.float64) for k in ROI_KEYS}
        conf_by_sub[sub] = load_confounds(str(conf_path))

    print(f"  Loaded {len(ts_by_sub)} Pixar subjects")
    return ts_by_sub, conf_by_sub


def load_mmmdata_run(sub, ses, run):
    """Load one MMMData run's timeseries and confounds."""
    stem = f"{sub}_{ses}_task-NATencoding_{run}"
    npz_path = MMMDATA_DIR / sub / f"{stem}_roi-timeseries.npz"
    conf_path = MMMDATA_DIR / sub / f"{stem}_confounds.tsv"

    if not npz_path.exists():
        return None, None

    data = np.load(str(npz_path))
    ts = {k: data[k].astype(np.float64) for k in ROI_KEYS}
    conf = load_confounds(str(conf_path))
    return ts, conf


# ── Pixar ISC ────────────────────────────────────────────────────────────

def compute_pixar_isc(ts_by_sub, conf_by_sub, n_trs_matched):
    """Compute ISC distributions for all C(n,3) Pixar subsamples.

    Returns DataFrame with columns:
        model, condition, roi, triplet, mean_r
    """
    subjects = sorted(ts_by_sub.keys())
    triplets = list(combinations(subjects, 3))
    n_triplets = len(triplets)
    print(f"  {n_triplets} triplets from {len(subjects)} subjects")

    # Pre-compute confound-regressed timeseries per model
    # ts_clean[model][subject][roi] = 1D array
    ts_clean = {}
    for model_id in MODELS:
        ts_clean[model_id] = {}
        for sub in subjects:
            raw_ts = ts_by_sub[sub]
            conf_df, conf_meta = conf_by_sub[sub]
            cmat = get_confound_matrix(conf_df, conf_meta, model_id)

            ts_clean[model_id][sub] = {}
            for roi in ROI_KEYS:
                ts_clean[model_id][sub][roi] = regress_confounds(raw_ts[roi], cmat)

    # Compute ISC for each triplet
    rows = []
    for ti, triplet in enumerate(triplets):
        if ti % 1000 == 0 and ti > 0:
            print(f"    triplet {ti}/{n_triplets}")

        for model_id in MODELS:
            for condition, n_trs in [("full-run", None), ("tr-matched", n_trs_matched)]:
                for roi in ROI_KEYS:
                    # Get timeseries for this triplet
                    ts_list = []
                    for sub in triplet:
                        t = ts_clean[model_id][sub][roi]
                        if n_trs is not None:
                            # Center-crop
                            start = (len(t) - n_trs) // 2
                            t = t[start:start + n_trs]
                        ts_list.append(t)

                    # Z-score each
                    ts_z = []
                    for t in ts_list:
                        if t.std() > 0:
                            ts_z.append((t - t.mean()) / t.std())
                        else:
                            ts_z.append(t)

                    # Mean of 3 pairwise correlations
                    pairs = list(combinations(range(3), 2))
                    rs = []
                    for i, j in pairs:
                        min_len = min(len(ts_z[i]), len(ts_z[j]))
                        if min_len < 10:
                            continue
                        r = np.corrcoef(ts_z[i][:min_len], ts_z[j][:min_len])[0, 1]
                        if np.isfinite(r):
                            rs.append(r)
                    if rs:
                        rows.append({
                            "model": model_id,
                            "condition": condition,
                            "roi": roi,
                            "triplet": f"{triplet[0]}|{triplet[1]}|{triplet[2]}",
                            "mean_r": np.mean(rs),
                        })

    df = pd.DataFrame(rows)
    print(f"  Pixar ISC: {len(df)} rows")
    return df


# ── MMMData ISC ──────────────────────────────────────────────────────────

def compute_mmmdata_isc():
    """Compute ISC for MMMData NATencoding using volumetric ROI timeseries.

    For each session × run × movie: compute pairwise ISC between all 3 subjects
    on the movie segment (extracted via events TSV). Then average across all
    observations to get per-ROI point estimates.

    Returns DataFrame with columns:
        model, roi, movie, session, run, subject_pair, r
    """
    rows = []

    for ses in MMMDATA_SESSIONS:
        for run in MMMDATA_RUNS:
            # Load all 3 subjects for this session/run
            all_ts = {}
            all_conf = {}
            events_df = None

            for sub in MMMDATA_SUBJECTS:
                ts, conf = load_mmmdata_run(sub, ses, run)
                if ts is None:
                    break
                all_ts[sub] = ts
                all_conf[sub] = conf

                # Load events (same structure for all subjects)
                if events_df is None:
                    events_path = (
                        BIDS_ROOT / sub / ses / "func"
                        / f"{sub}_{ses}_task-NATencoding_{run}_events.tsv"
                    )
                    if events_path.exists():
                        events_df = pd.read_csv(str(events_path), sep="\t")

            if len(all_ts) < 3 or events_df is None:
                continue

            # Extract movie segments
            movies = events_df[events_df["trial_type"] == "movie"]

            for model_id in MODELS:
                # Regress confounds for each subject (full run)
                clean_ts = {}
                for sub in MMMDATA_SUBJECTS:
                    conf_df, conf_meta = all_conf[sub]
                    cmat = get_confound_matrix(conf_df, conf_meta, model_id)
                    clean_ts[sub] = {}
                    for roi in ROI_KEYS:
                        clean_ts[sub][roi] = regress_confounds(
                            all_ts[sub][roi], cmat
                        )

                # For each movie, extract segment and compute ISC
                ses_idx = MMMDATA_SESSIONS.index(ses) + 1  # 1-indexed
                for _, mrow in movies.iterrows():
                    movie_name = mrow["movie_name"]
                    movie_key = movie_name.lower()
                    onset_vol = int(np.round(mrow["onset"] / MMMDATA_TR))
                    dur_vols = int(np.round(mrow["duration"] / MMMDATA_TR))

                    # Label repeated movies with session index
                    if movie_key in REPEATED_MOVIES:
                        movie_label = f"{movie_key} {ses_idx}"
                    else:
                        movie_label = movie_key

                    for roi in ROI_KEYS:
                        # Extract & z-score per subject
                        segments = {}
                        for sub in MMMDATA_SUBJECTS:
                            full = clean_ts[sub][roi]
                            end_vol = min(onset_vol + dur_vols, len(full))
                            seg = full[onset_vol:end_vol].copy()
                            if seg.std() > 0:
                                seg = (seg - seg.mean()) / seg.std()
                            segments[sub] = seg

                        # Pairwise correlations
                        for s1, s2 in combinations(MMMDATA_SUBJECTS, 2):
                            min_len = min(len(segments[s1]), len(segments[s2]))
                            if min_len < 10:
                                continue
                            r = np.corrcoef(
                                segments[s1][:min_len],
                                segments[s2][:min_len]
                            )[0, 1]
                            if np.isfinite(r):
                                rows.append({
                                    "model": model_id,
                                    "roi": roi,
                                    "movie": movie_label,
                                    "session": ses,
                                    "run": run,
                                    "subject_pair": f"{s1}|{s2}",
                                    "r": r,
                                })

    df = pd.DataFrame(rows)
    print(f"  MMMData ISC: {len(df)} rows")
    return df


# ── signal-averaged ISC for repeated movies ──────────────────────────────

def compute_mmmdata_averaged_isc():
    """Compute ISC on timeseries averaged across 10 repetitions of repeated movies.

    For each repeated movie (The Bench, From Dad To Son):
      1. For each subject × model: extract the movie segment from each of the
         10 sessions (after confound regression), then average across sessions.
      2. Compute pairwise ISC on the session-averaged timeseries.

    This tests whether signal averaging across repetitions improves ISC.

    Returns DataFrame with columns:
        model, roi, movie, subject_pair, r
    """
    rows = []

    # First pass: collect all segments for repeated movies
    # segments[model][movie_key][subject] = list of 1D arrays (one per session)
    segments = {m: {mk: {s: [] for s in MMMDATA_SUBJECTS}
                    for mk in REPEATED_MOVIES}
                for m in MODELS}

    for ses in MMMDATA_SESSIONS:
        for run in MMMDATA_RUNS:
            all_ts = {}
            all_conf = {}
            events_df = None

            for sub in MMMDATA_SUBJECTS:
                ts, conf = load_mmmdata_run(sub, ses, run)
                if ts is None:
                    break
                all_ts[sub] = ts
                all_conf[sub] = conf

                if events_df is None:
                    events_path = (
                        BIDS_ROOT / sub / ses / "func"
                        / f"{sub}_{ses}_task-NATencoding_{run}_events.tsv"
                    )
                    if events_path.exists():
                        events_df = pd.read_csv(str(events_path), sep="\t")

            if len(all_ts) < 3 or events_df is None:
                continue

            movies = events_df[events_df["trial_type"] == "movie"]

            for model_id in MODELS:
                clean_ts = {}
                for sub in MMMDATA_SUBJECTS:
                    conf_df, conf_meta = all_conf[sub]
                    cmat = get_confound_matrix(conf_df, conf_meta, model_id)
                    clean_ts[sub] = {}
                    for roi in ROI_KEYS:
                        clean_ts[sub][roi] = regress_confounds(
                            all_ts[sub][roi], cmat
                        )

                for _, mrow in movies.iterrows():
                    movie_key = mrow["movie_name"].lower()
                    if movie_key not in REPEATED_MOVIES:
                        continue

                    onset_vol = int(np.round(mrow["onset"] / MMMDATA_TR))
                    dur_vols = int(np.round(mrow["duration"] / MMMDATA_TR))

                    for sub in MMMDATA_SUBJECTS:
                        # Store per-ROI segments as a dict for this session
                        seg_dict = {}
                        for roi in ROI_KEYS:
                            full = clean_ts[sub][roi]
                            end_vol = min(onset_vol + dur_vols, len(full))
                            seg_dict[roi] = full[onset_vol:end_vol].copy()
                        segments[model_id][movie_key][sub].append(seg_dict)

    # Second pass: average across sessions and compute ISC
    for model_id in MODELS:
        for movie_key in REPEATED_MOVIES:
            for roi in ROI_KEYS:
                # Average timeseries across 10 sessions per subject
                averaged = {}
                for sub in MMMDATA_SUBJECTS:
                    segs = [s[roi] for s in segments[model_id][movie_key][sub]]
                    if not segs:
                        continue
                    # Truncate to shortest segment length (should be equal)
                    min_len = min(len(s) for s in segs)
                    stacked = np.array([s[:min_len] for s in segs])
                    avg_ts = stacked.mean(axis=0)
                    # Z-score the averaged timeseries
                    if avg_ts.std() > 0:
                        avg_ts = (avg_ts - avg_ts.mean()) / avg_ts.std()
                    averaged[sub] = avg_ts

                if len(averaged) < 2:
                    continue

                # Pairwise ISC
                for s1, s2 in combinations(MMMDATA_SUBJECTS, 2):
                    if s1 not in averaged or s2 not in averaged:
                        continue
                    min_len = min(len(averaged[s1]), len(averaged[s2]))
                    if min_len < 10:
                        continue
                    r = np.corrcoef(
                        averaged[s1][:min_len],
                        averaged[s2][:min_len]
                    )[0, 1]
                    if np.isfinite(r):
                        rows.append({
                            "model": model_id,
                            "roi": roi,
                            "movie": f"{movie_key} (avg 10 sessions)",
                            "subject_pair": f"{s1}|{s2}",
                            "r": r,
                        })

    df = pd.DataFrame(rows)
    print(f"  MMMData averaged ISC: {len(df)} rows")
    return df




def compute_mmmdata_summary(mmmdata_df):
    """Compute per-ROI mean ISC for MMMData (point estimates for overlay)."""
    rows = []
    for model_id in MODELS:
        mdf = mmmdata_df[mmmdata_df["model"] == model_id]
        for roi in ROI_KEYS:
            rdf = mdf[mdf["roi"] == roi]
            if len(rdf) == 0:
                continue
            # Fisher z-transform, average, back-transform
            zs = np.arctanh(rdf["r"].values)
            mean_z = zs.mean()
            rows.append({
                "model": model_id,
                "roi": roi,
                "mean_r": np.tanh(mean_z),
                "sd_r": np.tanh(zs.std()),
                "n_obs": len(rdf),
            })
    return pd.DataFrame(rows)


# ── main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=BENCHMARK_DIR / "results",
    )
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    # ── Compute mean MMMData clip duration in Pixar TRs ──────────────
    # Scan all events files to get average movie clip duration
    clip_durations_s = []
    for sub in MMMDATA_SUBJECTS:
        for ses in MMMDATA_SESSIONS:
            for run in MMMDATA_RUNS:
                events_path = (
                    BIDS_ROOT / sub / ses / "func"
                    / f"{sub}_{ses}_task-NATencoding_{run}_events.tsv"
                )
                if events_path.exists():
                    edf = pd.read_csv(str(events_path), sep="\t")
                    movies = edf[edf["trial_type"] == "movie"]
                    clip_durations_s.extend(movies["duration"].values)

    mean_clip_s = np.mean(clip_durations_s)
    n_trs_matched = int(np.round(mean_clip_s / PIXAR_TR))
    print(f"Mean MMMData clip duration: {mean_clip_s:.1f}s = {n_trs_matched} Pixar TRs (TR={PIXAR_TR}s)")

    # ── MMMData ISC ──────────────────────────────────────────────────
    print("\n=== Computing MMMData ISC (per-movie) ===")
    mmmdata_df = compute_mmmdata_isc()
    mmmdata_df.to_csv(out / "mmmdata_isc.tsv", sep="\t", index=False)

    mmmdata_summary = compute_mmmdata_summary(mmmdata_df)
    mmmdata_summary.to_csv(out / "mmmdata_isc_summary.tsv", sep="\t", index=False)
    print("\nMMMData ISC summary (volumetric ROIs):")
    print(mmmdata_summary.to_string(index=False))

    # ── MMMData signal-averaged ISC ──────────────────────────────────
    print("\n=== Computing MMMData signal-averaged ISC (repeated movies) ===")
    mmmdata_avg_df = compute_mmmdata_averaged_isc()
    mmmdata_avg_df.to_csv(out / "mmmdata_isc_averaged.tsv", sep="\t", index=False)
    print("\nSignal-averaged ISC (repeated movies, 10-session average):")
    for model_id in MODELS:
        adf = mmmdata_avg_df[mmmdata_avg_df["model"] == model_id]
        print(f"\n  Model {model_id}:")
        for movie in sorted(adf["movie"].unique()):
            mdf = adf[adf["movie"] == movie]
            for roi in ROI_KEYS:
                rdf = mdf[mdf["roi"] == roi]
                if len(rdf) > 0:
                    mean_r = rdf["r"].mean()
                    print(f"    {movie:40s} {roi:15s}  r = {mean_r:+.4f}")

    # ── Pixar ISC ────────────────────────────────────────────────────
    print("\n=== Computing Pixar ISC benchmark ===")
    print("Loading Pixar data...")
    ts_by_sub, conf_by_sub = load_pixar_data()

    print("Computing ISC distributions...")
    pixar_df = compute_pixar_isc(ts_by_sub, conf_by_sub, n_trs_matched)
    pixar_df.to_csv(out / "pixar_isc.tsv", sep="\t", index=False)

    # Summary stats
    pixar_summary = pixar_df.groupby(["model", "condition", "roi"])["mean_r"].agg(
        ["mean", "std", "count"]
    ).reset_index()
    pixar_summary.to_csv(out / "pixar_isc_summary.tsv", sep="\t", index=False)
    print("\nPixar ISC summary:")
    print(pixar_summary.to_string(index=False))

    print(f"\nResults saved to {out}")
    print("Done.")


if __name__ == "__main__":
    main()
