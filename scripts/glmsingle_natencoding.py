#!/usr/bin/env python3
"""
glmsingle_natencoding.py — Run GLMsingle on NATencoding with 4.5-s movie chunks.

Mirrors glmsingle_tbencoding.py, but conditions are 4.5-s pseudo-trials
("chunks") tiled over each movie, so NAT data yield single-"trial" betas
comparable to the TBencoding fits (4.5 s = the TB SOA = exactly 3 TRs).
See docs/doc/pattern-similarity-plan.md, Phase 2 / decision D7.

Key design features:
  - Per movie, n_chunks = floor(min presentation duration / 4.5); the tail
    remainder is dropped. Chunk onset volume = round(movie_onset/TR) + 3*k,
    a single rounding at movie onset keeps every chunk on the TR grid.
  - Repeated movies ("The Bench", "From Dad To Son"; condition==3) share
    condition columns across their 10 presentations — these repeats drive
    GLMsingle's cross-validated HRF/GLMdenoise/fracridge selection. All
    other movies' chunks are one-shot conditions.
  - sessionindicator: 10 NAT sessions x 2 runs.
  - No external confound regression (GLMdenoise; same as the TB fits).
  - Memory control: BOLD is passed as masked 2D arrays (V_brain, T) using
    the union of the runs' fMRIPrep brain masks (~4x fewer voxels than the
    full res-2 grid; the full-grid TB fit peaked at 200 GB). The mask and
    its flat C-order indices are saved beside the outputs so downstream
    loaders can restore volumetric geometry. --full-grid restores the
    TB-style 4D layout (needs a high-mem node).

Usage:
    python glmsingle_natencoding.py --subject sub-03 --pipeline original
    python glmsingle_natencoding.py --subject sub-03 --pipeline nordic --dry-run
"""

import argparse
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

# ── paths / constants ────────────────────────────────────────────────────────

BIDS_ROOT = Path("/gpfs/projects/hulacon/shared/mmmdata")
DERIV_ROOT = BIDS_ROOT / "derivatives"

PIPELINES = {
    "original": (DERIV_ROOT / "fmriprep", DERIV_ROOT / "glmsingle_nat"),
    "nordic":   (DERIV_ROOT / "fmriprep_nordic", DERIV_ROOT / "glmsingle_nat_nordic"),
}

SPACE = "MNI152NLin2009cAsym_res-2"
TASK = "NATencoding"
NAT_SESSIONS = [f"ses-{i:02d}" for i in range(19, 29)]  # ses-19..ses-28

TR = 1.5           # seconds
CHUNK_S = 4.5      # chunk duration == GLMsingle stimdur == TB SOA
CHUNK_TRS = 3      # 4.5 / 1.5 — exact TR stride between chunk onsets


# ── path helpers ─────────────────────────────────────────────────────────────

def bold_path(fmriprep_dir, subject, session, run):
    return (
        fmriprep_dir / subject / session / "func"
        / f"{subject}_{session}_task-{TASK}_{run}_space-{SPACE}_desc-preproc_bold.nii.gz"
    )


def brainmask_path(fmriprep_dir, subject, session, run):
    return (
        fmriprep_dir / subject / session / "func"
        / f"{subject}_{session}_task-{TASK}_{run}_space-{SPACE}_desc-brain_mask.nii.gz"
    )


def events_path(subject, session, run):
    return (
        BIDS_ROOT / subject / session / "func"
        / f"{subject}_{session}_task-{TASK}_{run}_events.tsv"
    )


def detect_runs(fmriprep_dir, subject, session):
    func_dir = fmriprep_dir / subject / session / "func"
    pattern = f"{subject}_{session}_task-{TASK}_run-*_space-{SPACE}_desc-preproc_bold.nii.gz"
    return sorted([p.name.split("_")[3] for p in func_dir.glob(pattern)])


def discover_sessions(fmriprep_dir, subject, sessions=None):
    """Return list of (session, [runs]) tuples with fMRIPrep output present."""
    if sessions is None:
        sessions = NAT_SESSIONS
    found = []
    for ses in sessions:
        runs = detect_runs(fmriprep_dir, subject, ses)
        if runs:
            found.append((ses, runs))
            print(f"  {ses}: {len(runs)} runs")
        else:
            print(f"  {ses}: no {TASK} runs found, skipping")
    return found


# ── events / chunk conditions ────────────────────────────────────────────────

def load_all_events(subject, session_runs):
    """One DataFrame of movie rows per run, with session/run columns added."""
    all_events = []
    for session, runs in session_runs:
        for run in runs:
            df = pd.read_csv(events_path(subject, session, run), sep="\t")
            mov = df[df["trial_type"] == "movie"].copy()
            mov["session"] = session
            mov["run"] = run
            all_events.append(mov)
    return all_events


def build_chunk_conditions(all_events):
    """Global (movie_name, chunk_idx) -> column mapping with shared repeats.

    n_chunks per movie = floor(min presentation duration / 4.5), computed over
    ALL presentations so repeated movies get one consistent chunk count even
    though recorded durations jitter by ~±0.02 s around the nominal length.

    Returns:
        cond_map: {(movie_name, chunk_idx): col_index}
        movie_chunks: {movie_name: n_chunks}
        condition_key: DataFrame [col_index, movie_name, chunk_idx,
                                  n_presentations]
    """
    presentations = pd.concat(all_events, ignore_index=True)
    stats = presentations.groupby("movie_name", sort=False)["duration"].agg(
        ["min", "size"]
    )

    cond_map, movie_chunks, rows = {}, {}, []
    for movie_name, (dur_min, n_pres) in stats.iterrows():
        n_chunks = int(np.floor(dur_min / CHUNK_S))
        assert n_chunks >= 1, f"{movie_name}: duration {dur_min}s too short"
        movie_chunks[movie_name] = n_chunks
        for k in range(n_chunks):
            col = len(cond_map)
            cond_map[(movie_name, k)] = col
            rows.append({
                "col_index": col,
                "movie_name": movie_name,
                "chunk_idx": k,
                "n_presentations": int(n_pres),
            })
    condition_key = pd.DataFrame(rows)

    n_rep_cond = int((condition_key["n_presentations"] > 1).sum())
    print(f"  {len(stats)} movies -> {len(cond_map)} chunk conditions "
          f"({n_rep_cond} repeated x "
          f"{int(condition_key['n_presentations'].max())} presentations)")
    return cond_map, movie_chunks, condition_key


def build_design_matrices(all_events, cond_map, movie_chunks,
                          n_volumes_per_run, run_labels):
    """Per-run (n_vols, n_conditions) binary onset matrices + chunk_info."""
    n_conditions = len(cond_map)
    designs, chunk_rows = [], []

    for run_idx, (events_df, n_vols) in enumerate(
        zip(all_events, n_volumes_per_run)
    ):
        design = np.zeros((n_vols, n_conditions), dtype=np.float32)
        for _, movie in events_df.iterrows():
            name = movie["movie_name"]
            movie_onset_vol = int(np.round(movie["onset"] / TR))
            for k in range(movie_chunks[name]):
                onset_vol = movie_onset_vol + CHUNK_TRS * k
                assert 0 <= onset_vol < n_vols, (
                    f"{run_labels[run_idx]} {name} chunk {k}: "
                    f"vol {onset_vol} outside run of {n_vols}"
                )
                col = cond_map[(name, k)]
                assert design[:, col].sum() == 0, (
                    f"{run_labels[run_idx]} {name} chunk {k}: "
                    f"condition column used twice in one run"
                )
                design[onset_vol, col] = 1.0
                chunk_rows.append({
                    "session": movie["session"],
                    "run": movie["run"],
                    "run_idx": run_idx,
                    "movie_name": name,
                    "chunk_idx": k,
                    "onset": movie["onset"] + CHUNK_S * k,
                    "onset_vol": onset_vol,
                    "col_index": col,
                })
        n_active = int((design.sum(axis=0) > 0).sum())
        print(f"  {run_labels[run_idx]}: design ({n_vols} x {n_conditions}), "
              f"{n_active} active conditions")
        designs.append(design)

    chunk_info = pd.DataFrame(chunk_rows)
    return designs, chunk_info


# ── brain mask / data loading ────────────────────────────────────────────────

def build_union_mask(fmriprep_dir, subject, run_list):
    """Union of the runs' fMRIPrep brain masks on the shared res-2 grid."""
    union, affine, header = None, None, None
    for session, run in run_list:
        img = nib.load(str(brainmask_path(fmriprep_dir, subject, session, run)))
        m = np.asarray(img.dataobj) > 0.5
        if union is None:
            union, affine, header = m, img.affine, img.header
        else:
            assert m.shape == union.shape and np.allclose(img.affine, affine), (
                f"{session}/{run}: brain mask grid differs"
            )
            union |= m
    return union, affine, header


# ── GLMsingle runner ─────────────────────────────────────────────────────────

def run_glmsingle(pipeline, subject, session_runs, output_dir,
                  full_grid=False, dry_run=False):
    fmriprep_dir, _ = PIPELINES[pipeline]

    run_list, run_labels, session_indices = [], [], []
    for ses_idx, (session, runs) in enumerate(session_runs):
        for run in runs:
            run_list.append((session, run))
            run_labels.append(f"{session}/{run}")
            session_indices.append(ses_idx + 1)
    n_total_runs, n_sessions = len(run_list), len(session_runs)

    print(f"\n{'=' * 70}")
    print(f"GLMsingle NAT: {subject} / {pipeline} — "
          f"{n_sessions} sessions, {n_total_runs} runs"
          f"{' [DRY RUN]' if dry_run else ''}")
    print(f"{'=' * 70}")

    # Volume counts from headers (cheap; full data loaded only when fitting)
    n_volumes_per_run = []
    for session, run in run_list:
        bp = bold_path(fmriprep_dir, subject, session, run)
        if not bp.exists():
            print(f"  ERROR: {bp} not found")
            sys.exit(1)
        n_volumes_per_run.append(nib.load(str(bp)).shape[-1])

    print("\nLoading events...")
    all_events = load_all_events(subject, session_runs)
    print(f"  {len(all_events)} run event files, "
          f"{sum(len(e) for e in all_events)} movie presentations")

    print("\nBuilding chunk conditions...")
    cond_map, movie_chunks, condition_key = build_chunk_conditions(all_events)

    print("\nBuilding design matrices...")
    designs, chunk_info = build_design_matrices(
        all_events, cond_map, movie_chunks, n_volumes_per_run, run_labels
    )

    sessionindicator = np.array(session_indices, dtype=int).reshape(1, -1)

    print(f"\nGLMsingle configuration:")
    print(f"  TR = {TR}s, stimdur = {CHUNK_S}s")
    print(f"  {n_total_runs} runs across {n_sessions} sessions, "
          f"{sum(n_volumes_per_run)} volumes total")
    print(f"  {len(cond_map)} conditions, {len(chunk_info)} chunk instances")
    print(f"  data layout: {'full 4D grid' if full_grid else 'masked 2D'}")

    if dry_run:
        print("\nDry run — no BOLD loaded, nothing written.")
        rep = condition_key[condition_key["n_presentations"] > 1]
        for name, grp in rep.groupby("movie_name"):
            print(f"  repeated: {name}: {len(grp)} chunks x "
                  f"{int(grp['n_presentations'].iloc[0])} presentations")
        return None

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Brain mask (masked-2D layout) ──
    mask = mask_affine = mask_header = None
    if not full_grid:
        print("\nBuilding union brain mask...")
        mask, mask_affine, mask_header = build_union_mask(
            fmriprep_dir, subject, run_list
        )
        n_vox = int(mask.sum())
        print(f"  {n_vox} voxels of {mask.size} "
              f"({100 * n_vox / mask.size:.1f}% of grid)")
        nib.save(
            nib.Nifti1Image(mask.astype(np.uint8), mask_affine, mask_header),
            output_dir / "brain_mask.nii.gz",
        )
        np.save(output_dir / "brain_mask_index.npy",
                np.flatnonzero(mask.ravel(order="C")))

    # ── Load BOLD ──
    print("\nLoading BOLD data...")
    data_list = []
    for i, (session, run) in enumerate(run_list):
        img = nib.load(str(bold_path(fmriprep_dir, subject, session, run)))
        data = img.get_fdata(dtype=np.float32)
        assert data.shape[-1] == n_volumes_per_run[i]
        if not full_grid:
            assert data.shape[:3] == mask.shape, (
                f"{session}/{run}: BOLD grid differs from mask grid"
            )
            data = data[mask]  # (V_brain, T)
        print(f"  {run_labels[i]}: {data.shape}")
        data_list.append(data)

    # ── Fit ──
    from glmsingle.glmsingle import GLM_single

    glmsingle_outdir = output_dir / "glmsingle_outputs"
    figuredir = output_dir / "glmsingle_figures"
    params = {
        "wantlibrary": 1,
        "wantglmdenoise": 1,
        "wantfracridge": 1,
        "wantfileoutputs": [1, 1, 1, 1],
        "wantmemoryoutputs": [0, 0, 0, 0],
        "sessionindicator": sessionindicator,
    }
    glm = GLM_single(params)

    print(f"\nRunning GLMsingle (output: {glmsingle_outdir})...")
    glm.fit(
        design=designs,
        data=data_list,
        stimdur=CHUNK_S,
        tr=TR,
        outputdir=str(glmsingle_outdir),
        figuredir=str(figuredir),
    )
    print("GLMsingle complete.")

    # ── Supplementary outputs ──
    condition_key.to_csv(output_dir / "condition_key.csv", index=False)
    chunk_info.to_csv(output_dir / "chunk_info.csv", index=False)

    metadata = {
        "subject": subject,
        "task": TASK,
        "pipeline": pipeline,
        "fmriprep_dir": str(fmriprep_dir),
        "sessions": [sr[0] for sr in session_runs],
        "runs_per_session": {sr[0]: sr[1] for sr in session_runs},
        "n_sessions": n_sessions,
        "n_total_runs": n_total_runs,
        "run_labels": run_labels,
        "session_indices": session_indices,
        "tr": TR,
        "stimdur": CHUNK_S,
        "chunk_trs": CHUNK_TRS,
        "n_conditions": len(cond_map),
        "n_chunk_instances": len(chunk_info),
        "n_repeated_conditions": int(
            (condition_key["n_presentations"] > 1).sum()
        ),
        "movie_chunks": movie_chunks,
        "confound_strategy": "none (GLMdenoise handles denoising)",
        "data_layout": "full_4d" if full_grid else "masked_2d",
        "n_voxels": None if full_grid else int(mask.sum()),
        "mask_files": None if full_grid else [
            "brain_mask.nii.gz", "brain_mask_index.npy"
        ],
    }
    with open(output_dir / "run_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved condition_key.csv, chunk_info.csv, run_metadata.json")


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--subject", required=True, help="e.g. sub-03")
    parser.add_argument("--pipeline", required=True,
                        choices=sorted(PIPELINES),
                        help="original (fmriprep) or nordic (fmriprep_nordic)")
    parser.add_argument("--sessions", nargs="+", default=None,
                        help="Specific sessions (default: ses-19..ses-28)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Override output dir "
                             "(default: derivatives/glmsingle_nat{,_nordic}/{subject})")
    parser.add_argument("--full-grid", action="store_true",
                        help="Pass full 4D volumes instead of masked 2D "
                             "(TB-fit layout; needs a high-mem node)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Build conditions/designs and report; no BOLD, "
                             "no outputs")
    return parser.parse_args()


def main():
    args = parse_args()
    fmriprep_dir, output_base = PIPELINES[args.pipeline]

    print(f"Discovering {TASK} sessions for {args.subject} ({args.pipeline})...")
    session_runs = discover_sessions(fmriprep_dir, args.subject, args.sessions)
    if not session_runs:
        print(f"ERROR: No {TASK} sessions found")
        sys.exit(1)

    output_dir = args.output_dir or (output_base / args.subject)
    run_glmsingle(args.pipeline, args.subject, session_runs, output_dir,
                  full_grid=args.full_grid, dry_run=args.dry_run)
    print("\nDone.")


if __name__ == "__main__":
    main()
