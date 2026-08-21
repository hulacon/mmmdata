#!/usr/bin/env python3
"""
glmsingle_tbretrieval.py — Single-trial betas for TBretrieval.

Companion to glmsingle_tbencoding.py. Retrieval differs from encoding in three
ways that drive every design choice here:

  1. Cue modality is crossed within session. Of the 4 retrieval runs in a
     typical session, 2 are image-cued (cueId=1) and 2 word-cued (cueId=2).
     Runs are cue-pure. Image and word cue trials are modelled SEPARATELY.
  2. Repeated conditions are scarce. 994 of 1000 items are retrieved exactly
     once per cue type; only 6 (mmmId 995-1000, sharedId=1 — the same anchor
     set encoding repeats 42x) repeat, 14x each per cue type. GLMsingle's
     GLMdenoise PC count and fracridge fraction cross-validate on repeated
     conditions, so those choices rest on 6 (split) or 12 (crossed) items.
     Stability is tested separately; see the workbench.
  3. The two cue types have different physical durations: image cues 3.00 s,
     word cues 0.54 s. GLMsingle's stimdur is a scalar, so both arms model the
     RETRIEVAL ATTEMPT at 3.0 s rather than the cue's physical length (median
     RT 1.38 s, 95th pct 2.86 s). --stimdur exposes this for sensitivity.

Two parameterizations, both run and compared (workbench decision 2026-08-21):

  split    two independent fits, 28 runs each, 1000 conditions, 6 repeated
           conditions per fit, separate noise pools and hyperparameters.
  crossed  one fit over all 56 runs, conditions = mmmId x cueType (2000
           columns; runs are cue-pure so only 1000 are active in any run),
           12 repeated conditions, one shared noise pool.

Both yield cue-specific betas. They differ only in whether hyperparameters and
the GLMdenoise noise pool are shared across cue types.

ses-18 is structurally different — 1 run per cue instead of 2, reCon all
"across", sharedId all 0, so NO super repeats and zero cross-validation
leverage. It is included by default and reported explicitly; --drop-ses18
excludes it.

Usage:
    python glmsingle_tbretrieval.py --subject sub-03 --parameterization split --cue image --dry-run
    python glmsingle_tbretrieval.py --subject sub-03 --parameterization split --cue word
    python glmsingle_tbretrieval.py --subject sub-03 --parameterization crossed
    python glmsingle_tbretrieval.py --subject sub-03 --parameterization split --cue image --sessions ses-04 ses-05
"""

import argparse
import json
import sys
import tomllib
from collections import Counter
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

# ── configuration (config TOMLs, not hard-coded paths) ───────────────────────

_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


def load_config():
    """base.toml with local.toml overlaid. Flattened across sections."""
    cfg = {}
    for name in ("base.toml", "local.toml"):
        p = _CONFIG_DIR / name
        if p.exists():
            with open(p, "rb") as f:
                for section in tomllib.load(f).values():
                    if isinstance(section, dict):
                        cfg.update(section)
    if "bids_project_dir" not in cfg:
        sys.exit(f"ERROR: no 'bids_project_dir' in {_CONFIG_DIR}/*.toml")
    return cfg


SPACE = "MNI152NLin2009cAsym_res-2"
TASK = "TBretrieval"
# ses-04..18: ses-18 exists for retrieval but not encoding.
TB_RET_SESSIONS = [f"ses-{i:02d}" for i in range(4, 19)]

TR = 1.5
STIMDUR_DEFAULT = 3.0   # the retrieval attempt, not the cue — see module docstring
FD_THRESHOLD = 0.5

CUE_LABELS = {1: "image", 2: "word"}   # cueId -> label (sidecar: 1=visual, 2=auditory)


# ── path helpers ─────────────────────────────────────────────────────────────

def bold_path(fmriprep_dir, subject, session, run):
    return (fmriprep_dir / subject / session / "func"
            / f"{subject}_{session}_task-{TASK}_{run}_space-{SPACE}_desc-preproc_bold.nii.gz")


def confounds_path(fmriprep_dir, subject, session, run):
    return (fmriprep_dir / subject / session / "func"
            / f"{subject}_{session}_task-{TASK}_{run}_desc-confounds_timeseries.tsv")


def events_path(bids_root, subject, session, run):
    return (bids_root / subject / session / "func"
            / f"{subject}_{session}_task-{TASK}_{run}_events.tsv")


def detect_runs(fmriprep_dir, subject, session):
    func_dir = fmriprep_dir / subject / session / "func"
    if not func_dir.is_dir():
        return []
    pattern = f"{subject}_{session}_task-{TASK}_run-*_space-{SPACE}_desc-preproc_bold.nii.gz"
    return sorted(p.name.split("_")[3] for p in func_dir.glob(pattern))


# ── discovery, with cue typing ───────────────────────────────────────────────

def run_cue_label(bids_root, subject, session, run):
    """Cue type of a run, from its events. Runs are cue-pure; assert that."""
    df = pd.read_csv(events_path(bids_root, subject, session, run), sep="\t")
    trials = df[df["trial_type"] != "rest"]
    ids = set(int(v) for v in trials["cueId"].dropna().unique())
    if len(ids) != 1:
        sys.exit(f"ERROR: {session}/{run} is not cue-pure (cueId={sorted(ids)}). "
                 "The split parameterization assumes cue-pure runs.")
    cue_id = ids.pop()
    if cue_id not in CUE_LABELS:
        sys.exit(f"ERROR: {session}/{run} has unknown cueId={cue_id}")
    return CUE_LABELS[cue_id]


def discover_sessions(bids_root, fmriprep_dir, subject, sessions=None, cue=None,
                      drop_ses18=False):
    """Discover TBretrieval sessions/runs, optionally filtered to one cue type.

    Returns list of (session, [(run, cue_label), ...]) tuples.
    """
    if sessions is None:
        sessions = TB_RET_SESSIONS
    if drop_ses18:
        sessions = [s for s in sessions if s != "ses-18"]

    found = []
    for ses in sessions:
        runs = detect_runs(fmriprep_dir, subject, ses)
        if not runs:
            print(f"  {ses}: no {TASK} runs found, skipping")
            continue
        typed = [(r, run_cue_label(bids_root, subject, ses, r)) for r in runs]
        if cue is not None:
            typed = [(r, c) for r, c in typed if c == cue]
        if typed:
            counts = Counter(c for _, c in typed)
            print(f"  {ses}: {len(typed)} runs ({dict(counts)})")
            found.append((ses, typed))
        else:
            print(f"  {ses}: no {cue}-cue runs, skipping")
    return found


def load_all_events(bids_root, subject, session_runs):
    """One DataFrame per run, non-rest trials only, with session/run/cue added."""
    all_events = []
    for session, typed in session_runs:
        for run, cue in typed:
            df = pd.read_csv(events_path(bids_root, subject, session, run), sep="\t")
            trials = df[df["trial_type"] != "rest"].copy()
            trials["session"] = session
            trials["run"] = run
            trials["cue"] = cue
            all_events.append(trials)
    return all_events


# ── design ───────────────────────────────────────────────────────────────────

def norm_mmm(v):
    """mmmId as a plain integer string: 998.0 / '998.0' / 998 -> '998'.

    TBencoding's condition_key.csv and trial_info.csv carry mmmId
    float-formatted ('998.0'), but the ROI pattern caches under
    derivatives/pattern_similarity/ normalize to '998'. The caches are what the
    reinstatement test and the sampling campaign actually consume, and a
    '998.0' vs '998' mismatch joins to zero rows SILENTLY. Normalize here so
    encoding and retrieval meet in the same key space.
    """
    if pd.isna(v):
        return "n/a"
    try:
        return str(int(float(v)))
    except (TypeError, ValueError):
        return str(v)


def condition_id(mmm_id, cue, parameterization):
    """Condition key. split -> mmmId; crossed -> mmmId x cueType."""
    mmm = norm_mmm(mmm_id)
    return mmm if parameterization == "split" else f"{mmm}_{cue}"


def build_condition_mapping(all_events, parameterization):
    seen = {}
    keys = []
    for ev in all_events:
        for mmm_id, cue in zip(ev["mmmId"].values, ev["cue"].values):
            k = condition_id(mmm_id, cue, parameterization)
            keys.append(k)
            if k not in seen:
                seen[k] = len(seen)

    counts = Counter(keys)
    rows = []
    for k, col_idx in sorted(seen.items(), key=lambda x: x[1]):
        mmm, _, cue = k.partition("_")
        rows.append({"col_index": col_idx, "condition_id": k, "mmmId": mmm,
                     "cue": cue or "n/a", "n_presentations": counts[k]})
    rows.sort(key=lambda r: r["col_index"])
    condition_key = pd.DataFrame(rows)

    n_repeated = int((condition_key["n_presentations"] > 1).sum())
    rep_dist = Counter(condition_key["n_presentations"].values)
    print(f"  {len(seen)} unique conditions, {n_repeated} with repetitions")
    print("  Repetition distribution: "
          + ", ".join(f"{n_items}x{n_reps}reps" for n_reps, n_items in sorted(rep_dist.items())))
    if n_repeated < 6:
        print(f"  WARNING: only {n_repeated} repeated conditions — GLMsingle's "
              "GLMdenoise and fracridge cross-validation rests on these.")
    return seen, condition_key


def build_design_matrices(all_events, cond_map, n_volumes_per_run, run_labels,
                          parameterization):
    n_conditions = len(cond_map)
    designs, trial_rows = [], []

    for run_idx, (ev, n_vols) in enumerate(zip(all_events, n_volumes_per_run)):
        design = np.zeros((n_vols, n_conditions), dtype=np.float32)
        for _, trial in ev.iterrows():
            k = condition_id(trial["mmmId"], trial["cue"], parameterization)
            col_idx = cond_map[k]
            onset_vol = int(np.round(trial["onset"] / TR))
            if 0 <= onset_vol < n_vols:
                design[onset_vol, col_idx] = 1.0
            trial_rows.append({
                "session": trial["session"], "run": trial["run"],
                "run_idx": run_idx, "cue": trial["cue"],
                "onset": trial["onset"], "duration": trial["duration"],
                "mmmId": norm_mmm(trial["mmmId"]), "condition_id": k,
                "col_index": col_idx,
                "word": trial.get("word", ""), "pairId": trial.get("pairId", ""),
                "sharedId": trial.get("sharedId", ""),
                "enCon": trial.get("enCon", ""), "reCon": trial.get("reCon", ""),
                "resp": trial.get("resp", ""), "resp_RT": trial.get("resp_RT", ""),
            })
        n_active = int((design.sum(axis=0) > 0).sum())
        print(f"  {run_labels[run_idx]}: design ({n_vols} x {n_conditions}), "
              f"{n_active} active conditions")
        designs.append(design)

    return designs, pd.DataFrame(trial_rows)


def build_spike_regressors(fmriprep_dir, subject, session, run, n_volumes):
    df = pd.read_csv(confounds_path(fmriprep_dir, subject, session, run), sep="\t")
    if len(df) != n_volumes:
        sys.exit(f"ERROR: confounds ({len(df)}) != BOLD ({n_volumes}) for {session}/{run}")
    fd = np.nan_to_num(df["framewise_displacement"].values, nan=0.0)
    outliers = np.where(fd > FD_THRESHOLD)[0]
    if len(outliers) == 0:
        return None
    spikes = np.zeros((n_volumes, len(outliers)), dtype=np.float64)
    for i, tr_idx in enumerate(outliers):
        spikes[tr_idx, i] = 1.0
    return spikes


# ── runner ───────────────────────────────────────────────────────────────────

def run(bids_root, fmriprep_dir, subject, session_runs, output_dir,
        parameterization, stimdur, use_spike_regressors=False, dry_run=False):
    run_list, run_labels, session_indices, run_cues = [], [], [], []
    for ses_idx, (session, typed) in enumerate(session_runs):
        for r, cue in typed:
            run_list.append((session, r))
            run_labels.append(f"{session}/{r}[{cue}]")
            session_indices.append(ses_idx + 1)
            run_cues.append(cue)

    n_total_runs, n_sessions = len(run_list), len(session_runs)
    print(f"\n{'=' * 70}")
    print(f"GLMsingle {TASK} [{parameterization}]: {subject} — "
          f"{n_sessions} sessions, {n_total_runs} runs")
    print(f"{'=' * 70}")

    print("\nLoading events...")
    all_events = load_all_events(bids_root, subject, session_runs)
    print(f"  {len(all_events)} run event files, "
          f"{sum(len(e) for e in all_events)} trials "
          f"({dict(Counter(run_cues))} runs by cue)")

    print("\nBuilding condition mapping...")
    cond_map, condition_key = build_condition_mapping(all_events, parameterization)

    has_ses18 = any(s == "ses-18" for s, _ in session_runs)
    if has_ses18:
        n18 = sum(len(t) for s, t in session_runs if s == "ses-18")
        print(f"\n  NOTE: ses-18 included ({n18} runs) — reCon all 'across', "
              "no super repeats, so it contributes data but no CV leverage.")

    print("\nResolving BOLD...")
    n_volumes_per_run, missing = [], []
    for session, r in run_list:
        bp = bold_path(fmriprep_dir, subject, session, r)
        if not bp.exists():
            missing.append(str(bp))
            n_volumes_per_run.append(None)
            continue
        n_volumes_per_run.append(nib.load(str(bp)).shape[-1])
    if missing:
        print(f"  ERROR: {len(missing)} BOLD files not found:")
        for m in missing[:5]:
            print(f"    {m}")
        sys.exit(1)
    print(f"  {n_total_runs} runs, volumes: {sorted(set(n_volumes_per_run))}")

    print("\nBuilding design matrices...")
    designs, trial_info = build_design_matrices(
        all_events, cond_map, n_volumes_per_run, run_labels, parameterization)

    sessionindicator = np.array(session_indices, dtype=int).reshape(1, -1)

    manifest = {
        "subject": subject, "task": TASK,
        "parameterization": parameterization,
        "fmriprep_dir": str(fmriprep_dir),
        "sessions": [s for s, _ in session_runs],
        "runs_per_session": {s: [r for r, _ in t] for s, t in session_runs},
        "run_labels": run_labels, "run_cues": run_cues,
        "session_indices": session_indices,
        "n_sessions": n_sessions, "n_total_runs": n_total_runs,
        "n_runs_by_cue": dict(Counter(run_cues)),
        "tr": TR, "stimdur": stimdur,
        "n_conditions": len(cond_map),
        "n_trials_total": int(len(trial_info)),
        "n_repeated_conditions": int((condition_key["n_presentations"] > 1).sum()),
        "repeated_condition_ids": sorted(
            condition_key.loc[condition_key["n_presentations"] > 1, "condition_id"]),
        "includes_ses18": has_ses18,
        "n_volumes_per_run": n_volumes_per_run,
        "confound_strategy": ("spike_regressors_only" if use_spike_regressors
                              else "none (GLMdenoise handles denoising)"),
        "fd_threshold": FD_THRESHOLD if use_spike_regressors else None,
    }

    if dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "dry_run_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"\n{'=' * 70}")
        print("DRY RUN — nothing fitted. Manifest:")
        print(f"  {output_dir / 'dry_run_manifest.json'}")
        print(f"  {n_total_runs} runs, {len(cond_map)} conditions, "
              f"{manifest['n_repeated_conditions']} repeated, "
              f"stimdur={stimdur}s, TR={TR}s")
        print(f"{'=' * 70}")
        return manifest

    from glmsingle.glmsingle import GLM_single

    print("\nLoading BOLD data...")
    data_list = []
    for session, r in run_list:
        data_list.append(nib.load(str(bold_path(fmriprep_dir, subject, session, r)))
                         .get_fdata(dtype=np.float32))

    extra_regressors, spike_counts = None, {}
    if use_spike_regressors:
        print("\nBuilding spike regressors...")
        extra_regressors = []
        for i, (session, r) in enumerate(run_list):
            spikes = build_spike_regressors(fmriprep_dir, subject, session, r,
                                            n_volumes_per_run[i])
            spike_counts[run_labels[i]] = 0 if spikes is None else spikes.shape[1]
            extra_regressors.append(
                spikes if spikes is not None
                else np.zeros((n_volumes_per_run[i], 0), dtype=np.float64))
        print(f"  Total spike regressors: {sum(spike_counts.values())}")
    else:
        print("\nNo external confound regression (per GLMsingle recommendation).")

    glmsingle_outdir = output_dir / "glmsingle_outputs"
    figuredir = output_dir / "glmsingle_figures"
    params = {
        "wantlibrary": 1, "wantglmdenoise": 1, "wantfracridge": 1,
        "wantfileoutputs": [1, 1, 1, 1], "wantmemoryoutputs": [0, 0, 0, 0],
        "sessionindicator": sessionindicator,
    }
    if extra_regressors is not None:
        params["extra_regressors"] = extra_regressors

    print(f"\nGLMsingle configuration:")
    print(f"  parameterization = {parameterization}")
    print(f"  TR = {TR}s, stimdur = {stimdur}s (retrieval attempt, not cue length)")
    print(f"  {n_total_runs} runs across {n_sessions} sessions")
    print(f"  {len(cond_map)} conditions, "
          f"{manifest['n_repeated_conditions']} repeated")
    print(f"  Output: {glmsingle_outdir}")

    glm = GLM_single(params)
    print(f"\nRunning GLMsingle ({n_total_runs} runs)...")
    results = glm.fit(design=designs, data=data_list, stimdur=stimdur, tr=TR,
                      outputdir=str(glmsingle_outdir), figuredir=str(figuredir))
    print("GLMsingle complete.")

    condition_key.to_csv(output_dir / "condition_key.csv", index=False)
    trial_info.to_csv(output_dir / "trial_info.csv", index=False)
    manifest["spike_counts"] = spike_counts if use_spike_regressors else None
    with open(output_dir / "run_metadata.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nSaved condition_key.csv, trial_info.csv, run_metadata.json")
    return results


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subject", required=True, help="e.g. sub-03")
    p.add_argument("--parameterization", choices=["split", "crossed"], required=True,
                   help="split: one fit per cue type. crossed: one fit, "
                        "conditions = mmmId x cueType.")
    p.add_argument("--cue", choices=["image", "word"], default=None,
                   help="Required for --parameterization split; ignored for crossed.")
    p.add_argument("--sessions", nargs="+", default=None,
                   help="Sessions to include (default: ses-04..ses-18)")
    p.add_argument("--drop-ses18", action="store_true",
                   help="Exclude ses-18 (no super repeats, no CV leverage)")
    p.add_argument("--stimdur", type=float, default=STIMDUR_DEFAULT,
                   help=f"Modelled event duration (default {STIMDUR_DEFAULT}s)")
    p.add_argument("--fmriprep-dir", default=None,
                   help="Override fMRIPrep derivatives dir (default: original variant)")
    p.add_argument("--output-base", default=None,
                   help="Override output root (default: derivatives/glmsingle_tbret)")
    p.add_argument("--spike-regressors", action="store_true",
                   help="Add spike regressors for FD > %.1f mm TRs" % FD_THRESHOLD)
    p.add_argument("--dry-run", action="store_true",
                   help="Resolve inputs, build the design, write a manifest, fit nothing")
    return p.parse_args()


def main():
    args = parse_args()
    if args.parameterization == "split" and args.cue is None:
        sys.exit("ERROR: --parameterization split requires --cue {image,word}")
    if args.parameterization == "crossed" and args.cue is not None:
        print("NOTE: --cue is ignored for --parameterization crossed")
        args.cue = None

    cfg = load_config()
    bids_root = Path(cfg["bids_project_dir"])
    fmriprep_dir = (Path(args.fmriprep_dir) if args.fmriprep_dir
                    else bids_root / "derivatives" / "fmriprep")
    output_base = (Path(args.output_base) if args.output_base
                   else bids_root / "derivatives" / "glmsingle_tbret")

    arm = args.cue if args.parameterization == "split" else "crossed"
    output_dir = output_base / args.subject / arm

    print(f"BIDS root:  {bids_root}")
    print(f"fMRIPrep:   {fmriprep_dir}")
    print(f"Output:     {output_dir}")
    print(f"\nDiscovering {TASK} sessions for {args.subject} "
          f"[{args.parameterization}"
          + (f", cue={args.cue}" if args.cue else "") + "]...")

    session_runs = discover_sessions(bids_root, fmriprep_dir, args.subject,
                                     args.sessions, args.cue, args.drop_ses18)
    if not session_runs:
        sys.exit(f"ERROR: no {TASK} runs found for {args.subject}")

    run(bids_root, fmriprep_dir, args.subject, session_runs, output_dir,
        args.parameterization, args.stimdur, args.spike_regressors, args.dry_run)


if __name__ == "__main__":
    main()
