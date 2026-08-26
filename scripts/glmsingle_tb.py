#!/usr/bin/env python3
"""
glmsingle_tb.py — Single-trial betas for the TB tasks (encoding + retrieval).

One runner, four arms, all on the post-campaign fMRIPrep tree
(derivatives/fmriprep, 25.2.5). Supersedes glmsingle_tbretrieval.py
(split/crossed) and, for new fits, glmsingle_tbencoding.py (24.1.1 tree).
Design record: mmmdata-agents/docs/workbench/retrieval-modeling/ (redesigned
2026-08-26 into a 3x2: beta ladder TYPEB/C/D x siloed/pooled).

Arms:

  enc        TBencoding only (ses-04..17, ~42 runs), conditions = mmmId
             (1000 columns, 664 repeated).
  ret-image  TBretrieval image-cued runs (ses-04..18, 28 runs), conditions =
             mmmId (1000 columns, 6 repeated — the binding scarcity).
  ret-word   TBretrieval word-cued runs, same shape as ret-image.
  pooled     One fit over ALL TB runs (~98), conditions = mmmId x subgroup
             where subgroup is enc / image / word (~3000 columns; repetitions
             fall within a subgroup, never across). Retrieval borrows
             encoding's repeated conditions for GLMdenoise/fracridge
             cross-validation and shares one noise pool.

The beta-type factor (TYPEB FITHRF / TYPEC +GLMdenoise / TYPED +fracridge) is
free: GLMsingle emits all of them from one fit (wantfileoutputs=[1,1,1,1]).

Retrieval facts that drive the design (measured 2026-08-21, see workbench log):
  - Runs are cue-pure (asserted here); cue modality is crossed within session.
  - 994 of 1000 items are retrieved exactly once per cue; only mmmId 995-1000
    (sharedId=1, the encoding anchor set) repeat, 14x each per cue.
  - Image cues are 3.00 s, word cues 0.54 s, but both are modelled at
    stimdur 3.0 s: the modelled event is the RETRIEVAL ATTEMPT (median RT
    1.38 s, 95th pct 2.86 s), matching encoding's 3.0 s so the pooled scalar
    stimdur is coherent and enc/ret betas are on comparable footing.
  - ses-18 is retrieval-only and structurally different (1 run/cue, reCon all
    "across", no super repeats — data but zero CV leverage). Included by
    default and reported; --drop-ses18 excludes it.

Usage:
    python glmsingle_tb.py --subject sub-03 --arm ret-image --dry-run
    python glmsingle_tb.py --subject sub-03 --arm enc
    python glmsingle_tb.py --subject sub-03 --arm pooled
    python glmsingle_tb.py --subject sub-03 --arm ret-word --sessions ses-04 ses-05
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
OUTPUT_TREE = "glmsingle_tb"

# ses-18 exists for retrieval but not encoding.
TASK_SESSIONS = {
    "TBencoding": [f"ses-{i:02d}" for i in range(4, 18)],
    "TBretrieval": [f"ses-{i:02d}" for i in range(4, 19)],
}

TR = 1.5
STIMDUR_DEFAULT = 3.0   # the retrieval attempt, not the cue — see module docstring
FD_THRESHOLD = 0.5

CUE_LABELS = {1: "image", 2: "word"}   # cueId -> label (sidecar: 1=visual, 2=auditory)

# arm -> list of (task, cue filter). Subgroup labels are "enc" for encoding
# runs and the cue label for retrieval runs.
ARM_SPECS = {
    "enc": [("TBencoding", None)],
    "ret-image": [("TBretrieval", "image")],
    "ret-word": [("TBretrieval", "word")],
    "pooled": [("TBencoding", None), ("TBretrieval", None)],
}


# ── path helpers ─────────────────────────────────────────────────────────────

def bold_path(fmriprep_dir, subject, session, task, run):
    return (fmriprep_dir / subject / session / "func"
            / f"{subject}_{session}_task-{task}_{run}_space-{SPACE}_desc-preproc_bold.nii.gz")


def confounds_path(fmriprep_dir, subject, session, task, run):
    return (fmriprep_dir / subject / session / "func"
            / f"{subject}_{session}_task-{task}_{run}_desc-confounds_timeseries.tsv")


def events_path(bids_root, subject, session, task, run):
    return (bids_root / subject / session / "func"
            / f"{subject}_{session}_task-{task}_{run}_events.tsv")


def detect_runs(fmriprep_dir, subject, session, task):
    func_dir = fmriprep_dir / subject / session / "func"
    if not func_dir.is_dir():
        return []
    pattern = f"{subject}_{session}_task-{task}_run-*_space-{SPACE}_desc-preproc_bold.nii.gz"
    return sorted(p.name.split("_")[3] for p in func_dir.glob(pattern))


# ── discovery, with subgroup typing ──────────────────────────────────────────

def run_cue_label(bids_root, subject, session, run):
    """Cue type of a retrieval run, from its events. Runs are cue-pure; assert."""
    df = pd.read_csv(events_path(bids_root, subject, session, "TBretrieval", run),
                     sep="\t")
    trials = df[df["trial_type"] != "rest"]
    ids = set(int(v) for v in trials["cueId"].dropna().unique())
    if len(ids) != 1:
        sys.exit(f"ERROR: {session}/{run} is not cue-pure (cueId={sorted(ids)}). "
                 "The siloed retrieval arms assume cue-pure runs.")
    cue_id = ids.pop()
    if cue_id not in CUE_LABELS:
        sys.exit(f"ERROR: {session}/{run} has unknown cueId={cue_id}")
    return CUE_LABELS[cue_id]


def discover_sessions(bids_root, fmriprep_dir, subject, arm, sessions=None,
                      drop_ses18=False):
    """Discover the arm's sessions/runs across its tasks.

    Returns list of (session, [(task, run, subgroup), ...]) tuples, sessions
    sorted, encoding runs before retrieval runs within a session.
    """
    specs = ARM_SPECS[arm]
    all_sessions = sorted({s for task, _ in specs for s in TASK_SESSIONS[task]})
    if sessions is not None:
        all_sessions = [s for s in all_sessions if s in set(sessions)]
    if drop_ses18:
        all_sessions = [s for s in all_sessions if s != "ses-18"]

    found = []
    for ses in all_sessions:
        typed = []
        for task, cue in specs:
            if ses not in TASK_SESSIONS[task]:
                continue
            for r in detect_runs(fmriprep_dir, subject, ses, task):
                if task == "TBencoding":
                    subgroup = "enc"
                else:
                    subgroup = run_cue_label(bids_root, subject, ses, r)
                    if cue is not None and subgroup != cue:
                        continue
                typed.append((task, r, subgroup))
        if typed:
            counts = Counter(sg for _, _, sg in typed)
            print(f"  {ses}: {len(typed)} runs ({dict(counts)})")
            found.append((ses, typed))
        else:
            print(f"  {ses}: no matching runs, skipping")
    return found


def load_all_events(bids_root, subject, session_runs):
    """One DataFrame per run, stimulus trials only, with bookkeeping columns.

    Encoding events keep trial_type == 'image'; retrieval events keep
    everything but 'rest'.
    """
    all_events = []
    for session, typed in session_runs:
        for task, run, subgroup in typed:
            df = pd.read_csv(events_path(bids_root, subject, session, task, run),
                             sep="\t")
            if task == "TBencoding":
                trials = df[df["trial_type"] == "image"].copy()
            else:
                trials = df[df["trial_type"] != "rest"].copy()
            trials["session"] = session
            trials["run"] = run
            trials["task"] = task
            trials["subgroup"] = subgroup
            all_events.append(trials)
    return all_events


# ── design ───────────────────────────────────────────────────────────────────

def norm_mmm(v):
    """mmmId as a plain integer string: 998.0 / '998.0' / 998 -> '998'.

    The old TBencoding tree's condition_key.csv / trial_info.csv carry mmmId
    float-formatted ('998.0'), but the ROI pattern caches under
    derivatives/pattern_similarity/ normalize to '998'. The caches are what
    the 6-cell benchmark and the sampling campaign actually consume, and a
    '998.0' vs '998' mismatch joins to zero rows SILENTLY. Normalize here so
    all arms meet the caches in the same key space.
    """
    if pd.isna(v):
        return "n/a"
    try:
        return str(int(float(v)))
    except (TypeError, ValueError):
        return str(v)


def condition_id(mmm_id, subgroup, arm):
    """Condition key. Siloed arms -> mmmId; pooled -> mmmId x subgroup
    ('998_enc' / '998_image' / '998_word'), so repetitions pool within a
    subgroup and never across."""
    mmm = norm_mmm(mmm_id)
    return f"{mmm}_{subgroup}" if arm == "pooled" else mmm


def build_condition_mapping(all_events, arm):
    seen = {}
    keys = []
    for ev in all_events:
        for mmm_id, subgroup in zip(ev["mmmId"].values, ev["subgroup"].values):
            k = condition_id(mmm_id, subgroup, arm)
            keys.append(k)
            if k not in seen:
                seen[k] = len(seen)

    counts = Counter(keys)
    rows = []
    for k, col_idx in sorted(seen.items(), key=lambda x: x[1]):
        mmm, _, subgroup = k.partition("_")
        rows.append({"col_index": col_idx, "condition_id": k, "mmmId": mmm,
                     "subgroup": subgroup or "n/a", "n_presentations": counts[k]})
    rows.sort(key=lambda r: r["col_index"])
    condition_key = pd.DataFrame(rows)

    n_repeated = int((condition_key["n_presentations"] > 1).sum())
    rep_dist = Counter(condition_key["n_presentations"].values)
    print(f"  {len(seen)} unique conditions, {n_repeated} with repetitions")
    print("  Repetition distribution: "
          + ", ".join(f"{n_items}x{n_reps}reps" for n_reps, n_items in sorted(rep_dist.items())))
    if arm == "pooled":
        for sg in ("enc", "image", "word"):
            sub = condition_key[condition_key["subgroup"] == sg]
            print(f"    {sg}: {len(sub)} conditions, "
                  f"{int((sub['n_presentations'] > 1).sum())} repeated")
    if n_repeated < 6:
        print(f"  WARNING: only {n_repeated} repeated conditions — GLMsingle's "
              "GLMdenoise and fracridge cross-validation rests on these.")
    return seen, condition_key


def build_design_matrices(all_events, cond_map, n_volumes_per_run, run_labels,
                          arm):
    n_conditions = len(cond_map)
    designs, trial_rows = [], []

    for run_idx, (ev, n_vols) in enumerate(zip(all_events, n_volumes_per_run)):
        design = np.zeros((n_vols, n_conditions), dtype=np.float32)
        for _, trial in ev.iterrows():
            k = condition_id(trial["mmmId"], trial["subgroup"], arm)
            col_idx = cond_map[k]
            onset_vol = int(np.round(trial["onset"] / TR))
            if 0 <= onset_vol < n_vols:
                design[onset_vol, col_idx] = 1.0
            trial_rows.append({
                "session": trial["session"], "run": trial["run"],
                "run_idx": run_idx, "task": trial["task"],
                "subgroup": trial["subgroup"],
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


def build_spike_regressors(fmriprep_dir, subject, session, task, run, n_volumes):
    df = pd.read_csv(confounds_path(fmriprep_dir, subject, session, task, run),
                     sep="\t")
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


# ── derivative-tree bookkeeping ──────────────────────────────────────────────

def ensure_dataset_description(output_base, fmriprep_dir):
    """The tree must be catalog-legible under Contract A from its first write
    — an undeclared derivative tree in the nightly rebuild advertises a
    pipeline that does not exist."""
    dd = output_base / "dataset_description.json"
    if dd.exists():
        return
    try:
        from importlib.metadata import version
        glmsingle_version = version("glmsingle")
    except Exception:
        glmsingle_version = "unknown"
    output_base.mkdir(parents=True, exist_ok=True)
    with open(dd, "w") as f:
        json.dump({
            "Name": "GLMsingle single-trial betas — TB encoding + retrieval "
                    "(siloed and pooled arms)",
            "BIDSVersion": "1.8.0",
            "DatasetType": "derivative",
            "GeneratedBy": [
                {"Name": "GLMsingle", "Version": glmsingle_version,
                 "CodeURL": "https://github.com/cvnlab/GLMsingle"},
                {"Name": "glmsingle_tb.py",
                 "Description": "mmmdata/scripts/glmsingle_tb.py; design record "
                                "in mmmdata-agents docs/workbench/retrieval-modeling/"},
            ],
            "SourceDatasets": [{"URL": str(fmriprep_dir)}],
        }, f, indent=2)
    print(f"  Wrote {dd}")


# ── runner ───────────────────────────────────────────────────────────────────

def run(bids_root, fmriprep_dir, subject, session_runs, output_dir, arm,
        stimdur, use_spike_regressors=False, dry_run=False):
    run_list, run_labels, session_indices, run_subgroups = [], [], [], []
    for ses_idx, (session, typed) in enumerate(session_runs):
        for task, r, subgroup in typed:
            run_list.append((session, task, r))
            run_labels.append(f"{session}/{r}[{subgroup}]")
            session_indices.append(ses_idx + 1)
            run_subgroups.append(subgroup)

    n_total_runs, n_sessions = len(run_list), len(session_runs)
    print(f"\n{'=' * 70}")
    print(f"GLMsingle TB [{arm}]: {subject} — "
          f"{n_sessions} sessions, {n_total_runs} runs")
    print(f"{'=' * 70}")

    print("\nLoading events...")
    all_events = load_all_events(bids_root, subject, session_runs)
    print(f"  {len(all_events)} run event files, "
          f"{sum(len(e) for e in all_events)} trials "
          f"({dict(Counter(run_subgroups))} runs by subgroup)")

    print("\nBuilding condition mapping...")
    cond_map, condition_key = build_condition_mapping(all_events, arm)

    has_ses18 = any(s == "ses-18" for s, _ in session_runs)
    if has_ses18:
        n18 = sum(len(t) for s, t in session_runs if s == "ses-18")
        print(f"\n  NOTE: ses-18 included ({n18} runs) — retrieval-only, reCon "
              "all 'across', no super repeats: data but no CV leverage.")

    print("\nResolving BOLD...")
    n_volumes_per_run, missing = [], []
    for session, task, r in run_list:
        bp = bold_path(fmriprep_dir, subject, session, task, r)
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
        all_events, cond_map, n_volumes_per_run, run_labels, arm)

    sessionindicator = np.array(session_indices, dtype=int).reshape(1, -1)

    manifest = {
        "subject": subject, "arm": arm,
        "tasks": sorted({t for _, t, _ in run_list}),
        "fmriprep_dir": str(fmriprep_dir),
        "sessions": [s for s, _ in session_runs],
        "runs_per_session": {s: [r for _, r, _ in t] for s, t in session_runs},
        "run_labels": run_labels, "run_subgroups": run_subgroups,
        "session_indices": session_indices,
        "n_sessions": n_sessions, "n_total_runs": n_total_runs,
        "n_runs_by_subgroup": dict(Counter(run_subgroups)),
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
    for session, task, r in run_list:
        data_list.append(
            nib.load(str(bold_path(fmriprep_dir, subject, session, task, r)))
            .get_fdata(dtype=np.float32))

    extra_regressors, spike_counts = None, {}
    if use_spike_regressors:
        print("\nBuilding spike regressors...")
        extra_regressors = []
        for i, (session, task, r) in enumerate(run_list):
            spikes = build_spike_regressors(fmriprep_dir, subject, session,
                                            task, r, n_volumes_per_run[i])
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
        # All four model types saved: the TYPEB/C/D beta ladder is the free
        # factor of the 3x2 comparison.
        "wantfileoutputs": [1, 1, 1, 1], "wantmemoryoutputs": [0, 0, 0, 0],
        "sessionindicator": sessionindicator,
    }
    if extra_regressors is not None:
        params["extra_regressors"] = extra_regressors

    print(f"\nGLMsingle configuration:")
    print(f"  arm = {arm}")
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
    p.add_argument("--arm", choices=sorted(ARM_SPECS), required=True,
                   help="enc | ret-image | ret-word (siloed fits) | pooled "
                        "(one fit, conditions = mmmId x subgroup)")
    p.add_argument("--sessions", nargs="+", default=None,
                   help="Sessions to include (default: all the arm's tasks have)")
    p.add_argument("--drop-ses18", action="store_true",
                   help="Exclude ses-18 (retrieval-only; no super repeats, no CV leverage)")
    p.add_argument("--stimdur", type=float, default=STIMDUR_DEFAULT,
                   help=f"Modelled event duration (default {STIMDUR_DEFAULT}s)")
    p.add_argument("--fmriprep-dir", default=None,
                   help="Override fMRIPrep derivatives dir (default: derivatives/fmriprep)")
    p.add_argument("--output-base", default=None,
                   help=f"Override output root (default: derivatives/{OUTPUT_TREE})")
    p.add_argument("--spike-regressors", action="store_true",
                   help="Add spike regressors for FD > %.1f mm TRs" % FD_THRESHOLD)
    p.add_argument("--dry-run", action="store_true",
                   help="Resolve inputs, build the design, write a manifest, fit nothing")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config()
    bids_root = Path(cfg["bids_project_dir"])
    fmriprep_dir = (Path(args.fmriprep_dir) if args.fmriprep_dir
                    else bids_root / "derivatives" / "fmriprep")
    output_base = (Path(args.output_base) if args.output_base
                   else bids_root / "derivatives" / OUTPUT_TREE)
    output_dir = output_base / args.subject / args.arm

    print(f"BIDS root:  {bids_root}")
    print(f"fMRIPrep:   {fmriprep_dir}")
    print(f"Output:     {output_dir}")

    ensure_dataset_description(output_base, fmriprep_dir)

    print(f"\nDiscovering TB runs for {args.subject} [{args.arm}]...")
    session_runs = discover_sessions(bids_root, fmriprep_dir, args.subject,
                                     args.arm, args.sessions, args.drop_ses18)
    if not session_runs:
        sys.exit(f"ERROR: no runs found for {args.subject} [{args.arm}]")

    run(bids_root, fmriprep_dir, args.subject, session_runs, output_dir,
        args.arm, args.stimdur, args.spike_regressors, args.dry_run)


if __name__ == "__main__":
    main()
