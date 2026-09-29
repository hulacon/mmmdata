#!/usr/bin/env python3
"""Tier 1 of the data-quality collection: per run × confound regime, rebuildable.

Verbs (all idempotent; state is on disk, never in this process):

  regimes   print the regime registry (name, status, columns, drift, version)
  plan      list the run × regime cells that are missing or stale, optionally
            writing a units file for the sbatch array
  run       clean ONE run under every requested regime (one BOLD load shared
            by all of them) and write its tier-1 outputs; skips cells that are
            current unless --force. A regime the run cannot carry (too few
            aCompCor components) gets a declared-absent marker instead, and
            the remaining regimes still run
  collect   flatten every sidecar into tier1_runs.tsv and tier1_parcels.tsv
            at the tree root
  motion    registry T1.7: one row per run (no regime) with raw and
            respiration-filtered FD, to tier1_motion.tsv (+ .json) at the
            tree root; for task runs also T1.8, the largest motion–task |r|.
            Reads confounds TSVs, events and respiratory recordings, never
            BOLD; a full rebuild replaces the table
  glm-plan  as plan, for the GLM cells (T1.5/T1.6): task runs with events
  glm       fit ONE task run's stand-in GLM under every requested regime
            (T1.5 task R², and T1.6 condition betas for floc/motor/tone);
            same skip / --force / declared-absent rules as run

Provisional regimes (not yet confirmed by whoever defined them) are excluded
from every verb unless --include-provisional is given.

Usage:
    python tier1.py regimes
    python tier1.py plan --units units.txt
    python tier1.py run --sub 03 --ses 19 --task NATencoding --run 01
    python tier1.py run --units units.txt --index 7          # sbatch array
    python tier1.py collect
    python tier1.py motion
    python tier1.py glm-plan --units glm_units.txt
    python tier1.py glm --units glm_units.txt --index 7      # sbatch array, VERB=glm

Library: src/python/neuroimaging/{confounds,data_quality,data_quality_glm}.py. Design record:
mmmdata-agents docs/workbench/data-quality/.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src" / "python") not in sys.path:  # idempotent: tests import this module repeatedly
    sys.path.insert(0, str(REPO_ROOT / "src" / "python"))

from core.config import load_config  # noqa: E402
from neuroimaging import data_quality as dq  # noqa: E402
from neuroimaging import data_quality_glm as dqg  # noqa: E402
from neuroimaging.confounds import (  # noqa: E402
    RegimeNotApplicable,
    confirmed_regimes,
    describe_regimes,
    load_regimes,
)
from neuroimaging.constants import DEFAULT_SPACE, DEFAULT_VARIANT  # noqa: E402
from neuroimaging.io import FmriprepRun, find_fmriprep_runs  # noqa: E402


# ---------------------------------------------------------------------------
# Paths and regimes
# ---------------------------------------------------------------------------

class Paths:
    def __init__(self, args: argparse.Namespace):
        cfg = load_config()["paths"]
        self.bids_root = Path(args.bids_root or cfg["bids_project_dir"])
        self.output_dir = Path(cfg["output_dir"])
        self.tree_root = Path(args.tree_root or self.output_dir / dq.TREE_NAME)
        self.atlases_dir = Path(args.atlases_dir or self.output_dir / "atlases")
        self.variant = args.variant
        self.fmriprep_tree = self.bids_root / "derivatives" / self.variant


def selected_regimes(args: argparse.Namespace) -> list:
    registry = load_regimes()
    if args.regimes:
        names = [n.strip() for n in args.regimes.split(",") if n.strip()]
        unknown = [n for n in names if n not in registry]
        if unknown:
            sys.exit(f"Unknown regime(s) {unknown}; the registry has {sorted(registry)}")
    else:
        names = list(registry)
    chosen = []
    for n in names:
        r = registry[n]
        if r.provisional and not args.include_provisional:
            if args.regimes:
                sys.exit(
                    f"Regime {n!r} is provisional (not yet confirmed by its source). "
                    "Pass --include-provisional to use it knowingly."
                )
            continue
        chosen.append(r)
    return chosen


def runs_for(args: argparse.Namespace, paths: Paths) -> list[FmriprepRun]:
    runs = find_fmriprep_runs(
        subject=args.sub, session=args.ses, task=args.task, run=args.run,
        variant=paths.variant, space=DEFAULT_SPACE, bids_root=paths.bids_root,
        allow_mixed_designs=True,
    )
    return [r for r in runs if r.bold is not None and r.mask is not None and r.confounds is not None]


def unit_line(run: FmriprepRun) -> str:
    return f"{run.subject}\t{run.session}\t{run.task}\t{run.run or ''}"


def run_from_unit(line: str, paths: Paths) -> FmriprepRun:
    sub, ses, task, run = (line.rstrip("\n").split("\t") + [""])[:4]
    runs = find_fmriprep_runs(
        subject=sub, session=ses, task=task, run=run or None,
        variant=paths.variant, space=DEFAULT_SPACE, bids_root=paths.bids_root,
        allow_mixed_designs=True,
    )
    runs = [r for r in runs if (r.run or "") == run]
    if len(runs) != 1:
        sys.exit(f"Unit {line.strip()!r} resolves to {len(runs)} runs under {paths.fmriprep_tree}; expected 1")
    return runs[0]


# ---------------------------------------------------------------------------
# Verbs
# ---------------------------------------------------------------------------

def cmd_regimes(args: argparse.Namespace) -> None:
    table = describe_regimes()
    print(table.to_string(index=False))
    print(f"\nconfirmed (usable without --include-provisional): {confirmed_regimes()}")


def cmd_plan(args: argparse.Namespace) -> None:
    paths = Paths(args)
    regimes = selected_regimes(args)
    runs = runs_for(args, paths)
    stale_runs, n_cells, n_missing = [], 0, 0
    atlases_sha = dq.atlases_sha256(paths.atlases_dir) if args.check_hashes else None
    for run in runs:
        sha = dq.file_sha256(run.bold) if args.check_hashes else None
        missing_here = 0
        for regime in regimes:
            n_cells += 1
            if sha is None:
                current = dq.cell_exists(paths.tree_root, run, regime.name)
            else:
                current = dq.is_current(paths.tree_root, run, regime, sha, atlases_sha=atlases_sha)
            if not current:
                missing_here += 1
        if missing_here:
            n_missing += missing_here
            stale_runs.append(run)
    print(f"runs: {len(runs)}  regimes: {[r.name for r in regimes]}  cells: {n_cells}  "
          f"missing/stale cells: {n_missing}  runs to (re)build: {len(stale_runs)}")
    if not args.check_hashes:
        print("(existence only; --check-hashes also compares input and atlas hashes and regime versions)")
    if args.units:
        Path(args.units).write_text("".join(unit_line(r) + "\n" for r in stale_runs))
        print(f"wrote {len(stale_runs)} units to {args.units}")


def cmd_run(args: argparse.Namespace) -> None:
    paths = Paths(args)
    regimes = selected_regimes(args)
    if args.units:
        if args.index is None:
            sys.exit("--units needs --index (1-based line number, e.g. $SLURM_ARRAY_TASK_ID)")
        lines = [ln for ln in Path(args.units).read_text().splitlines() if ln.strip()]
        if not 1 <= args.index <= len(lines):
            sys.exit(f"--index {args.index} is outside 1..{len(lines)} for {args.units}")
        runs = [run_from_unit(lines[args.index - 1], paths)]
    else:
        if not (args.sub and args.ses and args.task):
            sys.exit("run needs --sub, --ses and --task (and --run for multi-run tasks), or --units/--index")
        runs = runs_for(args, paths)
        if not runs:
            sys.exit(f"No completed fMRIPrep run matches under {paths.fmriprep_tree}")
    fmriprep_version = dq.pipeline_version(paths.fmriprep_tree)
    code_sha = dq.code_version(REPO_ROOT)
    dq.ensure_dataset_description(paths.tree_root, paths.fmriprep_tree, fmriprep_version, code_sha)

    atlases_sha = dq.atlases_sha256(paths.atlases_dir)
    for run in runs:
        t0 = time.time()
        sha = dq.file_sha256(run.bold)
        todo = [r for r in regimes
                if args.force or not dq.is_current(paths.tree_root, run, r, sha, atlases_sha=atlases_sha)]
        if not todo:
            print(f"{run.entity_prefix}: all {len(regimes)} regimes current, skipping")
            continue
        inputs = dq.load_run_inputs(run, paths.atlases_dir)
        assert inputs.input_bold_sha256 == sha and inputs.input_atlases_sha256 == atlases_sha
        for regime in todo:
            try:
                rec = dq.write_run_regime(
                    paths.tree_root, inputs, regime,
                    fmriprep_version=fmriprep_version, code_sha=code_sha,
                )
            except RegimeNotApplicable as exc:
                # The run cannot carry this regime; declare it and go on to the next one.
                dq.write_absent_regime(
                    paths.tree_root, inputs, regime, exc,
                    fmriprep_version=fmriprep_version, code_sha=code_sha,
                )
                print(f"{run.entity_prefix} {regime.name:10s} ABSENT ({exc.n_available} of "
                      f"{exc.n_required} aCompCor components)")
                continue
            print(f"{run.entity_prefix} {regime.name:10s} n_vol={rec['n_vol']} nss={rec['n_nss']} "
                  f"p={rec['n_regressors']} dof={rec['dof_resid']} "
                  f"tsnr_med={rec['tsnr_median_mask']:.2f} var_ratio_med={rec['var_ratio_median_mask']:.3f}")
        print(f"{run.entity_prefix}: {len(todo)} regime(s) in {time.time() - t0:.0f} s")


def cmd_motion(args: argparse.Namespace) -> None:
    """T1.7: one FD row per run (no regime), raw and respiration-filtered, to tier1_motion.tsv."""
    import json

    import nibabel as nib
    import numpy as np
    import pandas as pd

    from neuroimaging import data_quality_motion as dqm
    from neuroimaging.io import load_confounds

    t0 = time.time()
    paths = Paths(args)
    physio_tsv = Path(args.physio_reality or paths.output_dir / "duckbrain" / "physio" / "physio_reality.tsv")
    runs = runs_for(args, paths)
    if not runs:
        sys.exit(f"No completed fMRIPrep run matches under {paths.fmriprep_tree}")
    trs = {}
    for run in runs:
        zooms = nib.load(str(run.bold)).header.get_zooms()
        trs[dqm.run_key(run.subject, run.session, run.task, run.run)] = float(zooms[3])

    # Pass 1: a breathing peak for every run with a usable respiratory recording.
    physio = dqm.physio_index(physio_tsv)
    status = {dqm.run_key(r.sub, r.ses, r.task, r.run): "unusable" for r in physio.itertuples(index=False)}
    usable = physio[physio["verdict"] == "usable"]
    run_peaks, physio_sha, peak_rows = {}, {}, []
    for rec in usable.itertuples(index=False):
        key = dqm.run_key(rec.sub, rec.ses, rec.task, rec.run)
        if key not in trs:
            continue  # a recording for a run with no completed fMRIPrep output
        run = next(r for r in runs if dqm.run_key(r.subject, r.session, r.task, r.run) == key)
        n_vol = len(load_confounds(run, columns=["framewise_displacement"]))
        trace, fs = dqm.in_scan_trace(paths.bids_root / rec.relpath, n_vol, trs[key])
        run_peaks[key] = dqm.breathing_peak(trace, fs)
        status[key] = "peak" if np.isfinite(run_peaks[key]) else "edge"
        physio_sha[key] = dq.file_sha256(paths.bids_root / rec.relpath)
        peak_rows.append({"sub": rec.sub, "peak_hz": run_peaks[key]})
    per_sub, sample = dqm.fallback_bands(pd.DataFrame(peak_rows))
    print(f"breathing peaks: {sum(v == 'peak' for v in status.values())} runs "
          f"({sum(v == 'edge' for v in status.values())} usable recordings with an edge maximum); " + ", ".join(
        f"sub-{s} [{b.lo:.3f}, {b.hi:.3f}] Hz" for s, b in sorted(per_sub.items()))
        + f"; sample [{sample.lo:.3f}, {sample.hi:.3f}] Hz")

    # Pass 2: every run.
    fmriprep_version = dq.pipeline_version(paths.fmriprep_tree)
    code_sha = dq.code_version(REPO_ROOT)
    rows = []
    for run in runs:
        key = dqm.run_key(run.subject, run.session, run.task, run.run)
        band = dqm.band_for(key, run_peaks, per_sub, sample)
        confounds = load_confounds(run)
        try:
            row = dqm.motion_row(confounds, trs[key], band)
            row.update(motion_task_row(run, confounds, trs[key]))
        except (ValueError, KeyError) as exc:
            raise type(exc)(f"{run.entity_prefix}: {exc}") from exc
        rows.append({
            "schema_version": dqm.SCHEMA_VERSION, "sub": run.subject, "ses": run.session,
            "task": run.task, "run": run.run, "repetition_time": trs[key],
            "resp_status": status.get(key, "none"), **row,
            "fmriprep_version": fmriprep_version,
            "input_confounds_sha256": dq.file_sha256(run.confounds),
            "input_physio_sha256": physio_sha.get(key), "code_version": code_sha,
        })
    table = pd.DataFrame(rows).sort_values(["sub", "ses", "task", "run"], na_position="first")
    out = paths.tree_root / f"{dqm.TABLE_NAME}.tsv"
    table.to_csv(out, sep="\t", index=False, na_rep="n/a", float_format="%.6f")
    prov = {
        "schema_version": dqm.SCHEMA_VERSION, "code_version": code_sha, "fmriprep_version": fmriprep_version,
        "physio_reality": str(physio_tsv), "physio_reality_sha256": dq.file_sha256(physio_tsv),
        "parameters": {"radius_mm": dqm.RADIUS_MM, "fd_thresholds": list(dqm.FD_THRESHOLDS),
                       "resp_search_hz": list(dqm.RESP_SEARCH_HZ), "half_width_hz": dqm.HALF_WIDTH_HZ,
                       "welch_segment_s": dqm.WELCH_SEGMENT_S, "min_trace_s": dqm.MIN_TRACE_S, "filter_order": dqm.FILTER_ORDER},
        "fallback_bands": {**{f"sub-{s}": [b.lo, b.hi] for s, b in sorted(per_sub.items())},
                           "sample": [sample.lo, sample.hi]},
    }
    (paths.tree_root / f"{dqm.TABLE_NAME}.json").write_text(json.dumps(prov, indent=2) + "\n")
    print(f"{out}: {len(table)} runs in {time.time() - t0:.0f} s; band source "
          f"{table['band_source'].value_counts().to_dict()}; resp_status {table['resp_status'].value_counts().to_dict()}; filter {table['filter_kind'].value_counts().to_dict()}; "
          f"max |FD - fMRIPrep| {table['fd_check_max_abs_diff'].max():.2g} mm")


def motion_task_row(run: FmriprepRun, confounds, tr: float) -> dict:
    """T1.8 for a GLM-eligible run (regime-free: the task columns are the same under every regime)."""
    empty = {"motion_task_r_max": None, "motion_task_r_motion": None, "motion_task_r_condition": None}
    if not dqg.eligible(run):
        return empty
    events = dqg.read_events(run)
    model = dqg.model_for(run, events)
    dm = dqg.design_for(run, events, confounds, tr, model, load_regimes()["none"])
    return dqg.motion_task_correlation(dm, model.conditions, confounds)


def glm_runs(args: argparse.Namespace, paths: Paths) -> list[FmriprepRun]:
    return [r for r in runs_for(args, paths) if dqg.eligible(r)]


def glm_keys(run: FmriprepRun, model, atlases_sha: str, bold_sha: Optional[str] = None) -> "dqg.GlmKeys":
    return dqg.GlmKeys(
        bold_sha256=bold_sha or dq.file_sha256(run.bold),
        events_sha256=dq.file_sha256(run.events),
        model_sha256=dqg.model_sha256(model),
        atlases_sha256=atlases_sha,
    )


def cmd_glm_plan(args: argparse.Namespace) -> None:
    paths = Paths(args)
    regimes = selected_regimes(args)
    runs = glm_runs(args, paths)
    atlases_sha = dq.atlases_sha256(paths.atlases_dir) if args.check_hashes else None
    stale_runs, n_cells, n_missing = [], 0, 0
    for run in runs:
        keys = None
        if args.check_hashes:
            keys = glm_keys(run, dqg.model_for(run, dqg.read_events(run)), atlases_sha)
        missing_here = 0
        for regime in regimes:
            n_cells += 1
            current = (dqg.glm_is_current(paths.tree_root, run, regime, keys) if keys
                       else dqg.glm_cell_exists(paths.tree_root, run, regime.name))
            missing_here += not current
        if missing_here:
            n_missing += missing_here
            stale_runs.append(run)
    print(f"task runs with events (excluding {sorted(dqg.EXCLUDED_TASKS)}): {len(runs)}  "
          f"regimes: {len(regimes)}  cells: {n_cells}  missing/stale cells: {n_missing}  "
          f"runs to (re)build: {len(stale_runs)}")
    if args.units:
        Path(args.units).write_text("".join(unit_line(r) + "\n" for r in stale_runs))
        print(f"wrote {len(stale_runs)} units to {args.units}")


def cmd_glm(args: argparse.Namespace) -> None:
    from neuroimaging.glm.config import repetition_time

    paths = Paths(args)
    regimes = selected_regimes(args)
    if args.units:
        if args.index is None:
            sys.exit("--units needs --index (1-based line number, e.g. $SLURM_ARRAY_TASK_ID)")
        lines = [ln for ln in Path(args.units).read_text().splitlines() if ln.strip()]
        if not 1 <= args.index <= len(lines):
            sys.exit(f"--index {args.index} is outside 1..{len(lines)} for {args.units}")
        runs = [run_from_unit(lines[args.index - 1], paths)]
    else:
        if not (args.sub and args.ses and args.task):
            sys.exit("glm needs --sub, --ses and --task (and --run for multi-run tasks), or --units/--index")
        runs = runs_for(args, paths)
    runs = [r for r in runs if dqg.eligible(r)]
    if not runs:
        sys.exit("No GLM-eligible run (task run with events, task not excluded) matches")
    fmriprep_version = dq.pipeline_version(paths.fmriprep_tree)
    code_sha = dq.code_version(REPO_ROOT)
    dq.ensure_dataset_description(paths.tree_root, paths.fmriprep_tree, fmriprep_version, code_sha)
    atlases_sha = dq.atlases_sha256(paths.atlases_dir)
    for run in runs:
        t0 = time.time()
        events = dqg.read_events(run)
        model = dqg.model_for(run, events)
        keys = glm_keys(run, model, atlases_sha)
        todo = [r for r in regimes if args.force or not dqg.glm_is_current(paths.tree_root, run, r, keys)]
        if not todo:
            print(f"{run.entity_prefix}: all {len(regimes)} GLM regimes current, skipping")
            continue
        inputs = dq.load_run_inputs(run, paths.atlases_dir)
        assert inputs.input_bold_sha256 == keys.bold_sha256 and inputs.input_atlases_sha256 == atlases_sha
        tr = repetition_time(run, paths.bids_root)
        if abs(tr - inputs.tr) > 1e-3:
            sys.exit(f"{run.entity_prefix}: sidecar RepetitionTime {tr} disagrees with the NIfTI header {inputs.tr}")
        for regime in todo:
            try:
                rec = dqg.write_run_glm(paths.tree_root, inputs, events, model, regime, tr, keys,
                                        fmriprep_version=fmriprep_version, code_sha=code_sha)
            except RegimeNotApplicable as exc:
                dqg.write_absent_glm(paths.tree_root, inputs, model, regime, exc, tr, keys,
                                     fmriprep_version=fmriprep_version, code_sha=code_sha)
                print(f"{run.entity_prefix} {regime.name:14s} ABSENT ({exc.n_available} of "
                      f"{exc.n_required} aCompCor components)")
                continue
            print(f"{run.entity_prefix} {regime.name:14s} p={rec['n_regressors']} dof={rec['dof_resid']} "
                  f"r2adj_med={rec['task_r2adj_median']:.4f} r2adj_p99={rec['task_r2adj_p99']:.3f} "
                  f"(raw med {rec['task_r2_median']:.4f})")
        print(f"{run.entity_prefix}: {len(todo)} GLM regime(s) in {time.time() - t0:.0f} s")


def cmd_collect(args: argparse.Namespace) -> None:
    paths = Paths(args)
    runs, parcels = dq.collect(paths.tree_root)
    glm, glm_parcels = dqg.collect(paths.tree_root)
    if runs.empty and glm.empty:
        sys.exit(f"No tier-1 sidecars under {paths.tree_root}; run `tier1.py run` or `tier1.py glm` first")
    if not runs.empty:
        runs_path = paths.tree_root / "tier1_runs.tsv"
        parcels_path = paths.tree_root / "tier1_parcels.tsv"
        runs.to_csv(runs_path, sep="\t", index=False, na_rep="n/a")
        parcels.to_csv(parcels_path, sep="\t", index=False, na_rep="n/a")
        print(f"{runs_path}: {len(runs)} run x regime rows")
        print(f"{parcels_path}: {len(parcels)} parcel rows")
        if "regime" in runs:
            print(runs.groupby("regime").agg(rows=("absent", "size"), absent=("absent", "sum")).to_string())
    if not glm.empty:
        glm_path = paths.tree_root / f"{dqg.TABLE_NAME}.tsv"
        glm_parcels_path = paths.tree_root / f"{dqg.PARCELS_TABLE_NAME}.tsv"
        glm.to_csv(glm_path, sep="\t", index=False, na_rep="n/a")
        glm_parcels.to_csv(glm_parcels_path, sep="\t", index=False, na_rep="n/a", float_format="%.6g")
        print(f"{glm_path}: {len(glm)} task run x regime rows ({int(glm['absent'].sum())} absent)")
        print(f"{glm_parcels_path}: {len(glm_parcels)} parcel rows")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--bids-root", help="override config paths.bids_project_dir")
    p.add_argument("--tree-root", help="override <output_dir>/data_quality")
    p.add_argument("--atlases-dir", help="override <output_dir>/atlases")
    p.add_argument("--variant", default=DEFAULT_VARIANT, help="fMRIPrep tree name (default %(default)s)")
    p.add_argument("--regimes", help="comma-separated regime names (default: every confirmed regime)")
    p.add_argument("--include-provisional", action="store_true",
                   help="also use regimes whose status is provisional")


def _entities(p: argparse.ArgumentParser) -> None:
    p.add_argument("--sub", help="bare label, e.g. 03")
    p.add_argument("--ses", help="bare label, e.g. 19")
    p.add_argument("--task")
    p.add_argument("--run", help="bare label, e.g. 01")


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="verb", required=True)

    p = sub.add_parser("regimes", help="print the regime registry")
    p.set_defaults(func=cmd_regimes)

    p = sub.add_parser("plan", help="list missing/stale cells; optionally write a units file")
    _common(p); _entities(p)
    p.add_argument("--units", help="write one line per run to (re)build here")
    p.add_argument("--check-hashes", action="store_true",
                   help="compare input hashes and regime versions, not just existence "
                        "(reads every BOLD: ~20 min for the whole dataset)")
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser("run", help="clean one run under every requested regime")
    _common(p); _entities(p)
    p.add_argument("--units", help="units file written by plan")
    p.add_argument("--index", type=int, help="1-based line in --units")
    p.add_argument("--force", action="store_true", help="rebuild cells that are current")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("motion", help="T1.7: FD per run, raw and respiration-filtered, to tier1_motion.tsv")
    _common(p); _entities(p)
    p.add_argument("--physio-reality", help="override <output_dir>/duckbrain/physio/physio_reality.tsv")
    p.set_defaults(func=cmd_motion)

    p = sub.add_parser("glm-plan", help="list missing/stale GLM cells (T1.5/T1.6); optionally write a units file")
    _common(p); _entities(p)
    p.add_argument("--units", help="write one line per task run to (re)build here")
    p.add_argument("--check-hashes", action="store_true",
                   help="compare input, events, model and atlas hashes, not just existence")
    p.set_defaults(func=cmd_glm_plan)

    p = sub.add_parser("glm", help="fit one task run's stand-in GLM under every requested regime")
    _common(p); _entities(p)
    p.add_argument("--units", help="units file written by glm-plan")
    p.add_argument("--index", type=int, help="1-based line in --units")
    p.add_argument("--force", action="store_true", help="rebuild cells that are current")
    p.set_defaults(func=cmd_glm)

    p = sub.add_parser("collect", help="flatten sidecars into the tier-1 tables")
    _common(p)
    p.set_defaults(func=cmd_collect)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
