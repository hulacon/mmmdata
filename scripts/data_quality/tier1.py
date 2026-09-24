#!/usr/bin/env python3
"""Tier 1 of the data-quality collection: per run × confound regime, rebuildable.

Verbs (all idempotent; state is on disk, never in this process):

  regimes   print the regime registry (name, status, columns, drift, version)
  plan      list the run × regime cells that are missing or stale, optionally
            writing a units file for the sbatch array
  run       clean ONE run under every requested regime (one BOLD load shared
            by all of them) and write its tier-1 outputs; skips cells that are
            current unless --force
  collect   flatten every sidecar into tier1_runs.tsv and tier1_parcels.tsv
            at the tree root

Provisional regimes (not yet confirmed by whoever defined them) are excluded
from every verb unless --include-provisional is given.

Usage:
    python tier1.py regimes
    python tier1.py plan --units units.txt
    python tier1.py run --sub 03 --ses 19 --task NATencoding --run 01
    python tier1.py run --units units.txt --index 7          # sbatch array
    python tier1.py collect

Library: src/python/neuroimaging/{confounds,data_quality}.py. Design record:
mmmdata-agents docs/workbench/data-quality/.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src" / "python"))

from core.config import load_config  # noqa: E402
from neuroimaging import data_quality as dq  # noqa: E402
from neuroimaging.confounds import confirmed_regimes, describe_regimes, load_regimes  # noqa: E402
from neuroimaging.constants import DEFAULT_SPACE, DEFAULT_VARIANT  # noqa: E402
from neuroimaging.io import FmriprepRun, find_fmriprep_runs  # noqa: E402


# ---------------------------------------------------------------------------
# Paths and regimes
# ---------------------------------------------------------------------------

class Paths:
    def __init__(self, args: argparse.Namespace):
        cfg = load_config()["paths"]
        self.bids_root = Path(args.bids_root or cfg["bids_project_dir"])
        output_dir = Path(cfg["output_dir"])
        self.tree_root = Path(args.tree_root or output_dir / dq.TREE_NAME)
        self.atlases_dir = Path(args.atlases_dir or output_dir / "atlases")
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
    for run in runs:
        sha = dq.file_sha256(run.bold) if args.check_hashes else None
        missing_here = 0
        for regime in regimes:
            n_cells += 1
            if sha is None:
                nii, js = dq.tsnr_paths(paths.tree_root, run, regime.name)
                current = nii.exists() and js.exists()
            else:
                current = dq.is_current(paths.tree_root, run, regime, sha)
            if not current:
                missing_here += 1
        if missing_here:
            n_missing += missing_here
            stale_runs.append(run)
    print(f"runs: {len(runs)}  regimes: {[r.name for r in regimes]}  cells: {n_cells}  "
          f"missing/stale cells: {n_missing}  runs to (re)build: {len(stale_runs)}")
    if not args.check_hashes:
        print("(existence only; --check-hashes also compares input hashes and regime versions)")
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
    dq.ensure_dataset_description(paths.tree_root, fmriprep_version, code_sha)

    for run in runs:
        t0 = time.time()
        sha = dq.file_sha256(run.bold)
        todo = [r for r in regimes if args.force or not dq.is_current(paths.tree_root, run, r, sha)]
        if not todo:
            print(f"{run.entity_prefix}: all {len(regimes)} regimes current, skipping")
            continue
        inputs = dq.load_run_inputs(run, paths.atlases_dir)
        assert inputs.input_bold_sha256 == sha
        for regime in todo:
            rec = dq.write_run_regime(
                paths.tree_root, inputs, regime,
                fmriprep_version=fmriprep_version, code_sha=code_sha,
            )
            print(f"{run.entity_prefix} {regime.name:10s} n_vol={rec['n_vol']} nss={rec['n_nss']} "
                  f"p={rec['n_regressors']} dof={rec['dof_resid']} "
                  f"tsnr_med={rec['tsnr_median_mask']:.2f} var_ratio_med={rec['var_ratio_median_mask']:.3f}")
        print(f"{run.entity_prefix}: {len(todo)} regime(s) in {time.time() - t0:.0f} s")


def cmd_collect(args: argparse.Namespace) -> None:
    paths = Paths(args)
    runs, parcels = dq.collect(paths.tree_root)
    if runs.empty:
        sys.exit(f"No tier-1 sidecars under {paths.tree_root}; run `tier1.py run` first")
    runs_path = paths.tree_root / "tier1_runs.tsv"
    parcels_path = paths.tree_root / "tier1_parcels.tsv"
    runs.to_csv(runs_path, sep="\t", index=False, na_rep="n/a")
    parcels.to_csv(parcels_path, sep="\t", index=False, na_rep="n/a")
    print(f"{runs_path}: {len(runs)} run x regime rows")
    print(f"{parcels_path}: {len(parcels)} parcel rows")
    if "regime" in runs:
        print(runs.groupby("regime").size().to_string())


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

    p = sub.add_parser("collect", help="flatten sidecars into the tier-1 tables")
    _common(p)
    p.set_defaults(func=cmd_collect)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
