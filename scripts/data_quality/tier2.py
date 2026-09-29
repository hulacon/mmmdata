#!/usr/bin/env python3
"""Tier 2 of the data-quality collection, rebuilt from tier-1 caches.

Two parts, each in its own directory under <tree>/tier2/:

  naturalistic   LOO-ISFC and the audio-envelope lag scan for every film's
                 first viewing (registry T2.2, T2.4); repeat reliability of
                 the recurring films (T2.1) and within- vs between-film
                 discriminability per session (T2.3)
  connectivity   rest-run FC, within-subject QC-FC, fingerprinting and the
                 hippocampal FC profile (T2.5-T2.7); also reads
                 tier1_motion.tsv (`tier1.py motion`)

Verbs (idempotent; state is on disk, never in this process):

  build   every requested part x every regime in tier1_runs.tsv, written to
          <tree>/tier2/<part>/ (or <out-dir>/<part>/). A full rebuild
          replaces a part's tables; nothing is appended.
  diff    compare two tier-2 roots part by part, table by table and array by
          array; exit 1 on any difference. Settles-when 4 is
          `build --out-dir <tmp>` then `diff <tree>/tier2 <tmp>`.

Usage:
    python tier2.py build --provisional-subjects 06,07
    python tier2.py build --parts naturalistic --regimes base12fd,gsr --films bench --out-dir /tmp/t2
    python tier2.py diff A B

Library: src/python/neuroimaging/data_quality_{tier2,connectivity}.py. Design
record: mmmdata-agents docs/workbench/data-quality/.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import sys
import time
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src" / "python") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src" / "python"))

from core.config import load_config  # noqa: E402
from neuroimaging import data_quality as dq  # noqa: E402
from neuroimaging import data_quality_connectivity as dqc  # noqa: E402
from neuroimaging import data_quality_tier2 as t2  # noqa: E402
from neuroimaging.io import find_events_file  # noqa: E402


class Paths:
    def __init__(self, args: argparse.Namespace):
        cfg = load_config()["paths"]
        self.bids_root = Path(args.bids_root or cfg["bids_project_dir"])
        output_dir = Path(cfg["output_dir"])
        self.tree_root = Path(args.tree_root or output_dir / dq.TREE_NAME)
        self.registry = Path(args.registry or self.bids_root / "stimuli" / "stimulus_registry" / "movies.tsv")
        self.feature_store = Path(args.feature_store or output_dir / "stimuli_features" / "psytwill")
        self.atlases_dir = Path(args.atlases_dir or output_dir / "atlases")


PARTS = ("naturalistic", "connectivity")


def _csv(value: Optional[str]) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()] if value else []


def cmd_build(args: argparse.Namespace) -> None:
    parts = _csv(args.parts) or list(PARTS)
    unknown = sorted(set(parts) - set(PARTS))
    if unknown:
        sys.exit(f"Unknown part(s) {unknown}; choose from {PARTS}")
    root = Path(args.out_dir) if args.out_dir else None
    if "naturalistic" in parts:
        build_naturalistic(args, root / "naturalistic" if root else None)
    if "connectivity" in parts:
        build_connectivity(args, root / "connectivity" if root else None)


def build_connectivity(args: argparse.Namespace, dest: Optional[Path]) -> None:
    t0 = time.time()
    paths = Paths(args)
    runs = t2.load_tier1_runs(paths.tree_root)
    provisional = [s.removeprefix("sub-") for s in _csv(args.provisional_subjects)]
    rest = dqc.rest_runs(runs, dqc.load_motion(paths.tree_root), provisional)
    available = sorted(runs.loc[runs["task"].isin(dqc.REST_TASKS), "regime"].unique())
    regimes = _csv(args.regimes) or available
    unknown = sorted(set(regimes) - set(available))
    if unknown:
        sys.exit(f"Regime(s) {unknown} have no rest rows in tier1_runs.tsv; available: {available}")
    print(f"connectivity: {len(rest)} rest runs, subjects {rest['sub'].value_counts().sort_index().to_dict()}, "
          f"provisional {provisional or 'none'}; regimes: {', '.join(regimes)}")
    absent = set(runs.loc[runs["absent"], ["sub", "ses", "task", "run", "regime"]].itertuples(index=False, name=None))
    rename = t2.schaefer_current_names(paths.atlases_dir)
    centroids = dqc.parcel_centroids(paths.atlases_dir)
    centroids["name"] = centroids["name"].map(rename)
    result = dqc.compute(paths.tree_root, rest, regimes, centroids, absent, rename)
    dest = dest or dqc.out_dir(paths.tree_root)
    provenance = {
        "created": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "code_version": dq.code_version(REPO_ROOT),
        "inputs": {
            "tier1_runs": str(paths.tree_root / "tier1_runs.tsv"),
            "tier1_runs_sha256": t2.file_sha256(paths.tree_root / "tier1_runs.tsv"),
            "tier1_motion": str(paths.tree_root / "tier1_motion.tsv"),
            "tier1_motion_sha256": t2.file_sha256(paths.tree_root / "tier1_motion.tsv"),
            "schaefer_dseg": str(paths.atlases_dir / f"{dqc.SCHAEFER_DSEG}.nii.gz"),
        },
        "regimes": regimes,
        "provisional_subjects": provisional,
    }
    written = dqc.write(result, dest, provenance)
    print(f"{dest}: {len(written)} files; qcfc {len(result.qcfc)} rows, fingerprint {len(result.fingerprint)}, "
          f"hipp_profile {len(result.hipp_profile)}, {len(result.fc)} FC arrays; "
          f"{len(result.skipped)} skipped in {time.time() - t0:.0f} s")
    for s in result.skipped:
        print(f"  skipped: {s}")


def build_naturalistic(args: argparse.Namespace, dest: Optional[Path]) -> None:
    t0 = time.time()
    paths = Paths(args)
    runs = t2.load_tier1_runs(paths.tree_root)
    available = sorted(runs.loc[runs["task"] == t2.TASK, "regime"].unique())
    regimes = _csv(args.regimes) or available
    unknown = sorted(set(regimes) - set(available))
    if unknown:
        sys.exit(f"Regime(s) {unknown} have no {t2.TASK} rows in tier1_runs.tsv; available: {available}")

    names = t2.movie_name_index(paths.registry)
    viewings = t2.film_viewings(
        runs, lambda s, e, r: find_events_file(s, e, t2.TASK, r, bids_root=paths.bids_root), names)
    films = _csv(args.films)
    if films:
        missing = sorted(set(films) - set(viewings["stimulus_id"]))
        if missing:
            sys.exit(f"No {t2.TASK} showing of {missing}")
        viewings = viewings[viewings["stimulus_id"].isin(films)].reset_index(drop=True)
    first = viewings[viewings["first_viewing"]]
    print(f"{len(viewings)} showings, {first['stimulus_id'].nunique()} films, "
          f"{first['sub'].nunique()} subjects; regimes: {', '.join(regimes)}")

    envelopes = t2.load_envelopes(paths.feature_store, first["stimulus_id"])
    rename = t2.schaefer_current_names(paths.atlases_dir)
    result = t2.compute(paths.tree_root, viewings, envelopes, regimes, runs, rename)
    dest = dest or t2.out_dir(paths.tree_root)
    provenance = {
        "created": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "code_version": dq.code_version(REPO_ROOT),
        "inputs": {
            "tier1_runs": str(paths.tree_root / "tier1_runs.tsv"),
            "tier1_runs_sha256": t2.file_sha256(paths.tree_root / "tier1_runs.tsv"),
            "registry": str(paths.registry),
            "registry_sha256": t2.file_sha256(paths.registry),
            "feature_store": str(paths.feature_store / f"{t2.FEATURE_GROUP}_features.parquet"),
            "schaefer_names": str(paths.atlases_dir / t2.SCHAEFER_CURRENT_TABLE),
        },
        "regimes": regimes,
        "films": sorted(first["stimulus_id"].unique()),
    }
    written = t2.write(result, dest, provenance)
    print(f"naturalistic {dest}: {len(written)} files; loo_isc {len(result.loo_isc)} rows, "
          f"envelope_lag {len(result.envelope_lag)}, envelope_parcel {len(result.envelope_parcel)}, "
          f"wsc {len(result.wsc)}, wsc_pairs {len(result.wsc_pairs)}, discriminability {len(result.discriminability)}, "
          f"{len(result.isfc_group)} group ISFC matrices; {len(result.skipped)} skipped "
          f"in {time.time() - t0:.0f} s")
    for s in result.skipped:
        print(f"  skipped: {s}")


def cmd_diff(args: argparse.Namespace) -> None:
    a, b = Path(args.a), Path(args.b)
    modules = {"naturalistic": t2, "connectivity": dqc}
    problems = []
    for part, module in modules.items():
        pa, pb = a / part, b / part
        if pa.exists() or pb.exists():
            problems += [f"{part}: {p}" for p in module.diff(pa, pb)]
    if not any((a / p).exists() or (b / p).exists() for p in modules):
        problems.append(f"neither {a} nor {b} holds a tier-2 part ({', '.join(modules)})")
    for p in problems:
        print(p)
    if problems:
        sys.exit(1)
    print("identical")


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="verb", required=True)

    p = sub.add_parser("build", help="rebuild tier-2 parts from tier-1 caches")
    p.add_argument("--parts", help=f"comma-separated parts (default: {','.join(PARTS)})")
    p.add_argument("--provisional-subjects",
                   help="comma-separated subjects whose rows are provisional (fMRIPrep due a rerun), e.g. 06,07")
    p.add_argument("--bids-root", help="override config paths.bids_project_dir")
    p.add_argument("--tree-root", help="override <output_dir>/data_quality")
    p.add_argument("--registry", help="override <bids_root>/stimuli/stimulus_registry/movies.tsv")
    p.add_argument("--feature-store", help="override <output_dir>/stimuli_features/psytwill")
    p.add_argument("--atlases-dir", help="override <output_dir>/atlases (Schaefer name tables)")
    p.add_argument("--regimes", help="comma-separated regimes (default: every regime in tier1_runs.tsv)")
    p.add_argument("--films", help="naturalistic: comma-separated stimulus_ids (default: every film)")
    p.add_argument("--out-dir", help="tier-2 root to write the parts under, instead of <tree>/tier2")
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("diff", help="compare two tier-2 roots part by part; exit 1 on any difference")
    p.add_argument("a")
    p.add_argument("b")
    p.set_defaults(func=cmd_diff)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
