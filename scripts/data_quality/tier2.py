#!/usr/bin/env python3
"""Tier 2 of the data-quality collection: naturalistic measures, rebuilt from tier-1 caches.

Verbs (idempotent; state is on disk, never in this process):

  build   LOO-ISFC and the audio-envelope lag scan for every film's first
          viewing x every regime in tier1_runs.tsv, written to
          <tree>/tier2/naturalistic/ (or --out-dir). A full rebuild replaces
          the tables; nothing is appended.
  diff    compare two tier-2 trees table by table and matrix by matrix; exit 1
          on any difference. Settles-when 4 is `build --out-dir <tmp>` then
          `diff <tree>/tier2/naturalistic <tmp>`.

Usage:
    python tier2.py build
    python tier2.py build --regimes base12fd,gsr --films bench,negative-space --out-dir /tmp/t2
    python tier2.py diff A B

Library: src/python/neuroimaging/data_quality_tier2.py. Design record:
mmmdata-agents docs/workbench/data-quality/.
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


def _csv(value: Optional[str]) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()] if value else []


def cmd_build(args: argparse.Namespace) -> None:
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
    dest = Path(args.out_dir) if args.out_dir else t2.out_dir(paths.tree_root)
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
    print(f"{dest}: {len(written)} files; loo_isc {len(result.loo_isc)} rows, "
          f"envelope_lag {len(result.envelope_lag)}, envelope_parcel {len(result.envelope_parcel)}, "
          f"{len(result.isfc_group)} group ISFC matrices; {len(result.skipped)} skipped "
          f"in {time.time() - t0:.0f} s")
    for s in result.skipped:
        print(f"  skipped: {s}")


def cmd_diff(args: argparse.Namespace) -> None:
    problems = t2.diff(Path(args.a), Path(args.b))
    for p in problems:
        print(p)
    if problems:
        sys.exit(1)
    print("identical")


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="verb", required=True)

    p = sub.add_parser("build", help="rebuild the naturalistic tier-2 tables from tier-1 caches")
    p.add_argument("--bids-root", help="override config paths.bids_project_dir")
    p.add_argument("--tree-root", help="override <output_dir>/data_quality")
    p.add_argument("--registry", help="override <bids_root>/stimuli/stimulus_registry/movies.tsv")
    p.add_argument("--feature-store", help="override <output_dir>/stimuli_features/psytwill")
    p.add_argument("--atlases-dir", help="override <output_dir>/atlases (Schaefer name tables)")
    p.add_argument("--regimes", help="comma-separated regimes (default: every regime in tier1_runs.tsv)")
    p.add_argument("--films", help="comma-separated stimulus_ids (default: every film)")
    p.add_argument("--out-dir", help="write here instead of <tree>/tier2/naturalistic")
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("diff", help="compare two tier-2 trees; exit 1 on any difference")
    p.add_argument("a")
    p.add_argument("b")
    p.set_defaults(func=cmd_diff)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
