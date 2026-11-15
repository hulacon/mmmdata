#!/usr/bin/env python3
"""Inputs for ``psytwill bench boundary`` on the films: boundaries, shot cuts, scored spans.

Writes three tables into ``-o``:

  boundaries.csv  stimulus_id, level, onset, annotator    every annotated segment onset > 0, all levels
  cuts.csv        stimulus_id, time                        shot cuts the cut classifier accepted
  durations.csv   stimulus_id, duration                    each film's scored span (registry duration_s,
                                                           or a --cap STIM=SECONDS where given)

Sources: the store's segment annotations (``_inputs/movies_annot_all.tsv``),
the cut schema table (``corpus, id, time, classifier_accept``; films are
``corpus == mmmdata``), and the stimulus registry's ``duration_s``. Each film
must have exactly one annotator (the bench's film test nests films in
annotators), a registry duration, and at least one segment; a ``--cap``
naming an unknown film or longer than the film stops the build.

Usage:
    python film_boundaries.py -o <dir> [--cap STIM=SECONDS ...] [--store DIR] [--cut-schema FILE] [--registry FILE]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
if str(REPO_ROOT / "src" / "python") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src" / "python"))

SEGMENTS = "_inputs/movies_annot_all.tsv"
CUT_SCHEMA = "space-timescale/cuts/cut_schema_v1.parquet"
REGISTRY = "stimuli/stimulus_registry/movies.tsv"


def build(seg: pd.DataFrame, cuts: pd.DataFrame, reg: pd.DataFrame, caps: dict[str, float]):
    b = seg.loc[seg["onset"] > 0, ["stimulus_id", "level", "onset", "annotator"]].copy()
    films = sorted(seg["stimulus_id"].unique())
    n_ann = seg.groupby("stimulus_id")["annotator"].nunique()
    if (n_ann > 1).any():
        raise SystemExit(f"films with more than one annotator: {sorted(n_ann[n_ann > 1].index)}")
    dur = reg.set_index("stimulus_id")["duration_s"].astype(float)
    missing = sorted(set(films) - set(dur.index))
    if missing:
        raise SystemExit(f"{len(missing)} annotated film(s) have no registry duration (e.g. {missing[:3]})")
    d = dur.loc[films].copy()
    for sid, cap in caps.items():
        if sid not in d.index:
            raise SystemExit(f"--cap names {sid!r}, which is not an annotated film")
        if cap > d[sid]:
            raise SystemExit(f"--cap {sid}={cap} is longer than the film ({d[sid]} s)")
        d[sid] = cap
    c = cuts[(cuts["corpus"] == "mmmdata") & cuts["classifier_accept"].astype(bool)]
    c = c.rename(columns={"id": "stimulus_id"})[["stimulus_id", "time"]]
    unknown = sorted(set(c["stimulus_id"]) - set(films))
    if unknown:
        raise SystemExit(f"cuts name {len(unknown)} film(s) with no annotations (e.g. {unknown[:3]})")
    durations = pd.DataFrame({"stimulus_id": d.index, "duration": d.to_numpy()})
    return b.sort_values(["stimulus_id", "level", "onset"]), c.sort_values(["stimulus_id", "time"]), durations


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-o", "--output", required=True, help="output directory")
    ap.add_argument("--cap", action="append", default=[], metavar="STIM=SECONDS",
                    help="score this film on [0, SECONDS) only; repeatable")
    ap.add_argument("--store", help="stimuli_features root (default: <config output_dir>/stimuli_features)")
    ap.add_argument("--cut-schema", help=f"cut table (default: <config fit_corpora_dir>/{CUT_SCHEMA})")
    ap.add_argument("--registry", help=f"film registry (default: <config bids_project_dir>/{REGISTRY})")
    args = ap.parse_args(argv)
    if not (args.store and args.cut_schema and args.registry):
        from core.config import load_config

        paths = load_config()["paths"]
    store = Path(args.store) if args.store else Path(paths["output_dir"]) / "stimuli_features"
    cut_schema = Path(args.cut_schema) if args.cut_schema else Path(paths["fit_corpora_dir"]) / CUT_SCHEMA
    registry = Path(args.registry) if args.registry else Path(paths["bids_project_dir"]) / REGISTRY
    caps = {}
    for spec in args.cap:
        sid, sep, sec = spec.partition("=")
        if not sep:
            raise SystemExit(f"--cap {spec!r}: expected STIM=SECONDS")
        caps[sid] = float(sec)
    b, c, d = build(pd.read_csv(store / SEGMENTS, sep="\t"), pd.read_parquet(cut_schema),
                    pd.read_csv(registry, sep="\t"), caps)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    b.to_csv(out / "boundaries.csv", index=False)
    c.to_csv(out / "cuts.csv", index=False)
    d.to_csv(out / "durations.csv", index=False)
    print(f"  boundaries: {len(b)} over {b['stimulus_id'].nunique()} films ("
          + ", ".join(f"{lv} {n}" for lv, n in b.groupby("level").size().items()) + ")")
    print(f"  cuts: {len(c)} accepted over {c['stimulus_id'].nunique()} films; durations: {len(d)} films, caps {caps}")
    print(f"  -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
