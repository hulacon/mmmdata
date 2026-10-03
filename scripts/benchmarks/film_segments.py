#!/usr/bin/env python3
"""One row per annotated fine-grained film segment (SEG-C): the segments table for T5.

The input to ``psytwill bench retrieval --target-segments``: a human's SEG-C
description (a chunk in the store's annotation table, keyed
``stimulus_id, chunk_idx``) has to find its moment among the other segments
of the same film, where a moment is the film's frame grid mean-pooled over
``[onset, offset)``.

Columns: ``stimulus_id, chunk_idx, onset, offset, duration, annotator``.
``annotator`` is carried for the per-annotator breakdown: each film has one
annotator and nothing is double-coded, so it is a covariate, not a unit.

The store's chunk index is trusted only after it is proven to be the SEG-C
segment: per film, the chunks in ``chunk_idx`` order must carry exactly the
onsets and offsets of the SEG-C rows in ``seg_number`` order. Any mismatch,
or a film whose frame grid ends before its last segment, stops the build.

Usage:
    python film_segments.py -o <dir>/segc_segments.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
if str(REPO_ROOT / "src" / "python") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src" / "python"))

from core.config import load_config  # noqa: E402

ANNOT_TABLE = "psytwill/movies_annot_chunks_features.parquet"
GRID_TABLE = "psytwill/movies_frames_features.parquet"
SEGMENTS = "_inputs/movies_annot_all.tsv"


def store_chunks(path: Path) -> pd.DataFrame:
    """Distinct (stimulus_id, chunk_idx, onset, offset) of one model's rows."""
    pf = pq.ParquetFile(path)
    model = None
    for i in range(pf.num_row_groups):
        m = pf.read_row_group(i, columns=["model"])["model"]
        if len(m):
            model = m[0].as_py()
            break
    t = pq.read_table(path, columns=["stimulus_id", "chunk_idx", "onset", "offset", "model"],
                      filters=[("model", "=", model)])
    df = t.drop_columns(["model"]).to_pandas().drop_duplicates()
    if df.duplicated(["stimulus_id", "chunk_idx"]).any():
        raise SystemExit(f"{path}: a (stimulus_id, chunk_idx) carries two different time spans")
    return df


def grid_end(path: Path) -> pd.Series:
    """Last frame-grid stamp per film, from one model of the frame table."""
    first = pq.ParquetFile(path).read_row_group(0, columns=["model"])["model"][0].as_py()
    t = pq.read_table(path, columns=["stimulus_id", "time"], filters=[("model", "=", first)])
    return t.to_pandas().groupby("stimulus_id")["time"].max()


def build(store: Path) -> pd.DataFrame:
    seg = pd.read_csv(store / SEGMENTS, sep="\t")
    segc = seg[seg["level"] == "C"].sort_values(["stimulus_id", "seg_number"])
    chunks = store_chunks(store / ANNOT_TABLE).sort_values(["stimulus_id", "chunk_idx"])
    films_c, films_k = set(segc["stimulus_id"]), set(chunks["stimulus_id"])
    if films_c != films_k:
        raise SystemExit(f"SEG-C films and store chunk films differ: only SEG-C {sorted(films_c - films_k)}, "
                         f"only store {sorted(films_k - films_c)}")
    rows = []
    for film, c in segc.groupby("stimulus_id"):
        k = chunks[chunks["stimulus_id"] == film]
        if len(k) != len(c) or not (np.allclose(k["onset"], c["onset"]) and np.allclose(k["offset"], c["offset"])):
            raise SystemExit(f"{film}: store chunks do not match SEG-C rows in order "
                             f"({len(k)} chunks vs {len(c)} segments)")
        annot = c["annotator"].unique()
        if len(annot) != 1:
            raise SystemExit(f"{film}: SEG-C rows name {len(annot)} annotators; one per film was expected")
        for ci, r in zip(k["chunk_idx"], c.itertuples()):
            rows.append({"stimulus_id": film, "chunk_idx": int(ci), "onset": float(r.onset),
                         "offset": float(r.offset), "duration": float(r.offset - r.onset), "annotator": annot[0]})
    out = pd.DataFrame(rows)
    ends = grid_end(store / GRID_TABLE)
    over = out[out["onset"] > out["stimulus_id"].map(ends)]
    if not over.empty:
        raise SystemExit(f"{len(over)} segment(s) start after their film's last frame (e.g. "
                         f"{over[['stimulus_id', 'chunk_idx']].head(3).values.tolist()})")
    no_c = sorted(set(seg["stimulus_id"]) - films_c)
    print(f"  {len(out)} SEG-C segments over {out['stimulus_id'].nunique()} films"
          + (f"; films with no SEG-C (left out): {no_c}" if no_c else ""))
    print(f"  segments per film: median {out.groupby('stimulus_id').size().median():.0f}, "
          f"min {out.groupby('stimulus_id').size().min()}; duration median {out['duration'].median():.1f} s, "
          f"min {out['duration'].min():.1f} s")
    print("  per annotator (films / segments): " + ", ".join(
        f"{a} {g['stimulus_id'].nunique()}/{len(g)}" for a, g in out.groupby("annotator")))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-o", "--output", required=True, help="output CSV")
    ap.add_argument("--store", help="stimuli_features root (default: <config output_dir>/stimuli_features)")
    args = ap.parse_args(argv)
    store = Path(args.store) if args.store else Path(load_config()["paths"]["output_dir"]) / "stimuli_features"
    out = build(store)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"  -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
