#!/usr/bin/env python3
"""Speech share of each film segment, under the diarize frame gate.

Splits a segment-level film benchmark (``psytwill bench retrieval
--subset``) into speech and non-speech items. A segment's frames are the
film grid's bins whose start lies in ``[onset, offset)``, the same rows
``bench retrieval --*-segments`` pools. A frame is speech when the union of
the film's diarize turns (``<movies dir>/<film>/diarize_speakers.csv``,
``onset, offset`` per turn) covers at least ``--min-coverage`` of
``[t, t + window)``. Overlapping turns (two speakers) count once.

Columns: ``stimulus_id, chunk_idx, n_frames, n_speech_frames, speech_share``.
With ``--lists DIR``, also writes ``speech.txt`` and ``nonspeech.txt``
(``stimulus_id|chunk_idx`` labels) split at ``--split`` on
``speech_share`` (speech when ``>= split``).

A film without a diarize file stops the build: an absent file is not a film
without speech. A file with no turns is a film without speech.

Usage:
    python segment_speech.py --segments <segments.csv> -o <speech.csv> [--movies-dir DIR] [--lists DIR]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
if str(REPO_ROOT / "src" / "python") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src" / "python"))


def _union(turns: np.ndarray) -> np.ndarray:
    """Merge overlapping (onset, offset) intervals."""
    if len(turns) == 0:
        return turns.reshape(0, 2)
    t = turns[np.argsort(turns[:, 0])]
    out = [t[0].copy()]
    for on, off in t[1:]:
        if on <= out[-1][1]:
            out[-1][1] = max(out[-1][1], off)
        else:
            out.append(np.array([on, off]))
    return np.vstack(out)


def frame_coverage(turns: np.ndarray, starts: np.ndarray, window: float) -> np.ndarray:
    """Share of each frame ``[t, t + window)`` covered by the union of the turns."""
    u = _union(turns)
    cov = np.zeros(len(starts))
    for on, off in u:
        cov += np.clip(np.minimum(starts + window, off) - np.maximum(starts, on), 0, None)
    return cov / window


def build(segments: pd.DataFrame, movies_dir: Path, *, window: float, min_coverage: float) -> pd.DataFrame:
    need = {"stimulus_id", "chunk_idx", "onset", "offset"}
    if not need <= set(segments.columns):
        raise SystemExit(f"segments table lacks {sorted(need - set(segments.columns))}")
    rows = []
    for film, seg in segments.groupby("stimulus_id"):
        f = movies_dir / film / "diarize_speakers.csv"
        if not f.exists():
            raise SystemExit(f"{film}: no diarize file at {f}; an absent file is not a film without speech")
        d = pd.read_csv(f)
        if not {"onset", "offset"} <= set(d.columns):
            raise SystemExit(f"{f}: needs onset, offset columns")
        turns = d[["onset", "offset"]].to_numpy(dtype=float)
        for r in seg.itertuples(index=False):
            k0 = int(np.ceil(r.onset / window - 1e-9))
            starts = np.arange(k0, int(np.ceil(r.offset / window - 1e-9))) * window
            starts = starts[(starts >= r.onset - 1e-9) & (starts < r.offset - 1e-9)]
            if starts.size == 0:
                raise SystemExit(f"{film} segment {r.chunk_idx}: no {window} s frame starts in "
                                 f"[{r.onset}, {r.offset})")
            gated = frame_coverage(turns, starts, window) >= min_coverage
            rows.append({"stimulus_id": film, "chunk_idx": int(r.chunk_idx), "n_frames": int(starts.size),
                         "n_speech_frames": int(gated.sum()), "speech_share": float(gated.mean())})
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--segments", required=True, help="table with stimulus_id, chunk_idx, onset, offset")
    ap.add_argument("-o", "--output", required=True, help="output CSV")
    ap.add_argument("--movies-dir", help="per-film feature dirs (default: <config output_dir>/stimuli_features/movies)")
    ap.add_argument("--window", type=float, default=0.5, help="grid bin width, seconds")
    ap.add_argument("--min-coverage", type=float, default=0.25, help="turn coverage that gates a frame as speech")
    ap.add_argument("--lists", help="directory for speech.txt / nonspeech.txt item lists")
    ap.add_argument("--split", type=float, default=0.5, help="speech_share at or above which a segment is speech")
    args = ap.parse_args(argv)
    if args.movies_dir:
        movies = Path(args.movies_dir)
    else:
        from core.config import load_config

        movies = Path(load_config()["paths"]["output_dir"]) / "stimuli_features" / "movies"
    out = build(pd.read_csv(args.segments), movies, window=args.window, min_coverage=args.min_coverage)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    sp = out["speech_share"] >= args.split
    print(f"  {len(out)} segments over {out['stimulus_id'].nunique()} films; speech_share >= {args.split}: "
          f"{int(sp.sum())}, below: {int((~sp).sum())}, zero: {int((out['speech_share'] == 0).sum())}")
    if args.lists:
        d = Path(args.lists)
        d.mkdir(parents=True, exist_ok=True)
        lab = out["stimulus_id"] + "|" + out["chunk_idx"].astype(str)
        (d / "speech.txt").write_text("\n".join(lab[sp]) + "\n")
        (d / "nonspeech.txt").write_text("\n".join(lab[~sp]) + "\n")
        print(f"  lists -> {d}/speech.txt, nonspeech.txt")
    print(f"  -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
