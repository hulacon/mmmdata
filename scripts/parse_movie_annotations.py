#!/usr/bin/env python
"""Parse the human movie scene annotations into one tidy long table.

`stimuli/movies/movie_annotations/*.xlsx` holds a two-level human
segmentation per movie: SEG-B (coarse events, short label) and SEG-C (fine
sub-segments, rich free-text description), each with start/end times.

**The times are `m.ss`, not decimal minutes and not seconds.** `0.45` is
45 s; `1.06` is 66 s; `1.5` is 1 min 50 s (Excel dropped the trailing zero
of `1.50`). Anything that reads these as floats and multiplies by 60 is
wrong by up to 40%, silently. This module owns the conversion so no
analysis re-derives it -- and `--check` proves it against each movie's
registry `duration_s`.

Output columns (long, one row per segment):
    stimulus_id, level (B|C), seg_number, onset, offset, duration,
    description, annotator, source_file

Usage:
    python parse_movie_annotations.py --check           # validate, write nothing
    python parse_movie_annotations.py -o annotations.tsv
"""

from __future__ import annotations

import argparse
import datetime
import math
import re
import sys
from pathlib import Path

import pandas as pd

try:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "python"))
    from core.config import load_config

    _cfg = load_config()
    BIDS_ROOT = Path(_cfg["paths"]["bids_project_dir"])
except Exception:  # noqa: BLE001 - config is a convenience, not a dependency
    BIDS_ROOT = Path("/gpfs/projects/hulacon/shared/mmmdata")

ANNOT_DIR = BIDS_ROOT / "stimuli" / "movies" / "movie_annotations"
REGISTRY = BIDS_ROOT / "stimuli" / "stimulus_registry" / "movies.tsv"

# Times whose fractional part implies >= 60 seconds are not m.ss at all.
_MAX_SS = 59


def parse_mss(value) -> float | None:
    """`m.ss` -> seconds. Returns None for blanks; raises on out-of-range ss.

    Handles the four spellings seen in these files: a float from Excel
    (`1.5` meaning 1:50), a string (`"1.06"`), a colon form (`"1:06"`), and
    a `datetime.time` -- Excel autoformatted exactly one cell
    (Negative Space, `4.48` -> `04:48:00`), where the hour field carries the
    minutes and the minute field carries the seconds.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, datetime.time):
        if value.second:
            raise ValueError(
                f"{value!r}: seconds field set on an autoformatted cell; the "
                f"m.ss->time mapping (hour=minutes, minute=seconds) does not "
                f"cover it. Check by hand."
            )
        return value.hour * 60 + value.minute
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        if ":" in s:
            mins, _, secs = s.partition(":")
            return int(mins or 0) * 60 + float(secs or 0)
        value = float(s)
    value = float(value)
    minutes = math.floor(value)
    # Two decimal places is the whole point: .5 is 50 seconds, not 5.
    seconds = round(round(value - minutes, 4) * 100)
    if seconds > _MAX_SS:
        raise ValueError(
            f"{value!r} parses to {seconds}s in the fractional field, which is "
            f"not m.ss -- the file may use decimal minutes. Check by hand."
        )
    return minutes * 60 + seconds


def _annotator(path: Path) -> str | None:
    m = re.search(r"_master_([A-Za-z]+)\.xlsx$", path.name)
    return m.group(1).upper() if m else None


def _col(df: pd.DataFrame, *candidates: str) -> str | None:
    lowered = {c.lower().strip(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lowered:
            return lowered[cand.lower()]
    return None


def parse_file(path: Path, stimulus_id: str) -> pd.DataFrame:
    """One annotation workbook -> long rows for both levels."""
    raw = pd.read_excel(path)
    rows = []

    levels = (
        ("B", _col(raw, "SEG-B Number"), _col(raw, "Start Time (m.ss)"),
         _col(raw, "End Time (m.ss)"), _col(raw, "SEG-B Description")),
        ("C", _col(raw, "SEG-C Number"), _col(raw, "SEG-C Start Time (m.ss)"),
         _col(raw, "SEG-C End Time (m.ss)"), _col(raw, "SEG-C Description")),
    )
    for level, n_col, s_col, e_col, d_col in levels:
        if not all((n_col, s_col, e_col)):
            continue
        sub = raw[[c for c in (n_col, s_col, e_col, d_col) if c]].dropna(how="all")
        for _, r in sub.iterrows():
            if pd.isna(r[n_col]):
                continue
            try:
                onset = parse_mss(r[s_col])
                offset = parse_mss(r[e_col])
            except ValueError as exc:
                raise ValueError(f"{path.name} {level}{r[n_col]}: {exc}") from exc
            if onset is None or offset is None:
                continue
            rows.append({
                "stimulus_id": stimulus_id,
                "level": level,
                "seg_number": int(r[n_col]),
                "onset": onset,
                "offset": offset,
                "duration": offset - onset,
                "description": (str(r[d_col]).strip() if d_col and not pd.isna(r[d_col]) else ""),
                "annotator": _annotator(path),
                "source_file": path.name,
            })
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="validate against registry durations; write nothing")
    ap.add_argument("-o", "--out", type=Path, help="output TSV")
    ap.add_argument("--tolerance", type=float, default=15.0,
                    help="seconds the last offset may exceed/undershoot duration_s")
    args = ap.parse_args()

    reg = pd.read_csv(REGISTRY, sep="\t")
    have = reg[reg["annotation_file"].notna()]

    frames, problems = [], []
    for _, row in have.iterrows():
        path = BIDS_ROOT / "stimuli" / "movies" / row["annotation_file"]
        if not path.exists():
            problems.append(f"{row['stimulus_id']}: MISSING FILE {path.name}")
            continue
        try:
            df = parse_file(path, row["stimulus_id"])
        except ValueError as exc:
            problems.append(f"{row['stimulus_id']}: {exc}")
            continue
        if df.empty:
            problems.append(f"{row['stimulus_id']}: parsed 0 segments")
            continue

        dur = float(row["duration_s"])
        last = df["offset"].max()
        if abs(last - dur) > args.tolerance:
            problems.append(
                f"{row['stimulus_id']}: last offset {last:.0f}s vs registry "
                f"duration {dur:.0f}s (delta {last - dur:+.0f}s)")
        if (df["duration"] < 0).any():
            problems.append(f"{row['stimulus_id']}: negative segment duration")
        frames.append(df)

    missing = reg[reg["annotation_file"].isna()]["stimulus_id"].tolist()
    all_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    print(f"movies with an annotation file : {len(have)} / {len(reg)}")
    if missing:
        print(f"movies with NO annotation      : {', '.join(missing)}")
    if not all_df.empty:
        for level, g in all_df.groupby("level"):
            print(f"  SEG-{level}: {len(g)} segments, "
                  f"median {g.groupby('stimulus_id').size().median():.0f}/movie, "
                  f"median length {g['duration'].median():.0f}s")
        segc = all_df[all_df["level"] == "C"]
        if not segc.empty:
            print("\nannotator covariate (SEG-C):")
            by = segc.groupby("annotator").agg(
                movies=("stimulus_id", "nunique"),
                segments=("stimulus_id", "size"),
                med_seg_s=("duration", "median"),
                med_chars=("description", lambda s: s.str.len().median()),
            )
            by["segs_per_movie"] = (by["segments"] / by["movies"]).round(1)
            print(by.to_string())

    if problems:
        print(f"\nPROBLEMS ({len(problems)}):")
        for p in problems:
            print(f"  {p}")
    else:
        print("\nall parsed files agree with registry durations "
              f"(tolerance {args.tolerance:.0f}s)")

    if args.out and not all_df.empty:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        all_df.to_csv(args.out, sep="\t", index=False)
        print(f"\nwrote {len(all_df)} rows -> {args.out}")

    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
