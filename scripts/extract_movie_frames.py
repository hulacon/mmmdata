"""Extract per-film frame images (screengrabs) for stimulus-feature dashboards.

The *2psy dashboards can show a thumbnail per time point (``--frames-dir``),
but scoring extracted frames to a temp dir and discarded them — no sidecar in
the features tree records a ``saved_frames_dir``. This driver backfills them.

It walks ``<features-root>/movies/<slug>/``, reads any video sidecar
(``*.meta.json`` with ``input.type == "video"``) to learn which film the
directory describes and the frame interval the models saw, and writes
``frames/frame_<t>.jpg`` beside the features via
``viz2psy.video.extract_frames`` — the same extraction path the models used,
so thumbnails line up with feature rows exactly.

The features tree is the film↔slug mapping; nothing is re-derived from
titles. Idempotent: a directory whose ``frames/`` already holds the expected
count is skipped (``--force`` re-extracts).

Usage (one film, e.g. from a SLURM array task):
    python extract_movie_frames.py --features-root <dir> --index 3
    python extract_movie_frames.py --features-root <dir> --slug adventure-time
    python extract_movie_frames.py --features-root <dir> --list
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def film_dirs(features_root: Path) -> list[Path]:
    """Sorted per-film feature directories under <features-root>/movies/."""
    movies = features_root / "movies"
    if not movies.is_dir():
        sys.exit(f"error: {movies} does not exist — pass the "
                 "stimuli_features root (see config output_dir)")
    return sorted(p for p in movies.iterdir() if p.is_dir())


def video_sidecar(film_dir: Path) -> dict | None:
    """The input block of any video sidecar in the directory."""
    for meta_path in sorted(film_dir.glob("*.meta.json")):
        try:
            info = json.loads(meta_path.read_text()).get("input", {})
        except (OSError, json.JSONDecodeError):
            continue
        if info.get("type") == "video":
            return info
    return None


def extract_one(film_dir: Path, interval: float | None, force: bool) -> int:
    info = video_sidecar(film_dir)
    if info is None:
        print(f"skipping {film_dir.name}: no video sidecar")
        return 0

    video_path = Path(info["path"])
    if not video_path.exists():
        sys.exit(f"error: {video_path} (from {film_dir.name} sidecar) "
                 "not found — run where the stimuli tree is mounted")

    interval = interval or float(info.get("frame_interval_sec", 0.5))
    expected = int(info.get("n_frames", 0))
    out_dir = film_dir / "frames"

    existing = len(list(out_dir.glob("frame_*.jpg"))) if out_dir.exists() else 0
    if existing and not force:
        if expected and existing >= expected:
            print(f"{film_dir.name}: frames/ already complete "
                  f"({existing} frames), skipping")
            return existing
        sys.exit(f"error: {out_dir} holds {existing} frames but the sidecar "
                 f"expects {expected}; re-run with --force to re-extract")

    from viz2psy.video import extract_frames

    print(f"{film_dir.name}: extracting every {interval}s from "
          f"{video_path.name} -> {out_dir}")
    frames = extract_frames(video_path, frame_interval=interval,
                            save_dir=out_dir, quiet=True)
    print(f"{film_dir.name}: wrote {len(frames)} frames")
    if expected and abs(len(frames) - expected) > 1:
        sys.exit(f"error: wrote {len(frames)} frames but the sidecar "
                 f"recorded n_frames={expected} — interval mismatch?")
    return len(frames)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--features-root", type=Path, required=True,
                    help="stimuli_features root (contains movies/<slug>/)")
    sel = ap.add_mutually_exclusive_group(required=True)
    sel.add_argument("--index", type=int,
                     help="1-based index into the sorted film dirs "
                          "(SLURM_ARRAY_TASK_ID)")
    sel.add_argument("--slug", help="one film directory by name")
    sel.add_argument("--all", action="store_true", help="every film, serially")
    sel.add_argument("--list", action="store_true",
                     help="print index -> slug mapping and exit")
    ap.add_argument("--interval", type=float, default=None,
                    help="override the sidecar's frame_interval_sec")
    ap.add_argument("--force", action="store_true",
                    help="re-extract over an existing frames/ dir")
    args = ap.parse_args()

    dirs = film_dirs(args.features_root)
    if args.list:
        for i, d in enumerate(dirs, start=1):
            print(f"{i:3d}  {d.name}")
        return
    if args.slug:
        matches = [d for d in dirs if d.name == args.slug]
        if not matches:
            sys.exit(f"error: no film dir named {args.slug!r} under "
                     f"{args.features_root}/movies (see --list)")
        dirs = matches
    elif args.index is not None:
        if not 1 <= args.index <= len(dirs):
            sys.exit(f"error: --index {args.index} out of range 1..{len(dirs)}")
        dirs = [dirs[args.index - 1]]

    for d in dirs:
        extract_one(d, args.interval, args.force)


if __name__ == "__main__":
    main()
