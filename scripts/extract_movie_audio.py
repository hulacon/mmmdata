"""Extract per-film audio tracks for stimulus-feature viewers.

Companion to ``extract_movie_frames.py``: the frames/ screengrabs give a
viewer its visual scrub, but audio playback needs an audio file the browser
can seek, and the source ``.mov`` files are far too heavy to ship. This
driver walks ``<features-root>/movies/<slug>/``, resolves each film's source
video from its sidecar (same path the models scored), and writes an AAC
``audio.m4a`` beside ``frames/`` via ffmpeg.

The features tree is the film↔slug mapping; nothing is re-derived from
titles. Idempotent: a directory that already holds a non-empty ``audio.m4a``
is skipped (``--force`` re-encodes). Output is written to a temp name and
renamed, so an interrupted run never leaves a truncated file behind.

Requires an ``ffmpeg`` (and, for the duration check, ``ffprobe``) on PATH —
on Talapas, ``module load ffmpeg`` — or pointed at by ``$FFMPEG_BIN``.

Usage (one film, e.g. from a SLURM array task):
    python extract_movie_audio.py --features-root <dir> --index 3
    python extract_movie_audio.py --features-root <dir> --all
    python extract_movie_audio.py --features-root <dir> --list
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from extract_movie_frames import film_dirs, video_sidecar

AUDIO_NAME = "audio.m4a"


def ffmpeg_bin(probe: bool = False) -> str:
    name = "ffprobe" if probe else "ffmpeg"
    override = os.environ.get("FFMPEG_BIN")
    if override and not probe:
        return override
    found = shutil.which(name)
    if not found:
        sys.exit(f"error: no {name} on PATH — module load ffmpeg, "
                 "or set $FFMPEG_BIN")
    return found


def probe_duration(path: Path) -> float | None:
    """Container duration in seconds, or None if ffprobe is unusable."""
    try:
        out = subprocess.run(
            [ffmpeg_bin(probe=True), "-v", "error", "-show_entries",
             "format=duration", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, check=True).stdout.strip()
        return float(out)
    except (subprocess.CalledProcessError, ValueError):
        return None


def extract_one(film_dir: Path, bitrate: str, force: bool) -> None:
    info = video_sidecar(film_dir)
    if info is None:
        print(f"skipping {film_dir.name}: no video sidecar")
        return

    video_path = Path(info["path"])
    if not video_path.exists():
        sys.exit(f"error: {video_path} (from {film_dir.name} sidecar) "
                 "not found — run where the stimuli tree is mounted")

    out_path = film_dir / AUDIO_NAME
    if out_path.exists() and out_path.stat().st_size > 0 and not force:
        print(f"{film_dir.name}: {AUDIO_NAME} already exists, skipping")
        return

    tmp_path = out_path.with_name(f".{AUDIO_NAME}.tmp.m4a")
    cmd = [ffmpeg_bin(), "-y", "-v", "error", "-i", str(video_path),
           "-vn", "-c:a", "aac", "-b:a", bitrate,
           "-movflags", "+faststart", str(tmp_path)]
    print(f"{film_dir.name}: {video_path.name} -> {out_path.name}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        tmp_path.unlink(missing_ok=True)
        sys.exit(f"error: ffmpeg failed on {video_path.name} "
                 f"(exit {exc.returncode})")

    # The sidecar's frame grid implies the film duration; a big mismatch
    # means the sidecar points at the wrong file, not an encode hiccup.
    expected = int(info.get("n_frames", 0)) * float(
        info.get("frame_interval_sec", 0.5))
    got = probe_duration(tmp_path)
    if expected and got is not None and abs(got - expected) > 2.0:
        tmp_path.unlink(missing_ok=True)
        sys.exit(f"error: {film_dir.name} audio runs {got:.1f}s but the "
                 f"sidecar frame grid implies ~{expected:.1f}s")

    tmp_path.rename(out_path)
    size_mb = out_path.stat().st_size / 1e6
    print(f"{film_dir.name}: wrote {AUDIO_NAME} "
          f"({size_mb:.1f} MB, {got or 0:.1f}s)")


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
    ap.add_argument("--bitrate", default="192k",
                    help="AAC bitrate (default %(default)s)")
    ap.add_argument("--force", action="store_true",
                    help="re-encode over an existing audio file")
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
        extract_one(d, args.bitrate, args.force)


if __name__ == "__main__":
    main()
