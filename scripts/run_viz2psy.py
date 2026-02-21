#!/usr/bin/env python3
"""Run viz2psy feature extraction on MMMData stimuli.

Wraps the viz2psy CLI to score shared1000 images, movie clips, and
movie cue images. Designed for use on Talapas (UO HPC) via SLURM.

Usage:
    python scripts/run_viz2psy.py images [--models resmem clip ...] [--dry-run]
    python scripts/run_viz2psy.py movies [--models resmem clip ...] [--frame-interval 0.5] [--dry-run]
    python scripts/run_viz2psy.py cues   [--models resmem clip ...] [--dry-run]

Requires viz2psy to be installed in the active Python environment,
or uses the viz2psy venv at VIZ2PSY_VENV (default: ~/.local/envs/viz2psy).
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# --- Paths ----------------------------------------------------------------

BIDS_ROOT = Path("/gpfs/projects/hulacon/shared/mmmdata")
STIMULI = BIDS_ROOT / "stimuli"

# viz2psy venv (override with VIZ2PSY_VENV env var)
VIZ2PSY_VENV = Path(
    os.environ.get("VIZ2PSY_VENV", Path.home() / ".local" / "envs" / "viz2psy")
)
VIZ2PSY_BIN = VIZ2PSY_VENV / "bin" / "viz2psy"
VIZ2PSY_VIZ_BIN = VIZ2PSY_VENV / "bin" / "viz2psy-viz"

IMAGES_DIR = STIMULI / "shared1000" / "images"
IMAGES_OUTPUT = STIMULI / "shared1000" / "viz2psy_scores.csv"

MOVIES_DIR = STIMULI / "movies" / "movie_files"
MOVIES_SCORES_DIR = STIMULI / "movies" / "viz2psy_scores"

CUES_DIR = STIMULI / "movies" / "movie_cues"
CUES_OUTPUT = STIMULI / "movies" / "viz2psy_cue_scores.csv"

MOVIE_SUFFIX = "_trimmed_normalized_filtered"

ALL_MODELS = [
    "resmem", "emonet", "clip", "caption", "dinov2",
    "gist", "places", "llstat", "saliency", "aesthetics", "yolo",
]


def movie_stem(filename: str) -> str:
    """Extract clean movie name from filename.

    'Adventure_Time_trimmed_normalized_filtered.mov' -> 'Adventure_Time'
    """
    stem = Path(filename).stem
    if stem.endswith(MOVIE_SUFFIX):
        stem = stem[: -len(MOVIE_SUFFIX)]
    return stem


def run_cmd(cmd: list[str], dry_run: bool = False) -> int:
    """Run a command, or print it if dry_run."""
    cmd_str = " ".join(str(c) for c in cmd)
    if dry_run:
        print(f"[dry-run] {cmd_str}")
        return 0

    print(f"[run] {cmd_str}")
    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        print(f"  -> exited with code {result.returncode}", file=sys.stderr)
    return result.returncode


def _viz2psy_cmd() -> str:
    """Return path to viz2psy binary (venv or system)."""
    if VIZ2PSY_BIN.exists():
        return str(VIZ2PSY_BIN)
    return "viz2psy"


def _viz2psy_viz_cmd() -> str:
    """Return path to viz2psy-viz binary (venv or system)."""
    if VIZ2PSY_VIZ_BIN.exists():
        return str(VIZ2PSY_VIZ_BIN)
    return "viz2psy-viz"


def generate_dashboard(
    scores_csv: Path,
    image_root: Path | None = None,
    video_path: Path | None = None,
    dry_run: bool = False,
) -> None:
    """Generate a viz2psy-viz HTML dashboard for a scores CSV."""
    dashboard_path = scores_csv.parent / (scores_csv.stem + "_dashboard.html")

    cmd = [
        _viz2psy_viz_cmd(), "dashboard",
        str(scores_csv),
        "-o", str(dashboard_path),
    ]
    if image_root is not None:
        cmd.extend(["--image-root", str(image_root)])
    if video_path is not None:
        cmd.extend(["--video-path", str(video_path)])

    run_cmd(cmd, dry_run=dry_run)


def run_images(models: list[str], dry_run: bool = False) -> None:
    """Score shared1000 images with viz2psy."""
    images = sorted(IMAGES_DIR.glob("*.png"))
    if not images:
        print(f"No PNG files found in {IMAGES_DIR}", file=sys.stderr)
        sys.exit(1)

    print(f"Scoring {len(images)} images -> {IMAGES_OUTPUT}")

    cmd = [
        _viz2psy_cmd(), *models,
        *[str(p) for p in images],
        "-o", str(IMAGES_OUTPUT),
    ]
    run_cmd(cmd, dry_run=dry_run)

    # Generate HTML dashboard (includes embedded image browser)
    generate_dashboard(IMAGES_OUTPUT, image_root=IMAGES_DIR, dry_run=dry_run)


def run_movies(
    models: list[str],
    frame_interval: float = 0.5,
    dry_run: bool = False,
) -> None:
    """Score movie files with viz2psy (one job per movie)."""
    movies = sorted(MOVIES_DIR.glob("*.mov"))
    if not movies:
        print(f"No .mov files found in {MOVIES_DIR}", file=sys.stderr)
        sys.exit(1)

    if not dry_run:
        MOVIES_SCORES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Scoring {len(movies)} movies -> {MOVIES_SCORES_DIR}/")

    for mov in movies:
        stem = movie_stem(mov.name)
        output = MOVIES_SCORES_DIR / f"{stem}_scores.csv"

        cmd = [
            _viz2psy_cmd(), *models,
            str(mov),
            "-o", str(output),
            "--frame-interval", str(frame_interval),
        ]
        run_cmd(cmd, dry_run=dry_run)

        # Generate per-movie dashboard
        generate_dashboard(output, video_path=mov, dry_run=dry_run)

    print("All movies scored.")


def run_cues(models: list[str], dry_run: bool = False) -> None:
    """Score movie cue images with viz2psy."""
    cues = sorted(CUES_DIR.glob("*.jpg"))
    if not cues:
        print(f"No JPG files found in {CUES_DIR}", file=sys.stderr)
        sys.exit(1)

    print(f"Scoring {len(cues)} cue images -> {CUES_OUTPUT}")

    cmd = [
        _viz2psy_cmd(), *models,
        *[str(p) for p in cues],
        "-o", str(CUES_OUTPUT),
    ]
    run_cmd(cmd, dry_run=dry_run)

    # Generate HTML dashboard (includes embedded image browser)
    generate_dashboard(CUES_OUTPUT, image_root=CUES_DIR, dry_run=dry_run)


def main():
    parser = argparse.ArgumentParser(
        description="Run viz2psy feature extraction on MMMData stimuli",
    )
    parser.add_argument(
        "target",
        choices=["images", "movies", "cues", "all"],
        help="Which stimuli to score",
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        help=f"Models to run (default: all). Choose from: {', '.join(ALL_MODELS)}",
    )
    parser.add_argument(
        "--frame-interval", type=float, default=0.5,
        help="Frame extraction interval in seconds for movies (default: 0.5)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print commands without executing",
    )

    args = parser.parse_args()

    # Default to --all flag style (pass no model names, let viz2psy use --all)
    if args.models is None:
        models = ["--all"]
    else:
        models = args.models

    if args.target in ("images", "all"):
        run_images(models, dry_run=args.dry_run)

    if args.target in ("movies", "all"):
        run_movies(models, frame_interval=args.frame_interval, dry_run=args.dry_run)

    if args.target in ("cues", "all"):
        run_cues(models, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
