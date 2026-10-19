"""Find audio and video files in a dataset copy.

Copies shared outside the lab must contain no recordings of participants'
voices. Those recordings belong in the source-data tree, which is never
copied, so a hit here means something was misfiled or copied by mistake.

A file counts as audio/video when its extension is one (also under a
compression suffix, e.g. ``.wav.gz``), or when a file whose extension is not
a known dataset format starts with an audio/video signature (a recording
renamed to ``.bin`` or given no extension). Known dataset formats (NIfTI,
TSV, JSON, images, ...) are not opened, so the scan stays cheap on a large
tree.

The top-level ``stimuli/`` directory is excluded by default: its audio is
synthesized stimulus speech and its films are stimuli, not participant
recordings. ``.git`` and ``.datalad`` are skipped.
"""

from __future__ import annotations

import os
from pathlib import Path

from .sweep import SKIP_DIRS, Finding

AUDIO_VIDEO_EXTENSIONS = (
    ".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".oga", ".opus", ".wma",
    ".aif", ".aiff", ".amr", ".caf", ".mp4", ".m4v", ".mov", ".avi", ".mkv",
    ".webm", ".wmv", ".3gp", ".mpg", ".mpeg",
)
COMPRESSION_SUFFIXES = (".gz", ".bz2", ".xz", ".zip", ".zst")

# Formats the dataset is expected to hold; these are never opened.
KNOWN_DATA_EXTENSIONS = (
    ".nii", ".nii.gz", ".json", ".tsv", ".tsv.gz", ".csv", ".txt", ".md",
    ".html", ".rst", ".bval", ".bvec", ".png", ".jpg", ".jpeg", ".svg",
    ".gif", ".gii", ".mgz", ".mgh", ".h5", ".x5", ".mat", ".npy", ".npz",
    ".parquet", ".duckdb", ".edf", ".asc", ".log", ".yml", ".yaml", ".toml",
    ".cfg", ".pdf", ".label", ".annot", ".pial", ".white", ".sphere",
    ".dat", ".lta", ".xfm", ".ctab", ".stats",
)
DEFAULT_ALLOWED_DIRS = ("stimuli",)

_SIGNATURES = (
    (0, b"RIFF", 8, b"WAVE", "wav"),
    (0, b"RIFF", 8, b"AVI ", "avi"),
    (0, b"FORM", 8, b"AIFF", "aiff"),
    (0, b"ID3", None, None, "mp3"),
    (0, b"fLaC", None, None, "flac"),
    (0, b"OggS", None, None, "ogg"),
    (0, b"#!AMR", None, None, "amr"),
    (0, b"\x1aE\xdf\xa3", None, None, "matroska/webm"),
    (0, b"\x30\x26\xb2\x75", None, None, "asf/wma"),
    (4, b"ftyp", None, None, "mp4/m4a/mov"),
)


def _sniff(path: Path) -> str | None:
    """The audio/video format a file's first bytes announce, or None."""
    try:
        with open(path, "rb") as fh:
            head = fh.read(16)
    except OSError:
        return None
    for off, magic, off2, magic2, kind in _SIGNATURES:
        if head[off:off + len(magic)] == magic and (
                off2 is None or head[off2:off2 + len(magic2)] == magic2):
            return kind
    # MPEG audio frame sync (an mp3 without an ID3 tag)
    if len(head) >= 2 and head[0] == 0xFF and head[1] & 0xE0 == 0xE0 and head[1] & 0x06:
        return "mpeg-audio"
    return None


def _strip_compression(name: str) -> str:
    low = name.lower()
    for suf in COMPRESSION_SUFFIXES:
        if low.endswith(suf):
            return low[: -len(suf)]
    return low


def classify(path: Path) -> str | None:
    """Why ``path`` is audio/video (``"extension .wav"``, ``"content wav"``),
    or None if it is not."""
    low = path.name.lower()
    base = _strip_compression(low)
    for ext in AUDIO_VIDEO_EXTENSIONS:
        if base.endswith(ext):
            return f"extension {ext}"
    if low.endswith(KNOWN_DATA_EXTENSIONS):
        return None
    kind = _sniff(path)
    return f"content {kind}" if kind else None


def find_media(root: str | Path,
               allowed_dirs: tuple[str, ...] = DEFAULT_ALLOWED_DIRS) -> list[Finding]:
    """Every audio/video file under ``root`` outside the allowed top-level
    directories, as ``media`` findings (root-relative paths)."""
    root = Path(root)
    findings: list[Finding] = []
    for dirpath, dirnames, filenames in os.walk(root):
        here = Path(dirpath)
        if here == root:
            dirnames[:] = [d for d in dirnames if d not in allowed_dirs]
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        for name in sorted(filenames):
            p = here / name
            why = classify(p)
            if why:
                findings.append(Finding(p.relative_to(root).as_posix(), 0, "media", why))
    return findings
