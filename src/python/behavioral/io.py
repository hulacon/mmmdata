"""File discovery and loading for MMMData behavioral data.

All filesystem interaction is isolated here. Analysis modules receive
DataFrames and never touch the filesystem directly.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from .constants import (
    DEFAULT_BIDS_ROOT,
    SUBJECT_IDS,
    TB_SESSIONS,
    TB_ENCODING_SESSIONS,
    FINAL_SESSION,
    TB2AFC_TEMPLATE,
    FIN2AFC_TEMPLATE,
    FINTIMELINE_TEMPLATE,
    TBENCODING_GLOB,
    TBRETRIEVAL_GLOB,
    COLUMN_RENAMES,
)


# ---------------------------------------------------------------------------
# BIDS root resolution
# ---------------------------------------------------------------------------

def _resolve_bids_root(bids_root: Optional[Path] = None) -> Path:
    """Return BIDS root from argument, config, or fallback constant."""
    if bids_root is not None:
        return Path(bids_root)
    try:
        # Try loading from mmmdata's config
        code_root = Path(__file__).resolve().parents[3]  # src/python/behavioral -> mmmdata/
        if str(code_root) not in sys.path:
            sys.path.insert(0, str(code_root / "src" / "python"))
        from core.config import load_config
        config = load_config()
        return Path(config["paths"]["bids_project_dir"])
    except Exception:
        return DEFAULT_BIDS_ROOT


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------

_BIDS_FILENAME_RE = re.compile(
    r"sub-(?P<subject>\d+)_ses-(?P<session>\d+)"
)


def _parse_sub_ses(path: Path) -> dict[str, str]:
    """Extract subject and session IDs from a BIDS filename."""
    m = _BIDS_FILENAME_RE.search(path.name)
    if m:
        return {"subject": m.group("subject"), "session": m.group("session")}
    return {}


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def find_tb2afc_files(
    bids_root: Optional[Path] = None,
    subjects: Optional[Sequence[str]] = None,
    sessions: Optional[Sequence[str]] = None,
) -> list[Path]:
    """Discover TB2AFC behavioral files on disk.

    Parameters
    ----------
    bids_root : Path, optional
        BIDS dataset root.
    subjects : sequence of str, optional
        Subject IDs without ``sub-`` prefix (e.g., ``["03", "04"]``).
        Defaults to all subjects.
    sessions : sequence of str, optional
        Session IDs without ``ses-`` prefix. Defaults to ses-04..ses-18.

    Returns
    -------
    list of Path
        Sorted list of existing ``.tsv`` file paths.
    """
    root = _resolve_bids_root(bids_root)
    subs = subjects or SUBJECT_IDS
    sess = sessions or TB_SESSIONS
    paths = []
    for sub in subs:
        for ses in sess:
            p = root / TB2AFC_TEMPLATE.format(sub=sub, ses=ses)
            if p.exists():
                paths.append(p)
    return sorted(paths)


def find_encoding_files(
    bids_root: Optional[Path] = None,
    subjects: Optional[Sequence[str]] = None,
    sessions: Optional[Sequence[str]] = None,
) -> list[Path]:
    """Discover TBencoding event files. Globs across run numbers.

    Parameters
    ----------
    bids_root : Path, optional
    subjects : sequence of str, optional
    sessions : sequence of str, optional
        Defaults to ses-04..ses-17 (encoding absent in ses-18).

    Returns
    -------
    list of Path
    """
    root = _resolve_bids_root(bids_root)
    subs = subjects or SUBJECT_IDS
    sess = sessions or TB_ENCODING_SESSIONS
    paths = []
    for sub in subs:
        for ses in sess:
            pattern = TBENCODING_GLOB.format(sub=sub, ses=ses)
            paths.extend(root.glob(pattern))
    return sorted(paths)


def find_retrieval_files(
    bids_root: Optional[Path] = None,
    subjects: Optional[Sequence[str]] = None,
    sessions: Optional[Sequence[str]] = None,
) -> list[Path]:
    """Discover TBretrieval event files. Globs across run numbers.

    Parameters
    ----------
    bids_root : Path, optional
    subjects : sequence of str, optional
    sessions : sequence of str, optional
        Defaults to ses-04..ses-18.

    Returns
    -------
    list of Path
    """
    root = _resolve_bids_root(bids_root)
    subs = subjects or SUBJECT_IDS
    sess = sessions or TB_SESSIONS
    paths = []
    for sub in subs:
        for ses in sess:
            pattern = TBRETRIEVAL_GLOB.format(sub=sub, ses=ses)
            paths.extend(root.glob(pattern))
    return sorted(paths)


def find_fin2afc_files(
    bids_root: Optional[Path] = None,
    subjects: Optional[Sequence[str]] = None,
) -> list[Path]:
    """Discover FIN2AFC files (ses-30 only).

    Parameters
    ----------
    bids_root : Path, optional
    subjects : sequence of str, optional

    Returns
    -------
    list of Path
    """
    root = _resolve_bids_root(bids_root)
    subs = subjects or SUBJECT_IDS
    paths = []
    for sub in subs:
        p = root / FIN2AFC_TEMPLATE.format(sub=sub)
        if p.exists():
            paths.append(p)
    return sorted(paths)


def find_fintimeline_files(
    bids_root: Optional[Path] = None,
    subjects: Optional[Sequence[str]] = None,
) -> list[Path]:
    """Discover FINtimeline files (ses-30 only).

    Parameters
    ----------
    bids_root : Path, optional
    subjects : sequence of str, optional

    Returns
    -------
    list of Path
    """
    root = _resolve_bids_root(bids_root)
    subs = subjects or SUBJECT_IDS
    paths = []
    for sub in subs:
        p = root / FINTIMELINE_TEMPLATE.format(sub=sub)
        if p.exists():
            paths.append(p)
    return sorted(paths)


# ---------------------------------------------------------------------------
# Single-file loading
# ---------------------------------------------------------------------------

def load_tsv(path: Path) -> pd.DataFrame:
    """Load a single BIDS ``.tsv`` file into a DataFrame.

    Handles ``n/a`` as NaN. Always sets ``subject`` and ``session``
    from the BIDS filename (authoritative source), since column values
    may use different numbering schemes (e.g., ``session_num`` in TB2AFC
    is the behavioral session 1-15, not the BIDS session 04-18).

    Parameters
    ----------
    path : Path
        Path to the ``.tsv`` file.

    Returns
    -------
    pd.DataFrame
    """
    df = pd.read_csv(path, sep="\t", na_values=["n/a"])
    meta = _parse_sub_ses(path)
    # Always use filename-based subject/session (BIDS-authoritative)
    if "subject" in meta:
        df["subject"] = meta["subject"]
    if "session" in meta:
        df["session"] = meta["session"]
    return df


# ---------------------------------------------------------------------------
# Multi-file loading (concatenation)
# ---------------------------------------------------------------------------

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column types and drop redundant raw columns.

    Since ``load_tsv`` always injects ``subject`` and ``session``
    from the BIDS filename, the raw ``subject_id`` and ``session_num``
    columns (if present) are redundant. We drop them to avoid
    confusion but do NOT rename them (the filename-based values are
    already authoritative).
    """
    # Drop redundant raw columns that conflict with filename-based values
    for raw_col in ("subject_id", "session_num"):
        if raw_col in df.columns and COLUMN_RENAMES.get(raw_col) in df.columns:
            df = df.drop(columns=[raw_col])

    # Rename run_num -> run if present
    if "run_num" in df.columns:
        df = df.rename(columns={"run_num": "run"})

    # Ensure subject/session are zero-padded strings
    for col in ("subject", "session"):
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: f"{int(x):02d}" if pd.notna(x) else x
            )
    if "run" in df.columns:
        df["run"] = df["run"].apply(
            lambda x: f"{int(x):02d}" if pd.notna(x) else x
        )
    return df


def _load_and_concat(
    paths: list[Path],
    drop_rest: bool = False,
    trial_type_col: str = "trial_type",
    rest_values: tuple[str, ...] = ("rest",),
) -> pd.DataFrame:
    """Load multiple TSV files, concatenate, normalize, and optionally filter."""
    if not paths:
        return pd.DataFrame()
    dfs = []
    for p in paths:
        df = load_tsv(p)
        df["source_file"] = str(p)
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    combined = _normalize_columns(combined)
    if drop_rest and trial_type_col in combined.columns:
        combined = combined[~combined[trial_type_col].isin(rest_values)].reset_index(drop=True)
    return combined


def load_tb2afc(
    bids_root: Optional[Path] = None,
    subjects: Optional[Sequence[str]] = None,
    sessions: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Load and concatenate all TB2AFC files into a single DataFrame.

    Normalizes column names: ``subject_id`` -> ``subject``,
    ``session_num`` -> ``session``.

    Parameters
    ----------
    bids_root : Path, optional
    subjects : sequence of str, optional
    sessions : sequence of str, optional

    Returns
    -------
    pd.DataFrame
        Concatenated DataFrame with all TB2AFC trials.
    """
    paths = find_tb2afc_files(bids_root, subjects, sessions)
    return _load_and_concat(paths)


def load_encoding(
    bids_root: Optional[Path] = None,
    subjects: Optional[Sequence[str]] = None,
    sessions: Optional[Sequence[str]] = None,
    drop_rest: bool = True,
) -> pd.DataFrame:
    """Load and concatenate all TBencoding event files.

    Parameters
    ----------
    bids_root : Path, optional
    subjects : sequence of str, optional
    sessions : sequence of str, optional
    drop_rest : bool, default True
        If True, removes rows where ``trial_type == 'rest'``.

    Returns
    -------
    pd.DataFrame
    """
    paths = find_encoding_files(bids_root, subjects, sessions)
    return _load_and_concat(paths, drop_rest=drop_rest)


def load_retrieval(
    bids_root: Optional[Path] = None,
    subjects: Optional[Sequence[str]] = None,
    sessions: Optional[Sequence[str]] = None,
    drop_rest: bool = True,
) -> pd.DataFrame:
    """Load and concatenate all TBretrieval event files.

    Parameters
    ----------
    bids_root : Path, optional
    subjects : sequence of str, optional
    sessions : sequence of str, optional
    drop_rest : bool, default True
        If True, removes rows where ``trial_type == 'rest'``.

    Returns
    -------
    pd.DataFrame
    """
    paths = find_retrieval_files(bids_root, subjects, sessions)
    return _load_and_concat(paths, drop_rest=drop_rest)


def load_fin2afc(
    bids_root: Optional[Path] = None,
    subjects: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Load and concatenate all FIN2AFC files.

    Parameters
    ----------
    bids_root : Path, optional
    subjects : sequence of str, optional

    Returns
    -------
    pd.DataFrame
    """
    paths = find_fin2afc_files(bids_root, subjects)
    return _load_and_concat(paths)


def load_fintimeline(
    bids_root: Optional[Path] = None,
    subjects: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Load and concatenate all FINtimeline files.

    Parameters
    ----------
    bids_root : Path, optional
    subjects : sequence of str, optional

    Returns
    -------
    pd.DataFrame
    """
    paths = find_fintimeline_files(bids_root, subjects)
    return _load_and_concat(paths)
