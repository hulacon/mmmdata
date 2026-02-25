"""Preprocessing utilities for MMMData behavioral data.

Cleaning, filtering, column normalization, and validation functions
that prepare loaded DataFrames for analysis.
"""

from __future__ import annotations

import sys
from typing import Optional

import numpy as np
import pandas as pd

from .constants import (
    SCANNER_RESP_OFFSET,
    SESSION_ORDER,
    TB2AFC_COLUMNS,
    ENCODING_COLUMNS,
    RETRIEVAL_COLUMNS,
    FIN2AFC_COLUMNS,
    FINTIMELINE_COLUMNS,
    COLUMN_RENAMES,
)


# ---------------------------------------------------------------------------
# Scanner response remapping
# ---------------------------------------------------------------------------

def remap_scanner_resp(
    df: pd.DataFrame,
    resp_col: str = "resp",
    offset: int = SCANNER_RESP_OFFSET,
    output_col: Optional[str] = None,
) -> pd.DataFrame:
    """Remap in-scanner button box responses to semantic rating scale.

    The scanner button box records 6, 7, 8 for a 3-point scale.
    This subtracts the offset to yield 1, 2, 3.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with scanner response data.
    resp_col : str, default ``"resp"``
        Column containing raw button numbers.
    offset : int, default 5
        Value to subtract (6->1, 7->2, 8->3).
    output_col : str, optional
        Name for remapped column. If None, overwrites ``resp_col``.

    Returns
    -------
    pd.DataFrame
        Copy with remapped responses.
    """
    result = df.copy()
    target = output_col or resp_col
    result[target] = pd.to_numeric(result[resp_col], errors="coerce") - offset
    return result


# ---------------------------------------------------------------------------
# 2AFC response decomposition
# ---------------------------------------------------------------------------

def decompose_2afc_resp(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns from 2AFC response values.

    Adds:

    - ``chose_position``: which image position was selected (1 or 2)
    - ``confidence``: ``"sure"`` or ``"maybe"``
    - ``is_correct``: boolean accuracy check

    Parameters
    ----------
    df : pd.DataFrame
        Must have ``resp`` and ``correct_resp`` columns.

    Returns
    -------
    pd.DataFrame
        Copy with added columns.
    """
    result = df.copy()
    resp = pd.to_numeric(result["resp"], errors="coerce")
    correct = pd.to_numeric(result["correct_resp"], errors="coerce")

    # Position chosen: resp 1-2 -> position 1, resp 3-4 -> position 2
    result["chose_position"] = np.where(resp.isin([1, 2]), 1, 2)

    # Confidence: resp 1 or 4 = sure, resp 2 or 3 = maybe
    result["confidence"] = np.where(resp.isin([1, 4]), "sure", "maybe")

    # Accuracy check
    result["is_correct"] = result["chose_position"] == correct

    return result


# ---------------------------------------------------------------------------
# Reaction time filtering
# ---------------------------------------------------------------------------

def filter_rt(
    df: pd.DataFrame,
    rt_col: str = "resp_RT",
    min_rt: float = 0.2,
    max_rt: Optional[float] = None,
    max_sd: Optional[float] = 3.0,
    by: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Filter reaction times by absolute bounds and/or SD-based outliers.

    Parameters
    ----------
    df : pd.DataFrame
    rt_col : str, default ``"resp_RT"``
        Column containing reaction times in seconds.
    min_rt : float, default 0.2
        Minimum plausible RT (fast guesses).
    max_rt : float, optional
        Maximum absolute RT bound.
    max_sd : float, optional, default 3.0
        Remove trials > ``max_sd`` SDs from the mean.
        Applied per group if ``by`` is specified.
    by : list of str, optional
        Grouping columns for per-group SD filtering
        (e.g., ``["subject", "session"]``).

    Returns
    -------
    pd.DataFrame
        Filtered copy. Number of dropped trials printed to stderr.
    """
    result = df.copy()
    rt = pd.to_numeric(result[rt_col], errors="coerce")
    n_before = len(result)

    # Absolute minimum
    mask = rt >= min_rt

    # Absolute maximum
    if max_rt is not None:
        mask &= rt <= max_rt

    result = result[mask].reset_index(drop=True)

    # SD-based filtering
    if max_sd is not None:
        rt = pd.to_numeric(result[rt_col], errors="coerce")
        if by:
            groups = result.groupby(by, observed=True)
            means = groups[rt_col].transform("mean")
            sds = groups[rt_col].transform("std")
        else:
            means = rt.mean()
            sds = rt.std()
        deviation = (rt - means).abs()
        within_bounds = deviation <= max_sd * sds
        # Keep rows where SD is NaN (single-trial groups)
        within_bounds = within_bounds | sds.isna() if isinstance(sds, pd.Series) else within_bounds
        result = result[within_bounds].reset_index(drop=True)

    n_dropped = n_before - len(result)
    if n_dropped > 0:
        print(f"filter_rt: dropped {n_dropped}/{n_before} trials", file=sys.stderr)

    return result


# ---------------------------------------------------------------------------
# Session ordering
# ---------------------------------------------------------------------------

def add_session_order(
    df: pd.DataFrame,
    session_col: str = "session",
) -> pd.DataFrame:
    """Add a ``session_order`` column: 0-indexed sequential session number.

    Maps ses-04 -> 0, ses-05 -> 1, ..., ses-18 -> 14.
    Useful for regression / learning curve x-axes.

    Parameters
    ----------
    df : pd.DataFrame
    session_col : str

    Returns
    -------
    pd.DataFrame
        Copy with ``session_order`` (int) column.
    """
    result = df.copy()
    result["session_order"] = result[session_col].map(SESSION_ORDER)
    return result


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_dataframe(
    df: pd.DataFrame,
    task: str,
) -> list[str]:
    """Validate that a DataFrame has expected columns and value ranges.

    Parameters
    ----------
    df : pd.DataFrame
    task : str
        One of ``"tb2afc"``, ``"encoding"``, ``"retrieval"``,
        ``"fin2afc"``, ``"fintimeline"``.

    Returns
    -------
    list of str
        Warning messages (empty if everything checks out).
    """
    expected_map = {
        "tb2afc": TB2AFC_COLUMNS,
        "encoding": ENCODING_COLUMNS,
        "retrieval": RETRIEVAL_COLUMNS,
        "fin2afc": FIN2AFC_COLUMNS,
        "fintimeline": FINTIMELINE_COLUMNS,
    }

    # Also check against normalized names
    normalized_expected_map = {}
    for key, cols in expected_map.items():
        normalized_expected_map[key] = [COLUMN_RENAMES.get(c, c) for c in cols]

    warnings: list[str] = []

    if task not in expected_map:
        warnings.append(f"Unknown task '{task}'. Expected one of: {list(expected_map)}")
        return warnings

    expected = set(normalized_expected_map[task])
    actual = set(df.columns) - {"source_file"}  # source_file added by io.py
    missing = expected - actual
    if missing:
        warnings.append(f"Missing columns: {sorted(missing)}")

    # Value range checks
    if "trial_accuracy" in df.columns:
        vals = df["trial_accuracy"].dropna()
        if not vals.empty and not vals.isin([0, 0.0, 1, 1.0]).all():
            warnings.append(
                f"trial_accuracy has unexpected values: {sorted(vals.unique())}"
            )

    if "enCon" in df.columns:
        vals = df["enCon"].dropna()
        if not vals.empty and not vals.isin([1, 2, 3]).all():
            warnings.append(
                f"enCon has unexpected values: {sorted(vals.unique())}"
            )

    if "reCon" in df.columns:
        vals = df["reCon"].dropna()
        if not vals.empty and not vals.isin([1, 2]).all():
            warnings.append(
                f"reCon has unexpected values: {sorted(vals.unique())}"
            )

    return warnings
