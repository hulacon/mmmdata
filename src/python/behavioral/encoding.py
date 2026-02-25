"""Encoding analysis and subsequent memory effects.

Functions for analyzing encoding quality ratings and their relationship
to later recognition accuracy (subsequent memory effects).
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd


def encoding_rating_distribution(
    df: pd.DataFrame,
    rating_col: str = "resp",
    group_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Distribution of encoding association quality ratings (1-3).

    Assumes scanner resp has already been remapped via
    ``preprocessing.remap_scanner_resp()``.

    Parameters
    ----------
    df : pd.DataFrame
        Encoding trials with remapped resp column.
    rating_col : str, default ``"resp"``
    group_cols : sequence of str, optional

    Returns
    -------
    pd.DataFrame
        Columns: ``*group_cols``, ``rating``, ``count``, ``proportion``.
    """
    work = df.copy()
    work["_rating"] = pd.to_numeric(work[rating_col], errors="coerce")
    work = work.dropna(subset=["_rating"])

    gcols = list(group_cols or [])
    count_cols = gcols + ["_rating"]

    counts = work.groupby(count_cols, observed=True).size().reset_index(name="count")
    counts = counts.rename(columns={"_rating": "rating"})

    # Compute proportion within each group
    if gcols:
        totals = counts.groupby(gcols, observed=True)["count"].transform("sum")
    else:
        totals = counts["count"].sum()
    counts["proportion"] = counts["count"] / totals

    return counts


def subsequent_memory_effect(
    encoding_df: pd.DataFrame,
    recognition_df: pd.DataFrame,
    merge_on: Sequence[str] = ("subject", "pairId"),
    encoding_rating_col: str = "resp",
    accuracy_col: str = "trial_accuracy",
    group_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Compute subsequent memory effect: does encoding quality predict
    later recognition accuracy?

    Merges encoding ratings with recognition accuracy by ``pairId``,
    then computes recognition accuracy as a function of encoding rating.

    Parameters
    ----------
    encoding_df : pd.DataFrame
        Encoding trials (resp should be remapped to 1-3 via
        ``preprocessing.remap_scanner_resp()``).
    recognition_df : pd.DataFrame
        2AFC or retrieval data with ``trial_accuracy``.
    merge_on : sequence of str, default ``("subject", "pairId")``
        Columns to merge on.
    encoding_rating_col : str, default ``"resp"``
    accuracy_col : str, default ``"trial_accuracy"``
    group_cols : sequence of str, optional
        Additional grouping (e.g., ``["subject"]``, ``["subject", "enCon"]``).

    Returns
    -------
    pd.DataFrame
        Columns: ``*group_cols``, ``encoding_rating``, ``n_trials``,
        ``accuracy``, ``se``.
    """
    merge_keys = list(merge_on)

    # Prepare encoding side: keep only rating + merge keys
    enc = encoding_df[merge_keys + [encoding_rating_col]].copy()
    enc = enc.rename(columns={encoding_rating_col: "encoding_rating"})
    enc["encoding_rating"] = pd.to_numeric(enc["encoding_rating"], errors="coerce")
    enc = enc.dropna(subset=["encoding_rating"])

    # For items encoded multiple times (repeats/triplets), take the mean rating
    enc = enc.groupby(merge_keys, observed=True)["encoding_rating"].mean().reset_index()
    # Round to nearest integer for grouping
    enc["encoding_rating"] = enc["encoding_rating"].round().astype(int)

    # Prepare recognition side: keep accuracy + merge keys
    rec = recognition_df[merge_keys + [accuracy_col]].copy()
    rec[accuracy_col] = pd.to_numeric(rec[accuracy_col], errors="coerce")

    # Merge
    merged = enc.merge(rec, on=merge_keys, how="inner")

    if merged.empty:
        return pd.DataFrame(
            columns=list(group_cols or []) + [
                "encoding_rating", "n_trials", "accuracy", "se",
            ]
        )

    # Group by encoding rating (and optional extra groups)
    gcols = list(group_cols or []) + ["encoding_rating"]
    grouped = merged.groupby(gcols, observed=True)

    result = grouped[accuracy_col].agg(
        n_trials="count",
        accuracy="mean",
        se="sem",
    ).reset_index()

    return result


def retrieval_vividness_by_condition(
    df: pd.DataFrame,
    rating_col: str = "resp",
    group_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Analyze retrieval vividness ratings (1-3) by condition.

    Assumes scanner resp has already been remapped via
    ``preprocessing.remap_scanner_resp()``.

    Parameters
    ----------
    df : pd.DataFrame
        Retrieval trials with remapped resp column.
    rating_col : str, default ``"resp"``
    group_cols : sequence of str, optional
        (e.g., ``["subject", "enCon"]``, ``["subject", "enCon", "reCon"]``)

    Returns
    -------
    pd.DataFrame
        Columns: ``*group_cols``, ``mean_vividness``, ``se``, ``n``.
    """
    work = df.copy()
    work["_viv"] = pd.to_numeric(work[rating_col], errors="coerce")

    if group_cols:
        grouped = work.groupby(list(group_cols), observed=True)
    else:
        grouped = work.groupby(lambda _: "all")

    result = grouped["_viv"].agg(
        mean_vividness="mean",
        se="sem",
        n="count",
    ).reset_index()

    if not group_cols:
        result = result.drop(columns=["index"])

    return result
