"""Reaction time distribution analysis.

All functions accept DataFrames and optional ``group_cols``
for flexible condition-wise breakdowns.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd


def rt_summary(
    df: pd.DataFrame,
    rt_col: str = "resp_RT",
    group_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Compute RT distribution statistics per group.

    Parameters
    ----------
    df : pd.DataFrame
    rt_col : str, default ``"resp_RT"``
    group_cols : sequence of str, optional

    Returns
    -------
    pd.DataFrame
        Columns: ``*group_cols``, ``mean_rt``, ``median_rt``, ``sd_rt``,
        ``min_rt``, ``max_rt``, ``skewness``, ``n_trials``.
    """
    work = df.copy()
    work["_rt"] = pd.to_numeric(work[rt_col], errors="coerce")

    if group_cols:
        grouped = work.groupby(list(group_cols), observed=True)
    else:
        grouped = work.groupby(lambda _: "all")

    result = grouped["_rt"].agg(
        mean_rt="mean",
        median_rt="median",
        sd_rt="std",
        min_rt="min",
        max_rt="max",
        skewness="skew",
        n_trials="count",
    ).reset_index()

    if not group_cols:
        result = result.drop(columns=["index"])

    return result


def rt_by_accuracy(
    df: pd.DataFrame,
    rt_col: str = "resp_RT",
    accuracy_col: str = "trial_accuracy",
    group_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Compare RT distributions for correct vs. incorrect trials.

    Parameters
    ----------
    df : pd.DataFrame
    rt_col : str, default ``"resp_RT"``
    accuracy_col : str, default ``"trial_accuracy"``
    group_cols : sequence of str, optional

    Returns
    -------
    pd.DataFrame
        Columns: ``*group_cols``, ``accurate``, ``mean_rt``,
        ``median_rt``, ``sd_rt``, ``n_trials``.
    """
    work = df.copy()
    work["_rt"] = pd.to_numeric(work[rt_col], errors="coerce")
    work["accurate"] = pd.to_numeric(work[accuracy_col], errors="coerce").astype(bool)

    gcols = list(group_cols or []) + ["accurate"]
    grouped = work.groupby(gcols, observed=True)

    result = grouped["_rt"].agg(
        mean_rt="mean",
        median_rt="median",
        sd_rt="std",
        n_trials="count",
    ).reset_index()

    return result


def rt_sequential(
    df: pd.DataFrame,
    rt_col: str = "resp_RT",
    trial_col: str = "trial_id",
    group_cols: Optional[Sequence[str]] = None,
    window: int = 5,
) -> pd.DataFrame:
    """Compute trial-by-trial RT with rolling average for within-session trends.

    Parameters
    ----------
    df : pd.DataFrame
    rt_col : str, default ``"resp_RT"``
    trial_col : str, default ``"trial_id"``
    group_cols : sequence of str, optional
        Grouping for independent rolling windows
        (e.g., ``["subject", "session"]``).
    window : int, default 5
        Rolling window size.

    Returns
    -------
    pd.DataFrame
        Original data plus ``rt_rolling_mean`` column.
    """
    result = df.copy()
    result["_rt"] = pd.to_numeric(result[rt_col], errors="coerce")

    if group_cols:
        result = result.sort_values(list(group_cols) + [trial_col])
        result["rt_rolling_mean"] = (
            result.groupby(list(group_cols), observed=True)["_rt"]
            .transform(lambda x: x.rolling(window, min_periods=1).mean())
        )
    else:
        result = result.sort_values(trial_col)
        result["rt_rolling_mean"] = result["_rt"].rolling(window, min_periods=1).mean()

    result = result.drop(columns=["_rt"])
    return result
