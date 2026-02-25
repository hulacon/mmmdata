"""Cross-session learning curves and longitudinal analyses.

Functions for tracking behavioral metrics across the 15 trial-based
sessions (ses-04 through ses-18).
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm

from .constants import SESSION_ORDER


def session_learning_curve(
    df: pd.DataFrame,
    metric_col: str = "trial_accuracy",
    group_cols: Optional[Sequence[str]] = None,
    session_col: str = "session",
) -> pd.DataFrame:
    """Compute a metric (accuracy, RT, etc.) across sessions.

    Parameters
    ----------
    df : pd.DataFrame
    metric_col : str, default ``"trial_accuracy"``
        Column to aggregate (mean) per session.
    group_cols : sequence of str, optional
        Additional grouping beyond session (e.g., ``["subject"]``,
        ``["subject", "enCon"]``).
    session_col : str, default ``"session"``

    Returns
    -------
    pd.DataFrame
        Columns: ``session``, ``session_order``, ``*group_cols``,
        ``mean``, ``se``, ``n``.
    """
    work = df.copy()
    work["_metric"] = pd.to_numeric(work[metric_col], errors="coerce")

    gcols = [session_col] + list(group_cols or [])
    grouped = work.groupby(gcols, observed=True)

    result = grouped["_metric"].agg(
        mean="mean",
        se="sem",
        n="count",
    ).reset_index()

    # Add session order for plotting
    result["session_order"] = result[session_col].map(SESSION_ORDER)
    return result


def session_dprime_curve(
    df: pd.DataFrame,
    session_col: str = "session",
    group_cols: Optional[Sequence[str]] = None,
    accuracy_col: str = "trial_accuracy",
) -> pd.DataFrame:
    """Compute 2AFC d' at each session.

    Uses the standard 2AFC formula: ``d' = sqrt(2) * z(p_correct)``
    (Macmillan & Creelman, 2005).

    Parameters
    ----------
    df : pd.DataFrame
    session_col : str, default ``"session"``
    group_cols : sequence of str, optional
        (e.g., ``["subject"]``, ``["subject", "enCon"]``)
    accuracy_col : str, default ``"trial_accuracy"``

    Returns
    -------
    pd.DataFrame
        Columns: ``session``, ``session_order``, ``*group_cols``,
        ``accuracy``, ``dprime_2afc``, ``n``.
    """
    work = df.copy()
    work["_acc"] = pd.to_numeric(work[accuracy_col], errors="coerce")

    gcols = [session_col] + list(group_cols or [])
    grouped = work.groupby(gcols, observed=True)

    rows = []
    for name, grp in grouped:
        acc = grp["_acc"].dropna()
        n = len(acc)
        if n == 0:
            continue
        p_correct = acc.mean()
        p_corrected = np.clip(p_correct, 0.001, 0.999)
        d = float(np.sqrt(2) * norm.ppf(p_corrected))

        row = {"accuracy": float(p_correct), "dprime_2afc": d, "n": n}
        if isinstance(name, tuple):
            for col, val in zip(gcols, name):
                row[col] = val
        else:
            row[gcols[0]] = name
        rows.append(row)

    result = pd.DataFrame(rows)
    if not result.empty:
        result["session_order"] = result[session_col].map(SESSION_ORDER)
        cols = gcols + ["session_order"] + [
            c for c in result.columns if c not in gcols + ["session_order"]
        ]
        result = result[cols]
    return result


def compare_conditions_over_sessions(
    df: pd.DataFrame,
    condition_col: str = "enCon",
    metric_col: str = "trial_accuracy",
    session_col: str = "session",
    subject_col: str = "subject",
) -> pd.DataFrame:
    """Compute per-subject, per-session, per-condition metrics.

    Useful for plotting condition-comparison learning curves.

    Parameters
    ----------
    df : pd.DataFrame
    condition_col : str, default ``"enCon"``
    metric_col : str, default ``"trial_accuracy"``
    session_col : str, default ``"session"``
    subject_col : str, default ``"subject"``

    Returns
    -------
    pd.DataFrame
        Columns: ``subject``, ``session``, ``session_order``,
        ``condition``, ``mean``, ``se``, ``n``.
    """
    work = df.copy()
    work["_metric"] = pd.to_numeric(work[metric_col], errors="coerce")

    gcols = [subject_col, session_col, condition_col]
    grouped = work.groupby(gcols, observed=True)

    result = grouped["_metric"].agg(
        mean="mean",
        se="sem",
        n="count",
    ).reset_index()

    result = result.rename(columns={condition_col: "condition"})
    result["session_order"] = result[session_col].map(SESSION_ORDER)
    return result
