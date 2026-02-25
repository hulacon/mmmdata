"""Accuracy and signal detection metrics for recognition memory.

All functions accept DataFrames (from ``io`` module) and optional
``group_cols`` for flexible condition-wise breakdowns.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------------

def accuracy_by_condition(
    df: pd.DataFrame,
    accuracy_col: str = "trial_accuracy",
    group_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Compute mean accuracy grouped by condition columns.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``accuracy_col``.
    accuracy_col : str, default ``"trial_accuracy"``
        Binary accuracy column (0/1).
    group_cols : sequence of str, optional
        Columns to group by. Common patterns:

        - ``["subject"]``: per-subject overall
        - ``["subject", "enCon"]``: by encoding condition
        - ``["subject", "enCon", "reCon"]``: full factorial
        - ``["subject", "session"]``: per-session

    Returns
    -------
    pd.DataFrame
        Columns: ``*group_cols``, ``n_trials``, ``n_correct``,
        ``accuracy``, ``se``.
    """
    acc = pd.to_numeric(df[accuracy_col], errors="coerce")
    work = df.copy()
    work["_acc"] = acc

    if group_cols:
        grouped = work.groupby(list(group_cols), observed=True)
    else:
        grouped = work.groupby(lambda _: "all")

    result = grouped["_acc"].agg(
        n_trials="count",
        n_correct="sum",
        accuracy="mean",
        se="sem",
    ).reset_index()

    if not group_cols:
        result = result.drop(columns=["index"])

    result["n_correct"] = result["n_correct"].astype(int)
    return result


# ---------------------------------------------------------------------------
# Signal detection theory
# ---------------------------------------------------------------------------

def dprime(
    hit_rate: float,
    false_alarm_rate: float,
    correction: str = "loglinear",
    n_signal: Optional[int] = None,
    n_noise: Optional[int] = None,
) -> float:
    """Compute d-prime (d') signal detection sensitivity measure.

    Parameters
    ----------
    hit_rate : float
        Proportion of signal trials correctly identified.
    false_alarm_rate : float
        Proportion of noise trials incorrectly identified as signal.
    correction : str, default ``"loglinear"``
        How to handle rates of 0 or 1.

        - ``"loglinear"``: Add 0.5 to hits/FAs and 1 to signal/noise N
          before computing rates (Hautus, 1995). Requires ``n_signal``
          and ``n_noise``.
        - ``"clip"``: Clip to ``[1/(2N), 1 - 1/(2N)]``.
    n_signal : int, optional
        Number of signal trials (required for loglinear/clip).
    n_noise : int, optional
        Number of noise trials (required for loglinear/clip).

    Returns
    -------
    float
        d' value. Higher = better discrimination.
    """
    hr, far = _correct_rates(hit_rate, false_alarm_rate, correction,
                              n_signal, n_noise)
    return float(norm.ppf(hr) - norm.ppf(far))


def criterion(
    hit_rate: float,
    false_alarm_rate: float,
    correction: str = "loglinear",
    n_signal: Optional[int] = None,
    n_noise: Optional[int] = None,
) -> float:
    """Compute response criterion (c) from signal detection theory.

    Parameters
    ----------
    hit_rate : float
    false_alarm_rate : float
    correction : str, default ``"loglinear"``
    n_signal : int, optional
    n_noise : int, optional

    Returns
    -------
    float
        c value. Negative = liberal bias (tendency to say "signal").
    """
    hr, far = _correct_rates(hit_rate, false_alarm_rate, correction,
                              n_signal, n_noise)
    return float(-0.5 * (norm.ppf(hr) + norm.ppf(far)))


def _correct_rates(
    hr: float,
    far: float,
    correction: str,
    n_signal: Optional[int],
    n_noise: Optional[int],
) -> tuple[float, float]:
    """Apply boundary correction to hit rate and false alarm rate."""
    if correction == "loglinear":
        if n_signal is not None and n_noise is not None:
            # Hautus (1995) loglinear correction: always applied
            hr = (hr * n_signal + 0.5) / (n_signal + 1)
            far = (far * n_noise + 0.5) / (n_noise + 1)
        else:
            # Fallback: simple clip if N not provided
            hr = np.clip(hr, 0.001, 0.999)
            far = np.clip(far, 0.001, 0.999)
    elif correction == "clip":
        if n_signal is not None and n_noise is not None:
            floor_hr = 1 / (2 * n_signal)
            ceil_hr = 1 - 1 / (2 * n_signal)
            floor_far = 1 / (2 * n_noise)
            ceil_far = 1 - 1 / (2 * n_noise)
            hr = np.clip(hr, floor_hr, ceil_hr)
            far = np.clip(far, floor_far, ceil_far)
        else:
            hr = np.clip(hr, 0.001, 0.999)
            far = np.clip(far, 0.001, 0.999)
    return float(hr), float(far)


def compute_sdt_2afc(
    df: pd.DataFrame,
    group_cols: Optional[Sequence[str]] = None,
    accuracy_col: str = "trial_accuracy",
    correction: str = "loglinear",
) -> pd.DataFrame:
    """Compute signal detection theory metrics for 2AFC data.

    In 2AFC, accuracy maps directly to d':
    ``d' = sqrt(2) * z(proportion_correct)``
    (Macmillan & Creelman, 2005).

    Also computes a yes/no-equivalent d' using ``recog`` vs
    ``correct_resp`` if those columns are present, with hit/FA
    rates for richer diagnostics.

    Parameters
    ----------
    df : pd.DataFrame
        2AFC data with ``trial_accuracy`` and optionally ``recog``
        and ``correct_resp``.
    group_cols : sequence of str, optional
        Grouping columns (e.g., ``["subject"]``, ``["subject", "enCon"]``).
    accuracy_col : str, default ``"trial_accuracy"``
    correction : str, default ``"loglinear"``

    Returns
    -------
    pd.DataFrame
        Columns: ``*group_cols``, ``n_trials``, ``accuracy``,
        ``dprime_2afc``.
        If ``recog`` and ``correct_resp`` are present, also:
        ``hit_rate``, ``fa_rate``, ``dprime_yesno``, ``criterion``.
    """
    work = df.copy()
    work["_acc"] = pd.to_numeric(work[accuracy_col], errors="coerce")

    if group_cols:
        groups = work.groupby(list(group_cols), observed=True)
    else:
        groups = work.groupby(lambda _: "all")

    rows = []
    for name, grp in groups:
        acc = grp["_acc"].dropna()
        n = len(acc)
        if n == 0:
            continue
        p_correct = acc.mean()

        # 2AFC d': d' = sqrt(2) * z(p_correct)
        p_corrected = np.clip(p_correct, 0.001, 0.999)
        d_2afc = float(np.sqrt(2) * norm.ppf(p_corrected))

        row = {
            "n_trials": n,
            "accuracy": float(p_correct),
            "dprime_2afc": d_2afc,
        }

        # Yes/no equivalent if recog column available
        if "recog" in grp.columns and "correct_resp" in grp.columns:
            recog = pd.to_numeric(grp["recog"], errors="coerce")
            correct = pd.to_numeric(grp["correct_resp"], errors="coerce")
            valid = recog.notna() & correct.notna()
            if valid.any():
                # "Hit" = chose correct position, "FA" = chose incorrect
                # In 2AFC these sum to 1, but we compute for diagnostics
                n_signal = valid.sum()
                hits = ((recog == correct) & valid).sum()
                hr = hits / n_signal if n_signal > 0 else 0.5
                far = 1 - hr  # In 2AFC, FA = 1 - hit_rate

                row["hit_rate"] = float(hr)
                row["fa_rate"] = float(far)

        if isinstance(name, tuple):
            for col, val in zip(group_cols, name):
                row[col] = val
        elif group_cols:
            row[group_cols[0]] = name

        rows.append(row)

    result = pd.DataFrame(rows)
    if group_cols and not result.empty:
        cols = list(group_cols) + [c for c in result.columns if c not in group_cols]
        result = result[cols]
    return result


# ---------------------------------------------------------------------------
# Confidence analysis
# ---------------------------------------------------------------------------

def confidence_accuracy_curve(
    df: pd.DataFrame,
    group_cols: Optional[Sequence[str]] = None,
    resp_col: str = "resp",
    accuracy_col: str = "trial_accuracy",
) -> pd.DataFrame:
    """Compute accuracy at each confidence level (1-4) for 2AFC data.

    Confidence is derived from ``resp``:

    - ``resp`` 1 or 4 = ``"sure"``  (confidence 2)
    - ``resp`` 2 or 3 = ``"maybe"`` (confidence 1)

    Parameters
    ----------
    df : pd.DataFrame
    group_cols : sequence of str, optional
    resp_col : str, default ``"resp"``
    accuracy_col : str, default ``"trial_accuracy"``

    Returns
    -------
    pd.DataFrame
        Columns: ``*group_cols``, ``confidence_level`` (1=maybe, 2=sure),
        ``n_trials``, ``accuracy``, ``se``.
    """
    work = df.copy()
    resp = pd.to_numeric(work[resp_col], errors="coerce")
    work["confidence_level"] = np.where(resp.isin([1, 4]), 2, 1)
    work["_acc"] = pd.to_numeric(work[accuracy_col], errors="coerce")

    gcols = list(group_cols or []) + ["confidence_level"]
    grouped = work.groupby(gcols, observed=True)

    result = grouped["_acc"].agg(
        n_trials="count",
        accuracy="mean",
        se="sem",
    ).reset_index()

    return result
