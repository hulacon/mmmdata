"""Analyses specific to the final session (ses-30) tasks.

Compares final recognition (FIN2AFC) to session-by-session
recognition (TB2AFC), and analyzes temporal judgment (FINtimeline).
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd


def fin_vs_tb_accuracy(
    fin2afc_df: pd.DataFrame,
    tb2afc_df: pd.DataFrame,
    group_cols: Optional[Sequence[str]] = None,
    accuracy_col: str = "trial_accuracy",
    subject_col: str = "subject",
    session_col: str = "session",
) -> pd.DataFrame:
    """Compare FIN2AFC (ses-30) accuracy to session-by-session TB2AFC.

    Parameters
    ----------
    fin2afc_df : pd.DataFrame
        FIN2AFC data.
    tb2afc_df : pd.DataFrame
        Session-by-session TB2AFC data.
    group_cols : sequence of str, optional
        Condition grouping (e.g., ``["enCon"]``).
    accuracy_col : str
    subject_col : str
    session_col : str

    Returns
    -------
    pd.DataFrame
        Columns: ``subject``, ``source`` (``"TB-ses-XX"`` or ``"FIN-ses-30"``),
        ``*group_cols``, ``accuracy``, ``se``, ``n``.
    """
    gcols = list(group_cols or [])

    # TB2AFC: per-subject, per-session accuracy
    tb = tb2afc_df.copy()
    tb["_acc"] = pd.to_numeric(tb[accuracy_col], errors="coerce")
    tb_gcols = [subject_col, session_col] + gcols
    tb_agg = tb.groupby(tb_gcols, observed=True)["_acc"].agg(
        accuracy="mean", se="sem", n="count",
    ).reset_index()
    tb_agg["source"] = "TB-ses-" + tb_agg[session_col].astype(str)

    # FIN2AFC: per-subject accuracy
    fin = fin2afc_df.copy()
    fin["_acc"] = pd.to_numeric(fin[accuracy_col], errors="coerce")
    fin_gcols = [subject_col] + gcols
    fin_agg = fin.groupby(fin_gcols, observed=True)["_acc"].agg(
        accuracy="mean", se="sem", n="count",
    ).reset_index()
    fin_agg["source"] = "FIN-ses-30"
    fin_agg[session_col] = "30"

    # Combine
    cols = [subject_col, "source", session_col] + gcols + ["accuracy", "se", "n"]
    result = pd.concat(
        [tb_agg[[c for c in cols if c in tb_agg.columns]],
         fin_agg[[c for c in cols if c in fin_agg.columns]]],
        ignore_index=True,
    )
    return result


def timeline_analysis(
    df: pd.DataFrame,
    group_cols: Optional[Sequence[str]] = None,
    resp_col: str = "timeline_resp",
    accuracy_col: str = "trial_accuracy",
) -> pd.DataFrame:
    """Analyze temporal judgment accuracy from FINtimeline.

    ``timeline_resp`` is a 0-1 scale (early-to-late) reflecting the
    participant's judgment of when a stimulus was encountered.

    Parameters
    ----------
    df : pd.DataFrame
    group_cols : sequence of str, optional
    resp_col : str, default ``"timeline_resp"``
    accuracy_col : str, default ``"trial_accuracy"``

    Returns
    -------
    pd.DataFrame
        Columns: ``*group_cols``, ``mean_resp``, ``sd_resp``,
        ``accuracy``, ``n``.
    """
    work = df.copy()
    work["_resp"] = pd.to_numeric(work[resp_col], errors="coerce")
    work["_acc"] = pd.to_numeric(work[accuracy_col], errors="coerce")

    if group_cols:
        grouped = work.groupby(list(group_cols), observed=True)
    else:
        grouped = work.groupby(lambda _: "all")

    result = grouped.agg(
        mean_resp=("_resp", "mean"),
        sd_resp=("_resp", "std"),
        accuracy=("_acc", "mean"),
        n=("_resp", "count"),
    ).reset_index()

    if not group_cols:
        result = result.drop(columns=["index"])

    return result


def timeline_by_condition(
    df: pd.DataFrame,
    resp_col: str = "timeline_resp",
    accuracy_col: str = "trial_accuracy",
    subject_col: str = "subject",
) -> pd.DataFrame:
    """Timeline judgment accuracy and response by enCon x reCon.

    Parameters
    ----------
    df : pd.DataFrame
    resp_col : str
    accuracy_col : str
    subject_col : str

    Returns
    -------
    pd.DataFrame
        Columns: ``subject``, ``enCon``, ``reCon``, ``mean_resp``,
        ``sd_resp``, ``accuracy``, ``n``.
    """
    return timeline_analysis(
        df,
        group_cols=[subject_col, "enCon", "reCon"],
        resp_col=resp_col,
        accuracy_col=accuracy_col,
    )


def long_term_retention_curve(
    tb2afc_df: pd.DataFrame,
    fin2afc_df: pd.DataFrame,
    subject_col: str = "subject",
    session_col: str = "session",
    accuracy_col: str = "trial_accuracy",
    pair_col: str = "pairId",
) -> pd.DataFrame:
    """Track item-level accuracy from session-by-session through final.

    Matches items by ``pairId`` across TB sessions and FIN2AFC to
    create a retention curve.

    Parameters
    ----------
    tb2afc_df : pd.DataFrame
    fin2afc_df : pd.DataFrame
    subject_col : str
    session_col : str
    accuracy_col : str
    pair_col : str

    Returns
    -------
    pd.DataFrame
        Per-subject, per-pairId: ``initial_session``, ``initial_accuracy``,
        ``final_accuracy``, ``retention_delta``.
    """
    # TB: find the first session each item was tested
    tb = tb2afc_df[[subject_col, session_col, pair_col, accuracy_col]].copy()
    tb[accuracy_col] = pd.to_numeric(tb[accuracy_col], errors="coerce")
    tb[pair_col] = pd.to_numeric(tb[pair_col], errors="coerce")
    tb = tb.sort_values([subject_col, pair_col, session_col])
    tb_first = tb.groupby([subject_col, pair_col], observed=True).first().reset_index()
    tb_first = tb_first.rename(columns={
        session_col: "initial_session",
        accuracy_col: "initial_accuracy",
    })

    # FIN: final accuracy per item
    fin = fin2afc_df[[subject_col, pair_col, accuracy_col]].copy()
    fin[accuracy_col] = pd.to_numeric(fin[accuracy_col], errors="coerce")
    fin[pair_col] = pd.to_numeric(fin[pair_col], errors="coerce")
    fin = fin.rename(columns={accuracy_col: "final_accuracy"})

    # Merge
    merged = tb_first.merge(fin, on=[subject_col, pair_col], how="inner")
    merged["retention_delta"] = merged["final_accuracy"] - merged["initial_accuracy"]

    return merged
