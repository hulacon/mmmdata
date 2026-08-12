"""End-to-end behavioral report builders.

Each function chains the full pipeline for one report — load data
(``behavioral.io``) -> analysis (``behavioral.accuracy`` /
``behavioral.learning`` / ...) -> interactive figure
(``behavioral.plotting``) — and writes the result to an explicit
output path.

Inputs are explicit: a ``bids_root`` (or preloaded DataFrame(s), which
skip the load step), optional subject/session filters, and the exact
``save_path`` for the output file. No function here consults global
configuration or invents output locations.

All figures are written via ``backend="plotly"`` by default, producing
a standalone interactive HTML file at ``save_path``. Pass
``backend="matplotlib"`` (with an image-format ``save_path`` such as
``.png``) for static output instead.

Each function returns a dict containing at least ``"path"`` (the
written output path as a string) plus report-specific summary counts
(``n_subjects``, ``n_trials``, ``n_sessions`` as applicable). If the
required input data are missing or empty, :class:`ReportDataError`
is raised.

Usage::

    from behavioral import reports

    res = reports.accuracy_by_condition_report(
        "out/accuracy.html", bids_root=Path("/path/to/bids"),
        subjects=["03"],
    )
    print(res["path"], res["n_trials"])
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

import pandas as pd

from . import io as behav_io
from .accuracy import accuracy_by_condition, confidence_accuracy_curve
from .encoding import subsequent_memory_effect
from .final_session import fin_vs_tb_accuracy
from .learning import session_dprime_curve, session_learning_curve
from .preprocessing import remap_scanner_resp
from . import plotting

__all__ = [
    "ReportDataError",
    "accuracy_by_condition_report",
    "learning_curve_report",
    "rt_distribution_report",
    "dprime_curve_report",
    "subsequent_memory_report",
    "confidence_accuracy_report",
    "fin_comparison_report",
    "timeline_responses_report",
]

PathLike = Union[str, Path]


class ReportDataError(ValueError):
    """Raised when the data required for a report are missing or empty."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_tb2afc(
    data: Optional[pd.DataFrame],
    bids_root: Optional[Path],
    subjects: Optional[Sequence[str]],
    sessions: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Return preloaded TB2AFC data or load it, raising if empty."""
    df = data if data is not None else behav_io.load_tb2afc(
        bids_root=bids_root, subjects=subjects, sessions=sessions
    )
    if df.empty:
        raise ReportDataError("No TB2AFC data found for given filters.")
    return df


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def accuracy_by_condition_report(
    save_path: PathLike,
    bids_root: Optional[Path] = None,
    subjects: Optional[Sequence[str]] = None,
    sessions: Optional[Sequence[str]] = None,
    group_by: Optional[Sequence[str]] = None,
    title: Optional[str] = None,
    data: Optional[pd.DataFrame] = None,
    backend: str = "plotly",
) -> dict:
    """Recognition accuracy by experimental condition (bar plot).

    Parameters
    ----------
    save_path : path-like
        Output file to write (HTML for the default plotly backend).
    bids_root : Path, optional
        BIDS root for loading; ignored if ``data`` is given.
    subjects, sessions : sequence of str, optional
        Filters applied when loading.
    group_by : sequence of str, optional
        Grouping columns for the accuracy analysis.
        Default ``["subject", "enCon"]``.
    title : str, optional
    data : pd.DataFrame, optional
        Preloaded TB2AFC trials; skips loading.
    backend : str
        Plotting backend (see ``behavioral.plotting``).

    Returns
    -------
    dict
        ``{"path", "n_subjects", "n_trials"}``.
    """
    df = _load_tb2afc(data, bids_root, subjects, sessions)

    gcols = list(group_by) if group_by else ["subject", "enCon"]
    acc_df = accuracy_by_condition(df, group_cols=gcols)

    plotting.plot_accuracy_by_condition(
        acc_df, backend=backend, title=title, save_path=str(save_path)
    )

    return {
        "path": str(save_path),
        "n_subjects": int(df["subject"].nunique()),
        "n_trials": int(len(df)),
    }


def learning_curve_report(
    save_path: PathLike,
    bids_root: Optional[Path] = None,
    subjects: Optional[Sequence[str]] = None,
    sessions: Optional[Sequence[str]] = None,
    metric: str = "trial_accuracy",
    condition: Optional[str] = None,
    title: Optional[str] = None,
    data: Optional[pd.DataFrame] = None,
    backend: str = "plotly",
) -> dict:
    """A behavioral metric across sessions as a learning curve.

    Parameters as :func:`accuracy_by_condition_report`, plus:

    metric : str
        Column to track (e.g., ``"trial_accuracy"``, ``"resp_RT"``).
    condition : str, optional
        Condition column to split curves by (e.g., ``"enCon"``).

    Returns
    -------
    dict
        ``{"path", "metric", "n_sessions"}``.
    """
    df = _load_tb2afc(data, bids_root, subjects, sessions)

    gcols = ["subject"]
    if condition:
        gcols.append(condition)

    curve_df = session_learning_curve(df, metric_col=metric, group_cols=gcols)

    plotting.plot_learning_curve(
        curve_df, hue=condition, backend=backend, title=title,
        save_path=str(save_path),
    )

    return {
        "path": str(save_path),
        "metric": metric,
        "n_sessions": int(curve_df["session"].nunique()),
    }


def rt_distribution_report(
    save_path: PathLike,
    bids_root: Optional[Path] = None,
    subjects: Optional[Sequence[str]] = None,
    sessions: Optional[Sequence[str]] = None,
    group_by: Optional[str] = None,
    kind: str = "histogram",
    title: Optional[str] = None,
    data: Optional[pd.DataFrame] = None,
    backend: str = "plotly",
) -> dict:
    """Reaction time distribution (histogram or violin).

    Parameters as :func:`accuracy_by_condition_report`, plus:

    group_by : str, optional
        Column to group by (e.g., ``"subject"``, ``"enCon"``).
    kind : str
        ``"histogram"`` or ``"violin"``.

    Returns
    -------
    dict
        ``{"path", "kind", "n_subjects", "n_trials"}``.
    """
    df = _load_tb2afc(data, bids_root, subjects, sessions)

    plotting.plot_rt_distribution(
        df, group_col=group_by, backend=backend, kind=kind, title=title,
        save_path=str(save_path),
    )

    return {
        "path": str(save_path),
        "kind": kind,
        "n_subjects": int(df["subject"].nunique()),
        "n_trials": int(len(df)),
    }


def dprime_curve_report(
    save_path: PathLike,
    bids_root: Optional[Path] = None,
    subjects: Optional[Sequence[str]] = None,
    sessions: Optional[Sequence[str]] = None,
    condition: Optional[str] = None,
    title: Optional[str] = None,
    data: Optional[pd.DataFrame] = None,
    backend: str = "plotly",
) -> dict:
    """d-prime (signal detection sensitivity) across sessions.

    Parameters as :func:`learning_curve_report` (minus ``metric``).

    Returns
    -------
    dict
        ``{"path", "n_sessions"}``.
    """
    df = _load_tb2afc(data, bids_root, subjects, sessions)

    gcols = ["subject"]
    if condition:
        gcols.append(condition)

    dprime_df = session_dprime_curve(df, group_cols=gcols)

    plotting.plot_dprime_curve(
        dprime_df, hue=condition, backend=backend, title=title,
        save_path=str(save_path),
    )

    return {
        "path": str(save_path),
        "n_sessions": int(dprime_df["session"].nunique()),
    }


def subsequent_memory_report(
    save_path: PathLike,
    bids_root: Optional[Path] = None,
    subjects: Optional[Sequence[str]] = None,
    group_by: Optional[Sequence[str]] = None,
    title: Optional[str] = None,
    encoding_data: Optional[pd.DataFrame] = None,
    recognition_data: Optional[pd.DataFrame] = None,
    backend: str = "plotly",
) -> dict:
    """Subsequent memory effect: encoding ratings vs later recognition.

    Scanner button-box responses in the encoding data are remapped to
    the semantic rating scale before analysis.

    Parameters
    ----------
    save_path : path-like
    bids_root : Path, optional
    subjects : sequence of str, optional
    group_by : sequence of str, optional
        Grouping columns. Default ``["subject"]``.
    title : str, optional
    encoding_data, recognition_data : pd.DataFrame, optional
        Preloaded TBencoding / TB2AFC data; each skips its load step.
    backend : str

    Returns
    -------
    dict
        ``{"path", "n_subjects"}``.
    """
    enc_df = encoding_data if encoding_data is not None else (
        behav_io.load_encoding(bids_root=bids_root, subjects=subjects)
    )
    rec_df = recognition_data if recognition_data is not None else (
        behav_io.load_tb2afc(bids_root=bids_root, subjects=subjects)
    )

    if enc_df.empty or rec_df.empty:
        raise ReportDataError("Missing encoding or recognition data.")

    enc_df = remap_scanner_resp(enc_df)
    gcols = list(group_by) if group_by else ["subject"]
    sme_df = subsequent_memory_effect(enc_df, rec_df, group_cols=gcols)

    if sme_df.empty:
        raise ReportDataError("No matched encoding-recognition pairs found.")

    plotting.plot_subsequent_memory(
        sme_df, backend=backend, title=title, save_path=str(save_path)
    )

    return {
        "path": str(save_path),
        "n_subjects": int(sme_df["subject"].nunique()),
    }


def confidence_accuracy_report(
    save_path: PathLike,
    bids_root: Optional[Path] = None,
    subjects: Optional[Sequence[str]] = None,
    sessions: Optional[Sequence[str]] = None,
    title: Optional[str] = None,
    data: Optional[pd.DataFrame] = None,
    backend: str = "plotly",
) -> dict:
    """Confidence-accuracy calibration (per-subject curves).

    Parameters as :func:`accuracy_by_condition_report`.

    Returns
    -------
    dict
        ``{"path", "n_subjects"}``.
    """
    df = _load_tb2afc(data, bids_root, subjects, sessions)

    cal_df = confidence_accuracy_curve(df, group_cols=["subject"])

    plotting.plot_confidence_accuracy(
        cal_df, backend=backend, title=title, save_path=str(save_path)
    )

    return {
        "path": str(save_path),
        "n_subjects": int(cal_df["subject"].nunique()),
    }


def fin_comparison_report(
    save_path: PathLike,
    bids_root: Optional[Path] = None,
    subjects: Optional[Sequence[str]] = None,
    group_by: Optional[Sequence[str]] = None,
    title: Optional[str] = None,
    tb_data: Optional[pd.DataFrame] = None,
    fin_data: Optional[pd.DataFrame] = None,
    backend: str = "plotly",
) -> dict:
    """Session-by-session vs final-session (ses-30) recognition accuracy.

    Parameters
    ----------
    save_path : path-like
    bids_root : Path, optional
    subjects : sequence of str, optional
    group_by : sequence of str, optional
        Condition grouping (e.g., ``["enCon"]``).
    title : str, optional
    tb_data, fin_data : pd.DataFrame, optional
        Preloaded TB2AFC / FIN2AFC data; each skips its load step.
    backend : str

    Returns
    -------
    dict
        ``{"path", "n_subjects"}``.
    """
    tb_df = tb_data if tb_data is not None else (
        behav_io.load_tb2afc(bids_root=bids_root, subjects=subjects)
    )
    fin_df = fin_data if fin_data is not None else (
        behav_io.load_fin2afc(bids_root=bids_root, subjects=subjects)
    )

    if tb_df.empty or fin_df.empty:
        raise ReportDataError("Missing TB2AFC or FIN2AFC data.")

    comp_df = fin_vs_tb_accuracy(fin_df, tb_df, group_cols=group_by)

    plotting.plot_fin_comparison(
        comp_df, backend=backend, title=title, save_path=str(save_path)
    )

    return {
        "path": str(save_path),
        "n_subjects": int(comp_df["subject"].nunique()),
    }


def timeline_responses_report(
    save_path: PathLike,
    bids_root: Optional[Path] = None,
    subjects: Optional[Sequence[str]] = None,
    group_by: Optional[str] = "subject",
    title: Optional[str] = None,
    data: Optional[pd.DataFrame] = None,
    backend: str = "plotly",
) -> dict:
    """Temporal judgment responses from the FINtimeline task (ses-30).

    Parameters
    ----------
    save_path : path-like
    bids_root : Path, optional
    subjects : sequence of str, optional
    group_by : str, optional
        Column to group distributions by.
    title : str, optional
    data : pd.DataFrame, optional
        Preloaded FINtimeline data; skips loading.
    backend : str

    Returns
    -------
    dict
        ``{"path", "n_subjects", "n_trials"}``.
    """
    df = data if data is not None else behav_io.load_fintimeline(
        bids_root=bids_root, subjects=subjects
    )
    if df.empty:
        raise ReportDataError("No FINtimeline data found.")

    plotting.plot_timeline_responses(
        df, group_col=group_by, backend=backend, title=title,
        save_path=str(save_path),
    )

    return {
        "path": str(save_path),
        "n_subjects": int(df["subject"].nunique()),
        "n_trials": int(len(df)),
    }
