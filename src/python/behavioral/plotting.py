"""Visualization for behavioral analysis results.

Every public function accepts a ``backend`` parameter:

- ``"matplotlib"`` (default): returns a matplotlib Figure
- ``"plotly"``: returns a plotly Figure
- ``"both"``: returns a dict ``{"matplotlib": fig, "plotly": fig}``

All functions accept DataFrames (output of analysis functions),
NOT raw data files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Literal, Union

import numpy as np
import pandas as pd

Backend = Literal["matplotlib", "plotly", "both"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_matplotlib():
    """Lazy import matplotlib."""
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend for HPC
    import matplotlib.pyplot as plt
    import seaborn as sns
    return plt, sns


def _get_plotly():
    """Lazy import plotly."""
    import plotly.express as px
    import plotly.graph_objects as go
    return px, go


def _dispatch(mpl_func, plotly_func, backend, **kwargs):
    """Run the appropriate backend function(s) and return result."""
    if backend == "matplotlib":
        return mpl_func(**kwargs)
    elif backend == "plotly":
        return plotly_func(**kwargs)
    elif backend == "both":
        return {
            "matplotlib": mpl_func(**kwargs),
            "plotly": plotly_func(**kwargs),
        }
    else:
        raise ValueError(f"Unknown backend: {backend!r}")


# ---------------------------------------------------------------------------
# Accuracy by condition
# ---------------------------------------------------------------------------

def plot_accuracy_by_condition(
    df: pd.DataFrame,
    x: str = "enCon",
    hue: Optional[str] = "reCon",
    subject_col: str = "subject",
    accuracy_col: str = "accuracy",
    backend: Backend = "matplotlib",
    title: Optional[str] = None,
    save_path: Optional[str] = None,
):
    """Bar plot of accuracy by condition with per-subject points overlaid.

    Parameters
    ----------
    df : pd.DataFrame
        Output of ``accuracy.accuracy_by_condition()``.
    x : str
    hue : str, optional
    subject_col : str
    accuracy_col : str
    backend : Backend
    title : str, optional
    save_path : str, optional
    """
    _title = title or f"Accuracy by {x}"

    def _mpl(**kw):
        plt, sns = _get_matplotlib()
        fig, ax = plt.subplots(figsize=(8, 5))
        if hue and hue in df.columns:
            sns.barplot(data=df, x=x, y=accuracy_col, hue=hue, ax=ax,
                        errorbar="se", alpha=0.7)
            sns.stripplot(data=df, x=x, y=accuracy_col, hue=hue, ax=ax,
                          dodge=True, alpha=0.6, legend=False)
        else:
            sns.barplot(data=df, x=x, y=accuracy_col, ax=ax,
                        errorbar="se", alpha=0.7)
            sns.stripplot(data=df, x=x, y=accuracy_col, ax=ax,
                          alpha=0.6)
        ax.set_title(_title)
        ax.set_ylabel("Accuracy")
        ax.axhline(0.5, ls="--", color="gray", alpha=0.5, label="chance")
        ax.set_ylim(0, 1)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        return fig

    def _plotly(**kw):
        px, go = _get_plotly()
        fig = px.bar(df, x=x, y=accuracy_col, color=hue if hue and hue in df.columns else None,
                     barmode="group", title=_title,
                     hover_data=[subject_col] if subject_col in df.columns else None)
        fig.add_hline(y=0.5, line_dash="dash", line_color="gray",
                      annotation_text="chance")
        fig.update_yaxes(range=[0, 1])
        if save_path:
            html_path = str(save_path).rsplit(".", 1)[0] + ".html"
            fig.write_html(html_path)
        return fig

    return _dispatch(_mpl, _plotly, backend)


# ---------------------------------------------------------------------------
# Learning curve
# ---------------------------------------------------------------------------

def plot_learning_curve(
    df: pd.DataFrame,
    x: str = "session_order",
    y: str = "mean",
    hue: Optional[str] = None,
    subject_col: str = "subject",
    backend: Backend = "matplotlib",
    title: Optional[str] = None,
    save_path: Optional[str] = None,
):
    """Line plot of a metric across sessions (learning curve).

    Shows individual subjects as thin lines, group mean as thick line
    with error ribbon.

    Parameters
    ----------
    df : pd.DataFrame
        Output of ``learning.session_learning_curve()``.
    """
    _title = title or "Learning Curve"

    def _mpl(**kw):
        plt, sns = _get_matplotlib()
        fig, ax = plt.subplots(figsize=(10, 5))
        if hue and hue in df.columns:
            sns.lineplot(data=df, x=x, y=y, hue=hue, style=subject_col,
                         ax=ax, markers=True, dashes=False, alpha=0.6)
        elif subject_col in df.columns:
            # Individual subjects
            for sub, grp in df.groupby(subject_col):
                ax.plot(grp[x], grp[y], "o-", alpha=0.4, label=f"sub-{sub}")
            # Group mean
            group_mean = df.groupby(x)[y].agg(["mean", "sem"]).reset_index()
            ax.plot(group_mean[x], group_mean["mean"], "k-", linewidth=2,
                    label="group mean")
            ax.fill_between(group_mean[x],
                            group_mean["mean"] - group_mean["sem"],
                            group_mean["mean"] + group_mean["sem"],
                            alpha=0.2, color="black")
        else:
            ax.plot(df[x], df[y], "o-")
        ax.set_title(_title)
        ax.set_xlabel("Session")
        ax.set_ylabel(y)
        ax.legend(fontsize=8)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        return fig

    def _plotly(**kw):
        px, go = _get_plotly()
        color = hue if hue and hue in df.columns else subject_col if subject_col in df.columns else None
        fig = px.line(df, x=x, y=y, color=color, markers=True,
                      title=_title)
        if save_path:
            html_path = str(save_path).rsplit(".", 1)[0] + ".html"
            fig.write_html(html_path)
        return fig

    return _dispatch(_mpl, _plotly, backend)


# ---------------------------------------------------------------------------
# RT distribution
# ---------------------------------------------------------------------------

def plot_rt_distribution(
    df: pd.DataFrame,
    rt_col: str = "resp_RT",
    group_col: Optional[str] = None,
    backend: Backend = "matplotlib",
    kind: Literal["histogram", "violin"] = "histogram",
    title: Optional[str] = None,
    save_path: Optional[str] = None,
):
    """Plot RT distribution(s) as histogram or violin.

    Parameters
    ----------
    df : pd.DataFrame
        Raw trial data (not aggregated).
    rt_col : str
    group_col : str, optional
    backend : Backend
    kind : str
    """
    _title = title or "RT Distribution"

    def _mpl(**kw):
        plt, sns = _get_matplotlib()
        fig, ax = plt.subplots(figsize=(8, 5))
        if kind == "violin" and group_col:
            sns.violinplot(data=df, x=group_col, y=rt_col, ax=ax, inner="quart")
        elif kind == "violin":
            sns.violinplot(data=df, y=rt_col, ax=ax, inner="quart")
        elif group_col:
            for name, grp in df.groupby(group_col):
                ax.hist(grp[rt_col].dropna(), bins=30, alpha=0.5, label=str(name))
            ax.legend()
        else:
            ax.hist(df[rt_col].dropna(), bins=30, alpha=0.7)
        ax.set_title(_title)
        ax.set_xlabel("RT (s)")
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        return fig

    def _plotly(**kw):
        px, go = _get_plotly()
        if kind == "violin":
            fig = px.violin(df, y=rt_col, x=group_col, box=True,
                            title=_title)
        else:
            fig = px.histogram(df, x=rt_col, color=group_col,
                               barmode="overlay", title=_title, nbins=30)
        if save_path:
            html_path = str(save_path).rsplit(".", 1)[0] + ".html"
            fig.write_html(html_path)
        return fig

    return _dispatch(_mpl, _plotly, backend)


# ---------------------------------------------------------------------------
# d-prime curve
# ---------------------------------------------------------------------------

def plot_dprime_curve(
    df: pd.DataFrame,
    x: str = "session_order",
    y: str = "dprime_2afc",
    hue: Optional[str] = None,
    subject_col: str = "subject",
    backend: Backend = "matplotlib",
    title: Optional[str] = None,
    save_path: Optional[str] = None,
):
    """Line plot of d' across sessions.

    Parameters
    ----------
    df : pd.DataFrame
        Output of ``learning.session_dprime_curve()``.
    """
    _title = title or "d' Across Sessions"

    def _mpl(**kw):
        plt, sns = _get_matplotlib()
        fig, ax = plt.subplots(figsize=(10, 5))
        if subject_col in df.columns:
            for sub, grp in df.groupby(subject_col):
                ax.plot(grp[x], grp[y], "o-", alpha=0.5, label=f"sub-{sub}")
            group_mean = df.groupby(x)[y].agg(["mean", "sem"]).reset_index()
            ax.plot(group_mean[x], group_mean["mean"], "k-", linewidth=2,
                    label="group mean")
            ax.fill_between(group_mean[x],
                            group_mean["mean"] - group_mean["sem"],
                            group_mean["mean"] + group_mean["sem"],
                            alpha=0.2, color="black")
        else:
            ax.plot(df[x], df[y], "o-")
        ax.axhline(0, ls="--", color="gray", alpha=0.5)
        ax.set_title(_title)
        ax.set_xlabel("Session")
        ax.set_ylabel("d'")
        ax.legend(fontsize=8)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        return fig

    def _plotly(**kw):
        px, go = _get_plotly()
        color = hue if hue and hue in df.columns else subject_col if subject_col in df.columns else None
        fig = px.line(df, x=x, y=y, color=color, markers=True,
                      title=_title)
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        if save_path:
            html_path = str(save_path).rsplit(".", 1)[0] + ".html"
            fig.write_html(html_path)
        return fig

    return _dispatch(_mpl, _plotly, backend)


# ---------------------------------------------------------------------------
# Subsequent memory effect
# ---------------------------------------------------------------------------

def plot_subsequent_memory(
    df: pd.DataFrame,
    x: str = "encoding_rating",
    y: str = "accuracy",
    hue: Optional[str] = None,
    subject_col: str = "subject",
    backend: Backend = "matplotlib",
    title: Optional[str] = None,
    save_path: Optional[str] = None,
):
    """Bar plot of recognition accuracy as a function of encoding rating.

    Parameters
    ----------
    df : pd.DataFrame
        Output of ``encoding.subsequent_memory_effect()``.
    """
    _title = title or "Subsequent Memory Effect"

    def _mpl(**kw):
        plt, sns = _get_matplotlib()
        fig, ax = plt.subplots(figsize=(6, 5))
        if hue and hue in df.columns:
            sns.barplot(data=df, x=x, y=y, hue=hue, ax=ax,
                        errorbar="se", alpha=0.7)
        else:
            sns.barplot(data=df, x=x, y=y, ax=ax,
                        errorbar="se", alpha=0.7)
        ax.set_title(_title)
        ax.set_xlabel("Encoding Rating")
        ax.set_ylabel("Recognition Accuracy")
        ax.axhline(0.5, ls="--", color="gray", alpha=0.5)
        ax.set_ylim(0, 1)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        return fig

    def _plotly(**kw):
        px, go = _get_plotly()
        fig = px.bar(df, x=x, y=y, color=hue if hue and hue in df.columns else None,
                     barmode="group", title=_title)
        fig.add_hline(y=0.5, line_dash="dash", line_color="gray")
        fig.update_yaxes(range=[0, 1])
        if save_path:
            html_path = str(save_path).rsplit(".", 1)[0] + ".html"
            fig.write_html(html_path)
        return fig

    return _dispatch(_mpl, _plotly, backend)


# ---------------------------------------------------------------------------
# Confidence-accuracy calibration
# ---------------------------------------------------------------------------

def plot_confidence_accuracy(
    df: pd.DataFrame,
    subject_col: str = "subject",
    backend: Backend = "matplotlib",
    title: Optional[str] = None,
    save_path: Optional[str] = None,
):
    """Plot accuracy at each confidence level (calibration curve).

    Parameters
    ----------
    df : pd.DataFrame
        Output of ``accuracy.confidence_accuracy_curve()``.
        Must have ``confidence_level`` and ``accuracy`` columns.
    """
    _title = title or "Confidence-Accuracy Calibration"

    def _mpl(**kw):
        plt, sns = _get_matplotlib()
        fig, ax = plt.subplots(figsize=(6, 5))
        if subject_col in df.columns:
            sns.pointplot(data=df, x="confidence_level", y="accuracy",
                          hue=subject_col, ax=ax, dodge=0.1)
        else:
            sns.pointplot(data=df, x="confidence_level", y="accuracy", ax=ax)
        ax.set_title(_title)
        ax.set_xlabel("Confidence (1=maybe, 2=sure)")
        ax.set_ylabel("Accuracy")
        ax.axhline(0.5, ls="--", color="gray", alpha=0.5)
        ax.set_ylim(0, 1)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        return fig

    def _plotly(**kw):
        px, go = _get_plotly()
        color = subject_col if subject_col in df.columns else None
        fig = px.line(df, x="confidence_level", y="accuracy",
                      color=color, markers=True, title=_title)
        fig.add_hline(y=0.5, line_dash="dash", line_color="gray")
        fig.update_yaxes(range=[0, 1])
        if save_path:
            html_path = str(save_path).rsplit(".", 1)[0] + ".html"
            fig.write_html(html_path)
        return fig

    return _dispatch(_mpl, _plotly, backend)


# ---------------------------------------------------------------------------
# Final session comparison
# ---------------------------------------------------------------------------

def plot_fin_comparison(
    df: pd.DataFrame,
    subject_col: str = "subject",
    backend: Backend = "matplotlib",
    title: Optional[str] = None,
    save_path: Optional[str] = None,
):
    """Compare session-by-session accuracy to final session accuracy.

    Parameters
    ----------
    df : pd.DataFrame
        Output of ``final_session.fin_vs_tb_accuracy()``.
        Must have ``source`` and ``accuracy`` columns.
    """
    _title = title or "Session vs Final Recognition Accuracy"

    def _mpl(**kw):
        plt, sns = _get_matplotlib()
        fig, ax = plt.subplots(figsize=(12, 5))
        if subject_col in df.columns:
            for sub, grp in df.groupby(subject_col):
                ax.plot(range(len(grp)), grp["accuracy"].values, "o-",
                        alpha=0.5, label=f"sub-{sub}")
        else:
            ax.plot(range(len(df)), df["accuracy"].values, "o-")
        ax.set_xticks(range(len(df["source"].unique())))
        ax.set_xticklabels(sorted(df["source"].unique()), rotation=45, ha="right")
        ax.set_title(_title)
        ax.set_ylabel("Accuracy")
        ax.axhline(0.5, ls="--", color="gray", alpha=0.5)
        ax.legend(fontsize=8)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        return fig

    def _plotly(**kw):
        px, go = _get_plotly()
        color = subject_col if subject_col in df.columns else None
        fig = px.line(df, x="source", y="accuracy", color=color,
                      markers=True, title=_title)
        fig.add_hline(y=0.5, line_dash="dash", line_color="gray")
        if save_path:
            html_path = str(save_path).rsplit(".", 1)[0] + ".html"
            fig.write_html(html_path)
        return fig

    return _dispatch(_mpl, _plotly, backend)


# ---------------------------------------------------------------------------
# Timeline responses
# ---------------------------------------------------------------------------

def plot_timeline_responses(
    df: pd.DataFrame,
    resp_col: str = "timeline_resp",
    group_col: Optional[str] = "subject",
    backend: Backend = "matplotlib",
    title: Optional[str] = None,
    save_path: Optional[str] = None,
):
    """Distribution of timeline temporal judgments.

    Parameters
    ----------
    df : pd.DataFrame
        Raw FINtimeline trial data.
    resp_col : str
    group_col : str, optional
    """
    _title = title or "Timeline Temporal Judgments"

    def _mpl(**kw):
        plt, sns = _get_matplotlib()
        fig, ax = plt.subplots(figsize=(8, 5))
        if group_col and group_col in df.columns:
            sns.histplot(data=df, x=resp_col, hue=group_col, ax=ax,
                         bins=20, alpha=0.5, stat="density")
        else:
            ax.hist(df[resp_col].dropna(), bins=20, alpha=0.7, density=True)
        ax.set_title(_title)
        ax.set_xlabel("Timeline Response (0=early, 1=late)")
        ax.set_ylabel("Density")
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        return fig

    def _plotly(**kw):
        px, go = _get_plotly()
        fig = px.histogram(df, x=resp_col,
                           color=group_col if group_col and group_col in df.columns else None,
                           barmode="overlay", nbins=20, histnorm="probability density",
                           title=_title)
        if save_path:
            html_path = str(save_path).rsplit(".", 1)[0] + ".html"
            fig.write_html(html_path)
        return fig

    return _dispatch(_mpl, _plotly, backend)


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------

def save_figure(
    fig,
    path: str,
    dpi: int = 300,
    backend: Backend = "matplotlib",
) -> None:
    """Save figure to disk, auto-detecting backend.

    Parameters
    ----------
    fig : matplotlib.figure.Figure or plotly.graph_objects.Figure
        Figure to save.
    path : str
        Output file path.
    dpi : int, default 300
        DPI for raster formats (matplotlib only).
    backend : Backend
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if backend == "matplotlib":
        fig.savefig(str(path), dpi=dpi, bbox_inches="tight")
    elif backend == "plotly":
        if path.suffix == ".html":
            fig.write_html(str(path))
        else:
            fig.write_image(str(path))
