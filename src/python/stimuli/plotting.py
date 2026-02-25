"""Stimulus feature visualization for MMMData.

Wraps viz2psy's interactive visualization modules for use with
MMMData's stimulus data. Falls back to basic plotly if viz2psy
is not available.

viz2psy functions used:
- viz2psy.viz.interactive.timeseries.plot_timeseries_interactive
- viz2psy.viz.interactive.scatter.plot_scatter_interactive
- viz2psy.viz.heatmap.plot_heatmap

Data is loaded from viz2psy score CSVs (one per movie or one
combined CSV for images).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# High-dimensional embedding prefixes
_HIGHDIM_PREFIXES = ("clip_", "dinov2_", "gist_", "saliency_")


def _ensure_viz2psy(viz2psy_dir: Optional[str] = None) -> bool:
    """Add viz2psy to sys.path if available. Return True if importable."""
    if viz2psy_dir:
        src = str(Path(viz2psy_dir) / "src")
        if src not in sys.path:
            sys.path.insert(0, src)
    try:
        import viz2psy.viz  # noqa: F401
        return True
    except ImportError:
        return False


def _save_html(fig, save_path: Optional[str] = None) -> None:
    """Save plotly figure to HTML if path provided."""
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(save_path))


def _scalar_columns(df: pd.DataFrame) -> list[str]:
    """Return non-embedding numeric columns."""
    return [
        c for c in df.select_dtypes(include="number").columns
        if not any(c.startswith(p) for p in _HIGHDIM_PREFIXES)
        and c != "time"
    ]


def _load_image_scores(scores_dir: str) -> pd.DataFrame:
    """Load image scores CSV."""
    p = Path(scores_dir)
    if p.is_file() and p.suffix == ".csv":
        return pd.read_csv(p)
    csv = p / "viz2psy_scores.csv" if p.is_dir() else p
    if not csv.exists() and p.is_dir():
        csv = p.parent / "viz2psy_scores.csv"
    if not csv.exists():
        raise FileNotFoundError(f"Image scores not found at {csv}")
    return pd.read_csv(csv)


def _find_movie_csv(scores_dir: str, movie_name: str) -> Path:
    """Find a movie scores CSV by name."""
    d = Path(scores_dir)
    exact = d / f"{movie_name}_scores.csv"
    if exact.exists():
        return exact
    lower = movie_name.lower()
    for csv in d.glob("*_scores.csv"):
        if csv.stem.removesuffix("_scores").lower() == lower:
            return csv
    raise FileNotFoundError(
        f"Movie scores not found for '{movie_name}' in {d}"
    )


# ---------------------------------------------------------------------------
# Movie feature timeline
# ---------------------------------------------------------------------------

def plot_movie_feature_timeline(
    scores_dir: str,
    movie_name: str,
    features: Optional[list[str]] = None,
    time_range: Optional[list[float]] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    viz2psy_dir: Optional[str] = None,
):
    """Plot feature scores over time for a movie.

    Uses viz2psy's interactive timeseries if available, otherwise
    falls back to basic plotly.

    Parameters
    ----------
    scores_dir : str
        Path to directory containing movie score CSVs.
    movie_name : str
        Movie name (e.g., "Mr-Bean").
    features : list of str, optional
        Features to plot. Supports glob patterns if viz2psy available.
        Default: all scalar features.
    time_range : list of float, optional
        [start, end] in seconds.
    title : str, optional
    save_path : str, optional
    viz2psy_dir : str, optional
        Path to viz2psy repo for importing visualization modules.

    Returns
    -------
    plotly Figure
    """
    csv_path = _find_movie_csv(scores_dir, movie_name)
    df = pd.read_csv(csv_path)

    if time_range and len(time_range) == 2:
        df = df[(df["time"] >= time_range[0]) & (df["time"] <= time_range[1])]

    _title = title or f"Feature Timeline: {movie_name}"

    # Try viz2psy's interactive timeseries
    if _ensure_viz2psy(viz2psy_dir):
        from viz2psy.viz.interactive.timeseries import plot_timeseries_interactive
        from viz2psy.viz.sidecar import SidecarMetadata

        # Load sidecar if available
        sidecar = None
        meta_path = csv_path.with_suffix("").with_suffix(".meta.json")
        if not meta_path.exists():
            meta_path = csv_path.parent / f"{csv_path.stem}.meta.json"
        if meta_path.exists():
            sidecar = SidecarMetadata.from_file(str(meta_path))

        fig = plot_timeseries_interactive(
            df, features=features, time_col="time",
            title=_title, sidecar=sidecar,
        )
    else:
        # Fallback: basic plotly
        import plotly.graph_objects as go

        plot_cols = features if features else _scalar_columns(df)
        plot_cols = [f for f in plot_cols if f in df.columns]

        fig = go.Figure()
        for col in plot_cols:
            fig.add_trace(go.Scatter(
                x=df["time"], y=df[col], mode="lines",
                name=col, line=dict(width=1),
            ))
        fig.update_layout(
            title=_title, xaxis_title="Time (s)", yaxis_title="Score",
            template="plotly_white", hovermode="x unified",
        )

    _save_html(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Image feature comparison
# ---------------------------------------------------------------------------

def plot_image_feature_comparison(
    scores_dir: str,
    feature: str,
    top_n: Optional[int] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    viz2psy_dir: Optional[str] = None,
):
    """Plot a feature value across images, sorted by value.

    Parameters
    ----------
    scores_dir : str
        Path to scores CSV or directory containing it.
    feature : str
        Feature column name (e.g., "memorability", "Awe").
    top_n : int, optional
        Show only the top N images.
    title : str, optional
    save_path : str, optional
    viz2psy_dir : str, optional

    Returns
    -------
    plotly Figure
    """
    import plotly.express as px

    df = _load_image_scores(scores_dir)

    if feature not in df.columns:
        available = _scalar_columns(df)
        raise ValueError(
            f"Feature '{feature}' not found. Available: {available}"
        )

    df = df.sort_values(feature, ascending=False).reset_index(drop=True)
    if top_n:
        df = df.head(top_n)

    label_col = "filename" if "filename" in df.columns else df.index

    fig = px.bar(
        df, x=label_col, y=feature,
        title=title or f"{feature} Across Images",
        labels={feature: feature, "filename": "Image"},
    )
    fig.update_layout(xaxis_tickangle=-45, template="plotly_white")

    _save_html(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Similarity matrix
# ---------------------------------------------------------------------------

def plot_feature_similarity_matrix(
    scores_dir: str,
    model: str = "clip",
    n_stimuli: Optional[int] = 50,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    viz2psy_dir: Optional[str] = None,
):
    """Plot cosine similarity matrix for stimulus embeddings.

    Parameters
    ----------
    scores_dir : str
        Path to scores CSV or directory containing it.
    model : str, default "clip"
        Embedding prefix ("clip" or "dinov2").
    n_stimuli : int, optional, default 50
        Subsample to this many stimuli for readability.
    title : str, optional
    save_path : str, optional
    viz2psy_dir : str, optional

    Returns
    -------
    plotly Figure
    """
    import plotly.graph_objects as go

    df = _load_image_scores(scores_dir)

    prefix = f"{model}_"
    embed_cols = [c for c in df.columns if c.startswith(prefix)]
    if not embed_cols:
        raise ValueError(
            f"No embedding columns with prefix '{prefix}' found."
        )

    embeddings = df[embed_cols].values
    labels = df["filename"].tolist() if "filename" in df.columns else [
        str(i) for i in range(len(df))
    ]

    if n_stimuli and len(df) > n_stimuli:
        idx = np.random.default_rng(42).choice(
            len(df), n_stimuli, replace=False
        )
        idx.sort()
        embeddings = embeddings[idx]
        labels = [labels[i] for i in idx]

    # Cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    normalized = embeddings / norms
    sim_matrix = normalized @ normalized.T

    short_labels = [Path(l).stem[:20] for l in labels]

    fig = go.Figure(data=go.Heatmap(
        z=sim_matrix, x=short_labels, y=short_labels,
        colorscale="Viridis", zmin=0, zmax=1,
        colorbar=dict(title="Cosine Sim"),
    ))
    fig.update_layout(
        title=title or f"{model.upper()} Embedding Similarity ({len(labels)} stimuli)",
        template="plotly_white", height=700, width=750,
    )

    _save_html(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Feature distribution
# ---------------------------------------------------------------------------

def plot_feature_distribution(
    scores_dir: str,
    feature: str,
    group_by: Optional[str] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    viz2psy_dir: Optional[str] = None,
):
    """Plot the distribution of a feature across stimuli.

    Parameters
    ----------
    scores_dir : str
        Path to scores CSV or directory containing it.
    feature : str
        Feature column name.
    group_by : str, optional
        Column to group by for colored histograms.
    title : str, optional
    save_path : str, optional
    viz2psy_dir : str, optional

    Returns
    -------
    plotly Figure
    """
    import plotly.express as px

    df = _load_image_scores(scores_dir)

    if feature not in df.columns:
        available = _scalar_columns(df)
        raise ValueError(
            f"Feature '{feature}' not found. Available: {available}"
        )

    fig = px.histogram(
        df, x=feature, color=group_by,
        barmode="overlay", nbins=40,
        title=title or f"Distribution of {feature}",
        opacity=0.7,
    )
    fig.update_layout(template="plotly_white")

    _save_html(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Embedding scatter (DR projection)
# ---------------------------------------------------------------------------

def plot_embedding_scatter(
    scores_dir: str,
    model: str = "clip",
    method: str = "pca",
    color_by: Optional[str] = None,
    n_stimuli: Optional[int] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    viz2psy_dir: Optional[str] = None,
):
    """Plot 2D scatter of stimulus embeddings via dimensionality reduction.

    Uses viz2psy's interactive scatter if available (supports PCA, UMAP,
    t-SNE, MDS), otherwise falls back to PCA via sklearn.

    Parameters
    ----------
    scores_dir : str
        Path to scores CSV or directory containing it.
    model : str, default "clip"
        Embedding prefix ("clip" or "dinov2").
    method : str, default "pca"
        DR method: "pca", "umap", "tsne", "mds".
    color_by : str, optional
        Column to color points by (e.g., "memorability").
    n_stimuli : int, optional
        Subsample to this many stimuli.
    title : str, optional
    save_path : str, optional
    viz2psy_dir : str, optional

    Returns
    -------
    plotly Figure
    """
    df = _load_image_scores(scores_dir)

    if n_stimuli and len(df) > n_stimuli:
        df = df.sample(n=n_stimuli, random_state=42).reset_index(drop=True)

    prefix = f"{model}_"
    embed_cols = [c for c in df.columns if c.startswith(prefix)]
    if not embed_cols:
        raise ValueError(f"No embedding columns with prefix '{prefix}'")

    _title = title or f"{model.upper()} Embeddings ({method.upper()})"

    if _ensure_viz2psy(viz2psy_dir):
        from viz2psy.viz.interactive.scatter import plot_scatter_interactive
        fig = plot_scatter_interactive(
            df, features=[f"{model}_*"], method=method,
            color_by=color_by, title=_title,
        )
    else:
        # Fallback: PCA with sklearn
        import plotly.express as px
        from sklearn.decomposition import PCA

        embeddings = df[embed_cols].values
        pca = PCA(n_components=2)
        coords = pca.fit_transform(embeddings)
        df["_pc1"] = coords[:, 0]
        df["_pc2"] = coords[:, 1]

        fig = px.scatter(
            df, x="_pc1", y="_pc2", color=color_by,
            hover_data=["filename"] if "filename" in df.columns else None,
            title=_title,
            labels={"_pc1": "PC1", "_pc2": "PC2"},
        )
        fig.update_layout(template="plotly_white")

    _save_html(fig, save_path)
    return fig
