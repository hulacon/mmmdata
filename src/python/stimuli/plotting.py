"""Stimulus feature visualization for MMMData.

Wraps viz2psy's interactive visualization modules for use with
MMMData's stimulus data. Falls back to basic plotly if viz2psy
is not available.

viz2psy functions used:
- viz2psy.viz.interactive.timeseries.plot_timeseries_interactive
- viz2psy.viz.interactive.scatter.plot_scatter_interactive
- viz2psy.viz.heatmap.plot_heatmap

Data comes from the Contract B feature store's psytwill aggregates
(``derivatives/stimuli_features/psytwill/<group>_features.parquet``),
identified by ``store`` (the directory) and ``group``. Those tables are
long — one row per (stimulus, coordinate, model, feature) — so the loader
here pivots back to the wide frame the plotting code expects.

Superseded the pre-0.6.0 ``stimuli/*/viz2psy_scores*`` CSVs, which covered
only viz2psy, reached neither twp1000 nor any audio or text feature, and
used the column names viz2psy 0.6.0 renamed.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


#: Contract B §4.1 embedding columns: a model prefix plus fixed-width index
#: suffixes (``clip_000``, ``ebind_1023``, the ``saliency_23_23`` grid).
#: Named scores (``places_airfield``, ``llstat_b_mean``) do not match.
_EMBEDDING_RE = re.compile(r"_\d+(_\d+)*$")

#: Long-table columns that locate a row rather than measure it.
_KEY_COLUMNS = (
    "stimulus_id", "voice", "time", "onset", "offset", "chunk_idx", "word_idx",
)


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
    """Return plottable numeric columns: no embeddings, no key columns."""
    return [
        c for c in df.select_dtypes(include="number").columns
        if not _EMBEDDING_RE.search(c) and c not in _KEY_COLUMNS
    ]


def _group_path(store: str, group: str) -> Path:
    return Path(store) / f"{group}_features.parquet"


def available_groups(store: str) -> list[str]:
    """Group ids in the feature store, for an error message worth reading."""
    d = Path(store)
    if not d.exists():
        return []
    return sorted(p.name.removesuffix("_features.parquet")
                  for p in d.glob("*_features.parquet"))


def load_group(
    store: str,
    group: str,
    stimulus_id: Optional[str] = None,
    models: Optional[list[str]] = None,
    embeddings: bool = False,
) -> pd.DataFrame:
    """One feature group as a wide frame: a row per stimulus/coordinate.

    Filters are pushed into the scan rather than applied afterwards — the
    largest group is 237 M rows, and the embedding columns alone are most
    of them.
    """
    import duckdb

    path = _group_path(store, group)
    if not path.exists():
        groups = available_groups(store)
        raise FileNotFoundError(
            f"No feature group {group!r} in {store}. "
            + (f"Available: {', '.join(groups)}" if groups
               else "The store holds no aggregates; run "
                    "`stimfeat_campaign.py aggregate`.")
        )

    where, params = [], []
    if stimulus_id is not None:
        where.append("lower(stimulus_id) = lower(?)")
        params.append(stimulus_id)
    if models:
        where.append(f"model IN ({', '.join('?' for _ in models)})")
        params.extend(models)
    if not embeddings and not models:
        where.append(r"NOT regexp_matches(feature, '_\d+(_\d+)*$')")

    sql = (f"SELECT * FROM read_parquet('{path}') "
           f"WHERE {' AND '.join(where) if where else 'TRUE'}")
    con = duckdb.connect()
    try:
        con.execute("SET enable_progress_bar = false")
        long = con.execute(sql, params).df()
    finally:
        con.close()

    if long.empty:
        raise ValueError(
            f"No rows in group {group!r}"
            + (f" for stimulus {stimulus_id!r}" if stimulus_id else "")
        )

    keys = [c for c in _KEY_COLUMNS if long[c].notna().any()]
    long = long.copy()
    long["_v"] = long["value"].where(long["value"].notna(), long["value_str"])
    wide = long.pivot_table(
        index=keys, columns="feature", values="_v", aggfunc="first"
    ).reset_index()
    wide.columns.name = None
    # The long table carries numbers and strings in one column, so the pivot
    # comes back as object; restore numeric dtype where the whole column is
    # numeric, or _scalar_columns finds nothing to plot.
    for column in wide.columns:
        if column in keys or wide[column].dtype != object:
            continue
        coerced = pd.to_numeric(wide[column], errors="coerce")
        if coerced.notna().sum() == wide[column].notna().sum():
            wide[column] = coerced
    return wide


# ---------------------------------------------------------------------------
# Movie feature timeline
# ---------------------------------------------------------------------------

def plot_movie_feature_timeline(
    store: str,
    movie_name: str,
    group: str = "movies_frames",
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
    store : str
        The psytwill aggregates directory inside the feature store.
    movie_name : str
        Movie stimulus_id (e.g., "adventure-time").
    group : str
        Feature group. Default "movies_frames" (the 0.5 s visual grid);
        "movies_audio_frames" for acoustics.
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
    df = load_group(store, group, stimulus_id=movie_name)

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
    store: str,
    feature: str,
    group: str = "shared1000_image",
    top_n: Optional[int] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    viz2psy_dir: Optional[str] = None,
):
    """Plot a feature value across images, sorted by value.

    Parameters
    ----------
    store : str
        The psytwill aggregates directory inside the feature store.
    group : str
        Feature group. Default "shared1000_image".
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

    df = load_group(store, group)

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
    store: str,
    group: str = "shared1000_image",
    model: str = "clip",
    n_stimuli: Optional[int] = 50,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    viz2psy_dir: Optional[str] = None,
):
    """Plot cosine similarity matrix for stimulus embeddings.

    Parameters
    ----------
    store : str
        The psytwill aggregates directory inside the feature store.
    group : str
        Feature group. Default "shared1000_image".
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

    # Embeddings are excluded by default; naming the model opts in.
    df = load_group(store, group, models=[model])

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
    store: str,
    feature: str,
    group: str = "shared1000_image",
    group_by: Optional[str] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    viz2psy_dir: Optional[str] = None,
):
    """Plot the distribution of a feature across stimuli.

    Parameters
    ----------
    store : str
        The psytwill aggregates directory inside the feature store.
    group : str
        Feature group. Default "shared1000_image".
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

    df = load_group(store, group)

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
    store: str,
    group: str = "shared1000_image",
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
    store : str
        The psytwill aggregates directory inside the feature store.
    group : str
        Feature group. Default "shared1000_image".
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
    # The embedding model, plus whatever `color_by` names -- a scatter
    # coloured by a score needs that score loaded alongside the dimensions.
    wanted = [model] + ([color_by.split("_")[0]] if color_by else [])
    df = load_group(store, group, models=wanted)

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
