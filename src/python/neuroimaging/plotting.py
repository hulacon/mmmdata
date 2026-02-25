"""Neuroimaging visualization functions for MMMData.

All functions produce interactive HTML (plotly) output.
nilearn is used for data extraction (masking, timeseries);
plotly is used for rendering.

Functions accept file paths and return plotly Figure objects.
Each function has a ``save_path`` parameter for persisting output.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def _get_plotly():
    """Lazy import plotly."""
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    return go, px, make_subplots


def _save_html(fig, save_path: Optional[str] = None) -> None:
    """Save plotly figure to HTML if path provided."""
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(save_path))


# ---------------------------------------------------------------------------
# Brain map
# ---------------------------------------------------------------------------

def plot_brain_map(
    stat_map_path: str,
    threshold: Optional[float] = None,
    title: Optional[str] = None,
    colormap: str = "cold_hot",
    save_path: Optional[str] = None,
):
    """Plot a statistical brain map using nilearn's view_img.

    Produces an interactive HTML brain viewer.

    Parameters
    ----------
    stat_map_path : str
        Path to a NIfTI statistical map.
    threshold : float, optional
    title : str, optional
    colormap : str, default "cold_hot"
    save_path : str, optional
        Path to save the HTML output.

    Returns
    -------
    nilearn html viewer object
    """
    from nilearn import plotting

    view = plotting.view_img(
        stat_map_path,
        threshold=threshold,
        title=title or Path(stat_map_path).stem,
        colorbar=True,
        cmap=colormap,
    )

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        view.save_as_html(str(save_path))

    return view


# ---------------------------------------------------------------------------
# BOLD timeseries
# ---------------------------------------------------------------------------

def plot_bold_timeseries(
    bold_path: str,
    roi_name: Optional[str] = None,
    atlas_scale: int = 400,
    atlases_dir: Optional[str] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
):
    """Extract and plot BOLD time series from an ROI.

    Parameters
    ----------
    bold_path : str
        Path to 4D preprocessed BOLD NIfTI.
    roi_name : str, optional
        ROI label from Schaefer atlas. If None, plots global mean.
    atlas_scale : int, default 400
    atlases_dir : str, optional
    title : str, optional
    save_path : str, optional

    Returns
    -------
    plotly.graph_objects.Figure
    """
    import nibabel as nib
    from nilearn.maskers import NiftiLabelsMasker, NiftiMasker

    go, px, make_subplots = _get_plotly()

    bold_img = nib.load(bold_path)
    n_volumes = bold_img.shape[-1]
    tr = bold_img.header.get_zooms()[-1]
    time_points = np.arange(n_volumes) * tr

    if roi_name:
        from .atlas import load_schaefer_atlas, get_roi_index
        atlas_path, labels_df = load_schaefer_atlas(
            n_rois=atlas_scale, atlases_dir=atlases_dir,
        )
        roi_idx = get_roi_index(labels_df, roi_name)
        if roi_idx is None:
            raise ValueError(f"ROI '{roi_name}' not found in Schaefer {atlas_scale}")

        masker = NiftiLabelsMasker(atlas_path, standardize=True)
        all_ts = masker.fit_transform(bold_img)
        # ROI index is 1-based; masker output columns are 0-based
        ts = all_ts[:, roi_idx - 1]
        label = roi_name
    else:
        masker = NiftiMasker(standardize=True)
        all_voxels = masker.fit_transform(bold_img)
        ts = all_voxels.mean(axis=1)
        label = "Global mean"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time_points, y=ts, mode="lines",
        name=label, line=dict(width=1.5),
    ))
    fig.update_layout(
        title=title or f"BOLD Timeseries: {label}",
        xaxis_title="Time (s)",
        yaxis_title="BOLD signal (z-scored)",
        template="plotly_white",
    )

    _save_html(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Motion parameters
# ---------------------------------------------------------------------------

def plot_motion_parameters(
    confounds_path: str,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
):
    """Plot head motion parameters and framewise displacement.

    Parameters
    ----------
    confounds_path : str
        Path to fMRIPrep confounds TSV.
    title : str, optional
    save_path : str, optional

    Returns
    -------
    plotly.graph_objects.Figure
    """
    go, px, make_subplots = _get_plotly()

    df = pd.read_csv(confounds_path, sep="\t")

    # Translation parameters
    trans_cols = [c for c in df.columns if c.startswith("trans_")]
    # Rotation parameters
    rot_cols = [c for c in df.columns if c.startswith("rot_")]
    # Framewise displacement
    has_fd = "framewise_displacement" in df.columns

    n_rows = 2 + (1 if has_fd else 0)
    subtitles = ["Translation (mm)", "Rotation (rad)"]
    if has_fd:
        subtitles.append("Framewise Displacement (mm)")

    fig = make_subplots(
        rows=n_rows, cols=1, shared_xaxes=True,
        subplot_titles=subtitles,
        vertical_spacing=0.08,
    )

    volumes = np.arange(len(df))

    # Translation
    for col in trans_cols:
        fig.add_trace(go.Scatter(
            x=volumes, y=df[col], mode="lines",
            name=col, line=dict(width=1),
        ), row=1, col=1)

    # Rotation
    for col in rot_cols:
        fig.add_trace(go.Scatter(
            x=volumes, y=df[col], mode="lines",
            name=col, line=dict(width=1),
        ), row=2, col=1)

    # FD
    if has_fd:
        fd = df["framewise_displacement"].fillna(0)
        fig.add_trace(go.Scatter(
            x=volumes, y=fd, mode="lines",
            name="FD", line=dict(width=1, color="red"),
        ), row=3, col=1)
        # Threshold line at 0.5mm
        fig.add_hline(y=0.5, line_dash="dash", line_color="gray",
                       annotation_text="0.5mm", row=3, col=1)

    fig.update_layout(
        title=title or f"Motion Parameters: {Path(confounds_path).stem}",
        xaxis_title="Volume",
        height=200 * n_rows + 100,
        template="plotly_white",
        showlegend=True,
    )

    _save_html(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Carpet plot
# ---------------------------------------------------------------------------

def plot_carpet(
    bold_path: str,
    mask_path: Optional[str] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    max_voxels: int = 5000,
):
    """Generate a carpet (grayplot) of BOLD signal.

    Parameters
    ----------
    bold_path : str
        Path to 4D preprocessed BOLD NIfTI.
    mask_path : str, optional
        Path to brain mask. If None, auto-computed.
    title : str, optional
    save_path : str, optional
    max_voxels : int, default 5000
        Subsample to this many voxels for performance.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    import nibabel as nib
    from nilearn.maskers import NiftiMasker

    go, px, make_subplots = _get_plotly()

    masker_kwargs = {"standardize": True}
    if mask_path and Path(mask_path).exists():
        masker_kwargs["mask_img"] = mask_path

    masker = NiftiMasker(**masker_kwargs)
    bold_img = nib.load(bold_path)
    data = masker.fit_transform(bold_img)  # (n_volumes, n_voxels)

    n_volumes, n_voxels = data.shape
    tr = bold_img.header.get_zooms()[-1]

    # Subsample voxels if needed
    if n_voxels > max_voxels:
        idx = np.random.default_rng(42).choice(n_voxels, max_voxels, replace=False)
        idx.sort()
        data = data[:, idx]
        n_voxels = max_voxels

    # Sort voxels by mean signal for cleaner visualization
    sort_idx = np.argsort(data.mean(axis=0))
    data = data[:, sort_idx]

    fig = go.Figure(data=go.Heatmap(
        z=data.T,
        x=np.arange(n_volumes) * tr,
        colorscale="RdBu_r",
        zmin=-3, zmax=3,
        colorbar=dict(title="z-score"),
    ))

    fig.update_layout(
        title=title or f"Carpet Plot: {Path(bold_path).stem}",
        xaxis_title="Time (s)",
        yaxis_title=f"Voxels ({n_voxels})",
        template="plotly_white",
        height=600,
    )

    _save_html(fig, save_path)
    return fig
