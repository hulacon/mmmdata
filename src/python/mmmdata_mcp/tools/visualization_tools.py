"""Visualization tools for the MMMData MCP server.

Generates interactive HTML (plotly) figures and saves them to the
configured figures directory (``[mcp] figures_dir``), never into the
dataset. Tools are thin wrappers: resolve paths/config, delegate to the
mmmdata libraries, and return JSON.

Three domains:
- Behavioral: delegates full load -> analyze -> plot chains to
  mmmdata's behavioral.reports module
- Neuroimaging: brain maps, timeseries, motion QC via nilearn;
  fMRIPrep paths built by mmmdata's neuroimaging.fmriprep_layout
- Stimulus: viz2psy feature timelines, similarity matrices
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

from pydantic import Field

from ..paths import get_paths


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _timestamp() -> str:
    """Short timestamp for default filenames."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _ensure_dir(subdir: str) -> Path:
    """Create and return a subdirectory under figures_dir."""
    d = get_paths().figures_dir / subdir
    d.mkdir(parents=True, exist_ok=True)
    return d


def _result(file_path: str, plot_type: str, description: str,
            **extra: Any) -> str:
    """Standard JSON response for all visualization tools."""
    return json.dumps({
        "file_path": file_path,
        "plot_type": plot_type,
        "description": description,
        **extra,
    })


# ===================================================================
# BEHAVIORAL VISUALIZATION TOOLS
# ===================================================================

def plot_accuracy_by_condition(
    subjects: Annotated[
        list[str] | None,
        Field(description="Subject IDs (e.g., ['##', '##']). Default: all."),
    ] = None,
    sessions: Annotated[
        list[str] | None,
        Field(description="Session IDs (e.g., ['##', '##']). Default: all TB sessions."),
    ] = None,
    group_by: Annotated[
        list[str] | None,
        Field(description="Columns to group by (e.g., ['subject', 'enCon', 'reCon']). Default: ['subject', 'enCon']."),
    ] = None,
    title: Annotated[
        str | None, Field(description="Custom plot title.")
    ] = None,
    filename: Annotated[
        str | None, Field(description="Custom output filename.")
    ] = None,
) -> str:
    """Plot recognition accuracy by experimental condition (encoding
    condition, retrieval condition). Returns an interactive HTML bar
    plot with per-subject overlays.
    """
    from behavioral import reports

    paths = get_paths()
    out_dir = _ensure_dir("behavioral")
    fname = filename or f"accuracy_by_condition_{_timestamp()}.html"
    save_path = out_dir / fname

    try:
        res = reports.accuracy_by_condition_report(
            save_path, bids_root=paths.bids_root, subjects=subjects,
            sessions=sessions, group_by=group_by, title=title,
        )
    except reports.ReportDataError as exc:
        return json.dumps({"error": str(exc)})

    return _result(
        str(save_path), "accuracy_by_condition",
        f"Accuracy by condition for {res['n_subjects']} subjects ({res['n_trials']} trials)",
        n_subjects=res["n_subjects"], n_trials=res["n_trials"],
    )


def plot_learning_curve(
    subjects: Annotated[
        list[str] | None,
        Field(description="Subject IDs. Default: all."),
    ] = None,
    sessions: Annotated[
        list[str] | None,
        Field(description="Session IDs. Default: all TB sessions."),
    ] = None,
    metric: Annotated[
        str,
        Field(description="Column to track (e.g., 'trial_accuracy', 'resp_RT')."),
    ] = "trial_accuracy",
    condition: Annotated[
        str | None,
        Field(description="Condition column to split by (e.g., 'enCon')."),
    ] = None,
    title: str | None = None,
    filename: str | None = None,
) -> str:
    """Plot a behavioral metric across sessions as a learning curve.
    Shows individual subjects and group mean with error ribbon.
    """
    from behavioral import reports

    paths = get_paths()
    out_dir = _ensure_dir("behavioral")
    fname = filename or f"learning_curve_{metric}_{_timestamp()}.html"
    save_path = out_dir / fname

    try:
        res = reports.learning_curve_report(
            save_path, bids_root=paths.bids_root, subjects=subjects,
            sessions=sessions, metric=metric, condition=condition,
            title=title,
        )
    except reports.ReportDataError as exc:
        return json.dumps({"error": str(exc)})

    return _result(
        str(save_path), "learning_curve",
        f"Learning curve ({metric}) across {res['n_sessions']} sessions",
        metric=metric, n_sessions=res["n_sessions"],
    )


def plot_rt_distribution(
    subjects: list[str] | None = None,
    sessions: list[str] | None = None,
    group_by: Annotated[
        str | None,
        Field(description="Column to group by (e.g., 'subject', 'enCon', 'trial_accuracy')."),
    ] = None,
    kind: Annotated[
        str,
        Field(json_schema_extra={"enum": ["histogram", "violin"]}),
    ] = "histogram",
    title: str | None = None,
    filename: str | None = None,
) -> str:
    """Plot reaction time distribution as histogram or violin plot.
    Can be grouped by condition, subject, or accuracy.
    """
    from behavioral import reports

    paths = get_paths()
    out_dir = _ensure_dir("behavioral")
    fname = filename or f"rt_distribution_{_timestamp()}.html"
    save_path = out_dir / fname

    try:
        res = reports.rt_distribution_report(
            save_path, bids_root=paths.bids_root, subjects=subjects,
            sessions=sessions, group_by=group_by, kind=kind, title=title,
        )
    except reports.ReportDataError as exc:
        return json.dumps({"error": str(exc)})

    return _result(
        str(save_path), "rt_distribution",
        f"RT distribution ({kind}) for {res['n_subjects']} subjects",
        kind=kind, n_trials=res["n_trials"],
    )


def plot_dprime_curve(
    subjects: list[str] | None = None,
    sessions: list[str] | None = None,
    condition: Annotated[
        str | None,
        Field(description="Condition to split by (e.g., 'enCon')."),
    ] = None,
    title: str | None = None,
    filename: str | None = None,
) -> str:
    """Plot d-prime (signal detection sensitivity) across sessions.
    Shows discrimination ability over the course of the experiment.
    """
    from behavioral import reports

    paths = get_paths()
    out_dir = _ensure_dir("behavioral")
    fname = filename or f"dprime_curve_{_timestamp()}.html"
    save_path = out_dir / fname

    try:
        res = reports.dprime_curve_report(
            save_path, bids_root=paths.bids_root, subjects=subjects,
            sessions=sessions, condition=condition, title=title,
        )
    except reports.ReportDataError as exc:
        return json.dumps({"error": str(exc)})

    return _result(
        str(save_path), "dprime_curve",
        f"d' across {res['n_sessions']} sessions",
        n_sessions=res["n_sessions"],
    )


def plot_subsequent_memory(
    subjects: list[str] | None = None,
    group_by: Annotated[
        list[str] | None,
        Field(description="Grouping (e.g., ['subject', 'enCon']). Default: ['subject']."),
    ] = None,
    title: str | None = None,
    filename: str | None = None,
) -> str:
    """Plot subsequent memory effect: how encoding quality ratings
    predict later recognition accuracy.
    """
    from behavioral import reports

    paths = get_paths()
    out_dir = _ensure_dir("behavioral")
    fname = filename or f"subsequent_memory_{_timestamp()}.html"
    save_path = out_dir / fname

    try:
        res = reports.subsequent_memory_report(
            save_path, bids_root=paths.bids_root, subjects=subjects,
            group_by=group_by, title=title,
        )
    except reports.ReportDataError as exc:
        return json.dumps({"error": str(exc)})

    return _result(
        str(save_path), "subsequent_memory",
        f"Subsequent memory effect for {res['n_subjects']} subjects",
    )


def plot_confidence_accuracy(
    subjects: list[str] | None = None,
    sessions: list[str] | None = None,
    title: str | None = None,
    filename: str | None = None,
) -> str:
    """Plot confidence-accuracy calibration: accuracy at each
    confidence level (sure vs maybe). Shows per-subject curves.
    """
    from behavioral import reports

    paths = get_paths()
    out_dir = _ensure_dir("behavioral")
    fname = filename or f"confidence_accuracy_{_timestamp()}.html"
    save_path = out_dir / fname

    try:
        res = reports.confidence_accuracy_report(
            save_path, bids_root=paths.bids_root, subjects=subjects,
            sessions=sessions, title=title,
        )
    except reports.ReportDataError as exc:
        return json.dumps({"error": str(exc)})

    return _result(
        str(save_path), "confidence_accuracy",
        f"Confidence-accuracy calibration for {res['n_subjects']} subjects",
    )


def plot_fin_comparison(
    subjects: list[str] | None = None,
    group_by: Annotated[
        list[str] | None,
        Field(description="Condition grouping (e.g., ['enCon'])."),
    ] = None,
    title: str | None = None,
    filename: str | None = None,
) -> str:
    """Compare session-by-session recognition accuracy with final
    session (ses-30) accuracy. Shows retention over time.
    """
    from behavioral import reports

    paths = get_paths()
    out_dir = _ensure_dir("behavioral")
    fname = filename or f"fin_comparison_{_timestamp()}.html"
    save_path = out_dir / fname

    try:
        res = reports.fin_comparison_report(
            save_path, bids_root=paths.bids_root, subjects=subjects,
            group_by=group_by, title=title,
        )
    except reports.ReportDataError as exc:
        return json.dumps({"error": str(exc)})

    return _result(
        str(save_path), "fin_comparison",
        f"Session vs final accuracy for {res['n_subjects']} subjects",
    )


def plot_timeline_responses(
    subjects: list[str] | None = None,
    group_by: Annotated[
        str | None, Field(description="Column to group by.")
    ] = "subject",
    title: str | None = None,
    filename: str | None = None,
) -> str:
    """Plot distribution of temporal judgment responses from the
    FINtimeline task (ses-30). Shows how accurately participants
    place items on a study timeline.
    """
    from behavioral import reports

    paths = get_paths()
    out_dir = _ensure_dir("behavioral")
    fname = filename or f"timeline_responses_{_timestamp()}.html"
    save_path = out_dir / fname

    try:
        res = reports.timeline_responses_report(
            save_path, bids_root=paths.bids_root, subjects=subjects,
            group_by=group_by, title=title,
        )
    except reports.ReportDataError as exc:
        return json.dumps({"error": str(exc)})

    return _result(
        str(save_path), "timeline_responses",
        f"Timeline temporal judgments for {res['n_subjects']} subjects",
        n_trials=res["n_trials"],
    )


# ===================================================================
# NEUROIMAGING VISUALIZATION TOOLS
# ===================================================================

def plot_brain_map(
    stat_map_path: Annotated[
        str, Field(description="Path to NIfTI statistical map file.")
    ],
    threshold: Annotated[
        float | None,
        Field(description="Display threshold for the stat map."),
    ] = None,
    title: str | None = None,
    colormap: Annotated[
        str,
        Field(description="Colormap name (e.g., 'cold_hot', 'RdBu_r')."),
    ] = "cold_hot",
    filename: str | None = None,
) -> str:
    """Plot a statistical brain map (NIfTI) overlaid on a standard
    anatomical template. Produces an interactive glass-brain or
    slice view.
    """
    from neuroimaging.plotting import plot_brain_map as _plot

    out_dir = _ensure_dir("neuroimaging")
    fname = filename or f"brain_map_{_timestamp()}.html"
    save_path = out_dir / fname

    _plot(stat_map_path, threshold=threshold, title=title,
          colormap=colormap, save_path=str(save_path))

    return _result(
        str(save_path), "brain_map",
        f"Brain map: {Path(stat_map_path).name}",
    )


def plot_bold_timeseries(
    subject: Annotated[str, Field(description="Subject ID (e.g., '##').")],
    session: Annotated[str, Field(description="Session ID (e.g., '13').")],
    task: Annotated[str, Field(description="Task name (e.g., 'TBencoding').")],
    run: str = "01",
    roi_name: Annotated[
        str | None,
        Field(description="Schaefer parcel: its full CBIG name or the tail from an underscore (e.g., 'LH_Vis_1'); a name matching both hemispheres is an error. If omitted, plots global mean."),
    ] = None,
    atlas_scale: Annotated[
        int,
        Field(description="Number of ROIs in Schaefer parcellation."),
    ] = 400,
    variant: Annotated[
        str,
        Field(
            description="fMRIPrep variant: standard or NORDIC-denoised.",
            json_schema_extra={"enum": ["standard", "nordic"]},
        ),
    ] = "standard",
    title: str | None = None,
    filename: str | None = None,
) -> str:
    """Extract and plot BOLD time series from an ROI using the
    Schaefer atlas. Shows signal over time with optional events overlay.
    """
    from neuroimaging import fmriprep_layout
    from neuroimaging.plotting import plot_bold_timeseries as _plot

    paths = get_paths()
    bold_path = fmriprep_layout.bold_path(
        paths.fmriprep_dir_for(variant), subject, session, task, run=run
    )
    if not bold_path.exists():
        return json.dumps({"error": f"BOLD file not found: {bold_path}"})

    out_dir = _ensure_dir("neuroimaging")
    fname = filename or f"bold_ts_sub-{subject}_ses-{session}_{task}_{_timestamp()}.html"
    save_path = out_dir / fname

    _plot(
        str(bold_path), roi_name=roi_name, atlas_scale=atlas_scale,
        atlases_dir=str(paths.atlases_dir), title=title,
        save_path=str(save_path),
    )

    roi_label = roi_name or "global mean"
    return _result(
        str(save_path), "bold_timeseries",
        f"BOLD timeseries ({roi_label}) for sub-{subject} ses-{session} {task} run-{run}",
    )


def plot_motion_qc(
    subject: str,
    session: str,
    task: str,
    run: str = "01",
    variant: Annotated[
        str,
        Field(
            description="fMRIPrep variant: standard or NORDIC-denoised.",
            json_schema_extra={"enum": ["standard", "nordic"]},
        ),
    ] = "standard",
    title: str | None = None,
    filename: str | None = None,
) -> str:
    """Plot head motion parameters (6 rigid-body) and framewise
    displacement from fMRIPrep confounds. Useful for data quality inspection.
    """
    from neuroimaging import fmriprep_layout
    from neuroimaging.plotting import plot_motion_parameters as _plot

    paths = get_paths()
    confounds_path = fmriprep_layout.confounds_path(
        paths.fmriprep_dir_for(variant), subject, session, task, run=run
    )
    if not confounds_path.exists():
        return json.dumps({"error": f"Confounds file not found: {confounds_path}"})

    out_dir = _ensure_dir("neuroimaging")
    fname = filename or f"motion_sub-{subject}_ses-{session}_{task}_{_timestamp()}.html"
    save_path = out_dir / fname

    _plot(str(confounds_path), title=title, save_path=str(save_path))

    return _result(
        str(save_path), "motion_qc",
        f"Motion parameters for sub-{subject} ses-{session} {task} run-{run}",
    )


def plot_carpet(
    subject: str,
    session: str,
    task: str,
    run: str = "01",
    variant: Annotated[
        str,
        Field(
            description="fMRIPrep variant: standard or NORDIC-denoised.",
            json_schema_extra={"enum": ["standard", "nordic"]},
        ),
    ] = "standard",
    title: str | None = None,
    filename: str | None = None,
) -> str:
    """Generate a carpet plot (grayplot) showing BOLD signal intensity
    for all brain voxels over time. Useful for spotting motion artifacts
    and global signal fluctuations.
    """
    from neuroimaging import fmriprep_layout
    from neuroimaging.plotting import plot_carpet as _plot

    paths = get_paths()
    fmriprep_dir = paths.fmriprep_dir_for(variant)
    bold_path = fmriprep_layout.bold_path(
        fmriprep_dir, subject, session, task, run=run
    )
    mask_path = fmriprep_layout.mask_path(
        fmriprep_dir, subject, session, task, run=run
    )

    if not bold_path.exists():
        return json.dumps({"error": f"BOLD file not found: {bold_path}"})

    out_dir = _ensure_dir("neuroimaging")
    fname = filename or f"carpet_sub-{subject}_ses-{session}_{task}_{_timestamp()}.html"
    save_path = out_dir / fname

    _plot(
        str(bold_path),
        mask_path=str(mask_path) if mask_path.exists() else None,
        title=title, save_path=str(save_path),
    )

    return _result(
        str(save_path), "carpet_plot",
        f"Carpet plot for sub-{subject} ses-{session} {task} run-{run}",
    )


# ===================================================================
# STIMULUS FEATURE VISUALIZATION TOOLS
# ===================================================================

def plot_movie_features(
    movie_name: Annotated[
        str,
        Field(description="Movie stimulus_id (e.g., 'adventure-time')."),
    ],
    group: Annotated[
        str,
        Field(description=(
            "Contract B feature group. Default 'movies_frames' (the 0.5 s "
            "visual grid); 'movies_audio_frames' for acoustics."
        )),
    ] = "movies_frames",
    features: Annotated[
        list[str] | None,
        Field(description="Feature names to plot. Default: all scalar features."),
    ] = None,
    time_range: Annotated[
        list[float] | None,
        Field(description="[start, end] in seconds to zoom into."),
    ] = None,
    title: str | None = None,
    filename: str | None = None,
) -> str:
    """Plot feature scores over time for a movie stimulus.
    Shows how features like emotion, memorability, saliency vary
    across frames.
    """
    from stimuli.plotting import plot_movie_feature_timeline as _plot

    paths = get_paths()
    store = paths.stimuli_features_tables_dir

    out_dir = _ensure_dir("stimuli")
    fname = filename or f"movie_features_{movie_name}_{_timestamp()}.html"
    save_path = out_dir / fname

    _plot(
        str(store), movie_name, group=group, features=features,
        time_range=time_range, title=title, save_path=str(save_path),
        viz2psy_dir=paths.viz2psy_dir,
    )

    return _result(
        str(save_path), "movie_features",
        f"Feature timeline for movie: {movie_name}",
        movie=movie_name,
    )


def plot_image_features(
    feature: Annotated[
        str,
        Field(description="Feature to plot (e.g., 'memorability', 'valence')."),
    ],
    group: Annotated[
        str,
        Field(description=(
            "Contract B feature group. Default 'shared1000_image'; see "
            "list_stimulus_sets for the full set, including twp1000, audio, "
            "transcript, caption, and annotation groups."
        )),
    ] = "shared1000_image",
    top_n: Annotated[
        int | None,
        Field(description="Show only top N images by feature value."),
    ] = None,
    title: str | None = None,
    filename: str | None = None,
) -> str:
    """Plot a feature value (e.g., memorability, valence) across
    images in a stimulus set. Optionally show only top N.
    """
    from stimuli.plotting import plot_image_feature_comparison as _plot

    paths = get_paths()
    store = paths.stimuli_features_tables_dir

    out_dir = _ensure_dir("stimuli")
    fname = filename or f"image_features_{feature}_{_timestamp()}.html"
    save_path = out_dir / fname

    _plot(
        str(store), feature, group=group, top_n=top_n,
        title=title, save_path=str(save_path),
    )

    return _result(
        str(save_path), "image_features",
        f"{feature} across {group}",
        feature=feature, group=group,
    )


def plot_stimulus_similarity(
    group: Annotated[
        str,
        Field(description=(
            "Contract B feature group. Default 'shared1000_image'; see "
            "list_stimulus_sets for the full set, including twp1000, audio, "
            "transcript, caption, and annotation groups."
        )),
    ] = "shared1000_image",
    model: Annotated[
        str,
        Field(description="Embedding model to use.",
              json_schema_extra={"enum": ["clip", "dinov2"]}),
    ] = "clip",
    n_stimuli: Annotated[
        int | None,
        Field(description="Number of stimuli to include (for readability)."),
    ] = 50,
    title: str | None = None,
    filename: str | None = None,
) -> str:
    """Plot cosine similarity matrix for stimulus embeddings
    (CLIP or DINOv2). Shows how similar stimuli are to each other
    in representational space.
    """
    from stimuli.plotting import plot_feature_similarity_matrix as _plot

    paths = get_paths()
    store = paths.stimuli_features_tables_dir

    out_dir = _ensure_dir("stimuli")
    fname = filename or f"similarity_{model}_{_timestamp()}.html"
    save_path = out_dir / fname

    _plot(
        str(store), group=group, model=model, n_stimuli=n_stimuli,
        title=title, save_path=str(save_path),
    )

    return _result(
        str(save_path), "stimulus_similarity",
        f"{model.upper()} similarity matrix ({group})",
        model=model, group=group,
    )


def plot_feature_distribution(
    feature: Annotated[
        str,
        Field(description="Feature name (e.g., 'memorability', 'valence')."),
    ],
    group: Annotated[
        str,
        Field(description=(
            "Contract B feature group. Default 'shared1000_image'; see "
            "list_stimulus_sets for the full set, including twp1000, audio, "
            "transcript, caption, and annotation groups."
        )),
    ] = "shared1000_image",
    group_by: Annotated[
        str | None,
        Field(description="Column to group by (e.g., 'category')."),
    ] = None,
    title: str | None = None,
    filename: str | None = None,
) -> str:
    """Plot the distribution of a feature across a stimulus set.
    Shows histogram/density of values.
    """
    from stimuli.plotting import plot_feature_distribution as _plot

    paths = get_paths()
    store = paths.stimuli_features_tables_dir

    out_dir = _ensure_dir("stimuli")
    fname = filename or f"feature_dist_{feature}_{_timestamp()}.html"
    save_path = out_dir / fname

    _plot(
        str(store), feature, group=group, group_by=group_by,
        title=title, save_path=str(save_path),
    )

    return _result(
        str(save_path), "feature_distribution",
        f"Distribution of {feature} in {group}",
        feature=feature, group=group,
    )


def plot_embedding_scatter(
    group: Annotated[
        str,
        Field(description=(
            "Contract B feature group. Default 'shared1000_image'; see "
            "list_stimulus_sets for the full set, including twp1000, audio, "
            "transcript, caption, and annotation groups."
        )),
    ] = "shared1000_image",
    model: Annotated[
        str,
        Field(json_schema_extra={"enum": ["clip", "dinov2"]}),
    ] = "clip",
    method: Annotated[
        str,
        Field(description="Dimensionality reduction method.",
              json_schema_extra={"enum": ["pca", "umap", "tsne", "mds"]}),
    ] = "pca",
    color_by: Annotated[
        str | None,
        Field(description="Column to color points by (e.g., 'memorability')."),
    ] = None,
    n_stimuli: Annotated[
        int | None, Field(description="Subsample to N stimuli.")
    ] = None,
    title: str | None = None,
    filename: str | None = None,
) -> str:
    """Plot 2D scatter of stimulus embeddings using dimensionality
    reduction (PCA, UMAP, t-SNE, MDS). Leverages viz2psy's
    interactive scatter. Color points by any feature.
    """
    from stimuli.plotting import plot_embedding_scatter as _plot

    paths = get_paths()
    store = paths.stimuli_features_tables_dir

    out_dir = _ensure_dir("stimuli")
    fname = filename or f"scatter_{model}_{method}_{_timestamp()}.html"
    save_path = out_dir / fname

    _plot(
        str(store), group=group, model=model, method=method,
        color_by=color_by, n_stimuli=n_stimuli,
        title=title, save_path=str(save_path),
        viz2psy_dir=paths.viz2psy_dir,
    )

    return _result(
        str(save_path), "embedding_scatter",
        f"{model.upper()} embeddings ({method.upper()}) for {group}",
        model=model, method=method, group=group,
    )


TOOL_FUNCTIONS = [
    # Behavioral
    plot_accuracy_by_condition,
    plot_learning_curve,
    plot_rt_distribution,
    plot_dprime_curve,
    plot_subsequent_memory,
    plot_confidence_accuracy,
    plot_fin_comparison,
    plot_timeline_responses,
    # Neuroimaging
    plot_brain_map,
    plot_bold_timeseries,
    plot_motion_qc,
    plot_carpet,
    # Stimulus features
    plot_movie_features,
    plot_image_features,
    plot_stimulus_similarity,
    plot_feature_distribution,
    plot_embedding_scatter,
]
