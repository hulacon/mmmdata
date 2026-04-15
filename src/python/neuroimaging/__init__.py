"""Neuroimaging data access, QC, visualization, and analysis utilities.

Submodules
----------
io
    Discover and load preprocessed fMRIPrep outputs (``FmriprepRun``,
    ``find_fmriprep_runs``, ``load_confounds``, ``load_bold``, ``load_mask``).
constants
    Path templates, confound column groups, task-to-stream mappings.
qc
    MRIQC and fMRIPrep QC metrics, outlier detection, motion summaries.
plotting
    Plotly-based visualization helpers (brain maps, carpet plots, motion).
atlas
    Schaefer parcellation loader.
"""
