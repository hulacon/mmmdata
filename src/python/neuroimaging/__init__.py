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
glm
    Condition-level GLM: BIDS Stats Models specs, design construction, the
    estimator interface (nilearn first), fixed effects, Contract A-keyed
    outputs, split-half reliability.
"""
