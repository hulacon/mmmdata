"""Pipeline harness: step contracts, validators, status, orchestration.

This package formalizes the MMMData preprocessing pipeline as a DAG of
steps, each with its own postcondition validator. The registry in
``steps.py`` is the single source of truth for step dependencies and
resource requirements; ``validators.py`` checks that outputs exist and
are well-formed; ``status.py`` reports project-wide progress.

Typical usage::

    from pipeline.status import pipeline_status, runnable_sessions

    df = pipeline_status(subjects=["03"])        # full status table
    ready = runnable_sessions("fmriprep_nordic") # (sub, ses) pairs to submit
"""
