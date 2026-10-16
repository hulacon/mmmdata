"""Quality-control tools for the MMMData MCP server.

Thin wrappers around neuroimaging.qc and neuroimaging.qc_dashboard
that resolve paths, call the library functions, and serialize results
to JSON.

Two lists: ``TOOL_FUNCTIONS`` only read, and back the default ``qc``
toolset. ``WRITE_TOOL_FUNCTIONS`` write into the dataset tree (the
dashboard and the decision store both live under the BIDS root) and
back the opt-in ``qc_write`` toolset.
"""

import json
from typing import Annotated

from pydantic import Field

from ..paths import get_paths


def _import_qc():
    from neuroimaging import qc
    return qc


def _import_qc_dashboard():
    from neuroimaging import qc_dashboard
    return qc_dashboard


# ---------------------------------------------------------------------------
# Tool functions
# ---------------------------------------------------------------------------

def list_qc_reports(
    pipeline: Annotated[
        str,
        Field(
            description="Which pipeline's reports to list.",
            json_schema_extra={"enum": ["mriqc", "fmriprep", "both"]},
        ),
    ] = "mriqc",
    variant: Annotated[
        str,
        Field(
            description="fMRIPrep variant: standard or NORDIC-denoised.",
            json_schema_extra={"enum": ["standard", "nordic"]},
        ),
    ] = "standard",
) -> str:
    """List available QC HTML reports from MRIQC and/or fMRIPrep.
    Returns report filenames with parsed BIDS entities and
    summary counts by subject and modality.
    """
    qc = _import_qc()
    paths = get_paths()
    mriqc_dir = paths.mriqc_dir if pipeline in ("mriqc", "both") else None
    fmriprep_dir = (
        paths.fmriprep_dir_for(variant) if pipeline in ("fmriprep", "both") else None
    )
    result = qc.list_reports(mriqc_dir=mriqc_dir, fmriprep_dir=fmriprep_dir)
    return json.dumps(result, default=str)


def get_iqm_metrics(
    modality: Annotated[
        str,
        Field(
            description="Modality to query.",
            json_schema_extra={"enum": ["bold", "T1w", "T2w", "dwi"]},
        ),
    ] = "bold",
    subject: Annotated[
        str | None,
        Field(description="Subject ID without 'sub-' prefix. Default: all."),
    ] = None,
    session: Annotated[
        str | None,
        Field(description="Session ID without 'ses-' prefix. Default: all."),
    ] = None,
    metrics: Annotated[
        list[str] | None,
        Field(
            description=(
                "Specific IQM keys (e.g. ['fd_mean', 'tsnr']). "
                "Default: key metrics for the modality."
            ),
        ),
    ] = None,
) -> str:
    """Read MRIQC image quality metrics (IQMs) for individual runs.
    Returns a table of IQMs with BIDS entities
    (subject, session, task, run).
    """
    qc = _import_qc()
    table = qc.get_iqm_table(
        get_paths().mriqc_dir, modality,
        subject=subject, session=session, metrics=metrics,
    )
    return json.dumps({
        "modality": modality,
        "n_runs": len(table),
        "data": table,
    }, default=str)


def aggregate_iqms(
    modality: Annotated[
        str,
        Field(
            description="Modality to query.",
            json_schema_extra={"enum": ["bold", "T1w", "T2w", "dwi"]},
        ),
    ] = "bold",
    group_by: Annotated[
        str,
        Field(
            description=(
                "'subject': compare across subjects; "
                "'session': compare across sessions; "
                "'task': compare across tasks; "
                "'global': pool all runs."
            ),
            json_schema_extra={"enum": ["subject", "session", "task", "global"]},
        ),
    ] = "subject",
    metrics: Annotated[
        list[str] | None,
        Field(description="Specific IQM keys. Default: key metrics for modality."),
    ] = None,
    subject: Annotated[
        str | None,
        Field(
            description=(
                "Restrict to one subject (e.g. '##'). Useful with "
                "group_by='session' for longitudinal QC."
            ),
        ),
    ] = None,
) -> str:
    """Compute summary statistics (mean, std, median, min, max, quartiles)
    of MRIQC IQMs grouped by subject, session, task, or globally.
    Useful for comparing quality across subjects or tracking
    longitudinal trends within a densely-sampled subject.
    """
    qc = _import_qc()
    result = qc.aggregate_iqms(
        get_paths().mriqc_dir, modality,
        group_by=group_by, metrics=metrics, subject=subject,
    )
    return json.dumps(result, default=str)


def detect_iqm_outliers(
    modality: Annotated[
        str,
        Field(
            description="Modality to check.",
            json_schema_extra={"enum": ["bold", "T1w", "T2w", "dwi"]},
        ),
    ] = "bold",
    scope: Annotated[
        str,
        Field(
            description=(
                "'global': outliers relative to all runs. "
                "'within_subject': outliers relative to each "
                "subject's own runs."
            ),
            json_schema_extra={"enum": ["global", "within_subject"]},
        ),
    ] = "global",
    subject: Annotated[
        str | None,
        Field(description="Subject ID without 'sub-' prefix."),
    ] = None,
    metrics: Annotated[
        list[str] | None,
        Field(description="Specific IQM keys. Default: key metrics for modality."),
    ] = None,
    iqr_multiplier: Annotated[
        float | None,
        Field(
            description=(
                "IQR multiplier for the fence "
                "(1.5 = standard, 3.0 = extreme only). "
                "Defaults to the [qc] setting in base.toml."
            ),
        ),
    ] = None,
) -> str:
    """Flag runs whose MRIQC IQMs fall outside the IQR-based fence.
    Supports global detection (across all subjects) and
    within-subject detection (relative to each subject's own
    distribution). Reports flagged metrics, values, and thresholds.
    """
    qc = _import_qc()
    result = qc.detect_outliers(
        get_paths().mriqc_dir, modality,
        scope=scope, subject=subject, metrics=metrics,
        iqr_multiplier=iqr_multiplier,
    )
    return json.dumps(result, default=str)


def get_motion_summary(
    subject: Annotated[
        str | None,
        Field(description="Subject ID without 'sub-' prefix. Default: all."),
    ] = None,
    session: Annotated[
        str | None,
        Field(description="Session ID without 'ses-' prefix. Default: all."),
    ] = None,
    task: Annotated[
        str | None,
        Field(description="Task label (e.g. 'TBencoding'). Default: all."),
    ] = None,
    fd_threshold: Annotated[
        float | None,
        Field(
            description=(
                "FD threshold in mm for counting high-motion "
                "volumes. Defaults to the [qc] setting in base.toml."
            ),
        ),
    ] = None,
    variant: Annotated[
        str,
        Field(
            description="fMRIPrep variant: standard or NORDIC-denoised.",
            json_schema_extra={"enum": ["standard", "nordic"]},
        ),
    ] = "standard",
) -> str:
    """Summarize fMRIPrep motion confounds across BOLD runs.
    Reports mean/median/max framewise displacement, percent
    of high-motion volumes, DVARS, and RMSD per run.
    """
    qc = _import_qc()
    result = qc.summarize_motion(
        get_paths().fmriprep_dir_for(variant),
        subject=subject, session=session, task=task,
        fd_threshold=fd_threshold,
    )
    return json.dumps(result, default=str)


def get_processing_status(
    subject: Annotated[
        str | None,
        Field(description="Subject ID without 'sub-' prefix. Default: all."),
    ] = None,
    pipeline: Annotated[
        str,
        Field(
            description="Which pipeline(s) to check.",
            json_schema_extra={"enum": ["mriqc", "fmriprep", "both"]},
        ),
    ] = "both",
    variant: Annotated[
        str,
        Field(
            description="fMRIPrep variant: standard or NORDIC-denoised.",
            json_schema_extra={"enum": ["standard", "nordic"]},
        ),
    ] = "standard",
) -> str:
    """Compare raw BIDS data against MRIQC/fMRIPrep derivatives.
    Shows which sessions and runs have been processed and
    which are still pending, with completion percentages.
    """
    qc = _import_qc()
    paths = get_paths()
    result = qc.processing_status(
        paths.bids_root, paths.mriqc_dir, paths.fmriprep_dir_for(variant),
        subject=subject, pipeline=pipeline,
    )
    return json.dumps(result, default=str)


def get_run_details(
    subject: Annotated[
        str,
        Field(description="Subject ID without 'sub-' prefix."),
    ],
    session: Annotated[
        str,
        Field(description="Session ID without 'ses-' prefix."),
    ],
    task: Annotated[
        str,
        Field(description="Task label (e.g. 'TBencoding')."),
    ],
    run: Annotated[
        str | None,
        Field(description="Run number (e.g. '##'). Omit if task has no runs."),
    ] = None,
    suffix: Annotated[
        str,
        Field(
            description="Modality suffix.",
            json_schema_extra={"enum": ["bold", "T1w", "T2w", "dwi"]},
        ),
    ] = "bold",
    variant: Annotated[
        str,
        Field(
            description="fMRIPrep variant: standard or NORDIC-denoised.",
            json_schema_extra={"enum": ["standard", "nordic"]},
        ),
    ] = "standard",
) -> str:
    """Get full MRIQC IQMs and fMRIPrep motion summary for a
    specific run. Returns all available metrics (not just key
    ones). Use to deep-dive into a flagged or suspicious run.
    """
    qc = _import_qc()
    paths = get_paths()
    result = qc.run_details(
        paths.mriqc_dir, paths.fmriprep_dir_for(variant),
        subject=subject, session=session, task=task,
        run=run, suffix=suffix,
    )
    return json.dumps(result, default=str)


def generate_qc_dashboard(
    subject: Annotated[
        str | None,
        Field(description="Subject ID without 'sub-' prefix. Default: all subjects."),
    ] = None,
    modality: Annotated[
        str,
        Field(
            description="Modality to include in dashboard.",
            json_schema_extra={"enum": ["bold", "T1w", "T2w", "dwi"]},
        ),
    ] = "bold",
    iqr_multiplier: Annotated[
        float | None,
        Field(
            description=(
                "IQR multiplier for outlier detection. "
                "Defaults to the [qc] setting in base.toml."
            ),
        ),
    ] = None,
    fd_threshold: Annotated[
        float | None,
        Field(
            description=(
                "FD threshold in mm for motion flagging. "
                "Defaults to the [qc] setting in base.toml."
            ),
        ),
    ] = None,
    variant: Annotated[
        str,
        Field(
            description="fMRIPrep variant: standard or NORDIC-denoised.",
            json_schema_extra={"enum": ["standard", "nordic"]},
        ),
    ] = "standard",
) -> str:
    """Generate an interactive HTML QC dashboard showing processing
    status (MRIQC/fMRIPrep completion), per-subject QC summaries,
    per-run IQMs, outlier flags, motion summaries, and existing
    QC decisions. Produces a self-contained HTML file with sortable
    tables and Plotly charts. Open in VS Code Simple Browser or
    serve locally.
    """
    qc_dash = _import_qc_dashboard()
    paths = get_paths()

    subject_tag = f"sub-{subject}" if subject else "all"
    result_path = qc_dash.generate_review_dashboard(
        paths.bids_root,
        mriqc_dir=paths.mriqc_dir,
        fmriprep_dir=paths.fmriprep_dir_for(variant),
        subject=subject,
        modality=modality,
        iqr_multiplier=iqr_multiplier,
        fd_threshold=fd_threshold,
    )
    out_dir = qc_dash.dashboards_dir(paths.bids_root)

    return json.dumps({
        "file_path": str(result_path),
        "description": f"QC dashboard for {subject_tag} ({modality}, {variant} fMRIPrep)",
        "view_instructions": (
            "Open in VS Code: Ctrl+Shift+P > 'Simple Browser: Show' > "
            f"paste the file path. Or serve locally: python -m http.server 8080 "
            f"--directory {out_dir}"
        ),
    })


def save_qc_decision(
    subject: Annotated[
        str,
        Field(description="Subject ID without 'sub-' prefix."),
    ],
    session: Annotated[
        str,
        Field(description="Session ID without 'ses-' prefix."),
    ],
    task: Annotated[
        str,
        Field(description="Task label (e.g. 'TBencoding')."),
    ],
    decision: Annotated[
        str,
        Field(
            description="QC decision for this run. Agent-written records "
                        "may only use 'pending'.",
            json_schema_extra={
                "enum": ["keep", "exclude", "investigate", "pending"]},
        ),
    ],
    reason: Annotated[
        str,
        Field(description="Explanation for the decision."),
    ],
    run: Annotated[
        str | None,
        Field(description="Run number (e.g. '##'). Omit if no run entity."),
    ] = None,
    reviewer: Annotated[
        str,
        Field(description="Named human signing off. Leave empty for "
                          "agent-written records, which are saved as "
                          "automated and must use decision='pending'."),
    ] = "",
    recommendation: Annotated[
        str | None,
        Field(description="For automated 'pending' records: the decision "
                          "the automation would suggest. Advisory only."),
    ] = None,
    suffix: Annotated[
        str,
        Field(
            description="Modality suffix.",
            json_schema_extra={"enum": ["bold", "T1w", "T2w", "dwi"]},
        ),
    ] = "bold",
) -> str:
    """Record a QC decision for a run, in duckbrain's decision store
    (derivatives/duckbrain/qc/decisions/, append-only JSON audit trail).
    Without a named human reviewer the record is automated and may only
    be 'pending' (+ optional recommendation); keep/exclude/investigate
    require a reviewer. Query via the catalog's qc_decisions table.
    """
    qc_dash = _import_qc_dashboard()
    # "qc_agent" was this tool's historical default reviewer; it names
    # automation, not a person, so it must never read as a sign-off
    automated = (not reviewer.strip()
                 or reviewer.strip().lower() in {"qc_agent", "agent"})
    try:
        record = qc_dash.save_decision(
            qc_dash.decisions_dir(get_paths().bids_root),
            subject, session, task, run,
            decision, reason, reviewer, suffix,
            automated=automated, recommendation=recommendation,
        )
    except ValueError as exc:
        return json.dumps({"saved": False, "error": str(exc)})
    return json.dumps({"saved": True, "automated": automated, **record},
                      default=str)


def load_qc_decisions(
    subject: Annotated[
        str | None,
        Field(description="Subject ID without 'sub-' prefix. Default: all."),
    ] = None,
) -> str:
    """Load existing QC decisions from every known location (legacy
    preprocessing_qc/ and qc_review/ plus duckbrain's current store),
    newest last. Returns the latest decision for each run plus full
    history. Use to check what has already been reviewed.
    """
    qc_dash = _import_qc_dashboard()
    decisions = qc_dash.load_decisions(
        qc_dash.decision_search_dirs(get_paths().bids_root), subject=subject,
    )
    return json.dumps({
        "n_decisions": len(decisions),
        "decisions": decisions,
    }, default=str)


TOOL_FUNCTIONS = [
    list_qc_reports,
    get_iqm_metrics,
    aggregate_iqms,
    detect_iqm_outliers,
    get_motion_summary,
    get_processing_status,
    get_run_details,
    load_qc_decisions,
]

WRITE_TOOL_FUNCTIONS = [
    generate_qc_dashboard,
    save_qc_decision,
]
