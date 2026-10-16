"""Contract A catalog query tools for the MMMData MCP server.

Thin wrappers around the catalog query helpers in ``core.catalog``
that let an agent ask "what data exists, how was it processed, and
what did QC decide" from catalog.duckdb.

All real logic lives in ``core.catalog``; these wrappers only resolve
paths from config, delegate, and serialize the result to JSON. Subject
and session arguments accept both bare ("##") and prefixed ("sub-##")
forms.
"""

import json
from typing import Annotated

from pydantic import Field

from ..paths import get_paths

_STATUSES = ["present", "missing", "excepted", "surplus"]


def _catalog():
    """Return (catalog_db_path, core.catalog module)."""
    from core import catalog
    return get_paths().catalog_db, catalog


# ---------------------------------------------------------------------------
# Tool functions
# ---------------------------------------------------------------------------

def query_catalog(
    sql: Annotated[
        str,
        Field(
            description=(
                "A read-only DuckDB query. Example: "
                "\"SELECT sub, ses, task, COUNT(*) AS n "
                "FROM files WHERE suffix='bold' AND dataset_relpath='.' "
                "GROUP BY sub, ses, task\". Entity labels are bare "
                "('##', not 'sub-##'); run is an integer."
            )
        ),
    ],
    params: Annotated[
        list | None,
        Field(description="Positional bind parameters for the query."),
    ] = None,
) -> str:
    """Run a read-only SQL query against the MMMData catalog
    (catalog.duckdb, Contract A). Tables: files (all BIDS-parseable
    files across raw + derivatives, keyed by dataset_relpath with the
    full entity set incl. space/res/desc), datasets (per-tree
    provenance: pipeline, version, kind=canonical|internal), sessions
    (canonical sessions.tsv), expected_units + expectation_exceptions
    (declared expectations), qc_decisions, files_supplemental
    (non-BIDS files: category dark|sidecar|metadata). Views:
    observed_units, resolution (four-state judgment: present/missing/
    excepted/surplus with disposition accepted|pending). Returns
    columns and rows as JSON.
    """
    db, catalog = _catalog()
    try:
        columns, rows = catalog.run_select(db, sql, params)
    except ValueError as e:
        # read-only gate (enforced in core.catalog.run_select)
        return json.dumps({"error": str(e)})
    except FileNotFoundError:
        raise
    except Exception as e:
        return json.dumps({"error": str(e)})
    return json.dumps({"columns": columns, "row_count": len(rows), "rows": rows},
                      default=str)


def validate_dataset(
    subjects: Annotated[
        list[str] | None,
        Field(
            description=(
                "Subjects to report (e.g. ['##'] or ['sub-##']). "
                "Default: all declared subjects."
            )
        ),
    ] = None,
    sessions: Annotated[
        list[str] | None,
        Field(
            description=(
                "Sessions to report (e.g. ['##'] or ['ses-##']). "
                "Default: all sessions."
            )
        ),
    ] = None,
    statuses: Annotated[
        list[str] | None,
        Field(
            description=(
                "Resolution statuses to report: present, missing, "
                "excepted, surplus. Default: issues only (missing, "
                "surplus, excepted-pending)."
            )
        ),
    ] = None,
    include_all: Annotated[
        bool,
        Field(description="Report every resolution row, including present."),
    ] = False,
) -> str:
    """Report how the dataset resolves against its declared
    expectations (Contract A resolution view). Every expected unit is
    present, missing, excepted (with disposition accepted = will never
    exist, or pending = still to do), or surplus (present but
    undeclared — catches mislabels). Judged live inside the catalog;
    the declaration is <bids_root>/expectations/dataset.toml. Returns
    a status summary plus the reported unit rows.
    """
    db, catalog = _catalog()
    try:
        report = catalog.resolution_report(
            db, subjects=subjects, sessions=sessions,
            statuses=statuses, include_all=include_all,
        )
    except Exception as e:
        return json.dumps({"error": str(e), "available_statuses": _STATUSES})
    return json.dumps(report, default=str)


def get_session_summary(
    subject: Annotated[
        str, Field(description="Subject label (e.g. '##' or 'sub-##').")
    ],
    session: Annotated[
        str, Field(description="Session label (e.g. '##' or 'ses-##').")
    ],
) -> str:
    """Get a catalog-backed summary of one subject/session: raw BIDS
    files with entities, derivative coverage with provenance
    (pipeline + version per tree), supplemental/dark file counts,
    QC decisions, session metadata (from the canonical sessions.tsv),
    and the session's expectation resolution (anything not plainly
    present).
    """
    db, catalog = _catalog()
    summary = catalog.session_summary(db, subject, session)
    return json.dumps(summary, default=str)


def get_file_expectations(
    task: Annotated[
        str | None,
        Field(
            description=(
                "Task name (e.g. 'TBencoding', 'floc'). Returns the "
                "declared unit shapes for that task."
            )
        ),
    ] = None,
    session_type: Annotated[
        str | None,
        Field(
            description=(
                "Session type (e.g. 'localizer', 'cued_recall', "
                "'anat'). Returns its sessions and expected units."
            )
        ),
    ] = None,
) -> str:
    """Look up what the dataset's declared expectations say should
    exist for a task or session type, from the catalog's
    expected_units/sessions tables (the ingested form of
    <bids_root>/expectations/dataset.toml). With no arguments,
    returns an overview: declared tasks, session types, subjects,
    datasets carrying expectations, and total unit count.
    """
    db, catalog = _catalog()
    result = catalog.lookup_expectations(
        db, task=task, session_type=session_type
    )
    return json.dumps(result, default=str)


TOOL_FUNCTIONS = [
    query_catalog,
    validate_dataset,
    get_session_summary,
    get_file_expectations,
]
