"""BIDS dataset tools for the MMMData MCP server.

These functions wrap the mmmdata BIDS utilities (``core.bids_utils``)
and expose them as MCP tools.
"""

import json
from typing import Annotated

from pydantic import Field

from ..paths import get_paths


def _get_bids_layout():
    """Summarize the dataset at the configured BIDS root."""
    from core.bids_utils import summarize_bids_dataset

    summary = summarize_bids_dataset(
        bids_dir=str(get_paths().bids_root),
        verbose=False,
    )
    return summary


def summarize_dataset() -> str:
    """Summarize the BIDS neuroimaging dataset. Returns subject count,
    sessions, datatypes, modalities, and tasks as JSON."""
    summary = _get_bids_layout()
    # Remove non-serialisable BIDSLayout before returning
    result = {k: v for k, v in summary.items() if k != "layout"}
    return json.dumps(result, indent=2)


def list_subjects() -> str:
    """List all subject IDs in the BIDS dataset."""
    summary = _get_bids_layout()
    return json.dumps(summary["subjects"])


def get_subject_files(
    subject_id: Annotated[
        str,
        Field(description="Subject ID without 'sub-' prefix, e.g. '##'."),
    ],
) -> str:
    """Get a detailed list of all files for a specific subject.
    Returns a JSON table with BIDS entities and file paths."""
    from core.bids_utils import get_subject_summary

    df = get_subject_summary(subject_id, bids_dir=str(get_paths().bids_root))
    return df.to_json(orient="records", indent=2)


TOOL_FUNCTIONS = [summarize_dataset, list_subjects, get_subject_files]
