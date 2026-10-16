"""Sourcedata inspection tools for the MMMData MCP server.

Read-only tools for scanning what raw data exists in the source tree
(``[paths] source_dir``, kept outside the BIDS dataset so the BIDS tree
can be shared without exposing PII). Opt-in (``--toolsets sourcedata``):
the source tree is never part of a shared copy of the dataset.

Source layout: sub-##/ses-##/{audio,behavioral,dicom,eyetracking,other},
plus shared/ (cross-subject material) and archive/ (pilots, retired
conversion staging).

The substantive logic lives in ``core.sourcedata``; these functions
only resolve paths and serialize results to JSON.
"""

import json
from typing import Annotated

from pydantic import Field

from ..paths import get_paths


def scan_sourcedata(
    subject_id: Annotated[
        str,
        Field(description="Subject ID without 'sub-' prefix, e.g. '##'."),
    ],
    session_id: Annotated[
        str | None,
        Field(description="Optional session ID without 'ses-' prefix, e.g. '##'."),
    ] = None,
) -> str:
    """Scan the raw source data tree (mmmsourcedata) for a subject. Shows per-session data by modality (audio, behavioral, dicom, eyetracking, other) with file/series counts. Useful for checking what has been collected for not-yet-BIDSified subjects."""
    paths = get_paths()

    from core import sourcedata

    try:
        result = sourcedata.scan_sourcedata(
            source_dir=paths.sourcedata_dir,
            subject_id=subject_id,
            session_id=session_id,
        )
    except FileNotFoundError as exc:
        return json.dumps({
            "error": str(exc),
            "available_subjects": sourcedata.list_source_subjects(
                paths.sourcedata_dir
            ),
        })

    return json.dumps(result, indent=2)


TOOL_FUNCTIONS = [scan_sourcedata]
