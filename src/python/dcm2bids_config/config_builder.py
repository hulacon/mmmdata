"""Build dcm2bids configuration dicts from structured session definitions.

This module contains pure functions with no filesystem I/O.  All inputs are
dataclasses or simple dicts; the output is a plain ``dict`` ready for
``json.dumps()``.
"""

from __future__ import annotations

from .session_defs import AnatDef, SessionDef, TaskDef


def _b0_id(group: str, subject: str, session: str) -> str:
    """Build a B0FieldIdentifier string.

    >>> _b0_id("encoding", "sub-03", "ses-06")
    'B0map_encoding_sub03_ses06'
    """
    sub = subject.replace("-", "")
    ses = session.replace("-", "")
    return f"B0map_{group}_{sub}_{ses}"


def _build_bold_description(
    task: TaskDef,
    run: int,
    subject: str,
    session: str,
) -> dict:
    """Build a single BOLD description entry."""
    desc: dict = {
        "id": f"task_{task.task_label}"
        + (f"_run-{run}" if task.is_multi_run else ""),
        "datatype": "func",
        "suffix": "bold",
        "custom_entities": f"task-{task.task_label}",
        "criteria": {
            "ProtocolName": task.protocol_name(run),
            "MultibandAccelerationFactor": 3,
        },
        "sidecar_changes": {
            "TaskName": task.task_label,
        },
    }
    if task.fmap_group and task.fmap_group != "none":
        desc["sidecar_changes"]["B0FieldSource"] = _b0_id(
            task.fmap_group, subject, session
        )
    return desc


def _build_sbref_description(
    task: TaskDef,
    run: int,
    subject: str,
    session: str,
) -> dict:
    """Build a single SBRef description entry."""
    desc: dict = {
        "id": f"task_{task.task_label}"
        + (f"_run-{run}" if task.is_multi_run else ""),
        "datatype": "func",
        "suffix": "sbref",
        "custom_entities": f"task-{task.task_label}",
        "criteria": {
            "SeriesDescription": task.sbref_description(run),
        },
        "sidecar_changes": {
            "TaskName": task.task_label,
        },
    }
    if task.fmap_group and task.fmap_group != "none":
        desc["sidecar_changes"]["B0FieldSource"] = _b0_id(
            task.fmap_group, subject, session
        )
    return desc


def _build_anat_description(anat: AnatDef) -> dict:
    """Build an anatomical / DWI description entry."""
    return {
        "id": anat.suffix if anat.datatype == "anat" else f"{anat.suffix}_dir-{anat.acq}",
        "datatype": anat.datatype,
        "suffix": anat.suffix,
        "custom_entities": anat.custom_entities,
        "criteria": {
            "SeriesDescription": anat.series_description,
        },
    }


def _build_fmap_description_seriesnumber(
    direction: str,
    series_number: int,
    group: str,
    intended_for: list[str],
    subject: str,
    session: str,
) -> dict:
    """Build a fieldmap description matched by SeriesNumber."""
    series_desc = f"se_epi_{direction.lower()}"
    return {
        "datatype": "fmap",
        "suffix": "epi",
        "custom_entities": f"dir-{direction}",
        "criteria": {
            "SeriesDescription": series_desc,
            "SeriesNumber": str(series_number),
        },
        "sidecar_changes": {
            "B0FieldIdentifier": _b0_id(group, subject, session),
        },
    }


def _build_fmap_description_seriesdesc(
    direction: str,
    group: str,
    intended_for: list[str],
    subject: str,
    session: str,
    series_number: int | None = None,
) -> dict:
    """Build a fieldmap description matched by SeriesDescription suffix.

    When *series_number* is provided, it is added to ``criteria`` alongside
    ``SeriesDescription`` for hybrid matching (needed when two fmap pairs
    share the same SeriesDescription suffix).
    """
    series_desc = f"se_epi_{direction.lower()}_{group}"
    criteria: dict = {
        "SeriesDescription": series_desc,
    }
    if series_number is not None:
        criteria["SeriesNumber"] = str(series_number)
    return {
        "datatype": "fmap",
        "suffix": "epi",
        "custom_entities": f"dir-{direction}",
        "criteria": criteria,
        "sidecar_changes": {
            "B0FieldIdentifier": _b0_id(group, subject, session),
        },
    }


FieldmapInfo = dict[str, dict[str, int]]
"""Mapping from fmap group name to ``{"ap": series_num, "pa": series_num}``."""


def build_config(
    subject: str,
    session: str,
    session_def: SessionDef,
    fmap_info: FieldmapInfo | None = None,
) -> dict:
    """Build a complete dcm2bids config dict.

    This is a pure function: same inputs always produce the same output.

    Parameters
    ----------
    subject : str
        Subject ID including prefix (e.g. ``"sub-03"``).
    session : str
        Session ID including prefix (e.g. ``"ses-06"``).
    session_def : SessionDef
        The session type definition (from ``session_defs``).
    fmap_info : dict, optional
        Fieldmap series numbers, keyed by group name.  Required when
        ``session_def.fmap_strategy == "series_number"``.  Example::

            {"encoding": {"ap": 10, "pa": 11},
             "retrieval": {"ap": 28, "pa": 30}}

    Returns
    -------
    dict
        A dcm2bids-compatible config dict with a ``"descriptions"`` key.

    Raises
    ------
    ValueError
        If series_number strategy is used but ``fmap_info`` is missing
        required groups.
    """
    descriptions: list[dict] = []

    # --- Anatomical / DWI ---
    for anat in session_def.anat:
        descriptions.append(_build_anat_description(anat))

    # --- Functional tasks ---
    for task in session_def.tasks:
        for run in task.run_numbers():
            descriptions.append(
                _build_bold_description(task, run, subject, session)
            )
            if task.has_sbref:
                descriptions.append(
                    _build_sbref_description(task, run, subject, session)
                )

    # --- Fieldmaps ---
    for group in session_def.fmap_groups:
        intended_for = session_def.task_ids_for_fmap_group(group)

        if session_def.fmap_strategy == "series_number":
            if fmap_info is None or group not in fmap_info:
                raise ValueError(
                    f"fmap_info missing group {group!r} for "
                    f"{subject}/{session} (strategy=series_number)"
                )
            series = fmap_info[group]
            for direction, key in [("AP", "ap"), ("PA", "pa")]:
                descriptions.append(
                    _build_fmap_description_seriesnumber(
                        direction, series[key], group,
                        intended_for, subject, session,
                    )
                )
        elif session_def.fmap_strategy == "series_description":
            for direction, key in [("AP", "ap"), ("PA", "pa")]:
                sn = None
                if fmap_info and group in fmap_info:
                    sn = fmap_info[group].get(key)
                descriptions.append(
                    _build_fmap_description_seriesdesc(
                        direction, group,
                        intended_for, subject, session,
                        series_number=sn,
                    )
                )

    return {"descriptions": descriptions}
