"""Override file parsing for per-subject session idiosyncrasies.

Each subject may have an optional ``overrides.toml`` file that documents
and handles sessions deviating from the canonical schedule.  Example::

    [ses-10]
    note = "Re-entry after encoding run 1; 3 fieldmap pairs"
    fmap_groups = ["encoding_r1", "encoding_r2", "retrieval"]
    [ses-10.fmap_series.encoding_r1]
    ap = 5
    pa = 7
    [ses-10.fmap_series.encoding_r2]
    ap = 21
    pa = 23
    [ses-10.fmap_series.retrieval]
    ap = 37
    pa = 39

    [ses-02]
    note = "Localizer: PRF (3 runs) + auditory + tone"
    session_type = "localizer"
    [[ses-02.tasks]]
    task_label = "prf"
    protocol_base = "localizer_prf_run{n}"
    fmap_group = "first"
    runs = 3
    has_sbref = false
    [[ses-02.tasks]]
    task_label = "auditory"
    protocol_base = "localizer_auditory"
    fmap_group = "second"
    [[ses-02.tasks]]
    task_label = "tone"
    protocol_base = "localizer_tone"
    fmap_group = "second"
"""

from __future__ import annotations

import tomllib
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

from .session_defs import SessionDef, TaskDef, get_session_def


def load_overrides(overrides_path: Path) -> dict[str, Any]:
    """Load a subject's override TOML file.

    Parameters
    ----------
    overrides_path : Path
        Path to the ``overrides.toml`` file.

    Returns
    -------
    dict
        Parsed TOML, keyed by session ID (e.g. ``"ses-10"``).
        Returns empty dict if file doesn't exist.
    """
    if not overrides_path.exists():
        return {}
    with open(overrides_path, "rb") as f:
        return tomllib.load(f)


def apply_overrides(
    session_id: str,
    session_def: SessionDef,
    overrides: dict[str, Any],
) -> tuple[SessionDef, dict[str, dict[str, int]] | None]:
    """Apply overrides to a session definition.

    Parameters
    ----------
    session_id : str
        Session identifier (e.g. ``"ses-10"``).
    session_def : SessionDef
        Base session definition from the schedule.
    overrides : dict
        Full overrides dict (all sessions for a subject).

    Returns
    -------
    tuple[SessionDef, dict | None]
        Modified session definition and optional fmap_info override.
        If fmap_series is specified in overrides, returns the explicit
        series numbers; otherwise returns None (auto-detect).
    """
    if session_id not in overrides:
        return session_def, None

    ovr = overrides[session_id]

    # --- Session type override ---
    if "session_type" in ovr:
        from .session_defs import SESSION_TYPES
        session_def = SESSION_TYPES[ovr["session_type"]]

    # --- Task list override ---
    if "tasks" in ovr:
        tasks = tuple(
            TaskDef(
                task_label=t["task_label"],
                protocol_base=t["protocol_base"],
                fmap_group=t.get("fmap_group", "encoding"),
                runs=t.get("runs", 1),
                has_sbref=t.get("has_sbref", False),
            )
            for t in ovr["tasks"]
        )
        session_def = replace(session_def, tasks=tasks)

    # --- Fieldmap group override ---
    if "fmap_groups" in ovr:
        session_def = replace(
            session_def, fmap_groups=tuple(ovr["fmap_groups"])
        )

    # --- Fieldmap strategy override ---
    if "fmap_strategy" in ovr:
        session_def = replace(session_def, fmap_strategy=ovr["fmap_strategy"])

    # --- Explicit fieldmap series numbers ---
    fmap_info = None
    if "fmap_series" in ovr:
        fmap_info = {
            group: {"ap": spec["ap"], "pa": spec["pa"]}
            for group, spec in ovr["fmap_series"].items()
        }

    # --- Exclude tasks ---
    if "exclude_tasks" in ovr:
        excluded = set(ovr["exclude_tasks"])
        session_def = replace(
            session_def,
            tasks=tuple(t for t in session_def.tasks if t.task_label not in excluded),
        )

    return session_def, fmap_info
