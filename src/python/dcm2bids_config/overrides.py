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
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .session_defs import (
    ANAT_T1W,
    ANAT_T2W_COR,
    ANAT_T2W_SPC,
    DWI_AP,
    DWI_LR,
    DWI_PA,
    DWI_RL,
    AnatDef,
    SessionDef,
    TaskDef,
    get_session_def,
)

# Map short names (used in overrides.toml) to predefined AnatDef constants
ANAT_REGISTRY: dict[str, AnatDef] = {
    "T1w_MPR": ANAT_T1W,
    "T2w_SPC": ANAT_T2W_SPC,
    "T2w_oblcor": ANAT_T2W_COR,
    "dwi_AP": DWI_AP,
    "dwi_PA": DWI_PA,
    "dwi_RL": DWI_RL,
    "dwi_LR": DWI_LR,
}


@dataclass
class OverrideResult:
    """Result of applying overrides to a session definition."""

    session_def: SessionDef
    fmap_info: dict[str, dict[str, int]] | None = None
    run_protocols: dict[str, dict[int, str]] | None = None
    run_series: dict[str, dict[int, dict[str, int]]] | None = None
    fmap_desc_map: dict[str, str] | None = None


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
) -> OverrideResult:
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
    OverrideResult
        Modified session definition and optional override metadata.
    """
    if session_id not in overrides:
        return OverrideResult(session_def=session_def)

    ovr = overrides[session_id]

    # --- Session type override ---
    if "session_type" in ovr:
        from .session_defs import SESSION_TYPES
        session_def = SESSION_TYPES[ovr["session_type"]]

    # --- Task list override ---
    if "tasks" in ovr:
        tasks = []
        for t in ovr["tasks"]:
            runs_val = t.get("runs", 1)
            if isinstance(runs_val, list):
                runs_val = tuple(runs_val)
            tasks.append(
                TaskDef(
                    task_label=t["task_label"],
                    protocol_base=t["protocol_base"],
                    fmap_group=t.get("fmap_group", "encoding"),
                    runs=runs_val,
                    has_sbref=t.get("has_sbref", False),
                )
            )
        session_def = replace(session_def, tasks=tuple(tasks))

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

    # --- Add anatomical entries ---
    if "add_anat" in ovr:
        extra_anat = tuple(ANAT_REGISTRY[name] for name in ovr["add_anat"])
        session_def = replace(session_def, anat=session_def.anat + extra_anat)

    # --- Exclude tasks ---
    if "exclude_tasks" in ovr:
        excluded = set(ovr["exclude_tasks"])
        session_def = replace(
            session_def,
            tasks=tuple(t for t in session_def.tasks if t.task_label not in excluded),
        )

    # --- Run protocol overrides (TOML keys are strings, convert to int) ---
    run_protocols = None
    if "run_protocols" in ovr:
        run_protocols = {
            task_label: {int(k): v for k, v in runs.items()}
            for task_label, runs in ovr["run_protocols"].items()
        }

    # --- Run series number overrides (TOML keys are strings, convert to int) ---
    run_series = None
    if "run_series" in ovr:
        run_series = {
            task_label: {int(k): spec for k, spec in runs.items()}
            for task_label, runs in ovr["run_series"].items()
        }

    # --- Fieldmap description suffix map ---
    fmap_desc_map = ovr.get("fmap_desc_map")

    return OverrideResult(
        session_def=session_def,
        fmap_info=fmap_info,
        run_protocols=run_protocols,
        run_series=run_series,
        fmap_desc_map=fmap_desc_map,
    )
