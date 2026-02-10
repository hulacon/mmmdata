"""Session and task definitions for dcm2bids config generation.

This module is the single source of truth for:
- What tasks exist and how they map to DICOM protocol names
- What each session type contains
- Which session numbers map to which session type

All data here is pure Python with no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TaskDef:
    """Definition of a single functional task.

    Parameters
    ----------
    task_label : str
        BIDS task entity value (e.g. ``"TBencoding"``).
    protocol_base : str
        DICOM ProtocolName base. For multi-run tasks use ``{n}`` placeholder
        (e.g. ``"cued_recall_encoding_run{n}"``).
    fmap_group : str
        Which fieldmap group this task belongs to (e.g. ``"encoding"``,
        ``"retrieval"``). Used for B0FieldSource/B0FieldIdentifier.
    runs : int
        Number of runs. 1 means no ``run-`` entity in BIDS filename.
    has_sbref : bool
        Whether each run has an accompanying SBRef series.
    """

    task_label: str
    protocol_base: str
    fmap_group: str
    runs: int = 1
    has_sbref: bool = False

    def protocol_name(self, run: int) -> str:
        """Return the DICOM ProtocolName for a specific run number."""
        if "{n}" in self.protocol_base:
            return self.protocol_base.replace("{n}", str(run))
        return self.protocol_base

    def sbref_description(self, run: int) -> str:
        """Return the DICOM SeriesDescription for the SBRef of a run."""
        return self.protocol_name(run) + "_SBRef"


@dataclass(frozen=True)
class AnatDef:
    """Definition of an anatomical or DWI acquisition.

    Parameters
    ----------
    suffix : str
        BIDS suffix (e.g. ``"T1w"``, ``"T2w"``, ``"dwi"``).
    acq : str
        BIDS ``acq`` entity value (e.g. ``"MPR"``, ``"SPC"``).
    series_description : str
        DICOM SeriesDescription to match.
    datatype : str
        BIDS datatype (``"anat"`` or ``"dwi"``).
    custom_entities : str
        Full custom_entities string for dcm2bids (e.g. ``"acq-MPR"``).
    """

    suffix: str
    acq: str
    series_description: str
    datatype: str = "anat"
    custom_entities: str = ""

    def __post_init__(self):
        if not self.custom_entities:
            # Build from acq if datatype is dwi (uses dir-XX) or anat (uses acq-XX)
            if self.datatype == "dwi":
                object.__setattr__(self, "custom_entities", f"dir-{self.acq}")
            else:
                object.__setattr__(self, "custom_entities", f"acq-{self.acq}")


@dataclass(frozen=True)
class SessionDef:
    """Complete definition of a session type.

    Parameters
    ----------
    session_type : str
        Short identifier (e.g. ``"tb_first"``, ``"naturalistic"``).
    tasks : tuple[TaskDef, ...]
        Functional tasks in acquisition order.
    fmap_strategy : str
        How fieldmaps are matched:
        - ``"series_number"``: match by SeriesNumber (older sessions)
        - ``"series_description"``: match by SeriesDescription suffix (newer)
        - ``"none"``: no fieldmaps in this session
    anat : tuple[AnatDef, ...]
        Anatomical/DWI acquisitions (empty for most functional sessions).
    fmap_groups : tuple[str, ...]
        Ordered fieldmap group names. Each gets an AP+PA pair.
    """

    session_type: str
    tasks: tuple[TaskDef, ...] = ()
    fmap_strategy: str = "series_number"
    anat: tuple[AnatDef, ...] = ()
    fmap_groups: tuple[str, ...] = ()

    def task_ids_for_fmap_group(self, group: str) -> list[str]:
        """Return dcm2bids description IDs for all tasks in a fieldmap group."""
        ids = []
        for task in self.tasks:
            if task.fmap_group != group:
                continue
            if task.runs == 1:
                ids.append(f"task_{task.task_label}")
            else:
                for r in range(1, task.runs + 1):
                    ids.append(f"task_{task.task_label}_run-{r}")
        return ids


# ---------------------------------------------------------------------------
# Anatomical definitions (ses-01 and ses-28/ses-30)
# ---------------------------------------------------------------------------

ANAT_T1W = AnatDef("T1w", "MPR", "ABCD_T1w_MPR_vNav")
ANAT_T2W_SPC = AnatDef("T2w", "SPC", "ABCD_T2w_SPC_vNav")
ANAT_T2W_COR = AnatDef("T2w", "oblcor", "T2_coronal_1.8")

DWI_AP = AnatDef("dwi", "AP", "cmrr_diff_3shell_ap", datatype="dwi")
DWI_PA = AnatDef("dwi", "PA", "cmrr_diff_3shell_pa", datatype="dwi")
DWI_RL = AnatDef("dwi", "RL", "cmrr_diff_3shell_rl", datatype="dwi")
DWI_LR = AnatDef("dwi", "LR", "cmrr_diff_3shell_lr", datatype="dwi")


# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------

# Trial-based (TB) tasks — ses-04 through ses-18
TB_ENCODING = TaskDef("TBencoding", "cued_recall_encoding_run{n}", "encoding", runs=3, has_sbref=True)
TB_MATH = TaskDef("TBmath", "cued_recall_math", "encoding", has_sbref=True)
TB_RESTING = TaskDef("TBresting", "cued_recall_resting", "retrieval", has_sbref=True)
TB_RETRIEVAL_2 = TaskDef("TBretrieval", "cued_recall_retrieval_run{n}", "retrieval", runs=2, has_sbref=True)
TB_RETRIEVAL_4 = TaskDef("TBretrieval", "cued_recall_retrieval_run{n}", "retrieval", runs=4, has_sbref=False)

# Naturalistic (NAT) tasks — ses-19 through ses-28
NAT_ENCODING = TaskDef("NATencoding", "free_recall_encoding_run{n}", "encoding", runs=2, has_sbref=True)
NAT_MATH = TaskDef("NATmath", "free_recall_math", "encoding", has_sbref=True)
NAT_RESTING = TaskDef("NATresting", "free_recall_resting", "retrieval", has_sbref=True)
NAT_RETRIEVAL = TaskDef("NATretrieval", "free_recall_retrieval_run{n}", "retrieval", runs=1, has_sbref=True)

# Baseline resting (ses-01)
INIT_RESTING = TaskDef("INITresting", "Resting_baseline", "none", has_sbref=False)


# ---------------------------------------------------------------------------
# Session type definitions
# ---------------------------------------------------------------------------

ANATOMY = SessionDef(
    session_type="anatomy",
    anat=(ANAT_T1W, ANAT_T2W_SPC, ANAT_T2W_COR, DWI_AP, DWI_PA, DWI_RL, DWI_LR),
    tasks=(INIT_RESTING,),
    fmap_strategy="none",
    fmap_groups=(),
)

# TODO: Localizer sessions (ses-02, ses-03) have variable task sets across
# subjects. Needs a localizer template or per-subject override to define
# which localizer tasks were run.  Placeholder definition:
LOCALIZER = SessionDef(
    session_type="localizer",
    tasks=(),  # filled via overrides per subject
    fmap_strategy="series_number",
    fmap_groups=("first", "second"),
)

TB_FIRST = SessionDef(
    session_type="tb_first",
    tasks=(TB_ENCODING, TB_MATH, TB_RESTING, TB_RETRIEVAL_2),
    fmap_strategy="series_number",
    fmap_groups=("encoding", "retrieval"),
)

TB_MIDDLE = SessionDef(
    session_type="tb_middle",
    tasks=(
        TaskDef("TBencoding", "cued_recall_encoding_run{n}", "encoding", runs=3, has_sbref=True),
        TaskDef("TBmath", "cued_recall_math", "encoding", has_sbref=True),
        TaskDef("TBresting", "cued_recall_resting", "retrieval", has_sbref=True),
        TaskDef("TBretrieval", "cued_recall_retrieval_run{n}", "retrieval", runs=4, has_sbref=True),
    ),
    fmap_strategy="series_number",
    fmap_groups=("encoding", "retrieval"),
)

TB_LAST = SessionDef(
    session_type="tb_last",
    tasks=(
        TaskDef("TBmath", "cued_recall_math", "encoding", has_sbref=True),
        TaskDef("TBresting", "cued_recall_resting", "retrieval", has_sbref=True),
        TaskDef("TBretrieval", "cued_recall_retrieval_run{n}", "retrieval", runs=2, has_sbref=True),
    ),
    fmap_strategy="series_number",
    fmap_groups=("encoding", "retrieval"),
)

NATURALISTIC = SessionDef(
    session_type="naturalistic",
    tasks=(NAT_ENCODING, NAT_MATH, NAT_RESTING, NAT_RETRIEVAL),
    fmap_strategy="series_description",
    fmap_groups=("encoding", "retrieval"),
)

NATURALISTIC_FM = SessionDef(
    session_type="naturalistic_fm",
    tasks=(NAT_ENCODING, NAT_MATH, NAT_RESTING, NAT_RETRIEVAL),
    fmap_strategy="series_number",
    fmap_groups=("encoding", "retrieval"),
)

# TODO: Final session (ses-30) has variable content (final memory tests +
# makeup localizers). Needs per-subject override.  Placeholder:
FINAL = SessionDef(
    session_type="final",
    tasks=(),  # filled via overrides per subject
    fmap_strategy="series_description",
    fmap_groups=("encoding",),
)


# ---------------------------------------------------------------------------
# Session schedule: session number → SessionDef
# ---------------------------------------------------------------------------

SESSION_TYPES: dict[str, SessionDef] = {
    "anatomy": ANATOMY,
    "localizer": LOCALIZER,
    "tb_first": TB_FIRST,
    "tb_middle": TB_MIDDLE,
    "tb_last": TB_LAST,
    "naturalistic": NATURALISTIC,
    "naturalistic_fm": NATURALISTIC_FM,
    "final": FINAL,
}

SESSION_SCHEDULE: dict[str, str] = {
    "ses-01": "anatomy",
    "ses-02": "localizer",
    "ses-03": "localizer",
    "ses-04": "tb_first",
    **{f"ses-{i:02d}": "tb_middle" for i in range(5, 18)},
    "ses-18": "tb_last",
    **{f"ses-{i:02d}": "naturalistic" for i in range(19, 29)},
    # ses-29 is behavioral only (no imaging)
    "ses-30": "final",
}


def get_session_def(session_id: str) -> SessionDef:
    """Look up the SessionDef for a session ID.

    Parameters
    ----------
    session_id : str
        Session identifier (e.g. ``"ses-06"``).

    Returns
    -------
    SessionDef

    Raises
    ------
    KeyError
        If the session is not in the schedule (e.g. ses-29).
    """
    session_type = SESSION_SCHEDULE[session_id]
    return SESSION_TYPES[session_type]
