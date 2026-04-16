"""Pipeline step specifications and registry.

The ``STEPS`` dict is the authoritative DAG of MMMData processing steps.
Each :class:`StepSpec` records its prerequisites, variants, resource
requirements, and a callable validator that checks postconditions for a
given (subject, session).

The registry is populated lazily by :func:`_register_default_steps`, which
is called automatically on first access to ``STEPS``. This avoids an import
cycle: ``validators`` imports ``StepSpec`` and ``ValidationResult`` from
here, then this module imports the validators after definitions are ready.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Optional


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class ValidationResult:
    """Result of validating a single step's postconditions for one sub/ses.

    Attributes
    ----------
    step : str
        Registered step name.
    subject, session : str
        Zero-padded IDs without prefixes.
    status : str
        One of: "complete", "partial", "missing", "error", "skipped".

        * complete — expected outputs present and valid
        * partial  — some outputs present, some missing (details in ``details``)
        * missing  — no outputs produced yet (valid target for submission)
        * error    — outputs present but malformed
        * skipped  — not applicable to this sub/ses (e.g., no raw inputs)
    expected : int
        Expected output file count (0 if not applicable).
    found : int
        Actual output file count.
    details : list[str]
        Human-readable messages (diagnostics, missing files, etc.).
    metrics : dict
        Optional numeric spot-checks (e.g., {"tsnr_median": 87.2}).
    """

    step: str
    subject: str
    session: str
    status: str
    expected: int = 0
    found: int = 0
    details: list[str] = dataclasses.field(default_factory=list)
    metrics: dict[str, Any] = dataclasses.field(default_factory=dict)

    @property
    def is_complete(self) -> bool:
        return self.status == "complete"

    @property
    def is_blocking(self) -> bool:
        """True if downstream steps should not run."""
        return self.status in ("missing", "partial", "error")


@dataclasses.dataclass(frozen=True)
class StepSpec:
    """Specification for one pipeline step.

    Attributes
    ----------
    name : str
        Unique step identifier (used as key in STEPS).
    description : str
        Human-readable one-liner.
    depends_on : tuple[str, ...]
        Names of prerequisite steps. A sub/ses is runnable for this step
        iff all prerequisites are complete for that sub/ses.
    variants : tuple[str, ...]
        Named output variants (e.g., ("original", "nordic") for fmriprep).
        Empty tuple means no variants.
    validate : Callable
        Signature: ``(subject, session, bids_root=None, **kwargs) -> ValidationResult``.
    slurm_script : str, optional
        Relative path (from scripts/) of sbatch script that runs this step.
    slurm_resources : dict[str, str]
        Standard SLURM settings: time, mem, cpus, partition.
    env_vars : tuple[str, ...]
        Env vars the slurm_script expects (e.g., ("NORDIC_SUBJECT", "NORDIC_SESSION")).
    """

    name: str
    description: str
    depends_on: tuple[str, ...]
    variants: tuple[str, ...]
    validate: Callable[..., ValidationResult]
    slurm_script: Optional[str] = None
    slurm_resources: dict[str, str] = dataclasses.field(default_factory=dict)
    env_vars: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

STEPS: dict[str, StepSpec] = {}
_REGISTERED: bool = False


def register_step(spec: StepSpec) -> None:
    """Register a step. Idempotent for same-name same-spec re-registration."""
    if spec.name in STEPS and STEPS[spec.name] is not spec:
        raise ValueError(f"Step {spec.name!r} is already registered with a different spec")
    STEPS[spec.name] = spec


def get_step(name: str) -> StepSpec:
    """Fetch a registered step by name."""
    _ensure_registered()
    if name not in STEPS:
        raise KeyError(
            f"Unknown step {name!r}. Registered: {sorted(STEPS.keys())}"
        )
    return STEPS[name]


def all_steps() -> list[StepSpec]:
    """Return all registered steps in topological (dependency) order."""
    _ensure_registered()
    return [STEPS[name] for name in topological_order()]


def topological_order() -> list[str]:
    """Return step names in dependency order (Kahn's algorithm).

    Raises ValueError if the DAG contains a cycle.
    """
    _ensure_registered()
    in_degree = {name: 0 for name in STEPS}
    for spec in STEPS.values():
        for dep in spec.depends_on:
            if dep not in STEPS:
                raise KeyError(
                    f"Step {spec.name!r} depends on unregistered step {dep!r}"
                )
            in_degree[spec.name] += 1

    queue = [name for name, d in in_degree.items() if d == 0]
    order: list[str] = []
    while queue:
        queue.sort()  # deterministic order
        name = queue.pop(0)
        order.append(name)
        for other in STEPS.values():
            if name in other.depends_on:
                in_degree[other.name] -= 1
                if in_degree[other.name] == 0:
                    queue.append(other.name)

    if len(order) != len(STEPS):
        remaining = set(STEPS) - set(order)
        raise ValueError(f"Cycle detected among steps: {sorted(remaining)}")
    return order


def dag_adjacency() -> dict[str, list[str]]:
    """Return adjacency list mapping step -> list of direct dependents."""
    _ensure_registered()
    adj: dict[str, list[str]] = {name: [] for name in STEPS}
    for spec in STEPS.values():
        for dep in spec.depends_on:
            adj[dep].append(spec.name)
    return {k: sorted(v) for k, v in adj.items()}


# ---------------------------------------------------------------------------
# Default registration (deferred to avoid import cycle)
# ---------------------------------------------------------------------------

def _ensure_registered() -> None:
    global _REGISTERED
    if _REGISTERED:
        return
    _REGISTERED = True  # set first to prevent recursion
    from . import validators as v  # noqa: WPS433

    register_step(StepSpec(
        name="bidsification",
        description="Raw BIDS tree: BOLD NIfTIs, sidecars, events, fmaps.",
        depends_on=(),
        variants=(),
        validate=v.validate_bidsification,
        slurm_script="run_dcm2bids.py",
        slurm_resources={"time": "02:00:00", "mem": "16G", "cpus": "4"},
        env_vars=("SUBJECT", "SESSION"),
    ))

    register_step(StepSpec(
        name="mriqc",
        description="MRIQC IQM JSONs for every BOLD and anat scan.",
        depends_on=("bidsification",),
        variants=(),
        validate=v.validate_mriqc,
        slurm_script="mriqc_participant.sbatch",
        slurm_resources={"time": "06:00:00", "mem": "16G", "cpus": "4"},
        env_vars=("SUBJECT", "SESSION"),
    ))

    register_step(StepSpec(
        name="nordic_denoise",
        description="NORDIC PCA denoising of raw BOLD NIfTIs (MATLAB).",
        depends_on=("bidsification",),
        variants=(),
        validate=v.validate_nordic_denoise,
        slurm_script="nordic_denoise.sbatch",
        slurm_resources={"time": "02:00:00", "mem": "32G", "cpus": "4"},
        env_vars=("NORDIC_SUBJECT", "NORDIC_SESSION"),
    ))

    register_step(StepSpec(
        name="nordic_bids_input",
        description="BIDS input tree for fMRIPrep (hardlinked NORDIC BOLDs + sidecars).",
        depends_on=("nordic_denoise",),
        variants=(),
        validate=v.validate_nordic_bids_input,
        slurm_script="nordic_build_bids_input.sh",
        slurm_resources={"time": "00:30:00", "mem": "4G", "cpus": "1"},
        env_vars=("NORDIC_SUBJECT", "NORDIC_SESSION"),
    ))

    register_step(StepSpec(
        name="fmriprep",
        description="fMRIPrep on raw (non-NORDIC) BOLD.",
        depends_on=("bidsification",),
        variants=(),
        validate=v.validate_fmriprep,
        slurm_script="fmriprep_func.sbatch",
        slurm_resources={"time": "08:00:00", "mem": "48G", "cpus": "8"},
        env_vars=("FPREP_SUBJECT", "FPREP_SESSION"),
    ))

    register_step(StepSpec(
        name="fmriprep_nordic",
        description="fMRIPrep on NORDIC-denoised BOLD.",
        depends_on=("nordic_bids_input",),
        variants=(),
        validate=v.validate_fmriprep_nordic,
        slurm_script="fmriprep_nordic.sbatch",
        slurm_resources={"time": "08:00:00", "mem": "48G", "cpus": "8"},
        env_vars=("FPREP_SUBJECT", "FPREP_SESSION"),
    ))

    register_step(StepSpec(
        name="preprocessing_qc",
        description="Per-run QC decision JSONs (keep/exclude/investigate).",
        depends_on=("fmriprep_nordic",),
        variants=(),
        validate=v.validate_preprocessing_qc,
        slurm_script=None,  # generated by Python, not SLURM
        slurm_resources={},
        env_vars=(),
    ))

    register_step(StepSpec(
        name="stream_glmsingle",
        description="Layer 2 GLMsingle stream: curated confounds + outlier mask.",
        depends_on=("preprocessing_qc",),
        variants=(),
        validate=v.validate_stream_glmsingle,
        slurm_script="clean_stream.sbatch",
        slurm_resources={"time": "01:00:00", "mem": "16G", "cpus": "4"},
        env_vars=("STREAM", "SUBJECT", "SESSION"),
    ))

    register_step(StepSpec(
        name="stream_naturalistic",
        description="Layer 2 naturalistic stream: confound regression + high-pass.",
        depends_on=("preprocessing_qc",),
        variants=(),
        validate=v.validate_stream_naturalistic,
        slurm_script="clean_stream.sbatch",
        slurm_resources={"time": "02:00:00", "mem": "32G", "cpus": "4"},
        env_vars=("STREAM", "SUBJECT", "SESSION"),
    ))

    register_step(StepSpec(
        name="stream_connectivity",
        description="Layer 2 connectivity stream: regress + bandpass + smooth.",
        depends_on=("preprocessing_qc",),
        variants=(),
        validate=v.validate_stream_connectivity,
        slurm_script="clean_stream.sbatch",
        slurm_resources={"time": "02:00:00", "mem": "32G", "cpus": "4"},
        env_vars=("STREAM", "SUBJECT", "SESSION"),
    ))
