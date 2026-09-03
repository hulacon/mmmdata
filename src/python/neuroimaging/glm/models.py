"""Load and validate BIDS Stats Models specs for condition-level GLMs.

The specs live in ``<repo>/models/model-<name>_smdl.json``. This module
reads the subset of the BIDS Stats Models 1.0 grammar the runner
implements — one Run node with a ``Factor`` + ``Convolve`` transformation,
t contrasts over factor levels, and one Subject node of ``Type: "meta"``
(precision-weighted fixed effects) — and refuses anything outside it with
an error that names the fix. Refusing is the point: a spec the runner
silently half-implements would fit a model nobody declared.

pybids/fitlins are not dependencies. The spec format is the contract; the
executor is ours until fitlins' maintenance is vetted (glm-strategy log,
DECIDED 2026-08-25, item 1).
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Union

MODELS_DIR = Path(__file__).resolve().parents[4] / "models"
FACTOR_PREFIX_SEP = "."


class ModelSpecError(ValueError):
    """A BIDS Stats Model the runner cannot (or must not) execute."""


@dataclasses.dataclass(frozen=True)
class Contrast:
    name: str
    #: condition (factor level, e.g. ``"adult"``) -> weight
    weights: dict[str, float]
    test: str = "t"

    @property
    def conditions(self) -> tuple[str, ...]:
        return tuple(self.weights)


@dataclasses.dataclass(frozen=True)
class StatsModel:
    name: str
    description: str
    tasks: tuple[str, ...]
    #: events column the conditions are levels of (``trial_type``)
    factor: str
    #: factor levels in the run-level design, in ``Model.X`` order
    conditions: tuple[str, ...]
    hrf_model: str
    contrasts: tuple[Contrast, ...]
    #: True when the Subject node pools runs as fixed effects
    fixed_effects: bool
    path: Path

    @property
    def task(self) -> str:
        if len(self.tasks) != 1:
            raise ModelSpecError(
                f"{self.path.name}: Input.task names {list(self.tasks)}; the runner "
                "fits one task per invocation. Split the spec."
            )
        return self.tasks[0]

    def contrast(self, name: str) -> Contrast:
        for c in self.contrasts:
            if c.name == name:
                return c
        raise KeyError(f"{self.path.name} has no contrast {name!r}; has {[c.name for c in self.contrasts]}")


def list_models(models_dir: Path = MODELS_DIR) -> list[str]:
    """Names of the shipped specs (``model-<name>_smdl.json`` -> ``name``)."""
    return sorted(p.name[len("model-") : -len("_smdl.json")] for p in models_dir.glob("model-*_smdl.json"))


def model_path(name_or_path: Union[str, Path], models_dir: Path = MODELS_DIR) -> Path:
    p = Path(name_or_path)
    if p.suffix == ".json":
        if not p.exists():
            raise FileNotFoundError(f"Model spec not found: {p}")
        return p
    candidate = models_dir / f"model-{name_or_path}_smdl.json"
    if not candidate.exists():
        raise FileNotFoundError(
            f"No model named {name_or_path!r} in {models_dir}; available: {list_models(models_dir)}"
        )
    return candidate


def load_model(name_or_path: Union[str, Path], models_dir: Path = MODELS_DIR) -> StatsModel:
    """Read a spec and validate the subset the runner implements."""
    path = model_path(name_or_path, models_dir)
    with open(path) as f:
        spec = json.load(f)
    return parse_model(spec, path)


def _level(cond: str, factor: str, where: str) -> str:
    prefix = f"{factor}{FACTOR_PREFIX_SEP}"
    if not cond.startswith(prefix):
        raise ModelSpecError(
            f"{where}: {cond!r} is not a level of the Factor variable {factor!r} "
            f"(expected {prefix}<level>)"
        )
    return cond[len(prefix) :]


def parse_model(spec: dict, path: Path = Path("<memory>")) -> StatsModel:
    where = path.name
    for key in ("Name", "BIDSModelVersion", "Nodes"):
        if key not in spec:
            raise ModelSpecError(f"{where}: missing top-level {key!r}")

    tasks = tuple(spec.get("Input", {}).get("task", ()))
    if not tasks:
        raise ModelSpecError(f"{where}: Input.task must name the task the spec applies to")

    nodes = spec["Nodes"]
    run_nodes = [n for n in nodes if n.get("Level") == "Run"]
    subject_nodes = [n for n in nodes if n.get("Level") == "Subject"]
    other = [n.get("Level") for n in nodes if n.get("Level") not in ("Run", "Subject")]
    if len(run_nodes) != 1:
        raise ModelSpecError(f"{where}: expected exactly one Run node, found {len(run_nodes)}")
    if other:
        raise ModelSpecError(
            f"{where}: node levels {other} are not implemented; the runner stops at "
            "Subject-level fixed effects (cross-subject random effects are out of "
            "scope by decision, glm-strategy log 2026-08-25)"
        )
    run = run_nodes[0]

    # --- Transformations: Factor(one variable) + Convolve(levels) ---------
    instructions = run.get("Transformations", {}).get("Instructions", [])
    factors = [i for i in instructions if i.get("Name") == "Factor"]
    convolves = [i for i in instructions if i.get("Name") == "Convolve"]
    unknown = [i.get("Name") for i in instructions if i.get("Name") not in ("Factor", "Convolve")]
    if len(factors) != 1 or len(factors[0].get("Input", [])) != 1:
        raise ModelSpecError(f"{where}: Run node needs exactly one Factor over exactly one variable")
    if len(convolves) != 1:
        raise ModelSpecError(f"{where}: Run node needs exactly one Convolve instruction")
    if unknown:
        raise ModelSpecError(
            f"{where}: transformations {unknown} are not implemented (only Factor and Convolve)"
        )
    factor = factors[0]["Input"][0]
    hrf_model = convolves[0].get("Model", "spm")

    # --- Model.X: the convolved levels plus an intercept -------------------
    model = run.get("Model", {})
    if model.get("Type", "glm") != "glm":
        raise ModelSpecError(f"{where}: Run node Model.Type must be 'glm'")
    X = model.get("X", [])
    if 1 not in X:
        raise ModelSpecError(f"{where}: Run node Model.X must include the intercept 1")
    x_conditions = [_level(x, factor, where) for x in X if x != 1]
    convolved = [_level(x, factor, where) for x in convolves[0].get("Input", [])]
    not_convolved = [c for c in x_conditions if c not in convolved]
    if not_convolved:
        raise ModelSpecError(
            f"{where}: conditions in Model.X but not Convolve.Input: {not_convolved}"
        )
    if not x_conditions:
        raise ModelSpecError(f"{where}: Model.X names no conditions")

    # --- Contrasts: t tests over the conditions in X -----------------------
    contrasts: list[Contrast] = []
    for c in run.get("Contrasts", []):
        name = c.get("Name")
        if not name or not str(name).replace("_", "").isalnum():
            raise ModelSpecError(
                f"{where}: contrast name {name!r} must be alphanumeric (it becomes a "
                "BIDS filename entity value)"
            )
        if c.get("Test", "t") != "t":
            raise ModelSpecError(f"{where}: contrast {name}: only Test 't' is implemented")
        conds = [_level(x, factor, f"{where} contrast {name}") for x in c.get("ConditionList", [])]
        weights = c.get("Weights", [])
        if len(conds) != len(weights) or not conds:
            raise ModelSpecError(
                f"{where}: contrast {name}: ConditionList ({len(conds)}) and Weights "
                f"({len(weights)}) must be the same non-zero length"
            )
        missing = [k for k in conds if k not in x_conditions]
        if missing:
            raise ModelSpecError(
                f"{where}: contrast {name} uses conditions not in Model.X: {missing}"
            )
        contrasts.append(Contrast(name=name, weights=dict(zip(conds, map(float, weights)))))
    if not contrasts:
        raise ModelSpecError(f"{where}: Run node declares no Contrasts; nothing to estimate")
    names = [c.name for c in contrasts]
    if len(set(names)) != len(names):
        raise ModelSpecError(f"{where}: duplicate contrast names: {names}")

    # --- Subject node: meta (fixed effects) or absent ----------------------
    fixed_effects = False
    if len(subject_nodes) > 1:
        raise ModelSpecError(f"{where}: more than one Subject node")
    if subject_nodes:
        sm = subject_nodes[0].get("Model", {})
        if sm.get("Type") != "meta" or sm.get("X") != [1]:
            raise ModelSpecError(
                f"{where}: Subject node must be Model {{'Type': 'meta', 'X': [1]}} — "
                "precision-weighted fixed effects over runs is the only pooling implemented"
            )
        edges = spec.get("Edges", [])
        if not any(e.get("Source") == run.get("Name") and e.get("Destination") == subject_nodes[0].get("Name") for e in edges):
            raise ModelSpecError(f"{where}: no Edge from the Run node to the Subject node")
        fixed_effects = True

    return StatsModel(
        name=str(spec["Name"]),
        description=str(spec.get("Description", "")),
        tasks=tasks,
        factor=factor,
        conditions=tuple(x_conditions),
        hrf_model=hrf_model,
        contrasts=tuple(contrasts),
        fixed_effects=fixed_effects,
        path=path,
    )
