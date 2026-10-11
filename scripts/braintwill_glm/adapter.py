"""MMMData runs → braintwill run-block designs, under the frozen reference GLM specification.

braintwill must not know this dataset exists, so everything that resolves a path, a task
label or a confound column lives here, and braintwill is imported as a library (its
``CLAUDE.md``: "Anything that resolves an MMMData path ... belongs in
``mmmdata/scripts/braintwill_*/``").

The design is the stand-in's, column for column: the same events adapter, the same
``build_design_matrix`` call under ``reference_config(regime)``, and the same data
preparation nilearn's ``FirstLevelModel`` applies with the reference settings (masked by the
runs' brain-mask intersection, then ``mean_scaling`` — each voxel divided by its temporal
mean, ×100, minus 100, so effects are percent signal change). That identity is what lets a
braintwill fit be compared with ``derivatives/nilearn_glm`` at all.

The split nilearn never makes: the model's condition columns are braintwill's **task**
columns (``S``, shared across runs); every other column — motion, aCompCor, fMRIPrep's
cosines, non-steady-state spikes, the intercept — is the run's own **nuisance** (``N``),
eliminated within the run. A condition a run never presented (adapter-derived levels in
TBencoding) is an all-zero task column there, so a contrast that needs it is not estimable
from that run alone and braintwill refuses it — the same runs the stand-in skips.

Run under mmmdata's ``.venv`` (nilearn 0.13.1, the version that wrote the stand-in tree).
"""

from __future__ import annotations

import dataclasses
import os
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src" / "python") not in sys.path:
    sys.path.insert(0, str(_REPO / "src" / "python"))


def _import_braintwill():
    """braintwill from the sibling checkout (constellation layout: every repo under one
    ``code/`` directory), or from ``$BRAINTWILL_SRC``. Loud when neither exists."""
    try:
        import braintwill  # noqa: F401
        return
    except ImportError:
        pass
    src = Path(os.environ.get("BRAINTWILL_SRC", _REPO.parent / "braintwill"))
    if not (src / "braintwill" / "__init__.py").exists():
        raise ImportError(
            f"braintwill not importable and no checkout at {src}. Set BRAINTWILL_SRC to the "
            "braintwill repository root."
        )
    sys.path.insert(0, str(src))


_import_braintwill()

from braintwill import Basis, Coefficient, CoefficientAxis, Family, RunDesign  # noqa: E402

from neuroimaging.glm.adapters import adapt_events  # noqa: E402
from neuroimaging.glm.config import GlmConfig, repetition_time  # noqa: E402
from neuroimaging.glm.design import available_contrast_vectors, build_design_matrix, strict_for  # noqa: E402
from neuroimaging.glm.models import StatsModel, load_model  # noqa: E402
from neuroimaging.glm.reference import reference_config  # noqa: E402
from neuroimaging.io import FmriprepRun, find_fmriprep_runs, load_confounds, mask_intersection  # noqa: E402

#: Units of every condition coefficient: nilearn signal_scaling=0 (the reference spec).
UNITS = "percent signal change"


@dataclasses.dataclass
class AdaptedRun:
    """One run, both ways: the nilearn design matrix the stand-in fitted, and braintwill's
    split of it."""

    run: FmriprepRun
    t_r: float
    design_matrix: pd.DataFrame
    design: RunDesign
    #: contrasts this run can estimate, as weights over the coefficient axis
    contrasts: dict[str, np.ndarray]


@dataclasses.dataclass
class AdaptedCell:
    subject: str
    model: StatsModel
    cfg: GlmConfig
    axis: CoefficientAxis
    runs: list[AdaptedRun]
    mask_img: Any
    mask: np.ndarray

    @property
    def conditions(self) -> list[str]:
        return list(self.model.conditions)

    def contrast_names(self) -> list[str]:
        return [c.name for c in self.model.contrasts]

    def load_data(self, i: int) -> np.ndarray:
        """Run ``i`` masked and mean-scaled exactly as FirstLevelModel prepares it,
        (n_scans, n_voxels) float64."""
        return load_scaled(self.runs[i].run.bold, self.mask_img)


def config_bids_root() -> Path:
    """The BIDS root from config/*.toml (local.toml overlay included) — never hard-coded."""
    from core.config import load_config

    return Path(load_config()["paths"]["bids_project_dir"])


def coefficient_axis(model: StatsModel, cfg: GlmConfig) -> CoefficientAxis:
    return CoefficientAxis(
        Coefficient(term=model.factor, level=cond, family=Family.CONDITION,
                    basis=Basis(str(cfg.hrf_model)), units=UNITS)
        for cond in model.conditions
    )


def split_design(dm: pd.DataFrame, conditions: Sequence[str]) -> RunDesign:
    """Condition columns → task ``S`` (zero where the run lacks one); the rest → nuisance."""
    n = len(dm)
    S = np.zeros((n, len(conditions)))
    for j, cond in enumerate(conditions):
        if cond in dm.columns:
            S[:, j] = dm[cond].to_numpy(dtype=float)
    nuisance = [c for c in dm.columns if c not in set(conditions)]
    return RunDesign(S=S, N=dm[nuisance].to_numpy(dtype=float), nuisance=tuple(nuisance))


def load_scaled(bold: Path, mask_img: Any) -> np.ndarray:
    from nilearn.glm.first_level.first_level import mean_scaling
    from nilearn.masking import apply_mask

    Y = apply_mask(str(bold), mask_img)
    Y, _ = mean_scaling(Y, 0)
    return np.asarray(Y, dtype=float)


def adapt(subject: str, model_name: str, *, space: str = "T1w", sessions: Optional[Sequence[str]] = None,
          regime: str = "reference", bids_root: Optional[Path] = None,
          runs: Optional[Sequence[FmriprepRun]] = None) -> AdaptedCell:
    """Every run of one subject × model (optionally restricted to ``sessions``), as the
    stand-in selected and designed them. ``runs`` overrides discovery (e.g. a stand-in's
    recorded run list)."""
    import nibabel as nib

    if bids_root is None:
        bids_root = config_bids_root()
    model = load_model(model_name)
    cfg = dataclasses.replace(reference_config(regime), space=space, hrf_model=model.hrf_model)
    subject = subject.removeprefix("sub-")
    if runs is None:
        found = find_fmriprep_runs(subject=subject, task=model.task, variant=cfg.variant, space=space,
                                   bids_root=bids_root)
        if sessions:
            wanted = {s.removeprefix("ses-") for s in sessions}
            found = [r for r in found if r.session in wanted]
        runs = found
    runs = list(runs)
    if not runs:
        raise FileNotFoundError(f"no fMRIPrep runs for sub-{subject} task-{model.task} space-{space}")
    missing = [r.entity_prefix for r in runs if r.events is None or r.bold is None or r.mask is None
               or r.confounds is None]
    if missing:
        raise FileNotFoundError("runs lacking events, BOLD, mask or confounds: " + ", ".join(missing))
    mask_img, mask = mask_intersection(runs)
    axis = coefficient_axis(model, cfg)
    all_events = adapt_events(model.adapter, [pd.read_csv(r.events, sep="\t", na_values=["n/a"]) for r in runs])
    adapted = []
    for run, events in zip(runs, all_events):
        t_r = repetition_time(run, bids_root)
        n_scans = nib.load(str(run.bold)).shape[-1]
        dm = build_design_matrix(events, load_confounds(run), t_r, n_scans, model, cfg,
                                 strict=strict_for(model))
        vectors, _ = available_contrast_vectors(model, list(dm.columns))
        on_axis = {name: _on_axis(vec, list(dm.columns), model.conditions) for name, vec in vectors.items()}
        adapted.append(AdaptedRun(run=run, t_r=t_r, design_matrix=dm,
                                  design=split_design(dm, model.conditions), contrasts=on_axis))
    return AdaptedCell(subject=subject, model=model, cfg=cfg, axis=axis, runs=adapted,
                       mask_img=mask_img, mask=mask)


def _on_axis(vec: np.ndarray, columns: list[str], conditions: Sequence[str]) -> np.ndarray:
    """A contrast over the design columns → weights over the condition axis. Refuses a
    contrast that weights a non-condition column (none of the models do)."""
    out = np.zeros(len(conditions))
    for w, col in zip(vec, columns):
        if w == 0:
            continue
        if col not in conditions:
            raise ValueError(f"contrast weights non-condition column {col!r}")
        out[list(conditions).index(col)] = w
    return out
