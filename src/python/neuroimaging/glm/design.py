"""Events + confounds -> a first-level design matrix, and contrasts over it.

Thin on purpose: nilearn builds the matrix, this module decides what goes in
(the spec's conditions, the config's confounds) and refuses the cases where a
quietly different model would result — a contrast condition the run never
presented, a confound column the TSV lacks.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .config import GlmConfig
from .models import Contrast, StatsModel


class DesignError(ValueError):
    """The design cannot be built as declared."""


def model_events(events: pd.DataFrame, model: StatsModel, *, strict: bool = True) -> pd.DataFrame:
    """The events rows the model regresses, as nilearn wants them.

    Rows whose factor level is not one of the model's conditions become the
    implicit baseline (fLoc's ``baseline`` blocks). A condition the model
    names but the run never presented is an error under ``strict`` — the
    contrast weights would then reference a column that does not exist, and
    nilearn would fit a different model without complaint.
    """
    for col in ("onset", "duration", model.factor):
        if col not in events.columns:
            raise DesignError(f"events lack column {col!r}; have {list(events.columns)}")
    levels = events[model.factor].astype(str)
    kept = events[levels.isin(model.conditions)]
    if strict:
        present = set(kept[model.factor].astype(str))
        missing = [c for c in model.conditions if c not in present]
        if missing:
            raise DesignError(
                f"conditions declared by {model.name} but absent from this run's events: "
                f"{missing}. Present levels: {sorted(set(levels))}. Either the run is "
                "truncated (check its sidecar) or the spec names the wrong levels."
            )
    out = pd.DataFrame(
        {
            "onset": kept["onset"].astype(float).to_numpy(),
            "duration": kept["duration"].astype(float).to_numpy(),
            "trial_type": kept[model.factor].astype(str).to_numpy(),
        }
    )
    return out


def confound_regressors(confounds: pd.DataFrame, cfg: GlmConfig) -> pd.DataFrame:
    """The confound columns the config asks for, NaN-free.

    fMRIPrep writes ``n/a`` for the first volume of derivative columns; the
    six raw motion parameters have none, but a caller may configure columns
    that do. Zero is the right fill for a regressor's undefined first sample.
    """
    cols = cfg.confound_columns(list(confounds.columns))
    return confounds[cols].astype(float).fillna(0.0)


def build_design_matrix(
    events: pd.DataFrame,
    confounds: Optional[pd.DataFrame],
    t_r: float,
    n_scans: int,
    model: StatsModel,
    cfg: GlmConfig,
    *,
    strict: bool = True,
) -> pd.DataFrame:
    """One run's design matrix: convolved conditions, confounds, intercept."""
    from nilearn.glm.first_level import make_first_level_design_matrix

    ev = model_events(events, model, strict=strict)
    frame_times = np.arange(n_scans) * t_r
    add_regs = None
    add_reg_names = None
    if confounds is not None:
        regs = confound_regressors(confounds, cfg)
        if len(regs) != n_scans:
            raise DesignError(
                f"confounds have {len(regs)} rows but the BOLD has {n_scans} volumes"
            )
        add_regs = regs.to_numpy()
        add_reg_names = list(regs.columns)
    dm = make_first_level_design_matrix(
        frame_times,
        events=ev,
        hrf_model=model.hrf_model if cfg.hrf_model == model.hrf_model else cfg.hrf_model,
        drift_model=cfg.drift_model,
        high_pass=cfg.high_pass,
        add_regs=add_regs,
        add_reg_names=add_reg_names,
    )
    return dm


def contrast_vector(contrast: Contrast, design_columns: list[str]) -> np.ndarray:
    """Weights over the design columns for one t contrast.

    Raises DesignError if a weighted condition has no column — which is what
    happens when ``strict=False`` let a truncated run through.
    """
    vec = np.zeros(len(design_columns))
    missing = []
    for cond, w in contrast.weights.items():
        if cond in design_columns:
            vec[design_columns.index(cond)] = w
        else:
            missing.append(cond)
    if missing:
        raise DesignError(
            f"contrast {contrast.name}: conditions {missing} have no design column; "
            f"columns are {design_columns}"
        )
    return vec


def contrast_vectors(model: StatsModel, design_columns: list[str]) -> dict[str, np.ndarray]:
    return {c.name: contrast_vector(c, design_columns) for c in model.contrasts}
