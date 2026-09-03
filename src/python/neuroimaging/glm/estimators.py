"""The estimator interface, its nilearn implementation, and fixed effects.

The interface is first-class by decision (glm-strategy log, 2026-08-25):
(design matrix, data, covariance model) -> (effect, variance, dof) per
contrast. Every engine in the bake-off — this nilearn wrapper, a future
REMLfit wrapper, braintwill's GLS core — implements :class:`Estimator`, and
the harness that scores them lives here in mmmdata so braintwill is never
both contestant and referee.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Optional, Protocol

import numpy as np
import pandas as pd

from .config import GlmConfig


@dataclasses.dataclass(frozen=True)
class ContrastEstimate:
    """One contrast's estimate for one run: the fixed-effects inputs."""

    effect: Any  # nibabel image
    variance: Any  # nibabel image
    dof: Optional[float] = None
    #: optional convenience maps the engine already computed
    stat: Any = None
    z: Any = None


class Estimator(Protocol):
    """Fit one run and return every requested contrast."""

    name: str

    def fit_run(
        self,
        bold: Any,
        design: pd.DataFrame,
        contrasts: dict[str, np.ndarray],
        *,
        t_r: float,
        mask: Any = None,
        cfg: GlmConfig,
    ) -> dict[str, ContrastEstimate]: ...


class NilearnEstimator:
    """nilearn ``FirstLevelModel`` with the config's noise model and smoothing.

    ``noise_model="ar1"`` is prewhitening; ``"ols"`` is the iid control. The
    design matrix is built by :mod:`.design` and passed in whole, so what
    nilearn fits is exactly what was declared.
    """

    name = "nilearn"

    def fit_run(
        self,
        bold: Any,
        design: pd.DataFrame,
        contrasts: dict[str, np.ndarray],
        *,
        t_r: float,
        mask: Any = None,
        cfg: GlmConfig,
    ) -> dict[str, ContrastEstimate]:
        from nilearn.glm.first_level import FirstLevelModel

        # t_r, hrf_model and drift_model are deliberately NOT passed: the
        # design matrix built by `design.build_design_matrix` already encodes
        # them, and nilearn ignores (and warns about) the constructor values
        # when a design is supplied. `t_r` stays in the signature because the
        # interface is engine-neutral and other engines need it.
        del t_r
        flm = FirstLevelModel(
            noise_model=cfg.noise_model,
            smoothing_fwhm=cfg.smoothing_fwhm,
            mask_img=mask if mask is not None else False,
            minimize_memory=False,
            standardize=False,
            signal_scaling=0,
        )
        flm.fit(bold, design_matrices=design)
        out: dict[str, ContrastEstimate] = {}
        for name, vec in contrasts.items():
            maps = flm.compute_contrast(vec, stat_type="t", output_type="all")
            out[name] = ContrastEstimate(
                effect=maps["effect_size"],
                variance=maps["effect_variance"],
                dof=_dof(flm),
                stat=maps.get("stat"),
                z=maps.get("z_score"),
            )
        return out


def _dof(flm: Any) -> Optional[float]:
    """Residual degrees of freedom of a fitted FirstLevelModel, if exposed."""
    try:
        results = flm.results_[0]
        first = next(iter(results.values()))
        return float(first.df_residuals)
    except Exception:
        return None


@dataclasses.dataclass(frozen=True)
class FixedEffectsResult:
    effect: Any
    variance: Any
    stat: Any
    z: Any
    n_runs: int


def fixed_effects(estimates: list[ContrastEstimate], mask: Any = None) -> FixedEffectsResult:
    """Precision-weighted fixed effects across runs (nilearn ``compute_fixed_effects``).

    One run is passed through unchanged rather than "pooled", so a
    single-run subject still gets the same artifact set.
    """
    if not estimates:
        raise ValueError("fixed_effects needs at least one run")
    from nilearn.glm.contrasts import compute_fixed_effects

    effects = [e.effect for e in estimates]
    variances = [e.variance for e in estimates]
    dofs = [e.dof for e in estimates]
    kwargs: dict[str, Any] = {"mask": mask, "precision_weighted": True}
    if all(d is not None for d in dofs):
        kwargs["dofs"] = dofs
    try:
        res = compute_fixed_effects(effects, variances, return_z_score=True, **kwargs)
    except TypeError:  # older nilearn without return_z_score
        res = compute_fixed_effects(effects, variances, **kwargs)
    fx_effect, fx_variance, fx_stat = res[0], res[1], res[2]
    fx_z = res[3] if len(res) > 3 else None
    return FixedEffectsResult(
        effect=fx_effect, variance=fx_variance, stat=fx_stat, z=fx_z, n_runs=len(estimates)
    )


ESTIMATORS: dict[str, type] = {"nilearn": NilearnEstimator}


def get_estimator(name: str) -> Estimator:
    try:
        return ESTIMATORS[name]()
    except KeyError:
        raise KeyError(f"Unknown estimator {name!r}; available: {sorted(ESTIMATORS)}") from None
