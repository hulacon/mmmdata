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
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Optional, Protocol

import numpy as np
import pandas as pd

from .config import GlmConfig

NILEARN_NOISE_MODELS = ("ols", "ar1")
REMLFIT_NOISE_MODELS = ("ols", "arma11")


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

        if cfg.noise_model not in NILEARN_NOISE_MODELS:
            raise ValueError(
                f"nilearn estimator takes noise_model in {NILEARN_NOISE_MODELS}, got "
                f"{cfg.noise_model!r}; 'arma11' is the remlfit estimator's"
            )
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


def t_to_z(t: np.ndarray, dof: float) -> np.ndarray:
    """Signed z equivalent of a t statistic with ``dof`` degrees of freedom.

    Two-tailed p mapped back through the normal, tail-symmetric, with the
    tail probability floored so a huge t gives a large finite z, not inf.
    """
    from scipy import stats

    t = np.asarray(t, dtype=float)
    p_tail = np.clip(stats.t.sf(np.abs(t), dof), 1e-300, 0.5)
    return np.sign(t) * stats.norm.isf(p_tail)


def write_afni_matrix(path: Path, design: pd.DataFrame, t_r: float, stim_columns: list[str]) -> list[str]:
    """Write a design as the ``.xmat.1D`` 3dREMLfit reads with ``-matrix``.

    The header mimics ``3dDeconvolve -x1D`` closely enough for ``-gltsym
    'SYM: ...'`` to address columns by label (verified against AFNI 24.1.22).
    Labels are the design columns with characters SYM cannot parse replaced;
    the returned list is the label per column, in order, for building GLTs.
    ``stim_columns`` are marked as stimuli (ColumnGroups 1); everything else
    is baseline (-1), which is what ``-nobout`` drops from buckets.
    """
    X = design.to_numpy(dtype=float)
    n, k = X.shape
    labels = [_sym_label(c) for c in design.columns]
    if len(set(labels)) != len(labels):
        raise ValueError(f"design column labels collide after AFNI sanitising: {labels}")
    groups = [1 if c in stim_columns else -1 for c in design.columns]
    stim_idx = [i for i, c in enumerate(design.columns) if c in stim_columns]
    header = [
        "# <matrix",
        f'#  ni_type = "{k}*double"',
        f'#  ni_dimen = "{n}"',
        '#  ColumnLabels = "' + " ; ".join(labels) + '"',
        '#  ColumnGroups = "' + ",".join(map(str, groups)) + '"',
        f'#  RowTR = "{t_r}"',
        f'#  GoodList = "0..{n - 1}"',
        f'#  NRowFull = "{n}"',
        '#  RunStart = "0"',
        f'#  Nstim = "{len(stim_idx)}"',
        '#  StimBots = "' + ",".join(str(i) for i in stim_idx) + '"',
        '#  StimTops = "' + ",".join(str(i) for i in stim_idx) + '"',
        '#  StimLabels = "' + " ; ".join(labels[i] for i in stim_idx) + '"',
        '#  CommandLine = "mmmdata neuroimaging.glm.estimators.RemlfitEstimator"',
        "# >",
    ]
    with open(path, "w") as f:
        f.write("\n".join(header) + "\n")
        np.savetxt(f, X, fmt="%.10g")
        f.write("# </matrix>\n")
    return labels


def _sym_label(column: str) -> str:
    out = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in str(column))
    return out if out[0].isalpha() or out[0] == "_" else "c_" + out


def gltsym_expression(weights: np.ndarray, labels: list[str]) -> str:
    """``SYM: +0.5*adult +0.5*child -0.5*car -0.5*instrument`` for a weight vector."""
    terms = [f"{w:+.10g}*{labels[i]}" for i, w in enumerate(weights) if w != 0]
    if not terms:
        raise ValueError("a contrast with no non-zero weight cannot be expressed as a GLT")
    return "SYM: " + " ".join(terms)


class RemlfitEstimator:
    """AFNI ``3dREMLfit``: ARMA(1,1) prewhitening per voxel, or its OLS output.

    Same inputs and outputs as :class:`NilearnEstimator`, made comparable on
    purpose: the data are smoothed with nilearn's ``smooth_img`` and scaled
    to percent signal change per voxel (nilearn's ``mean_scaling``, the
    ``signal_scaling=0`` the nilearn wrapper uses) before AFNI sees them, so
    effect maps share units across engines and the OLS outputs of the two
    engines agree to numerical precision (tested). Contrast variance is
    recovered from the bucket as (Coef / Tstat)^2; the residual dof is
    n_scans minus the design rank, which is what AFNI stamps on the t bricks.

    Needs ``3dREMLfit`` on PATH — on Talapas, ``module load afni/24.1.22``.
    """

    name = "remlfit"
    executable = "3dREMLfit"

    def __init__(self, n_threads: Optional[int] = None, keep_workdir: bool = False):
        self.n_threads = n_threads
        self.keep_workdir = keep_workdir

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
        import nibabel as nib
        from nilearn.glm.first_level.first_level import mean_scaling
        from nilearn.image import smooth_img

        exe = shutil.which(self.executable)
        if exe is None:
            raise RuntimeError(
                f"{self.executable} not on PATH; on Talapas run `module load afni/24.1.22` "
                "(or prepend /packages/afni/24.1.22) before using the remlfit estimator"
            )
        if cfg.noise_model not in REMLFIT_NOISE_MODELS:
            raise ValueError(f"remlfit takes noise_model in {REMLFIT_NOISE_MODELS}, got {cfg.noise_model!r}")
        if mask is None:
            raise ValueError("remlfit needs a brain mask image; fMRIPrep writes one per run")

        img = bold if isinstance(bold, nib.Nifti1Image) else nib.load(str(bold))
        if cfg.smoothing_fwhm:
            img = smooth_img(img, cfg.smoothing_fwhm)
        data = np.asarray(img.dataobj, dtype=np.float32)
        mask_arr = np.asarray(mask.dataobj).astype(bool)
        n_scans = data.shape[-1]
        if len(design) != n_scans:
            raise ValueError(f"design has {len(design)} rows but the BOLD has {n_scans} volumes")
        flat = data[mask_arr].T  # (time, voxels)
        flat, _ = mean_scaling(flat, axis=0)
        scaled = np.zeros_like(data)
        scaled[mask_arr] = flat.T
        dof = float(n_scans - np.linalg.matrix_rank(design.to_numpy(dtype=float)))
        weighted = np.any([np.asarray(v) != 0 for v in contrasts.values()], axis=0)
        stim_columns = [c for c, w in zip(design.columns, weighted) if w]

        workdir = Path(tempfile.mkdtemp(prefix="remlfit_", dir=os.environ.get("TMPDIR")))
        try:
            bold_path = workdir / "bold.nii.gz"
            nib.Nifti1Image(scaled, img.affine).to_filename(str(bold_path))
            mask_path = workdir / "mask.nii.gz"
            nib.Nifti1Image(mask_arr.astype(np.uint8), mask.affine).to_filename(str(mask_path))
            labels = write_afni_matrix(workdir / "design.xmat.1D", design, t_r, stim_columns)
            out_flag = "-Rglt" if cfg.noise_model == "arma11" else "-Oglt"
            bucket = workdir / "glt.nii.gz"
            cmd = [exe, "-matrix", str(workdir / "design.xmat.1D"), "-input", str(bold_path),
                   "-mask", str(mask_path), "-tout", "-noFDR", "-nobout", "-quiet", out_flag, str(bucket)]
            names = list(contrasts)
            for name in names:
                cmd += ["-gltsym", gltsym_expression(contrasts[name], labels), name]
            env = dict(os.environ)
            if self.n_threads:
                env["OMP_NUM_THREADS"] = str(self.n_threads)
            env.setdefault("AFNI_GLTSYM_PRINT", "NO")
            res = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=workdir)
            if res.returncode != 0 or not bucket.exists():
                raise RuntimeError(
                    f"3dREMLfit failed (rc {res.returncode}) in {workdir}:\n" + res.stderr[-3000:]
                )
            arr = np.asarray(nib.load(str(bucket)).dataobj, dtype=np.float32)
            arr = arr.reshape(arr.shape[:3] + (-1,))  # AFNI writes a 5-D bucket (x, y, z, 1, k)
            if arr.shape[-1] != 2 * len(names):
                raise RuntimeError(
                    f"expected {2 * len(names)} bucket bricks (Coef, Tstat per contrast), got {arr.shape[-1]}"
                )
        finally:
            if not self.keep_workdir:
                shutil.rmtree(workdir, ignore_errors=True)

        out: dict[str, ContrastEstimate] = {}
        for j, name in enumerate(names):
            coef = arr[..., 2 * j].astype(float)
            tstat = arr[..., 2 * j + 1].astype(float)
            with np.errstate(divide="ignore", invalid="ignore"):
                var = np.where(tstat != 0, (coef / tstat) ** 2, np.nan)
            z = t_to_z(tstat, dof)
            for a in (coef, tstat, var, z):
                a[~mask_arr] = 0.0
            var[~mask_arr] = np.nan
            mk = lambda a: nib.Nifti1Image(a.astype(np.float32), img.affine)  # noqa: E731
            out[name] = ContrastEstimate(effect=mk(coef), variance=mk(var), dof=dof, stat=mk(tstat), z=mk(z))
        return out


ESTIMATORS: dict[str, type] = {"nilearn": NilearnEstimator, "remlfit": RemlfitEstimator}

#: The bake-off's engine factor: name -> (estimator, noise_model).
ENGINES: dict[str, tuple[str, str]] = {
    "nilearn-ols": ("nilearn", "ols"),
    "nilearn-ar1": ("nilearn", "ar1"),
    "remlfit-arma11": ("remlfit", "arma11"),
}


def get_estimator(name: str) -> Estimator:
    try:
        return ESTIMATORS[name]()
    except KeyError:
        raise KeyError(f"Unknown estimator {name!r}; available: {sorted(ESTIMATORS)}") from None
