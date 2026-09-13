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
#: FILM noise models. ``--sa`` pools only the Tukey path, so ``tukey`` /
#: ``tukey-smoothed`` is the pooling pair and ``ar1`` has no pooled twin
#: (MEASURED 2026-09-12; see :meth:`FilmEstimator.flags`).
FILM_NOISE_MODELS = ("ols", "ar1", "tukey", "tukey-smoothed")


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

    After an ``ar1`` fit, :attr:`last_noise_map` holds that run's per-voxel
    AR(1) coefficient as an image (``None`` after an ``ols`` fit). It is a
    diagnostic, not an output of the estimator interface: the runner writes
    it only under ``--keep-per-run``.
    """

    name = "nilearn"

    def __init__(self) -> None:
        self.last_noise_map: Any = None

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

        self.last_noise_map = None
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
        self.last_noise_map = _ar1_map(flm) if cfg.noise_model == "ar1" else None
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


def _ar1_map(flm: Any) -> Any:
    """One run's per-voxel AR(1) coefficient, as an image in the fit's mask.

    nilearn bins the Yule-Walker AR(1) estimate and carries it as the voxel's
    ``labels_`` entry so voxels sharing a coefficient share a whitening matrix
    (``nilearn/glm/first_level/first_level.py``, ``run_glm``); the label *is*
    the coefficient, binned to 0.01. Recovering it costs nothing, and it is
    the only per-voxel noise parameter nilearn exposes.

    Returns ``None`` rather than raising if nilearn's internals move: this is
    a diagnostic and must never fail a fit.
    """
    import numpy as np

    try:
        labels = flm.labels_[0]
        ar = np.asarray([float(v) for v in np.asarray(labels).ravel()], dtype=np.float32)
        return flm.masker_.inverse_transform(ar)
    except Exception:
        return None


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


# ------------------------------------------------------------------ FSL FILM
def write_fsl_design(path: Path, design: pd.DataFrame) -> None:
    """One run's design as the ``.mat`` ``film_gls --pd`` reads.

    ``PPheights`` is each column's peak-to-peak range (1.0 for a constant
    column, whose range is 0); FILM uses it only to report a required effect
    size, never in the fit.
    """
    X = design.to_numpy(dtype=float)
    heights = np.ptp(X, axis=0)
    heights[heights == 0] = 1.0
    with open(path, "w") as f:
        f.write(f"/NumWaves\t{X.shape[1]}\n/NumPoints\t{X.shape[0]}\n")
        f.write("/PPheights\t" + "\t".join(f"{h:.10g}" for h in heights) + "\n\n/Matrix\n")
        np.savetxt(f, X, fmt="%.10g", delimiter="\t")


def write_fsl_contrasts(path: Path, contrasts: dict[str, np.ndarray], n_waves: int) -> list[str]:
    """The ``.con`` for ``film_gls --con``; returns the contrast names in file order.

    FILM numbers its outputs ``cope1..copeK`` by row, so the returned order is
    what maps a brick back to a contrast name.
    """
    names = list(contrasts)
    rows = []
    for name in names:
        vec = np.asarray(contrasts[name], dtype=float)
        if vec.shape != (n_waves,):
            raise ValueError(f"contrast {name!r} has {vec.shape} weights for a {n_waves}-column design")
        rows.append(vec)
    with open(path, "w") as f:
        for i, name in enumerate(names, start=1):
            f.write(f"/ContrastName{i}\t{name}\n")
        f.write(f"/NumWaves\t{n_waves}\n/NumContrasts\t{len(names)}\n")
        f.write("/PPheights\t" + "\t".join("1" for _ in names) + "\n")
        f.write("/RequiredEffect\t" + "\t".join("1" for _ in names) + "\n\n/Matrix\n")
        np.savetxt(f, np.vstack(rows), fmt="%.10g", delimiter=" ")
    return names


#: Data are handed to FILM in percent signal change plus this offset, so the
#: mean-based ``--thr`` mask is exactly the brain mask (out-of-mask voxels are
#: left at 0). A contrast never weights the intercept, so the offset changes
#: no estimate (verified by the OLS-equivalence test).
FILM_OFFSET = 100.0
FILM_THRESHOLD = 10.0


class FilmEstimator:
    """FSL ``film_gls``: FILM prewhitening, per voxel or spatially pooled.

    The Track A engine (glm-strategy pass-2 design, 2026-09-11). Four noise
    models, which differ only in how the autocorrelation is estimated:

    ``ols``
        ``--noest`` — no autocorrelation estimated. The contract arm: it
        reproduces analytic (and nilearn) OLS to numerical precision, which
        is what makes the other three comparable to the pass-1 table.
    ``ar1``
        ``--ar`` — AR(1) per voxel, nothing pooled. Same noise model as
        ``nilearn-ar1`` in a different implementation, so the pair measures
        implementation rather than model. It has no pooled twin: see
        :meth:`flags`.
    ``tukey`` / ``tukey-smoothed``
        FILM's default Tukey taper (M = sqrt(n_scans)), unpooled and pooled.
        This is the pooling pair — same taper, ``--sa`` off and on — and
        ``tukey-smoothed`` is FEAT's production setting.

    Comparability, as for :class:`RemlfitEstimator`: the data are smoothed
    with nilearn's ``smooth_img`` and scaled to percent signal change per
    voxel (``signal_scaling=0``) before FILM sees them, so effect maps share
    units with the nilearn engine.

    **Divergence from FEAT, deliberate.** FEAT passes ``--epith``, a
    brightness threshold that makes ``--sa`` a SUSAN (edge-preserving)
    smooth. Per-voxel percent-signal-change data are spatially flat by
    construction — every in-mask voxel has the same mean — so brightness
    gating has nothing to act on and is not passed. ``--sa`` is therefore
    unweighted local pooling of the autocorrelation estimates, which is the
    pooling the track is about, without a tissue-boundary confound.

    Needs ``film_gls`` on PATH — on Talapas, ``module load fsl/6.0.7.9``.
    """

    name = "film"
    executable = "film_gls"

    def __init__(self, susan_mask_size: int = 5, keep_workdir: bool = False):
        self.susan_mask_size = susan_mask_size
        self.keep_workdir = keep_workdir

    def flags(self, noise_model: str) -> list[str]:
        """The ``film_gls`` flags for one noise model (its whole definition).

        ``ar1-smoothed`` is refused rather than silently accepted: FILM
        applies ``--sa`` inside the Tukey/multitaper estimator only, so
        ``--ar --sa`` is byte-identical to ``--ar`` (MEASURED 2026-09-12 on
        synthetic AR(0.4) data — varcope max |difference| exactly 0, while
        the same pair on the Tukey path cuts the variance-map SD from 0.0156
        to 0.0085). A caller asking for a pooled AR(1) wants the Tukey pair.
        """
        if noise_model == "ar1-smoothed":
            raise ValueError(
                "film's --sa does not pool the --ar path (`--ar --sa` is identical to `--ar`); "
                "the pooling pair within FILM is 'tukey' vs 'tukey-smoothed'"
            )
        if noise_model not in FILM_NOISE_MODELS:
            raise ValueError(f"film takes noise_model in {FILM_NOISE_MODELS}, got {noise_model!r}")
        flags = {"ols": ["--noest"], "ar1": ["--ar"], "tukey": [], "tukey-smoothed": []}[noise_model]
        if noise_model.endswith("-smoothed"):
            flags = flags + ["--sa", f"--ms={self.susan_mask_size}"]
        return flags

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

        del t_r  # the design matrix already encodes it; the interface is engine-neutral
        exe = shutil.which(self.executable)
        if exe is None:
            raise RuntimeError(
                f"{self.executable} not on PATH; on Talapas run `module load fsl/6.0.7.9` "
                "(or prepend $FSLDIR/bin) before using the film estimator"
            )
        flags = self.flags(cfg.noise_model)
        if mask is None:
            raise ValueError("film needs a brain mask image; fMRIPrep writes one per run")

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
        scaled[mask_arr] = flat.T + FILM_OFFSET

        workdir = Path(tempfile.mkdtemp(prefix="film_", dir=os.environ.get("TMPDIR")))
        try:
            bold_path = workdir / "bold.nii.gz"
            nib.Nifti1Image(scaled, img.affine).to_filename(str(bold_path))
            write_fsl_design(workdir / "design.mat", design)
            names = write_fsl_contrasts(workdir / "design.con", contrasts, design.shape[1])
            results = workdir / "stats"
            cmd = [exe, f"--in={bold_path}", f"--rn={results}", f"--pd={workdir / 'design.mat'}",
                   f"--con={workdir / 'design.con'}", f"--thr={FILM_THRESHOLD}"] + flags
            res = subprocess.run(cmd, capture_output=True, text=True, cwd=workdir)
            dof_file = results / "dof"
            if res.returncode != 0 or not dof_file.exists():
                raise RuntimeError(
                    f"film_gls failed (rc {res.returncode}) in {workdir}:\n"
                    + (res.stderr or res.stdout)[-3000:]
                )
            dof = float(dof_file.read_text().strip())
            out: dict[str, ContrastEstimate] = {}
            for j, name in enumerate(names, start=1):
                maps = {}
                for stat, stem in (("effect", "cope"), ("variance", "varcope"), ("stat", "tstat"), ("z", "zstat")):
                    path = results / f"{stem}{j}.nii.gz"
                    if not path.exists():
                        raise RuntimeError(f"film_gls wrote no {path.name} for contrast {name!r} in {workdir}")
                    arr = np.asarray(nib.load(str(path)).dataobj, dtype=np.float32)
                    arr[~mask_arr] = np.nan if stat == "variance" else 0.0
                    maps[stat] = nib.Nifti1Image(arr, img.affine)
                out[name] = ContrastEstimate(dof=dof, **maps)
        finally:
            if not self.keep_workdir:
                shutil.rmtree(workdir, ignore_errors=True)
        return out


ESTIMATORS: dict[str, type] = {
    "nilearn": NilearnEstimator,
    "remlfit": RemlfitEstimator,
    "film": FilmEstimator,
}

#: The bake-off's engine factor: name -> (estimator, noise_model).
ENGINES: dict[str, tuple[str, str]] = {
    "nilearn-ols": ("nilearn", "ols"),
    "nilearn-ar1": ("nilearn", "ar1"),
    "remlfit-arma11": ("remlfit", "arma11"),
    # Track A (pass-2 design, 2026-09-11, corrected 2026-09-12): the pooling
    # contrast is `film-tukey` vs `film-smoothed` — same taper, --sa off and
    # on — because FILM pools only the Tukey path. `film-pervoxel` (--ar) is
    # the implementation bridge to `nilearn-ar1`: same noise model, other code.
    "film-pervoxel": ("film", "ar1"),
    "film-tukey": ("film", "tukey"),
    "film-smoothed": ("film", "tukey-smoothed"),
}


def get_estimator(name: str) -> Estimator:
    try:
        return ESTIMATORS[name]()
    except KeyError:
        raise KeyError(f"Unknown estimator {name!r}; available: {sorted(ESTIMATORS)}") from None
