"""Tier 1 GLM rows of the data-quality collection: T1.5 task R², T1.6 localizer betas, T1.8 motion–task r.

The stand-in GLM (nilearn OLS + SPM canonical, the frozen reference spec) fitted
to one run under each confound regime: the regime supplies the confound and drift
columns through :func:`neuroimaging.glm.reference.reference_config`, and nothing
else about the model changes. Measure registry v0.2 (mmmdata-agents
``docs/archive/workbench/data-quality/out/measure-registry.md``); design choices
D1–D4 DECIDED 2026-09-29 (data-quality log).

* **Which runs.** Every run with an events file, except the tasks in
  :data:`EXCLUDED_TASKS`. Localizers with a shipped model (:data:`LOCALIZER_MODELS`)
  use it; every other task uses the generic rule: one regressor per
  ``trial_type`` except the baseline types in :data:`BASELINE_TYPES`.
* **T1.5, task R²** is voxelwise partial R²: the share of the variance left
  after the regime's nuisance columns (confounds, drift, intercept, one spike per
  lead-in volume) that the task regressors explain. The headline is the
  **adjusted** form, ``1 - (RSS_full / dof_full) / (RSS_nuisance / dof_nuisance)``,
  whose null expectation is ~0 under every regime. The raw form,
  ``1 - RSS_full / RSS_nuisance``, has null expectation ~``k / dof_nuisance``
  (k task columns), so it rises with every nuisance column a regime adds and
  would rank regimes by their regressor count (pilot 2026-09-29: floc's median
  raw R² sat at that chance level, 0.053 → 0.061 from ``none`` to ``acc20``).
  Even the adjusted form's median and p99 skew with dof on short runs (the median
  of an F ratio lies below its mean), so the **headline is the fraction of
  voxels whose task-block partial F has p < :data:`F_ALPHA`**, whose null level is
  ``F_ALPHA`` at any dof (DECIDED Ben 2026-09-29: keep all three, F fraction
  first). It assumes iid noise: regimes that leave slow drift (``none``, ``gsr``)
  leave more autocorrelated residuals and so more false positives. Summarised
  over the brain mask and per parcel; no map is written.
* **T1.6, localizer betas** (localizer runs only): every condition's beta as a
  4D NIfTI (volume order in the sidecar), the residual variance ``sigma²`` as a
  3D NIfTI, and the unscaled covariance of the condition betas in the sidecar,
  so tier 2 can form any contrast and its variance, ``c'β`` and
  ``sigma² c' C c``, without refitting. Units are percent signal change, as
  nilearn's ``signal_scaling=0`` gives the stand-in. Percent signal change is
  undefined where a voxel's temporal mean is not meaningfully positive: MNI
  resampling leaves ~0.1 % of brain-mask voxels at the mask edge with means
  near or below zero (down to -3,477 on the pilot run), where the scaled betas
  reach ~10⁴ and the float32/float64 choice alone moves them by thousands. Betas
  and ``sigma²`` are NaN (masked, never filled) where the mean is below
  :data:`PSC_FLOOR_FRACTION` of the run's median in-mask mean; R², being
  scale-free, is kept. Elsewhere the fit equals nilearn's to ~1e-5
  (pilot 2026-09-29, sub-03 ses-03 floc run-01: r = 0.9999999999998).
* **T1.8, motion–task r** is regime-free: the largest ``|r|`` between any of
  the six motion columns and any convolved task regressor, over the steady-state
  volumes. It lands in ``tier1_motion.tsv``, not here.

Per run × regime, beside the tier-1 cleaning outputs in ``sub-XX/ses-YY/func/``::

    <prefix>_space-S_desc-<regime>_glm.json            T1.5 summaries, design facts, provenance
    <prefix>_space-S_desc-<regime>_glm.tsv             per-parcel T1.5 (+ T1.6 contrast means)
    <prefix>_space-S_desc-<regime>_betas.nii.gz        T1.6 only: 4D condition betas
    <prefix>_space-S_desc-<regime>_sigmasquared.nii.gz T1.6 only: residual variance

or, for a run that cannot carry the regime, the declared-absent pair
``_glmabsent.tsv`` / ``_glmabsent.json`` (same rule as the cleaning cells).
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from . import data_quality as dq
from .confounds import Regime, RegimeNotApplicable, non_steady_state_mask, regime_design
from .glm.design import build_design_matrix, contrast_vectors
from .glm.models import StatsModel, load_model
from .glm.reference import reference_config
from .io import FmriprepRun

SCHEMA_VERSION = "1.1"  # 1.1: task-F fraction (2026-09-29)
TABLE_NAME = "tier1_glm"
PARCELS_TABLE_NAME = "tier1_glm_parcels"

#: trial_type levels the generic rule leaves in the implicit baseline.
BASELINE_TYPES: frozenset[str] = frozenset({"rest", "fixation", "baseline", "blank"})

#: task -> shipped BIDS Stats Model whose conditions and contrasts T1.5/T1.6 use.
LOCALIZER_MODELS: dict[str, str] = {"floc": "floc", "motor": "motor", "tone": "tone"}

#: Tasks with events that get no GLM row, and why (D4, 2026-09-29).
EXCLUDED_TASKS: dict[str, str] = {
    "auditory": "one ~562 s stimulus block then ~50 s of fixation: the only task regressor is a step "
                "nearly collinear with linear + quadratic drift, so its fit would measure the drift "
                "model, not the regime",
    "fixation": "a single `calibration` event spanning the run: no task to model",
}

#: T1.5's headline threshold: a voxel counts when its task-block partial F has p below this.
F_ALPHA = 0.001

#: A voxel whose temporal mean is below this fraction of the run's median in-mask mean
#: gets no beta: its percent signal change is undefined or explosive (see module doc).
PSC_FLOOR_FRACTION = 0.01

MOTION_COLUMNS: tuple[str, ...] = ("trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z")


# ---------------------------------------------------------------------------
# Which runs, which model
# ---------------------------------------------------------------------------

def eligible(run: FmriprepRun) -> bool:
    """True when the run has events and its task is not excluded."""
    return run.events is not None and run.task not in EXCLUDED_TASKS


def read_events(run: FmriprepRun) -> pd.DataFrame:
    return pd.read_csv(run.events, sep="\t", na_values=["n/a"])


def generic_model(task: str, events: pd.DataFrame) -> StatsModel:
    """One condition per non-baseline ``trial_type`` the run presented, in sorted order."""
    levels = sorted(set(events["trial_type"].dropna().astype(str)) - BASELINE_TYPES)
    if not levels:
        raise ValueError(f"task {task!r}: no trial_type outside the baseline types {sorted(BASELINE_TYPES)}")
    return StatsModel(
        name=f"generic-{task}",
        description="data-quality generic rule: one regressor per non-baseline trial_type",
        tasks=(task,),
        factor="trial_type",
        conditions=tuple(levels),
        hrf_model="spm",
        contrasts=(),
        fixed_effects=False,
        path=Path(f"<generic {task}>"),
    )


def model_for(run: FmriprepRun, events: pd.DataFrame) -> StatsModel:
    if run.task in LOCALIZER_MODELS:
        return load_model(LOCALIZER_MODELS[run.task])
    return generic_model(run.task, events)


def model_sha256(model: StatsModel) -> str:
    """The shipped spec's file hash, or the generic rule's identity (its condition list)."""
    if model.name.startswith("generic-"):
        import hashlib

        return hashlib.sha256(json.dumps([model.name, list(model.conditions)]).encode()).hexdigest()
    return dq.file_sha256(model.path)


def design_for(run: FmriprepRun, events: pd.DataFrame, confounds: pd.DataFrame, tr: float,
               model: StatsModel, regime: Regime) -> pd.DataFrame:
    """The stand-in's design matrix under ``regime``; raises RegimeNotApplicable like the cleaner.

    ``regime_design`` is called first only for its checks: it raises
    :class:`RegimeNotApplicable` for a run with too few aCompCor components, where
    GlmConfig would raise a plain KeyError, so both tiers declare the same cells absent.
    """
    regime_design(regime, confounds)
    strict = run.task in LOCALIZER_MODELS
    return build_design_matrix(events, confounds, tr, len(confounds), model,
                               reference_config(regime.name), strict=strict)


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------

def percent_signal_change(Y: np.ndarray) -> np.ndarray:
    """nilearn's ``mean_scaling`` (``signal_scaling=0``): ``100 * (Y / mean - 1)`` per voxel.

    The mean is over every volume, lead-in included, as nilearn takes it. A voxel
    whose mean is 0 is scaled by 1, as nilearn does.
    """
    mean = Y.mean(axis=0)
    mean = np.where(mean == 0, 1.0, mean)
    return 100.0 * (Y / mean - 1.0)


@dataclasses.dataclass
class GlmFit:
    conditions: tuple[str, ...]
    betas: np.ndarray  # (n_conditions, n_voxels) float32, percent signal change
    sigma2: np.ndarray  # (n_voxels,) residual variance, RSS / dof
    cov_unscaled: np.ndarray  # (n_conditions, n_conditions): [(X'X)^+] on the condition block
    r2_task: np.ndarray  # (n_voxels,) partial R² of the task block (raw)
    r2_task_adj: np.ndarray  # (n_voxels,) the same, adjusted for both models' residual dof
    task_p: np.ndarray  # (n_voxels,) p of the task-block partial F, (df_task, dof)
    df_task: int
    psc_defined: np.ndarray  # (n_voxels,) bool: mean above the PSC floor; betas/sigma2 NaN elsewhere
    psc_floor: float
    dof: int
    dof_nuisance: int
    rank: int
    n_regressors: int


def fit(data: np.ndarray, design: pd.DataFrame, conditions: tuple[str, ...], chunk: int = 20_000) -> GlmFit:
    """OLS of every voxel (``data`` is n_vol × n_voxels) on ``design``, and on its nuisance part alone.

    The same model nilearn's OLS fits (pseudo-inverse, ``dof = n - rank``); the
    nuisance-only fit gives the partial R² of the task block.
    """
    X = design.to_numpy(dtype=np.float64)
    n, p = X.shape
    if len(data) != n:
        raise ValueError(f"design has {n} rows but the data has {len(data)} volumes")
    cond_idx = [design.columns.get_loc(c) for c in conditions]
    Xn = np.delete(X, cond_idx, axis=1)
    from scipy.stats import f as f_dist

    rank = int(np.linalg.matrix_rank(X))
    dof = n - rank
    dof_n = n - int(np.linalg.matrix_rank(Xn))
    df_task = dof_n - dof
    if df_task < 1:
        raise ValueError("the task columns add no rank to the nuisance design")
    if dof < 1:
        raise ValueError(f"the design leaves {dof} residual degrees of freedom on a {n}-volume run")
    pinv = np.linalg.pinv(X)
    pinv_n = np.linalg.pinv(Xn)
    cov = (pinv @ pinv.T)[np.ix_(cond_idx, cond_idx)]
    n_vox = data.shape[1]
    means = data.mean(axis=0, dtype=np.float64)
    positive = means[means > 0]
    floor = PSC_FLOOR_FRACTION * float(np.median(positive)) if positive.size else np.inf
    defined = means > floor
    betas = np.empty((len(conditions), n_vox), dtype=np.float32)
    sigma2 = np.empty(n_vox)
    r2 = np.empty(n_vox)
    r2_adj = np.empty(n_vox)
    task_p = np.empty(n_vox)
    for start in range(0, n_vox, chunk):
        sl = slice(start, min(start + chunk, n_vox))
        Y = percent_signal_change(data[:, sl].astype(np.float64))
        B = pinv @ Y
        R = Y - X @ B
        rss = np.einsum("ij,ij->j", R, R)
        Rn = Y - Xn @ (pinv_n @ Y)
        rss_n = np.einsum("ij,ij->j", Rn, Rn)
        betas[:, sl] = B[cond_idx].astype(np.float32)
        sigma2[sl] = rss / dof
        with np.errstate(divide="ignore", invalid="ignore"):
            r2[sl] = 1.0 - rss / rss_n
            r2_adj[sl] = 1.0 - (rss / dof) / (rss_n / dof_n)
            fstat = ((rss_n - rss) / df_task) / (rss / dof)
        task_p[sl] = f_dist.sf(fstat, df_task, dof)
    r2[~np.isfinite(r2)] = np.nan
    r2_adj[~np.isfinite(r2_adj)] = np.nan
    task_p[~np.isfinite(task_p)] = np.nan
    betas[:, ~defined] = np.nan
    sigma2[~defined] = np.nan
    return GlmFit(conditions=tuple(conditions), betas=betas, sigma2=sigma2, cov_unscaled=cov,
                  r2_task=r2, r2_task_adj=r2_adj, task_p=task_p, df_task=df_task, psc_defined=defined, psc_floor=floor,
                  dof=dof, dof_nuisance=dof_n, rank=rank, n_regressors=p)


def contrast(fitres: GlmFit, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(effect, variance)`` per voxel for condition weights ``weights`` (one per condition)."""
    w = np.asarray(weights, dtype=np.float64)
    effect = w @ fitres.betas.astype(np.float64)
    variance = fitres.sigma2 * float(w @ fitres.cov_unscaled @ w)
    return effect, variance


def condition_weights(model: StatsModel, design_columns: list[str], conditions: tuple[str, ...]) -> dict[str, np.ndarray]:
    """The model's contrasts as weights over ``conditions`` (the design's own vectors, restricted)."""
    idx = [design_columns.index(c) for c in conditions]
    out = {}
    for name, vec in contrast_vectors(model, design_columns).items():
        full = np.asarray(vec, dtype=np.float64)
        if np.any(np.delete(full, idx)):
            raise ValueError(f"contrast {name!r} weights a non-condition column; T1.6 contrasts are condition-only")
        out[name] = full[idx]
    return out


# ---------------------------------------------------------------------------
# T1.8
# ---------------------------------------------------------------------------

def motion_task_correlation(design: pd.DataFrame, conditions: tuple[str, ...],
                            confounds: pd.DataFrame) -> dict[str, Any]:
    """Largest |r| between a motion column and a convolved task regressor, over steady-state volumes."""
    keep = ~non_steady_state_mask(confounds)
    motion = confounds[list(MOTION_COLUMNS)].astype(float).fillna(0.0).to_numpy()[keep]
    best = {"motion_task_r_max": float("nan"), "motion_task_r_motion": None, "motion_task_r_condition": None}
    for cond in conditions:
        reg = design[cond].to_numpy(dtype=np.float64)[keep]
        if reg.std() == 0:
            continue
        for j, col in enumerate(MOTION_COLUMNS):
            m = motion[:, j]
            if m.std() == 0:
                continue
            r = abs(float(np.corrcoef(reg, m)[0, 1]))
            if not np.isfinite(best["motion_task_r_max"]) or r > best["motion_task_r_max"]:
                best = {"motion_task_r_max": r, "motion_task_r_motion": col, "motion_task_r_condition": cond}
    return best


# ---------------------------------------------------------------------------
# Output layout and currency
# ---------------------------------------------------------------------------

def _paths(tree_root: Path, run: FmriprepRun, regime: str, suffix: str, ext: str) -> Path:
    return dq.output_dir(tree_root, run) / f"{dq.output_stem(run, regime)}_{suffix}{ext}"


def glm_paths(tree_root: Path, run: FmriprepRun, regime: str) -> tuple[Path, Path]:
    return _paths(tree_root, run, regime, "glm", ".tsv"), _paths(tree_root, run, regime, "glm", ".json")


def beta_paths(tree_root: Path, run: FmriprepRun, regime: str) -> tuple[Path, Path]:
    return (_paths(tree_root, run, regime, "betas", ".nii.gz"),
            _paths(tree_root, run, regime, "sigmasquared", ".nii.gz"))


def glm_absent_paths(tree_root: Path, run: FmriprepRun, regime: str) -> tuple[Path, Path]:
    return _paths(tree_root, run, regime, "glmabsent", ".tsv"), _paths(tree_root, run, regime, "glmabsent", ".json")


def _outputs(tree_root: Path, run: FmriprepRun, regime: str) -> list[Path]:
    return [*glm_paths(tree_root, run, regime), *beta_paths(tree_root, run, regime)]


@dataclasses.dataclass(frozen=True)
class GlmKeys:
    """The input identities a GLM cell is current against."""

    bold_sha256: str
    events_sha256: str
    model_sha256: str
    atlases_sha256: str


def glm_cell_exists(tree_root: Path, run: FmriprepRun, regime: str) -> bool:
    return all(p.exists() for p in glm_paths(tree_root, run, regime)) or all(
        p.exists() for p in glm_absent_paths(tree_root, run, regime))


def glm_is_current(tree_root: Path, run: FmriprepRun, regime: Regime, keys: GlmKeys) -> bool:
    absent = all(p.exists() for p in glm_absent_paths(tree_root, run, regime.name))
    if absent:
        js = glm_absent_paths(tree_root, run, regime.name)[1]
    else:
        if not all(p.exists() for p in glm_paths(tree_root, run, regime.name)):
            return False
        js = glm_paths(tree_root, run, regime.name)[1]
    try:
        meta = json.loads(js.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    if not absent and meta.get("localizer") and not all(p.exists() for p in beta_paths(tree_root, run, regime.name)):
        return False
    return (
        meta.get("schema_version") == SCHEMA_VERSION
        and meta.get("regime_version") == regime.version
        and meta.get("input_bold_sha256") == keys.bold_sha256
        and meta.get("input_events_sha256") == keys.events_sha256
        and meta.get("model_sha256") == keys.model_sha256
        and (absent or meta.get("input_atlases_sha256") == keys.atlases_sha256)
    )


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def _summary(x: np.ndarray, name: str) -> dict[str, float]:
    finite = x[np.isfinite(x)]
    stats = {"mean": np.mean, "median": np.median,
             "p90": lambda v: np.percentile(v, 90), "p99": lambda v: np.percentile(v, 99)}
    return {f"{name}_{k}": (float(f(finite)) if finite.size else float("nan")) for k, f in stats.items()}


def _frac_sig(p: np.ndarray) -> float:
    finite = p[np.isfinite(p)]
    return float((finite < F_ALPHA).mean()) if finite.size else float("nan")


def parcel_table(inputs: dq.RunInputs, fitres: GlmFit, effects: dict[str, np.ndarray]) -> pd.DataFrame:
    """One row per parcel of every tier-1 parcellation: mean task R², and each contrast's mean effect."""
    rows = []
    for seg, parc in inputs.parcellations.items():
        labels = parc.labels_in_mask
        for idx, name, n_mask in zip(parc.table["index"], parc.table["name"], parc.table["n_voxels_mask"]):
            sel = labels == int(idx)
            row: dict[str, Any] = {"atlas": seg, "parcel": str(name), "index": int(idx), "n_voxels_mask": int(n_mask)}
            if sel.any():
                row["task_frac_p001"] = _frac_sig(fitres.task_p[sel])
                row["task_r2adj_mean"] = float(np.nanmean(fitres.r2_task_adj[sel]))
                row["task_r2_mean"] = float(np.nanmean(fitres.r2_task[sel]))
                for cname, eff in effects.items():
                    row[f"effect_{cname}"] = float(np.nanmean(eff[sel]))
            else:
                row["task_frac_p001"] = float("nan")
                row["task_r2adj_mean"] = float("nan")
                row["task_r2_mean"] = float("nan")
                for cname in effects:
                    row[f"effect_{cname}"] = float("nan")
            rows.append(row)
    return pd.DataFrame(rows)


def _provenance(inputs: dq.RunInputs, regime: Regime, keys: GlmKeys, model: StatsModel, tr: float,
                fmriprep_version: str, code_sha: str) -> dict[str, Any]:
    run = inputs.run
    return {
        "schema_version": SCHEMA_VERSION,
        "sub": run.subject, "ses": run.session, "task": run.task, "run": run.run,
        "space": run.space, "variant": run.variant,
        "regime": regime.name, "regime_status": regime.status, "regime_version": regime.version,
        "model": model.name, "conditions": list(model.conditions),
        "localizer": run.task in LOCALIZER_MODELS,
        "repetition_time": tr,
        "fmriprep_version": fmriprep_version,
        "input_bold": str(run.bold),
        "input_bold_sha256": keys.bold_sha256,
        "input_confounds_sha256": inputs.input_confounds_sha256,
        "input_events": str(run.events),
        "input_events_sha256": keys.events_sha256,
        "model_sha256": keys.model_sha256,
        "input_atlases_sha256": keys.atlases_sha256,
        "code_version": code_sha,
        "created": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    }


def write_run_glm(tree_root: Path, inputs: dq.RunInputs, events: pd.DataFrame, model: StatsModel,
                  regime: Regime, tr: float, keys: GlmKeys, *, fmriprep_version: str,
                  code_sha: str) -> dict[str, Any]:
    """Fit one loaded run under one regime and write its GLM cell. Returns the sidecar record."""
    import nibabel as nib

    run = inputs.run
    dm = design_for(run, events, inputs.confounds, tr, model, regime)
    fitres = fit(inputs.data, dm, model.conditions)
    localizer = run.task in LOCALIZER_MODELS
    weights = condition_weights(model, list(dm.columns), model.conditions) if localizer else {}
    effects = {name: contrast(fitres, w)[0] for name, w in weights.items()}

    out = dq.output_dir(tree_root, run)
    out.mkdir(parents=True, exist_ok=True)
    tsv, js = glm_paths(tree_root, run, regime.name)
    parcel_table(inputs, fitres, effects).to_csv(tsv, sep="\t", index=False, float_format="%.6g", na_rep="n/a")

    record = _provenance(inputs, regime, keys, model, tr, fmriprep_version, code_sha)
    nss = non_steady_state_mask(inputs.confounds)
    record.update({
        "design_columns": list(dm.columns),
        "n_vol": len(dm), "n_nss": int(nss.sum()),
        "n_regressors": fitres.n_regressors, "rank": fitres.rank, "dof_resid": fitres.dof,
        "dof_nuisance": fitres.dof_nuisance, "n_task_columns": len(model.conditions),
        "mask_n_voxels": int(inputs.mask_bool.sum()),
        "task_f_df": [fitres.df_task, fitres.dof], "f_alpha": F_ALPHA,
        "task_frac_p001": _frac_sig(fitres.task_p),
        "psc_floor": fitres.psc_floor, "n_voxels_below_psc_floor": int((~fitres.psc_defined).sum()),
        **_summary(fitres.r2_task_adj, "task_r2adj"),
        **_summary(fitres.r2_task, "task_r2"),
    })
    if localizer:
        betas_path, s2_path = beta_paths(tree_root, run, regime.name)
        shape = inputs.mask_bool.shape
        vol = np.zeros(shape + (len(model.conditions),), dtype=np.float32)
        vol[inputs.mask_bool] = fitres.betas.T
        img = nib.Nifti1Image(vol, inputs.affine)
        img.set_data_dtype(np.float32)
        img.to_filename(str(betas_path))
        s2 = np.zeros(shape, dtype=np.float32)
        s2[inputs.mask_bool] = fitres.sigma2.astype(np.float32)
        img = nib.Nifti1Image(s2, inputs.affine)
        img.set_data_dtype(np.float32)
        img.to_filename(str(s2_path))
        record.update({
            "beta_volumes": list(model.conditions),
            "cov_unscaled": fitres.cov_unscaled.tolist(),
            "units": "percent signal change (nilearn signal_scaling=0)",
            "contrasts": {name: w.tolist() for name, w in weights.items()},
            "contrast_effect_median_mask": {name: float(np.nanmedian(e)) for name, e in effects.items()},
        })
    js.write_text(json.dumps(record, indent=2) + "\n")
    for marker in glm_absent_paths(tree_root, run, regime.name):
        marker.unlink(missing_ok=True)
    return record


def write_absent_glm(tree_root: Path, inputs: dq.RunInputs, model: StatsModel, regime: Regime,
                     why: RegimeNotApplicable, tr: float, keys: GlmKeys, *, fmriprep_version: str,
                     code_sha: str) -> dict[str, Any]:
    """Declare that this run cannot carry ``regime`` in the GLM; remove any earlier outputs."""
    run = inputs.run
    dq.output_dir(tree_root, run).mkdir(parents=True, exist_ok=True)
    for stale in _outputs(tree_root, run, regime.name):
        stale.unlink(missing_ok=True)
    record = _provenance(inputs, regime, keys, model, tr, fmriprep_version, code_sha)
    record.update({
        "absent": True, "absent_reason": str(why),
        "acompcor_available": why.n_available, "acompcor_required": why.n_required,
        "n_vol": len(inputs.confounds), "n_nss": int(non_steady_state_mask(inputs.confounds).sum()),
    })
    tsv, js = glm_absent_paths(tree_root, run, regime.name)
    pd.DataFrame([{"regime": regime.name, "reason": str(why), "acompcor_available": why.n_available,
                   "acompcor_required": why.n_required}]).to_csv(tsv, sep="\t", index=False)
    js.write_text(json.dumps(record, indent=2) + "\n")
    return record


# ---------------------------------------------------------------------------
# Collect
# ---------------------------------------------------------------------------

_DROP_FROM_TABLE = ("design_columns", "conditions", "cov_unscaled", "contrasts", "beta_volumes",
                    "contrast_effect_median_mask")


def collect(tree_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Flatten every GLM sidecar into ``(tier1_glm, tier1_glm_parcels)``; absent cells get a row."""
    tree_root = Path(tree_root)
    runs, parcels = [], []
    for js in sorted(tree_root.glob("sub-*/ses-*/func/*_glm.json")):
        rec = json.loads(js.read_text())
        row = {k: v for k, v in rec.items() if k not in _DROP_FROM_TABLE}
        row["n_conditions"] = len(rec.get("conditions", []))
        runs.append({**row, "absent": False})
        tbl = pd.read_csv(js.with_suffix(".tsv"), sep="\t", na_values=["n/a"])
        key = {k: rec.get(k) for k in ("sub", "ses", "task", "run", "regime")}
        parcels.append(tbl.assign(**key))
    for js in sorted(tree_root.glob("sub-*/ses-*/func/*_glmabsent.json")):
        runs.append(json.loads(js.read_text()))
    parcel_df = pd.concat(parcels, ignore_index=True) if parcels else pd.DataFrame()
    if not parcel_df.empty:
        lead = ["sub", "ses", "task", "run", "regime"]
        parcel_df = parcel_df[lead + [c for c in parcel_df.columns if c not in lead]]
    return pd.DataFrame(runs), parcel_df
