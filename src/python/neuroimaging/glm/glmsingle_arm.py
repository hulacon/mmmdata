"""The bake-off's standalone GLMsingle arms, emitted in the statmap schema.

Two arms that do not factor with the HRF/confound/engine design
(glm-strategy log, DECIDED 2026-09-08, item 5):

* **GLMsingle on fLoc.** Each half's three runs go through GLMsingle
  (TYPED: library HRF, GLMdenoise, ridge) as a block design; every block is
  a "trial" with a beta. A contrast is a weighted difference of per-condition
  block means, its variance the weighted sum of per-condition block
  variances of the mean — the estimator's own error model is not used
  because GLMsingle does not expose one.
* **Two-sample test on the existing TBencoding per-trial betas.** The
  encoding fits already hold one beta per trial (``betasmd``, columns =
  ``trial_info.csv`` rows — the TRAP from ../retrieval-modeling/: match on
  (session, run, onset), never ``col_index``). Per half, first vs later
  presentations (adapter labels) give a Welch t map.

Both produce the same per-half maps the factorial cells do, so the harness
scores them identically.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from .estimators import ContrastEstimate, t_to_z
from .models import StatsModel


def block_design(events: pd.DataFrame, conditions: Sequence[str], t_r: float, n_scans: int) -> np.ndarray:
    """GLMsingle's (time x conditions) onset matrix: 1 at each block's onset TR.

    Onsets round to the nearest TR; two events of one condition landing on
    the same TR is a modelling error and is refused.
    """
    design = np.zeros((n_scans, len(conditions)), dtype=np.float32)
    for _, row in events.iterrows():
        cond = str(row["trial_type"])
        if cond not in conditions:
            continue
        tr_index = int(round(float(row["onset"]) / t_r))
        if tr_index >= n_scans:
            raise ValueError(f"onset {row['onset']} s is beyond the run's {n_scans} volumes")
        j = conditions.index(cond)
        if design[tr_index, j]:
            raise ValueError(f"two {cond} events round to TR {tr_index}")
        design[tr_index, j] = 1.0
    return design


def trial_conditions(designs: Sequence[np.ndarray]) -> np.ndarray:
    """Condition index of every trial in GLMsingle's beta order (time within run, runs concatenated)."""
    conds = []
    for d in designs:
        rows, cols = np.nonzero(d)
        order = np.argsort(rows, kind="stable")
        conds.extend(cols[order].tolist())
    return np.asarray(conds, dtype=int)


def contrast_from_trial_betas(
    betas: np.ndarray,
    trial_condition: np.ndarray,
    weights: dict[int, float],
    affine: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> ContrastEstimate:
    """Weighted difference of per-condition trial means, variance from trial spread.

    ``betas`` is (x, y, z, trials); ``weights`` maps condition index to
    contrast weight (zero-weight conditions are ignored). dof is the sum of
    (n_c - 1) over weighted conditions — the pooled-variance count, used
    only to map t to z.
    """
    import nibabel as nib

    effect = np.zeros(betas.shape[:3], dtype=float)
    var = np.zeros(betas.shape[:3], dtype=float)
    dof = 0
    for cond, w in weights.items():
        if w == 0:
            continue
        sel = trial_condition == cond
        n = int(sel.sum())
        if n < 2:
            raise ValueError(f"condition {cond} has {n} trial(s); a variance needs at least 2")
        x = betas[..., sel].astype(float)
        effect += w * x.mean(axis=-1)
        var += (w ** 2) * x.var(axis=-1, ddof=1) / n
        dof += n - 1
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(var > 0, effect / np.sqrt(var), 0.0)
    z = t_to_z(t, dof)
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        for a in (effect, t, z):
            a[~m] = 0.0
        var[~m] = np.nan
    mk = lambda a: nib.Nifti1Image(a.astype(np.float32), affine)  # noqa: E731
    return ContrastEstimate(effect=mk(effect), variance=mk(var), dof=float(dof), stat=mk(t), z=mk(z))


def welch_contrast(
    betas_a: np.ndarray, betas_b: np.ndarray, affine: np.ndarray, mask: Optional[np.ndarray] = None
) -> ContrastEstimate:
    """A − B two-sample Welch t over the last axis of two beta stacks."""
    import nibabel as nib

    na, nb = betas_a.shape[-1], betas_b.shape[-1]
    if na < 2 or nb < 2:
        raise ValueError(f"both groups need at least 2 trials; got {na} and {nb}")
    ma, mb = betas_a.mean(-1), betas_b.mean(-1)
    va, vb = betas_a.var(-1, ddof=1) / na, betas_b.var(-1, ddof=1) / nb
    var = va + vb
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(var > 0, (ma - mb) / np.sqrt(var), 0.0)
        dof_map = np.where(var > 0, var ** 2 / (va ** 2 / (na - 1) + vb ** 2 / (nb - 1)), na + nb - 2)
    dof = float(np.nanmedian(dof_map[np.asarray(mask, dtype=bool)] if mask is not None else dof_map))
    z = t_to_z(t, dof)
    effect = ma - mb
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        for a in (effect, t, z):
            a[~m] = 0.0
        var[~m] = np.nan
    mk = lambda a: nib.Nifti1Image(a.astype(np.float32), affine)  # noqa: E731
    return ContrastEstimate(effect=mk(effect), variance=mk(var), dof=dof, stat=mk(t), z=mk(z))


def fit_glmsingle_half(
    bolds: Sequence[Any],
    designs: Sequence[np.ndarray],
    stimdur: float,
    t_r: float,
    model: StatsModel,
    *,
    mask: Optional[np.ndarray] = None,
    smoothing_fwhm: Optional[float] = None,
    workdir: Optional[str] = None,
) -> dict[str, ContrastEstimate]:
    """GLMsingle TYPED over one half's runs, then every model contrast from the trial betas.

    Runs are pre-smoothed with the harness FWHM so the comparison isolates
    the estimator rather than the smoothing; GLMsingle itself is run as the
    encoding fits were (library HRF, GLMdenoise, fracridge), file outputs off.
    """
    import nibabel as nib
    from glmsingle.glmsingle import GLM_single
    from nilearn.image import smooth_img

    data = []
    affine = None
    for b in bolds:
        img = b if isinstance(b, nib.Nifti1Image) else nib.load(str(b))
        if smoothing_fwhm:
            img = smooth_img(img, smoothing_fwhm)
        affine = img.affine if affine is None else affine
        data.append(np.asarray(img.dataobj, dtype=np.float32))
    params = {
        "wantlibrary": 1, "wantglmdenoise": 1, "wantfracridge": 1,
        "wantfileoutputs": [0, 0, 0, 0], "wantmemoryoutputs": [0, 0, 0, 1],
    }
    # No brainexclude: the Python port tests it with `if not params[...]`,
    # which raises on an array. GLMsingle picks its own noise pool; the
    # harness mask is applied to the contrast maps below.
    results = GLM_single(params).fit(design=list(designs), data=data, stimdur=stimdur, tr=t_r, outputdir=workdir)
    betas = np.asarray(results["typed"]["betasmd"])
    cond_index = trial_conditions(designs)
    out = {}
    for c in model.contrasts:
        weights = {model.conditions.index(k): w for k, w in c.weights.items()}
        out[c.name] = contrast_from_trial_betas(betas, cond_index, weights, affine, mask)
    return out


def tb_trial_labels(trial_info: pd.DataFrame, adapted_events: Sequence[pd.DataFrame]) -> pd.Series:
    """Adapter labels aligned to ``trial_info`` rows by (session, run, onset).

    Refuses any trial the adapter did not label: an unmatched row means the
    beta order and the events disagree, which is the trap this guards.
    """
    key_cols = ["session", "run", "onset"]
    lab = {}
    for ev in adapted_events:
        for r in ev.itertuples(index=False):
            lab[(f"ses-{int(r.ses_num):02d}", f"run-{int(r.run_idx):02d}", round(float(r.onset), 3))] = r.trial_type
    labels = []
    for r in trial_info.itertuples(index=False):
        key = (r.session, r.run, round(float(r.onset), 3))
        if key not in lab:
            raise KeyError(f"trial_info row {key} has no adapted event; beta columns and events disagree")
        labels.append(lab[key])
    return pd.Series(labels, index=trial_info.index, name="trial_type")
