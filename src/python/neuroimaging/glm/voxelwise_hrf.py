"""Per-voxel HRF fits: one design per GLMsingle library kernel, stitched by HRFindex.

The bake-off's third HRF level (glm-strategy log, DECIDED 2026-09-08). The
subject's encoding GLMsingle fit chose one of 20 library kernels per voxel
(``HRFindex``); here every voxel is fitted with the design convolved with
*its* kernel. Twenty disjoint voxel groups sum to one brain, so the cost is
about one full fit plus twenty rounds of per-group overhead.

Engine-neutral: any :class:`~.estimators.Estimator` fits each group through
``fit_run`` with a group mask, and the maps are stitched here. Smoothing is
applied once to the whole run before the groups are cut, which is the same
operation the engines apply inside a single-group fit (nilearn smooths the
unmasked image before masking; the remlfit wrapper does the same).
"""

from __future__ import annotations

import dataclasses
from typing import Any, Optional

import numpy as np
import pandas as pd

from .config import GlmConfig
from .design import available_contrast_vectors, build_design_matrix, strict_for
from .estimators import ContrastEstimate, Estimator
from .hrf import GLMSINGLE_PREFIX, LIBRARY_SIZE
from .models import StatsModel

VOXELWISE = "voxelwise"  # the cfg.hrf_model value that selects this path


def fit_run_voxelwise(
    estimator: Estimator,
    bold: Any,
    events: pd.DataFrame,
    confounds: Optional[pd.DataFrame],
    t_r: float,
    model: StatsModel,
    cfg: GlmConfig,
    hrfindex: Any,
    mask: Any,
    *,
    min_group_voxels: int = 1,
) -> dict[str, ContrastEstimate]:
    """Fit one run with each voxel's own library kernel; return stitched maps.

    ``hrfindex`` and ``mask`` are images on the BOLD's grid (checked). Groups
    with fewer than ``min_group_voxels`` inside the mask are skipped and left
    zero (variance NaN) — a handful of stray voxels is not worth a fit, and
    NaN variance drops them from fixed effects rather than counting them.
    """
    import nibabel as nib
    from nilearn.image import smooth_img

    img = bold if isinstance(bold, nib.Nifti1Image) else nib.load(str(bold))
    mask_arr = np.asarray(mask.dataobj).astype(bool)
    idx = np.asarray(hrfindex.dataobj).astype(int)
    if idx.shape != img.shape[:3] or mask_arr.shape != img.shape[:3]:
        raise ValueError(
            f"grid mismatch: BOLD {img.shape[:3]}, HRFindex {idx.shape}, mask {mask_arr.shape}"
        )
    if cfg.smoothing_fwhm:
        img = smooth_img(img, cfg.smoothing_fwhm)
    group_cfg = dataclasses.replace(cfg, smoothing_fwhm=None)
    n_scans = img.shape[-1]

    stitched: dict[str, dict[str, np.ndarray]] = {}
    dof: Optional[float] = None
    fitted_groups = 0
    for k in range(LIBRARY_SIZE):
        group = mask_arr & (idx == k)
        if group.sum() < min_group_voxels:
            continue
        kcfg = dataclasses.replace(group_cfg, hrf_model=f"{GLMSINGLE_PREFIX}{k}")
        dm = build_design_matrix(events, confounds, t_r, n_scans, model, kcfg, strict=strict_for(model))
        vectors, _skipped = available_contrast_vectors(model, list(dm.columns))
        if not vectors:
            return {}  # no contrast is estimable from this run; the caller skips it
        gmask = nib.Nifti1Image(group.astype(np.uint8), mask.affine)
        est = estimator.fit_run(img, dm, vectors, t_r=t_r, mask=gmask, cfg=kcfg)
        fitted_groups += 1
        for name, ce in est.items():
            slot = stitched.setdefault(name, {})
            for stat, image in (("effect", ce.effect), ("variance", ce.variance), ("stat", ce.stat), ("z", ce.z)):
                if image is None:
                    continue
                arr = np.asarray(image.dataobj, dtype=float)
                if stat not in slot:
                    slot[stat] = np.full(arr.shape, np.nan if stat == "variance" else 0.0)
                slot[stat][group] = arr[group]
            if dof is None:
                dof = ce.dof
    if not fitted_groups:
        raise ValueError("no HRFindex group had voxels inside the mask; nothing was fitted")

    out: dict[str, ContrastEstimate] = {}
    for name, maps in stitched.items():
        mk = lambda a: nib.Nifti1Image(a.astype(np.float32), img.affine)  # noqa: E731
        out[name] = ContrastEstimate(
            effect=mk(maps["effect"]),
            variance=mk(maps["variance"]),
            dof=dof,
            stat=mk(maps["stat"]) if "stat" in maps else None,
            z=mk(maps["z"]) if "z" in maps else None,
        )
    return out
