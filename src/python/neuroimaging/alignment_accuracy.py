"""Anatomical accuracy of a BOLD->T1w alignment: pial/CSF edge r and ribbon specificity.

Two measures that compare alignment arms on one subject's T1w BOLD grid, where
only the boldref->T1w transform differs between arms: the anatomy stays fixed
and the EPI moves. Design record: mmmdata-agents
``docs/workbench/alignment-strategy/``.

**Anatomy on the BOLD grid.** fMRIPrep's ``desc-ribbon_mask`` (cortical GM
between the FreeSurfer white and pial surfaces), FreeSurfer cerebral WM
(``desc-aseg_dseg`` labels :data:`ASEG_WM`) and outside-brain
(``desc-aseg_dseg`` labels :data:`ASEG_OUTSIDE`: unlabelled, which includes
sulcal CSF, and extracerebral CSF). Each 1 mm mask is put on the BOLD grid as a
**coverage fraction** (:func:`coverage_fraction`), and a voxel belongs to a
compartment when its fraction is at least :data:`FRACTION_IN`. GM wins ties.

* **wm_band** — WM within :data:`WM_BAND_VOX` voxels (6-connected) of GM: the
  juxtacortical WM that misaligned cortical signal falls into.
* **csf_side** — outside-brain within :data:`CSF_SIDE_VOX` voxels of GM.
* **pial_band** — GM voxels 6-adjacent to outside-brain (the ribbon's outer
  boundary), dilated by :data:`PIAL_BAND_VOX`, kept to GM and outside-brain. It
  leaves out the GM/WM boundary that BBR's cost uses, and the ventricles (aseg
  labels them, so they are not outside-brain).

**Pial/CSF edge r** (per run): Pearson r of the Gaussian gradient magnitudes
(:data:`EDGE_SIGMA_MM`) of the run's T1w boldref and fMRIPrep's
``label-CSF_probseg``, inside the pial band. EPI is bright in CSF, so the
GM/CSF interface is a strong EPI edge.

**Ribbon specificity** (per fit): among voxels whose ncsnr beats its shuffle
null, inside the signal parcels, ``n(GM) / (n(GM) + n(wm_band))``. The
compartments do not change between arms, so the share needs no normalising.
Reported beside it, not gating: the rate ratio
``P(exceed | GM) / P(exceed | wm_band)`` and the CSF-side share
``n(csf_side) / (n(GM) + n(csf_side))``. Its spread is a bootstrap over parcels,
paired across arms (:func:`bootstrap_specificity`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from . import data_quality_alignment as dqa

SCHEMA_VERSION = "1.0"
#: Sub-voxel samples per axis when putting a fine mask on the BOLD grid.
SUPERSAMPLE = 4
FRACTION_IN = 0.5
WM_BAND_VOX = 2
CSF_SIDE_VOX = 2
PIAL_BAND_VOX = 1
#: Narrower than data_quality_alignment's 2 mm; why is in the design record (its shift check).
EDGE_SIGMA_MM = 1.0
ASEG_WM = (2, 41)
ASEG_OUTSIDE = (0, 24)
N_BOOT = 2000
RANDOM_SEED = dqa.RANDOM_SEED
COUNT_COLUMNS = ["n_gm", "n_wm_band", "n_csf_side", "x_gm", "x_wm_band", "x_csf_side"]


def coverage_fraction(mask: np.ndarray, affine: np.ndarray, shape: tuple, ref_affine: np.ndarray,
                      k: int = SUPERSAMPLE) -> np.ndarray:
    """Share of each voxel of the ``(shape, ref_affine)`` grid covered by a fine binary mask.

    Each coarse voxel is sampled at ``k**3`` evenly spaced sub-voxel points;
    each point takes the nearest fine voxel. Points outside the fine grid count
    as uncovered.
    """
    mask = np.asarray(mask, bool)
    offsets = (np.arange(k) + 0.5) / k - 0.5
    coarse = np.indices(shape).reshape(3, -1).astype(np.float64)
    to_fine = np.linalg.inv(affine) @ ref_affine
    hits = np.zeros(coarse.shape[1], np.float64)
    for dx in offsets:
        for dy in offsets:
            for dz in offsets:
                p = coarse + np.array([[dx], [dy], [dz]])
                ijk = np.rint(to_fine[:3, :3] @ p + to_fine[:3, 3:]).astype(np.int64)
                ok = np.all((ijk >= 0) & (ijk < np.array(mask.shape)[:, None]), axis=0)
                hits[ok] += mask[ijk[0, ok], ijk[1, ok], ijk[2, ok]]
    return (hits / k ** 3).reshape(shape)


@dataclass
class Compartments:
    gm: np.ndarray
    wm: np.ndarray
    outside: np.ndarray
    wm_band: np.ndarray
    csf_side: np.ndarray
    pial_band: np.ndarray

    def sizes(self) -> dict[str, int]:
        return {k: int(getattr(self, k).sum()) for k in ("gm", "wm", "outside", "wm_band", "csf_side", "pial_band")}


def compartments(ribbon_frac: np.ndarray, wm_frac: np.ndarray, outside_frac: np.ndarray,
                 wm_band_vox: int = WM_BAND_VOX, csf_side_vox: int = CSF_SIDE_VOX,
                 pial_band_vox: int = PIAL_BAND_VOX) -> Compartments:
    """Compartment masks from coverage fractions (see the module docstring)."""
    from scipy.ndimage import binary_dilation

    gm = ribbon_frac >= FRACTION_IN
    wm = (wm_frac >= FRACTION_IN) & ~gm
    outside = (outside_frac >= FRACTION_IN) & ~gm & ~wm
    wm_band = wm & binary_dilation(gm, iterations=wm_band_vox)
    csf_side = outside & binary_dilation(gm, iterations=csf_side_vox)
    outer = gm & binary_dilation(outside)
    pial_band = binary_dilation(outer, iterations=pial_band_vox) & (gm | outside)
    return Compartments(gm, wm, outside, wm_band, csf_side, pial_band)


def anatomy_compartments(ribbon: np.ndarray, ribbon_affine: np.ndarray, aseg: np.ndarray,
                         aseg_affine: np.ndarray, shape: tuple, ref_affine: np.ndarray,
                         shift: Optional[tuple[int, float]] = None) -> Compartments:
    """Compartments on the ``(shape, ref_affine)`` grid from the fine ribbon mask and aseg labels.

    ``shift`` is ``(world axis, mm)``: the anatomy is translated by that much
    before it is put on the grid. Because the fine masks go through
    :func:`coverage_fraction`, a sub-voxel shift changes the fractions and so
    which boundary voxels pass :data:`FRACTION_IN`.
    """
    rib_aff, aseg_aff = ribbon_affine.copy(), aseg_affine.copy()
    if shift is not None:
        axis, d = shift
        rib_aff[axis, 3] += d
        aseg_aff[axis, 3] += d
    return compartments(coverage_fraction(np.asarray(ribbon) > 0, rib_aff, shape, ref_affine),
                        coverage_fraction(np.isin(aseg, ASEG_WM), aseg_aff, shape, ref_affine),
                        coverage_fraction(np.isin(aseg, ASEG_OUTSIDE), aseg_aff, shape, ref_affine))


def nearest_labels(labels: np.ndarray, where: np.ndarray) -> np.ndarray:
    """``labels`` with every ``where`` voxel given its nearest nonzero label (others 0)."""
    from scipy.ndimage import distance_transform_edt

    labels = np.asarray(labels)
    if not (labels > 0).any():
        return np.zeros_like(labels)
    idx = distance_transform_edt(labels == 0, return_distances=False, return_indices=True)
    out = labels[tuple(idx)]
    return np.where(where, out, 0)


def pial_edge_r(run: np.ndarray, csf_prob: np.ndarray, band: np.ndarray, zooms: np.ndarray) -> float:
    """Edge correlation of a T1w-space boldref with the CSF probability map, inside the pial band."""
    return dqa.edge_correlation(run, csf_prob, band, EDGE_SIGMA_MM / np.asarray(zooms, float))


def region_edge_r(run: np.ndarray, csf_prob: np.ndarray, band: np.ndarray, zooms: np.ndarray,
                  band_labels: np.ndarray, groups: dict[str, list[int]]) -> dict[str, float]:
    """:func:`pial_edge_r` over the band voxels of each group of parcel labels.

    The gradients are taken over the whole image once; only the voxel set
    changes by group.
    """
    from scipy.ndimage import gaussian_gradient_magnitude

    sigma = EDGE_SIGMA_MM / np.asarray(zooms, float)
    ga = gaussian_gradient_magnitude(np.nan_to_num(np.asarray(run, np.float64)), sigma=sigma)
    gb = gaussian_gradient_magnitude(np.nan_to_num(np.asarray(csf_prob, np.float64)), sigma=sigma)
    out = {}
    for name, labs in groups.items():
        sel = band & np.isin(band_labels, labs)
        x, y = ga[sel], gb[sel]
        out[name] = (float(np.corrcoef(x, y)[0, 1]) if x.size >= 3 and x.std() > 0 and y.std() > 0
                     else np.nan)
    return out


def network_groups(names: pd.Series) -> dict[str, list[int]]:
    """Schaefer-17 parcel indices by network (``17Networks_LH_SomMotA_1`` -> ``SomMotA``).

    ``names`` is indexed by parcel label. Pass the current CBIG names: the
    TemplateFlow MNI table's component names predate CBIG's 2019 fix.
    """
    groups: dict[str, list[int]] = {}
    for idx, name in names.items():
        parts = str(name).split("_")
        if len(parts) < 3:
            raise ValueError(f"not a Schaefer 17-network name: {name!r}")
        groups.setdefault(parts[2], []).append(int(idx))
    return groups


def component_group(names: pd.Series, component: str) -> list[int]:
    """Parcel indices whose name has ``component`` as its 4th field (e.g. ``PFCd``)."""
    return [int(i) for i, n in names.items() if len(str(n).split("_")) > 3 and str(n).split("_")[3] == component]


def parcel_counts(exceed: np.ndarray, keep: np.ndarray, comp: Compartments, labels: np.ndarray,
                  parcels: Optional[list[int]] = None) -> pd.DataFrame:
    """Per parcel: voxels (``n_*``) and null-beating voxels (``x_*``) in GM, wm_band and csf_side.

    ``keep`` is the voxel set the null was computed over (brain mask and
    meanvol floor); voxels outside it count nowhere. WM-band and CSF-side
    voxels lie outside the parcels' GM, so each takes its nearest parcel.
    """
    exceed = np.asarray(exceed, bool) & keep
    lab = nearest_labels(labels, comp.gm | comp.wm_band | comp.csf_side)
    if parcels is None:
        parcels = sorted(int(v) for v in np.unique(labels) if v > 0)
    rows = []
    for p in parcels:
        in_p = (lab == p) & keep
        row = {"parcel": p}
        for name in ("gm", "wm_band", "csf_side"):
            sel = in_p & getattr(comp, name)
            row[f"n_{name}"] = int(sel.sum())
            row[f"x_{name}"] = int((exceed & sel).sum())
        rows.append(row)
    return pd.DataFrame(rows, columns=["parcel", *COUNT_COLUMNS])


def specificity(counts: pd.DataFrame) -> dict[str, float]:
    """Ribbon specificity, rate ratio and CSF-side share from summed per-parcel counts."""
    s = counts[COUNT_COLUMNS].sum()
    share = s.x_gm / (s.x_gm + s.x_wm_band) if s.x_gm + s.x_wm_band else np.nan
    rate_gm = s.x_gm / s.n_gm if s.n_gm else np.nan
    rate_band = s.x_wm_band / s.n_wm_band if s.n_wm_band else np.nan
    return {"specificity": float(share),
            "rate_ratio": float(rate_gm / rate_band) if rate_band else np.nan,
            "csf_share": float(s.x_csf_side / (s.x_gm + s.x_csf_side)) if s.x_gm + s.x_csf_side else np.nan,
            **{c: int(s[c]) for c in COUNT_COLUMNS}}


def bootstrap_specificity(counts_by_arm: dict[str, pd.DataFrame], n_boot: int = N_BOOT,
                          seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Specificity per arm over parcel resamples, paired: every arm sees the same draw.

    Every arm's table must list the same parcels in the same order. Returns one
    row per draw, one column per arm.
    """
    arms = list(counts_by_arm)
    ref = counts_by_arm[arms[0]]["parcel"].to_numpy()
    for a in arms[1:]:
        if not np.array_equal(counts_by_arm[a]["parcel"].to_numpy(), ref):
            raise ValueError(f"arm {a} lists different parcels from arm {arms[0]}; the bootstrap must be paired")
    x = {a: counts_by_arm[a][["x_gm", "x_wm_band"]].to_numpy(np.float64) for a in arms}
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(ref), size=(n_boot, len(ref)))
    out = {}
    for a in arms:
        s = x[a][draws].sum(axis=1)  # (n_boot, 2)
        tot = s.sum(axis=1)
        out[a] = np.where(tot > 0, s[:, 0] / np.where(tot > 0, tot, 1), np.nan)
    return pd.DataFrame(out)


#: An arm beats A on ribbon specificity by at least this much (absolute share), with its paired CI above 0.
MIN_SPECIFICITY_GAIN = 0.010
#: ncsnr no-worse margin, as a share of A's median ncsnr over the signal parcels.
NCSNR_MARGIN_FRAC = 0.10


def compare_arms(counts_a: pd.DataFrame, counts_b: pd.DataFrame, ncsnr_a: pd.Series, ncsnr_b: pd.Series,
                 n_boot: int = N_BOOT, seed: int = RANDOM_SEED) -> dict:
    """Arm B against A on one subject: ribbon specificity gain and the ncsnr no-worse check, both paired by parcel.

    ``counts_*`` are :func:`parcel_counts` tables over the same signal parcels; ``ncsnr_*`` are per-parcel median
    ncsnr indexed by parcel. Specificity: B's minus A's pooled share, with a paired parcel bootstrap CI. ncsnr:
    the median over parcels of B - A, with a bootstrap CI over the same parcel draws; the margin is
    :data:`NCSNR_MARGIN_FRAC` of A's median. ``wins`` needs both, before visual review.
    """
    boot = bootstrap_specificity({"A": counts_a, "B": counts_b}, n_boot=n_boot, seed=seed)
    gain = specificity(counts_b)["specificity"] - specificity(counts_a)["specificity"]
    g_ci = np.nanpercentile(boot["B"] - boot["A"], [2.5, 97.5])
    parcels = counts_a["parcel"].to_numpy()
    missing = set(parcels) - set(ncsnr_a.dropna().index) | set(parcels) - set(ncsnr_b.dropna().index)
    if missing:
        raise ValueError(f"no ncsnr for signal parcels {sorted(missing)} in one of the arms")
    d = (ncsnr_b.reindex(parcels) - ncsnr_a.reindex(parcels)).to_numpy(np.float64)
    draws = np.random.default_rng(seed).integers(0, len(parcels), size=(n_boot, len(parcels)))
    n_ci = np.percentile(np.median(d[draws], axis=1), [2.5, 97.5])
    margin = NCSNR_MARGIN_FRAC * float(np.median(ncsnr_a.reindex(parcels)))
    spec_ok = bool(gain >= MIN_SPECIFICITY_GAIN and g_ci[0] > 0)
    ncsnr_ok = bool(n_ci[0] >= -margin)
    return {"n_parcels": int(len(parcels)),
            "specificity_A": specificity(counts_a)["specificity"], "specificity_B": specificity(counts_b)["specificity"],
            "specificity_gain": float(gain), "specificity_gain_ci95": [float(g_ci[0]), float(g_ci[1])],
            "min_gain": MIN_SPECIFICITY_GAIN, "specificity_passes": spec_ok,
            "ncsnr_median_A": float(np.median(ncsnr_a.reindex(parcels))),
            "ncsnr_median_diff": float(np.median(d)), "ncsnr_median_diff_ci95": [float(n_ci[0]), float(n_ci[1])],
            "ncsnr_margin": margin, "ncsnr_no_worse": ncsnr_ok, "wins": spec_ok and ncsnr_ok}
