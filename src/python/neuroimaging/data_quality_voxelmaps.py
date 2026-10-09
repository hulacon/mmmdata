"""Voxel and vertex quality maps of the data-quality collection (T1w + fsnative).

The tier-1 cleaning grid (:mod:`neuroimaging.data_quality`) holds run-level tSNR
in MNI only. This module adds the subject's own spaces, for two questions: where
does each session lose signal (dropout), and do the fsnative surfaces sample the
volume signal adequately. Design record: mmmdata-agents
``docs/workbench/voxel-quality/``.

**Run cell** (``tier1.py voxelmaps``, one per run, no atlas, no resampling).
The run's T1w BOLD and its two fsnative ``func.gii`` series are cleaned with
:func:`data_quality.clean` under :data:`REGIMES`. Written beside the MNI cells in
``sub-##/ses-##/func/``::

    <prefix>_space-T1w_mean.nii.gz                     temporal mean (fitted volumes)
    <prefix>_space-T1w_desc-<regime>_tsnr.nii.gz       tSNR, one per regime
    <prefix>_hemi-<H>_space-fsnative_mean.func.gii
    <prefix>_hemi-<H>_space-fsnative_desc-<regime>_tsnr.func.gii

each with a ``.json`` sidecar carrying ``"cell": "voxelmaps"``. The T1w mean
sidecar is the cell's record: input hashes, regime versions, summaries. Outside
the brain mask a volume is NaN, never 0. The temporal mean does not depend on the
regime (the same volumes are fitted under every one), so there is one.

**Subject cell** (``tier1.py voxelmaps-subject``, one per subject). Reads only run
cells and fMRIPrep anatomy, and writes:

* Session maps (median over the session's runs) and subject maps (median over
  sessions), for the normalised mean and each regime's tSNR. A run's mean is
  divided by its own median over the subject's consensus mask (cortex vertices on
  the surface) first, because receive gain varies run to run. A voxel outside a
  run's mask is NaN in that run, so a session median uses the runs that cover it.
* Relative maps, session ÷ subject, and the dropped mask: relative mean signal
  below :data:`DROP_FLOOR` (RATIFIED 2026-10-09) inside the consensus mask.
* Surface vs volume, per run and then median over runs: each run's T1w mean and
  ``drift`` tSNR sampled onto fsnative along the white→pial ribbon
  (:data:`DEPTHS`), the ratio surface ÷ sampled per vertex, and the fraction of
  the ribbon samples inside the run's brain mask.
* Parcel rows: Schaefer-400 17n and HOSPA warped MNI→T1w by nearest label (as the
  pRF cell does); fsnative vertices take the Schaefer label at their midthickness
  position.
* The session and subject maps warped T1w→MNI152NLin2009cAsym res-2 with
  fMRIPrep's transform, for cross-subject views. The tier-1 MNI cells hold tSNR
  only, so MNI dropout is read from these.

Runs on a grid other than the subject's majority T1w grid (sub-03's 1.7 mm
sessions) are resampled onto it at aggregation only (trilinear; the run mask
trilinear and kept above 0.5), and the subject sidecar names them.
"""

from __future__ import annotations

import collections
import datetime as _dt
import json
import tempfile
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from . import data_quality as dq
from .confounds import Regime, regime_design
from .io import FmriprepRun, load_confounds

SCHEMA_VERSION = "1.0"
CELL = "voxelmaps"
REGIMES: tuple[str, ...] = ("none", "drift", "reference")
#: The regime whose tSNR is sampled onto the surface for the surface-vs-volume ratio.
SAMPLED_REGIME = "drift"
VOLUME_SPACE = "T1w"
SURFACE_SPACE = "fsnative"
MNI_SPACE = "MNI152NLin2009cAsym"
MNI_RES = "2"
HEMIS: tuple[str, ...] = ("L", "R")
#: Fractions of the way from white to pial at which the volume is sampled
#: (FreeSurfer ``--projfrac-avg 0 1 0.2``, fMRIPrep's fsnative sampling).
DEPTHS: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
#: A voxel or vertex is dropped in a session below this relative mean (RATIFIED 2026-10-09).
DROP_FLOOR = 0.8
#: The subject's consensus mask: voxels inside this fraction of its runs' brain masks.
CONSENSUS = 0.9
#: A session is examined when its dropped fraction exceeds this multiple of the subject median.
EXAMINE_MULTIPLE = 2.0
#: Ribbon in-mask fraction below which a vertex is mapped as under-covered (RATIFIED 2026-10-09).
RIBBON_FLOOR = 0.8

RUNS_TABLE = "tier1_voxelmaps"
SESSIONS_TABLE = "tier1_voxelmaps_sessions"
PARCELS_TABLE = "tier1_voxelmaps_parcels"
SURFVOL_TABLE = "tier1_voxelmaps_surfvol"

MEASURES: tuple[str, ...] = ("mean",) + tuple(f"tsnr_{r}" for r in REGIMES)


# ---------------------------------------------------------------------------
# Pure helpers (tested on synthetic data)
# ---------------------------------------------------------------------------

def run_scale(mean: np.ndarray, domain: np.ndarray) -> float:
    """The run's gain: median of its temporal mean over ``domain`` (finite values only)."""
    v = np.asarray(mean, dtype=np.float64)[domain]
    v = v[np.isfinite(v)]
    if v.size == 0:
        raise ValueError("no finite value in the normalisation domain")
    return float(np.median(v))


def nanmedian_stack(maps: list[np.ndarray]) -> np.ndarray:
    """Element-wise median over maps, ignoring NaN; NaN where every map is NaN."""
    stack = np.stack([np.asarray(m, dtype=np.float32) for m in maps])
    out = np.full(stack.shape[1:], np.nan, dtype=np.float32)
    any_finite = np.isfinite(stack).any(axis=0)
    if any_finite.any():
        out[any_finite] = np.nanmedian(stack[:, any_finite], axis=0)
    return out


def relative(session: np.ndarray, subject: np.ndarray) -> np.ndarray:
    """session ÷ subject; NaN where either is NaN or the subject value is not positive."""
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(subject > 0, session / subject, np.nan)
    out = np.asarray(out, dtype=np.float32)
    out[~np.isfinite(out)] = np.nan
    return out


def dropout_summary(rel_mean: np.ndarray, domain: np.ndarray, floor: float = DROP_FLOOR) -> dict[str, Any]:
    """Within ``domain``: fraction dropped (finite and below floor), uncovered (NaN), and their union."""
    r = np.asarray(rel_mean, dtype=np.float64)[domain]
    n = int(r.size)
    fin = np.isfinite(r)
    dropped = fin & (r < floor)
    if n == 0:
        return {"n": 0, "frac_dropped": np.nan, "frac_uncovered": np.nan, "frac_lost": np.nan,
                "rel_mean_median": np.nan}
    return {
        "n": n,
        "frac_dropped": float(dropped.sum() / n),
        "frac_uncovered": float((~fin).sum() / n),
        "frac_lost": float((dropped | ~fin).sum() / n),
        "rel_mean_median": float(np.median(r[fin])) if fin.any() else np.nan,
    }


def examine_flags(frac: pd.Series, multiple: float = EXAMINE_MULTIPLE) -> pd.Series:
    """True where a session's fraction exceeds ``multiple`` × the subject median (median 0: any > 0)."""
    med = float(np.nanmedian(frac)) if len(frac) else np.nan
    if not np.isfinite(med):
        return pd.Series(False, index=frac.index)
    return frac > (multiple * med if med > 0 else 0.0)


def ribbon_points(white: np.ndarray, pial: np.ndarray, depths: tuple[float, ...] = DEPTHS) -> np.ndarray:
    """(n_depth, n_vertex, 3) world coordinates from white (0) to pial (1)."""
    white = np.asarray(white, dtype=np.float64)
    pial = np.asarray(pial, dtype=np.float64)
    if white.shape != pial.shape:
        raise ValueError(f"white {white.shape} and pial {pial.shape} surfaces differ in vertex count")
    return np.stack([white + d * (pial - white) for d in depths])


def sample_volume(values: np.ndarray, mask: np.ndarray, affine: np.ndarray, points: np.ndarray
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Trilinear samples of a masked volume at world ``points`` (..., 3).

    Returns ``(value, inmask)``: ``inmask`` is the trilinear weight of the mask at
    the point, and ``value`` the mask-weighted mean of the in-mask neighbours (NaN
    where no neighbour is in the mask), so out-of-mask voxels never enter a sample.
    """
    from scipy.ndimage import map_coordinates

    pts = np.asarray(points, dtype=np.float64)
    flat = pts.reshape(-1, 3)
    ijk = (np.linalg.inv(affine) @ np.c_[flat, np.ones(len(flat))].T)[:3]
    m = np.asarray(mask, dtype=np.float64)
    v = np.where(mask & np.isfinite(values), values, 0.0).astype(np.float64)
    w = map_coordinates(m, ijk, order=1, mode="constant", cval=0.0)
    s = map_coordinates(v, ijk, order=1, mode="constant", cval=0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        val = np.where(w > 1e-6, s / w, np.nan)
    return val.reshape(pts.shape[:-1]), w.reshape(pts.shape[:-1])


def ribbon_sample(values: np.ndarray, mask: np.ndarray, affine: np.ndarray, points: np.ndarray
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Depth-averaged ribbon sample per vertex and the ribbon's in-mask fraction.

    ``points`` is :func:`ribbon_points`' (n_depth, n_vertex, 3). The value averages
    the depths that have an in-mask neighbour; the fraction averages the mask weight
    over all depths.
    """
    val, w = sample_volume(values, mask, affine, points)
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(np.where(np.isfinite(val), val, np.nan), axis=0)
    return mean.astype(np.float32), w.mean(axis=0).astype(np.float32)


def spearman(a: np.ndarray, b: np.ndarray) -> tuple[float, int]:
    """Spearman ρ over pairs finite in both, and the pair count."""
    from scipy.stats import spearmanr

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return np.nan, int(ok.sum())
    return float(spearmanr(a[ok], b[ok]).statistic), int(ok.sum())


def map_summary(values: np.ndarray) -> dict[str, Any]:
    v = np.asarray(values, dtype=np.float64).ravel()
    fin = v[np.isfinite(v)]
    return {"n": int(v.size), "n_nan": int(v.size - fin.size),
            "median": float(np.median(fin)) if fin.size else np.nan,
            "p05": float(np.percentile(fin, 5)) if fin.size else np.nan,
            "p95": float(np.percentile(fin, 95)) if fin.size else np.nan}


# ---------------------------------------------------------------------------
# File layout
# ---------------------------------------------------------------------------

def surface_bold(run: FmriprepRun, hemi: str) -> Path:
    """The run's fsnative ``func.gii`` (``FmriprepRun.surface_*`` point at fsaverage6)."""
    return run.bold.parent / f"{run.entity_prefix}_hemi-{hemi}_space-{SURFACE_SPACE}_bold.func.gii"


def run_files(run: FmriprepRun) -> dict[str, Path]:
    """Every fMRIPrep file the run cell reads. Missing = loud."""
    if run.space != VOLUME_SPACE:
        raise ValueError(f"{run.entity_prefix}: the voxelmaps cell reads space-{VOLUME_SPACE} runs, got {run.space}")
    files = {"bold": run.bold, "mask": run.mask, "confounds": run.confounds}
    for h in HEMIS:
        files[f"surf_{h}"] = surface_bold(run, h)
    missing = [f"{k}: {p}" for k, p in files.items() if p is None or not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"{run.entity_prefix}: voxelmaps inputs missing: {missing}")
    return {k: Path(p) for k, p in files.items()}


def _run_dir(tree_root: Path, run: FmriprepRun) -> Path:
    return Path(tree_root) / f"sub-{run.subject}" / f"ses-{run.session}" / "func"


def run_map_paths(tree_root: Path, run: FmriprepRun) -> dict[str, tuple[Path, Path]]:
    """``key -> (map, sidecar)`` for every output of a run cell.

    Keys: ``T1w_<measure>`` and ``<hemi>_<measure>``, measure in :data:`MEASURES`.
    """
    d = _run_dir(tree_root, run)
    out: dict[str, tuple[Path, Path]] = {}
    spaces = [("T1w", f"_space-{VOLUME_SPACE}", ".nii.gz")] + [
        (h, f"_hemi-{h}_space-{SURFACE_SPACE}", ".func.gii") for h in HEMIS]
    for key, frag, ext in spaces:
        for measure in MEASURES:
            name = f"{run.entity_prefix}{frag}_mean" if measure == "mean" else \
                f"{run.entity_prefix}{frag}_desc-{measure.removeprefix('tsnr_')}_tsnr"
            out[f"{key}_{measure}"] = (d / f"{name}{ext}", d / f"{name}.json")
    return out


def record_path(tree_root: Path, run: FmriprepRun) -> Path:
    """The cell's record: the T1w mean map's sidecar."""
    return run_map_paths(tree_root, run)["T1w_mean"][1]


def regime_versions(regimes: dict[str, Regime]) -> dict[str, str]:
    return {name: regimes[name].version for name in REGIMES}


def run_is_current(tree_root: Path, run: FmriprepRun, keys: Optional[dict[str, str]],
                   versions: dict[str, str]) -> bool:
    """Every output exists and (when ``keys`` is given) the record matches these inputs and regimes."""
    paths = run_map_paths(tree_root, run)
    if not all(m.exists() and s.exists() for m, s in paths.values()):
        return False
    if keys is None:
        return True
    try:
        rec = json.loads(record_path(tree_root, run).read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return (rec.get("schema_version") == SCHEMA_VERSION and rec.get("input_keys") == keys
            and rec.get("regime_versions") == versions)


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def save_volume(values: np.ndarray, affine: np.ndarray, path: Path) -> None:
    import nibabel as nib

    img = nib.Nifti1Image(np.asarray(values, dtype=np.float32), affine)
    img.set_data_dtype(np.float32)
    img.set_qform(affine, code=1)
    img.set_sform(affine, code=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(img, str(path))


def save_surface(values: np.ndarray, path: Path) -> None:
    import nibabel as nib

    da = nib.gifti.GiftiDataArray(np.asarray(values, dtype=np.float32),
                                  intent="NIFTI_INTENT_NONE", datatype="NIFTI_TYPE_FLOAT32")
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.gifti.GiftiImage(darrays=[da]), str(path))


def load_surface_series(path: Path) -> np.ndarray:
    """(n_vol, n_vertex) float32 from a per-timepoint ``func.gii``."""
    import nibabel as nib

    img = nib.load(str(path))
    return np.vstack([np.asarray(d.data, dtype=np.float32) for d in img.darrays])


def load_surface_map(path: Path) -> np.ndarray:
    import nibabel as nib

    return np.asarray(nib.load(str(path)).darrays[0].data, dtype=np.float32)


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Run cell
# ---------------------------------------------------------------------------

def _clean_maps(data: np.ndarray, designs: dict[str, Any]) -> tuple[np.ndarray, dict[str, np.ndarray], dict]:
    """Temporal mean and per-regime tSNR of (n_vol, n_elem) data; one regime's residuals at a time."""
    mean, tsnr, dof = None, {}, {}
    for name, design in designs.items():
        res = dq.clean(data, design)
        if mean is None:
            mean = res.mean.astype(np.float32)
        tsnr[name] = res.tsnr.astype(np.float32)
        dof[name] = {"n_regressors": design.n_regressors, "dof_resid": design.dof_resid,
                     "dof_loss": round(design.dof_loss, 6)}
        del res
    return mean, tsnr, dof


def build_run_cell(tree_root: Path, run: FmriprepRun, regimes: dict[str, Regime], provenance: dict,
                   force: bool = False, log: Callable[[str], None] = print) -> Optional[dict]:
    """Write one run's T1w + fsnative mean and tSNR maps. None when current."""
    import nibabel as nib

    files = run_files(run)
    keys = {k: dq.file_sha256(p) for k, p in sorted(files.items())}
    versions = regime_versions(regimes)
    if not force and run_is_current(tree_root, run, keys, versions):
        log(f"{run.entity_prefix}: voxelmaps current, skipping")
        return None
    confounds = load_confounds(run)
    # A regime the run cannot carry raises here (RegimeNotApplicable). None of
    # REGIMES needs more than 6 aCompCor components, which every run has; a
    # failure is a new fact to look at, not a cell to declare absent silently.
    designs = {name: regime_design(regimes[name], confounds) for name in REGIMES}
    n_vol = len(confounds)
    n_nss = int(next(iter(designs.values())).n_nss)
    paths = run_map_paths(tree_root, run)
    base = {"schema_version": SCHEMA_VERSION, "cell": CELL, "sub": run.subject, "ses": run.session,
            "task": run.task, "run": run.run, "variant": run.variant}
    summaries: dict[str, Any] = {}
    dofs: dict[str, Any] = {}

    # Volume.
    bold = nib.load(str(files["bold"]))
    mask_img = nib.load(str(files["mask"]))
    mask = np.asarray(mask_img.dataobj) > 0
    if bold.shape[:3] != mask.shape or not np.allclose(bold.affine, mask_img.affine, atol=1e-3):
        raise ValueError(f"{run.entity_prefix}: T1w BOLD and brain mask are on different grids")
    data = dq._masked_timeseries(bold, mask)
    if data.shape[0] != n_vol:
        raise ValueError(f"{run.entity_prefix}: confounds has {n_vol} rows, T1w BOLD {data.shape[0]} volumes")
    tr = float(bold.header.get_zooms()[3])
    mean, tsnr, dof = _clean_maps(data, designs)
    del data
    dofs = dof
    vol_maps = {"mean": mean, **{f"tsnr_{r}": tsnr[r] for r in REGIMES}}
    for measure, v in vol_maps.items():
        full = np.full(mask.shape, np.nan, dtype=np.float32)
        full[mask] = v
        save_volume(full, bold.affine, paths[f"T1w_{measure}"][0])
        summaries[f"T1w_{measure}"] = map_summary(v)

    # Surface.
    for h in HEMIS:
        series = load_surface_series(files[f"surf_{h}"])
        if series.shape[0] != n_vol:
            raise ValueError(f"{run.entity_prefix} hemi-{h}: {series.shape[0]} surface volumes, {n_vol} confounds rows")
        mean_s, tsnr_s, _ = _clean_maps(series, designs)
        del series
        surf_maps = {"mean": mean_s, **{f"tsnr_{r}": tsnr_s[r] for r in REGIMES}}
        for measure, v in surf_maps.items():
            save_surface(v, paths[f"{h}_{measure}"][0])
            summaries[f"{h}_{measure}"] = map_summary(v)

    created = _now()
    record = {**base, "space": VOLUME_SPACE, "measure": "mean",
              "n_vol": n_vol, "n_nss": n_nss, "repetition_time": tr,
              "mask_n_voxels": int(mask.sum()), "grid_shape": list(mask.shape),
              "grid_zooms": [float(z) for z in bold.header.get_zooms()[:3]],
              "regimes": list(REGIMES), "regime_versions": versions, "dof": dofs,
              "summaries": summaries, "input_keys": keys, "inputs": {k: str(p) for k, p in files.items()},
              **provenance, "created": created}
    for key, (_, side) in paths.items():
        space_key, measure = key.split("_", 1)
        meta = {**base, "measure": "mean" if measure == "mean" else "tsnr",
                "regime": None if measure == "mean" else measure.removeprefix("tsnr_"),
                "space": VOLUME_SPACE if space_key == "T1w" else SURFACE_SPACE,
                "hemi": None if space_key == "T1w" else space_key,
                "summary": summaries[key], "input_keys": keys, **provenance, "created": created}
        if measure != "mean":
            meta["regime_version"] = versions[meta["regime"]]
            meta.update(dofs[meta["regime"]])
        side.write_text(json.dumps(record if key == "T1w_mean" else meta, indent=2, default=float) + "\n")
    log(f"{run.entity_prefix}: voxelmaps written (T1w mask {int(mask.sum())} voxels, "
        f"tSNR drift median {summaries['T1w_tsnr_drift']['median']:.1f} vol / "
        f"{summaries['L_tsnr_drift']['median']:.1f} L / {summaries['R_tsnr_drift']['median']:.1f} R)")
    return record


def collect_runs(tree_root: Path) -> pd.DataFrame:
    """One row per run × space × measure from the run-cell records."""
    rows = []
    for js in sorted(Path(tree_root).glob(f"sub-*/ses-*/func/*_space-{VOLUME_SPACE}_mean.json")):
        rec = json.loads(js.read_text())
        if rec.get("cell") != CELL:
            continue
        for key, s in rec["summaries"].items():
            space_key, measure = key.split("_", 1)
            rows.append({"sub": rec["sub"], "ses": rec["ses"], "task": rec["task"], "run": rec["run"],
                         "space": VOLUME_SPACE if space_key == "T1w" else SURFACE_SPACE,
                         "hemi": None if space_key == "T1w" else space_key, "measure": measure,
                         **s, "n_vol": rec["n_vol"], "n_nss": rec["n_nss"],
                         "grid_zooms": "x".join(f"{z:.3f}" for z in rec["grid_zooms"]),
                         "schema_version": rec["schema_version"], "code_version": rec.get("code_version"),
                         "fmriprep_version": rec.get("fmriprep_version")})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Subject cell: anatomy
# ---------------------------------------------------------------------------

def _anat_one(sub_dir: Path, subject: str, pattern: str) -> Path:
    """One fMRIPrep anat file, under ``anat/`` or (single-anat-session subjects) ``ses-*/anat/``."""
    hits = [p for p in sorted((sub_dir / "anat").glob(f"sub-{subject}*{pattern}")) if "_space-" not in p.name]
    if not hits:
        hits = [p for p in sorted(sub_dir.glob(f"ses-*/anat/sub-{subject}*{pattern}")) if "_space-" not in p.name]
    if len(hits) != 1:
        raise FileNotFoundError(f"expected one *{pattern} under {sub_dir}/anat or ses-*/anat, found "
                                f"{[str(p) for p in hits]}")
    return hits[0]


def anat_files(fmriprep_tree: Path, subject: str) -> dict[str, Path]:
    """Surfaces (T1w world coordinates), ribbon, MNI xfms and cortex labels."""
    sub_dir = Path(fmriprep_tree) / f"sub-{subject}"
    out: dict[str, Path] = {
        "ribbon": _anat_one(sub_dir, subject, "_desc-ribbon_mask.nii.gz"),
        "wm": _anat_one(sub_dir, subject, "_label-WM_probseg.nii.gz"),
    }
    for direction, pattern in (("xfm_mni2t1w", "_from-MNI152NLin2009cAsym_to-T1w_mode-image_xfm.h5"),
                               ("xfm_t1w2mni", "_from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5")):
        hits = sorted((sub_dir / "anat").glob(f"sub-{subject}*{pattern}")) or \
            sorted(sub_dir.glob(f"ses-*/anat/sub-{subject}*{pattern}"))
        if len(hits) != 1:
            raise FileNotFoundError(f"expected one *{pattern} for sub-{subject}, found {[str(p) for p in hits]}")
        out[direction] = hits[0]
    for h, fs in zip(HEMIS, ("lh", "rh")):
        for surf in ("white", "pial", "midthickness"):
            out[f"{surf}_{h}"] = _anat_one(sub_dir, subject, f"_hemi-{h}_{surf}.surf.gii")
        label = Path(fmriprep_tree) / "sourcedata" / "freesurfer" / f"sub-{subject}" / "label" / f"{fs}.cortex.label"
        if not label.exists():
            raise FileNotFoundError(f"FreeSurfer cortex label missing: {label}")
        out[f"cortex_{h}"] = label
    return out


def mni_reference(atlases_dir: Path) -> Path:
    """The MNI res-2 grid the tier-1 cells use: the Schaefer label volume defines it."""
    return Path(atlases_dir) / f"{dq.PARCELLATIONS['Schaefer17n400']['stem']}.nii.gz"


def reference_grid(runs: list[FmriprepRun]) -> tuple[tuple[int, ...], np.ndarray, list[str]]:
    """The subject's majority T1w grid, and the runs on any other grid."""
    import nibabel as nib

    grids: dict[tuple, list[FmriprepRun]] = collections.defaultdict(list)
    affines: dict[tuple, np.ndarray] = {}
    for r in runs:
        img = nib.load(str(r.mask))
        key = (img.shape, tuple(np.round(img.affine, 3).ravel()))
        grids[key].append(r)
        affines[key] = img.affine
    major = max(grids, key=lambda k: (len(grids[k]), k))
    off = sorted(r.entity_prefix for k, rs in grids.items() if k != major for r in rs)
    return major[0], affines[major], off


def onto_grid(values: np.ndarray, mask: np.ndarray, affine: np.ndarray, shape: tuple, ref_affine: np.ndarray
              ) -> tuple[np.ndarray, np.ndarray]:
    """Resample a masked map onto another grid: trilinear, mask-weighted, mask kept above 0.5."""
    from scipy.ndimage import map_coordinates

    if tuple(values.shape) == tuple(shape) and np.allclose(affine, ref_affine, atol=1e-3):
        return values, mask
    ii, jj, kk = np.meshgrid(*[np.arange(n) for n in shape], indexing="ij")
    world = ref_affine @ np.c_[ii.ravel(), jj.ravel(), kk.ravel(), np.ones(ii.size)].T
    ijk = (np.linalg.inv(affine) @ world)[:3]
    w = map_coordinates(mask.astype(np.float64), ijk, order=1, mode="constant", cval=0.0).reshape(shape)
    v = np.where(mask & np.isfinite(values), values, 0.0)
    s = map_coordinates(v.astype(np.float64), ijk, order=1, mode="constant", cval=0.0).reshape(shape)
    keep = w > 0.5
    out = np.full(shape, np.nan, dtype=np.float32)
    out[keep] = (s[keep] / w[keep]).astype(np.float32)
    return out, keep


# ---------------------------------------------------------------------------
# Subject cell: outputs
# ---------------------------------------------------------------------------

def subject_dir(tree_root: Path, subject: str) -> Path:
    return Path(tree_root) / f"sub-{subject}" / "func"


def session_map_path(tree_root: Path, subject: str, session: Optional[str], kind: str, space: str,
                     hemi: Optional[str] = None) -> Path:
    """Aggregate map path. ``kind``: ``mean`` | ``tsnr_<regime>`` | ``rel_<measure>`` | ``dropped``.

    Session maps (``session`` given) sit in ``ses-##/func/`` with no task entity;
    subject maps (``session`` None) in ``sub-##/func/``.
    """
    d = (Path(tree_root) / f"sub-{subject}" / f"ses-{session}" / "func") if session else subject_dir(tree_root, subject)
    ent = f"sub-{subject}" + (f"_ses-{session}" if session else "")
    if hemi:
        ent += f"_hemi-{hemi}"
    ent += f"_space-{space}" + (f"_res-{MNI_RES}" if space == MNI_SPACE else "")
    ext = ".func.gii" if hemi else ".nii.gz"
    if kind == "mean":
        name = f"{ent}_desc-norm_mean"
    elif kind.startswith("tsnr_"):
        name = f"{ent}_desc-{kind.removeprefix('tsnr_')}_tsnr"
    elif kind.startswith("rel_"):
        m = kind.removeprefix("rel_")
        name = f"{ent}_desc-{m.replace('_', '')}_relative"
    elif kind == "dropped":
        name = f"{ent}_desc-dropped_mask"
    elif kind in ("surfvol_mean", "surfvol_tsnr", "ribbon_inmask"):
        name = f"{ent}_desc-{kind.replace('_', '')}_ratio" if kind != "ribbon_inmask" else f"{ent}_desc-ribboninmask_frac"
    else:
        raise ValueError(f"unknown aggregate map kind {kind!r}")
    return d / f"{name}{ext}"


def subject_table_paths(tree_root: Path, subject: str) -> dict[str, Path]:
    d = subject_dir(tree_root, subject)
    return {"sidecar": d / f"sub-{subject}_desc-voxelmaps_qc.json",
            "sessions": d / f"sub-{subject}_desc-voxelmapssessions_qc.tsv",
            "parcels": d / f"sub-{subject}_desc-voxelmapsparcels_qc.tsv",
            "surfvol": d / f"sub-{subject}_desc-voxelmapssurfvol_qc.tsv"}


def subject_is_current(tree_root: Path, subject: str, keys: dict) -> bool:
    paths = subject_table_paths(tree_root, subject)
    if not all(p.exists() for p in paths.values()):
        return False
    side = json.loads(paths["sidecar"].read_text())
    return side.get("schema_version") == SCHEMA_VERSION and side.get("input_keys") == keys


def _parcel_medians(values: np.ndarray, labels: np.ndarray, domain: np.ndarray, index: np.ndarray) -> np.ndarray:
    out = np.full(len(index), np.nan)
    lab = labels[domain]
    val = np.asarray(values, dtype=np.float64)[domain]
    for i, idx in enumerate(index):
        v = val[lab == idx]
        v = v[np.isfinite(v)]
        if v.size:
            out[i] = np.median(v)
    return out


def _warp_to(src: Path, ref: Path, xfm: Path, out: Path, interp: str) -> None:
    import shutil
    import subprocess

    exe = shutil.which("antsApplyTransforms")
    if exe is None:
        raise RuntimeError("antsApplyTransforms is not on PATH; `module load ants/2.5.2` BEFORE activating the venv")
    subprocess.run([exe, "-d", "3", "-i", str(src), "-r", str(ref), "-t", str(xfm), "-n", interp,
                    "-o", str(out), "--float", "1"], check=True, capture_output=True)


def build_subject_cell(tree_root: Path, fmriprep_tree: Path, atlases_dir: Path, subject: str,
                       runs: list[FmriprepRun], provenance: dict, force: bool = False,
                       log: Callable[[str], None] = print) -> Optional[dict]:
    """Aggregate one subject's run cells into session/subject maps, dropout, surface-vs-volume rows."""
    import nibabel as nib

    from . import data_quality_prf as dqp

    missing = [r.entity_prefix for r in runs if not run_is_current(tree_root, r, None, {})]
    if missing:
        raise FileNotFoundError(f"sub-{subject}: {len(missing)} run cells missing (first: {missing[:3]}); "
                                "run `tier1.py voxelmaps` for them first")
    anat = anat_files(fmriprep_tree, subject)
    run_records = {r.entity_prefix: record_path(tree_root, r) for r in runs}
    keys = {"anat": {k: dq.file_sha256(p) for k, p in sorted(anat.items())},
            "runs": {k: json.loads(p.read_text())["input_keys"]["bold"] for k, p in sorted(run_records.items())},
            "atlases": dq.atlases_sha256(atlases_dir)}
    if not force and subject_is_current(tree_root, subject, keys):
        log(f"sub-{subject}: voxelmaps subject cell current, skipping")
        return None

    shape, ref_affine, off_grid = reference_grid(runs)
    log(f"sub-{subject}: {len(runs)} runs, grid {shape}, {len(off_grid)} off-grid: {off_grid}")

    # Pass 1: run masks on the reference grid -> consensus mask.
    masks: dict[str, np.ndarray] = {}
    for r in runs:
        img = nib.load(str(r.mask))
        m = np.asarray(img.dataobj) > 0
        masks[r.entity_prefix] = onto_grid(m.astype(np.float32), m, img.affine, shape, ref_affine)[1]
    count = np.sum([m for m in masks.values()], axis=0)
    consensus = count >= CONSENSUS * len(runs)
    log(f"sub-{subject}: consensus mask {int(consensus.sum())} voxels (in ≥ {CONSENSUS:.0%} of runs)")

    # Anatomy on the reference grid.
    ribbon_img = nib.load(str(anat["ribbon"]))
    ribbon = onto_grid(np.ones(ribbon_img.shape, np.float32), np.asarray(ribbon_img.dataobj) > 0,
                       ribbon_img.affine, shape, ref_affine)[1]
    surfs = {f"{s}_{h}": np.asarray(nib.load(str(anat[f"{s}_{h}"])).darrays[0].data, dtype=np.float64)
             for s in ("white", "pial", "midthickness") for h in HEMIS}
    cortex = {h: np.asarray(nib.freesurfer.read_label(str(anat[f"cortex_{h}"])), dtype=np.int64) for h in HEMIS}
    cortex_mask = {}
    for h in HEMIS:
        cm = np.zeros(len(surfs[f"white_{h}"]), dtype=bool)
        cm[cortex[h]] = True
        cortex_mask[h] = cm
    points = {h: ribbon_points(surfs[f"white_{h}"], surfs[f"pial_{h}"]) for h in HEMIS}

    sessions: dict[str, list[FmriprepRun]] = collections.defaultdict(list)
    for r in runs:
        sessions[r.session].append(r)
    vol_kinds = MEASURES
    sess_vol: dict[str, dict[str, np.ndarray]] = {}
    sess_surf: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    sess_sv: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    run_rows: list[dict] = []

    # Pass 2: per session, load run maps, normalise, median.
    for ses in sorted(sessions):
        per_kind: dict[str, list[np.ndarray]] = collections.defaultdict(list)
        per_surf: dict[str, dict[str, list[np.ndarray]]] = {h: collections.defaultdict(list) for h in HEMIS}
        per_sv: dict[str, dict[str, list[np.ndarray]]] = {h: collections.defaultdict(list) for h in HEMIS}
        for r in sessions[ses]:
            paths = run_map_paths(tree_root, r)
            mimg = nib.load(str(paths["T1w_mean"][0]))
            rmask_native = np.asarray(nib.load(str(r.mask)).dataobj) > 0
            native = {k: np.asarray(nib.load(str(paths[f"T1w_{k}"][0])).dataobj, dtype=np.float32) for k in vol_kinds}
            on = {k: onto_grid(v, rmask_native, mimg.affine, shape, ref_affine)[0] for k, v in native.items()}
            scale = run_scale(on["mean"], consensus & masks[r.entity_prefix])
            on["mean"] = on["mean"] / scale
            for k in vol_kinds:
                per_kind[k].append(on[k])
            row = {"sub": subject, "ses": ses, "task": r.task, "run": r.run, "vol_scale": scale,
                   "off_grid": r.entity_prefix in off_grid}
            for h in HEMIS:
                smaps = {k: load_surface_map(paths[f"{h}_{k}"][0]) for k in vol_kinds}
                if smaps["mean"].size != len(surfs[f"white_{h}"]):
                    raise ValueError(f"{r.entity_prefix} hemi-{h}: {smaps['mean'].size} values, "
                                     f"{len(surfs[f'white_{h}'])} surface vertices")
                # Surface vs volume, on the run's own (native) grid: raw means, so the gain cancels.
                vs_mean, inmask = ribbon_sample(native["mean"], rmask_native, mimg.affine, points[h])
                vs_tsnr, _ = ribbon_sample(native[f"tsnr_{SAMPLED_REGIME}"], rmask_native, mimg.affine, points[h])
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratio_mean = np.where(vs_mean > 0, smaps["mean"] / vs_mean, np.nan).astype(np.float32)
                    ratio_tsnr = np.where(vs_tsnr > 0, smaps[f"tsnr_{SAMPLED_REGIME}"] / vs_tsnr, np.nan).astype(np.float32)
                per_sv[h]["surfvol_mean"].append(ratio_mean)
                per_sv[h]["surfvol_tsnr"].append(ratio_tsnr)
                per_sv[h]["ribbon_inmask"].append(inmask)
                s_scale = run_scale(smaps["mean"], cortex_mask[h])
                smaps["mean"] = smaps["mean"] / s_scale
                for k in vol_kinds:
                    per_surf[h][k].append(np.where(cortex_mask[h], smaps[k], np.nan).astype(np.float32))
                cm = cortex_mask[h]
                row[f"surf_scale_{h}"] = s_scale
                row[f"surfvol_mean_median_{h}"] = float(np.nanmedian(ratio_mean[cm]))
                row[f"ribbon_inmask_median_{h}"] = float(np.median(inmask[cm]))
            run_rows.append(row)
        sess_vol[ses] = {k: nanmedian_stack(v) for k, v in per_kind.items()}
        sess_surf[ses] = {h: {k: nanmedian_stack(v) for k, v in per_surf[h].items()} for h in HEMIS}
        sess_sv[ses] = {h: {k: nanmedian_stack(v) for k, v in per_sv[h].items()} for h in HEMIS}
        log(f"sub-{subject} ses-{ses}: {len(sessions[ses])} runs aggregated")

    subj_vol = {k: nanmedian_stack([sess_vol[s][k] for s in sess_vol]) for k in vol_kinds}
    subj_surf = {h: {k: nanmedian_stack([sess_surf[s][h][k] for s in sess_surf]) for k in vol_kinds} for h in HEMIS}
    subj_sv = {h: {k: nanmedian_stack([sess_sv[s][h][k] for s in sess_sv]) for k in sess_sv[next(iter(sess_sv))][h]}
               for h in HEMIS}

    # Parcellations: volume labels warped MNI -> reference grid; vertex labels from the midthickness.
    with tempfile.TemporaryDirectory() as tmp:
        ref_path = Path(tmp) / "ref.nii.gz"
        save_volume(np.zeros(shape, np.float32), ref_affine, ref_path)
        labels = {}
        for name, spec in dq.PARCELLATIONS.items():
            out = Path(tmp) / f"{name}.nii.gz"
            dqp.warp_labels(Path(atlases_dir) / f"{spec['stem']}.nii.gz", ref_path, anat["xfm_mni2t1w"], out)
            labels[name] = np.asarray(nib.load(str(out)).dataobj).round().astype(np.int32)
    tables = {name: dqp.kept_table(name, atlases_dir) for name in dq.PARCELLATIONS}
    vlabels = {}
    for h in HEMIS:
        ijk = np.rint((np.linalg.inv(ref_affine) @ np.c_[surfs[f"midthickness_{h}"],
                                                          np.ones(len(surfs[f"midthickness_{h}"]))].T)[:3]).astype(int)
        inside = np.all((ijk >= 0) & (ijk < np.array(shape)[:, None]), axis=0)
        lab = np.zeros(ijk.shape[1], dtype=np.int32)
        lab[inside] = labels["Schaefer17n400"][ijk[0, inside], ijk[1, inside], ijk[2, inside]]
        vlabels[h] = lab

    # Tables.
    sess_rows, parcel_rows = [], []
    for ses in sorted(sess_vol):
        rel_vol = relative(sess_vol[ses]["mean"], subj_vol["mean"])
        rows = {"sub": subject, "ses": ses, "n_runs": len(sessions[ses])}
        sess_rows.append({**rows, "space": VOLUME_SPACE, "hemi": None, "domain": "consensus_mask",
                          **dropout_summary(rel_vol, consensus),
                          **{f"rel_{k}_median": float(np.nanmedian(relative(sess_vol[ses][k], subj_vol[k])[consensus]))
                             for k in vol_kinds if k != "mean"}})
        for h in HEMIS:
            rel_s = relative(sess_surf[ses][h]["mean"], subj_surf[h]["mean"])
            cm = cortex_mask[h]
            sess_rows.append({**rows, "space": SURFACE_SPACE, "hemi": h, "domain": "cortex",
                              **dropout_summary(rel_s, cm),
                              **{f"rel_{k}_median": float(np.nanmedian(relative(sess_surf[ses][h][k], subj_surf[h][k])[cm]))
                                 for k in vol_kinds if k != "mean"},
                              "surfvol_mean_median": float(np.nanmedian(sess_sv[ses][h]["surfvol_mean"][cm])),
                              "surfvol_tsnr_median": float(np.nanmedian(sess_sv[ses][h]["surfvol_tsnr"][cm])),
                              "ribbon_inmask_median": float(np.nanmedian(sess_sv[ses][h]["ribbon_inmask"][cm])),
                              "frac_ribbon_below_floor": float(np.mean(sess_sv[ses][h]["ribbon_inmask"][cm] < RIBBON_FLOOR))})
        for name, lab in labels.items():
            t = tables[name]
            idx = t["index"].to_numpy()
            for i, (pidx, pname) in enumerate(zip(idx, t["name"])):
                sel = consensus & (lab == pidx)
                d = dropout_summary(rel_vol, sel)
                parcel_rows.append({"sub": subject, "ses": ses, "space": VOLUME_SPACE, "hemi": None,
                                    "atlas": name, "parcel": pname, **d})
        for h in HEMIS:
            rel_s = relative(sess_surf[ses][h]["mean"], subj_surf[h]["mean"])
            t = tables["Schaefer17n400"]
            for pidx, pname in zip(t["index"], t["name"]):
                sel = cortex_mask[h] & (vlabels[h] == pidx)
                if not sel.any():
                    continue
                parcel_rows.append({"sub": subject, "ses": ses, "space": SURFACE_SPACE, "hemi": h,
                                    "atlas": "Schaefer17n400", "parcel": pname, **dropout_summary(rel_s, sel)})
    sess_df = pd.DataFrame(sess_rows)
    sess_df["examine"] = False
    for (space, hemi), g in sess_df.groupby(["space", sess_df["hemi"].fillna("-")]):
        sess_df.loc[g.index, "examine"] = examine_flags(g["frac_lost"]).to_numpy()

    # Surface vs volume, subject level.
    t = tables["Schaefer17n400"]
    idx = t["index"].to_numpy()
    sv_rows = []
    for h in HEMIS:
        cm = cortex_mask[h]
        surf_par = _parcel_medians(subj_surf[h]["mean"], vlabels[h], cm, idx)
        # Volume side, two ways: the ribbon voxels of each parcel (no surface involved), and the
        # surface-sampled volume (the same vertices). Normalised means on both sides.
        hemi_vox = labels["Schaefer17n400"] > 0
        names = t["name"].astype(str).to_numpy()
        hemi_sel = np.array([("LH" in n) if h == "L" else ("RH" in n) for n in names])
        vol_par = _parcel_medians(subj_vol["mean"], labels["Schaefer17n400"], consensus & ribbon & hemi_vox, idx)
        rho_ribbon, n_ribbon = spearman(surf_par[hemi_sel], vol_par[hemi_sel])
        sampled_norm, _ = ribbon_sample(subj_vol["mean"], consensus, ref_affine, points[h])
        samp_par = _parcel_medians(sampled_norm, vlabels[h], cm, idx)
        rho_sampled, n_sampled = spearman(surf_par[hemi_sel], samp_par[hemi_sel])
        ratio = subj_sv[h]["surfvol_mean"][cm]
        inm = subj_sv[h]["ribbon_inmask"][cm]
        # Depth profile: the subject's normalised mean at each ribbon depth, relative to the depth average.
        depth_vals, _ = sample_volume(subj_vol["mean"], consensus, ref_affine, points[h])
        avg = np.nanmean(depth_vals, axis=0)
        prof = {f"depth_{d:.1f}_rel": float(np.nanmedian((depth_vals[i] / avg)[cm])) for i, d in enumerate(DEPTHS)}
        sv_rows.append({
            "sub": subject, "hemi": h, "n_cortex_vertices": int(cm.sum()),
            "surfvol_mean_median": float(np.nanmedian(ratio)),
            "surfvol_mean_p05": float(np.nanpercentile(ratio, 5)), "surfvol_mean_p95": float(np.nanpercentile(ratio, 95)),
            "surfvol_tsnr_median": float(np.nanmedian(subj_sv[h]["surfvol_tsnr"][cm])),
            "rho_parcel_sampled": rho_sampled, "n_parcel_sampled": n_sampled,
            "rho_parcel_ribbon": rho_ribbon, "n_parcel_ribbon": n_ribbon,
            "ribbon_inmask_median": float(np.nanmedian(inm)),
            "frac_ribbon_below_floor": float(np.mean(inm < RIBBON_FLOOR)),
            "adequate": bool(0.9 <= float(np.nanmedian(ratio)) <= 1.1 and np.isfinite(rho_sampled) and rho_sampled >= 0.9),
            **prof,
        })
        for pidx, pname, sv, vv, sp in zip(idx, names, surf_par, vol_par, samp_par):
            if (("LH" in pname) if h == "L" else ("RH" in pname)):
                sel = cm & (vlabels[h] == pidx)
                parcel_rows.append({"sub": subject, "ses": None, "space": SURFACE_SPACE, "hemi": h,
                                    "atlas": "Schaefer17n400", "parcel": pname,
                                    "surf_mean_norm": sv, "vol_ribbon_mean_norm": vv, "vol_sampled_mean_norm": sp,
                                    "n": int(sel.sum()),
                                    "ribbon_inmask_median": float(np.nanmedian(subj_sv[h]["ribbon_inmask"][sel])) if sel.any() else np.nan,
                                    "frac_ribbon_below_floor": float(np.mean(subj_sv[h]["ribbon_inmask"][sel] < RIBBON_FLOOR)) if sel.any() else np.nan})
    sv_df = pd.DataFrame(sv_rows)

    # Write maps: session + subject, volume + surface, then MNI warps of the volume ones.
    written: list[Path] = []
    for ses in sorted(sess_vol):
        rel_vol = relative(sess_vol[ses]["mean"], subj_vol["mean"])
        vol_out = {**sess_vol[ses], **{f"rel_{k}": relative(sess_vol[ses][k], subj_vol[k]) for k in vol_kinds},
                   "dropped": np.where(consensus, (np.isfinite(rel_vol) & (rel_vol < DROP_FLOOR)) | ~np.isfinite(rel_vol), np.nan).astype(np.float32)}
        for k, v in vol_out.items():
            p = session_map_path(tree_root, subject, ses, k, VOLUME_SPACE)
            save_volume(v, ref_affine, p)
            written.append(p)
        for h in HEMIS:
            rel_s = relative(sess_surf[ses][h]["mean"], subj_surf[h]["mean"])
            surf_out = {**sess_surf[ses][h],
                        **{f"rel_{k}": relative(sess_surf[ses][h][k], subj_surf[h][k]) for k in vol_kinds},
                        "dropped": np.where(cortex_mask[h], (np.isfinite(rel_s) & (rel_s < DROP_FLOOR)) | ~np.isfinite(rel_s), np.nan).astype(np.float32)}
            for k, v in surf_out.items():
                save_surface(v, session_map_path(tree_root, subject, ses, k, SURFACE_SPACE, h))
    for k, v in subj_vol.items():
        p = session_map_path(tree_root, subject, None, k, VOLUME_SPACE)
        save_volume(v, ref_affine, p)
        written.append(p)
    for h in HEMIS:
        for k, v in subj_surf[h].items():
            save_surface(v, session_map_path(tree_root, subject, None, k, SURFACE_SPACE, h))
        for k, v in subj_sv[h].items():
            save_surface(v, session_map_path(tree_root, subject, None, k, SURFACE_SPACE, h))
    mni_ref = mni_reference(atlases_dir)
    for p in written:
        ent = p.name.split("_space-")[0]
        tail = p.name.split(f"_space-{VOLUME_SPACE}")[1]
        out = p.with_name(f"{ent}_space-{MNI_SPACE}_res-{MNI_RES}{tail}")
        # NaN does not survive interpolation: warp the finite part and its support, keep support > 0.5.
        img = nib.load(str(p))
        v = np.asarray(img.dataobj, dtype=np.float32)
        fin = np.isfinite(v)
        with tempfile.TemporaryDirectory() as tmp:
            src_v, src_w = Path(tmp) / "v.nii.gz", Path(tmp) / "w.nii.gz"
            save_volume(np.where(fin, v, 0), ref_affine, src_v)
            save_volume(fin.astype(np.float32), ref_affine, src_w)
            dst_v, dst_w = Path(tmp) / "vo.nii.gz", Path(tmp) / "wo.nii.gz"
            interp = "NearestNeighbor" if p.name.endswith("_mask.nii.gz") else "Linear"
            _warp_to(src_v, mni_ref, anat["xfm_t1w2mni"], dst_v, interp)
            _warp_to(src_w, mni_ref, anat["xfm_t1w2mni"], dst_w, interp)
            wv = np.asarray(nib.load(str(dst_v)).dataobj, dtype=np.float32)
            ww = np.asarray(nib.load(str(dst_w)).dataobj, dtype=np.float32)
            mni_aff = nib.load(str(dst_v)).affine
        res = np.full(wv.shape, np.nan, dtype=np.float32)
        keep = ww > 0.5
        res[keep] = wv[keep] / ww[keep] if interp == "Linear" else wv[keep]
        save_volume(res, mni_aff, out)

    paths = subject_table_paths(tree_root, subject)
    paths["sessions"].parent.mkdir(parents=True, exist_ok=True)
    sess_df.to_csv(paths["sessions"], sep="\t", index=False, na_rep="n/a", float_format="%.6g")
    pd.DataFrame(parcel_rows).to_csv(paths["parcels"], sep="\t", index=False, na_rep="n/a", float_format="%.6g")
    sv_df.to_csv(paths["surfvol"], sep="\t", index=False, na_rep="n/a", float_format="%.6g")
    side = {"schema_version": SCHEMA_VERSION, "cell": CELL, "subject": subject,
            "n_runs": len(runs), "n_sessions": len(sess_vol), "grid_shape": list(shape),
            "grid_affine": np.round(ref_affine, 6).tolist(), "off_grid_runs": off_grid,
            "consensus_fraction": CONSENSUS, "consensus_n_voxels": int(consensus.sum()),
            "drop_floor": DROP_FLOOR, "examine_multiple": EXAMINE_MULTIPLE, "ribbon_floor": RIBBON_FLOOR,
            "depths": list(DEPTHS), "sampled_regime": SAMPLED_REGIME, "regimes": list(REGIMES),
            "runs": run_rows, "input_keys": keys, "inputs": {k: str(p) for k, p in anat.items()},
            **provenance, "created": _now()}
    paths["sidecar"].write_text(json.dumps(side, indent=2, default=float) + "\n")
    log(f"sub-{subject}: voxelmaps subject cell written ({len(sess_vol)} sessions; "
        f"{int(sess_df['examine'].sum())} session x space rows flagged to examine)")
    return side


def collect_subjects(tree_root: Path) -> dict[str, pd.DataFrame]:
    """The three subject tables, concatenated over subjects."""
    out = {}
    for key, table in (("sessions", SESSIONS_TABLE), ("parcels", PARCELS_TABLE), ("surfvol", SURFVOL_TABLE)):
        frames = [pd.read_csv(p, sep="\t", na_values=["n/a"], dtype={"sub": str, "ses": str})
                  for p in sorted(Path(tree_root).glob(f"sub-*/func/sub-*_desc-voxelmaps{key}_qc.tsv"))]
        out[table] = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return out
