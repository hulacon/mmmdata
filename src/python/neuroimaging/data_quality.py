"""Tier 1 of the data-quality collection: clean one run under one regime, measure it.

The collection (``derivatives/data_quality/``, mmmdata-agents
``docs/workbench/data-quality/``) answers "what does each confound regime do
to this dataset" with one regenerable tree keyed like the catalog. Tier 1 is
per run × regime and is rebuilt when that run's fMRIPrep input changes; tier 2
pools tier-1 caches and never touches voxels. This module is tier 1's library:
the cleaning step, the three per-run measures, and the writer. The driver is
``scripts/data_quality/tier1.py``.

Per run × regime it writes, beside each other in ``sub-XX/ses-YY/func/``::

    <prefix>_space-S_desc-<regime>_tsnr.nii.gz       voxelwise tSNR (float32)
    <prefix>_space-S_desc-<regime>_tsnr.json         run-level facts + provenance
    <prefix>_space-S_seg-<atlas>_desc-<regime>_timeseries.tsv   parcel means
    <prefix>_space-S_seg-<atlas>_desc-<regime>_timeseries.json  per-parcel coverage,
                                                                variance removed

Definitions (measure registry v0.2, data-quality ``out/measure-registry.md``):

* **Cleaning** is ordinary least squares of every in-mask voxel on
  ``[intercept, regime columns]`` over the volumes fMRIPrep did not flag as
  non-steady-state. Flagged volumes are excluded from the fit and are NaN in
  every residual series — the time axis is never trimmed, so stimulus and
  cross-subject alignment survive (DECIDED 2026-09-23).
* **tSNR** is the voxel's temporal mean over the fitted volumes divided by
  the residual standard deviation with ``dof_resid`` in the denominator
  (``sqrt(RSS / (n_fit - n_regressors - 1))``). Under the ``none`` regime
  that is the classic mean/sd; under any other it is what the regression
  bought, read beside the degrees of freedom it cost.
* **Parcel time series** are the mean residual over the parcel's in-mask
  voxels. ``coverage`` is the fraction of the parcel's atlas voxels inside
  the run's brain mask; a parcel with no in-mask voxel is all-NaN.
* **Variance removed** per parcel is the mean over its in-mask voxels of
  ``1 - var(resid) / var(raw)`` (plain variances, ddof 0).
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from .confounds import Regime, RegimeDesign, regime_design
from .fmriprep_layout import space_part
from .io import FmriprepRun, load_bold, load_confounds, load_mask

TREE_NAME = "data_quality"
SCHEMA_VERSION = "1.0"

#: The parcellations every tier-1 cache carries (DECIDED 2026-09-23). Keys are
#: the ``seg-`` label of the output; values locate the dseg pair under
#: ``derivatives/atlases`` and name label rows to leave out.
PARCELLATIONS: dict[str, dict[str, Any]] = {
    "Schaefer17n400": {
        "stem": "tpl-MNI152NLin2009cAsym/anat/tpl-MNI152NLin2009cAsym_atlas-Schaefer2018_seg-17n_scale-400_res-2_dseg",
        "exclude_substrings": (),
    },
    "HOSPA": {
        "stem": "tpl-MNI152NLin2009cAsym/anat/tpl-MNI152NLin2009cAsym_atlas-HOSPA_res-2_desc-th25_dseg",
        # The maxprob subcortical atlas also labels whole-hemisphere tissue
        # classes; only the structures are parcels here.
        "exclude_substrings": ("White Matter", "Cerebral Cortex", "Ventricle"),
    },
}


# ---------------------------------------------------------------------------
# Atlases
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Parcellation:
    """A label image restricted to one run's mask, with its label table."""

    name: str
    labels_in_mask: np.ndarray  # int per in-mask voxel, 0 = no parcel
    table: pd.DataFrame  # index, name, n_voxels_atlas, n_voxels_mask, coverage


def load_parcellation(
    name: str, atlases_dir: Path, mask_bool: np.ndarray, affine: np.ndarray
) -> Parcellation:
    """Load one of :data:`PARCELLATIONS` onto a run's mask.

    Raises ValueError when the atlas grid is not the run's grid: parcels
    resampled on the fly would silently move between runs.
    """
    import nibabel as nib

    spec = PARCELLATIONS[name]
    nifti = Path(atlases_dir) / f"{spec['stem']}.nii.gz"
    tsv = Path(atlases_dir) / f"{spec['stem']}.tsv"
    for p in (nifti, tsv):
        if not p.exists():
            raise FileNotFoundError(f"Atlas file missing: {p}. Stage it under derivatives/atlases first.")
    img = nib.load(str(nifti))
    if img.shape != mask_bool.shape or not np.allclose(img.affine, affine, atol=1e-3):
        raise ValueError(
            f"Atlas {name} is on a different grid ({img.shape}) from the run's mask "
            f"({mask_bool.shape}); tier 1 does not resample."
        )
    labels = np.asarray(img.dataobj).astype(np.int32)
    table = pd.read_csv(tsv, sep="\t")
    keep = ~table["name"].astype(str).apply(
        lambda n: any(s in n for s in spec["exclude_substrings"])
    )
    table = table.loc[keep, ["index", "name"]].reset_index(drop=True)
    in_mask = labels[mask_bool]
    n_atlas = {int(i): int((labels == i).sum()) for i in table["index"]}
    n_mask = {int(i): int((in_mask == i).sum()) for i in table["index"]}
    table["n_voxels_atlas"] = table["index"].map(n_atlas)
    table["n_voxels_mask"] = table["index"].map(n_mask)
    table["coverage"] = table["n_voxels_mask"] / table["n_voxels_atlas"].replace(0, np.nan)
    # Labels not in the kept table become 0 so they never form a parcel.
    kept = set(int(i) for i in table["index"])
    in_mask = np.where(np.isin(in_mask, list(kept)), in_mask, 0)
    return Parcellation(name=name, labels_in_mask=in_mask, table=table)


# ---------------------------------------------------------------------------
# Cleaning and measures
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class CleanResult:
    """One run under one regime: residuals and the voxelwise facts about them."""

    design: RegimeDesign
    residuals: np.ndarray  # (n_vol, n_voxels) float32, NaN rows at non-steady-state volumes
    mean: np.ndarray  # voxel temporal mean over the fitted volumes
    sd_resid: np.ndarray  # dof-corrected residual sd
    var_ratio: np.ndarray  # var(resid) / var(raw) over the fitted volumes, ddof 0

    @property
    def tsnr(self) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            out = self.mean / self.sd_resid
        out[~np.isfinite(out)] = np.nan
        return out


def clean(data: np.ndarray, design: RegimeDesign, chunk: int = 20_000) -> CleanResult:
    """Residualise ``data`` (n_vol × n_voxels) on the regime's design plus an intercept."""
    data = np.asarray(data)
    n_vol = data.shape[0]
    if len(design.columns) != n_vol:
        raise ValueError(
            f"Design has {len(design.columns)} rows but the BOLD has {n_vol} volumes; "
            "the confounds TSV and the NIfTI do not belong to the same run."
        )
    dof = design.dof_resid
    if dof < 1:
        raise ValueError(
            f"Regime {design.regime.name!r} leaves {dof} residual degrees of freedom on a "
            f"{n_vol}-volume run; it cannot be fitted here."
        )
    fit = ~design.nss
    X = np.column_stack([np.ones(n_vol), design.columns.to_numpy(dtype=np.float64)])[fit]
    # One pseudo-inverse serves every voxel; the voxel axis is chunked so the
    # float64 temporaries stay bounded on long runs (the longest is ~1,900
    # volumes x ~240k voxels).
    pinv = np.linalg.pinv(X)
    n_vox = data.shape[1]
    residuals = np.full(data.shape, np.nan, dtype=np.float32)
    mean = np.empty(n_vox)
    sd_resid = np.empty(n_vox)
    var_ratio = np.empty(n_vox)
    for start in range(0, n_vox, chunk):
        sl = slice(start, min(start + chunk, n_vox))
        Yf = data[fit, sl].astype(np.float64)
        resid = Yf - X @ (pinv @ Yf)
        residuals[fit, sl] = resid.astype(np.float32)
        mean[sl] = Yf.mean(axis=0)
        sd_resid[sl] = np.sqrt(np.einsum("ij,ij->j", resid, resid) / dof)
        with np.errstate(divide="ignore", invalid="ignore"):
            var_ratio[sl] = resid.var(axis=0) / Yf.var(axis=0)
    var_ratio[~np.isfinite(var_ratio)] = np.nan
    return CleanResult(
        design=design,
        residuals=residuals,
        mean=mean,
        sd_resid=sd_resid,
        var_ratio=var_ratio,
    )


def parcel_timeseries(result: CleanResult, parc: Parcellation) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mean residual per parcel per volume, and the per-parcel table.

    Returns ``(timeseries, table)``: ``timeseries`` has one column per parcel
    name, NaN rows at non-steady-state volumes and all-NaN columns for
    parcels with no in-mask voxel; ``table`` adds ``var_removed`` to the
    parcellation's coverage columns.
    """
    labels = parc.labels_in_mask
    table = parc.table.copy()
    series = {}
    removed = []
    for idx, name in zip(table["index"], table["name"]):
        sel = labels == int(idx)
        if sel.any():
            series[str(name)] = result.residuals[:, sel].mean(axis=1)
            removed.append(float(np.nanmean(1.0 - result.var_ratio[sel])))
        else:
            series[str(name)] = np.full(result.residuals.shape[0], np.nan, dtype=np.float32)
            removed.append(np.nan)
    table["var_removed"] = removed
    return pd.DataFrame(series), table


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

def file_sha256(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def pipeline_version(derivatives_tree: Path) -> str:
    """The ``GeneratedBy[0].Version`` of a derivative tree's dataset_description.json.

    Required, not guessed: every tier-1 row must say which fMRIPrep made its
    input, or the rebuild-when-input-changes rule has nothing to compare.
    """
    desc = Path(derivatives_tree) / "dataset_description.json"
    if not desc.exists():
        raise FileNotFoundError(
            f"{desc} is missing, so the pipeline version of the input cannot be recorded. "
            "A derivative tree without dataset_description.json is not a valid input."
        )
    meta = json.loads(desc.read_text())
    try:
        return str(meta["GeneratedBy"][0]["Version"])
    except (KeyError, IndexError, TypeError):
        raise KeyError(f"{desc} has no GeneratedBy[0].Version to record") from None


def code_version(repo_root: Optional[Path] = None) -> str:
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[3]
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True, timeout=10,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Output layout
# ---------------------------------------------------------------------------

def output_dir(tree_root: Path, run: FmriprepRun) -> Path:
    return Path(tree_root) / f"sub-{run.subject}" / f"ses-{run.session}" / "func"


def output_stem(run: FmriprepRun, regime: str, seg: Optional[str] = None) -> str:
    """``sub-XX_ses-YY_task-T[_run-RR][_space-S][_seg-A]_desc-<regime>``."""
    stem = run.entity_prefix + space_part(run.space)  # the fragment carries its own "_"
    if seg:
        stem += f"_seg-{seg}"
    return f"{stem}_desc-{regime}"


def tsnr_paths(tree_root: Path, run: FmriprepRun, regime: str) -> tuple[Path, Path]:
    stem = output_dir(tree_root, run) / f"{output_stem(run, regime)}_tsnr"
    return stem.with_name(stem.name + ".nii.gz"), stem.with_name(stem.name + ".json")


def timeseries_paths(tree_root: Path, run: FmriprepRun, regime: str, seg: str) -> tuple[Path, Path]:
    stem = output_dir(tree_root, run) / f"{output_stem(run, regime, seg)}_timeseries"
    return stem.with_name(stem.name + ".tsv"), stem.with_name(stem.name + ".json")


def ensure_dataset_description(tree_root: Path, fmriprep_version: str, code_sha: str) -> Path:
    """Write the tree's ``dataset_description.json`` once so the catalog indexes it."""
    tree_root = Path(tree_root)
    tree_root.mkdir(parents=True, exist_ok=True)
    desc = tree_root / "dataset_description.json"
    if desc.exists():
        return desc
    desc.write_text(json.dumps({
        "Name": "MMMData data-quality collection",
        "BIDSVersion": "1.9.0",
        "DatasetType": "derivative",
        "GeneratedBy": [{
            "Name": "mmmdata data_quality tier1",
            "Version": code_sha,
            "CodeURL": "https://github.com/jhutchin/mmmdata",
            "Description": (
                "Per-run x confound-regime cleaning, tSNR and parcel caches; "
                "the measure registry lives in mmmdata-agents docs/workbench/data-quality/."
            ),
        }],
        "SourceDatasets": [{"URL": "derivatives/fmriprep", "Version": fmriprep_version}],
        "SchemaVersion": SCHEMA_VERSION,
    }, indent=2) + "\n")
    return desc


def is_current(tree_root: Path, run: FmriprepRun, regime: Regime, input_sha: str) -> bool:
    """True when every output for this run × regime exists and was built from this input."""
    nii, js = tsnr_paths(tree_root, run, regime.name)
    if not (nii.exists() and js.exists()):
        return False
    for seg in PARCELLATIONS:
        tsv, sj = timeseries_paths(tree_root, run, regime.name, seg)
        if not (tsv.exists() and sj.exists()):
            return False
    try:
        meta = json.loads(js.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return (
        meta.get("input_bold_sha256") == input_sha
        and meta.get("regime_version") == regime.version
        and meta.get("schema_version") == SCHEMA_VERSION
    )


# ---------------------------------------------------------------------------
# One run, every regime
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class RunInputs:
    """Everything loaded once per run and shared by every regime."""

    run: FmriprepRun
    data: np.ndarray  # (n_vol, n_voxels) float32, in-mask voxels only
    mask_bool: np.ndarray
    affine: np.ndarray
    confounds: pd.DataFrame
    tr: float
    input_bold_sha256: str
    input_confounds_sha256: str
    parcellations: dict[str, Parcellation]


def _masked_timeseries(bold: Any, mask_bool: np.ndarray) -> np.ndarray:
    """In-mask voxels x time as (n_vol, n_voxels) float32.

    One full read, deliberately: a gzipped NIfTI's array proxy re-decompresses
    from the file's start for every slab, so reading the 4D in volume blocks
    turned a 45 s run into 2 min and the longest run into 14 min (pilot
    47405564). The transient is the whole 4D as float32 (~8 G for the
    1,858-volume run); the solver's voxel chunking keeps everything after
    this bounded.
    """
    vols = np.asarray(bold.dataobj, dtype=np.float32)
    masked = vols[mask_bool]  # (n_voxels, n_vol)
    del vols  # the 4D transient goes before the transpose copy is made
    return np.ascontiguousarray(masked.T)


def load_run_inputs(run: FmriprepRun, atlases_dir: Path) -> RunInputs:
    """Load BOLD, mask, confounds and the parcellations for one run."""
    bold = load_bold(run)
    mask_img = load_mask(run)
    mask_bool = np.asarray(mask_img.dataobj).astype(bool)
    if bold.shape[:3] != mask_bool.shape or not np.allclose(bold.affine, mask_img.affine, atol=1e-3):
        raise ValueError(f"{run.entity_prefix}: BOLD and brain mask are on different grids")
    data = _masked_timeseries(bold, mask_bool)  # (n_vol, n_voxels) float32
    confounds = load_confounds(run)
    if len(confounds) != data.shape[0]:
        raise ValueError(
            f"{run.entity_prefix}: confounds TSV has {len(confounds)} rows, BOLD has "
            f"{data.shape[0]} volumes"
        )
    tr = float(bold.header.get_zooms()[3]) if len(bold.header.get_zooms()) > 3 else float("nan")
    parcs = {
        name: load_parcellation(name, atlases_dir, mask_bool, bold.affine) for name in PARCELLATIONS
    }
    return RunInputs(
        run=run,
        data=data,
        mask_bool=mask_bool,
        affine=bold.affine,
        confounds=confounds,
        tr=tr,
        input_bold_sha256=file_sha256(run.bold),
        input_confounds_sha256=file_sha256(run.confounds),
        parcellations=parcs,
    )


def write_run_regime(
    tree_root: Path,
    inputs: RunInputs,
    regime: Regime,
    *,
    fmriprep_version: str,
    code_sha: str,
) -> dict[str, Any]:
    """Clean one loaded run under one regime and write its tier-1 outputs.

    Returns the run-level record (what the tSNR sidecar holds).
    """
    import nibabel as nib

    run = inputs.run
    design = regime_design(regime, inputs.confounds)
    result = clean(inputs.data, design)

    out = output_dir(tree_root, run)
    out.mkdir(parents=True, exist_ok=True)

    tsnr_vol = np.zeros(inputs.mask_bool.shape, dtype=np.float32)
    tsnr_vol[inputs.mask_bool] = np.nan_to_num(result.tsnr, nan=0.0).astype(np.float32)
    img = nib.Nifti1Image(tsnr_vol, inputs.affine)
    img.set_data_dtype(np.float32)
    nii, js = tsnr_paths(tree_root, run, regime.name)
    img.to_filename(str(nii))

    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "sub": run.subject, "ses": run.session, "task": run.task, "run": run.run,
        "space": run.space, "variant": run.variant,
        "regime": regime.name, "regime_status": regime.status, "regime_version": regime.version,
        "regime_columns": list(design.columns.columns),
        "n_vol": design.n_vol, "n_nss": design.n_nss,
        "n_regressors": design.n_regressors, "n_drift": design.n_drift,
        "dof_resid": design.dof_resid, "dof_loss": round(design.dof_loss, 6),
        "mask_n_voxels": int(inputs.mask_bool.sum()),
        "tsnr_median_mask": float(np.nanmedian(result.tsnr)),
        "var_ratio_median_mask": float(np.nanmedian(result.var_ratio)),
        "repetition_time": inputs.tr,
        "fmriprep_version": fmriprep_version,
        "input_bold": str(run.bold),
        "input_bold_sha256": inputs.input_bold_sha256,
        "input_confounds_sha256": inputs.input_confounds_sha256,
        "code_version": code_sha,
        "created": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    }
    js.write_text(json.dumps(record, indent=2) + "\n")

    for seg, parc in inputs.parcellations.items():
        ts, table = parcel_timeseries(result, parc)
        tsv, sj = timeseries_paths(tree_root, run, regime.name, seg)
        ts.to_csv(tsv, sep="\t", index=False, float_format="%.6g", na_rep="n/a")
        parcels = {
            str(r["name"]): {
                "index": int(r["index"]),
                "n_voxels_atlas": int(r["n_voxels_atlas"]),
                "n_voxels_mask": int(r["n_voxels_mask"]),
                "coverage": None if pd.isna(r["coverage"]) else round(float(r["coverage"]), 6),
                "var_removed": None if pd.isna(r["var_removed"]) else round(float(r["var_removed"]), 6),
            }
            for _, r in table.iterrows()
        }
        sj.write_text(json.dumps({
            "schema_version": SCHEMA_VERSION,
            "SamplingFrequency": (1.0 / inputs.tr) if inputs.tr and np.isfinite(inputs.tr) else None,
            "StartTime": 0.0,
            "Description": (
                f"Mean residual per parcel after the {regime.name!r} regime; non-steady-state "
                "volumes are n/a; a parcel with no in-mask voxel is all n/a."
            ),
            "atlas": seg,
            "regime": regime.name, "regime_version": regime.version,
            "input_bold_sha256": inputs.input_bold_sha256,
            "n_nss": design.n_nss,
            "parcels": parcels,
        }, indent=2) + "\n")
    return record


# ---------------------------------------------------------------------------
# Collect
# ---------------------------------------------------------------------------

def collect(tree_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Flatten every sidecar under the tree into the two tier-1 tables.

    Returns ``(runs, parcels)``; the driver writes them as
    ``tier1_runs.tsv`` and ``tier1_parcels.tsv`` at the tree root. A rebuild
    is a re-walk: nothing here is cached.
    """
    tree_root = Path(tree_root)
    runs, parcels = [], []
    for js in sorted(tree_root.glob("sub-*/ses-*/func/*_tsnr.json")):
        rec = json.loads(js.read_text())
        rec = {k: v for k, v in rec.items() if k != "regime_columns"}
        runs.append(rec)
    for sj in sorted(tree_root.glob("sub-*/ses-*/func/*_timeseries.json")):
        meta = json.loads(sj.read_text())
        key = _entities_from_name(sj.name)
        for name, p in meta.get("parcels", {}).items():
            parcels.append({
                **key, "atlas": meta.get("atlas"), "regime": meta.get("regime"),
                "regime_version": meta.get("regime_version"), "parcel": name, **p,
            })
    return pd.DataFrame(runs), pd.DataFrame(parcels)


def _entities_from_name(name: str) -> dict[str, Optional[str]]:
    out: dict[str, Optional[str]] = {"sub": None, "ses": None, "task": None, "run": None}
    for part in name.split("_"):
        if "-" in part:
            k, v = part.split("-", 1)
            if k in out:
                out[k] = v
    return out
