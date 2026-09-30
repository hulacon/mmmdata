"""pRF rows of the data-quality collection: registry T1.11 (adopt, a join).

The pRF fits (``derivatives/prf``, analyzePRF, one pooled fit per subject, two
polarities ``prf`` and ``negprf``) do their own preprocessing, so their rows carry
``regime = prf`` and never enter the (runs x regimes) count. Nothing is refit and
no pRF value is resampled: this module reads the fits' R² maps and summarises
them. Design record: mmmdata-agents ``docs/workbench/data-quality/`` (T1.11
DECIDED 2026-09-29).

**Tier 1, one cell per subject** (``tier1.py prf``):

* **Surface rows**, per hemisphere x polarity, from the ``space-fsnative`` R²
  GIfTIs, restricted to FreeSurfer's ``cortex.label`` (the medial wall is not
  cortex). No resampling.
* **Volume rows**, per polarity, from the ``space-T1w`` R² volume over the fit's
  mask: the intersection of the brain masks of the pooled pRF runs that sit on
  the fit grid (runs resampled onto it before pooling have their own grid and are
  left out of the mask; the fit sidecar names them).
* **Parcel rows**, per polarity x parcel: Schaefer-400 17n and HOSPA, the
  collection's parcellations, warped MNI152NLin2009cAsym -> the fit's T1w grid
  by nearest label (``antsApplyTransforms -n GenericLabel`` with fMRIPrep's
  ``from-MNI152NLin2009cAsym_to-T1w`` transform). The atlas moves; the R² values
  do not. The warped label volumes are written into the cell.

Summaries are over finite values only: analyzePRF leaves some in-mask voxels and
cortical vertices unfit (NaN), and those are counted (``n_nan``), never filled.
Per domain: R² median and p90, and the fraction with R² above each of
:data:`R2_THRESHOLDS` (percent). The headline is ``frac_r2_gt20`` on the ``prf``
polarity, the prf README's contralaterality anchor. ``negprf`` sits beside it; the
two are nested fits, not a partition. No floor is applied to the maps.

Per cell, under ``<tree>/sub-##/anat/``::

    sub-##_task-prf_desc-prf_stat.json                        summaries, inputs, provenance
    sub-##_task-prf_space-T1w_desc-prf_parcels.tsv            per polarity x parcel
    sub-##_space-T1w_seg-<parcellation>_desc-prf_dseg.nii.gz  the warped labels

``collect`` flattens them into ``tier1_prf.tsv`` (subject x polarity x domain)
and ``tier1_prf_parcels.tsv``.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from . import data_quality as dq

SCHEMA_VERSION = "1.0"
REGIME = "prf"
TABLE_NAME = "tier1_prf"
PARCELS_TABLE_NAME = "tier1_prf_parcels"
FIT_TREE = "prf"

POLARITIES: tuple[str, ...] = ("prf", "negprf")
HEMIS: tuple[str, ...] = ("L", "R")

#: R² thresholds (percent, the maps' unit) for the ``frac_r2_gt<t>`` columns.
R2_THRESHOLDS: tuple[int, ...] = (10, 20, 30)

#: The MNI -> subject-T1w transform fMRIPrep writes per subject.
XFM_GLOB = "*from-MNI152NLin2009cAsym_to-T1w_mode-image_xfm.h5"


# ---------------------------------------------------------------------------
# Estimators (pure; tested on synthetic data)
# ---------------------------------------------------------------------------

def r2_summary(r2: np.ndarray) -> dict[str, Any]:
    """Summaries of one domain's R² values (percent). NaN = unfit: counted, left out."""
    r2 = np.asarray(r2, dtype=np.float64).ravel()
    fin = r2[np.isfinite(r2)]
    row: dict[str, Any] = {"n": int(r2.size), "n_nan": int(r2.size - fin.size)}
    if fin.size:
        row["r2_median"] = float(np.median(fin))
        row["r2_p90"] = float(np.quantile(fin, .9))
        for t in R2_THRESHOLDS:
            row[f"frac_r2_gt{t}"] = float(np.mean(fin > t))
    else:
        row["r2_median"] = row["r2_p90"] = float("nan")
        for t in R2_THRESHOLDS:
            row[f"frac_r2_gt{t}"] = float("nan")
    return row


def parcel_rows(r2: np.ndarray, labels: np.ndarray, mask: np.ndarray, table: pd.DataFrame,
                atlas: str, polarity: str) -> list[dict]:
    """Per parcel: R² summaries over the parcel's in-mask voxels.

    ``labels`` is the warped label volume on the fit grid; ``table`` holds the kept
    parcels (``index``, ``name``). ``coverage`` = in-mask / all warped voxels of the
    parcel, as in the other tier-1 parcel tables.
    """
    rows = []
    for rec in table.itertuples(index=False):
        in_parcel = labels == int(rec.index)
        n_atlas = int(in_parcel.sum())
        sel = in_parcel & mask
        row = {"polarity": polarity, "atlas": atlas, "parcel": int(rec.index), "name": str(rec.name).strip(),
               "n_voxels_atlas": n_atlas, "n_voxels_mask": int(sel.sum()),
               "coverage": float(sel.sum() / n_atlas) if n_atlas else float("nan")}
        s = r2_summary(r2[sel])
        s.pop("n")
        row.update(s)
        rows.append(row)
    return rows


def kept_table(name: str, atlases_dir: Path) -> pd.DataFrame:
    """The parcellation's label table with the collection's exclusions applied."""
    spec = dq.PARCELLATIONS[name]
    tsv = Path(atlases_dir) / f"{spec['stem']}.tsv"
    if not tsv.exists():
        raise FileNotFoundError(f"Atlas table missing: {tsv}. Stage it under derivatives/atlases first.")
    table = pd.read_csv(tsv, sep="\t")
    keep = ~table["name"].astype(str).apply(lambda n: any(s in n for s in spec["exclude_substrings"]))
    return table.loc[keep, ["index", "name"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def find_subjects(prf_root: Path, subject: Optional[str] = None) -> list[str]:
    subs = sorted(p.name.removeprefix("sub-") for p in Path(prf_root).glob("sub-*") if p.is_dir())
    return [s for s in subs if subject is None or s == subject]


def fit_files(prf_root: Path, subject: str) -> dict[str, Path]:
    """Every R² map and T1w sidecar the cell reads. Missing = loud."""
    d = Path(prf_root) / f"sub-{subject}"
    out: dict[str, Path] = {}
    for pol in POLARITIES:
        out[f"T1w_{pol}"] = d / f"sub-{subject}_task-prf_space-T1w_desc-R2_{pol}.nii.gz"
        out[f"T1w_{pol}_json"] = d / f"sub-{subject}_task-prf_space-T1w_{pol}.json"
        for h in HEMIS:
            out[f"fsnative_{h}_{pol}"] = d / f"sub-{subject}_task-prf_space-fsnative_hemi-{h}_desc-R2_{pol}.shape.gii"
    missing = [str(p) for p in out.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"pRF fit for sub-{subject} is incomplete; missing: {missing}")
    return out


def subject_inputs(fmriprep_tree: Path, subject: str) -> dict[str, Path]:
    """fMRIPrep's MNI -> T1w transform and FreeSurfer's cortex labels. Missing or ambiguous = loud."""
    # A subject with anatomy from one session has its anat outputs under ses-##/anat, not anat/.
    sub_dir = Path(fmriprep_tree) / f"sub-{subject}"
    xfm = sorted((sub_dir / "anat").glob(f"sub-{subject}_{XFM_GLOB}"))
    if not xfm:
        xfm = sorted(sub_dir.glob(f"ses-*/anat/sub-{subject}_ses-*_{XFM_GLOB}"))
    if len(xfm) != 1:
        raise FileNotFoundError(f"expected one {XFM_GLOB} under {sub_dir}/anat or {sub_dir}/ses-*/anat, "
                                f"found {len(xfm)}: {[str(p) for p in xfm]}")
    label_dir = Path(fmriprep_tree) / "sourcedata" / "freesurfer" / f"sub-{subject}" / "label"
    out = {"xfm": xfm[0]}
    for h, fs in zip(HEMIS, ("lh", "rh")):
        p = label_dir / f"{fs}.cortex.label"
        if not p.exists():
            raise FileNotFoundError(f"FreeSurfer cortex label missing: {p}")
        out[f"cortex_{h}"] = p
    return out


def input_keys(files: dict[str, Path]) -> dict[str, str]:
    return {k: dq.file_sha256(p) for k, p in sorted(files.items())}


def warp_labels(src: Path, ref: Path, xfm: Path, out: Path) -> None:
    """Nearest-label warp of an MNI label volume onto ``ref``'s grid (ANTs GenericLabel)."""
    exe = shutil.which("antsApplyTransforms")
    if exe is None:
        raise RuntimeError("antsApplyTransforms is not on PATH; `module load ants/2.5.2` BEFORE activating the venv")
    subprocess.run([exe, "-d", "3", "-i", str(src), "-r", str(ref), "-t", str(xfm),
                    "-n", "GenericLabel", "-o", str(out)], check=True, capture_output=True)


# ---------------------------------------------------------------------------
# Cells
# ---------------------------------------------------------------------------

def cell_paths(tree_root: Path, subject: str) -> dict[str, Path]:
    d = Path(tree_root) / f"sub-{subject}" / "anat"
    out = {"sidecar": d / f"sub-{subject}_task-prf_desc-{REGIME}_stat.json",
           "parcels": d / f"sub-{subject}_task-prf_space-T1w_desc-{REGIME}_parcels.tsv"}
    for name in dq.PARCELLATIONS:
        out[f"dseg_{name}"] = d / f"sub-{subject}_space-T1w_seg-{name}_desc-{REGIME}_dseg.nii.gz"
    return out


def is_current(tree_root: Path, subject: str, keys: dict, atlases_sha: str) -> bool:
    paths = cell_paths(tree_root, subject)
    if not all(p.exists() for p in paths.values()):
        return False
    side = json.loads(paths["sidecar"].read_text())
    return (side.get("schema_version") == SCHEMA_VERSION and side.get("input_keys") == keys
            and side.get("input_atlases_sha256") == atlases_sha)


def fit_mask(fmriprep_tree: Path, subject: str, sidecar: dict, ref_img: Any) -> tuple[np.ndarray, list[str]]:
    """Intersection of the pooled runs' T1w brain masks that lie on the fit grid."""
    import nibabel as nib

    mask, used = None, []
    for tag in sidecar["Runs"]:
        ses, run = tag.split("_")
        hits = sorted((Path(fmriprep_tree) / f"sub-{subject}" / ses / "func").glob(
            f"sub-{subject}_{ses}_task-prf_{run}_space-T1w_desc-brain_mask.nii.gz"))
        if len(hits) != 1:
            raise FileNotFoundError(f"sub-{subject} {tag}: {len(hits)} T1w brain masks; expected 1")
        img = nib.load(str(hits[0]))
        if img.shape != ref_img.shape or not np.allclose(img.affine, ref_img.affine, atol=1e-3):
            continue
        m = np.asarray(img.dataobj) > 0
        mask = m if mask is None else mask & m
        used.append(tag)
    if mask is None:
        raise ValueError(f"sub-{subject}: no pooled run's brain mask is on the pRF T1w grid")
    return mask, used


def write_cell(tree_root: Path, subject: str, r2_vol: dict[str, np.ndarray], r2_surf: dict[str, np.ndarray],
               cortex: dict[str, np.ndarray], mask: np.ndarray, labels: dict[str, np.ndarray], affine: np.ndarray,
               atlases_dir: Path, provenance: dict) -> dict:
    """Summarise and write one subject's cell. Returns the sidecar.

    ``r2_vol``: polarity -> T1w R² volume; ``r2_surf``: ``<hemi>_<polarity>`` -> per-vertex R²;
    ``cortex``: hemi -> cortex vertex indices; ``labels``: parcellation -> warped label volume.
    """
    import nibabel as nib

    domains: list[dict] = []
    parcels: list[dict] = []
    for pol in POLARITIES:
        for h in HEMIS:
            v = r2_surf[f"{h}_{pol}"]
            idx = cortex[h]
            if idx.size and idx.max() >= v.size:
                raise ValueError(f"hemi-{h} cortex label indexes vertex {idx.max()} of a {v.size}-vertex surface")
            domains.append({"polarity": pol, "domain": f"fsnative_hemi-{h}_cortex", **r2_summary(v[idx])})
        vol = r2_vol[pol]
        if vol.shape != mask.shape:
            raise ValueError(f"{pol} R² volume {vol.shape} is not on the mask grid {mask.shape}")
        domains.append({"polarity": pol, "domain": "T1w_mask", **r2_summary(vol[mask])})
        for name, lab in labels.items():
            parcels.extend(parcel_rows(vol, lab, mask, kept_table(name, atlases_dir), name, pol))

    paths = cell_paths(tree_root, subject)
    paths["sidecar"].parent.mkdir(parents=True, exist_ok=True)
    for name, lab in labels.items():
        img = nib.Nifti1Image(lab.astype(np.int16), affine)
        img.set_qform(affine, code=1)
        img.set_sform(affine, code=1)
        nib.save(img, str(paths[f"dseg_{name}"]))
    pd.DataFrame(parcels).to_csv(paths["parcels"], sep="\t", index=False, na_rep="n/a", float_format="%.6g")
    sidecar = {
        "schema_version": SCHEMA_VERSION, "regime": REGIME, "subject": subject,
        "r2_unit": "percent", "r2_thresholds": list(R2_THRESHOLDS),
        "n_voxels_mask": int(mask.sum()),
        "domains": domains,
        **provenance,
    }
    paths["sidecar"].write_text(json.dumps(sidecar, indent=2, default=float) + "\n")
    return sidecar


def collect(tree_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Flatten every cell into (subject x polarity x domain) and parcel tables."""
    rows, parcels = [], []
    for js in sorted(Path(tree_root).glob(f"sub-*/anat/*_task-prf_desc-{REGIME}_stat.json")):
        side = json.loads(js.read_text())
        for d in side["domains"]:
            rows.append({"sub": side["subject"], "regime": REGIME, **d,
                         "schema_version": side["schema_version"], "code_version": side.get("code_version"),
                         "fmriprep_version": side.get("fmriprep_version")})
        pt = pd.read_csv(cell_paths(tree_root, side["subject"])["parcels"], sep="\t", na_values=["n/a"])
        pt.insert(0, "sub", side["subject"])
        parcels.append(pt)
    return pd.DataFrame(rows), (pd.concat(parcels, ignore_index=True) if parcels else pd.DataFrame())


def build_cell(tree_root: Path, prf_root: Path, fmriprep_tree: Path, atlases_dir: Path, subject: str,
               provenance: dict, force: bool = False, log=print) -> Optional[dict]:
    """Read one subject's fit, warp the parcellations, write the cell. None when current."""
    import nibabel as nib

    files = {**fit_files(prf_root, subject), **subject_inputs(fmriprep_tree, subject)}
    keys = input_keys(files)
    atlases_sha = dq.atlases_sha256(atlases_dir)
    if not force and is_current(tree_root, subject, keys, atlases_sha):
        log(f"sub-{subject}: cell current, skipping")
        return None
    ref = nib.load(str(files["T1w_prf"]))
    for pol in POLARITIES[1:]:
        other = nib.load(str(files[f"T1w_{pol}"]))
        if other.shape != ref.shape or not np.allclose(other.affine, ref.affine, atol=1e-3):
            raise ValueError(f"sub-{subject}: {pol} T1w map is not on the prf map's grid")
    fit_side = json.loads(files["T1w_prf_json"].read_text())
    mask, mask_runs = fit_mask(fmriprep_tree, subject, fit_side, ref)
    r2_vol = {pol: np.asarray(nib.load(str(files[f"T1w_{pol}"])).dataobj, dtype=np.float64) for pol in POLARITIES}
    r2_surf = {f"{h}_{pol}": np.asarray(nib.load(str(files[f"fsnative_{h}_{pol}"])).darrays[0].data, dtype=np.float64)
               for pol in POLARITIES for h in HEMIS}
    cortex = {h: np.asarray(nib.freesurfer.read_label(str(files[f"cortex_{h}"])), dtype=np.int64) for h in HEMIS}
    labels = {}
    with tempfile.TemporaryDirectory() as tmp:
        for name, spec in dq.PARCELLATIONS.items():
            out = Path(tmp) / f"{name}.nii.gz"
            warp_labels(Path(atlases_dir) / f"{spec['stem']}.nii.gz", files["T1w_prf"], files["xfm"], out)
            labels[name] = np.asarray(nib.load(str(out)).dataobj).round().astype(np.int32)
    prov = {**provenance, "input_keys": keys, "input_atlases_sha256": atlases_sha,
            "inputs": {k: str(p) for k, p in files.items()}, "mask_runs": mask_runs,
            "warp": "antsApplyTransforms -n GenericLabel, MNI152NLin2009cAsym res-2 dseg -> pRF T1w grid"}
    side = write_cell(tree_root, subject, r2_vol, r2_surf, cortex, mask, labels, ref.affine, atlases_dir, prov)
    log(f"sub-{subject}: cell written ({side['n_voxels_mask']} mask voxels, runs {mask_runs})")
    return side
