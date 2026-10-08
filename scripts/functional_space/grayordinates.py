"""Grayordinate geometry, cleaning helpers and the loader for functional-space time series.

The functional-space template is a cortical-coordinate object built from three
pieces of the same fMRIPrep runs (mmmdata-agents
``docs/archive/workbench/functional-space/out/preregistration.md`` §5):

* **cortex** -- fMRIPrep ``space-fsaverage6`` surface BOLD, all 40,962 vertices
  per hemisphere. The medial wall (label 0 of the CBIG Schaefer-400 fsaverage6
  annotation staged in ``derivatives/atlases/tpl-fsaverage``) is NaN, so vertex
  ``i`` of the file is fsaverage6 vertex ``i`` and every atlas ``.label.gii``
  indexes it directly.
* **subcortex** -- ``MNI152NLin2009cAsym_res-2`` voxels of six Harvard-Oxford
  subcortical structures per hemisphere (``HOSPA`` th25: thalamus, caudate,
  putamen, pallidum, accumbens, amygdala). The voxel set is the atlas's, fixed
  across runs; voxels outside a run's brain mask are NaN in that run.
* **hippocampus** -- hippunfold's unfolded hippocampus at its standard 2 mm
  density (419 vertices per hemisphere), whose vertices correspond across
  subjects. Each run's ``space-T1w`` BOLD is sampled onto the subject's 0.5 mm
  hippocampal ribbon (``wb_command -volume-to-surface-mapping
  -ribbon-constrained``) and averaged, surface-area weighted, onto the 2 mm
  vertices.

hippunfold's ``space-T1w`` is NOT fMRIPrep's ``space-T1w``: hippunfold took the
ses-01 run-01 MPRAGE as its reference, fMRIPrep an unbiased template of the
ses-01 MPRAGEs. The surfaces are mapped into fMRIPrep T1w with the inverse of
fMRIPrep's ``from-orig_to-T1w`` transform for the run hippunfold used
(:func:`hippunfold_to_fmriprep_t1w`), checked against fMRIPrep's aseg.

Written per run (``<tree>/sub-XX/ses-YY/func/``)::

    <prefix>_space-fsaverage6_desc-<regime>_bold.dtseries.nii   CIFTI-2, float32
    <prefix>_space-fsaverage6_desc-<regime>_bold.json            provenance

Full-run residuals: nothing is trimmed. Non-steady-state volumes are NaN rows;
film title/fixation trimming and lag buffers belong to the route loaders.
Masked grayordinates are NaN columns -- never zero-filled.
"""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

#: Sub-tree of the derivatives directory this writer owns.
TREE_NAME = Path("functional_space") / "cleaned_timeseries"
SCHEMA_VERSION = "1.0"

FSAVERAGE6_N = 40962
HEMIS = ("L", "R")

#: HOSPA row name -> CIFTI structure name (nibabel spelling).
SUBCORTICAL = {
    "Left Thalamus": "CIFTI_STRUCTURE_THALAMUS_LEFT",
    "Left Caudate": "CIFTI_STRUCTURE_CAUDATE_LEFT",
    "Left Putamen": "CIFTI_STRUCTURE_PUTAMEN_LEFT",
    "Left Pallidum": "CIFTI_STRUCTURE_PALLIDUM_LEFT",
    "Left Accumbens": "CIFTI_STRUCTURE_ACCUMBENS_LEFT",
    "Left Amygdala": "CIFTI_STRUCTURE_AMYGDALA_LEFT",
    "Right Thalamus": "CIFTI_STRUCTURE_THALAMUS_RIGHT",
    "Right Caudate": "CIFTI_STRUCTURE_CAUDATE_RIGHT",
    "Right Putamen": "CIFTI_STRUCTURE_PUTAMEN_RIGHT",
    "Right Pallidum": "CIFTI_STRUCTURE_PALLIDUM_RIGHT",
    "Right Accumbens": "CIFTI_STRUCTURE_ACCUMBENS_RIGHT",
    "Right Amygdala": "CIFTI_STRUCTURE_AMYGDALA_RIGHT",
}
CORTEX = {"L": "CIFTI_STRUCTURE_CORTEX_LEFT", "R": "CIFTI_STRUCTURE_CORTEX_RIGHT"}
HIPPOCAMPUS = {"L": "CIFTI_STRUCTURE_HIPPOCAMPUS_LEFT", "R": "CIFTI_STRUCTURE_HIPPOCAMPUS_RIGHT"}

HOSPA_STEM = "tpl-MNI152NLin2009cAsym/anat/tpl-MNI152NLin2009cAsym_atlas-HOSPA_res-2_desc-th25_dseg"
MEDIAL_WALL_LABEL = (
    "tpl-fsaverage/anat/tpl-fsaverage_hemi-{hemi}_den-41k_atlas-Schaefer2018_seg-7n_scale-400_dseg.label.gii"
)

#: hippunfold's packaged unfold-space templates (inside its container).
HIPPUNFOLD_CONTAINER = "hippunfold-v1.5.2.sif"
HIPPUNFOLD_TEMPLATE_DIR = "/opt/conda/lib/python3.9/site-packages/hippunfold/resources/unfold_template_hipp"
HIPP_DENSITY = "2mm"
HIPP_SOURCE_DENSITY = "0p5mm"
#: The shared env that supplies wb_command (conda-forge connectome-workbench),
#: a sibling of the stimfeat env under the shared envs directory. Override
#: with $WB_COMMAND.
WB_ENV = "functional-space"
#: FreeSurfer aseg labels for the left/right hippocampus (placement check only).
ASEG_HIPPOCAMPUS = {"L": 17, "R": 53}
#: A mapped hippocampal midthickness must put at least this fraction of its
#: vertices inside fMRIPrep's aseg hippocampus, or the geometry step refuses.
MIN_ASEG_OVERLAP = 0.80

_LPS = np.diag([-1.0, -1.0, 1.0])


# ---------------------------------------------------------------------------
# Tree layout
# ---------------------------------------------------------------------------

def tree_root(output_dir: Path) -> Path:
    return Path(output_dir) / TREE_NAME


def grayordinates_path(root: Path) -> Path:
    return Path(root) / "grayordinates.tsv"


def hipp_template_path(root: Path) -> Path:
    return Path(root) / f"tpl-hippunfold_space-unfold_den-{HIPP_DENSITY}_label-hipp_midthickness.surf.gii"


def anat_dir(root: Path, subject: str) -> Path:
    return Path(root) / f"sub-{subject}" / "anat"


def hipp_surface_path(root: Path, subject: str, hemi: str, surf: str) -> Path:
    return anat_dir(root, subject) / (
        f"sub-{subject}_hemi-{hemi}_space-T1w_den-{HIPP_SOURCE_DENSITY}_label-hipp_{surf}.surf.gii"
    )


def hipp_assignment_path(root: Path, subject: str, hemi: str) -> Path:
    return anat_dir(root, subject) / (
        f"sub-{subject}_hemi-{hemi}_den-{HIPP_SOURCE_DENSITY}_label-hipp_to-{HIPP_DENSITY}_weights.tsv"
    )


def run_paths(root: Path, run: Any, regime: str) -> tuple[Path, Path]:
    stem = (
        Path(root) / f"sub-{run.subject}" / f"ses-{run.session}" / "func"
        / f"{run.entity_prefix}_space-fsaverage6_desc-{regime}_bold"
    )
    return stem.with_name(stem.name + ".dtseries.nii"), stem.with_name(stem.name + ".json")


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def _surf_arrays(path: Path) -> tuple[np.ndarray, np.ndarray]:
    import nibabel as nib

    g = nib.load(str(path))
    coords = g.get_arrays_from_intent("NIFTI_INTENT_POINTSET")[0].data
    faces = g.get_arrays_from_intent("NIFTI_INTENT_TRIANGLE")[0].data
    return np.asarray(coords, dtype=np.float64), np.asarray(faces)


def medial_wall(atlases_dir: Path, hemi: str) -> np.ndarray:
    """Boolean per fsaverage6 vertex: True on the medial wall (Schaefer label 0)."""
    import nibabel as nib

    lab = np.asarray(nib.load(str(Path(atlases_dir) / MEDIAL_WALL_LABEL.format(hemi=hemi))).agg_data())
    if lab.shape != (FSAVERAGE6_N,):
        raise ValueError(f"medial-wall label for hemi-{hemi} has shape {lab.shape}, not ({FSAVERAGE6_N},)")
    return lab == 0


def subcortical_masks(atlases_dir: Path) -> tuple[dict[str, np.ndarray], np.ndarray, tuple[int, ...]]:
    """{CIFTI structure: 3-D bool mask} for the HOSPA structures, plus affine and shape."""
    import nibabel as nib

    img = nib.load(str(Path(atlases_dir) / f"{HOSPA_STEM}.nii.gz"))
    lab = np.asarray(img.dataobj).astype(int)
    table = pd.read_csv(Path(atlases_dir) / f"{HOSPA_STEM}.tsv", sep="\t")
    masks = {}
    for name, structure in SUBCORTICAL.items():
        rows = table.loc[table["name"] == name, "index"]
        if len(rows) != 1:
            raise KeyError(f"HOSPA table has {len(rows)} rows named {name!r}")
        masks[structure] = lab == int(rows.iloc[0])
        if not masks[structure].any():
            raise ValueError(f"HOSPA label {name!r} is empty")
    return masks, img.affine, lab.shape


def brain_model_axis(atlases_dir: Path, n_hipp: int) -> Any:
    """The CIFTI brain-model axis every run shares: cortex, hippocampus, subcortex."""
    from nibabel import cifti2

    parts = [cifti2.BrainModelAxis.from_surface(np.arange(FSAVERAGE6_N), FSAVERAGE6_N, CORTEX[h]) for h in HEMIS]
    parts += [cifti2.BrainModelAxis.from_surface(np.arange(n_hipp), n_hipp, HIPPOCAMPUS[h]) for h in HEMIS]
    masks, affine, _ = subcortical_masks(atlases_dir)
    parts += [cifti2.BrainModelAxis.from_mask(m, name=s, affine=affine) for s, m in masks.items()]
    axis = parts[0]
    for p in parts[1:]:
        axis = axis + p
    return axis


def grayordinate_table(axis: Any) -> pd.DataFrame:
    """One row per grayordinate: piece, structure, hemisphere, vertex or voxel ijk."""
    rows = []
    for structure, slc, bm in axis.iter_structures():
        idx = np.arange(slc.start or 0, slc.stop if slc.stop is not None else len(axis))
        short = structure.replace("CIFTI_STRUCTURE_", "")
        hemi = "L" if short.endswith("_LEFT") else "R" if short.endswith("_RIGHT") else "n/a"
        piece = "cortex" if short.startswith("CORTEX") else "hippocampus" if short.startswith("HIPPOCAMPUS") else "subcortex"
        if bm.volume_shape is None:
            df = pd.DataFrame({"vertex": bm.vertex, "i": -1, "j": -1, "k": -1})
        else:
            v = bm.voxel
            df = pd.DataFrame({"vertex": -1, "i": v[:, 0], "j": v[:, 1], "k": v[:, 2]})
        df.insert(0, "grayordinate", idx)
        df.insert(1, "piece", piece)
        df.insert(2, "structure", short)
        df.insert(3, "hemi", hemi)
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def hippunfold_to_fmriprep_t1w(coords_ras: np.ndarray, xfm_path: Path) -> np.ndarray:
    """Map points from hippunfold's space-T1w into fMRIPrep's space-T1w.

    ``xfm_path`` is fMRIPrep's ``from-orig_to-T1w_mode-image_xfm.txt`` for the
    MPRAGE hippunfold used. As an ITK image transform it maps a point of the
    fixed image (fMRIPrep T1w) to the moving one (the original MPRAGE):
    ``x_orig = A (x_T1w - c) + c + t`` in LPS. The surfaces live in the original
    MPRAGE, so they need the inverse.
    """
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src" / "python"))
    from neuroimaging.io import read_itk_affine

    A, t, c = read_itk_affine(xfm_path)
    x_orig = coords_ras @ _LPS.T
    x_t1w = (np.linalg.solve(A, (x_orig - c - t).T)).T + c
    return x_t1w @ _LPS.T


def aseg_overlap(coords_ras: np.ndarray, aseg_path: Path, label: int) -> float:
    """Fraction of points whose nearest aseg voxel carries ``label``."""
    import nibabel as nib

    img = nib.load(str(aseg_path))
    lab = np.asarray(img.dataobj)
    ijk = np.rint(nib.affines.apply_affine(np.linalg.inv(img.affine), coords_ras)).astype(int)
    inside = np.all((ijk >= 0) & (ijk < np.array(lab.shape)), axis=1)
    hit = np.zeros(len(ijk), bool)
    hit[inside] = lab[ijk[inside, 0], ijk[inside, 1], ijk[inside, 2]] == label
    return float(hit.mean())


def write_surface(coords: np.ndarray, faces: np.ndarray, path: Path) -> None:
    import nibabel as nib

    g = nib.gifti.GiftiImage()
    g.add_gifti_data_array(nib.gifti.GiftiDataArray(coords.astype(np.float32), intent="NIFTI_INTENT_POINTSET"))
    g.add_gifti_data_array(nib.gifti.GiftiDataArray(faces.astype(np.int32), intent="NIFTI_INTENT_TRIANGLE"))
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(g, str(path))


def hipp_weights(sub_unfold: np.ndarray, tpl_unfold: np.ndarray, area: np.ndarray) -> pd.DataFrame:
    """Assign each 0.5 mm vertex to its nearest 2 mm template vertex in unfold space.

    The subject's unfold coordinates (after hippunfold's unfold-space
    registration) place its vertices; the template's 2 mm vertices are the
    shared coordinate. Weights are the 0.5 mm vertices' surface areas, so a
    2 mm value is the area-weighted mean of the ribbon samples nearest it.
    """
    from scipy.spatial import cKDTree

    _, nearest = cKDTree(tpl_unfold[:, :2]).query(sub_unfold[:, :2])
    df = pd.DataFrame({
        "vertex_source": np.arange(len(sub_unfold)),
        "vertex_target": nearest.astype(int),
        "area": area.astype(float),
    })
    empty = sorted(set(range(len(tpl_unfold))) - set(df["vertex_target"]))
    if empty:
        raise ValueError(f"{len(empty)} template vertices received no source vertex (first: {empty[:5]})")
    return df


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def wb_command_path(envs_dir: Path) -> Path:
    """``$WB_COMMAND``, else ``<envs_dir>/functional-space/bin/wb_command``.

    Called by absolute path so activating nothing disturbs the caller's interpreter.
    """
    path = Path(os.environ.get("WB_COMMAND") or Path(envs_dir) / WB_ENV / "bin" / "wb_command")
    if not path.exists():
        raise FileNotFoundError(f"wb_command not found at {path}; build the {WB_ENV} env or set $WB_COMMAND")
    return path


def wb_version(wb: Path) -> str:
    out = subprocess.run([str(wb), "-version"], check=True, capture_output=True, text=True).stdout
    for line in out.splitlines():
        if line.strip().startswith("Version:"):
            return line.split(":", 1)[1].strip()
    return "unknown"


def sample_hippocampus(
    t1w_bold: Path, root: Path, subject: str, wb: Path, scratch: Optional[Path] = None,
) -> dict[str, np.ndarray]:
    """{hemi: (n_vol, n_2mm) float32} ribbon samples averaged onto the 2 mm vertices.

    A 0.5 mm vertex whose sampled series is constant (outside the BOLD field
    of view or brain) is left out of its 2 mm average; a 2 mm vertex with no
    varying source vertex is NaN.
    """
    import nibabel as nib

    out: dict[str, np.ndarray] = {}
    tmp = Path(tempfile.mkdtemp(prefix="fs_hipp_", dir=scratch))
    try:
        for hemi in HEMIS:
            metric = tmp / f"hemi-{hemi}.func.gii"
            subprocess.run(
                [
                    str(wb), "-volume-to-surface-mapping", str(t1w_bold),
                    str(hipp_surface_path(root, subject, hemi, "midthickness")), str(metric),
                    "-ribbon-constrained",
                    str(hipp_surface_path(root, subject, hemi, "inner")),
                    str(hipp_surface_path(root, subject, hemi, "outer")),
                ],
                check=True, capture_output=True, text=True,
            )
            w = pd.read_csv(hipp_assignment_path(root, subject, hemi), sep="\t")
            data = np.asarray(nib.load(str(metric)).agg_data(), dtype=np.float64)
            if data.ndim == 1:
                data = data[:, None]
            if data.shape[0] != len(w):  # agg_data stacks columns last; guard the orientation
                data = data.T
            if data.shape[0] != len(w):
                raise ValueError(f"hemi-{hemi}: sampled {data.shape} does not match {len(w)} source vertices")
            varying = np.ptp(data, axis=1) > 0
            n_tgt = int(w["vertex_target"].max()) + 1
            tgt = w["vertex_target"].to_numpy()
            area = np.where(varying, w["area"].to_numpy(), 0.0)
            num = np.zeros((n_tgt, data.shape[1]))
            np.add.at(num, tgt, data * area[:, None])
            den = np.bincount(tgt, weights=area, minlength=n_tgt)
            with np.errstate(invalid="ignore", divide="ignore"):
                avg = num / den[:, None]
            avg[den == 0] = np.nan
            out[hemi] = avg.T.astype(np.float32)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return out


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class CleanedRun:
    """One run's cleaned grayordinate series with its index and provenance."""

    data: np.ndarray  # (n_vol, n_grayordinates) float32; NaN rows = non-steady-state, NaN columns = masked
    grayordinates: pd.DataFrame
    sidecar: dict
    path: Path

    def piece(self, name: str) -> tuple[np.ndarray, pd.DataFrame]:
        """Columns and index rows of one piece ('cortex', 'subcortex', 'hippocampus')."""
        sel = self.grayordinates["piece"].to_numpy() == name
        if not sel.any():
            raise KeyError(f"no grayordinates in piece {name!r}")
        return self.data[:, sel], self.grayordinates.loc[sel].reset_index(drop=True)


def load_manifest(root: Path) -> pd.DataFrame:
    """The tree's run manifest. Missing is a loud error, never an empty table."""
    path = Path(root) / "manifest.tsv"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run `clean_timeseries.py collect` after the array finishes")
    return pd.read_csv(path, sep="\t", dtype={"sub": str, "ses": str, "run": str})


def load_run(path: Path, root: Optional[Path] = None) -> CleanedRun:
    """Load one ``*_bold.dtseries.nii`` written by ``clean_timeseries.py``."""
    import nibabel as nib

    path = Path(path)
    root = Path(root) if root is not None else path.parents[3]
    table = pd.read_csv(grayordinates_path(root), sep="\t")
    img = nib.load(str(path))
    data = np.asarray(img.dataobj, dtype=np.float32)
    if data.shape[1] != len(table):
        raise ValueError(f"{path.name}: {data.shape[1]} grayordinates, index has {len(table)}")
    side = json.loads(path.with_name(path.name.replace(".dtseries.nii", ".json")).read_text())
    return CleanedRun(data=data, grayordinates=table, sidecar=side, path=path)


def write_dtseries(data: np.ndarray, axis: Any, tr: float, path: Path) -> None:
    from nibabel import cifti2

    series = cifti2.SeriesAxis(start=0.0, step=float(tr), size=data.shape[0], unit="SECOND")
    img = cifti2.Cifti2Image(np.asarray(data, dtype=np.float32), header=(series, axis))
    img.nifti_header.set_data_dtype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial.dtseries.nii")
    img.to_filename(str(tmp))
    os.replace(tmp, path)
