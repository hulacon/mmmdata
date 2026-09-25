"""Surface fits: fMRIPrep's per-hemisphere GIfTI BOLD as a nilearn ``SurfaceImage``.

nilearn >= 0.11 fits ``FirstLevelModel`` and ``compute_fixed_effects`` on a
``SurfaceImage`` directly, so the volume estimator and fixed effects run
unchanged (checked on nilearn 0.13.1, 2026-09-25). What a surface space needs
beyond the volume path is only here: finding the two hemisphere files, the
subject's mesh, a vertex mask, and writing ``.func.gii`` outputs.

Only subject-native surfaces are supported. ``fsnative`` pairs with the
subject's own FreeSurfer mesh, which fMRIPrep writes to ``anat/``; a template
surface (fsaverage6) would need a template mesh fetched from the network, which
compute nodes do not have.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

SURFACE_SPACES = ("fsnative",)
HEMIS = {"L": "left", "R": "right"}
#: The mesh the fit is carried on. The fit uses only vertex identity (nothing
#: is smoothed), so the choice matters for display, not for the numbers.
FIT_MESH = "midthickness"


def is_surface_space(space: str) -> bool:
    return space in SURFACE_SPACES


def surface_bold_path(run: Any, hemi: str, space: str) -> Path:
    """``<run prefix>_hemi-<L|R>_space-<space>_bold.func.gii`` beside the run's confounds."""
    return Path(run.confounds).parent / f"{run.entity_prefix}_hemi-{hemi}_space-{space}_bold.func.gii"


def subject_mesh_paths(fmriprep_dir: Path, subject: str, kind: str = FIT_MESH) -> dict[str, Path]:
    """The subject's ``hemi-<L|R>_<kind>.surf.gii`` in fMRIPrep's ``anat/``.

    Exactly one file per hemisphere, whatever other entities (``acq-``) the
    anatomical reference carries; zero or several is an error naming the glob.
    """
    anat = Path(fmriprep_dir) / f"sub-{subject}" / "anat"
    out = {}
    for hemi in HEMIS:
        pattern = f"sub-{subject}_*hemi-{hemi}_{kind}.surf.gii"
        hits = sorted(p for p in anat.glob(pattern) if "_space-" not in p.name and "_desc-" not in p.name)
        if len(hits) != 1:
            raise FileNotFoundError(f"expected one {pattern} under {anat}, found {len(hits)}: {[h.name for h in hits]}")
        out[hemi] = hits[0]
    return out


def load_mesh(paths: dict[str, Path]) -> Any:
    from nilearn.surface import PolyMesh

    return PolyMesh(**{HEMIS[h]: str(p) for h, p in paths.items()})


def load_surface_bold(run: Any, space: str, mesh: Any) -> Any:
    """One run's BOLD as a ``SurfaceImage`` (vertices x time per hemisphere)."""
    import nibabel as nib
    from nilearn.surface import SurfaceImage

    data = {}
    for hemi, part in HEMIS.items():
        g = nib.load(str(surface_bold_path(run, hemi, space)))
        data[part] = np.column_stack([d.data for d in g.darrays]).astype(np.float32)
    return SurfaceImage(mesh=mesh, data=data)


def n_scans_surface(run: Any, space: str) -> int:
    """Time points in a run's left-hemisphere file (the right must match; the fit checks)."""
    import nibabel as nib

    return len(nib.load(str(surface_bold_path(run, "L", space))).darrays)


def surface_mask_intersection(runs: Sequence[Any], space: str, mesh: Any) -> Any:
    """Vertices with finite, non-constant signal in every run, as a boolean ``SurfaceImage``.

    The surface analogue of ``io.mask_intersection``: fMRIPrep writes no
    surface brain mask, and a vertex with no signal variance (medial wall,
    sampling outside the volume) cannot be fitted. Runs are loaded one at a
    time so a 42-run pool never holds more than one run in memory.
    """
    from nilearn.surface import SurfaceImage

    keep: dict[str, np.ndarray] = {}
    for run in runs:
        img = load_surface_bold(run, space, mesh)
        for part, arr in img.data.parts.items():
            a = np.asarray(arr)
            ok = np.isfinite(a).all(axis=1) & (a.std(axis=1) > 0)
            if part in keep and keep[part].shape != ok.shape:
                raise ValueError(f"{run.entity_prefix}: {part} hemisphere has {ok.size} vertices, "
                                 f"expected {keep[part].size}; runs pooled into one map must share a mesh")
            keep[part] = ok if part not in keep else keep[part] & ok
    return SurfaceImage(mesh=mesh, data=keep)


def save_surface_statmap(img: Any, path_for_hemi: dict[str, Path]) -> list[Path]:
    """Write each hemisphere of a ``SurfaceImage`` map as a float32 ``.func.gii``."""
    import nibabel as nib

    written = []
    for hemi, part in HEMIS.items():
        arr = np.asarray(img.data.parts[part], dtype=np.float32).ravel()
        gii = nib.gifti.GiftiImage(darrays=[nib.gifti.GiftiDataArray(arr, intent="NIFTI_INTENT_NONE",
                                                                     datatype="NIFTI_TYPE_FLOAT32")])
        path = Path(path_for_hemi[hemi])
        nib.save(gii, str(path))
        written.append(path)
    return written
