"""Procrustes pieces and CHA connectivity targets over the grayordinate index.

Both are one label per grayordinate, in the row order of the cleaned tree's
``grayordinates.tsv``, so every route indexes columns the same way. The design
record is mmmdata-agents ``docs/archive/workbench/functional-space/`` (pre-registration
§5; CHA targets DECIDED 2026-09-29).

Pieces (piecewise alignment):

  cortex       Schaefer-400 17-network parcels on fsaverage6; medial wall = none
  subcortex    each HOSPA structure (per hemisphere, as the grayordinate
               ``structure`` column already splits them)
  hippocampus  per hemisphere, thirds along the unfolded long axis (the
               unfold template's x, the longer of its two in-plane axes),
               labelled ``ax1..ax3`` by increasing x

Connectivity targets (CHA): each target's signal is the mean of its members.

  cortex       Voronoi tiles on the fsaverage6 sphere around icosahedral
               centres. FreeSurfer's icosahedra nest, so the ico-k centres are
               the first ``ICO[k]`` fsaverage6 vertices of each hemisphere.
               Centres on the medial wall are dropped; tiles never cross
               hemispheres; medial-wall vertices belong to no target.
  structures   the 12 HOSPA structures and the 2 hippocampi, one target each
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

#: Icosahedral level -> centres per hemisphere (the first N fsaverage6 vertices).
ICO = {"ico3": 642, "ico4": 2562, "ico5": 10242}
CHA_LEVELS = ("ico3", "ico4", "ico5")
SCHAEFER17_LABEL = (
    "tpl-fsaverage/anat/tpl-fsaverage_hemi-{hemi}_den-41k_atlas-Schaefer2018_seg-17n_scale-400_dseg.label.gii"
)
SCHAEFER17_TABLE = "tpl-fsaverage/anat/tpl-fsaverage_den-41k_atlas-Schaefer2018_seg-17n_scale-400_dseg.tsv"
HIPP_THIRDS = 3
NONE = ""


def _hemi_vertices(g: pd.DataFrame, piece: str, hemi: str) -> tuple[np.ndarray, np.ndarray]:
    """Row positions and vertex numbers of one piece's hemisphere, in index order."""
    sel = ((g["piece"] == piece) & (g["hemi"] == hemi)).to_numpy()
    rows = np.flatnonzero(sel)
    return rows, g["vertex"].to_numpy()[rows].astype(int)


def piece_labels(grayordinates: pd.DataFrame, cortex_labels: dict[str, np.ndarray],
                 cortex_names: dict[int, str], hipp_unfold_x: np.ndarray) -> np.ndarray:
    """One Procrustes-piece label per grayordinate (``""`` = no piece).

    ``cortex_labels``: hemi -> per-fsaverage6-vertex parcel index (0 = medial wall).
    ``hipp_unfold_x``: the unfold template's long-axis coordinate per template vertex.
    """
    g = grayordinates
    out = np.full(len(g), NONE, dtype=object)
    for hemi in ("L", "R"):
        rows, vert = _hemi_vertices(g, "cortex", hemi)
        lab = np.asarray(cortex_labels[hemi])[vert]
        out[rows] = [cortex_names[int(k)] if k else NONE for k in lab]
        rows, vert = _hemi_vertices(g, "hippocampus", hemi)
        edges = np.quantile(hipp_unfold_x, np.linspace(0, 1, HIPP_THIRDS + 1)[1:-1])
        third = np.digitize(np.asarray(hipp_unfold_x)[vert], edges) + 1
        out[rows] = [f"HIPPOCAMPUS_{hemi}_ax{t}" for t in third]
    sub = (g["piece"] == "subcortex").to_numpy()
    out[sub] = g.loc[sub, "structure"].to_numpy()
    return out


def cortex_tiles(sphere: dict[str, np.ndarray], medial_wall: dict[str, np.ndarray], n_centres: int
                 ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """hemi -> (centre vertices, per-vertex tile index into those centres; -1 on the medial wall)."""
    from scipy.spatial import cKDTree

    out = {}
    for hemi in ("L", "R"):
        xyz = np.asarray(sphere[hemi])
        wall = np.asarray(medial_wall[hemi], dtype=bool)
        centres = np.flatnonzero(~wall[:n_centres])
        _, tile = cKDTree(xyz[centres]).query(xyz)
        tile = tile.astype(int)
        tile[wall] = -1
        out[hemi] = (centres, tile)
    return out


def target_assignment(grayordinates: pd.DataFrame, tiles: dict[str, tuple[np.ndarray, np.ndarray]]
                      ) -> tuple[np.ndarray, list[str]]:
    """Per grayordinate, the index of its CHA target (-1 = none), and the target names.

    Cortical tiles come first (L then R), then one target per subcortical
    structure and per hippocampus.
    """
    g = grayordinates
    assign = np.full(len(g), -1, dtype=int)
    names: list[str] = []
    for hemi in ("L", "R"):
        centres, tile = tiles[hemi]
        rows, vert = _hemi_vertices(g, "cortex", hemi)
        t = tile[vert]
        assign[rows] = np.where(t >= 0, t + len(names), -1)
        names += [f"cortex_{hemi}_v{c}" for c in centres]
    for piece in ("subcortex", "hippocampus"):
        sel = (g["piece"] == piece).to_numpy()
        for structure in pd.unique(g.loc[sel, "structure"]):
            assign[sel & (g["structure"] == structure).to_numpy()] = len(names)
            names.append(str(structure))
    return assign, names


# ---------------------------------------------------------------------------
# loaders from the staged files
# ---------------------------------------------------------------------------

def load_cortex_parcels(atlases_dir: Path) -> tuple[dict[str, np.ndarray], dict[int, str]]:
    import nibabel as nib

    labels = {h: np.asarray(nib.load(str(Path(atlases_dir) / SCHAEFER17_LABEL.format(hemi=h))).agg_data()).astype(int)
              for h in ("L", "R")}
    table = pd.read_csv(Path(atlases_dir) / SCHAEFER17_TABLE, sep="\t")
    return labels, dict(zip(table["index"].astype(int), table["name"]))


def load_sphere(freesurfer_dir: Path) -> dict[str, np.ndarray]:
    """fsaverage6 sphere coordinates per hemisphere (FreeSurfer ``?h.sphere``)."""
    import nibabel.freesurfer as fs

    return {h: fs.read_geometry(str(Path(freesurfer_dir) / "fsaverage6" / "surf" / f"{h.lower()}h.sphere"))[0]
            for h in ("L", "R")}


def load_hipp_unfold_x(template_path: Path) -> np.ndarray:
    import nibabel as nib

    coords = nib.load(str(template_path)).get_arrays_from_intent("NIFTI_INTENT_POINTSET")[0].data
    span = np.ptp(coords[:, :2], axis=0)
    if not span[0] > span[1]:
        raise ValueError(f"unfold template in-plane spans {span}: x is not the long axis")
    return np.asarray(coords[:, 0], dtype=float)
