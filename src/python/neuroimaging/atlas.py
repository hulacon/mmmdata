"""Atlas loading utilities for MMMData.

Loads Schaefer parcellations from the derivatives/atlases/ directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


# Default atlases location (can be overridden)
_DEFAULT_ATLASES_DIR = Path("/gpfs/projects/hulacon/shared/mmmdata/derivatives/atlases")

# Actual filenames on disk follow the pattern:
#   tpl-MNI152NLin2009cAsym_atlas-Schaefer2018_seg-{networks}n_scale-{n_rois}_res-2_dseg.{ext}
# Files live under: atlases/tpl-MNI152NLin2009cAsym/anat/
_SCHAEFER_SUBDIR = "tpl-MNI152NLin2009cAsym/anat"
_SCHAEFER_NIFTI_TEMPLATE = (
    "tpl-MNI152NLin2009cAsym_atlas-Schaefer2018_seg-{networks}n_scale-{n_rois}_res-2_dseg.nii.gz"
)
_SCHAEFER_TSV_TEMPLATE = (
    "tpl-MNI152NLin2009cAsym_atlas-Schaefer2018_seg-{networks}n_scale-{n_rois}_res-2_dseg.tsv"
)


def load_schaefer_atlas(
    n_rois: int = 400,
    networks: int = 7,
    atlases_dir: Optional[str] = None,
) -> tuple:
    """Load a Schaefer parcellation atlas.

    Parameters
    ----------
    n_rois : int, default 400
        Number of parcels (100 to 1000 in steps of 100).
    networks : int, default 7
        Network resolution (7 or 17).
    atlases_dir : str, optional
        Path to atlases directory. Default: derivatives/atlases/.

    Returns
    -------
    tuple of (str, pd.DataFrame)
        (path_to_nifti, labels_dataframe)
    """
    atlas_dir = Path(atlases_dir) if atlases_dir else _DEFAULT_ATLASES_DIR
    anat_dir = atlas_dir / _SCHAEFER_SUBDIR

    nifti_name = _SCHAEFER_NIFTI_TEMPLATE.format(n_rois=n_rois, networks=networks)
    tsv_name = _SCHAEFER_TSV_TEMPLATE.format(n_rois=n_rois, networks=networks)

    nifti_path = anat_dir / nifti_name
    tsv_path = anat_dir / tsv_name

    if not nifti_path.exists():
        raise FileNotFoundError(f"Atlas NIfTI not found: {nifti_path}")

    labels_df = pd.DataFrame()
    if tsv_path.exists():
        labels_df = pd.read_csv(tsv_path, sep="\t")

    return str(nifti_path), labels_df


def get_roi_index(labels_df: pd.DataFrame, roi_name: str) -> int | None:
    """Find the one parcel a name identifies in the labels table.

    ``roi_name`` matches a parcel's full name, or its trailing ``_``-separated
    fields (``LH_Vis_1`` for ``7Networks_LH_Vis_1``), case-insensitively. A
    substring is not enough: ``Vis_1`` must not pick ``Vis_10``.

    Parameters
    ----------
    labels_df : pd.DataFrame
        Labels table from ``load_schaefer_atlas()``.
    roi_name : str
        Full parcel name, or its tail from a ``_`` boundary.

    Returns
    -------
    int or None
        ROI index (1-based), or None if no parcel matches.

    Raises
    ------
    ValueError
        If the name matches more than one parcel (e.g. ``Vis_1`` names one in
        each hemisphere); the message lists them.
    """
    name_col = None
    for col in labels_df.columns:
        if "name" in col.lower() or "label" in col.lower():
            name_col = col
            break
    if name_col is None:
        return None

    names = labels_df[name_col].astype(str).str.lower()
    query = roi_name.lower()
    matches = labels_df[(names == query) | names.str.endswith("_" + query)]
    if matches.empty:
        return None
    if len(matches) > 1:
        raise ValueError(f"ROI {roi_name!r} matches {len(matches)} parcels: "
                         f"{list(matches[name_col])[:6]}; give more of the name")

    # Return the index column value (usually first column)
    idx_col = labels_df.columns[0]
    return int(matches.iloc[0][idx_col])
