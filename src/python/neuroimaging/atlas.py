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
        Number of parcels (100, 200, 300, 400, 500, 800, 1000).
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
    """Find ROI index by partial name match in the labels table.

    Parameters
    ----------
    labels_df : pd.DataFrame
        Labels table from ``load_schaefer_atlas()``.
    roi_name : str
        Partial ROI name to search for (case-insensitive).

    Returns
    -------
    int or None
        ROI index (1-based), or None if not found.
    """
    name_col = None
    for col in labels_df.columns:
        if "name" in col.lower() or "label" in col.lower():
            name_col = col
            break
    if name_col is None:
        return None

    matches = labels_df[
        labels_df[name_col].str.contains(roi_name, case=False, na=False)
    ]
    if matches.empty:
        return None

    # Return the index column value (usually first column)
    idx_col = labels_df.columns[0]
    return int(matches.iloc[0][idx_col])
