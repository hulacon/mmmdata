"""roi(): regions of interest under one naming scheme, ``<atlas>:<region>``.

Names:

``schaefer<parcels>-<networks>:<parcel>``
    A Schaefer 2018 parcel, e.g. ``schaefer400-7:LH_Vis_1``. The parcel is
    its full label or a tail of it (``neuroimaging.atlas.get_roi_index``).
    Template atlas, so only in ``MNI152NLin2009cAsym_res-2``.
``hipp:<LH|RH>_<subfield>``
    A hippocampal subfield from the segmentation named in config
    (``[mmmget.hippocampal_subfields]``), e.g. ``hipp:LH_CA1``. With no
    such table, raises :class:`~mmmget.errors.SourceNotConfigured`.
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path
from typing import Any, Optional

from core import catalog
from neuroimaging.atlas import get_roi_index, load_schaefer_atlas
from neuroimaging.constants import DEFAULT_SPACE

from . import _config
from .provenance import record
from .catalog_runs import split_space
from .errors import SourceNotConfigured

SCHAEFER_SPACE = "MNI152NLin2009cAsym_res-2"
_SCHAEFER = re.compile(r"^schaefer(\d+)-(7|17)$")
_HIPP_REGION = re.compile(r"^(LH|RH)_(.+)$")
SUBFIELD_KEYS = ("dataset", "desc", "space", "labels")


@dataclasses.dataclass(frozen=True)
class ROI:
    """One region: the label image it lives in and its value there."""

    name: str
    subject: Optional[str]
    space: str
    path: Path
    index: int
    source: str

    def mask(self):
        """The region as a boolean nibabel image on the label image's grid."""
        import nibabel as nib
        import numpy as np

        img = nib.load(str(self.path))
        data = np.asarray(img.dataobj) == self.index
        return nib.Nifti1Image(data.astype(np.uint8), img.affine, img.header)


def _schaefer(subject, name, n_rois, networks, region, space, cfg) -> ROI:
    if space != SCHAEFER_SPACE:
        raise SourceNotConfigured(
            f"{name}: Schaefer parcels exist only in {SCHAEFER_SPACE} (a "
            f"template atlas); no projection into {space!r} is built")
    atlases = Path(_config.mmmget_section(cfg).get(
        "atlases_dir", Path(cfg["paths"]["output_dir"]) / "atlases"))
    path, labels = load_schaefer_atlas(n_rois=n_rois, networks=networks,
                                       atlases_dir=str(atlases))
    if labels.empty:
        raise SourceNotConfigured(
            f"{name}: the Schaefer labels TSV beside {path} is missing, so "
            "parcel names cannot be resolved")
    index = get_roi_index(labels, region)
    if index is None:
        raise LookupError(f"{name}: no parcel named {region!r} in {Path(path).name}")
    return ROI(name=name, subject=subject, space=space, path=Path(path),
               index=index, source=f"Schaefer2018 {n_rois} parcels, {networks} networks")


def _hipp(subject, name, region, space, cfg) -> ROI:
    src: dict[str, Any] = _config.mmmget_section(cfg).get("hippocampal_subfields") or {}
    missing = [k for k in SUBFIELD_KEYS if k not in src]
    if missing:
        raise SourceNotConfigured(
            f"{name}: no hippocampal subfield source is configured. Add a "
            "[mmmget.hippocampal_subfields] table to config (local.toml or "
            f"base.toml) with {', '.join(SUBFIELD_KEYS)}: the catalog dataset "
            "holding the chosen segmentation, its desc- label, its space, and "
            f"the label value of each subfield (missing: {', '.join(missing)})")
    m = _HIPP_REGION.match(region)
    if not m:
        raise ValueError(f"{name}: a subfield region is <LH|RH>_<subfield>, e.g. hipp:LH_CA1")
    hemi_label, subfield = m.groups()
    labels: dict[str, int] = src["labels"]
    if subfield not in labels:
        raise LookupError(f"{name}: subfield {subfield!r} is not one of {sorted(labels)}")
    if space != src["space"]:
        raise SourceNotConfigured(
            f"{name}: the configured subfield segmentation is in {src['space']!r}, "
            f"not {space!r}; no resampling is built")
    if subject is None:
        raise ValueError(f"{name}: subfields are per subject; pass subject=")
    sub = catalog.bare_label(subject, "sub-")
    cat_space, cat_res = split_space(space)
    root = _config.bids_root()
    db = catalog.catalog_path(root)
    _, rows = catalog.run_select(
        db,
        "SELECT dataset_relpath, path FROM files WHERE dataset_relpath = ? "
        "AND sub = ? AND \"desc\" = ? AND suffix = 'dseg' AND hemi = ? "
        "AND ext = '.nii.gz' AND space IS NOT DISTINCT FROM ? "
        "AND res IS NOT DISTINCT FROM ?",
        [src["dataset"], sub, src["desc"], hemi_label[0], cat_space, cat_res],
    )
    if len(rows) != 1:
        found = ", ".join(Path(r["path"]).name for r in rows) or "none"
        raise SourceNotConfigured(
            f"{name}: expected one desc-{src['desc']} hemi-{hemi_label[0]} dseg "
            f"for sub-{sub} in {src['dataset']} (catalog {db}); found {found}")
    path = root / rows[0]["dataset_relpath"] / rows[0]["path"]
    return ROI(name=name, subject=sub, space=space, path=path,
               index=int(labels[subfield]),
               source=f"{src['dataset']} desc-{src['desc']}")


def roi(subject: Optional[str], name: str, space: str = DEFAULT_SPACE) -> ROI:
    """One region of interest by its ``<atlas>:<region>`` name.

    Parameters
    ----------
    subject : str or None
        ``"##"`` or ``"sub-##"``. Template atlases ignore it; per-subject
        sources (subfields) require it.
    name : str
        ``schaefer<parcels>-<networks>:<parcel>`` or
        ``hipp:<LH|RH>_<subfield>`` (see the module docstring).
    space : str
        Space label, as for :func:`mmmget.runs`.

    Raises
    ------
    SourceNotConfigured
        The region has no source in *space*, or none is configured.
    """
    atlas, sep, region = name.partition(":")
    if not sep or not region:
        raise ValueError(f"ROI name {name!r} is not <atlas>:<region>")
    cfg = _config.config()
    if m := _SCHAEFER.match(atlas):
        out = _schaefer(subject, name, int(m.group(1)), int(m.group(2)), region, space, cfg)
    elif atlas == "hipp":
        out = _hipp(subject, name, region, space, cfg)
    else:
        raise ValueError(
            f"ROI atlas {atlas!r} is not known; names are "
            "schaefer<parcels>-<networks>:<parcel> or hipp:<LH|RH>_<subfield>")
    record("roi", standard=True, name=name, space=space, source=out.source)
    return out
