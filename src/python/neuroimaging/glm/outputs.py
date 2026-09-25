"""Where contrast maps go and what they are called.

Contract A keys in every filename (``subject, session, task, space``) plus
the two entities that make a statistical map self-describing, ``contrast``
and ``stat``, following the BIDS derivatives convention for ``statmap``
files. The tree gets a ``dataset_description.json`` on first write so the
nightly catalog rebuild indexes it instead of finding an undeclared
directory.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

STATS = ("effect", "variance", "t", "z")


def save_statmap(img: Any, path: Path) -> Path:
    """Write a statistical map as float32 with a fresh header.

    nilearn >= 0.13 builds every output image with ``copy_header=True`` from
    the mask it was given, so a uint8 brain mask hands its ``uint8`` data
    type to the effect, variance, t and z maps; ``to_filename`` then scales
    each map onto 255 levels. In memory the arrays are exact, so anything
    scored from the fitted objects is unaffected, but a map read back from
    disk is not the map that was fitted (t maps written before 2026-09-12
    carry ~230 distinct values). Every stat map goes through here.
    """
    import nibabel as nib
    import numpy as np

    data = np.asarray(img.dataobj, dtype=np.float32)
    out = nib.Nifti1Image(data, img.affine)
    out.set_data_dtype(np.float32)
    path = Path(path)
    out.to_filename(str(path))
    return path


def statmap_name(
    subject: str,
    task: str,
    space: str,
    contrast: str,
    stat: str,
    session: Optional[str] = None,
    run: Optional[str] = None,
    ext: str = ".nii.gz",
    hemi: Optional[str] = None,
    desc: Optional[str] = None,
) -> str:
    """``sub-XX[_ses-YY]_task-T[_run-RR][_hemi-H]_space-S_contrast-C_stat-X[_desc-D]_statmap.nii.gz``.

    Bare labels in, prefixes added here — the same rule the QC tools use.
    A fixed-effects map over runs carries no ``run``; one pooled over sessions
    carries no ``session`` either. A surface map carries ``hemi`` (before
    ``space``, as fMRIPrep orders it) and ``ext=".func.gii"``. ``desc`` names
    the processing variant (:func:`glm_desc`), so several variants share a tree.
    """
    if stat not in STATS:
        raise ValueError(f"stat must be one of {STATS}, got {stat!r}")
    parts = [f"sub-{_bare(subject, 'sub')}"]
    if session:
        parts.append(f"ses-{_bare(session, 'ses')}")
    parts.append(f"task-{task}")
    if run:
        parts.append(f"run-{_bare(run, 'run')}")
    if hemi:
        parts.append(f"hemi-{hemi}")
    parts += [f"space-{space}", f"contrast-{contrast}", f"stat-{stat}"]
    if desc:
        parts.append(f"desc-{desc}")
    parts.append("statmap")
    return "_".join(parts) + ext


def noise_map_name(entity_prefix: str, space: str, param: str) -> str:
    """``<run prefix>_space-S_param-P_noisemap.nii.gz``.

    Not a ``statmap``: this is a fitted noise-model parameter (the per-voxel
    AR(1) coefficient), not an estimate of an effect, and nothing scores it.
    It is written only under ``--keep-per-run`` and only by engines that
    expose one.
    """
    return f"{entity_prefix}_space-{space}_param-{param}_noisemap.nii.gz"


def _bare(label: str, prefix: str) -> str:
    return label[len(prefix) + 1 :] if label.startswith(prefix + "-") else label


ENGINE_LABELS = {"ols": "OLS", "ar1": "AR1"}


def glm_desc(regime: str, noise_model: str, smoothing_fwhm: Optional[float] = None,
             variant: str = "fmriprep") -> str:
    """The ``desc-`` label of a fit: confound regime + engine, plus any departure.

    ``referenceAR1``, ``gsrOLS``; a smoothed or non-default-input fit appends
    ``Fwhm5`` / ``Nordic`` so it can never overwrite a reference-spec map.
    BIDS asks ``desc`` to distinguish versions of processing of the same
    input; every choice the runner exposes that changes the numbers is in it.
    """
    label = regime + ENGINE_LABELS.get(noise_model, noise_model.upper())
    if smoothing_fwhm:
        label += "Fwhm" + f"{smoothing_fwhm:g}".replace(".", "p")
    if variant != "fmriprep":
        label += "".join(w.capitalize() for w in variant.split("_") if w != "fmriprep")
    if not label.isalnum():
        raise ValueError(f"desc label must be alphanumeric, got {label!r}")
    return label


def describe_glm_desc(regime: str, noise_model: str, smoothing_fwhm: Optional[float] = None,
                      variant: str = "fmriprep") -> str:
    """One line for ``descriptions.tsv``."""
    engine = {"ols": "OLS (no serial-correlation model)", "ar1": "AR(1) prewhitening"}.get(noise_model, noise_model)
    smooth = f"{smoothing_fwhm:g} mm FWHM smoothing" if smoothing_fwhm else "unsmoothed"
    return (f"nilearn first-level GLM, {engine}; confound regime '{regime}' from "
            f"neuroimaging/glm/reference_spec.json; SPM canonical HRF; {smooth}; input {variant}")


@contextmanager
def _tree_lock(out_base: Path):
    """Exclusive POSIX lock on the tree's index files; GPFS honours it across nodes."""
    import fcntl

    out_base.mkdir(parents=True, exist_ok=True)
    with open(out_base / ".index.lock", "a") as fh:
        fcntl.lockf(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.lockf(fh, fcntl.LOCK_UN)


def _entities(name: str) -> dict[str, str]:
    stem = name.split(".", 1)[0]
    parts = stem.split("_")
    ents = dict(p.split("-", 1) for p in parts[:-1] if "-" in p)
    ents["suffix"] = parts[-1]
    return ents


MAPS_INDEX_COLUMNS = ("subject", "session", "task", "run", "hemi", "space", "res", "contrast", "stat", "desc", "path")


def update_tree_index(out_base: Path, desc: str, description: str) -> tuple[Path, Path]:
    """Refresh ``descriptions.tsv`` (upsert ``desc``) and ``maps.tsv`` (every statmap in the tree).

    ``maps.tsv`` is rebuilt from a scan, never appended, so it cannot drift
    from the tree; both files are written under one lock and renamed into
    place, so concurrent fits in an array cannot interleave. A map's place in
    the tree follows BIDS (a one-session pool under ``ses-``, a cross-session
    pool at subject level); this table is how to find one without knowing which.
    """
    import csv
    import os

    out_base = Path(out_base)
    desc_path, maps_path = out_base / "descriptions.tsv", out_base / "maps.tsv"
    with _tree_lock(out_base):
        rows = {}
        if desc_path.exists():
            with open(desc_path, newline="") as fh:
                rows = {r["desc_id"]: r["description"] for r in csv.DictReader(fh, delimiter="\t")}
        rows[desc] = description
        tmp = desc_path.with_suffix(".tsv.tmp")
        with open(tmp, "w", newline="") as fh:
            w = csv.writer(fh, delimiter="\t", lineterminator="\n")
            w.writerow(["desc_id", "description"])
            w.writerows(sorted(rows.items()))
        os.replace(tmp, desc_path)

        entries = []
        for p in sorted(out_base.glob("sub-*/**/*_statmap.*")):
            if not (p.name.endswith(".nii.gz") or p.name.endswith(".func.gii")):
                continue
            e = _entities(p.name)
            entries.append([e.get("sub", ""), e.get("ses", ""), e.get("task", ""), e.get("run", ""), e.get("hemi", ""),
                            e.get("space", ""), e.get("res", ""), e.get("contrast", ""), e.get("stat", ""), e.get("desc", ""),
                            str(p.relative_to(out_base))])
        tmp = maps_path.with_suffix(".tsv.tmp")
        with open(tmp, "w", newline="") as fh:
            w = csv.writer(fh, delimiter="\t", lineterminator="\n")
            w.writerow(MAPS_INDEX_COLUMNS)
            w.writerows(entries)
        os.replace(tmp, maps_path)
    return desc_path, maps_path


def output_dir(derivatives_dir: Path, tree: str, subject: str, session: Optional[str] = None) -> Path:
    d = Path(derivatives_dir) / tree / f"sub-{_bare(subject, 'sub')}"
    if session:
        d = d / f"ses-{_bare(session, 'ses')}"
    return d / "func"


def ensure_dataset_description(
    out_base: Path, fmriprep_dir: Path, model_name: str, estimator: str
) -> Path:
    """Write the tree-level description once, the way ``glmsingle_tb.py`` does."""
    dd = Path(out_base) / "dataset_description.json"
    if dd.exists():
        return dd
    try:
        from importlib.metadata import version

        nilearn_version = version("nilearn")
    except Exception:
        nilearn_version = "unknown"
    out_base.mkdir(parents=True, exist_ok=True)
    dd.write_text(
        json.dumps(
            {
                "Name": "nilearn first-level GLM contrast maps",
                "BIDSVersion": "1.8.0",
                "DatasetType": "derivative",
                "GeneratedBy": [
                    {
                        "Name": "nilearn",
                        "Version": nilearn_version,
                        "Description": f"estimator '{estimator}' via neuroimaging.glm",
                    },
                    {
                        "Name": "glm_contrast_maps.py",
                        "Description": "mmmdata/scripts/glm_contrast_maps.py; models in "
                        "mmmdata/models/ (BIDS Stats Models); design record in "
                        "mmmdata-agents docs/workbench/glm-strategy/",
                    },
                ],
                "SourceDatasets": [{"URL": str(fmriprep_dir)}],
                "HowToAcknowledge": f"First fit written by model {model_name}",
            },
            indent=2,
        )
    )
    return dd


def write_run_metadata(path: Path, payload: dict[str, Any]) -> Path:
    """The fit's own record: model, config, runs, estimator, versions."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path
