#!/usr/bin/env python3
"""
localizer_splithalf.py — per-run and split-half localizer contrast maps on
fsaverage6, for the within-subject ceiling of a localizer-map metric.

Fits one subject's localizer runs (a BIDS Stats Model from mmmdata/models/)
on fMRIPrep's ``space-fsaverage6`` surface BOLD with the frozen reference
specification (neuroimaging/glm/reference_spec.json: the effect engine, SPM
canonical, the ``reference`` confound regime, unsmoothed), exactly as
``scripts/glm_contrast_maps.py`` does for fsnative. Then it pools runs by
precision-weighted fixed effects three ways:

* every run (the full-data map);
* every complementary split of the runs into two equal halves (for 6 runs,
  the 10 unique 3-vs-3 pairs; for 2 runs, the single 1-vs-1 pair).

Outputs (under ``<derivatives>/<output-tree>/``, default
``functional_space/localizer_splithalf``):

* per run: ``..._run-RR_hemi-H_space-fsaverage6_contrast-C_stat-{effect,variance}_desc-<D>_statmap.func.gii``
* full data: ``sub-XX_task-T_hemi-H_space-fsaverage6_contrast-C_stat-{effect,variance,t}_desc-<D>_statmap.func.gii``
* split halves: the same name with ``desc-<D>Splithalf``, one GIfTI data array
  per (split, half) in the order the sidecar JSON lists
* ``..._desc-<D>_mask.func.gii``: vertices with signal in every run

Vertices outside the mask are NaN in every map (never 0).

Why this is not ``glm_contrast_maps.py --space fsaverage6``: that runner's
surface path supports subject-native meshes only (neuroimaging.glm.surface),
because a template mesh would be fetched from the network. The fit uses
vertex identity only (nothing is smoothed), so this script reads the
fsaverage6 FreeSurfer subject that fMRIPrep itself resampled onto
(``<fmriprep>/sourcedata/freesurfer/fsaverage6/surf/?h.white``) purely to
build the nilearn surface object.

Usage:
    python localizer_splithalf.py --subject sub-## --model floc --dry-run
    python localizer_splithalf.py --subject sub-## --model motor
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src" / "python") not in sys.path:
    sys.path.insert(0, str(_REPO / "src" / "python"))

from neuroimaging.constants import DERIVATIVES_DIRS  # noqa: E402
from neuroimaging.glm.adapters import adapt_events  # noqa: E402
from neuroimaging.glm.config import repetition_time  # noqa: E402
from neuroimaging.glm.design import available_contrast_vectors, build_design_matrix, strict_for  # noqa: E402
from neuroimaging.glm.estimators import get_estimator  # noqa: E402
from neuroimaging.glm.models import load_model  # noqa: E402
from neuroimaging.glm.outputs import glm_desc, statmap_name  # noqa: E402
from neuroimaging.glm.reference import reference_config  # noqa: E402
from neuroimaging.glm.surface import (  # noqa: E402
    HEMIS,
    load_surface_bold,
    n_scans_surface,
    surface_bold_path,
    surface_mask_intersection,
)
from neuroimaging.io import find_fmriprep_runs, load_confounds  # noqa: E402

SPACE = "fsaverage6"
MESH_KIND = "white"  # vertex identity only; any fsaverage6 mesh gives the same numbers
FS_HEMI = {"L": "lh", "R": "rh"}


def _bare(label: str, prefix: str) -> str:
    return label[len(prefix) + 1 :] if label.startswith(prefix + "-") else label


def _config_paths() -> tuple[Path, Path]:
    """(bids_root, derivatives_dir) from config/*.toml — never hard-coded."""
    from core.config import load_config

    cfg = load_config()
    bids_root = Path(cfg["paths"]["bids_project_dir"])
    derivatives = Path(cfg["paths"].get("output_dir", bids_root / "derivatives"))
    return bids_root, derivatives


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subject", required=True, help="sub-## or ##")
    p.add_argument("--model", required=True, help="model name in models/ (floc, motor) or a path")
    p.add_argument("--sessions", nargs="*", default=None, help="restrict to these sessions")
    p.add_argument("--regime", default="reference")
    p.add_argument("--output-tree", default="functional_space/localizer_splithalf")
    p.add_argument("--template-mesh-dir", type=Path, default=None,
                   help=f"FreeSurfer surf/ dir of fsaverage6 holding ?h.{MESH_KIND} "
                        "(default: <fmriprep>/sourcedata/freesurfer/fsaverage6/surf)")
    p.add_argument("--dry-run", action="store_true", help="discover runs, build designs, list splits; fit nothing")
    p.add_argument("--bids-root", type=Path, default=None)
    p.add_argument("--derivatives-dir", type=Path, default=None)
    return p.parse_args(argv)


def load_template_mesh(mesh_dir: Path):
    from nilearn.surface import PolyMesh

    paths = {h: mesh_dir / f"{FS_HEMI[h]}.{MESH_KIND}" for h in HEMIS}
    missing = [str(p) for p in paths.values() if not p.exists() or p.stat().st_size == 0]
    if missing:
        sys.exit("ERROR: fsaverage6 mesh not found or empty: " + ", ".join(missing)
                 + ". Pass --template-mesh-dir pointing at an fsaverage6 FreeSurfer surf/ directory.")
    return PolyMesh(**{HEMIS[h]: str(p) for h, p in paths.items()}), paths


def complementary_splits(n: int) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    """Every split of range(n) into two equal halves, each unordered pair once.

    Pairs are enumerated with run 0 in the first half, so 6 runs give the 10
    unique 3-vs-3 pairs and 2 runs give the single 1-vs-1 pair.
    """
    if n < 2 or n % 2:
        raise ValueError(f"split halves need an even number of runs >= 2, got {n}")
    idx = tuple(range(n))
    out = []
    for a in itertools.combinations(idx, n // 2):
        if 0 not in a:
            continue
        b = tuple(i for i in idx if i not in a)
        out.append((a, b))
    return out


def fixed_effects_arrays(effects: list[np.ndarray], variances: list[np.ndarray]) -> tuple[np.ndarray, ...]:
    """Precision-weighted fixed effects, as nilearn ``compute_fixed_effects(precision_weighted=True)``."""
    e = np.stack(effects)
    v = np.stack(variances)
    with np.errstate(divide="ignore", invalid="ignore"):
        w = 1.0 / v
        var = 1.0 / w.sum(axis=0)
        eff = var * (w * e).sum(axis=0)
        t = eff / np.sqrt(var)
    return eff, var, t


def _surface_arrays(img, mask: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Per-hemisphere float64 arrays from a SurfaceImage map, NaN outside the mask."""
    out = {}
    for hemi, part in HEMIS.items():
        a = np.asarray(img.data.parts[part], dtype=np.float64).ravel()
        a = a.copy()
        a[~mask[hemi]] = np.nan
        out[hemi] = a
    return out


def save_gifti(arrays: list[np.ndarray], path: Path) -> Path:
    import nibabel as nib

    darrays = [nib.gifti.GiftiDataArray(np.asarray(a, dtype=np.float32), intent="NIFTI_INTENT_NONE",
                                        datatype="NIFTI_TYPE_FLOAT32") for a in arrays]
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.gifti.GiftiImage(darrays=darrays), str(path))
    return path


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "-C", str(_REPO), "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def ensure_dataset_description(out_base: Path, fmriprep_dir: Path) -> None:
    dd = out_base / "dataset_description.json"
    if dd.exists():
        return
    from importlib.metadata import version

    out_base.mkdir(parents=True, exist_ok=True)
    dd.write_text(json.dumps({
        "Name": "Localizer contrast maps on fsaverage6: per run, full data, and complementary split halves",
        "BIDSVersion": "1.9.0",
        "DatasetType": "derivative",
        "GeneratedBy": [
            {"Name": "nilearn", "Version": version("nilearn"),
             "Description": "first-level fits via neuroimaging.glm under the frozen reference spec"},
            {"Name": "localizer_splithalf.py",
             "CodeURL": "https://github.com/hulacon/mmmdata/tree/main/scripts/functional_space",
             "Description": "per-run fits on space-fsaverage6 BOLD; precision-weighted fixed effects "
                            "over all runs and over each complementary split half"},
        ],
        "SourceDatasets": [{"URL": f"bids:derivatives/{fmriprep_dir.name}"}],
    }, indent=2) + "\n")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.bids_root is not None:
        bids_root = args.bids_root
        derivatives = args.derivatives_dir or bids_root / "derivatives"
    else:
        bids_root, derivatives = _config_paths()
        if args.derivatives_dir is not None:
            derivatives = args.derivatives_dir

    model = load_model(args.model)
    cfg = dataclasses.replace(reference_config(args.regime), space=SPACE, hrf_model=model.hrf_model,
                              output_tree=args.output_tree)
    desc = glm_desc(args.regime, cfg.noise_model, cfg.smoothing_fwhm, cfg.variant)
    subject = _bare(args.subject, "sub")

    runs = find_fmriprep_runs(subject=subject, task=model.task, variant=cfg.variant, space=SPACE,
                              bids_root=bids_root)
    if args.sessions:
        keep = {_bare(s, "ses") for s in args.sessions}
        runs = [r for r in runs if r.session in keep]
    if not runs:
        sys.exit(f"ERROR: no fMRIPrep runs for sub-{subject} task-{model.task} under {bids_root}")
    bad = [r.entity_prefix for r in runs if r.events is None or r.confounds is None
           or not all(surface_bold_path(r, h, SPACE).exists() for h in HEMIS)]
    if bad:
        sys.exit("ERROR: runs missing events, confounds or space-fsaverage6 BOLD: " + ", ".join(bad))
    splits = complementary_splits(len(runs))

    fmriprep_dir = bids_root / DERIVATIVES_DIRS[cfg.variant]
    mesh_dir = args.template_mesh_dir or fmriprep_dir / "sourcedata" / "freesurfer" / "fsaverage6" / "surf"
    mesh, mesh_paths = load_template_mesh(mesh_dir)

    print(f"model {model.name}: task-{model.task}, contrasts {', '.join(c.name for c in model.contrasts)}")
    print(f"runs ({len(runs)}): " + ", ".join(r.entity_prefix for r in runs))
    print(f"splits ({len(splits)}): " + "; ".join(f"{a} vs {b}" for a, b in splits))
    print(f"config: {json.dumps(cfg.to_dict())}")
    print(f"desc: {desc} -> {args.output_tree}")

    all_events = adapt_events(model.adapter, [pd.read_csv(r.events, sep="\t", na_values=["n/a"]) for r in runs])
    designs = []
    for run, events in zip(runs, all_events):
        t_r = repetition_time(run, bids_root)
        n_scans = n_scans_surface(run, SPACE)
        dm = build_design_matrix(events, load_confounds(run), t_r, n_scans, model, cfg, strict=strict_for(model))
        vectors, skipped = available_contrast_vectors(model, list(dm.columns))
        if skipped:
            sys.exit(f"ERROR: {run.entity_prefix} cannot estimate {skipped}; split halves need every "
                     "contrast in every run")
        designs.append((run, t_r, dm, vectors))
        print(f"  {run.entity_prefix}: TR {t_r} s, {n_scans} volumes, {dm.shape[1]} design columns")

    if args.dry_run:
        print("dry run: designs built, nothing fitted or written")
        return 0

    mask_img = surface_mask_intersection(runs, SPACE, mesh)
    mask = {h: np.asarray(mask_img.data.parts[p], dtype=bool).ravel() for h, p in HEMIS.items()}

    out_base = derivatives / args.output_tree
    ensure_dataset_description(out_base, fmriprep_dir)

    estimator = get_estimator("nilearn")
    per_run: dict[str, list[tuple[dict, dict]]] = {c.name: [] for c in model.contrasts}
    for run, t_r, dm, vectors in designs:
        bold = load_surface_bold(run, SPACE, mesh)
        est = estimator.fit_run(bold, dm, vectors, t_r=t_r, mask=mask_img, cfg=cfg)
        d = out_base / f"sub-{subject}" / f"ses-{run.session}" / "func"
        for name, ce in est.items():
            eff = _surface_arrays(ce.effect, mask)
            var = _surface_arrays(ce.variance, mask)
            per_run[name].append((eff, var))
            for stat, arrs in (("effect", eff), ("variance", var)):
                for h in HEMIS:
                    save_gifti([arrs[h]], d / statmap_name(subject, model.task, SPACE, name, stat,
                                                           session=run.session, run=run.run, hemi=h,
                                                           ext=".func.gii", desc=desc))
        print(f"  fitted {run.entity_prefix}")

    d = out_base / f"sub-{subject}" / "func"
    for h in HEMIS:
        save_gifti([mask[h].astype(np.float32)],
                   d / f"sub-{subject}_task-{model.task}_hemi-{h}_space-{SPACE}_desc-{desc}_mask.func.gii")

    split_desc = desc + "Splithalf"
    n_written = 0
    for name, estimates in per_run.items():
        for h in HEMIS:
            effs = [e[h] for e, _ in estimates]
            vars_ = [v[h] for _, v in estimates]
            full = fixed_effects_arrays(effs, vars_)
            for stat, arr in zip(("effect", "variance", "t"), full):
                save_gifti([arr], d / statmap_name(subject, model.task, SPACE, name, stat, hemi=h,
                                                   ext=".func.gii", desc=desc))
                n_written += 1
            halves = {"effect": [], "t": []}
            for a, b in splits:
                for half in (a, b):
                    eff, _, t = fixed_effects_arrays([effs[i] for i in half], [vars_[i] for i in half])
                    halves["effect"].append(eff)
                    halves["t"].append(t)
            for stat, arrs in halves.items():
                save_gifti(arrs, d / statmap_name(subject, model.task, SPACE, name, stat, hemi=h,
                                                  ext=".func.gii", desc=split_desc))
                n_written += 1

    run_ids = [r.entity_prefix for r in runs]
    sidecar = {
        "model": model.name,
        "model_path": str(model.path),
        "task": model.task,
        "space": SPACE,
        "desc": desc,
        "split_desc": split_desc,
        "config": cfg.to_dict(),
        "runs": run_ids,
        "splits": [{"split": k, "half_a": [run_ids[i] for i in a], "half_b": [run_ids[i] for i in b]}
                   for k, (a, b) in enumerate(splits)],
        "darray_order": "split-half maps hold 2 data arrays per split: [split0 half_a, split0 half_b, "
                        "split1 half_a, ...]",
        "pooling": "precision-weighted fixed effects (1/variance weights); t = effect / sqrt(pooled variance)",
        "masked_value": "NaN",
        "template_mesh": {h: str(p) for h, p in mesh_paths.items()},
        "driver": "scripts/functional_space/localizer_splithalf.py",
        "mmmdata_git_sha": _git_sha(),
    }
    (d / f"sub-{subject}_task-{model.task}_space-{SPACE}_desc-{split_desc}_statmap.json").write_text(
        json.dumps(sidecar, indent=2) + "\n")
    print(f"wrote {n_written} subject-level maps to {d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
