#!/usr/bin/env python3
"""
glm_contrast_maps.py — condition-level contrast maps for one subject and one
BIDS Stats Model, pooled across runs by precision-weighted fixed effects.

The first production runner on the glm-strategy architecture (mmmdata-agents
docs/workbench/glm-strategy/, DECIDED 2026-08-25): a model spec from
mmmdata/models/ declares the conditions and contrasts; neuroimaging.glm
builds the design from BIDS events + fMRIPrep confounds, fits with the
chosen estimator, and pools runs with nilearn compute_fixed_effects.

Defaults are the frozen reference specification
(neuroimaging/glm/reference_spec.json): OLS, SPM canonical, the `reference`
confound regime, unsmoothed, every run fitted and pooled inside the
intersection of the runs' brain masks. `--noise-model ar1` is the spec's
calibrated-inference engine. Any other flag departs from the reference, and
the fit's metadata records the whole config so the departure is visible.
Every output filename carries Contract A keys plus `contrast-` and `stat-`
entities, and `desc-<regime><ENGINE>` (e.g. `desc-referenceAR1`) naming the
processing variant, so every engine, regime and space shares one tree,
derivatives/nilearn_glm. After each fit the tree's `descriptions.tsv` (one row
per desc label) and `maps.tsv` (every map: subject, session, task, space,
contrast, stat, desc, path) are refreshed under a lock. Maps sit where BIDS
puts them — a one-session pool under ses-XX/, a cross-session pool at
sub-XX/ — so find them through maps.tsv or the catalog.

`--space` takes any volumetric space fMRIPrep wrote (the MNI reference,
`T1w`, `func`) or the subject surface `fsnative`. A surface fit reads the
per-hemisphere `.func.gii` BOLD on the subject's midthickness mesh, masks to
vertices with signal in every run, and writes `hemi-L`/`hemi-R` `.func.gii`
maps (neuroimaging.glm.surface); the estimator and fixed effects are the
volume ones.

Run discovery goes through neuroimaging.io.find_fmriprep_runs, which
refuses a task-motor or task-auditory selection spanning both session
groups unless --sessions or --allow-mixed-designs says so. For motor the
two-protocol claim behind that guard was withdrawn 2026-09-08
(OPEN-QUESTIONS Q19); the guard stays as a caution.

Usage:
    python glm_contrast_maps.py --subject sub-03 --model motor --sessions ses-30 --dry-run
    python glm_contrast_maps.py --subject sub-03 --model tbrepetition   # adapter needs all 42 runs
    python glm_contrast_maps.py --subject sub-03 --model floc
    python glm_contrast_maps.py --subject sub-03 --model floc --noise-model ar1
    python glm_contrast_maps.py --subject sub-03 --model floc --space T1w
    python glm_contrast_maps.py --subject sub-03 --model floc --space fsnative
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO / "src" / "python") not in sys.path:
    sys.path.insert(0, str(_REPO / "src" / "python"))

from neuroimaging.constants import DERIVATIVES_DIRS  # noqa: E402
from neuroimaging.glm.adapters import adapt_events  # noqa: E402
from neuroimaging.glm.config import repetition_time  # noqa: E402
from neuroimaging.glm.design import available_contrast_vectors, build_design_matrix, strict_for  # noqa: E402
from neuroimaging.glm.estimators import fixed_effects, get_estimator  # noqa: E402
from neuroimaging.glm.models import list_models, load_model  # noqa: E402
from neuroimaging.glm.reference import load_reference_spec, reference_config  # noqa: E402
from neuroimaging.glm.outputs import (  # noqa: E402
    describe_glm_desc,
    ensure_dataset_description,
    glm_desc,
    output_dir,
    update_tree_index,
    save_statmap,
    statmap_name,
    write_run_metadata,
)
from neuroimaging.glm.surface import (  # noqa: E402
    HEMIS,
    is_surface_space,
    load_mesh,
    load_surface_bold,
    n_scans_surface,
    save_surface_statmap,
    subject_mesh_paths,
    surface_bold_path,
    surface_mask_intersection,
)
from neuroimaging.io import FmriprepRun, find_fmriprep_runs, load_confounds, mask_intersection  # noqa: E402


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
    p.add_argument("--subject", required=True, help="sub-03 or 03")
    p.add_argument("--model", required=True, help=f"model name in models/ ({', '.join(list_models())}) or a path")
    p.add_argument("--sessions", nargs="*", default=None, help="restrict to these sessions (ses-30 or 30)")
    ref = reference_config()
    p.add_argument("--space", default=ref.space)
    p.add_argument("--variant", default=ref.variant, help="fmriprep tree to read")
    p.add_argument("--estimator", default="nilearn")
    p.add_argument("--noise-model", default=ref.noise_model, choices=["ar1", "ols"],
                   help="ols = the reference effect engine; ar1 = its calibrated-z/t engine")
    p.add_argument("--regime", default="reference", choices=sorted(load_reference_spec()["confound_regimes"]),
                   help="named confound regime from the reference spec")
    p.add_argument("--smoothing-fwhm", type=float, default=ref.smoothing_fwhm,
                   help="mm; 0 or unset = unsmoothed (the reference)")
    p.add_argument("--output-tree", default="nilearn_glm")
    p.add_argument("--allow-mixed-designs", action="store_true",
                   help="pool a split-design task across both session groups (see find_fmriprep_runs)")
    p.add_argument("--per-run-maps", action="store_true", help="also write each run's maps")
    p.add_argument("--dry-run", action="store_true", help="discover, build designs, print the plan; fit nothing")
    # Path overrides, for tests and off-config trees. Production reads config/*.toml.
    p.add_argument("--bids-root", type=Path, default=None)
    p.add_argument("--derivatives-dir", type=Path, default=None)
    return p.parse_args(argv)


def select_runs(args: argparse.Namespace, task: str, bids_root: Path) -> list[FmriprepRun]:
    subject = _bare(args.subject, "sub")
    sessions = {_bare(s, "ses") for s in args.sessions} if args.sessions else None
    runs: list[FmriprepRun] = []
    if sessions and len(sessions) == 1:
        runs = find_fmriprep_runs(subject=subject, session=next(iter(sessions)), task=task,
                                  variant=args.variant, space=args.space, bids_root=bids_root,
                                  allow_mixed_designs=args.allow_mixed_designs)
    else:
        runs = find_fmriprep_runs(subject=subject, task=task, variant=args.variant, space=args.space,
                                  bids_root=bids_root, allow_mixed_designs=args.allow_mixed_designs)
        if sessions:
            runs = [r for r in runs if r.session in sessions]
    if not runs:
        sys.exit(f"ERROR: no fMRIPrep runs for sub-{subject} task-{task} in {args.variant} "
                 f"(space {args.space}) under {bids_root}. Check the tree, the variant, and the space.")
    missing = [r.entity_prefix for r in runs if r.events is None]
    if missing:
        sys.exit("ERROR: runs without an events.tsv cannot be modelled: " + ", ".join(missing)
                 + ". Generate events first (raw2bids_converters) or exclude them with --sessions.")
    if is_surface_space(args.space):
        incomplete = [r.entity_prefix for r in runs if r.confounds is None
                      or not all(surface_bold_path(r, h, args.space).exists() for h in HEMIS)]
    else:
        incomplete = [r.entity_prefix for r in runs if r.bold is None or r.mask is None or r.confounds is None]
    if incomplete:
        sys.exit("ERROR: runs missing BOLD, mask or confounds in the requested space: " + ", ".join(incomplete))
    return runs


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
    cfg = dataclasses.replace(
        reference_config(args.regime),
        space=args.space,
        variant=args.variant,
        noise_model=args.noise_model,
        smoothing_fwhm=None if not args.smoothing_fwhm else args.smoothing_fwhm,
        hrf_model=model.hrf_model,
        output_tree=args.output_tree,
    )
    desc = glm_desc(args.regime, cfg.noise_model, cfg.smoothing_fwhm, cfg.variant)
    runs = select_runs(args, model.task, bids_root)
    subject = runs[0].subject
    fmriprep_dir = bids_root / DERIVATIVES_DIRS[args.variant]
    surface = is_surface_space(cfg.space)
    # One mask for every run and for the pool: fitting each run inside its own
    # mask and pooling under one run's leaves edge voxels estimated from a
    # varying subset of runs. A surface mask needs every run's data, so it is
    # built after the dry-run exit below.
    mesh, mesh_paths, mask_img = None, None, None
    try:
        if surface:
            mesh_paths = subject_mesh_paths(fmriprep_dir, subject)
            mesh = load_mesh(mesh_paths)
        else:
            mask_img, _ = mask_intersection(runs)
    except (ValueError, FileNotFoundError) as e:
        sys.exit(f"ERROR: {e}")

    print(f"model {model.name}: task-{model.task}, {len(model.conditions)} conditions, "
          f"{len(model.contrasts)} contrasts ({', '.join(c.name for c in model.contrasts)})")
    print(f"runs ({len(runs)}): " + ", ".join(r.entity_prefix for r in runs))
    print(f"config: {json.dumps(cfg.to_dict())}")
    print(f"desc: {desc} -> {cfg.output_tree}")

    # Designs first, for every run, before any fitting: a bad run fails the
    # whole job here rather than after an hour of estimation.
    import nibabel as nib

    # A model's events adapter needs every run's events at once (anchors,
    # presentation order), so events are read for all runs before any design.
    all_events = adapt_events(model.adapter, [pd.read_csv(r.events, sep="\t", na_values=["n/a"]) for r in runs])
    designs = []
    for run, events in zip(runs, all_events):
        t_r = repetition_time(run, bids_root)
        n_scans = n_scans_surface(run, cfg.space) if surface else nib.load(str(run.bold)).shape[-1]
        confounds = load_confounds(run)
        dm = build_design_matrix(events, confounds, t_r, n_scans, model, cfg, strict=strict_for(model))
        # Adapter-derived levels may be absent from a run (a TBencoding run with
        # no first presentation); such a run is skipped for that contrast only.
        vectors, skipped = available_contrast_vectors(model, list(dm.columns))
        if skipped:
            print(f"  {run.entity_prefix}: cannot estimate {skipped} (conditions absent); skipped for those")
        designs.append((run, t_r, dm, vectors))
        print(f"  {run.entity_prefix}: TR {t_r} s, {n_scans} volumes, "
              f"{dm.shape[1]} design columns ({len(model.conditions)} conditions, "
              f"{dm.shape[1] - len(model.conditions) - 1} confounds/drift, 1 intercept)")

    if args.dry_run:
        print("dry run: designs built, nothing fitted or written")
        return 0

    if surface:
        try:
            mask_img = surface_mask_intersection(runs, cfg.space, mesh)
        except ValueError as e:
            sys.exit(f"ERROR: {e}")

    def write_map(img, d: Path, name: str, stat: str, session, run=None) -> list[str]:
        if surface:
            paths = {h: d / statmap_name(subject, model.task, cfg.space, name, stat, session=session, run=run,
                                         hemi=h, ext=".func.gii", desc=desc) for h in HEMIS}
            return [p.name for p in save_surface_statmap(img, paths)]
        path = d / statmap_name(subject, model.task, cfg.space, name, stat, session=session, run=run, desc=desc)
        return [save_statmap(img, path).name]

    estimator = get_estimator(args.estimator)
    out_base = derivatives / cfg.output_tree
    ensure_dataset_description(out_base, fmriprep_dir, model.name, estimator.name)

    per_contrast: dict[str, list] = {c.name: [] for c in model.contrasts}
    for run, t_r, dm, vectors in designs:
        bold = load_surface_bold(run, cfg.space, mesh) if surface else nib.load(str(run.bold))
        est = estimator.fit_run(bold, dm, vectors, t_r=t_r, mask=mask_img, cfg=cfg)
        for name, ce in est.items():
            per_contrast[name].append(ce)
            if args.per_run_maps:
                d = output_dir(derivatives, cfg.output_tree, run.subject, run.session)
                d.mkdir(parents=True, exist_ok=True)
                for stat, img in (("effect", ce.effect), ("variance", ce.variance), ("t", ce.stat), ("z", ce.z)):
                    if img is not None:
                        write_map(img, d, name, stat, run.session, run.run)
        print(f"  fitted {run.entity_prefix}")

    # Fixed effects across every run selected: sessions pool together, so the
    # subject-level map carries no ses- entity. Session-level pooling is a
    # --sessions call per session.
    sessions = sorted({r.session for r in runs})
    fx_session = sessions[0] if len(sessions) == 1 else None
    d = output_dir(derivatives, cfg.output_tree, subject, fx_session)
    d.mkdir(parents=True, exist_ok=True)
    written = []
    for name, estimates in per_contrast.items():
        if not estimates:
            sys.exit(f"ERROR: no run could estimate contrast {name}; its conditions are absent everywhere")
        fx = fixed_effects(estimates, mask=mask_img) if model.fixed_effects or len(estimates) > 1 else None
        maps = (
            (("effect", fx.effect), ("variance", fx.variance), ("t", fx.stat), ("z", fx.z))
            if fx is not None
            else (("effect", estimates[0].effect), ("variance", estimates[0].variance),
                  ("t", estimates[0].stat), ("z", estimates[0].z))
        )
        for stat, img in maps:
            if img is None:
                continue
            written += write_map(img, d, name, stat, fx_session)

    meta = {
        "model": model.name,
        "model_path": str(model.path),
        "task": model.task,
        "estimator": estimator.name,
        "regime": args.regime,
        "desc": desc,
        "config": cfg.to_dict(),
        "runs": [{"subject": r.subject, "session": r.session, "run": r.run, "events": str(r.events)} for r in runs],
        "fixed_effects": model.fixed_effects,
        "contrasts": {c.name: c.weights for c in model.contrasts},
        "outputs": written,
    }
    if surface:
        meta["surface_mesh"] = {h: str(p) for h, p in mesh_paths.items()}
    # space- and desc- in the name: fits in several spaces and variants share a directory.
    write_run_metadata(d / f"sub-{subject}_task-{model.task}_space-{cfg.space}_desc-{desc}_model-{model.name}"
                           "_run_metadata.json", meta)
    update_tree_index(out_base, desc, describe_glm_desc(args.regime, cfg.noise_model, cfg.smoothing_fwhm, cfg.variant))
    print(f"wrote {len(written)} maps to {d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
