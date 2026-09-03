#!/usr/bin/env python3
"""
glm_contrast_maps.py — condition-level contrast maps for one subject and one
BIDS Stats Model, pooled across runs by precision-weighted fixed effects.

The first production runner on the glm-strategy architecture (mmmdata-agents
docs/workbench/glm-strategy/, DECIDED 2026-08-25): a model spec from
mmmdata/models/ declares the conditions and contrasts; neuroimaging.glm
builds the design from BIDS events + fMRIPrep confounds, fits with the
chosen estimator (nilearn FirstLevelModel, AR(1), by default), and pools runs
with nilearn compute_fixed_effects. Every output filename carries Contract A
keys plus `contrast-` and `stat-` entities.

Run discovery goes through neuroimaging.io.find_fmriprep_runs, so a
task-motor or task-auditory selection that spans both session groups is
refused (those labels cover two protocols); pass --sessions or
--allow-mixed-designs deliberately.

Nothing here has run on real data yet. The first real fit is a cluster step:
mmmdata-agents docs/cluster-reentry.md R15.

Usage:
    python glm_contrast_maps.py --subject sub-03 --model motor --sessions ses-30 --dry-run
    python glm_contrast_maps.py --subject sub-03 --model floc
    python glm_contrast_maps.py --subject sub-03 --model floc --estimator nilearn --noise-model ols
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
from neuroimaging.glm.config import DEFAULT_CONFIG, GlmConfig, repetition_time  # noqa: E402
from neuroimaging.glm.design import build_design_matrix, contrast_vectors  # noqa: E402
from neuroimaging.glm.estimators import fixed_effects, get_estimator  # noqa: E402
from neuroimaging.glm.models import list_models, load_model  # noqa: E402
from neuroimaging.glm.outputs import (  # noqa: E402
    ensure_dataset_description,
    output_dir,
    statmap_name,
    write_run_metadata,
)
from neuroimaging.io import FmriprepRun, find_fmriprep_runs, load_confounds  # noqa: E402


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
    p.add_argument("--space", default=DEFAULT_CONFIG.space)
    p.add_argument("--variant", default=DEFAULT_CONFIG.variant, help="fmriprep tree to read")
    p.add_argument("--estimator", default="nilearn")
    p.add_argument("--noise-model", default=DEFAULT_CONFIG.noise_model, choices=["ar1", "ols"])
    p.add_argument("--smoothing-fwhm", type=float, default=DEFAULT_CONFIG.smoothing_fwhm,
                   help="mm; 0 disables")
    p.add_argument("--output-tree", default=DEFAULT_CONFIG.output_tree)
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
        DEFAULT_CONFIG,
        space=args.space,
        variant=args.variant,
        noise_model=args.noise_model,
        smoothing_fwhm=None if not args.smoothing_fwhm else args.smoothing_fwhm,
        hrf_model=model.hrf_model,
        output_tree=args.output_tree,
    )
    runs = select_runs(args, model.task, bids_root)
    subject = runs[0].subject
    fmriprep_dir = bids_root / DERIVATIVES_DIRS[args.variant]

    print(f"model {model.name}: task-{model.task}, {len(model.conditions)} conditions, "
          f"{len(model.contrasts)} contrasts ({', '.join(c.name for c in model.contrasts)})")
    print(f"runs ({len(runs)}): " + ", ".join(r.entity_prefix for r in runs))
    print(f"config: {json.dumps(cfg.to_dict())}")

    # Designs first, for every run, before any fitting: a bad run fails the
    # whole job here rather than after an hour of estimation.
    import nibabel as nib

    designs = []
    for run in runs:
        t_r = repetition_time(run, bids_root)
        n_scans = nib.load(str(run.bold)).shape[-1]
        events = pd.read_csv(run.events, sep="\t", na_values=["n/a"])
        confounds = load_confounds(run)
        dm = build_design_matrix(events, confounds, t_r, n_scans, model, cfg)
        vectors = contrast_vectors(model, list(dm.columns))
        designs.append((run, t_r, dm, vectors))
        print(f"  {run.entity_prefix}: TR {t_r} s, {n_scans} volumes, "
              f"{dm.shape[1]} design columns ({len(model.conditions)} conditions, "
              f"{dm.shape[1] - len(model.conditions) - 1} confounds/drift, 1 intercept)")

    if args.dry_run:
        print("dry run: designs built, nothing fitted or written")
        return 0

    estimator = get_estimator(args.estimator)
    out_base = derivatives / cfg.output_tree
    ensure_dataset_description(out_base, fmriprep_dir, model.name, estimator.name)

    per_contrast: dict[str, list] = {c.name: [] for c in model.contrasts}
    for run, t_r, dm, vectors in designs:
        mask = nib.load(str(run.mask))
        bold = nib.load(str(run.bold))
        est = estimator.fit_run(bold, dm, vectors, t_r=t_r, mask=mask, cfg=cfg)
        for name, ce in est.items():
            per_contrast[name].append(ce)
            if args.per_run_maps:
                d = output_dir(derivatives, cfg.output_tree, run.subject, run.session)
                d.mkdir(parents=True, exist_ok=True)
                for stat, img in (("effect", ce.effect), ("variance", ce.variance), ("t", ce.stat), ("z", ce.z)):
                    if img is not None:
                        img.to_filename(str(d / statmap_name(run.subject, model.task, cfg.space, name, stat,
                                                              session=run.session, run=run.run)))
        print(f"  fitted {run.entity_prefix}")

    # Fixed effects across every run selected: sessions pool together, so the
    # subject-level map carries no ses- entity. Session-level pooling is a
    # --sessions call per session.
    sessions = sorted({r.session for r in runs})
    fx_session = sessions[0] if len(sessions) == 1 else None
    d = output_dir(derivatives, cfg.output_tree, subject, fx_session)
    d.mkdir(parents=True, exist_ok=True)
    mask_img = nib.load(str(runs[0].mask))
    written = []
    for name, estimates in per_contrast.items():
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
            path = d / statmap_name(subject, model.task, cfg.space, name, stat, session=fx_session)
            img.to_filename(str(path))
            written.append(path.name)

    meta = {
        "model": model.name,
        "model_path": str(model.path),
        "task": model.task,
        "estimator": estimator.name,
        "config": cfg.to_dict(),
        "runs": [{"subject": r.subject, "session": r.session, "run": r.run, "events": str(r.events)} for r in runs],
        "fixed_effects": model.fixed_effects,
        "contrasts": {c.name: c.weights for c in model.contrasts},
        "outputs": written,
    }
    write_run_metadata(d / f"sub-{subject}_task-{model.task}_model-{model.name}_run_metadata.json", meta)
    print(f"wrote {len(written)} maps to {d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
