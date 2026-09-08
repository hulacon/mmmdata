#!/usr/bin/env python3
"""
glm_bakeoff.py — the one-pass GLM estimator bake-off, one cell per invocation.

Design record: mmmdata-agents docs/workbench/glm-strategy/ (scope DECIDED
2026-09-08). A factorial of HRF (spm / spm+derivative / per-voxel GLMsingle
library kernel) x confounds (motion6 / motion24 / motion6+aCompCor) x engine
(nilearn OLS / nilearn AR(1) / 3dREMLfit ARMA(1,1)), plus two standalone
GLMsingle arms, fitted on fLoc, motor and TBencoding first-vs-later, frozen
on fmriprep 25.2.5 MNI152NLin2009cAsym res-2. Every cell writes split-half
t maps and the pre-registered scores (neuroimaging.glm.harness) into
<derivatives>/glm_bakeoff/; nothing here ranks anything — `collect` tables
the scores and the ranking is read off the table.

Verbs:
    plan      freeze the harness into the output tree and write units.txt
              (one "sub-XX cell-id" per line) for the SLURM array
    prep      per subject, once, on a large-memory node: extract HRFindex
              from the encoding GLMsingle fit as a NIfTI (the per-voxel HRF
              arm reads it) and score the glmsingle-betas arm from the same
              load (TBencoding per-trial betas, first vs later per half)
    fit       one cell for one subject (--cell, or --unit N from units.txt)
    collect   gather every scores.json into scores.tsv and print a summary

Usage:
    python glm_bakeoff.py plan
    python glm_bakeoff.py prep --subject sub-03
    python glm_bakeoff.py fit --subject sub-03 --cell model-floc_hrf-spm_conf-motion6_engine-nilearn-ar1
    python glm_bakeoff.py fit --unit 17
    python glm_bakeoff.py collect
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO / "src" / "python") not in sys.path:
    sys.path.insert(0, str(_REPO / "src" / "python"))
if str(_REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO / "scripts"))

from glm_contrast_maps import _bare, _config_paths, select_runs  # noqa: E402
from neuroimaging.constants import DEFAULT_SPACE, DEFAULT_VARIANT  # noqa: E402
from neuroimaging.glm import harness  # noqa: E402
from neuroimaging.glm.adapters import adapt_events  # noqa: E402
from neuroimaging.glm.config import DEFAULT_CONFIG, repetition_time  # noqa: E402
from neuroimaging.glm.design import build_design_matrix, contrast_vectors  # noqa: E402
from neuroimaging.glm.estimators import ENGINES, ContrastEstimate, fixed_effects, get_estimator  # noqa: E402
from neuroimaging.glm.glmsingle_arm import (  # noqa: E402
    block_design,
    fit_glmsingle_half,
    tb_trial_labels,
    welch_contrast,
)
from neuroimaging.glm.hrf import hrfindex_to_image, load_hrfindex  # noqa: E402
from neuroimaging.glm.models import load_model  # noqa: E402
from neuroimaging.glm.outputs import write_run_metadata  # noqa: E402
from neuroimaging.glm.voxelwise_hrf import VOXELWISE, fit_run_voxelwise  # noqa: E402
from neuroimaging.io import FmriprepRun, load_confounds  # noqa: E402

OUTPUT_TREE = "glm_bakeoff"
GLMSINGLE_TREE = "glmsingle_tb"
MODEL_SESSIONS: dict[str, Optional[list[str]]] = {"motor": ["ses-30"], "floc": None, "tbrepetition": None}
STATS = ("effect", "variance", "t", "z")


# --------------------------------------------------------------------- paths
def _paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    if args.bids_root is not None:
        bids_root = args.bids_root
        derivatives = args.derivatives_dir or bids_root / "derivatives"
    else:
        bids_root, derivatives = _config_paths()
        if args.derivatives_dir is not None:
            derivatives = args.derivatives_dir
    return bids_root, derivatives, derivatives / OUTPUT_TREE


def hrfindex_path(derivatives: Path, subject: str, space: str) -> Path:
    s = _bare(subject, "sub")
    return (derivatives / GLMSINGLE_TREE / f"sub-{s}" / "enc"
            / f"sub-{s}_task-TBencoding_space-{space}_desc-hrfindex_dseg.nii.gz")


def _runs_for(model_name: str, subject: str, args: argparse.Namespace, bids_root: Path) -> tuple[Any, list[FmriprepRun]]:
    model = load_model(model_name)
    ns = argparse.Namespace(subject=subject, sessions=args.sessions or MODEL_SESSIONS.get(model_name),
                            variant=args.variant, space=args.space, allow_mixed_designs=False)
    runs = select_runs(ns, model.task, bids_root)
    runs = sorted(runs, key=lambda r: (r.session, r.run or ""))
    return model, runs


def _mask_intersection(runs: list[FmriprepRun]):
    import nibabel as nib

    first = nib.load(str(runs[0].mask))
    inter = np.asarray(first.dataobj).astype(bool)
    for r in runs[1:]:
        m = nib.load(str(r.mask))
        if m.shape != first.shape or not np.allclose(m.affine, first.affine, atol=1e-3):
            raise SystemExit(f"ERROR: {r.entity_prefix} mask grid differs from {runs[0].entity_prefix}; "
                             "the frozen input must be one space and resolution")
        inter &= np.asarray(m.dataobj).astype(bool)
    return nib.Nifti1Image(inter.astype(np.uint8), first.affine), inter


def _write_half_maps(out_dir: Path, subject: str, task: str, space: str, half: int,
                     estimates: dict[str, ContrastEstimate]) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, ce in estimates.items():
        for stat, img in (("effect", ce.effect), ("variance", ce.variance), ("t", ce.stat), ("z", ce.z)):
            if img is None:
                continue
            fn = harness.half_name(_bare(subject, "sub"), task, space, half, name, stat)
            img.to_filename(str(out_dir / fn))
            written.append(fn)
    return written


def _score(halves: tuple[dict, dict], mask_arr: np.ndarray, n_set) -> dict[str, Any]:
    scores = {}
    for name in halves[0]:
        a, b = halves[0][name], halves[1][name]
        t1, t2 = np.asarray(a.stat.dataobj, dtype=float), np.asarray(b.stat.dataobj, dtype=float)
        z1 = np.asarray(a.z.dataobj, dtype=float) if a.z is not None else None
        z2 = np.asarray(b.z.dataobj, dtype=float) if b.z is not None else None
        scores[name] = harness.score_halves(t1, t2, mask_arr, n_set, z1, z2)
    return scores


def _pool(estimates: list[ContrastEstimate], mask_img) -> ContrastEstimate:
    if len(estimates) == 1:
        return estimates[0]
    fx = fixed_effects(estimates, mask=mask_img)
    return ContrastEstimate(effect=fx.effect, variance=fx.variance, dof=None, stat=fx.stat, z=fx.z)


# --------------------------------------------------------------------- verbs
def cmd_plan(args: argparse.Namespace) -> int:
    _, derivatives, out_base = _paths(args)
    out_base.mkdir(parents=True, exist_ok=True)
    spec_path = harness.freeze(out_base)
    models = args.models or list(harness.MODELS)
    cells = harness.factorial_cells(models)
    subjects = [_bare(s, "sub") for s in args.subjects]
    prep_cells = {c.id for c in cells if c.hrf == "glmsingle-betas"}
    lines = [f"sub-{s} {c.id}" for s in subjects for c in cells if c.id not in prep_cells]
    units = out_base / "units.txt"
    units.write_text("\n".join(lines) + "\n")
    n_fact = sum(1 for c in cells if not c.standalone)
    print(f"harness frozen at {spec_path} (sha {json.loads(spec_path.read_text())['sha256']})")
    print(f"{len(cells)} cells per subject ({n_fact} factorial + {len(cells) - n_fact} standalone); "
          f"{len(prep_cells)} run under `prep`")
    print(f"{len(lines)} array units -> {units}")
    return 0


def cmd_prep(args: argparse.Namespace) -> int:
    import nibabel as nib

    bids_root, derivatives, out_base = _paths(args)
    harness.check_frozen(out_base)
    subject = _bare(args.subject, "sub")
    model, runs = _runs_for("tbrepetition", subject, args, bids_root)
    enc_dir = derivatives / GLMSINGLE_TREE / f"sub-{subject}" / "enc"
    typed = enc_dir / "glmsingle_outputs" / "TYPED_FITHRF_GLMDENOISE_RR.npy"
    trial_info_path = enc_dir / "trial_info.csv"
    for p in (typed, trial_info_path):
        if not p.exists():
            raise SystemExit(f"ERROR: {p} missing; the encoding GLMsingle fit is the source of both "
                             "HRFindex and the per-trial betas")
    print(f"sub-{subject}: {len(runs)} TBencoding runs; loading {typed} (whole pickled dict, ~12 GB)")
    t0 = time.time()
    d = np.load(typed, allow_pickle=True).item()
    print(f"  loaded in {time.time() - t0:.0f} s; keys {sorted(d)}")
    mask_img, mask_arr = _mask_intersection(runs)

    # 1. HRFindex -> NIfTI on the frozen grid
    hidx = hrfindex_to_image(np.asarray(d["HRFindex"]), mask_img)
    hpath = hrfindex_path(derivatives, subject, args.space)
    hidx.to_filename(str(hpath))
    counts = np.bincount(np.asarray(hidx.dataobj)[mask_arr].ravel(), minlength=20)
    print(f"  HRFindex -> {hpath}; voxels per kernel in mask: {counts.tolist()}")

    # 2. glmsingle-betas arm: first vs later per half from the per-trial betas
    trial_info = pd.read_csv(trial_info_path)
    betas = np.asarray(d["betasmd"])
    del d
    if betas.shape[-1] != len(trial_info):
        raise SystemExit(f"ERROR: betasmd has {betas.shape[-1]} trials but trial_info.csv {len(trial_info)} rows")
    events = [pd.read_csv(r.events, sep="\t", na_values=["n/a"]) for r in runs]
    adapted = adapt_events(model.adapter, events)
    labels = tb_trial_labels(trial_info, adapted)
    run_key = [(f"ses-{r.session}", f"run-{r.run}") for r in runs]
    h1, h2 = harness.split_runs(len(runs))
    half_of_run = {run_key[i]: 1 for i in h1} | {run_key[i]: 2 for i in h2}
    trial_half = np.array([half_of_run[(s, r)] for s, r in zip(trial_info["session"], trial_info["run"])])
    cell = harness.Cell("tbrepetition", "glmsingle-betas", "na", "na")
    out_dir = harness.cell_dir(out_base, subject, cell)
    halves = []
    for h in (1, 2):
        first = betas[..., (trial_half == h) & (labels.to_numpy() == "first")]
        later = betas[..., (trial_half == h) & (labels.to_numpy() == "later")]
        print(f"  half {h}: {first.shape[-1]} first vs {later.shape[-1]} later trials")
        est = {"firstVsLater": welch_contrast(first, later, mask_img.affine, mask_arr)}
        _write_half_maps(out_dir, subject, model.task, args.space, h, est)
        halves.append(est)
    scores = _score((halves[0], halves[1]), mask_arr, harness.N_SETS["tbrepetition"])
    record = {
        "cell": cell.id, "subject": subject, "model": model.name, "task": model.task, "space": args.space,
        "contrasts": scores, "harness_sha256": json.loads((out_base / "harness.json").read_text())["sha256"],
        "runs": [r.entity_prefix for r in runs], "halves": [[run_key[i] for i in h1], [run_key[i] for i in h2]],
        "source": str(typed), "hrfindex": str(hpath), "elapsed_s": round(time.time() - t0, 1),
    }
    harness.write_scores(out_dir / "scores.json", record)
    print(f"  scores -> {out_dir / 'scores.json'}: " + json.dumps(scores["firstVsLater"]))
    return 0


def cmd_fit(args: argparse.Namespace) -> int:
    import nibabel as nib

    bids_root, derivatives, out_base = _paths(args)
    frozen = harness.check_frozen(out_base)
    if args.unit is not None:
        lines = (out_base / "units.txt").read_text().splitlines()
        try:
            subject, cell_id = lines[args.unit - 1].split()
        except (IndexError, ValueError):
            raise SystemExit(f"ERROR: unit {args.unit} not in {out_base / 'units.txt'} ({len(lines)} lines)")
    else:
        if not (args.subject and args.cell):
            raise SystemExit("ERROR: fit needs --unit N or both --subject and --cell")
        subject, cell_id = args.subject, args.cell
    subject = _bare(subject, "sub")
    cell = harness.Cell.parse(cell_id)
    model, runs = _runs_for(cell.model, subject, args, bids_root)
    mask_img, mask_arr = _mask_intersection(runs)
    out_dir = harness.cell_dir(out_base, subject, cell)
    if (out_dir / "scores.json").exists() and not args.force:
        print(f"{out_dir / 'scores.json'} exists; pass --force to refit")
        return 0
    h1, h2 = harness.split_runs(len(runs))
    n_set = harness.N_SETS[cell.model]
    print(f"cell {cell.id} sub-{subject}: {len(runs)} runs, halves {len(h1)}/{len(h2)}, "
          f"mask {int(mask_arr.sum())} voxels, N set {n_set}")
    t0 = time.time()
    events = [pd.read_csv(r.events, sep="\t", na_values=["n/a"]) for r in runs]
    adapted = adapt_events(model.adapter, events)
    timings: list[float] = []

    if cell.hrf == "glmsingle":
        halves = []
        for h, idx in ((1, h1), (2, h2)):
            bolds, designs, t_r = [], [], None
            for i in idx:
                r = runs[i]
                t_r = repetition_time(r, bids_root)
                n_scans = nib.load(str(r.bold)).shape[-1]
                designs.append(block_design(adapted[i], model.conditions, t_r, n_scans))
                bolds.append(r.bold)
            stimdur = float(np.median([adapted[i]["duration"].median() for i in idx]))
            tt = time.time()
            est = fit_glmsingle_half(bolds, designs, stimdur, t_r, model, mask=mask_arr,
                                     smoothing_fwhm=DEFAULT_CONFIG.smoothing_fwhm)
            timings.append(time.time() - tt)
            _write_half_maps(out_dir, subject, model.task, args.space, h, est)
            halves.append(est)
        cfg_dict = {"smoothing_fwhm": DEFAULT_CONFIG.smoothing_fwhm, "stimdur": stimdur, "engine": "glmsingle TYPED"}
    else:
        est_name, noise = ENGINES[cell.engine]
        hrf_model = harness.HRF_LEVELS[cell.hrf]
        cfg = dataclasses.replace(
            DEFAULT_CONFIG, space=args.space, variant=args.variant, noise_model=noise,
            hrf_model="spm" if hrf_model == VOXELWISE else hrf_model, output_tree=OUTPUT_TREE,
        ).with_confounds(cell.confounds)
        estimator = get_estimator(est_name)
        hrfindex = None
        if hrf_model == VOXELWISE:
            hp = hrfindex_path(derivatives, subject, args.space)
            if not hp.exists():
                raise SystemExit(f"ERROR: {hp} missing; run `glm_bakeoff.py prep --subject sub-{subject}` first")
            hrfindex = load_hrfindex(hp, reference=mask_img)
        per_run: list[dict[str, ContrastEstimate]] = []
        for i, r in enumerate(runs):
            t_r = repetition_time(r, bids_root)
            bold = nib.load(str(r.bold))
            confounds = load_confounds(r)
            tt = time.time()
            if hrfindex is not None:
                est = fit_run_voxelwise(estimator, bold, adapted[i], confounds, t_r, model, cfg, hrfindex, mask_img)
            else:
                dm = build_design_matrix(adapted[i], confounds, t_r, bold.shape[-1], model, cfg)
                est = estimator.fit_run(bold, dm, contrast_vectors(model, list(dm.columns)),
                                        t_r=t_r, mask=mask_img, cfg=cfg)
            timings.append(time.time() - tt)
            per_run.append(est)
            print(f"  {r.entity_prefix}: {timings[-1]:.0f} s", flush=True)
        halves = []
        for h, idx in ((1, h1), (2, h2)):
            pooled = {c.name: _pool([per_run[i][c.name] for i in idx], mask_img) for c in model.contrasts}
            _write_half_maps(out_dir, subject, model.task, args.space, h, pooled)
            halves.append(pooled)
        cfg_dict = cfg.to_dict()

    scores = _score((halves[0], halves[1]), mask_arr, n_set)
    record = {
        "cell": cell.id, "subject": subject, "model": model.name, "task": model.task, "space": args.space,
        "contrasts": scores, "harness_sha256": frozen["sha256"], "config": cfg_dict,
        "runs": [r.entity_prefix for r in runs],
        "halves": [[runs[i].entity_prefix for i in h1], [runs[i].entity_prefix for i in h2]],
        "fit_seconds": [round(t, 1) for t in timings], "elapsed_s": round(time.time() - t0, 1),
    }
    harness.write_scores(out_dir / "scores.json", record)
    write_run_metadata(out_dir / "run_metadata.json", record)
    for name, s in scores.items():
        print(f"  {name}: r={s['r']:.3f} " + " ".join(f"{k}={v:.3f}" for k, v in s.items() if k.startswith("dice")))
    print(f"done in {time.time() - t0:.0f} s -> {out_dir}")
    return 0


def cmd_collect(args: argparse.Namespace) -> int:
    _, _, out_base = _paths(args)
    rows = harness.collect_scores(out_base)
    if not rows:
        raise SystemExit(f"ERROR: no scores.json under {out_base}; nothing fitted yet")
    df = pd.DataFrame(rows)
    path = out_base / "scores.tsv"
    df.to_csv(path, sep="\t", index=False)
    print(f"{len(df)} score rows from {df['cell'].nunique()} cells x subjects -> {path}")
    summary = (df[df["metric"].isin(["r"]) | df["metric"].str.startswith("dice")]
               .groupby(["model", "hrf", "confounds", "engine", "metric"])["value"].mean().unstack("metric"))
    with pd.option_context("display.width", 200, "display.max_rows", 500):
        print(summary.round(3))
    return 0


# ---------------------------------------------------------------------- main
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--space", default=DEFAULT_SPACE)
    p.add_argument("--variant", default=DEFAULT_VARIANT)
    p.add_argument("--sessions", nargs="*", default=None, help="override the model's session selection")
    p.add_argument("--bids-root", type=Path, default=None)
    p.add_argument("--derivatives-dir", type=Path, default=None)
    sub = p.add_subparsers(dest="verb", required=True)
    s = sub.add_parser("plan"); s.add_argument("--subjects", nargs="+", default=["sub-03", "sub-04", "sub-05"])
    s.add_argument("--models", nargs="*", default=None)
    s = sub.add_parser("prep"); s.add_argument("--subject", required=True)
    s = sub.add_parser("fit"); s.add_argument("--subject"); s.add_argument("--cell"); s.add_argument("--unit", type=int)
    s.add_argument("--force", action="store_true")
    sub.add_parser("collect")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return {"plan": cmd_plan, "prep": cmd_prep, "fit": cmd_fit, "collect": cmd_collect}[args.verb](args)


if __name__ == "__main__":
    sys.exit(main())
