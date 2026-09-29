#!/usr/bin/env python3
"""Cleaned grayordinate time series for the functional-space alignment routes.

One CIFTI-2 dtseries per run: fsaverage6 cortex + unfolded hippocampus (2 mm)
+ MNI subcortical voxels, residualised under one confound regime with the
data-quality library (``neuroimaging.confounds.regime_design`` +
``neuroimaging.data_quality.clean``; nothing re-implemented). Geometry and the
loader live in ``grayordinates.py``; the design record is mmmdata-agents
``docs/workbench/functional-space/``.

Verbs (idempotent; state is on disk):

  geometry  once per subject: hippocampal surfaces mapped into fMRIPrep T1w,
            checked against the aseg, and their 0.5 mm -> 2 mm weights; once
            per tree: the 2 mm unfold template, grayordinates.tsv and
            dataset_description.json
  plan      list the runs whose output is missing; --units writes the array
            units file and --manifest a dry-run manifest (one row per run with
            inputs, volumes and the expected output size)
  run       clean ONE run (a --units line, or --sub/--ses/--task/--run)
  collect   gather every sidecar into manifest.tsv at the tree root
  mni       MNI res-2 cortex (Schaefer-400) + HOSPA hippocampus voxels for the
            held-out film runs, cleaned the same way: the MNI voxel-identity
            baseline's test data (DECIDED 2026-09-29). Writes
            mni_voxels.tsv at the tree root once, then one dtseries per run

Usage:
    python clean_timeseries.py geometry --sub 03 --sub 04 --sub 05
    python clean_timeseries.py plan --units units.txt --manifest dryrun.tsv
    python clean_timeseries.py run --units units.txt --index 7
    python clean_timeseries.py collect
    python clean_timeseries.py mni                 # every held-out NATencoding run
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
for p in (REPO_ROOT / "src" / "python", HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import grayordinates as go  # noqa: E402
from core.config import load_config  # noqa: E402
from neuroimaging import data_quality as dq  # noqa: E402
from neuroimaging.confounds import get_regime, regime_design  # noqa: E402
from neuroimaging.io import FmriprepRun, find_fmriprep_runs, load_confounds  # noqa: E402

#: Tasks whose runs feed alignment (pre-registration §3.1): films and every rest run.
TASKS = ("NATencoding", "INITresting", "TBresting", "NATresting", "FINresting")
SUBJECTS = ("03", "04", "05")
DEFAULT_REGIME = "reference"
#: The MPRAGE hippunfold took as its T1w (its config: ses-01 acq-MPR run-01).
HIPPUNFOLD_T1W = {"ses": "01", "acq": "MPR", "run": "01"}


class Paths:
    def __init__(self, args: argparse.Namespace):
        cfg = load_config()["paths"]
        self.bids_root = Path(cfg["bids_project_dir"])
        self.derivatives = Path(cfg["output_dir"])
        self.containers = Path(cfg["singularity_dir"])
        self.atlases = self.derivatives / "atlases"
        self.fmriprep = self.derivatives / "fmriprep"
        self.hippunfold = self.derivatives / "hippunfold" / "hippunfold"
        self.envs = Path(cfg["stimfeat_env"]).parent
        self.root = Path(args.tree_root) if getattr(args, "tree_root", None) else go.tree_root(self.derivatives)


def find_runs(sub=None, ses=None, task=None, run=None) -> list[FmriprepRun]:
    tasks = [task] if task else TASKS
    subs = [sub] if sub else SUBJECTS
    out = []
    for s in subs:
        for t in tasks:
            out += find_fmriprep_runs(subject=s, session=ses, task=t, run=run, allow_mixed_designs=True)
    return [r for r in out if r.confounds is not None]


def t1w_bold(run: FmriprepRun) -> Path:
    return Path(run.confounds).parent / f"{run.entity_prefix}_space-T1w_desc-preproc_bold.nii.gz"


def check_inputs(run: FmriprepRun) -> list[str]:
    missing = []
    for label, p in (("MNI bold", run.bold), ("MNI mask", run.mask), ("fsaverage6 L", run.surface_L),
                     ("fsaverage6 R", run.surface_R), ("T1w bold", t1w_bold(run))):
        if p is None or not Path(p).exists():
            missing.append(label)
    return missing


def unit_line(run: FmriprepRun) -> str:
    return f"{run.subject}\t{run.session}\t{run.task}\t{run.run or ''}"


def run_from_unit(line: str) -> FmriprepRun:
    sub, ses, task, run = (line.rstrip("\n").split("\t") + [""])[:4]
    runs = [r for r in find_runs(sub, ses, task, run or None) if (r.run or "") == run]
    if len(runs) != 1:
        sys.exit(f"Unit {line.strip()!r} resolves to {len(runs)} runs; expected 1")
    return runs[0]


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------

def cmd_geometry(args: argparse.Namespace) -> None:
    paths = Paths(args)
    root = paths.root
    root.mkdir(parents=True, exist_ok=True)

    tpl = go.hipp_template_path(root)
    if not tpl.exists():
        subprocess.run(
            ["apptainer", "exec", "-B", str(root), str(paths.containers / go.HIPPUNFOLD_CONTAINER),
             "cp", f"{go.HIPPUNFOLD_TEMPLATE_DIR}/tpl-avg_space-unfold_den-{go.HIPP_DENSITY}_midthickness.surf.gii",
             str(tpl)],
            check=True,
        )
    tpl_coords, _ = go._surf_arrays(tpl)
    n_hipp = len(tpl_coords)
    print(f"hippocampal template: {tpl.name}, {n_hipp} vertices")

    axis = go.brain_model_axis(paths.atlases, n_hipp)
    table = go.grayordinate_table(axis)
    table.to_csv(go.grayordinates_path(root), sep="\t", index=False)
    print(f"grayordinates: {len(table)} "
          + ", ".join(f"{k}={v}" for k, v in table["piece"].value_counts().items()))
    ensure_description(root, paths)

    for sub in args.sub or SUBJECTS:
        facts: dict = {"subject": sub, "hippunfold_t1w": HIPPUNFOLD_T1W, "hemis": {}}
        xfm = (paths.fmriprep / f"sub-{sub}" / f"ses-{HIPPUNFOLD_T1W['ses']}" / "anat"
               / f"sub-{sub}_ses-{HIPPUNFOLD_T1W['ses']}_acq-{HIPPUNFOLD_T1W['acq']}_run-{HIPPUNFOLD_T1W['run']}"
                 "_from-orig_to-T1w_mode-image_xfm.txt")
        aseg = paths.fmriprep / f"sub-{sub}" / "anat" / f"sub-{sub}_acq-MPR_desc-aseg_dseg.nii.gz"
        surf_dir = paths.hippunfold / f"sub-{sub}" / "ses-01" / "surf"
        for p in (xfm, aseg, surf_dir):
            if not p.exists():
                sys.exit(f"sub-{sub}: {p} is missing")
        for hemi in go.HEMIS:
            stem = f"sub-{sub}_ses-01_hemi-{hemi}_space-T1w_den-{go.HIPP_SOURCE_DENSITY}_label-hipp"
            overlaps = {}
            for surf in ("inner", "midthickness", "outer"):
                coords, faces = go._surf_arrays(surf_dir / f"{stem}_{surf}.surf.gii")
                mapped = go.hippunfold_to_fmriprep_t1w(coords, xfm)
                overlaps[surf] = {
                    "identity": go.aseg_overlap(coords, aseg, go.ASEG_HIPPOCAMPUS[hemi]),
                    "mapped": go.aseg_overlap(mapped, aseg, go.ASEG_HIPPOCAMPUS[hemi]),
                }
                go.write_surface(mapped, faces, go.hipp_surface_path(root, sub, hemi, surf))
            mid = overlaps["midthickness"]["mapped"]
            if mid < go.MIN_ASEG_OVERLAP:
                sys.exit(f"sub-{sub} hemi-{hemi}: mapped midthickness overlaps the aseg hippocampus "
                         f"{mid:.2f} < {go.MIN_ASEG_OVERLAP}; refusing to build on this geometry")
            import nibabel as nib
            unfold, _ = go._surf_arrays(
                surf_dir / f"sub-{sub}_ses-01_hemi-{hemi}_space-unfold_den-{go.HIPP_SOURCE_DENSITY}_label-hipp_midthickness.surf.gii")
            area = np.asarray(nib.load(str(surf_dir / f"{stem}_surfarea.shape.gii")).agg_data(), dtype=float)
            w = go.hipp_weights(unfold, tpl_coords, area)
            w.to_csv(go.hipp_assignment_path(root, sub, hemi), sep="\t", index=False)
            per_target = w.groupby("vertex_target").size()
            facts["hemis"][hemi] = {
                "aseg_overlap": overlaps,
                "n_source": len(w), "n_target": n_hipp,
                "sources_per_target_min": int(per_target.min()),
                "sources_per_target_median": float(per_target.median()),
            }
            print(f"sub-{sub} hemi-{hemi}: aseg overlap identity {overlaps['midthickness']['identity']:.2f} "
                  f"-> mapped {mid:.2f}; sources per 2 mm vertex min {per_target.min()} "
                  f"median {per_target.median():.0f}")
        facts.update({
            "transform": str(xfm),
            "direction": "inverse of the ITK image transform (fMRIPrep T1w <- original MPRAGE), LPS<->RAS",
            "source_surfaces": str(surf_dir),
            "created": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        })
        (go.anat_dir(root, sub) / f"sub-{sub}_label-hipp_geometry.json").write_text(json.dumps(facts, indent=2) + "\n")


def ensure_description(root: Path, paths: Paths) -> None:
    desc = root / "dataset_description.json"
    if desc.exists():
        return
    desc.write_text(json.dumps({
        "Name": "Functional-space cleaned grayordinate time series (fsaverage6 cortex, unfolded hippocampus, MNI subcortex)",
        "BIDSVersion": "1.9.0",
        "DatasetType": "derivative",
        "GeneratedBy": [{
            "Name": "clean_timeseries.py",
            "Version": dq.code_version(REPO_ROOT),
            "CodeURL": "https://github.com/hulacon/mmmdata/tree/main/scripts/functional_space",
            "Description": (
                "Per-run residuals under one data-quality confound regime (regime_design + clean); "
                "full runs, non-steady-state volumes NaN, masked grayordinates NaN."
            ),
        }, {
            "Name": "Connectome Workbench wb_command",
            "Version": go.wb_version(go.wb_command_path(paths.envs)),
            "CodeURL": "conda-forge connectome-workbench, shared env " + go.WB_ENV,
            "Description": "ribbon-constrained volume-to-surface mapping of space-T1w BOLD onto the hippocampal ribbon",
        }],
        "SourceDatasets": [
            {"URL": "bids:derivatives/fmriprep", "Version": dq.pipeline_version(paths.fmriprep)},
            {"URL": "bids:derivatives/hippunfold", "Version": "1.5.2"},
            {"URL": "bids:derivatives/atlases"},
        ],
        "SchemaVersion": go.SCHEMA_VERSION,
    }, indent=2) + "\n")


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------

def cmd_plan(args: argparse.Namespace) -> None:
    paths = Paths(args)
    runs = find_runs(args.sub_one, args.ses, args.task, args.run)
    n_gray = len(pd.read_csv(go.grayordinates_path(paths.root), sep="\t")) if go.grayordinates_path(paths.root).exists() else None
    rows, todo = [], []
    for r in runs:
        out, _ = go.run_paths(paths.root, r, args.regime)
        n_vol = sum(1 for _ in open(r.confounds)) - 1
        missing = check_inputs(r)
        rows.append({
            "sub": r.subject, "ses": r.session, "task": r.task, "run": r.run or "",
            "n_vol": n_vol, "inputs_missing": ";".join(missing) or "",
            "exists": out.exists(),
            "expected_mb": round(n_vol * n_gray * 4 / 1e6, 1) if n_gray else None,
            "output": str(out),
        })
        if not out.exists() and not missing:
            todo.append(r)
    df = pd.DataFrame(rows)
    print(df.groupby(["sub", "task"]).agg(runs=("n_vol", "size"), vols=("n_vol", "sum"),
                                           done=("exists", "sum")).to_string())
    bad = df[df["inputs_missing"] != ""]
    if len(bad):
        print(f"\n{len(bad)} runs lack inputs:\n{bad[['sub', 'ses', 'task', 'run', 'inputs_missing']].to_string(index=False)}")
    if n_gray:
        print(f"\ngrayordinates {n_gray}; total volumes {df['n_vol'].sum()}; "
              f"expected size {df['expected_mb'].sum() / 1e3:.1f} GB; runs to build {len(todo)}")
    if args.manifest:
        df.to_csv(args.manifest, sep="\t", index=False)
        print(f"wrote dry-run manifest {args.manifest}")
    if args.units:
        Path(args.units).write_text("".join(unit_line(r) + "\n" for r in todo))
        print(f"wrote {len(todo)} units to {args.units}")


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

def load_surface(path: Path) -> np.ndarray:
    import nibabel as nib

    arr = np.asarray(nib.load(str(path)).agg_data(), dtype=np.float32)
    if arr.shape[0] != go.FSAVERAGE6_N:
        arr = arr.T
    if arr.shape[0] != go.FSAVERAGE6_N:
        raise ValueError(f"{path.name}: shape {arr.shape} is not fsaverage6")
    return np.ascontiguousarray(arr.T)  # (n_vol, n_vertices)


def clean_columns(raw: np.ndarray, reason: np.ndarray, design) -> tuple[np.ndarray, np.ndarray]:
    """Residualise the valid columns; the rest stay NaN. Returns ``(cleaned, reason)``.

    ``reason`` is 0 for a usable column; this adds 3 (non-finite or constant
    over the fitted volumes) to the codes the caller set.
    """
    reason = reason.copy()
    fit = ~design.nss
    finite = np.isfinite(raw[fit]).all(axis=0)
    varying = np.ptp(np.where(np.isfinite(raw[fit]), raw[fit], 0), axis=0) > 0
    reason[(reason == 0) & ~(finite & varying)] = 3  # 3 = non-finite or constant over the fitted volumes
    valid = reason == 0
    result = dq.clean(raw[:, valid], design)
    out = np.full(raw.shape, np.nan, dtype=np.float32)
    out[:, valid] = result.residuals
    return out, reason


def clean_one(run: FmriprepRun, paths: Paths, regime_name: str, scratch: Path | None) -> dict:
    import nibabel as nib

    t0 = time.time()
    root = paths.root
    missing = check_inputs(run)
    if missing:
        raise FileNotFoundError(f"{run.entity_prefix}: inputs missing: {missing}")
    table = pd.read_csv(go.grayordinates_path(root), sep="\t")
    tpl_n = int((table["piece"] == "hippocampus").sum() // 2)
    axis = go.brain_model_axis(paths.atlases, tpl_n)
    if len(axis) != len(table):
        raise ValueError("grayordinates.tsv does not match the atlas-derived brain-model axis; rerun geometry")

    regime = get_regime(regime_name)
    confounds = load_confounds(run)
    design = regime_design(regime, confounds)
    n_vol = design.n_vol

    blocks: dict[str, np.ndarray] = {}
    invalid_reason: dict[str, np.ndarray] = {}
    # cortex
    for hemi, p in (("L", run.surface_L), ("R", run.surface_R)):
        x = load_surface(Path(p))
        wall = go.medial_wall(paths.atlases, hemi)
        blocks[f"cortex_{hemi}"] = x
        invalid_reason[f"cortex_{hemi}"] = np.where(wall, 1, 0)  # 1 = medial wall
    # hippocampus
    wb = go.wb_command_path(paths.envs)
    hipp = go.sample_hippocampus(t1w_bold(run), root, run.subject, wb, scratch)
    for hemi in go.HEMIS:
        blocks[f"hipp_{hemi}"] = hipp[hemi]
        invalid_reason[f"hipp_{hemi}"] = np.zeros(hipp[hemi].shape[1], int)
    # subcortex
    bold = nib.load(str(run.bold))
    mask_img = nib.load(str(run.mask))
    masks, affine, shape = go.subcortical_masks(paths.atlases)
    if bold.shape[:3] != shape or not np.allclose(bold.affine, affine, atol=1e-3):
        raise ValueError(f"{run.entity_prefix}: MNI BOLD grid differs from the HOSPA atlas grid")
    brain = np.asarray(mask_img.dataobj).astype(bool)
    vols = np.asarray(bold.dataobj, dtype=np.float32)
    for structure, m in masks.items():
        key = f"sub_{structure}"
        blocks[key] = np.ascontiguousarray(vols[m].T)  # np.nonzero order == BrainModelAxis.from_mask order
        invalid_reason[key] = np.where(brain[m], 0, 2)  # 2 = outside the run's brain mask
    del vols

    order = ["cortex_L", "cortex_R", "hipp_L", "hipp_R"] + [f"sub_{s}" for s in masks]
    raw = np.concatenate([blocks[k] for k in order], axis=1)
    reason = np.concatenate([invalid_reason[k] for k in order])
    del blocks
    if raw.shape != (n_vol, len(table)):
        raise ValueError(f"{run.entity_prefix}: assembled {raw.shape}, expected ({n_vol}, {len(table)})")
    out, reason = clean_columns(raw, reason, design)
    valid = reason == 0
    del raw

    tr = float(bold.header.get_zooms()[3])
    nii, js = go.run_paths(root, run, regime.name)
    go.write_dtseries(out, axis, tr, nii)

    pieces = table["piece"].to_numpy()
    counts = {
        piece: {
            "n": int((pieces == piece).sum()),
            "n_valid": int(((pieces == piece) & valid).sum()),
            "medial_wall": int(((pieces == piece) & (reason == 1)).sum()),
            "outside_brain_mask": int(((pieces == piece) & (reason == 2)).sum()),
            "constant_or_nonfinite": int(((pieces == piece) & (reason == 3)).sum()),
        }
        for piece in ("cortex", "hippocampus", "subcortex")
    }
    record = {
        "schema_version": go.SCHEMA_VERSION,
        "sub": run.subject, "ses": run.session, "task": run.task, "run": run.run or "",
        "RepetitionTime": tr,
        "regime": regime.name, "regime_status": regime.status, "regime_version": regime.version,
        "regime_columns": list(design.columns.columns),
        "n_vol": n_vol, "n_nss": design.n_nss,
        "nss_volumes": [int(i) for i in np.flatnonzero(design.nss)],
        "n_regressors": design.n_regressors, "n_drift": design.n_drift, "dof_resid": design.dof_resid,
        "grayordinates": counts,
        "trimming": (
            "none: full run. Non-steady-state volumes are NaN rows. Film title/fixation "
            "trimming and film-boundary lag buffers are applied by the route loaders."
        ),
        "masking": (
            "NaN columns, never zero-filled: cortex medial wall (Schaefer-400 fsaverage6 label 0); "
            "subcortical voxels outside this run's MNI brain mask; any grayordinate non-finite or "
            "constant over the fitted volumes. Hippocampal 2 mm vertices average only varying "
            "0.5 mm ribbon samples."
        ),
        "inputs": {
            "surface_L": str(run.surface_L), "surface_R": str(run.surface_R),
            "bold_MNI": str(run.bold), "mask_MNI": str(run.mask), "bold_T1w": str(t1w_bold(run)),
            "confounds": str(run.confounds), "confounds_sha256": dq.file_sha256(Path(run.confounds)),
        },
        "fmriprep_version": dq.pipeline_version(paths.fmriprep),
        "code_version": dq.code_version(REPO_ROOT),
        "wb_command": {"path": str(wb), "version": go.wb_version(wb)},
        "elapsed_s": round(time.time() - t0, 1),
        "created": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    }
    js.write_text(json.dumps(record, indent=2) + "\n")
    return record


def cmd_run(args: argparse.Namespace) -> None:
    paths = Paths(args)
    if args.units:
        if args.index is None:
            sys.exit("--units needs --index (1-based, e.g. $SLURM_ARRAY_TASK_ID)")
        lines = [ln for ln in Path(args.units).read_text().splitlines() if ln.strip()]
        if not 1 <= args.index <= len(lines):
            sys.exit(f"--index {args.index} outside 1..{len(lines)}")
        runs = [run_from_unit(lines[args.index - 1])]
    else:
        if not (args.sub_one and args.ses and args.task):
            sys.exit("run needs --sub/--ses/--task[/--run] or --units/--index")
        runs = find_runs(args.sub_one, args.ses, args.task, args.run)
    scratch = Path(os.environ["TMPDIR"]) if os.environ.get("TMPDIR") else None
    for r in runs:
        nii, _ = go.run_paths(paths.root, r, args.regime)
        if nii.exists() and not args.force:
            print(f"{r.entity_prefix}: exists, skipping")
            continue
        rec = clean_one(r, paths, args.regime, scratch)
        g = rec["grayordinates"]
        print(f"{r.entity_prefix}: n_vol={rec['n_vol']} nss={rec['n_nss']} dof={rec['dof_resid']} "
              + " ".join(f"{k}={v['n_valid']}/{v['n']}" for k, v in g.items())
              + f" in {rec['elapsed_s']:.0f} s")


# ---------------------------------------------------------------------------
# collect
# ---------------------------------------------------------------------------

def cmd_collect(args: argparse.Namespace) -> None:
    paths = Paths(args)
    rows = []
    for js in sorted(paths.root.glob("sub-*/ses-*/func/*_space-fsaverage6_*_bold.json")):
        rec = json.loads(js.read_text())
        row = {k: rec[k] for k in ("sub", "ses", "task", "run", "regime", "regime_version", "n_vol",
                                   "n_nss", "dof_resid", "RepetitionTime", "fmriprep_version",
                                   "code_version", "created")}
        for piece, c in rec["grayordinates"].items():
            row[f"{piece}_valid"] = c["n_valid"]
        row["path"] = str(js.with_name(js.name.replace(".json", ".dtseries.nii")).relative_to(paths.root))
        rows.append(row)
    if not rows:
        sys.exit(f"No sidecars under {paths.root}")
    df = pd.DataFrame(rows)
    df.to_csv(paths.root / "manifest.tsv", sep="\t", index=False)
    print(f"{paths.root / 'manifest.tsv'}: {len(df)} runs")
    print(df.groupby(["sub", "task"]).size().to_string())


# ---------------------------------------------------------------------------
# mni: the MNI voxel-identity baseline's test data
# ---------------------------------------------------------------------------

#: The MNI baseline needs test data only: the held-out film sessions (pre-registration §3.3).
MNI_TASK = "NATencoding"
MNI_TEMPLATE = "MNI152NLin2009cAsym"
SCHAEFER_MNI_STEM = "tpl-MNI152NLin2009cAsym/anat/tpl-MNI152NLin2009cAsym_atlas-Schaefer2018_seg-{seg}_scale-400_res-2_dseg"
HOSPA_HIPPOCAMPUS = {"Left Hippocampus": "L", "Right Hippocampus": "R"}


def mni_voxels(atlases_dir: Path) -> tuple[pd.DataFrame, Any, np.ndarray, tuple[int, ...]]:
    """Voxel index of the MNI output, its CIFTI axis, affine and grid shape.

    Cortex = Schaefer-400 7n label > 0, split by hemisphere from the label
    names; hippocampus = the HOSPA hippocampi. A voxel in both is kept as
    hippocampus, so no voxel is counted twice.
    """
    import nibabel as nib
    from nibabel import cifti2

    labs, names = {}, {}
    for seg in ("7n", "17n"):
        stem = Path(atlases_dir) / SCHAEFER_MNI_STEM.format(seg=seg)
        img = nib.load(str(stem) + ".nii.gz")
        labs[seg] = np.asarray(img.dataobj).astype(int)
        names[seg] = pd.read_csv(str(stem) + ".tsv", sep="\t").set_index("index")["name"]
        affine, shape = img.affine, labs[seg].shape
    hospa = nib.load(str(Path(atlases_dir) / f"{go.HOSPA_STEM}.nii.gz"))
    if hospa.shape != shape or not np.allclose(hospa.affine, affine, atol=1e-3):
        raise ValueError("HOSPA and Schaefer MNI res-2 grids differ")
    hlab = np.asarray(hospa.dataobj).astype(int)
    htable = pd.read_csv(Path(atlases_dir) / f"{go.HOSPA_STEM}.tsv", sep="\t")
    hipp = {}
    for name, hemi in HOSPA_HIPPOCAMPUS.items():
        rows = htable.loc[htable["name"] == name, "index"]
        if len(rows) != 1:
            raise KeyError(f"HOSPA table has {len(rows)} rows named {name!r}")
        hipp[hemi] = hlab == int(rows.iloc[0])
    in_hipp = hipp["L"] | hipp["R"]
    lh = np.isin(labs["7n"], [i for i, n in names["7n"].items() if "_LH_" in n])
    rh = np.isin(labs["7n"], [i for i, n in names["7n"].items() if "_RH_" in n])
    masks = {
        go.CORTEX["L"]: lh & ~in_hipp, go.CORTEX["R"]: rh & ~in_hipp,
        go.HIPPOCAMPUS["L"]: hipp["L"], go.HIPPOCAMPUS["R"]: hipp["R"],
    }
    parts, rows = [], []
    for structure, m in masks.items():
        if not m.any():
            raise ValueError(f"MNI mask for {structure} is empty")
        parts.append(cifti2.BrainModelAxis.from_mask(m, name=structure, affine=affine))
        ijk = np.argwhere(m)  # C order == np.nonzero order == BrainModelAxis.from_mask order
        for i, j, k in ijk:
            l7, l17 = int(labs["7n"][i, j, k]), int(labs["17n"][i, j, k])
            rows.append({
                "structure": structure, "i": int(i), "j": int(j), "k": int(k),
                "schaefer7n": names["7n"].get(l7, "") if l7 else "",
                "schaefer17n": names["17n"].get(l17, "") if l17 else "",
            })
    axis = parts[0]
    for p in parts[1:]:
        axis = axis + p
    table = pd.DataFrame(rows)
    table.insert(0, "piece", np.where(table["structure"].str.contains("HIPPOCAMPUS"), "hippocampus", "cortex"))
    return table, axis, affine, shape


def mni_paths(root: Path, run: FmriprepRun, regime: str) -> tuple[Path, Path]:
    stem = (Path(root) / f"sub-{run.subject}" / f"ses-{run.session}" / "func"
            / f"{run.entity_prefix}_space-{MNI_TEMPLATE}_res-2_desc-{regime}_bold")
    return stem.with_name(stem.name + ".dtseries.nii"), stem.with_name(stem.name + ".json")


def clean_mni(run: FmriprepRun, paths: Paths, regime_name: str, table: pd.DataFrame, axis: Any,
              affine: np.ndarray, shape: tuple[int, ...]) -> dict:
    import nibabel as nib

    t0 = time.time()
    for label, p in (("MNI bold", run.bold), ("MNI mask", run.mask), ("confounds", run.confounds)):
        if p is None or not Path(p).exists():
            raise FileNotFoundError(f"{run.entity_prefix}: {label} missing")
    regime = get_regime(regime_name)
    design = regime_design(regime, load_confounds(run))
    bold = nib.load(str(run.bold))
    if bold.shape[:3] != shape or not np.allclose(bold.affine, affine, atol=1e-3):
        raise ValueError(f"{run.entity_prefix}: MNI BOLD grid differs from the atlas grid")
    brain = np.asarray(nib.load(str(run.mask)).dataobj).astype(bool)
    ijk = table[["i", "j", "k"]].to_numpy()
    vols = np.asarray(bold.dataobj, dtype=np.float32)
    raw = np.ascontiguousarray(vols[ijk[:, 0], ijk[:, 1], ijk[:, 2], :].T)
    del vols
    if raw.shape[0] != design.n_vol:
        raise ValueError(f"{run.entity_prefix}: {raw.shape[0]} volumes, design has {design.n_vol}")
    reason = np.where(brain[ijk[:, 0], ijk[:, 1], ijk[:, 2]], 0, 2)  # 2 = outside the run's brain mask
    out, reason = clean_columns(raw, reason, design)
    del raw
    tr = float(bold.header.get_zooms()[3])
    nii, js = mni_paths(paths.root, run, regime.name)
    nii.parent.mkdir(parents=True, exist_ok=True)
    go.write_dtseries(out, axis, tr, nii)
    pieces = table["piece"].to_numpy()
    record = {
        "schema_version": go.SCHEMA_VERSION,
        "sub": run.subject, "ses": run.session, "task": run.task, "run": run.run or "",
        "space": MNI_TEMPLATE, "res": "2", "RepetitionTime": tr,
        "regime": regime.name, "regime_status": regime.status, "regime_version": regime.version,
        "regime_columns": list(design.columns.columns),
        "n_vol": design.n_vol, "n_nss": design.n_nss,
        "nss_volumes": [int(i) for i in np.flatnonzero(design.nss)],
        "n_regressors": design.n_regressors, "n_drift": design.n_drift, "dof_resid": design.dof_resid,
        "voxels": {
            piece: {
                "n": int((pieces == piece).sum()),
                "n_valid": int(((pieces == piece) & (reason == 0)).sum()),
                "outside_brain_mask": int(((pieces == piece) & (reason == 2)).sum()),
                "constant_or_nonfinite": int(((pieces == piece) & (reason == 3)).sum()),
            }
            for piece in ("cortex", "hippocampus")
        },
        "voxel_index": "mni_voxels.tsv (tree root); column order = its row order",
        "purpose": "MNI voxel-identity baseline test data (functional-space pre-registration §6, §11.4)",
        "trimming": "none: full run. Non-steady-state volumes are NaN rows.",
        "masking": "NaN columns, never zero-filled: outside this run's MNI brain mask, or non-finite/constant.",
        "inputs": {"bold_MNI": str(run.bold), "mask_MNI": str(run.mask), "confounds": str(run.confounds),
                   "confounds_sha256": dq.file_sha256(Path(run.confounds))},
        "fmriprep_version": dq.pipeline_version(paths.fmriprep),
        "code_version": dq.code_version(REPO_ROOT),
        "elapsed_s": round(time.time() - t0, 1),
        "created": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    }
    js.write_text(json.dumps(record, indent=2) + "\n")
    return record


def cmd_mni(args: argparse.Namespace) -> None:
    from films import HELDOUT_SESSIONS

    paths = Paths(args)
    table, axis, affine, shape = mni_voxels(paths.atlases)
    index = paths.root / "mni_voxels.tsv"
    if index.exists():
        old = pd.read_csv(index, sep="\t", keep_default_na=False)
        if not old[["structure", "i", "j", "k"]].equals(table[["structure", "i", "j", "k"]]):
            sys.exit(f"{index} differs from the atlas-derived voxel set; the atlases changed. Move it aside and rerun.")
    else:
        table.to_csv(index, sep="\t", index=False)
    sessions = [args.ses] if args.ses else list(HELDOUT_SESSIONS)
    runs = [r for s in sessions for r in find_runs(args.sub_one, s, MNI_TASK, args.run)]
    if not runs:
        sys.exit(f"No {MNI_TASK} runs in sessions {sessions}")
    for r in runs:
        nii, _ = mni_paths(paths.root, r, args.regime)
        if nii.exists() and not args.force:
            print(f"{r.entity_prefix}: exists, skipping")
            continue
        rec = clean_mni(r, paths, args.regime, table, axis, affine, shape)
        print(f"{r.entity_prefix}: n_vol={rec['n_vol']} nss={rec['n_nss']} "
              + " ".join(f"{k}={v['n_valid']}/{v['n']}" for k, v in rec["voxels"].items())
              + f" in {rec['elapsed_s']:.0f} s")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tree-root", help="override the output tree (default <derivatives>/functional_space/cleaned_timeseries)")
    sub = ap.add_subparsers(dest="verb", required=True)
    g = sub.add_parser("geometry")
    g.add_argument("--sub", action="append", help="subject label without 'sub-' (repeatable)")
    for name in ("plan", "run"):
        p = sub.add_parser(name)
        p.add_argument("--sub", dest="sub_one")
        p.add_argument("--ses")
        p.add_argument("--task")
        p.add_argument("--run")
        p.add_argument("--regime", default=DEFAULT_REGIME)
        if name == "plan":
            p.add_argument("--units")
            p.add_argument("--manifest")
        else:
            p.add_argument("--units")
            p.add_argument("--index", type=int)
            p.add_argument("--force", action="store_true")
    sub.add_parser("collect")
    m = sub.add_parser("mni")
    m.add_argument("--sub", dest="sub_one")
    m.add_argument("--ses", help="default: the held-out film sessions")
    m.add_argument("--run")
    m.add_argument("--regime", default=DEFAULT_REGIME)
    m.add_argument("--force", action="store_true")
    args = ap.parse_args()
    {"geometry": cmd_geometry, "plan": cmd_plan, "run": cmd_run, "collect": cmd_collect,
     "mni": cmd_mni}[args.verb](args)


if __name__ == "__main__":
    main()
