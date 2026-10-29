"""Candidate BOLD->anatomy alignments, compared with fMRIPrep's per-run BBR (transforms only).

fMRIPrep registers every run to the T1w separately (bbregister, 6 dof). This
module estimates two alternatives per subject and measures how far each would
move every run's voxels relative to the current outputs. It resamples no BOLD
series. Design record: mmmdata-agents ``docs/workbench/alignment-strategy/``.

**Arms** (A = the current per-run BBR):

* **B, unbiased EPI template.** Session means of the runs' boldrefs, an
  iterated median template over sessions (rigid, 6 dof), one bbregister of the
  template to the T1w, then each run rigid to the template.
* **C, NSD-style.** Each run rigid to its session mean; each session mean
  affine (12 dof, global correlation) to a reference session's mean; one
  functional reference (the reference session plus the next sessions, aligned
  to it) registered to the T2w with ANTs BSplineSyN; the T2w rigid to the T1w.

**Everything is estimated in T1w world coordinates, from A's own T1w boldrefs.**
A is rigid, so re-registering an A-placed boldref gives a correction that,
composed with A, is the transform an arm would estimate from the native boldref
(up to one extra interpolation of the input, which moves an estimate far less
than the effects measured here). The difference from A is therefore the
correction alone: for a T1w point ``p``, arm B samples the A-placed run at
``D_r(T(p))`` and arm C at ``D_r(S_s(W(p)))``, so the displacement is the distance
from ``p``. Every transform here is a fixed->moving point map in LPS, the ITK
convention, so chains compose left to right in the order written.

Outputs under ``<out>/sub-##/`` (one cell per subject)::

    sub-##_desc-stage0_deltas.tsv     one row per run: arm B and C displacements and their parts
    sub-##_desc-stage0_deltas.json    provenance, shared components, settings
    xfm/                              every estimated transform (ANTs .mat, LTA, warp)
    sub-##_desc-{templateB,funcrefC,t2wInT1w}_*.nii.gz   the images registered against
"""

from __future__ import annotations

import collections
import datetime as _dt
import json
import os
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np
import pandas as pd

from . import data_quality_alignment as dqa
from . import data_quality_voxelmaps as dqv
from .io import FmriprepRun

SCHEMA_VERSION = "1.0"
#: Rounds of register-to-median for arm B's template (round 0 is the unaligned median).
TEMPLATE_ROUNDS = 2
#: Sessions in arm C's functional reference: the reference session and the next ones (NSD used five).
FUNCREF_SESSIONS = 5
#: Brain-mask voxels are sampled on every Nth voxel of the T1w grid per axis.
POINT_STRIDE = 2
RANDOM_SEED = dqa.RANDOM_SEED

_GC_STAGE = ["--convergence", "[200x100x50,1e-7,10]", "--shrink-factors", "4x2x1", "--smoothing-sigmas", "2x1x0vox"]
#: Arm C's session step: rigid then affine, global correlation, dense.
ANTS_AFFINE_GC = [
    "--dimensionality", "3", "--float", "1", "--collapse-output-transforms", "1",
    "--interpolation", "Linear", "--use-histogram-matching", "0", "--winsorize-image-intensities", "[0.005,0.995]",
    "--random-seed", str(RANDOM_SEED),
    "--transform", "Rigid[0.1]", "--metric", "GC[{fixed},{moving},1,1,None]", *_GC_STAGE,
    "--transform", "Affine[0.1]", "--metric", "GC[{fixed},{moving},1,1,None]", *_GC_STAGE,
]
#: Arm C's functional -> T2w link: NSD's BSplineSyN [0.1, 400, 0, 3] after a rigid stage.
ANTS_SYN_T2W = [
    "--dimensionality", "3", "--float", "1", "--collapse-output-transforms", "1",
    "--interpolation", "Linear", "--use-histogram-matching", "0", "--winsorize-image-intensities", "[0.005,0.995]",
    "--random-seed", str(RANDOM_SEED),
    "--transform", "Rigid[0.1]", "--metric", "MI[{fixed},{moving},1,32,Regular,0.25]",
    "--convergence", "[1000x500x250,1e-7,10]", "--shrink-factors", "4x2x1", "--smoothing-sigmas", "2x1x0vox",
    "--transform", "BSplineSyN[0.1,400,0,3]", "--metric", "MI[{fixed},{moving},1,32]",
    "--convergence", "[100x70x50x20,1e-6,10]", "--shrink-factors", "8x4x2x1", "--smoothing-sigmas", "3x2x1x0vox",
]
#: T2w -> T1w, same session, same scanner frame: rigid from identity.
ANTS_RIGID_MI = [*dqa.ANTS_RIGID, "--metric", "MI[{fixed},{moving},1,32,Regular,0.25]"]


# ---------------------------------------------------------------------------
# Pure helpers (tested on synthetic data)
# ---------------------------------------------------------------------------

def read_itk_txt(path: Path) -> np.ndarray:
    """4x4 LPS matrix of a one-transform ITK text file (affine, 12 params + centre)."""
    text = Path(path).read_text()
    kinds = re.findall(r"^Transform:\s*(\S+)", text, flags=re.M)
    if len(kinds) != 1 or not re.match(r"(Affine|MatrixOffset)Transform", kinds[0]):
        raise ValueError(f"{path}: expected one affine transform, found {kinds}")
    params = np.array(re.search(r"^Parameters:(.*)$", text, flags=re.M).group(1).split(), dtype=np.float64)
    fixed = np.array(re.search(r"^FixedParameters:(.*)$", text, flags=re.M).group(1).split(), dtype=np.float64)
    return dqa.itk_affine_to_matrix(params, fixed)


def read_ants_mat(path: Path) -> np.ndarray:
    """4x4 LPS matrix of an ANTs ``*GenericAffine.mat``."""
    from scipy.io import loadmat

    mat = loadmat(str(path))
    key = next(k for k in mat if k.startswith("AffineTransform") or k.startswith("MatrixOffsetTransform"))
    return dqa.itk_affine_to_matrix(mat[key], mat["fixed"])


def write_itk_txt(M: np.ndarray, path: Path) -> Path:
    """Write a 4x4 LPS point map as a one-transform ITK text file (zero centre), as fMRIPrep writes its xfms."""
    M = np.asarray(M, dtype=np.float64)
    params = " ".join(f"{v:.10g}" for v in [*M[:3, :3].ravel(), *M[:3, 3]])
    Path(path).write_text("#Insight Transform File V1.0\n#Transform 0\nTransform: AffineTransform_double_3_3\n"
                          f"Parameters: {params}\nFixedParameters: 0 0 0\n")
    return Path(path)


def arm_b_coreg(coreg_a: np.ndarray, run_to_template: np.ndarray, template_to_t1w: np.ndarray) -> np.ndarray:
    """Arm B's boldref->T1w transform (T1w point -> native boldref point) from A's and the stage 0 corrections.

    ``template_to_t1w`` maps a T1w point to the B template (bbregister), ``run_to_template`` maps a template
    point to the A-placed run (the run's rigid registration to the template), and ``coreg_a`` (fMRIPrep's
    ``from-boldref_to-T1w`` xfm) maps a T1w point to the native boldref. The point maps chain in that order.
    """
    return coreg_a @ run_to_template @ template_to_t1w


def apply_affine(M: np.ndarray, points: np.ndarray) -> np.ndarray:
    p = np.asarray(points, dtype=np.float64)
    return (M @ np.c_[p, np.ones(len(p))].T).T[:, :3]


def summarise(moved: np.ndarray, points: np.ndarray, prefix: str) -> dict[str, float]:
    d = np.linalg.norm(np.asarray(moved) - np.asarray(points), axis=1)
    return {f"{prefix}_mean_mm": float(d.mean()), f"{prefix}_max_mm": float(d.max())}


def mask_points_lps(mask: np.ndarray, affine: np.ndarray, stride: int = POINT_STRIDE) -> tuple[np.ndarray, np.ndarray]:
    """World points (LPS) of mask voxels on every ``stride``-th voxel, and their voxel indices."""
    sub = np.zeros(mask.shape, bool)
    sub[::stride, ::stride, ::stride] = True
    ijk = np.argwhere(np.asarray(mask, bool) & sub)
    world = apply_affine(affine, ijk)
    return dqa.ras_to_lps(world), ijk


def sample_field_at_voxels(field: np.ndarray, ijk: np.ndarray) -> np.ndarray:
    """Displacement vectors of an ITK displacement field (x, y, z, [1,] 3) at integer voxels."""
    f = np.asarray(field)
    if f.ndim == 5:
        f = f[:, :, :, 0, :]
    return f[ijk[:, 0], ijk[:, 1], ijk[:, 2], :].astype(np.float64)


def reference_and_followers(sessions: Sequence[str], reference: str, n: int = FUNCREF_SESSIONS) -> list[str]:
    """The reference session and the sessions after it, ``n`` in all, in label order."""
    ordered = sorted(sessions)
    if reference not in ordered:
        raise ValueError(f"reference session {reference!r} not among {ordered}")
    picked = ordered[ordered.index(reference):][:n]
    if len(picked) < n:
        raise ValueError(f"only {len(picked)} sessions from {reference!r} on; arm C's reference needs {n}")
    return picked


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

def _exe(name: str, hint: str) -> str:
    exe = shutil.which(name)
    if exe is None:
        raise RuntimeError(f"{name} is not on PATH; {hint}")
    return exe


def _run(cmd: list[str], what: str, threads: int = 1, extra_env: Optional[dict] = None) -> None:
    env = {**os.environ, "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS": str(threads), **(extra_env or {})}
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"{what} failed ({cmd[0]}): {proc.stderr[-2000:] or proc.stdout[-2000:]}")


def ants_register(fixed: Path, moving: Path, prefix: Path, stages: list[str], fixed_mask: Optional[Path] = None,
                  threads: int = 1) -> Path:
    """Run ``antsRegistration`` with ``stages`` ({fixed}/{moving} filled in); returns the output prefix."""
    args = [s.format(fixed=fixed, moving=moving) for s in stages]
    cmd = [_exe("antsRegistration", "`module load ants/2.5.2` BEFORE activating the venv"), *args,
           "--output", f"[{prefix}]"]
    if fixed_mask is not None:
        cmd += ["--masks", f"[{fixed_mask},NULL]"]
    _run(cmd, f"antsRegistration {Path(moving).name} -> {Path(fixed).name}", threads)
    return prefix


def ants_apply(moving: Path, reference: Path, transforms: list[Path], out: Path, field: bool = False) -> Path:
    """Resample ``moving`` onto ``reference`` (or, ``field=True``, write the composite displacement field)."""
    cmd = [_exe("antsApplyTransforms", "`module load ants/2.5.2`"), "-d", "3", "-r", str(reference),
           "-o", f"[{out},1]" if field else str(out)]
    if not field:
        cmd += ["-i", str(moving), "-n", "Linear"]
    for t in transforms:
        cmd += ["-t", str(t)]
    _run(cmd, f"antsApplyTransforms -> {Path(out).name}")
    return out


def bbregister_to_t1w(moving: Path, subjects_dir: Path, subject: str, fsnative_to_t1w: Path, workdir: Path,
                      dof: int = 6) -> tuple[np.ndarray, float]:
    """bbregister an EPI image (in T1w world) to the subject's FreeSurfer anatomy; returns T1w->moving (LPS) + cost.

    bbregister gives moving->fsnative as an LTA; ``lta_convert --outitk`` writes it as the fsnative->moving point
    map. fMRIPrep's ``from-fsnative_to-T1w`` image transform is the T1w->fsnative point map. Their product maps a
    T1w point to the moving image.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    lta, itk = workdir / "bbreg.lta", workdir / "bbreg_itk.txt"
    # The freesurfer/8.2.0 module runs in apptainer, whose FreeSurfer setup resets SUBJECTS_DIR; APPTAINERENV_*
    # overrides it inside the container. The caller must also bind /gpfs (APPTAINER_BINDPATH).
    env = {"SUBJECTS_DIR": str(subjects_dir), "APPTAINERENV_SUBJECTS_DIR": str(subjects_dir)}
    if os.environ.get("FS_LICENSE"):
        env["APPTAINERENV_FS_LICENSE"] = os.environ["FS_LICENSE"]
    _run([_exe("bbregister", "`module load freesurfer/8.2.0`"), "--s", f"sub-{subject}", "--mov", str(moving),
          "--bold", f"--{dof}", "--init-coreg", "--lta", str(lta)], f"bbregister {Path(moving).name}",
         extra_env=env)
    _run([_exe("lta_convert", "`module load freesurfer/8.2.0`"), "--inlta", str(lta), "--outitk", str(itk)],
         "lta_convert", extra_env=env)
    costs = sorted(workdir.glob("*.mincost"))
    if len(costs) != 1:
        raise FileNotFoundError(f"expected one bbregister .mincost in {workdir}, found {costs}")
    cost = float(costs[0].read_text().split()[0])
    return bbreg_t1w_map(itk, fsnative_to_t1w), cost


def bbreg_t1w_map(bbreg_itk: Path, fsnative_to_t1w: Path) -> np.ndarray:
    """T1w point -> moving-image point, from bbregister's ITK export and fMRIPrep's fsnative->T1w xfm."""
    return read_itk_txt(bbreg_itk) @ read_itk_txt(fsnative_to_t1w)


# ---------------------------------------------------------------------------
# Cell
# ---------------------------------------------------------------------------

def _anat(fmriprep_tree: Path, subject: str) -> dict[str, Path]:
    sub_dir = Path(fmriprep_tree) / f"sub-{subject}"
    return {"t1w": dqv._anat_one(sub_dir, subject, "_desc-preproc_T1w.nii.gz"),
            "brain_mask": dqv._anat_one(sub_dir, subject, "_desc-brain_mask.nii.gz"),
            "fsnative_to_t1w": dqv._anat_one(sub_dir, subject, "_from-fsnative_to-T1w_mode-image_xfm.txt")}


def find_t2w(bids_root: Path, subject: str, acq: str) -> Path:
    hits = sorted(Path(bids_root).glob(f"sub-{subject}/ses-*/anat/sub-{subject}_*acq-{acq}_*T2w.nii.gz"))
    if len(hits) != 1:
        raise FileNotFoundError(f"expected one acq-{acq} T2w for sub-{subject}, found {[str(h) for h in hits]}")
    return hits[0]


def cell_paths(out_root: Path, subject: str) -> dict[str, Path]:
    d = Path(out_root) / f"sub-{subject}"
    return {"dir": d, "xfm": d / "xfm", "table": d / f"sub-{subject}_desc-stage0_deltas.tsv",
            "sidecar": d / f"sub-{subject}_desc-stage0_deltas.json",
            "templateB": d / f"sub-{subject}_desc-templateB_boldref.nii.gz",
            "funcrefC": d / f"sub-{subject}_desc-funcrefC_boldref.nii.gz",
            "t2w_in_t1w": d / f"sub-{subject}_desc-t2wInT1w_T2w.nii.gz",
            "check": d / "check"}


#: fMRIPrep's per-run boldref->T1w transform, the one file an arm tree replaces.
COREG_SUFFIX = "_from-boldref_to-T1w_mode-image_desc-coreg_xfm.txt"
#: Production files that stay out of an arm tree (reports; FreeSurfer is passed with --fs-subjects-dir).
ARM_TREE_SKIP = ("figures", "log", "logs", "sourcedata")
#: Largest |arm-tree displacement - stage 0 displacement| accepted, mm (the stage 0 table has 4 decimals).
ARM_TREE_TOL_MM = 1e-3


def build_arm_tree(fmriprep_tree: Path, stage0_root: Path, subject: str, arm: str, sessions: Sequence[str],
                   tasks: Sequence[str], out_tree: Path, log: Callable[[str], None] = print) -> pd.DataFrame:
    """An fMRIPrep derivatives tree that differs from production only in the coreg xfm of the chosen runs.

    Every production file of the subject's anatomy and of ``sessions`` is symlinked; for runs of ``tasks``
    the ``from-boldref_to-T1w`` xfm is replaced by the arm's (arm ``A`` replaces nothing: a control tree).
    fMRIPrep given this tree with ``--derivatives`` reuses every fit and resamples once with the arm's
    placement. Each written xfm's displacement from A over the T1w brain mask must reproduce the stage 0
    table's, or this raises. Returns one row per replaced (or, for A, matched) run.
    """
    import nibabel as nib

    if arm not in ("A", "B"):
        raise ValueError(f"arm must be A (control) or B; got {arm!r}")
    fmriprep_tree, out_tree = Path(fmriprep_tree), Path(out_tree)
    if out_tree.resolve() == fmriprep_tree.resolve():
        raise ValueError("the arm tree must not be the production tree")
    sub_src, sub_dst = fmriprep_tree / f"sub-{subject}", out_tree / f"sub-{subject}"

    def link_tree(src: Path, dst: Path) -> None:
        for f in sorted(src.rglob("*")):
            if f.is_file():
                d = dst / f.relative_to(src)
                d.parent.mkdir(parents=True, exist_ok=True)
                if d.is_symlink() or d.exists():
                    d.unlink()
                d.symlink_to(f)

    out_tree.mkdir(parents=True, exist_ok=True)
    for f in fmriprep_tree.iterdir():
        if f.is_file() and f.suffix in (".json", ".tsv"):
            (out_tree / f.name).unlink(missing_ok=True)
            (out_tree / f.name).symlink_to(f)
    for d in sub_src.iterdir():
        if d.is_dir() and d.name not in ARM_TREE_SKIP and not d.name.startswith("ses-"):
            link_tree(d, sub_dst / d.name)
    for s in sessions:
        if not (sub_src / f"ses-{s}").is_dir():
            raise FileNotFoundError(f"no production session {sub_src / f'ses-{s}'}")
        link_tree(sub_src / f"ses-{s}", sub_dst / f"ses-{s}")

    anat = _anat(fmriprep_tree, subject)
    bm = nib.load(str(anat["brain_mask"]))
    pts, _ = mask_points_lps(np.asarray(bm.dataobj) > 0, bm.affine)
    paths = cell_paths(stage0_root, subject)
    stage0 = pd.read_csv(paths["table"], sep="\t", dtype=str)
    T_B = bbreg_t1w_map(paths["xfm"] / "B_bbreg" / "bbreg_itk.txt", anat["fsnative_to_t1w"]) if arm == "B" else None
    rows = []
    for s in sessions:
        for coreg in sorted((sub_dst / f"ses-{s}" / "func").glob(f"sub-{subject}_ses-{s}_*{COREG_SUFFIX}")):
            prefix = coreg.name[: -len(COREG_SUFFIX)]
            ent = dict(kv.split("-", 1) for kv in prefix.split("_"))
            if ent["task"] not in tasks:
                continue
            C_A = read_itk_txt(coreg.resolve())
            if arm == "B":
                D = read_ants_mat(paths["xfm"] / f"B_run_{prefix}_0GenericAffine.mat")
                M = arm_b_coreg(C_A, D, T_B)
                coreg.unlink()
                write_itk_txt(M, coreg)
            else:
                M = C_A
            disp = summarise(apply_affine(M, pts), apply_affine(C_A, pts), "disp")
            hit = stage0[(stage0.ses == s) & (stage0.task == ent["task"]) & (stage0.run == ent.get("run", "n/a"))]
            ref = float(hit["B_mean_mm"].iloc[0]) if arm == "B" and len(hit) == 1 else 0.0
            if abs(disp["disp_mean_mm"] - ref) > ARM_TREE_TOL_MM:
                raise ValueError(f"{prefix}: arm {arm} xfm moves the brain {disp['disp_mean_mm']:.4f} mm from A, "
                                 f"stage 0 says {ref:.4f}")
            rows.append({"ses": s, "task": ent["task"], "run": ent.get("run", "n/a"), **disp,
                         "stage0_mean_mm": ref, "coreg": str(coreg)})
    if not rows:
        raise ValueError(f"sub-{subject}: no {COREG_SUFFIX} for tasks {list(tasks)} in sessions {list(sessions)}")
    log(f"sub-{subject}: arm {arm} tree at {out_tree}: {len(rows)} runs, displacement from A "
        f"median {np.median([r['disp_mean_mm'] for r in rows]):.3f} mm")
    return pd.DataFrame(rows)


def build_cell(out_root: Path, fmriprep_tree: Path, bids_root: Path, subject: str, runs: list[FmriprepRun],
               reference_session: str, provenance: dict, t2w_acq: str = "SPC", n_jobs: int = 1,
               check_run: Optional[str] = None, arms: Sequence[str] = ("B", "C"),
               log: Callable[[str], None] = print) -> dict:
    """Estimate arms B and/or C for one subject and write the per-run displacement table.

    Arm B also gets a 9-dof bbregister of its template (a diagnostic, not an arm): the global EPI-vs-anatomy
    scale that the WM-boundary criterion prefers, which a 6-dof registration cannot express.
    """
    import nibabel as nib

    paths = cell_paths(out_root, subject)
    for k in ("dir", "xfm", "check"):
        paths[k].mkdir(parents=True, exist_ok=True)
    xfm = paths["xfm"]
    anat = _anat(fmriprep_tree, subject)
    unknown = set(arms) - {"B", "C"}
    if unknown or not arms:
        raise ValueError(f"arms must be a non-empty subset of B, C; got {list(arms)}")
    t2w = find_t2w(bids_root, subject, t2w_acq) if "C" in arms else None
    subjects_dir = Path(fmriprep_tree) / "sourcedata" / "freesurfer"

    shape, ref_affine, off_grid = dqv.reference_grid(runs)
    order = sorted(runs, key=lambda r: (r.session, r.task, r.run or ""))
    sessions: dict[str, list[FmriprepRun]] = collections.defaultdict(list)
    for r in order:
        sessions[r.session].append(r)
    if reference_session not in sessions:
        raise ValueError(f"sub-{subject}: reference session {reference_session!r} has no runs")

    # Inputs on one grid: whole boldrefs (skull kept, as in the alignment cell), scaled by their brain median.
    norm: dict[str, Path] = {}
    norm_arr: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}
    img_dir = paths["dir"] / "inputs"
    for r in order:
        img = nib.load(str(r.boldref))
        m = np.asarray(nib.load(str(r.mask)).dataobj) > 0
        v = np.asarray(img.dataobj, dtype=np.float32)
        v = v / dqv.run_scale(v, m)
        arr = dqv.onto_grid(v, np.ones(v.shape, bool), img.affine, shape, ref_affine)[0]
        masks[r.entity_prefix] = dqv.onto_grid(m.astype(np.float32), m, img.affine, shape, ref_affine)[1]
        norm_arr[r.entity_prefix] = arr
        norm[r.entity_prefix] = img_dir / f"{r.entity_prefix}_norm.nii.gz"
        dqv.save_volume(np.nan_to_num(arr), ref_affine, norm[r.entity_prefix])

    def save(arr: np.ndarray, path: Path) -> Path:
        dqv.save_volume(np.nan_to_num(arr), ref_affine, path)
        return path

    ses_mean: dict[str, Path] = {}
    ses_mask: dict[str, Path] = {}
    for s, rs in sessions.items():
        ses_mean[s] = save(dqv.nanmedian_stack([norm_arr[r.entity_prefix] for r in rs]), img_dir / f"ses-{s}_mean.nii.gz")
        frac = np.mean([masks[r.entity_prefix] for r in rs], axis=0)
        ses_mask[s] = save((frac >= dqa.MASK_MAJORITY).astype(np.float32), img_dir / f"ses-{s}_mask.nii.gz")
    all_mask = save((np.mean([nib.load(str(p)).get_fdata() for p in ses_mask.values()], axis=0)
                     >= dqa.MASK_MAJORITY).astype(np.float32), img_dir / "sessions_mask.nii.gz")

    def pmap(fn, items):
        with ThreadPoolExecutor(max_workers=max(1, n_jobs)) as pool:
            return list(pool.map(fn, items))

    def run_to(fixed: Path, fixed_mask: Path, r: FmriprepRun, tag: str) -> np.ndarray:
        pre = ants_register(fixed, norm[r.entity_prefix], xfm / f"{tag}_{r.entity_prefix}_",
                            [*dqa.ANTS_RIGID, "--metric", f"MI[{fixed},{norm[r.entity_prefix]},1,32,None]"],
                            fixed_mask=fixed_mask)
        return read_ants_mat(Path(f"{pre}0GenericAffine.mat"))

    if "B" in arms:
        # ---- Arm B: iterated median template, one bbregister, runs rigid to it.
        template = save(dqv.nanmedian_stack([nib.load(str(p)).get_fdata() for p in ses_mean.values()]),
                        img_dir / "templateB_round0.nii.gz")
        for k in range(1, TEMPLATE_ROUNDS + 1):
            def to_template(s: str, template=template, k=k) -> Path:
                pre = ants_register(template, ses_mean[s], xfm / f"B_round{k}_ses-{s}_", [*dqa.ANTS_RIGID,
                                    "--metric", f"MI[{template},{ses_mean[s]},1,32,None]"], fixed_mask=all_mask)
                return ants_apply(ses_mean[s], template, [Path(f"{pre}0GenericAffine.mat")],
                                  img_dir / f"B_round{k}_ses-{s}.nii.gz")
            moved = pmap(to_template, sorted(sessions))
            template = save(dqv.nanmedian_stack([nib.load(str(p)).get_fdata() for p in moved]),
                            img_dir / f"templateB_round{k}.nii.gz")
        shutil.copyfile(template, paths["templateB"])
        log(f"sub-{subject}: arm B template built ({TEMPLATE_ROUNDS} rounds, {len(sessions)} sessions)")
        T_B, cost_B = bbregister_to_t1w(paths["templateB"], subjects_dir, subject, anat["fsnative_to_t1w"], xfm / "B_bbreg")
        log(f"sub-{subject}: arm B template -> T1w bbregister cost {cost_B:.3f}")
        T_B9, cost_B9 = bbregister_to_t1w(paths["templateB"], subjects_dir, subject, anat["fsnative_to_t1w"],
                                          xfm / "B_bbreg9", dof=9)
        log(f"sub-{subject}: arm B template 9-dof diagnostic: cost {cost_B9:.3f}, scale % "
            f"{np.round((np.diag(T_B9[:3, :3]) - 1) * 100, 2)}")

        D_B = dict(zip([r.entity_prefix for r in order], pmap(lambda r: run_to(paths["templateB"], all_mask, r, "B_run"),
                                                              order)))
        log(f"sub-{subject}: arm B runs registered ({len(order)})")

    funcref_ses: list[str] = []
    if "C" in arms:
        # ---- Arm C: runs rigid to their session mean; session means affine to the reference; funcref -> T2w -> T1w.
        D_C = dict(zip([r.entity_prefix for r in order],
                       pmap(lambda r: run_to(ses_mean[r.session], ses_mask[r.session], r, "C_run"), order)))
        ref_mean = ses_mean[reference_session]

        def ses_to_ref(s: str) -> tuple[np.ndarray, Path]:
            if s == reference_session:
                return np.eye(4), ref_mean
            pre = ants_register(ref_mean, ses_mean[s], xfm / f"C_ses-{s}_", ANTS_AFFINE_GC, fixed_mask=ses_mask[reference_session])
            mat = Path(f"{pre}0GenericAffine.mat")
            return read_ants_mat(mat), ants_apply(ses_mean[s], ref_mean, [mat], img_dir / f"C_ses-{s}_in_ref.nii.gz")
        S_C = dict(zip(sorted(sessions), pmap(ses_to_ref, sorted(sessions))))
        funcref_ses = reference_and_followers(list(sessions), reference_session)
        funcref = save(dqv.nanmedian_stack([nib.load(str(S_C[s][1])).get_fdata() for s in funcref_ses]), paths["funcrefC"])
        log(f"sub-{subject}: arm C sessions registered; funcref from ses {funcref_ses}")

        pre_t2 = ants_register(anat["t1w"], t2w, xfm / "C_t2w_to_t1w_", ANTS_RIGID_MI, fixed_mask=anat["brain_mask"],
                               threads=max(1, n_jobs))
        ants_apply(t2w, anat["t1w"], [Path(f"{pre_t2}0GenericAffine.mat")], paths["t2w_in_t1w"])
        pre_w = ants_register(paths["t2w_in_t1w"], funcref, xfm / "C_funcref_to_t2w_", ANTS_SYN_T2W,
                              fixed_mask=anat["brain_mask"], threads=max(1, n_jobs))
        w_field = ants_apply(funcref, paths["t2w_in_t1w"], [Path(f"{pre_w}1Warp.nii.gz"), Path(f"{pre_w}0GenericAffine.mat")],
                             xfm / "C_funcref_to_t2w_composite.nii.gz", field=True)
        log(f"sub-{subject}: arm C funcref -> T2w (BSplineSyN) done")

    # ---- Displacements at T1w brain-mask points.
    bm_img = nib.load(str(anat["brain_mask"]))
    pts, ijk = mask_points_lps(np.asarray(bm_img.dataobj) > 0, bm_img.affine)
    shared: dict[str, Any] = {}
    if "B" in arms:
        T_pts = apply_affine(T_B, pts)
        shared.update({**summarise(T_pts, pts, "B_shared"), "B_bbregister_cost": cost_B,
                       **summarise(apply_affine(T_B9, pts), pts, "B9_shared"), "B9_bbregister_cost": cost_B9,
                       "B9_scale_diag_pct": ((np.diag(T_B9[:3, :3]) - 1) * 100).tolist(),
                       "B9_det_pct": float((np.linalg.det(T_B9[:3, :3]) - 1) * 100)})
    if "C" in arms:
        fld_img = nib.load(str(w_field))
        if fld_img.shape[:3] != bm_img.shape or not np.allclose(fld_img.affine, bm_img.affine, atol=1e-3):
            raise ValueError("composite warp is not on the T1w brain-mask grid")
        W_pts = pts + sample_field_at_voxels(np.asarray(fld_img.dataobj), ijk)
        shared.update(summarise(W_pts, pts, "C_shared"))
    rows: list[dict[str, Any]] = []
    for r in order:
        key = r.entity_prefix
        row: dict[str, Any] = {"sub": subject, "ses": r.session, "task": r.task, "run": r.run}
        if "B" in arms:
            row.update({**summarise(apply_affine(D_B[key], T_pts), pts, "B"),
                        **summarise(apply_affine(D_B[key], pts), pts, "B_run")})
        if "C" in arms:
            S = S_C[r.session][0]
            row.update({**summarise(apply_affine(D_C[key], apply_affine(S, W_pts)), pts, "C"),
                        **summarise(apply_affine(S, pts), pts, "C_session"),
                        **summarise(apply_affine(D_C[key], pts), pts, "C_run")})
        rows.append({**row, "off_grid": key in off_grid})
    table = pd.DataFrame(rows)
    table.to_csv(paths["table"], sep="\t", index=False, na_rep="n/a", float_format="%.4f")

    check = None
    if check_run is not None:
        # Convention check: bbregister of one A-placed run must land where A put it (~0 mm).
        r = next(r for r in order if r.entity_prefix == check_run)
        T_chk, cost_chk = bbregister_to_t1w(Path(r.boldref), subjects_dir, subject, anat["fsnative_to_t1w"],
                                            paths["check"])
        check = {"run": check_run, "bbregister_cost": cost_chk, **summarise(apply_affine(T_chk, pts), pts, "disp")}
        log(f"sub-{subject}: convention check {check_run}: {check['disp_mean_mm']:.2f}/{check['disp_max_mm']:.2f} mm "
            f"(cost {cost_chk:.3f})")

    side = {"schema_version": SCHEMA_VERSION, "subject": subject, "n_runs": len(order), "n_sessions": len(sessions),
            "arms": list(arms), "reference_session": reference_session, "funcref_sessions": funcref_ses,
            "t2w": str(t2w) if t2w else None,
            "template_rounds": TEMPLATE_ROUNDS, "point_stride": POINT_STRIDE, "n_points": int(len(pts)),
            "off_grid_runs": off_grid, "shared": shared, "convention_check": check,
            "ants": {"rigid": dqa.ANTS_RIGID, "affine_gc": ANTS_AFFINE_GC, "syn_t2w": ANTS_SYN_T2W,
                     "rigid_mi": ANTS_RIGID_MI},
            "inputs": {k: str(v) for k, v in anat.items()}, **provenance,
            "created": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")}
    paths["sidecar"].write_text(json.dumps(side, indent=2, default=float) + "\n")
    log(f"sub-{subject}: stage 0 written: " + ", ".join(
        f"{a} median {table[f'{a}_mean_mm'].median():.2f} mm (shared {shared[f'{a}_shared_mean_mm']:.2f})"
        for a in arms))
    return side
