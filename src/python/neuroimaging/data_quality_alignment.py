"""Inter-session alignment of the data-quality collection (T1w space).

Every run's fMRIPrep T1w boldref should sit where the subject's other sessions
sit: they share one anatomical reference. This module measures how far each run
is from that consensus. Design record: mmmdata-agents
``docs/workbench/voxel-quality/``.

**One cell per subject** (``tier1.py alignment``; needs ANTs on PATH).

1. Each run's boldref is divided by its median over its brain mask (EPI
   intensity varies with receive gain) and put on the subject's majority T1w grid.
2. Session means: the median over the session's runs. Session masks: voxels in
   at least half the session's run masks.
3. For each session, a **leave-own-session-out template**: the median over the
   *other* sessions' means, and the template mask (voxels in at least half of
   the other sessions' masks). A run is never compared with itself or its
   session mates.
4. Per run, against its template:

   * **Residual rigid displacement** (primary): ``antsRegistration`` rigid, from
     identity (no centre-of-mass initialisation, so the current position is
     what is measured), Mattes MI on dense sampling with a fixed seed. Reported
     as the mean and max distance a template-mask voxel moves (mm), plus the
     mask centroid's shift and the rotation angle.
   * **Edge correlation** (secondary): Pearson r of the gradient magnitudes
     (Gaussian, :data:`EDGE_SIGMA_MM`) of run and template inside the template
     mask, before any registration. Sensitive to sub-voxel shifts.
   * **Mask Dice** against the template mask, labelled coverage: it mixes field
     of view placement with misregistration.

   ``flag`` = mean displacement > :data:`DISPLACEMENT_FLAG_MM` (RATIFIED
   2026-10-09: half the 1.8 mm in-plane voxel).

Outputs under ``<tree>/sub-##/func/``::

    sub-##_space-T1w_desc-alignment_qc.tsv           one row per run
    sub-##_space-T1w_desc-alignment_qc.json          provenance, template facts
    sub-##_space-T1w_desc-runs_boldref.nii.gz        4D, one normalised boldref per run
    sub-##_space-T1w_desc-runs_boldref.tsv           volume index -> run
    sub-##_space-T1w_desc-sessions_boldref.nii.gz    4D, one session mean per session
    sub-##_space-T1w_desc-sessions_boldref.tsv
    sub-##_space-T1w_desc-template_boldref.nii.gz    median over all sessions
    sub-##_space-T1w_desc-{sessions,runs}_boldref.gif  cycling movie, fixed WM outline
"""

from __future__ import annotations

import collections
import datetime as _dt
import json
import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from . import data_quality as dq
from . import data_quality_voxelmaps as dqv
from .io import FmriprepRun

SCHEMA_VERSION = "1.0"
CELL = "alignment"
TABLE_NAME = "tier1_alignment"
#: Mean residual displacement above which a run is flagged (RATIFIED 2026-10-09).
DISPLACEMENT_FLAG_MM = 0.9
EDGE_SIGMA_MM = 2.0
#: A voxel is in a session (template) mask when in at least this fraction of its runs (sessions).
MASK_MAJORITY = 0.5
#: Fixed so a rebuild registers identically.
RANDOM_SEED = 13
ANTS_RIGID = [
    "--dimensionality", "3", "--float", "1", "--collapse-output-transforms", "1",
    "--interpolation", "Linear", "--use-histogram-matching", "0", "--winsorize-image-intensities", "[0.005,0.995]",
    "--random-seed", str(RANDOM_SEED),
    "--transform", "Rigid[0.1]",
    "--convergence", "[200x100x50,1e-7,10]", "--shrink-factors", "4x2x1", "--smoothing-sigmas", "2x1x0vox",
]


# ---------------------------------------------------------------------------
# Pure helpers (tested on synthetic data)
# ---------------------------------------------------------------------------

def itk_affine_to_matrix(params: np.ndarray, center: np.ndarray) -> np.ndarray:
    """4x4 LPS matrix of an ITK ``AffineTransform_double_3_3`` (12 params + fixed centre).

    ITK maps a point p (fixed space) to ``A (p - c) + c + t``.
    """
    params = np.asarray(params, dtype=np.float64).ravel()
    center = np.asarray(center, dtype=np.float64).ravel()
    A = params[:9].reshape(3, 3)
    t = params[9:12]
    M = np.eye(4)
    M[:3, :3] = A
    M[:3, 3] = t + center - A @ center
    return M


def ras_to_lps(points: np.ndarray) -> np.ndarray:
    p = np.array(points, dtype=np.float64, copy=True)
    p[..., :2] *= -1
    return p


def displacement(M: np.ndarray, points_lps: np.ndarray) -> np.ndarray:
    """Distance each point moves under the 4x4 transform."""
    p = np.asarray(points_lps, dtype=np.float64)
    moved = (M @ np.c_[p, np.ones(len(p))].T).T[:, :3]
    return np.linalg.norm(moved - p, axis=1)


def rotation_deg(M: np.ndarray) -> float:
    R = M[:3, :3]
    c = (np.trace(R) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def dice(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, bool)
    b = np.asarray(b, bool)
    s = a.sum() + b.sum()
    return float(2.0 * (a & b).sum() / s) if s else np.nan


def edge_correlation(a: np.ndarray, b: np.ndarray, mask: np.ndarray, sigma_vox: np.ndarray) -> float:
    """Pearson r of Gaussian gradient magnitudes inside ``mask`` (NaN treated as 0 before filtering)."""
    from scipy.ndimage import gaussian_gradient_magnitude

    ga = gaussian_gradient_magnitude(np.nan_to_num(np.asarray(a, np.float64)), sigma=sigma_vox)
    gb = gaussian_gradient_magnitude(np.nan_to_num(np.asarray(b, np.float64)), sigma=sigma_vox)
    x, y = ga[mask], gb[mask]
    if x.size < 3 or x.std() == 0 or y.std() == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def loso_template(session_means: dict[str, np.ndarray], session_masks: dict[str, np.ndarray], left_out: str
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Median of the other sessions' means, and voxels in ≥ MASK_MAJORITY of the other sessions' masks."""
    others = [s for s in session_means if s != left_out]
    if not others:
        raise ValueError(f"no session other than {left_out!r} to build a template from")
    tmpl = dqv.nanmedian_stack([session_means[s] for s in others])
    frac = np.mean([session_masks[s] for s in others], axis=0)
    return tmpl, frac >= MASK_MAJORITY


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def _ants() -> str:
    exe = shutil.which("antsRegistration")
    if exe is None:
        raise RuntimeError("antsRegistration is not on PATH; `module load ants/2.5.2` BEFORE activating the venv")
    return exe


def register_rigid(fixed: Path, moving: Path, fixed_mask: Path, prefix: Path) -> np.ndarray:
    """Rigid moving -> fixed from identity; returns the fixed->moving 4x4 (LPS)."""
    from scipy.io import loadmat

    cmd = [_ants(), *ANTS_RIGID,
           "--metric", f"MI[{fixed},{moving},1,32,None]",
           "--masks", f"[{fixed_mask},NULL]",
           "--output", f"[{prefix}]"]
    env = {**os.environ, "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS": "1"}
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"antsRegistration failed for {moving}: {proc.stderr[-2000:]}")
    mat = loadmat(f"{prefix}0GenericAffine.mat")
    key = next(k for k in mat if k.startswith("AffineTransform"))
    return itk_affine_to_matrix(mat[key], mat["fixed"])


# ---------------------------------------------------------------------------
# Cell
# ---------------------------------------------------------------------------

def cell_paths(tree_root: Path, subject: str) -> dict[str, Path]:
    d = Path(tree_root) / f"sub-{subject}" / "func"
    stem = d / f"sub-{subject}_space-{dqv.VOLUME_SPACE}"
    return {"table": Path(f"{stem}_desc-alignment_qc.tsv"), "sidecar": Path(f"{stem}_desc-alignment_qc.json"),
            "runs_nii": Path(f"{stem}_desc-runs_boldref.nii.gz"), "runs_tsv": Path(f"{stem}_desc-runs_boldref.tsv"),
            "sessions_nii": Path(f"{stem}_desc-sessions_boldref.nii.gz"),
            "sessions_tsv": Path(f"{stem}_desc-sessions_boldref.tsv"),
            "template": Path(f"{stem}_desc-template_boldref.nii.gz"),
            "sessions_gif": Path(f"{stem}_desc-sessions_boldref.gif"), "runs_gif": Path(f"{stem}_desc-runs_boldref.gif")}


def is_current(tree_root: Path, subject: str, keys: dict) -> bool:
    paths = cell_paths(tree_root, subject)
    if not all(p.exists() for p in paths.values()):
        return False
    side = json.loads(paths["sidecar"].read_text())
    return side.get("schema_version") == SCHEMA_VERSION and side.get("input_keys") == keys


def _save4d(vols: list[np.ndarray], affine: np.ndarray, path: Path) -> None:
    dqv.save_volume(np.stack(vols, axis=-1), affine, path)


def build_cell(tree_root: Path, fmriprep_tree: Path, subject: str, runs: list[FmriprepRun], provenance: dict,
               n_jobs: int = 1, force: bool = False, log: Callable[[str], None] = print) -> Optional[dict]:
    import nibabel as nib

    anat = dqv.anat_files(fmriprep_tree, subject)
    keys = {"boldref": {r.entity_prefix: dq.file_sha256(r.boldref) for r in runs},
            "mask": {r.entity_prefix: dq.file_sha256(r.mask) for r in runs},
            "wm": dq.file_sha256(anat["wm"])}
    if not force and is_current(tree_root, subject, keys):
        log(f"sub-{subject}: alignment cell current, skipping")
        return None
    shape, ref_affine, off_grid = dqv.reference_grid(runs)
    zooms = np.sqrt((ref_affine[:3, :3] ** 2).sum(axis=0))
    order = sorted(runs, key=lambda r: (r.session, r.task, r.run or ""))

    norm: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}
    for r in order:
        img = nib.load(str(r.boldref))
        m = np.asarray(nib.load(str(r.mask)).dataobj) > 0
        v = np.asarray(img.dataobj, dtype=np.float32)
        v = v / dqv.run_scale(v, m)
        # The whole boldref (skull included) is kept: edges outside the brain carry alignment information.
        full = np.ones(v.shape, dtype=bool)
        norm[r.entity_prefix] = dqv.onto_grid(v, full, img.affine, shape, ref_affine)[0]
        masks[r.entity_prefix] = dqv.onto_grid(m.astype(np.float32), m, img.affine, shape, ref_affine)[1]

    sessions: dict[str, list[FmriprepRun]] = collections.defaultdict(list)
    for r in order:
        sessions[r.session].append(r)
    if len(sessions) < 2:
        raise ValueError(f"sub-{subject}: one session only; there is no other session to align to")
    ses_mean = {s: dqv.nanmedian_stack([norm[r.entity_prefix] for r in rs]) for s, rs in sessions.items()}
    ses_mask = {s: np.mean([masks[r.entity_prefix] for r in rs], axis=0) >= MASK_MAJORITY for s, rs in sessions.items()}
    template_all = dqv.nanmedian_stack(list(ses_mean.values()))

    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as tmp:
        tmp = Path(tmp)
        tmpl_files: dict[str, tuple[Path, Path, np.ndarray, np.ndarray]] = {}
        for s in sessions:
            t, tm = loso_template(ses_mean, ses_mask, s)
            tp, mp = tmp / f"tmpl_{s}.nii.gz", tmp / f"tmask_{s}.nii.gz"
            dqv.save_volume(np.nan_to_num(t), ref_affine, tp)
            dqv.save_volume(tm.astype(np.float32), ref_affine, mp)
            tmpl_files[s] = (tp, mp, t, tm)

        def one(r: FmriprepRun) -> dict[str, Any]:
            tp, mp, t, tm = tmpl_files[r.session]
            M = register_rigid(tp, Path(r.boldref), mp, tmp / f"{r.entity_prefix}_")
            ijk = np.argwhere(tm)
            world = (ref_affine @ np.c_[ijk, np.ones(len(ijk))].T).T[:, :3]
            d = displacement(M, ras_to_lps(world))
            return {"sub": subject, "ses": r.session, "task": r.task, "run": r.run,
                    "disp_mean_mm": float(d.mean()), "disp_max_mm": float(d.max()),
                    "centroid_shift_mm": float(displacement(M, ras_to_lps(world.mean(axis=0, keepdims=True)))[0]),
                    "rotation_deg": rotation_deg(M),
                    "edge_r": edge_correlation(norm[r.entity_prefix], t, tm, EDGE_SIGMA_MM / zooms),
                    "mask_dice": dice(masks[r.entity_prefix], tm),
                    "off_grid": r.entity_prefix in off_grid,
                    "flag": bool(d.mean() > DISPLACEMENT_FLAG_MM)}

        with ThreadPoolExecutor(max_workers=max(1, n_jobs)) as pool:
            for row in pool.map(one, order):
                rows.append(row)
                log(f"  {row['ses']} {row['task']} {row['run'] or ''}: disp {row['disp_mean_mm']:.2f}/"
                    f"{row['disp_max_mm']:.2f} mm, edge r {row['edge_r']:.3f}, dice {row['mask_dice']:.3f}"
                    + ("  FLAG" if row["flag"] else ""))

    table = pd.DataFrame(rows)
    paths = cell_paths(tree_root, subject)
    paths["table"].parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(paths["table"], sep="\t", index=False, na_rep="n/a", float_format="%.6g")
    _save4d([norm[r.entity_prefix] for r in order], ref_affine, paths["runs_nii"])
    pd.DataFrame([{"volume": i, "ses": r.session, "task": r.task, "run": r.run or "n/a",
                   "disp_mean_mm": table.loc[i, "disp_mean_mm"]} for i, r in enumerate(order)]
                 ).to_csv(paths["runs_tsv"], sep="\t", index=False, float_format="%.4g")
    ses_order = sorted(sessions)
    _save4d([ses_mean[s] for s in ses_order], ref_affine, paths["sessions_nii"])
    ses_disp = table.groupby("ses")["disp_mean_mm"].median()
    pd.DataFrame([{"volume": i, "ses": s, "n_runs": len(sessions[s]), "disp_mean_mm_median": ses_disp[s]}
                  for i, s in enumerate(ses_order)]).to_csv(paths["sessions_tsv"], sep="\t", index=False,
                                                            float_format="%.4g")
    dqv.save_volume(template_all, ref_affine, paths["template"])

    wm_img = nib.load(str(anat["wm"]))
    wm = dqv.onto_grid(np.asarray(wm_img.dataobj, dtype=np.float32), np.ones(wm_img.shape, bool),
                       wm_img.affine, shape, ref_affine)[0]
    write_movie([ses_mean[s] for s in ses_order],
                [f"sub-{subject} ses-{s}  median disp {ses_disp[s]:.2f} mm" for s in ses_order],
                wm, template_all, paths["sessions_gif"], fps=2)
    write_movie([norm[r.entity_prefix] for r in order],
                [f"sub-{subject} ses-{r.session} {r.task} {r.run or ''}  disp {table.loc[i, 'disp_mean_mm']:.2f} mm"
                 + ("  FLAG" if table.loc[i, "flag"] else "") for i, r in enumerate(order)],
                wm, template_all, paths["runs_gif"], fps=4)

    side = {"schema_version": SCHEMA_VERSION, "cell": CELL, "subject": subject, "n_runs": len(order),
            "n_sessions": len(sessions), "grid_shape": list(shape), "off_grid_runs": off_grid,
            "displacement_flag_mm": DISPLACEMENT_FLAG_MM, "edge_sigma_mm": EDGE_SIGMA_MM,
            "mask_majority": MASK_MAJORITY, "ants": ANTS_RIGID + ["--metric", "MI[fixed,moving,1,32,None]"],
            "n_flagged": int(table["flag"].sum()), "input_keys": keys,
            "inputs": {"wm": str(anat["wm"])}, **provenance,
            "created": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")}
    paths["sidecar"].write_text(json.dumps(side, indent=2, default=float) + "\n")
    log(f"sub-{subject}: alignment written, {len(order)} runs, {side['n_flagged']} flagged "
        f"(disp mean median {table['disp_mean_mm'].median():.2f} mm, max {table['disp_mean_mm'].max():.2f})")
    return side


def write_movie(frames: list[np.ndarray], titles: list[str], wm: np.ndarray, template: np.ndarray,
                path: Path, fps: int = 3) -> None:
    """GIF cycling ``frames`` in three orthogonal slices, the WM boundary (probseg 0.5) fixed on top."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    fin = np.isfinite(template)
    coords = np.argwhere(fin & (template > np.nanpercentile(template[fin], 50)))
    c = np.round(coords.mean(axis=0)).astype(int) if len(coords) else np.array(template.shape) // 2
    step = max(1, len(frames) // 20)
    lo, hi = np.nanpercentile(np.stack(frames[::step])[:, fin], [1, 99])

    def cuts(v: np.ndarray) -> list[np.ndarray]:
        return [np.rot90(v[c[0], :, :]), np.rot90(v[:, c[1], :]), np.rot90(v[:, :, c[2]])]

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.8), dpi=80)
    ims = []
    for ax, sl, wsl in zip(axes, cuts(np.nan_to_num(frames[0])), cuts(wm)):
        ims.append(ax.imshow(sl, cmap="gray", vmin=lo, vmax=hi, interpolation="nearest"))
        ax.contour(wsl, levels=[0.5], colors="#e8a33d", linewidths=0.8)
        ax.set_axis_off()
    title = fig.suptitle(titles[0], fontsize=10)
    fig.tight_layout()

    def update(i: int):
        for im, sl in zip(ims, cuts(np.nan_to_num(frames[i]))):
            im.set_data(sl)
        title.set_text(titles[i])
        return ims + [title]

    anim = FuncAnimation(fig, update, frames=len(frames), blit=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(path), writer=PillowWriter(fps=fps))
    plt.close(fig)


def collect(tree_root: Path) -> pd.DataFrame:
    frames = [pd.read_csv(p, sep="\t", na_values=["n/a"], dtype={"sub": str, "ses": str, "run": str})
              for p in sorted(Path(tree_root).glob(f"sub-*/func/sub-*_space-{dqv.VOLUME_SPACE}_desc-alignment_qc.tsv"))]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
