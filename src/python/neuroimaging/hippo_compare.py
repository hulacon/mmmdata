"""Hippocampal segmentations from several tools on one grid: volumes, Dice, voxel-assignment agreement.

Every arm is resampled **once**, from its finest output, into fMRIPrep's T1w
space, and compared there on two grids:

* **fine**: fMRIPrep T1w axes at ``FINE_MM`` isotropic, one box per hemisphere
  (the FreeSurfer aseg hippocampus, padded by ``BOX_PAD_MM``). Labels are
  resampled with ANTs ``GenericLabel``.
* **bold**: the subject's ``space-T1w`` BOLD grid (the one most of its runs use). Each BOLD voxel takes the
  **plurality** canonical label of the fine voxels whose centres fall in it,
  background included (ties go to the lower class, background first). This is
  the label a voxel would carry in a subfield BOLD analysis.

**Arms** (``ARMS``): HippUnfold 1.5.2 and 2.2.0, each on the SPC and the
oblique-coronal T2w; HSF on each T2w; FreeSurfer aseg (whole hippocampus only).

**Transforms into fMRIPrep T1w.** HippUnfold's T1w is the raw ses-01 MPR
run-01 image, not fMRIPrep's run-01/run-02 template, so its labels go through
fMRIPrep's ``run-01 from-orig_to-T1w``. That transform is used **forward**:
antsApplyTransforms resamples an image from ``orig`` to ``T1w`` with it (the
*inverse* is what maps points such as surface vertices). HSF labels live on the
raw T2w grid; each T2w is registered rigidly (mutual information) to the
fMRIPrep T1w, and the registration is checked against HippUnfold's own
``space-T1w_desc-preproc_T2w`` (correlation inside the box). aseg is already in
fMRIPrep T1w space.

**Canonical labels** (``CANON``). HippUnfold multihist7 CA4 joins DG because
HSF has no CA4 class (its five are DG, CA1, CA2, CA3, Sub), so hilar tissue can
only sit in HSF's DG or CA3. This is a protocol mapping, not a claim that the
two DG definitions agree. HippUnfold cyst is background. FreeSurfer contributes
one undivided ``hipp`` class.

**Regions** (``REGIONS``): ``hipp`` (HippUnfold incl. SRLM), ``hippNoSRLM``,
and the five shared subfields. Subfield regions are scored only between arms
that have subfields.

Outputs under ``<out>/sub-##/ses-01/anat/`` (``<out>`` is an internal-only
derivative tree)::

    *_hemi-{L,R}_space-T1w_res-{fine,bold}_desc-<arm>_dseg.nii.gz   canonical labels per arm
    *_hemi-{L,R}_space-T1w_res-fine_desc-{SPC,oblcor}_T2w.nii.gz     registered T2w (montage/3D background)
    *_hemi-{L,R}_space-T1w_res-fine_desc-hippAgreement_mask.nii.gz   number of arms calling a voxel hippocampus
    *_from-T2w{SPC,oblcor}_to-T1w_mode-image_xfm.mat                 HSF's T2w -> T1w rigid
    *_desc-{volumes,dice,assignment,registration}_stats.tsv          one subject's tables
    *_desc-hippcompare.json                                          provenance and settings
    *_desc-montage_hippcompare.png                                   coronal montage, every arm

Design record: mmmdata-agents ``docs/workbench/hippunfold-eval/`` (settle 2).
"""

from __future__ import annotations

import datetime as _dt
import itertools
import json
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

import nibabel as nib
import numpy as np
import pandas as pd

from . import alignment_candidates as ac
from . import data_quality_voxelmaps as dqv

SCHEMA_VERSION = "1.0"
#: Fine-grid spacing (mm). HippUnfold's cropT1w is ~0.2 mm; HSF and aseg are coarser.
FINE_MM = 0.3
#: Padding (mm) around the aseg hippocampus that defines each hemisphere's box.
BOX_PAD_MM = 10.0
#: Every arm segments the ses-01 anatomy (charter).
SESSION = "01"
HEMIS = ("L", "R")

CANON = {0: "bg", 1: "Sub", 2: "CA1", 3: "CA2", 4: "CA3", 5: "DG", 6: "SRLM", 7: "hipp"}
CANON_ID = {v: k for k, v in CANON.items()}
SUBFIELDS = ("Sub", "CA1", "CA2", "CA3", "DG")
N_CLASSES = len(CANON)
REGIONS = {
    "hipp": (1, 2, 3, 4, 5, 6, 7),
    "hippNoSRLM": (1, 2, 3, 4, 5, 7),
    **{s: (CANON_ID[s],) for s in SUBFIELDS},
}
#: Native label -> canonical label, per tool; labels not listed are background.
NATIVE_TO_CANON = {
    # multihist7: 1 Sub, 2 CA1, 3 CA2, 4 CA3, 5 CA4, 6 DG, 7 SRLM, 8 Cyst
    "hippunfold": {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 5, 7: 6},
    # hsf/segment.py: 0 bg, 1 dg, 2 ca1, 3 ca2, 4 ca3, 5 sub
    "hsf": {1: 5, 2: 2, 3: 3, 4: 4, 5: 1},
    # aseg Left-/Right-Hippocampus
    "fs_L": {17: 7},
    "fs_R": {53: 7},
}
FS_HEMI_LABEL = {"L": 17, "R": 53}

#: Montage colours per canonical class (categorical slots 1-7, validated light: CVD dE >= 9.1 adjacent).
CANON_COLOURS = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a", 4: "#eda100", 5: "#e87ba4", 6: "#008300", 7: "#6250d6"}


@dataclass(frozen=True)
class Arm:
    name: str           # BIDS desc label
    tool: str           # hippunfold | hsf | fs
    tree: str           # relative to the derivatives dir
    t2w_acq: Optional[str]

    @property
    def has_subfields(self) -> bool:
        return self.tool != "fs"


ARMS = (
    Arm("U1spc", "hippunfold", "hippunfold/hippunfold", "SPC"),
    Arm("U1obl", "hippunfold", "hippunfold_oblcor/hippunfold", "oblcor"),
    Arm("U2spc", "hippunfold", "hippunfold_v2", "SPC"),
    Arm("U2obl", "hippunfold", "hippunfold_v2_oblcor", "oblcor"),
    Arm("Hspc", "hsf", "hsf", "SPC"),
    Arm("Hobl", "hsf", "hsf_oblcor", "oblcor"),
    Arm("FS", "fs", "fmriprep", None),
)
#: Tree whose ``space-T1w_desc-preproc_T2w`` checks each T2w registration.
T2W_CHECK_TREE = {"SPC": "hippunfold_v2", "oblcor": "hippunfold_v2_oblcor"}


# ---------------------------------------------------------------------------
# Pure helpers (tested on synthetic data)
# ---------------------------------------------------------------------------

def dice(a: np.ndarray, b: np.ndarray) -> float:
    """Dice of two boolean masks; NaN when both are empty."""
    total = int(a.sum()) + int(b.sum())
    return float("nan") if total == 0 else 2.0 * int(np.logical_and(a, b).sum()) / total


def to_canon(native: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    """Map native integer labels to canonical ones; anything unmapped becomes background."""
    lut = np.zeros(int(max(max(mapping), native.max(initial=0))) + 1, dtype=np.uint8)
    for k, v in mapping.items():
        lut[k] = v
    return lut[np.clip(native, 0, None).astype(np.int64)]


def box_affine(mask: np.ndarray, affine: np.ndarray, spacing: float, pad: float) -> tuple[np.ndarray, tuple[int, ...]]:
    """A grid along ``affine``'s axes at ``spacing`` covering ``mask``'s voxels (whole voxels) plus ``pad`` mm."""
    if not mask.any():
        raise ValueError("cannot build a box around an empty mask")
    zooms = np.linalg.norm(affine[:3, :3], axis=0)
    rot = affine[:3, :3] / zooms
    ijk = np.argwhere(mask)
    corners = np.concatenate([ijk - 0.5, ijk + 0.5])
    u = (rot.T @ (nib.affines.apply_affine(affine, corners).T)).T  # coordinates along the image axes (mm)
    lo, hi = u.min(0) - pad, u.max(0) + pad
    shape = tuple(int(np.ceil((h - l) / spacing)) + 1 for l, h in zip(lo, hi))
    out = np.eye(4)
    out[:3, :3] = rot * spacing
    out[:3, 3] = rot @ lo
    return out, shape


@dataclass
class CoarseMap:
    """Which coarse voxel each fine voxel's centre falls in (fine voxels outside the coarse grid dropped)."""
    keep: np.ndarray        # flat fine indices that land inside the coarse grid
    group: np.ndarray       # per kept fine voxel: index into ``flat``
    flat: np.ndarray        # flat coarse indices touched, sorted
    coarse_shape: tuple[int, ...]


def coarse_map(fine_shape: Sequence[int], fine_affine: np.ndarray, coarse_shape: Sequence[int],
               coarse_affine: np.ndarray) -> CoarseMap:
    ijk = np.indices(fine_shape).reshape(3, -1).T
    c = np.floor(nib.affines.apply_affine(np.linalg.inv(coarse_affine) @ fine_affine, ijk) + 0.5).astype(np.int64)
    inside = np.all((c >= 0) & (c < np.asarray(coarse_shape)), axis=1)
    flat_all = np.ravel_multi_index(c[inside].T, tuple(coarse_shape))
    flat, group = np.unique(flat_all, return_inverse=True)
    return CoarseMap(np.flatnonzero(inside), group, flat, tuple(coarse_shape))


def plurality(canon_fine: np.ndarray, cmap: CoarseMap, n_classes: int = N_CLASSES) -> np.ndarray:
    """Plurality canonical label per touched coarse voxel (background counted; ties -> lower class)."""
    labels = canon_fine.ravel()[cmap.keep].astype(np.int64)
    counts = np.bincount(cmap.group * n_classes + labels, minlength=len(cmap.flat) * n_classes)
    return counts.reshape(len(cmap.flat), n_classes).argmax(1).astype(np.uint8)


def crop_coarse(values: np.ndarray, cmap: CoarseMap, coarse_affine: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Scatter per-touched-voxel ``values`` into the bounding box of the touched coarse voxels."""
    ijk = np.array(np.unravel_index(cmap.flat, cmap.coarse_shape)).T
    lo, hi = ijk.min(0), ijk.max(0) + 1
    out = np.zeros(tuple(hi - lo), dtype=values.dtype)
    out[tuple((ijk - lo).T)] = values
    aff = coarse_affine.copy()
    aff[:3, 3] = nib.affines.apply_affine(coarse_affine, lo)
    return out, aff


def region_mask(canon: np.ndarray, region: str) -> np.ndarray:
    return np.isin(canon, REGIONS[region])


def pair_rows(canon: dict[str, np.ndarray], arms: Sequence[Arm]) -> list[dict]:
    """Dice for every arm pair and every region both arms can express."""
    rows = []
    for a, b in itertools.combinations(arms, 2):
        regions = ["hipp", "hippNoSRLM"] + (list(SUBFIELDS) if a.has_subfields and b.has_subfields else [])
        for r in regions:
            ma, mb = region_mask(canon[a.name], r), region_mask(canon[b.name], r)
            rows.append({"arm_a": a.name, "arm_b": b.name, "region": r, "dice": dice(ma, mb),
                         "n_a": int(ma.sum()), "n_b": int(mb.sum())})
    return rows


def assignment_rows(canon: dict[str, np.ndarray], arms: Sequence[Arm]) -> list[dict]:
    """Voxel-assignment agreement between subfield arms: same subfield, among voxels either/both call a subfield."""
    sub_ids = [CANON_ID[s] for s in SUBFIELDS]
    rows = []
    for a, b in itertools.combinations([x for x in arms if x.has_subfields], 2):
        la, lb = canon[a.name], canon[b.name]
        ia, ib = np.isin(la, sub_ids), np.isin(lb, sub_ids)
        either, both = ia | ib, ia & ib
        same = (la == lb) & both
        rows.append({"arm_a": a.name, "arm_b": b.name, "n_either": int(either.sum()), "n_both": int(both.sum()),
                     "agree_either": float(same.sum() / either.sum()) if either.any() else float("nan"),
                     "agree_both": float(same.sum() / both.sum()) if both.any() else float("nan")})
    return rows


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def _one(pattern_dir: Path, pattern: str) -> Path:
    hits = sorted(Path(pattern_dir).glob(pattern))
    if len(hits) != 1:
        raise FileNotFoundError(f"expected one {pattern} under {pattern_dir}, found {[str(h) for h in hits]}")
    return hits[0]


def arm_input(arm: Arm, deriv: Path, subject: str, hemi: str) -> Path:
    tree = Path(deriv) / arm.tree
    if arm.tool == "hippunfold":
        return _one(tree / f"sub-{subject}" / f"ses-{SESSION}" / "anat",
                    f"sub-{subject}_ses-{SESSION}_hemi-{hemi}_space-cropT1w_*desc-subfields_atlas-multihist7_dseg.nii.gz")
    if arm.tool == "hsf":
        side = {"L": "left", "R": "right"}[hemi]
        return _one(tree / f"sub-{subject}",
                    f"sub-{subject}_ses-{SESSION}_acq-{arm.t2w_acq}_T2w_{side}_hippocampus_seg.nii.gz")
    return dqv._anat_one(tree / f"sub-{subject}", subject, "_desc-aseg_dseg.nii.gz")


def tree_version(tree: Path, deriv: Path) -> str:
    """``GeneratedBy[0].Version`` from the nearest dataset_description.json at or above ``tree`` (within deriv)."""
    for d in [Path(tree), *Path(tree).parents]:
        desc = d / "dataset_description.json"
        if desc.exists():
            gb = json.loads(desc.read_text()).get("GeneratedBy") or [{}]
            return str(gb[0].get("Version", "unknown"))
        if d == Path(deriv):
            break
    return "unknown"


def bold_reference(fmriprep_tree: Path, subject: str) -> tuple[Path, list[dict]]:
    """A space-T1w boldref on the grid most of the subject's runs use, plus a count of runs per grid.

    Grids can differ within a subject (sub-03's first sessions used another EPI geometry); the comparison is made
    on the majority grid and the others are recorded, not silently dropped.
    """
    refs = sorted((Path(fmriprep_tree) / f"sub-{subject}").glob("ses-*/func/*_space-T1w_boldref.nii.gz"))
    if not refs:
        raise FileNotFoundError(f"no space-T1w boldref under {fmriprep_tree}/sub-{subject}")
    grids: dict[tuple, list[Path]] = {}
    for r in refs:
        img = nib.load(str(r))
        grids.setdefault((img.shape[:3], tuple(np.round(img.affine, 3).ravel())), []).append(r)
    ranked = sorted(grids.items(), key=lambda kv: -len(kv[1]))
    summary = [{"shape": list(k[0]), "zooms_mm": [round(float(z), 3) for z in np.linalg.norm(
                    np.array(k[1]).reshape(4, 4)[:3, :3], axis=0)],
                "n_boldrefs": len(v), "sessions": sorted({p.parts[-3] for p in v}), "used": i == 0}
               for i, (k, v) in enumerate(ranked)]
    return ranked[0][1][0], summary


# ---------------------------------------------------------------------------
# ANTs
# ---------------------------------------------------------------------------

def ants_resample(moving: Path, reference: Path, transforms: Sequence[str], out: Path, labels: bool) -> Path:
    cmd = [ac._exe("antsApplyTransforms", "`module load ants/2.5.2` BEFORE activating the venv"), "-d", "3",
           "-i", str(moving), "-r", str(reference), "-o", str(out),
           "-n", "GenericLabel" if labels else "Linear", "-u", "int" if labels else "float"]
    for t in transforms or ["identity"]:
        cmd += ["-t", str(t)]
    ac._run(cmd, f"antsApplyTransforms {Path(moving).name} -> {Path(out).name}")
    return out


# ---------------------------------------------------------------------------
# Subject
# ---------------------------------------------------------------------------

def _rel(p: Path, root: Path) -> str:
    try:
        return str(Path(p).relative_to(root))
    except ValueError:
        return str(p)


def subject_paths(out_root: Path, subject: str) -> dict[str, Path]:
    d = Path(out_root) / f"sub-{subject}" / f"ses-{SESSION}" / "anat"
    stem = f"sub-{subject}_ses-{SESSION}"
    return {"dir": d, "stem": d / stem,
            **{k: d / f"{stem}_desc-{k}_stats.tsv" for k in ("volumes", "dice", "assignment", "registration")},
            "sidecar": d / f"{stem}_desc-hippcompare.json",
            "montage": d / f"{stem}_desc-montage_hippcompare.png"}


def _save(arr: np.ndarray, affine: np.ndarray, path: Path) -> Path:
    img = nib.Nifti1Image(arr, affine)
    img.header.set_xyzt_units("mm")
    nib.save(img, str(path))
    return path


def build_subject(out_root: Path, bids_root: Path, deriv: Path, subject: str, provenance: dict,
                  fmriprep_variant: str = "fmriprep", arms: Sequence[Arm] = ARMS, n_jobs: int = 1,
                  log: Callable[[str], None] = print) -> dict[str, Path]:
    """Resample every arm into fMRIPrep T1w (fine + BOLD grids) and write one subject's tables and maps."""
    paths = subject_paths(out_root, subject)
    paths["dir"].mkdir(parents=True, exist_ok=True)
    fmriprep_tree = Path(deriv) / fmriprep_variant
    sub_dir = fmriprep_tree / f"sub-{subject}"
    t1w = dqv._anat_one(sub_dir, subject, "_desc-preproc_T1w.nii.gz")
    brain = dqv._anat_one(sub_dir, subject, "_desc-brain_mask.nii.gz")
    aseg = dqv._anat_one(sub_dir, subject, "_desc-aseg_dseg.nii.gz")
    orig_to_t1w = dqv._anat_one(sub_dir, subject, "_run-01_from-orig_to-T1w_mode-image_xfm.txt")
    bold_ref, bold_grids = bold_reference(fmriprep_tree, subject)
    inputs = {(a.name, h): arm_input(a, deriv, subject, h) for a in arms for h in HEMIS}
    acqs = sorted({a.t2w_acq for a in arms if a.t2w_acq})
    t2w = {q: ac.find_t2w(bids_root, subject, q) for q in acqs}
    stem = paths["stem"].name

    def pmap(fn, items):
        with ThreadPoolExecutor(max_workers=max(1, n_jobs)) as ex:
            return list(ex.map(fn, items))

    # ---- T2w -> fMRIPrep T1w, rigid MI (HSF's route; montage background).
    xfm = {}
    with tempfile.TemporaryDirectory(prefix=f"hippcmp_{subject}_") as tmp:
        tmp = Path(tmp)

        def register(q):
            pre = ac.ants_register(t1w, t2w[q], tmp / f"t2w{q}_", ac.ANTS_RIGID_MI, fixed_mask=brain)
            dst = paths["dir"] / f"{stem}_from-T2w{q}_to-T1w_mode-image_xfm.mat"
            dst.write_bytes(Path(f"{pre}0GenericAffine.mat").read_bytes())
            return q, dst
        xfm = dict(pmap(register, acqs))
        log(f"sub-{subject}: T2w registrations done ({', '.join(acqs)})")

        def chain(arm: Arm) -> list[str]:
            if arm.tool == "hippunfold":
                return [str(orig_to_t1w)]
            if arm.tool == "hsf":
                return [str(xfm[arm.t2w_acq])]
            return []

        # ---- Per-hemisphere fine boxes from the aseg hippocampus.
        aseg_img = nib.load(str(aseg))
        aseg_data = np.asarray(aseg_img.dataobj)
        boxes = {}
        for h in HEMIS:
            aff, shape = box_affine(aseg_data == FS_HEMI_LABEL[h], aseg_img.affine, FINE_MM, BOX_PAD_MM)
            ref = _save(np.zeros(shape, np.uint8), aff, tmp / f"box_{h}.nii.gz")
            boxes[h] = (aff, shape, ref)

        jobs = [(a, h) for a in arms for h in HEMIS]

        def resample(job):
            a, h = job
            out = ants_resample(inputs[(a.name, h)], boxes[h][2], chain(a), tmp / f"{a.name}_{h}.nii.gz", labels=True)
            return job, np.asarray(nib.load(str(out)).dataobj).astype(np.int32)
        native_fine = dict(pmap(resample, jobs))
        log(f"sub-{subject}: {len(jobs)} label resamplings done")

        def background(job):
            q, h = job
            out = ants_resample(t2w[q], boxes[h][2], [str(xfm[q])], tmp / f"T2w{q}_{h}.nii.gz", labels=False)
            chk_src = _one(Path(deriv) / T2W_CHECK_TREE[q] / f"sub-{subject}" / f"ses-{SESSION}" / "anat",
                           f"sub-{subject}_ses-{SESSION}_space-T1w_desc-preproc_T2w.nii.gz")
            chk = ants_resample(chk_src, boxes[h][2], [str(orig_to_t1w)], tmp / f"chk{q}_{h}.nii.gz", labels=False)
            return job, np.asarray(nib.load(str(out)).dataobj), np.asarray(nib.load(str(chk)).dataobj), chk_src
        backgrounds = pmap(background, [(q, h) for q in acqs for h in HEMIS])

    # ---- Registration check: our T2w vs HippUnfold's T2w-in-T1w, inside the box where both have signal.
    reg_rows, t2w_fine = [], {}
    for (q, h), ours, theirs, chk_src in backgrounds:
        t2w_fine[(q, h)] = ours
        m = (ours > 0) & (theirs > 0)
        r = float(np.corrcoef(ours[m], theirs[m])[0, 1]) if m.sum() > 10 else float("nan")
        reg_rows.append({"subject": subject, "hemi": h, "t2w_acq": q, "r_vs_hippunfold_T2w": r, "n_vox": int(m.sum()),
                         "reference": _rel(chk_src, bids_root)})
        _save(ours.astype(np.float32), boxes[h][0], paths["dir"] / f"{stem}_hemi-{h}_space-T1w_res-fine_desc-{q}_T2w.nii.gz")

    # ---- Canonical labels, maps, tables.
    bold_img = nib.load(str(bold_ref))
    vol_rows, dice_rows, assign_rows, montage_inputs = [], [], [], {}
    fine_vox = FINE_MM ** 3
    bold_vox = float(abs(np.linalg.det(bold_img.affine[:3, :3])))
    for h in HEMIS:
        aff, shape, _ = boxes[h]
        cmap = coarse_map(shape, aff, bold_img.shape[:3], bold_img.affine)
        canon_fine, canon_bold = {}, {}
        for a in arms:
            mapping = NATIVE_TO_CANON[f"fs_{h}" if a.tool == "fs" else a.tool]
            canon_fine[a.name] = to_canon(native_fine[(a, h)], mapping)
            canon_bold[a.name] = plurality(canon_fine[a.name], cmap)
            _save(canon_fine[a.name], aff, paths["dir"] / f"{stem}_hemi-{h}_space-T1w_res-fine_desc-{a.name}_dseg.nii.gz")
            arr, baff = crop_coarse(canon_bold[a.name], cmap, bold_img.affine)
            _save(arr, baff, paths["dir"] / f"{stem}_hemi-{h}_space-T1w_res-bold_desc-{a.name}_dseg.nii.gz")
            # native volume: on the source grid, for this hemisphere only, so the fine column checks box + resampling
            src = nib.load(str(inputs[(a.name, h)]))
            src_canon = to_canon(np.asarray(src.dataobj).astype(np.int32), mapping)
            src_vox = float(abs(np.linalg.det(src.affine[:3, :3])))
            for r in REGIONS:
                if r in SUBFIELDS and not a.has_subfields:
                    continue
                vol_rows.append({"subject": subject, "hemi": h, "arm": a.name, "region": r,
                                 "native_mm3": float(region_mask(src_canon, r).sum()) * src_vox,
                                 "fine_mm3": float(region_mask(canon_fine[a.name], r).sum()) * fine_vox,
                                 "bold_mm3": float(region_mask(canon_bold[a.name], r).sum()) * bold_vox})
        for grid, canon in (("fine", canon_fine), ("bold", canon_bold)):
            dice_rows += [{"subject": subject, "hemi": h, "grid": grid, **row} for row in pair_rows(canon, arms)]
        assign_rows += [{"subject": subject, "hemi": h, "grid": "bold", **row} for row in assignment_rows(canon_bold, arms)]
        agree = np.sum([region_mask(canon_fine[a.name], "hipp") for a in arms], axis=0).astype(np.uint8)
        _save(agree, aff, paths["dir"] / f"{stem}_hemi-{h}_space-T1w_res-fine_desc-hippAgreement_mask.nii.gz")
        montage_inputs[h] = (canon_fine, t2w_fine.get(("SPC", h), next(iter(t2w_fine.values()))))

    pd.DataFrame(vol_rows).to_csv(paths["volumes"], sep="\t", index=False, float_format="%.6g")
    pd.DataFrame(dice_rows).to_csv(paths["dice"], sep="\t", index=False, float_format="%.6g")
    pd.DataFrame(assign_rows).to_csv(paths["assignment"], sep="\t", index=False, float_format="%.6g")
    pd.DataFrame(reg_rows).to_csv(paths["registration"], sep="\t", index=False, float_format="%.6g")
    montage(montage_inputs, arms, paths["montage"], title=f"sub-{subject}")

    sidecar = {
        "schema_version": SCHEMA_VERSION,
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        **provenance,
        "subject": subject,
        "arms": [{"name": a.name, "tool": a.tool, "tree": f"derivatives/{a.tree}", "t2w_acq": a.t2w_acq,
                  "version": tree_version(Path(deriv) / a.tree, Path(deriv)),
                  "inputs": {h: _rel(inputs[(a.name, h)], bids_root) for h in HEMIS}} for a in arms],
        "fmriprep": {"t1w": _rel(t1w, bids_root), "orig_to_t1w": _rel(orig_to_t1w, bids_root),
                     "bold_reference": _rel(bold_ref, bids_root), "version": tree_version(fmriprep_tree, Path(deriv))},
        "t2w": {q: _rel(t2w[q], bids_root) for q in acqs},
        "grids": {"fine": {"spacing_mm": FINE_MM, "box": f"aseg hippocampus per hemisphere + {BOX_PAD_MM} mm",
                           "interp": "ANTs GenericLabel, one resampling per arm"},
                  "bold": {"zooms_mm": [float(z) for z in bold_img.header.get_zooms()[:3]], "grids": bold_grids,
                           "rule": "plurality of fine-voxel canonical labels per BOLD voxel, background counted, "
                                   "ties to the lower class"}},
        "canon": CANON, "native_to_canon": {k: {str(n): c for n, c in v.items()} for k, v in NATIVE_TO_CANON.items()},
        "regions": {k: list(v) for k, v in REGIONS.items()},
        "transforms": {"hippunfold": "fMRIPrep run-01 from-orig_to-T1w, forward",
                       "hsf": "ANTs rigid MI, T2w -> fMRIPrep T1w (brain mask on fixed)", "fs": "identity"},
        "ants_rigid_mi": ac.ANTS_RIGID_MI,
    }
    paths["sidecar"].write_text(json.dumps(sidecar, indent=2) + "\n")
    log(f"sub-{subject}: wrote {paths['dir']}")
    return paths


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def montage(per_hemi: dict, arms: Sequence[Arm], out: Path, title: str, n_slices: int = 6) -> Path:
    """Coronal slices through the hippocampus, one row per arm, both hemispheres, canonical labels over the T2w."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    cmap = ListedColormap(["#000000"] + [CANON_COLOURS[k] for k in range(1, N_CLASSES)])
    cols = []
    for h in HEMIS:
        canon, bg = per_hemi[h]
        union = np.any([canon[a.name] > 0 for a in arms], axis=0)
        ys = np.flatnonzero(union.any(axis=(0, 2)))
        xs, zs = np.flatnonzero(union.any(axis=(1, 2))), np.flatnonzero(union.any(axis=(0, 1)))
        pad = int(round(3 / FINE_MM))
        xr = slice(max(xs[0] - pad, 0), xs[-1] + pad + 1)
        zr = slice(max(zs[0] - pad, 0), zs[-1] + pad + 1)
        for y in np.linspace(ys[0], ys[-1], n_slices + 2)[1:-1].round().astype(int):
            cols.append((h, y, xr, zr))
    fig, axes = plt.subplots(len(arms), len(cols), figsize=(1.6 * len(cols), 1.5 * len(arms) + 0.8), squeeze=False)
    for j, (h, y, xr, zr) in enumerate(cols):
        canon, bg = per_hemi[h]
        b = bg[xr, y, zr].T
        lo, hi = np.percentile(bg[bg > 0], [1, 99.5]) if (bg > 0).any() else (0, 1)
        for i, a in enumerate(arms):
            ax = axes[i, j]
            ax.imshow(b, cmap="gray", vmin=lo, vmax=hi, origin="lower", interpolation="nearest")
            lab = canon[a.name][xr, y, zr].T
            ax.imshow(np.ma.masked_equal(lab, 0), cmap=cmap, vmin=0, vmax=N_CLASSES - 1, alpha=0.55,
                      origin="lower", interpolation="nearest")
            ax.set_xticks([]), ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(a.name, fontsize=9)
            if i == 0:
                ax.set_title(f"{h}  y={y * FINE_MM:.0f} mm", fontsize=8)
    handles = [Patch(color=CANON_COLOURS[k], label=CANON[k] + (" (FS)" if CANON[k] == "hipp" else "")) for k in range(1, N_CLASSES)]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=8, frameon=False)
    fig.suptitle(f"{title}: canonical labels over T2w (SPC); y = mm from the posterior box edge (tail -> head)", fontsize=10)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Tree
# ---------------------------------------------------------------------------

def ensure_dataset_description(out_root: Path, provenance: dict) -> Path:
    desc = Path(out_root) / "dataset_description.json"
    if desc.exists():
        return desc
    desc.parent.mkdir(parents=True, exist_ok=True)
    desc.write_text(json.dumps({
        "Name": "Hippocampal segmentation comparison (internal)",
        "BIDSVersion": "1.10.0",
        "DatasetType": "derivative",
        "GeneratedBy": [{"Name": "compare_hippo_segmentations", "Version": provenance.get("code_version", "unknown"),
                         "CodeURL": "https://github.com/hulacon/mmmdata"}],
        "SourceDatasets": [{"URL": f"derivatives/{t}"} for t in
                           ("hippunfold", "hippunfold_oblcor", "hippunfold_v2", "hippunfold_v2_oblcor", "hsf",
                            "hsf_oblcor", "fmriprep")],
        "Description": "INTERNAL-ONLY, not part of the public release. HippUnfold 1.5.2/2.2.0, HSF and FreeSurfer "
                       "aseg hippocampus resampled into fMRIPrep T1w space and compared (volumes, Dice, voxel "
                       "assignment on the BOLD grid).",
    }, indent=2) + "\n")
    return desc


def collect(out_root: Path) -> dict[str, pd.DataFrame]:
    """Concatenate every subject's tables."""
    out = {}
    for k in ("volumes", "dice", "assignment", "registration"):
        files = sorted(Path(out_root).glob(f"sub-*/ses-{SESSION}/anat/*_desc-{k}_stats.tsv"))
        if not files:
            raise FileNotFoundError(f"no *_desc-{k}_stats.tsv under {out_root}")
        out[k] = pd.concat([pd.read_csv(f, sep="\t", dtype={"subject": str}) for f in files], ignore_index=True)
    return out


#: Sequential ramp (one hue, light -> dark) for Dice matrices.
DICE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def group_figures(tables: dict[str, pd.DataFrame], out_dir: Path, arms: Sequence[Arm] = ARMS) -> list[Path]:
    """Median pairwise Dice per region (one figure per grid) and per-arm volumes, across every hemisphere."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    names = [a.name for a in arms]
    ramp = LinearSegmentedColormap.from_list("dice", DICE_RAMP)
    dice_df, vol = tables["dice"], tables["volumes"]
    n_hemi = dice_df[["subject", "hemi"]].drop_duplicates().shape[0]
    written = []
    for grid in ("fine", "bold"):
        d = dice_df[dice_df["grid"] == grid]
        regions = list(REGIONS)
        fig, axes = plt.subplots(1, len(regions), figsize=(3.3 * len(regions), 3.9))
        for ax, r in zip(axes, regions):
            these = [n for n in names if r not in SUBFIELDS or next(a for a in arms if a.name == n).has_subfields]
            med = d[d["region"] == r].groupby(["arm_a", "arm_b"])["dice"].median()
            # lower triangle without its empty first row / last column: rows these[1:], columns these[:-1]
            rows, cols = these[1:], these[:-1]
            m = np.full((len(rows), len(cols)), np.nan)
            for i, a in enumerate(rows):
                for j, b in enumerate(cols):
                    if these.index(b) < these.index(a):
                        m[i, j] = med.get((b, a), med.get((a, b), np.nan))
            ax.imshow(np.ma.masked_invalid(m), cmap=ramp, vmin=0, vmax=1)
            for i in range(len(rows)):
                for j in range(len(cols)):
                    if np.isfinite(m[i, j]):
                        ax.text(j, i, f"{m[i, j]:.2f}"[1:] if m[i, j] < 1 else "1", ha="center", va="center",
                                fontsize=8, color="#ffffff" if m[i, j] > 0.55 else "#1a1a19")
            ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right", fontsize=8)
            ax.set_yticks(range(len(rows)), rows, fontsize=8)
            ax.set_title(r, fontsize=10)
            for s in ax.spines.values():
                s.set_visible(False)
        fig.suptitle(f"Median pairwise Dice, {grid} grid ({'0.3 mm' if grid == 'fine' else 'BOLD, plurality label'}), "
                     f"{n_hemi} hemispheres; hipp = HippUnfold incl. SRLM", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        p = Path(out_dir) / f"group_desc-dice{grid.capitalize()}_hippcompare.png"
        fig.savefig(p, dpi=120, bbox_inches="tight")
        plt.close(fig)
        written.append(p)

    regions = ["hipp", "hippNoSRLM", *SUBFIELDS]
    fig, axes = plt.subplots(1, len(regions), figsize=(2.6 * len(regions), 3.4), sharey=False)
    for ax, r in zip(axes, regions):
        v = vol[vol["region"] == r]
        shown = [n for n in names if (v["arm"] == n).any()]
        for k, n in enumerate(shown):
            y = v.loc[v["arm"] == n, "fine_mm3"].to_numpy()
            if len(y):
                x = k + np.linspace(-0.18, 0.18, len(y))
                ax.scatter(x, y, s=14, color="#2a78d6", edgecolors="#ffffff", linewidths=0.6, zorder=3)
                ax.plot([k - 0.3, k + 0.3], [np.median(y)] * 2, color="#1a1a19", lw=1.2, zorder=4)
        ax.set_xticks(range(len(shown)), shown, rotation=45, ha="right", fontsize=8)
        ax.set_title(r, fontsize=10)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="y", color="#e4e3df", lw=0.6, zorder=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("mm³ (fine grid)", fontsize=8)
    fig.suptitle(f"Volumes per arm, one dot per hemisphere ({n_hemi}), bar = median", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    p = Path(out_dir) / "group_desc-volumes_hippcompare.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    written.append(p)
    return written
