"""Hippocampal segmentations scored against NSD's manual MTL labels (accuracy, not agreement).

Companion to :mod:`neuroimaging.hippo_compare`, which measures agreement between tools on MMMData. Here the
reference is NSD's single-rater manual segmentation of the 7T high-resolution T2 (``HRT2``,
0.357 x 0.357 x 1.5 mm, oblique coronal), and every arm is scored against it on three grids:

* **hrt2** (primary): the manual labels' own grid, never resampled. Each arm reaches it once, from its finest
  output, onto the HRT2 grid supersampled ``SUBDIV`` (the 1.5 mm slice is thicker than any arm's grid), and
  each HRT2 voxel takes the **plurality** of its sub-voxels (exact block reduction).
* **t1_0p8** (cross-check): NSD's own transfer of the manual labels to the 0.8 mm T1 grid (``roi/{lh,rh}.MTL``);
  arms reach it by plurality from a 0.3 mm T1-axis grid.
* **bold**: a synthetic ``BOLD_ZOOMS`` grid on T1 axes at the eight ``BOLD_OFFSETS`` (half-voxel lattice).
  Manual labels and arms both reach it through the same 0.3 mm grid; label = plurality.

**Manual label coding.** ``HRT2/MTL_rawlabels`` holds values 1-20, not ``MTL.ctab`` codes: ctab label =
ceil(k / 2), **odd = right hemisphere, even = left** (:func:`decode_manual`). Hippocampus = SUB, CA1, CA2, CA3,
DG (head and body) + HT (tail, no subfields).

**HRT2 <-> T1.** NSD's HRT2 header is in the 7T session's scanner space and NSD ships no transform to the anat
grid. ``HRT2/T2matched`` is ``T2_0pt8`` after NSD's own rigid registration onto the HRT2 grid, so a rigid
``T2_0pt8 -> T2matched`` registration recovers it (:func:`register_t1_to_hrt2`; checked against ``roi/MTL``).

**Scoring rules** (fixed before any output was read; design record below):

* Whole hippocampus: manual SUB+CA1+CA2+CA3+DG+HT vs arm grey matter incl. SRLM (``hipp``) and excl.
  (``hippNoSRLM``); cyst is background. On T1-axis grids, restricted to the HRT2 field of view.
* Subfields (``SUBFIELDS``, incl. merged ``CA23``) inside the **domain**: per hemisphere, the HRT2 slices that
  hold any manual SUB/CA/DG label, minus manual HT voxels. Voxels an arm calls SRLM are dropped from both sides
  (FreeSurfer's molecular layer likewise). HippUnfold CA4 -> DG. ``sub_coverage`` = share of manual SUB the arm
  calls SUB (manual SUB includes presubiculum).
* Arm-M gate (``gate``): inside the domain, among voxels whose labels differ, the share where exactly one side
  calls the voxel hippocampus (arm SRLM counts as hippocampus; arm-SRLM-vs-manual-hippocampus voxels dropped).
* BOLD grid: per-subfield Dice, voxel-assignment agreement, **purity** (for voxels an arm labels X, mean
  sub-voxel share that is manual X) and voxel counts, in BOLD voxels whose majority lies in the domain.
* Sensitivity variant ``alt``: SRLM goes to the nearest non-SRLM arm label (instead of being dropped) and CA4
  goes to CA3 (instead of DG). Every metric is written for ``primary`` and ``alt``.

Design record: mmmdata-agents ``docs/workbench/hippunfold-eval/`` (settle 2b; DECIDED 2026-10-13).
"""

from __future__ import annotations

import datetime as _dt
import itertools
import json
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage

from . import alignment_candidates as ac
from . import hippo_compare as hc

SCHEMA_VERSION = "1.0"
#: HRT2 supersampling per axis (0.357, 0.357, 1.5 mm -> ~0.18, 0.18, 0.3 mm).
SUBDIV = (2, 2, 5)
#: Padding (mm) around the manual hippocampus for each hemisphere's box (both HRT2 and T1 boxes).
PAD_MM = 10.0
T1_FINE_MM = 0.3
BOLD_ZOOMS = (1.8, 1.8, 2.0)
#: BOLD lattice offsets, in BOLD voxels.
BOLD_OFFSETS = tuple(itertools.product((0.0, 0.5), repeat=3))
HEMIS = hc.HEMIS
SESSION = hc.SESSION
VARIANTS = ("primary", "alt")

CANON = {0: "bg", 1: "Sub", 2: "CA1", 3: "CA2", 4: "CA3", 5: "DG", 6: "SRLM", 7: "hipp", 8: "CA4", 9: "CA23"}
CANON_ID = {v: k for k, v in CANON.items()}
N_CLASSES = len(CANON)
SRLM, HIPP, CA4, CA23 = CANON_ID["SRLM"], CANON_ID["hipp"], CANON_ID["CA4"], CANON_ID["CA23"]
#: Canonical ids per scored subfield; CA23 also collects arms that only have a merged CA2/3 class.
SUBFIELDS = {"Sub": (1,), "CA1": (2,), "CA2": (3,), "CA3": (4,), "DG": (5,), "CA23": (3, 4, 9)}
#: Labels that count as a subfield in voxel-assignment agreement.
SUBFIELD_LABELS = (1, 2, 3, 4, 5, 9)
#: Manual hippocampal labels with subfields (domain definition).
MANUAL_SUBFIELD_IDS = (1, 2, 3, 4, 5)

#: Native label -> canonical, per source. Unlisted labels are background.
NATIVE_TO_CANON = {
    # multihist7: 1 Sub, 2 CA1, 3 CA2, 4 CA3, 5 CA4, 6 DG, 7 SRLM, 8 Cyst (CA4 kept apart until collapse())
    "hippunfold": {1: 1, 2: 2, 3: 3, 4: 4, 5: CA4, 6: 5, 7: SRLM},
    "hsf": hc.NATIVE_TO_CANON["hsf"],
    "fs_aseg_L": {17: HIPP},
    "fs_aseg_R": {53: HIPP},
    # FreeSurfer hippoSfLabels: para/pre/subiculum -> Sub; CA3 (holds CA2) -> CA23; molecular layer = the lumped
    # layer, treated like SRLM; HATA, fimbria, fissure -> background
    "fs_sf": {203: 1, 204: 1, 205: 1, 206: 2, 208: CA23, 209: CA4, 210: 5, 214: SRLM, 226: HIPP},
    # MTL.ctab: 4 SUB, 5 CA1, 6 CA2, 7 CA3, 8 DG, 10 HT; ERC/BA35/BA36/PHC -> background
    "manual": {4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 10: HIPP},
}
#: Montage colours (hippo_compare's slots 1-7 + two more for CA4 and CA23).
CANON_COLOURS = {**hc.CANON_COLOURS, 8: "#a35f2b", 9: "#5c5c5c"}


@dataclass(frozen=True)
class NsdArm:
    name: str
    tool: str                    # hippunfold | hsf | fs_aseg | fs_sf
    tree: Optional[str] = None   # relative to the derivatives root (hippunfold, hsf)
    t2w_acq: Optional[str] = None
    expresses: tuple = field(default=("Sub", "CA1", "CA2", "CA3", "DG", "CA23"))

    @property
    def has_subfields(self) -> bool:
        return bool(self.expresses)

    @property
    def native_space(self) -> str:
        return "hrt2" if self.t2w_acq == "HRT2" and self.tool == "hsf" else "t1"


ARMS = (
    NsdArm("U1iso", "hippunfold", "hippunfold_iso/hippunfold", "iso"),
    NsdArm("U1hr", "hippunfold", "hippunfold_HRT2/hippunfold", "HRT2"),
    NsdArm("U2iso", "hippunfold", "hippunfold_v2_iso", "iso"),
    NsdArm("U2hr", "hippunfold", "hippunfold_v2_HRT2", "HRT2"),
    NsdArm("Hiso", "hsf", "hsf_iso", "iso"),
    NsdArm("Hhr", "hsf", "hsf_HRT2", "HRT2"),
    NsdArm("FSaseg", "fs_aseg", expresses=()),
    NsdArm("FSsf", "fs_sf", expresses=("Sub", "CA1", "DG", "CA23")),
)


# ---------------------------------------------------------------------------
# Pure helpers (tested on synthetic data)
# ---------------------------------------------------------------------------

def decode_manual(raw: np.ndarray) -> dict[str, np.ndarray]:
    """``MTL_rawlabels`` (1-20) -> canonical labels per hemisphere: ctab = ceil(k/2), odd = R, even = L."""
    raw = np.asarray(raw).astype(np.int64)
    ctab = np.where(raw > 0, (raw + 1) // 2, 0)
    out = {}
    for h, parity in (("L", 0), ("R", 1)):
        out[h] = hc.to_canon(np.where((raw > 0) & (raw % 2 == parity), ctab, 0), NATIVE_TO_CANON["manual"])
    return out


def collapse(canon: np.ndarray, variant: str, spacing: Sequence[float] = (1.0, 1.0, 1.0)) -> np.ndarray:
    """Apply a scoring variant to an arm's canonical labels.

    ``primary``: CA4 -> DG; SRLM kept (scoring drops it). ``alt``: CA4 -> CA3; each SRLM voxel takes the label of
    the nearest non-SRLM hippocampal voxel of the same arm (Euclidean, ``spacing`` in mm).
    """
    out = np.asarray(canon).copy()
    if variant == "primary":
        out[out == CA4] = 5
        return out
    if variant != "alt":
        raise ValueError(f"unknown variant {variant!r}")
    out[out == CA4] = 4
    srlm = out == SRLM
    src = (out > 0) & ~srlm
    if srlm.any() and src.any():
        nz = np.argwhere(out > 0)
        lo, hi = nz.min(0), nz.max(0) + 1
        sl = tuple(slice(a, b) for a, b in zip(lo, hi))
        sub, s_src, s_srlm = out[sl], src[sl], srlm[sl]
        _, idx = ndimage.distance_transform_edt(~s_src, sampling=spacing, return_indices=True)
        sub[s_srlm] = sub[tuple(i[s_srlm] for i in idx)]
    return out


def express(labels: np.ndarray, arm: NsdArm) -> np.ndarray:
    """Labels at the resolution ``arm`` can express: CA2/CA3 merge to CA23 for an arm without them apart."""
    if "CA2" in arm.expresses:
        return labels
    out = np.asarray(labels).copy()
    out[np.isin(out, (3, 4))] = CA23
    return out


def subfield_domain(manual: np.ndarray, slice_axis: int = 2) -> np.ndarray:
    """Slices (along ``slice_axis``) holding any manual subfield label, minus manual tail (HT) voxels."""
    has = np.isin(manual, MANUAL_SUBFIELD_IDS)
    other = tuple(a for a in range(manual.ndim) if a != slice_axis)
    sl = has.any(axis=other)
    shape = [1] * manual.ndim
    shape[slice_axis] = -1
    return np.broadcast_to(sl.reshape(shape), manual.shape) & (manual != HIPP)


def supersample(affine: np.ndarray, shape: Sequence[int], subdiv: Sequence[int]) -> tuple[np.ndarray, tuple[int, ...]]:
    """A grid that splits every voxel of (affine, shape) into ``subdiv`` sub-voxels, exactly nested."""
    n = np.asarray(subdiv, float)
    out = affine.copy()
    out[:3, :3] = affine[:3, :3] / n            # scale each column
    out[:3, 3] = nib.affines.apply_affine(affine, -0.5 + 0.5 / n)
    return out, tuple(int(s) * int(k) for s, k in zip(shape, subdiv))


def block_counts(fine: np.ndarray, subdiv: Sequence[int], n_classes: int = N_CLASSES) -> np.ndarray:
    """Per coarse voxel, the count of each label among its sub-voxels: shape (*coarse, n_classes)."""
    nx, ny, nz = subdiv
    X, Y, Z = (s // k for s, k in zip(fine.shape, subdiv))
    b = fine.reshape(X, nx, Y, ny, Z, nz)
    return np.stack([(b == k).sum(axis=(1, 3, 5), dtype=np.int16) for k in range(n_classes)], axis=-1)


def counts_plurality(counts: np.ndarray) -> np.ndarray:
    """Plurality label from a counts array (ties -> lower class, background first)."""
    return counts.argmax(-1).astype(np.uint8)


def map_counts(fine: np.ndarray, cmap: hc.CoarseMap, n_classes: int = N_CLASSES) -> np.ndarray:
    """Per touched coarse voxel (``cmap.flat`` order), the count of each label among fine voxels in it."""
    labels = fine.ravel()[cmap.keep].astype(np.int64)
    c = np.bincount(cmap.group * n_classes + labels, minlength=len(cmap.flat) * n_classes)
    return c.reshape(len(cmap.flat), n_classes)


def region_ids(region: str) -> tuple:
    if region == "hipp":
        return tuple(k for k in CANON if k)
    if region == "hippNoSRLM":
        return tuple(k for k in CANON if k and k != SRLM)
    return SUBFIELDS[region]


def score_rows(manual: np.ndarray, arm_lab: np.ndarray, domain: np.ndarray, arm: NsdArm,
               fov: Optional[np.ndarray] = None) -> list[dict]:
    """Whole-hippocampus Dice (inside ``fov``), subfield Dice and SUB coverage (inside ``domain``, arm-SRLM dropped)."""
    rows = []
    in_fov = np.ones(manual.shape, bool) if fov is None else fov
    m_h = (manual > 0) & in_fov
    for r in ("hipp", "hippNoSRLM"):
        a = np.isin(arm_lab, region_ids(r)) & in_fov
        rows.append({"metric": "dice", "region": r, "value": hc.dice(a, m_h), "n_arm": int(a.sum()),
                     "n_manual": int(m_h.sum())})
    if not arm.has_subfields:
        return rows
    keep = domain & (arm_lab != SRLM)
    for s in arm.expresses:
        ids = SUBFIELDS[s]
        a, m = np.isin(arm_lab, ids) & keep, np.isin(manual, ids) & keep
        rows.append({"metric": "dice", "region": s, "value": hc.dice(a, m), "n_arm": int(a.sum()), "n_manual": int(m.sum())})
    a, m = (arm_lab == 1) & keep, (manual == 1) & keep
    rows.append({"metric": "sub_coverage", "region": "Sub", "value": float((a & m).sum() / m.sum()) if m.any() else np.nan,
                 "n_arm": int(a.sum()), "n_manual": int(m.sum())})
    return rows


def assignment_row(manual: np.ndarray, arm_lab: np.ndarray, keep: np.ndarray, arm: NsdArm) -> dict:
    """Same subfield, among kept voxels that either side calls a subfield (labels at the arm's resolution)."""
    a, m = express(arm_lab, arm), express(manual, arm)
    either = keep & (np.isin(a, SUBFIELD_LABELS) | np.isin(m, SUBFIELD_LABELS))
    same = either & (a == m)
    return {"n_either": int(either.sum()), "agree": float(same.sum() / either.sum()) if either.any() else np.nan}


def gate_row(manual: np.ndarray, arm_lab: np.ndarray, domain: np.ndarray, arm: NsdArm) -> dict:
    """Tissue share of label disagreements inside the domain (arm SRLM = hippocampus; SRLM-vs-manual-hipp dropped)."""
    a, m = express(arm_lab, arm), express(manual, arm)
    considered = domain & ~((arm_lab == SRLM) & (manual > 0))
    differ = considered & (a != m)
    tissue = differ & ((a > 0) != (m > 0))
    return {"n_differ": int(differ.sum()), "n_tissue": int(tissue.sum()),
            "tissue_share": float(tissue.sum() / differ.sum()) if differ.any() else np.nan}


def purity_rows(manual_counts: np.ndarray, arm_coarse: np.ndarray, manual_coarse: np.ndarray, keep: np.ndarray,
                arm: NsdArm) -> list[dict]:
    """Per subfield: voxel counts and purity = mean share of manual-X sub-voxels in the voxels the arm labels X."""
    total = manual_counts.sum(-1)
    rows = []
    for s in arm.expresses:
        ids = SUBFIELDS[s]
        sel = keep & np.isin(arm_coarse, ids) & (total > 0)
        share = manual_counts[sel][:, list(ids)].sum(-1) / total[sel]
        rows.append({"region": s, "n_vox_arm": int(sel.sum()),
                     "n_vox_manual": int((keep & np.isin(manual_coarse, ids)).sum()),
                     "purity": float(share.mean()) if sel.any() else np.nan})
    return rows


def bold_affine(t1_affine: np.ndarray, zooms: Sequence[float], offset: Sequence[float]) -> np.ndarray:
    """A grid on ``t1_affine``'s axes at ``zooms`` mm whose voxel 0 sits at the T1 origin + ``offset`` voxels."""
    rot = t1_affine[:3, :3] / np.linalg.norm(t1_affine[:3, :3], axis=0)
    out = np.eye(4)
    out[:3, :3] = rot * np.asarray(zooms)
    out[:3, 3] = t1_affine[:3, 3] + out[:3, :3] @ np.asarray(offset, float)
    return out


def coarse_grid_over(fine_shape: Sequence[int], fine_affine: np.ndarray, coarse_affine: np.ndarray) -> tuple:
    """Shift ``coarse_affine`` by whole voxels so its grid (returned shape) covers the fine box."""
    corners = np.array(list(itertools.product(*[(-0.5, s - 0.5) for s in fine_shape])))
    c = nib.affines.apply_affine(np.linalg.inv(coarse_affine) @ fine_affine, corners)
    lo, hi = np.floor(c.min(0)).astype(int) - 1, np.ceil(c.max(0)).astype(int) + 1
    aff = coarse_affine.copy()
    aff[:3, 3] = nib.affines.apply_affine(coarse_affine, lo)
    return aff, tuple(int(x) for x in hi - lo + 1)


# ---------------------------------------------------------------------------
# Inputs and registration
# ---------------------------------------------------------------------------

def nsd_subject(subject: str) -> str:
    """BIDS label ``0N`` -> NSD ``subj0N``."""
    return f"subj{int(subject):02d}"


def arm_inputs(arm: NsdArm, deriv: Path, ppdata: Path, subject: str) -> dict[str, Path]:
    """Native label file per hemisphere (FS arms: one whole-brain file, the same for both)."""
    if arm.tool in ("hippunfold", "hsf"):
        a = hc.Arm(arm.name, arm.tool, arm.tree, arm.t2w_acq)
        return {h: hc.arm_input(a, deriv, subject, h) for h in HEMIS}
    name = {"fs_aseg": "aseg_0pt8.nii.gz", "fs_sf": "hippoSfLabels_0pt8.nii.gz"}[arm.tool]
    p = Path(ppdata) / nsd_subject(subject) / "anat" / name
    if not p.exists():
        raise FileNotFoundError(p)
    return {h: p for h in HEMIS}


def register_t1_to_hrt2(ppdata: Path, subject: str, out: Path, threads: int = 1) -> Path:
    """Rigid ``T2_0pt8 -> HRT2/T2matched`` (fixed = HRT2 grid): resamples T1-space images onto HRT2 forward."""
    a = Path(ppdata) / nsd_subject(subject) / "anat"
    fixed, moving = a / "HRT2" / "T2matched.nii.gz", a / "T2_0pt8_masked.nii.gz"
    stages = ["--dimensionality", "3", "--float", "1", "--collapse-output-transforms", "1", "--random-seed", "1",
              "--interpolation", "Linear", "--winsorize-image-intensities", "[0.005,0.995]",
              "--initial-moving-transform", "[{fixed},{moving},1]",
              "--transform", "Rigid[0.1]", "--metric", "MI[{fixed},{moving},1,32,Regular,0.25]",
              "--convergence", "[1000x500x250x100,1e-8,10]", "--shrink-factors", "8x4x2x1",
              "--smoothing-sigmas", "3x2x1x0vox"]
    with tempfile.TemporaryDirectory(prefix=f"t1hrt2_{subject}_") as tmp:
        pre = ac.ants_register(fixed, moving, Path(tmp) / "x_", stages, fixed_mask=a / "HRT2" / "HRT2_mask.nii.gz",
                               threads=threads)
        out.write_bytes(Path(f"{pre}0GenericAffine.mat").read_bytes())
    return out


# ---------------------------------------------------------------------------
# Subject
# ---------------------------------------------------------------------------

def subject_paths(out_root: Path, subject: str) -> dict[str, Path]:
    d = Path(out_root) / f"sub-{subject}"
    stem = f"sub-{subject}"
    return {"dir": d, **{k: d / f"{stem}_desc-{k}_stats.tsv" for k in
                         ("volumes", "scores", "assignment", "purity", "gate", "registration")},
            "xfm": d / f"{stem}_from-T1w_to-HRT2_mode-image_xfm.mat",
            "sidecar": d / f"{stem}_desc-hippnsd.json", "montage": d / f"{stem}_desc-montage_hippnsd.png"}


def _save(arr: np.ndarray, affine: np.ndarray, path: Path) -> Path:
    return hc._save(arr, affine, path)


def _hemi_split(native: Path, hemi: str, tmp: Path) -> Path:
    """A whole-brain label image with the other hemisphere zeroed (world x: L < 0 < R)."""
    img = nib.load(str(native))
    data = np.asarray(img.dataobj).astype(np.int32)
    x = nib.affines.apply_affine(img.affine, np.indices(data.shape[:3]).reshape(3, -1).T)[:, 0].reshape(data.shape[:3])
    data[(x > 0) if hemi == "L" else (x < 0)] = 0
    return _save(data, img.affine, tmp / f"{Path(native).name.split('.')[0]}_{hemi}.nii.gz")


def build_subject(out_root: Path, deriv: Path, ppdata: Path, subject: str, provenance: dict,
                  arms: Sequence[NsdArm] = ARMS, n_jobs: int = 1, log: Callable[[str], None] = print) -> dict[str, Path]:
    """Score every arm against the manual labels for one NSD subject (both hemispheres, all grids, both variants)."""
    paths = subject_paths(out_root, subject)
    paths["dir"].mkdir(parents=True, exist_ok=True)
    anat = Path(ppdata) / nsd_subject(subject) / "anat"
    hrt2_img = nib.load(str(anat / "HRT2" / "HRT2_raw.nii.gz"))
    raw_img = nib.load(str(anat / "HRT2" / "MTL_rawlabels.nii.gz"))
    if not np.allclose(raw_img.affine, hrt2_img.affine) or raw_img.shape != hrt2_img.shape:
        raise ValueError("MTL_rawlabels is not on the HRT2_raw grid")
    t1_img = nib.load(str(anat / "T1_0pt8_masked.nii.gz"))
    manual_hrt2 = decode_manual(np.asarray(raw_img.dataobj))
    inputs = {a.name: arm_inputs(a, deriv, ppdata, subject) for a in arms}
    slice_axis = int(np.argmax(hrt2_img.header.get_zooms()[:3]))
    hrt2_zooms = np.asarray(hrt2_img.header.get_zooms()[:3], float)
    fine_hrt2_spacing = hrt2_zooms / np.asarray(SUBDIV)

    def pmap(fn, items):
        with ThreadPoolExecutor(max_workers=max(1, n_jobs)) as ex:
            return list(ex.map(fn, items))

    log(f"sub-{subject}: registering T1 space -> HRT2")
    xfm = register_t1_to_hrt2(ppdata, subject, paths["xfm"], threads=max(1, n_jobs))

    score_rows_all, assign_rows, purity_all, gate_rows, vol_rows, reg_rows = [], [], [], [], [], []
    montage_inputs = {}
    with tempfile.TemporaryDirectory(prefix=f"hippnsd_{subject}_") as tmp:
        tmp = Path(tmp)
        # registration check: T2 through the recovered xfm vs T2matched, and manual labels vs NSD's roi/MTL
        warped = hc.ants_resample(anat / "T2_0pt8_masked.nii.gz", anat / "HRT2" / "T2matched.nii.gz", [str(xfm)],
                                  tmp / "t2_on_hrt2.nii.gz", labels=False)
        W = np.asarray(nib.load(str(warped)).dataobj)
        T = np.asarray(nib.load(str(anat / "HRT2" / "T2matched.nii.gz")).dataobj)
        M = np.asarray(nib.load(str(anat / "HRT2" / "HRT2_mask.nii.gz")).dataobj) > 0
        m = M & (W > 0) & (T > 0)
        reg_rows.append({"subject": subject, "check": "r_T2_vs_T2matched", "hemi": "", "region": "",
                         "value": float(np.corrcoef(W[m], T[m])[0, 1])})

        # FS arms are whole-brain: split by hemisphere first
        native = {}
        for a in arms:
            for h in HEMIS:
                native[(a.name, h)] = _hemi_split(inputs[a.name][h], h, tmp) if a.tool == "fs_sf" else inputs[a.name][h]

        for h in HEMIS:
            man = manual_hrt2[h]
            if not (man > 0).any():
                raise ValueError(f"sub-{subject} hemi-{h}: no manual hippocampal labels")
            # ---- HRT2 box + supersampled grid
            zooms = hrt2_zooms
            pad = np.ceil(PAD_MM / zooms).astype(int)
            ijk = np.argwhere(man > 0)
            lo = np.maximum(ijk.min(0) - pad, 0)
            hi = np.minimum(ijk.max(0) + pad + 1, man.shape)
            box = tuple(slice(a_, b_) for a_, b_ in zip(lo, hi))
            box_aff = hrt2_img.affine.copy()
            box_aff[:3, 3] = nib.affines.apply_affine(hrt2_img.affine, lo)
            box_shape = tuple(int(x) for x in hi - lo)
            fine_aff, fine_shape = supersample(box_aff, box_shape, SUBDIV)
            ref_hrt2 = _save(np.zeros(fine_shape, np.uint8), fine_aff, tmp / f"ref_hrt2_{h}.nii.gz")
            man_box = man[box]
            dom_box = subfield_domain(man_box, slice_axis)
            # manual + domain + FOV saved on the HRT2 box for transfer to T1 space
            man_p = _save(man_box.astype(np.uint8), box_aff, tmp / f"man_{h}.nii.gz")
            dom_p = _save(dom_box.astype(np.uint8), box_aff, tmp / f"dom_{h}.nii.gz")
            fov_p = _save(np.ones(man.shape, np.uint8), hrt2_img.affine, tmp / f"fov_{h}.nii.gz")

            # ---- T1 0.3 mm box around NSD's own T1-grid manual hippocampus
            roi = nib.load(str(anat / "roi" / f"{'lh' if h == 'L' else 'rh'}.MTL.nii.gz"))
            if not np.allclose(roi.affine, t1_img.affine):
                raise ValueError("roi/MTL is not on the T1_0pt8 grid")
            roi_canon = hc.to_canon(np.asarray(roi.dataobj).astype(np.int32), NATIVE_TO_CANON["manual"])
            t1f_aff, t1f_shape = hc.box_affine(roi_canon > 0, roi.affine, T1_FINE_MM, PAD_MM)
            ref_t1 = _save(np.zeros(t1f_shape, np.uint8), t1f_aff, tmp / f"ref_t1_{h}.nii.gz")

            def to_grid(job):
                a, grid = job
                src = native[(a.name, h)]
                if grid == "hrt2":
                    ref, chain = ref_hrt2, ([str(xfm)] if a.native_space == "t1" else [])
                else:
                    ref, chain = ref_t1, ([] if a.native_space == "t1" else [f"[{xfm},1]"])
                out = hc.ants_resample(src, ref, chain, tmp / f"{a.name}_{h}_{grid}.nii.gz", labels=True)
                key = f"fs_aseg_{h}" if a.tool == "fs_aseg" else a.tool
                return job, hc.to_canon(np.asarray(nib.load(str(out)).dataobj).astype(np.int32), NATIVE_TO_CANON[key])

            arm_fine = dict(pmap(to_grid, [(a, g) for a in arms for g in ("hrt2", "t1")]))

            def ref_to_t1(p):
                out = hc.ants_resample(p, ref_t1, [f"[{xfm},1]"], tmp / f"{Path(p).name.split('.')[0]}_t1.nii.gz", labels=True)
                return np.asarray(nib.load(str(out)).dataobj).astype(np.uint8)
            man_t1, dom_t1, fov_t1 = pmap(ref_to_t1, [man_p, dom_p, fov_p])
            dom_t1, fov_t1 = dom_t1 > 0, fov_t1 > 0
            log(f"sub-{subject} hemi-{h}: {len(arm_fine)} arm resamplings done")

            # registration check 2: manual pushed to 0.8 mm vs NSD's roi/MTL
            cm08 = hc.coarse_map(t1f_shape, t1f_aff, roi.shape[:3], roi.affine)
            man08 = np.zeros(roi.shape[:3], np.uint8)
            man08.ravel()[cm08.flat] = counts_plurality(map_counts(man_t1, cm08))
            for s in ("Sub", "CA1", "CA2", "CA3", "DG"):
                reg_rows.append({"subject": subject, "check": "dice_manual_vs_roiMTL", "hemi": h, "region": s,
                                 "value": hc.dice(np.isin(man08, SUBFIELDS[s]), np.isin(roi_canon, SUBFIELDS[s]))})
            reg_rows.append({"subject": subject, "check": "dice_manual_vs_roiMTL", "hemi": h, "region": "hipp",
                             "value": hc.dice(man08 > 0, roi_canon > 0)})

            # ---- coarse maps for the T1-axis grids
            dom08 = np.zeros(roi.shape[:3], bool)
            dom08.ravel()[cm08.flat] = counts_plurality(map_counts(dom_t1.astype(np.uint8), cm08, 2)) > 0
            fov08 = np.zeros(roi.shape[:3], bool)
            fov08.ravel()[cm08.flat] = counts_plurality(map_counts(fov_t1.astype(np.uint8), cm08, 2)) > 0
            touched08 = np.zeros(roi.shape[:3], bool)
            touched08.ravel()[cm08.flat] = True
            bold_maps = []
            for off in BOLD_OFFSETS:
                baff, bshape = coarse_grid_over(t1f_shape, t1f_aff, bold_affine(t1_img.affine, BOLD_ZOOMS, off))
                cmb = hc.coarse_map(t1f_shape, t1f_aff, bshape, baff)
                bold_maps.append((off, cmb, map_counts(man_t1, cmb),
                                  counts_plurality(map_counts(dom_t1.astype(np.uint8), cmb, 2)) > 0,
                                  counts_plurality(map_counts(fov_t1.astype(np.uint8), cmb, 2)) > 0))

            for a in arms:
                for variant in VARIANTS:
                    a_hrt2 = collapse(arm_fine[(a, "hrt2")], variant, fine_hrt2_spacing)
                    a_t1 = collapse(arm_fine[(a, "t1")], variant, (T1_FINE_MM,) * 3)
                    base = {"subject": subject, "hemi": h, "arm": a.name, "variant": variant}
                    # hrt2 grid (primary)
                    a_cnt = block_counts(a_hrt2, SUBDIV)
                    a_co = counts_plurality(a_cnt)
                    for row in score_rows(man_box, a_co, dom_box, a):
                        score_rows_all.append({**base, "grid": "hrt2", "offset": "", **row})
                    if a.has_subfields:
                        keep = dom_box & (a_co != SRLM)
                        assign_rows.append({**base, "grid": "hrt2", "offset": "", **assignment_row(man_box, a_co, keep, a)})
                        gate_rows.append({**base, **gate_row(man_box, a_co, dom_box, a)})
                    # 0.8 mm cross-check
                    a08 = np.zeros(roi.shape[:3], np.uint8)
                    a08.ravel()[cm08.flat] = counts_plurality(map_counts(a_t1, cm08))
                    for row in score_rows(np.where(touched08, roi_canon, 0), a08, dom08 & touched08, a, fov=fov08):
                        score_rows_all.append({**base, "grid": "t1_0p8", "offset": "", **row})
                    # BOLD grids
                    for off, cmb, m_cnt, dom_b, fov_b in bold_maps:
                        m_co = counts_plurality(m_cnt)
                        a_cb = counts_plurality(map_counts(a_t1, cmb))
                        o = "".join("h" if x else "0" for x in off)
                        for row in score_rows(m_co, a_cb, dom_b, a, fov=fov_b):
                            score_rows_all.append({**base, "grid": "bold", "offset": o, **row})
                        if a.has_subfields:
                            keep = dom_b & (a_cb != SRLM)
                            assign_rows.append({**base, "grid": "bold", "offset": o, **assignment_row(m_co, a_cb, keep, a)})
                            m_cnt_e = m_cnt
                            if "CA2" not in a.expresses:   # manual CA2/CA3 counts pooled into CA23 for this arm
                                m_cnt_e = m_cnt.copy()
                                m_cnt_e[:, CA23] += m_cnt_e[:, 3] + m_cnt_e[:, 4]
                                m_cnt_e[:, 3] = m_cnt_e[:, 4] = 0
                            for row in purity_rows(m_cnt_e, express(a_cb, a), express(m_co, a), keep, a):
                                purity_all.append({**base, "offset": o, **row})
                    if variant == "primary":
                        fine_vox_hrt2 = float(np.prod(fine_hrt2_spacing))
                        for r in ["hipp", "hippNoSRLM", *(a.expresses or ())]:
                            ids = region_ids(r)
                            t1_mm3 = float(np.isin(a_t1, ids).sum()) * T1_FINE_MM ** 3
                            in_fov = float((np.isin(a_t1, ids) & fov_t1).sum()) * T1_FINE_MM ** 3
                            vol_rows.append({"subject": subject, "hemi": h, "arm": a.name, "region": r,
                                             "hrt2_mm3": float(np.isin(a_hrt2, ids).sum()) * fine_vox_hrt2,
                                             "t1_fine_mm3": t1_mm3, "frac_in_hrt2_fov": in_fov / t1_mm3 if t1_mm3 else np.nan})
            for r in ["hipp", *SUBFIELDS]:
                ids = (1, 2, 3, 4, 5, HIPP) if r == "hipp" else SUBFIELDS[r]
                vol_rows.append({"subject": subject, "hemi": h, "arm": "manual", "region": r,
                                 "hrt2_mm3": float(np.isin(man_box, ids).sum()) * float(np.prod(zooms)),
                                 "t1_fine_mm3": float(np.isin(man_t1, ids).sum()) * T1_FINE_MM ** 3, "frac_in_hrt2_fov": 1.0})
            montage_inputs[h] = (man_box, {a.name: counts_plurality(block_counts(collapse(arm_fine[(a, "hrt2")], "primary"),
                                                                                SUBDIV)) for a in arms},
                                 np.asarray(hrt2_img.dataobj)[box], dom_box, slice_axis)

    for key, rows in (("scores", score_rows_all), ("assignment", assign_rows), ("purity", purity_all),
                      ("gate", gate_rows), ("volumes", vol_rows), ("registration", reg_rows)):
        pd.DataFrame(rows).to_csv(paths[key], sep="\t", index=False, float_format="%.6g")
    montage(montage_inputs, arms, paths["montage"], title=f"sub-{subject} ({nsd_subject(subject)})")
    sidecar = {
        "schema_version": SCHEMA_VERSION, "created": _dt.datetime.now().isoformat(timespec="seconds"), **provenance,
        "subject": subject, "nsd_subject": nsd_subject(subject),
        "arms": [{"name": a.name, "tool": a.tool, "tree": a.tree, "t2w_acq": a.t2w_acq, "expresses": list(a.expresses),
                  "version": hc.tree_version(Path(deriv) / a.tree, Path(deriv)) if a.tree else "NSD ppdata",
                  "inputs": {h: str(inputs[a.name][h]) for h in HEMIS}} for a in arms],
        "reference": {"labels": str(anat / "HRT2" / "MTL_rawlabels.nii.gz"),
                      "decoding": "ctab = ceil(k/2); odd = R, even = L", "crosscheck": "roi/{lh,rh}.MTL (0.8 mm)"},
        "xfm": {"file": str(paths["xfm"]), "method": "ANTs rigid MI, T2_0pt8 -> HRT2/T2matched, HRT2_mask on fixed"},
        "grids": {"hrt2": {"subdiv": list(SUBDIV), "rule": "block plurality, ties to the lower class"},
                  "t1_fine_mm": T1_FINE_MM, "pad_mm": PAD_MM, "bold": {"zooms_mm": list(BOLD_ZOOMS),
                  "offsets_vox": [list(o) for o in BOLD_OFFSETS]}},
        "canon": CANON, "native_to_canon": {k: {str(n): c for n, c in v.items()} for k, v in NATIVE_TO_CANON.items()},
        "subfields": {k: list(v) for k, v in SUBFIELDS.items()}, "variants": {
            "primary": "CA4 -> DG; voxels an arm calls SRLM dropped from subfield scoring on both sides",
            "alt": "CA4 -> CA3; SRLM -> nearest non-SRLM label of the same arm"},
    }
    paths["sidecar"].write_text(json.dumps(sidecar, indent=2) + "\n")
    log(f"sub-{subject}: wrote {paths['dir']}")
    return paths


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def montage(per_hemi: dict, arms: Sequence[NsdArm], out: Path, title: str, n_slices: int = 6) -> Path:
    """HRT2 slices through the subfield domain: manual + every arm (primary collapse), labels over HRT2."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    cmap = ListedColormap(["#000000"] + [CANON_COLOURS[k] for k in range(1, N_CLASSES)])
    rows = ["manual", *[a.name for a in arms]]
    cols = []
    for h in HEMIS:
        man, labs, bg, dom, ax_ = per_hemi[h]
        sl = np.flatnonzero(dom.any(axis=tuple(a for a in range(3) if a != ax_)))
        for k in np.linspace(sl[0], sl[-1], n_slices + 2)[1:-1].round().astype(int):
            cols.append((h, k))
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(1.7 * len(cols), 1.5 * len(rows) + 0.8), squeeze=False)
    for j, (h, k) in enumerate(cols):
        man, labs, bg, dom, ax_ = per_hemi[h]
        take = lambda v: np.take(v, k, axis=ax_)  # noqa: E731
        b = take(bg).astype(float)
        union = take(man > 0) | np.any([take(labs[n] > 0) for n in labs], axis=0)
        ii, jj = np.nonzero(union) if union.any() else (np.arange(b.shape[0]), np.arange(b.shape[1]))
        p = 12
        r0, r1 = max(ii.min() - p, 0), ii.max() + p + 1
        c0, c1 = max(jj.min() - p, 0), jj.max() + p + 1
        lo, hi = np.percentile(b[b > 0], [1, 99.5]) if (b > 0).any() else (0, 1)
        for i, name in enumerate(rows):
            lab = take(man) if name == "manual" else take(labs[name])
            ax = axes[i, j]
            ax.imshow(b[r0:r1, c0:c1].T, cmap="gray", vmin=lo, vmax=hi, origin="lower", interpolation="nearest",
                      aspect="equal")
            ax.imshow(np.ma.masked_equal(lab[r0:r1, c0:c1], 0).T, cmap=cmap, vmin=0, vmax=N_CLASSES - 1, alpha=0.55,
                      origin="lower", interpolation="nearest", aspect="equal")
            ax.set_xticks([]), ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(name, fontsize=9)
            if i == 0:
                ax.set_title(f"{h} slice {k}", fontsize=8)
    shown = [1, 2, 3, 4, 5, 6, 7, 9]
    fig.legend(handles=[Patch(color=CANON_COLOURS[k], label=CANON[k]) for k in shown], loc="lower center",
               ncol=len(shown), fontsize=8, frameon=False)
    fig.suptitle(f"{title}: labels on the HRT2 grid (plurality; CA4 -> DG), slices within the manual subfield domain",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Tree
# ---------------------------------------------------------------------------

def collect(out_root: Path) -> dict[str, pd.DataFrame]:
    """Concatenate every subject's tables."""
    out = {}
    for k in ("volumes", "scores", "assignment", "purity", "gate", "registration"):
        files = sorted(Path(out_root).glob(f"sub-*/*_desc-{k}_stats.tsv"))
        if not files:
            raise FileNotFoundError(f"no *_desc-{k}_stats.tsv under {out_root}")
        out[k] = pd.concat([pd.read_csv(f, sep="\t", dtype={"subject": str, "offset": str}) for f in files],
                           ignore_index=True)
    return out
