#!/usr/bin/env python3
"""
deface_anat.py — deface every whole-head anatomical volume in the BIDS tree,
in place, with the faced originals moved byte-identical to the sourcedata
sibling first.

Workbench: mmmdata-agents/docs/workbench/anat-defacing/ (DECIDED 2026-09-11).

WHAT GETS DEFACED, and where each volume's face mask comes from
----------------------------------------------------------------
One face mask per subject is computed by pydeface on the fMRIPrep
`desc-preproc_T1w` (T1w space) and CARRIED to every other grid, so the same
face voxels go everywhere. Legs, by target grid:

  T1w space      fMRIPrep desc-preproc_T1w, session-level desc-preproc_T2w
                 (same grid, asserted)                       <- pydeface mask
  MNI res-2      fMRIPrep space-MNI..._desc-preproc_T1w      <- ANTs, T1w->MNI h5
  fsnative       FreeSurfer orig/T1/nu/T2/T2.norm/T2.prenorm <- ANTs, T1w->fsnative ITK affine
  raw T1w run k  sub-XX/ses-YY/anat/*_T1w.nii.gz             <- ANTs, INVERSE of
                 FreeSurfer orig/00k.mgz (byte-equal to run k,   from-orig_to-T1w
                 asserted), rawavg.mgz (run-01 grid, asserted)
  raw T2w        same-session T2w on their own grids         <- FSL flirt 6-dof
                 FreeSurfer orig/T2raw.mgz (byte-equal to the   T2w->T1w run-01,
                 first T2w, asserted)                           mask carried back

ANTs lives in the fMRIPrep container (`singularity_dir` in config); flirt and
convert_xfm are FSL on PATH; pydeface is a Python dependency of this venv.

`--raw-only` is the CONVERTER GUARD for subjects that have no fMRIPrep yet:
raw T1w runs get pydeface directly, same-session T2w get the flirt carry, and
the brain-mask invariance check is deferred to `verify` once fMRIPrep exists.

INVARIANTS, checked before any byte is written and again after
--------------------------------------------------------------
1. face ∩ brain = ∅ in every grid (fMRIPrep brain mask, carried by the same
   transform as the face mask; FreeSurfer brainmask.mgz on the conformed grid).
   A single voxel of overlap aborts the subject.
2. Every transform leg is direction-checked: the T1w intensity image is
   resampled by the same call and must correlate >= DIRECTION_R_MIN with the
   target's intensities inside the brain. An inverted leg fails loudly rather
   than defacing the back of the head.
3. Bytes outside the face region are UNCHANGED: NIfTI files are rewritten by
   zeroing voxels in the on-disk byte buffer (header bytes untouched, dtype
   and scaling untouched); MGH files go through nibabel with the header carried
   and are checked field by field on reload.
4. The original is at its sourcedata mirror path with the recorded sha256
   before the in-place write happens; an existing mirror file with a
   DIFFERENT sha is an error, never overwritten.

Idempotent: a target whose current sha256 equals the provenance's `sha_after`
is skipped; equal to `sha_before` is (re)defaced; anything else is an error.

Provenance, masks and QC montages live OUTSIDE the BIDS tree, under
`<source_dir>/anat_defacing/sub-XX/` — the "before" montages show faces.

Usage:
  deface_anat.py run     --subject 03 --dry-run   # masks + checks + montages, no writes in BIDS
  deface_anat.py run     --subject 03             # the real thing
  deface_anat.py verify  --subject 03             # re-check invariants from provenance
  deface_anat.py restore --subject 03             # put the originals back from their mirrors
  deface_anat.py reports --subject 03 [--dry-run] # move anat-rendering reports to sourcedata
  deface_anat.py audit                            # every anat volume in the tree has provenance
  deface_anat.py run     --subject 08 --raw-only  # converter guard: raw anat only, no fMRIPrep yet
"""


import argparse
import datetime as _dt
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import nibabel as nib

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src" / "python") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src" / "python"))

FMRIPREP_CONTAINER = "fmriprep-25.2.5.sif"
DIRECTION_R_MIN = 0.80          # T1w resampled through a leg vs the target, inside brain
FLIRT_MAX_SHIFT_MM = 15.0       # rigid refinement beyond the header alignment that we trust
FLIRT_MAX_ROT_DEG = 15.0
FS_CONFORMED_WHOLE_HEAD = ["orig.mgz", "T1.mgz", "nu.mgz", "T2.mgz", "T2.norm.mgz", "T2.prenorm.mgz"]
FS_WHOLE_HEAD_FRACTION = 0.30   # nonzero voxels outside brainmask -> whole-head (audit)
PROVENANCE_NAME = "provenance.json"


# --------------------------------------------------------------------------- #
# configuration                                                               #
# --------------------------------------------------------------------------- #
@dataclass
class Roots:
    bids: Path
    source: Path
    deriv: Path
    containers: Path

    @property
    def fmriprep(self) -> Path:
        return self.deriv / "fmriprep"

    @property
    def freesurfer(self) -> Path:
        return self.fmriprep / "sourcedata" / "freesurfer"

    def defacing_dir(self, subject: str) -> Path:
        return self.source / "anat_defacing" / f"sub-{subject}"

    def provenance_path(self, subject: str) -> Path:
        return self.defacing_dir(subject) / PROVENANCE_NAME

    def mirror(self, path: Path) -> Path:
        """Sourcedata mirror path for a faced original. Raw anatomy mirrors
        beside the session's other raw inputs; when that session directory
        belongs to someone else and is not group-writable, the mirror falls
        back to `anat_defacing/sub-XX/anat_faced/ses-YY/` (the provenance
        records whichever path was used, and `restore`/`verify` read it)."""
        path = Path(path)
        if path.is_relative_to(self.deriv):
            return self.source / "derivatives_faced" / path.relative_to(self.deriv)
        if path.is_relative_to(self.bids):
            rel = path.relative_to(self.bids)          # sub-XX/ses-YY/anat/<file>
            parts = list(rel.parts)
            if "anat" in parts:
                parts[parts.index("anat")] = "anat_faced"
            preferred = self.source.joinpath(*parts)
            if _writable_ancestor(preferred.parent):
                return preferred
            subject, session = parts[0], parts[1]
            return self.defacing_dir(subject[4:]) / "anat_faced" / session / path.name
        raise ValueError(f"{path} is under neither the BIDS root nor derivatives")


def _writable_ancestor(directory: Path) -> bool:
    """True if `directory` exists and is writable, or can be created because
    its nearest existing ancestor is writable."""
    d = Path(directory)
    while not d.exists():
        if d.parent == d:
            return False
        d = d.parent
    return os.access(d, os.W_OK)


def load_roots() -> Roots:
    from core.config import load_config
    cfg = load_config(config_dir=_REPO_ROOT / "config")["paths"]
    return Roots(
        bids=Path(cfg["bids_project_dir"]),
        source=Path(cfg["source_dir"]),
        deriv=Path(cfg["output_dir"]),
        containers=Path(cfg["singularity_dir"]),
    )


# --------------------------------------------------------------------------- #
# small utilities                                                             #
# --------------------------------------------------------------------------- #
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def now() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def run(cmd: list[str], log=print) -> subprocess.CompletedProcess:
    log("  $ " + " ".join(str(c) for c in cmd))
    proc = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(map(str, cmd))}\n"
                           f"stdout:\n{proc.stdout[-2000:]}\nstderr:\n{proc.stderr[-2000:]}")
    return proc


def same_grid(a: nib.spatialimages.SpatialImage, b: nib.spatialimages.SpatialImage, atol=1e-3) -> bool:
    return tuple(a.shape[:3]) == tuple(b.shape[:3]) and np.allclose(a.affine, b.affine, atol=atol)


def voxels_equal(a: Path, b: Path) -> bool:
    ia, ib = nib.load(a), nib.load(b)
    if not same_grid(ia, ib):
        return False
    da = np.asanyarray(ia.dataobj).squeeze()
    db = np.asanyarray(ib.dataobj).squeeze()
    return da.shape == db.shape and np.array_equal(da, db)


def load_bool(path: Path) -> np.ndarray:
    return np.asanyarray(nib.load(path).dataobj).squeeze() > 0.5


def save_mask(mask: np.ndarray, like: nib.spatialimages.SpatialImage, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.Nifti1Image(mask.astype(np.uint8), like.affine).to_filename(path)
    return path


def as_nifti(path: Path, out: Path) -> Path:
    """A NIfTI copy of an MGH volume (ANTs' reference must be NIfTI)."""
    img = nib.load(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    nib.Nifti1Image(np.asanyarray(img.dataobj), img.affine).to_filename(out)
    return out


def masked_corr(x: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float:
    xv = x[mask].astype(np.float64)
    yv = y[mask].astype(np.float64)
    if xv.size < 1000 or xv.std() == 0 or yv.std() == 0:
        return float("nan")
    return float(np.corrcoef(xv, yv)[0, 1])


# --------------------------------------------------------------------------- #
# byte-exact voxel zeroing                                                    #
# --------------------------------------------------------------------------- #
def zero_voxels_nifti(src: Path, dst: Path, face: np.ndarray) -> None:
    """Rewrite `src` to `dst` with the voxels in `face` zeroed in the ON-DISK
    byte buffer. Header bytes, dtype, byte order, scaling and any trailing
    extension bytes are copied verbatim. `face` is a bool array in the image's
    first three dimensions."""
    img = nib.load(src)
    hdr = img.header
    offset = int(img.dataobj.offset)
    shape = tuple(int(s) for s in hdr.get_data_shape())
    dt = np.dtype(hdr.get_data_dtype()).newbyteorder(hdr.endianness)
    opener = gzip.open if str(src).endswith(".gz") else open
    with opener(src, "rb") as fh:
        raw = fh.read()
    nbytes = int(np.prod(shape)) * dt.itemsize
    head, body, tail = raw[:offset], raw[offset:offset + nbytes], raw[offset + nbytes:]
    if len(body) != nbytes:
        raise RuntimeError(f"{src}: expected {nbytes} data bytes at offset {offset}, found {len(body)}")
    arr = np.frombuffer(body, dtype=dt).reshape(shape, order="F").copy()
    if face.shape != shape[:3]:
        raise ValueError(f"{src}: face mask {face.shape} does not match volume {shape[:3]}")
    view = arr.reshape(shape[:3] + (-1,), order="F")
    view[face, :] = 0
    out = head + arr.tobytes(order="F") + tail
    tmp = dst.with_name(dst.name + ".tmp")
    if str(dst).endswith(".gz"):
        with gzip.open(tmp, "wb", compresslevel=6) as fh:
            fh.write(out)
    else:
        with open(tmp, "wb") as fh:
            fh.write(out)
    os.replace(tmp, dst)


def zero_voxels_mgh(src: Path, dst: Path, face: np.ndarray) -> None:
    img = nib.load(src)
    data = np.asanyarray(img.dataobj).copy()
    if face.shape != data.shape[:3]:
        raise ValueError(f"{src}: face mask {face.shape} does not match volume {data.shape[:3]}")
    data.reshape(data.shape[:3] + (-1,))[face, :] = 0
    tmp = dst.with_name(dst.name + ".tmp.mgz")
    nib.MGHImage(data, img.affine, img.header).to_filename(tmp)
    os.replace(tmp, dst)


def zero_voxels(src: Path, dst: Path, face: np.ndarray) -> None:
    if str(src).endswith((".mgz", ".mgh")):
        zero_voxels_mgh(src, dst, face)
    else:
        zero_voxels_nifti(src, dst, face)


def check_rewrite(original: Path, defaced: Path, face: np.ndarray, brain: np.ndarray | None) -> dict:
    """The post-write invariants: identical outside the face, zero inside it,
    identical inside the brain, header preserved."""
    a, b = nib.load(original), nib.load(defaced)
    da = np.asanyarray(a.dataobj).reshape(a.shape[:3] + (-1,))
    db = np.asanyarray(b.dataobj).reshape(b.shape[:3] + (-1,))
    if str(original).endswith((".mgz", ".mgh")):
        header_ok = all(np.array_equal(a.header[k], b.header[k]) for k in a.header.keys())
    else:
        header_ok = a.header.binaryblock == b.header.binaryblock
    out = {
        "outside_face_identical": bool(np.array_equal(da[~face], db[~face])),
        "face_zeroed": bool((db[face] == 0).all()),
        "header_preserved": bool(header_ok),
        "max_abs_diff_in_brain": None,
    }
    if brain is not None:
        out["max_abs_diff_in_brain"] = float(np.abs(da[brain].astype(np.float64) - db[brain].astype(np.float64)).max())
    return out


# --------------------------------------------------------------------------- #
# transform legs                                                              #
# --------------------------------------------------------------------------- #
class Legs:
    """The registration/resampling calls, each returning file paths."""

    def __init__(self, roots: Roots, work: Path, log=print):
        self.roots = roots
        self.work = work
        self.log = log
        work.mkdir(parents=True, exist_ok=True)

    # -- pydeface ------------------------------------------------------------
    def pydeface_mask(self, image: Path, tag: str) -> Path:
        """Run pydeface's template registration on `image`; return the FACE
        mask (1 = remove) on the image's grid. pydeface's own defaced output
        is discarded — the write happens in `zero_voxels`."""
        from pydeface.utils import deface_image
        scratch = self.work / f"pydeface_{tag}"
        scratch.mkdir(parents=True, exist_ok=True)
        outfile = scratch / "defaced.nii.gz"
        self.log(f"  pydeface {image.name} -> {scratch}")
        _, warped_mask, template_reg, template_reg_mat = deface_image(
            str(image), outfile=str(outfile), force=True, forcecleanup=False, verbose=False)
        keep = np.asanyarray(nib.load(warped_mask).dataobj).squeeze() > 0.5   # pydeface: 1 = keep
        out = save_mask(~keep, nib.load(image), scratch / "facemask.nii.gz")
        for f in (warped_mask, template_reg, template_reg_mat):
            try:
                os.remove(f)
            except OSError:
                pass
        return out

    # -- ANTs (fMRIPrep container) -------------------------------------------
    def ants(self, moving: Path, reference: Path, transforms: list[str], out: Path, nearest: bool) -> Path:
        sif = self.roots.containers / FMRIPREP_CONTAINER
        if not sif.exists():
            raise FileNotFoundError(f"fMRIPrep container not found: {sif} (config paths.singularity_dir)")
        binds = sorted({str(p) for p in (self.roots.bids, self.roots.source, self.work) if p.exists()})
        cmd = ["singularity", "exec", "--cleanenv"]
        for b in binds:
            cmd += ["-B", f"{b}:{b}"]
        cmd += [sif, "antsApplyTransforms", "-d", "3", "-i", moving, "-r", reference, "-o", out,
                "-n", "NearestNeighbor" if nearest else "Linear"]
        for t in transforms:
            cmd += ["-t", t]
        out.parent.mkdir(parents=True, exist_ok=True)
        run(cmd, self.log)
        return out

    def carry_ants(self, face_t1w: Path, brain_t1w: Path, t1w_image: Path, reference: Path,
                   transforms: list[str], tag: str) -> tuple[Path, Path, Path]:
        """Carry face + brain masks (nearest) and the T1w intensities (linear)
        onto `reference`'s grid. Returns (face, brain, t1w_resampled)."""
        d = self.work / f"leg_{tag}"
        face = self.ants(face_t1w, reference, transforms, d / "facemask.nii.gz", nearest=True)
        brain = self.ants(brain_t1w, reference, transforms, d / "brainmask.nii.gz", nearest=True)
        t1 = self.ants(t1w_image, reference, transforms, d / "t1w_resampled.nii.gz", nearest=False)
        return face, brain, t1

    # -- FSL flirt (same-session T2w) ----------------------------------------
    def carry_flirt(self, face_src: Path, brain_src: Path, src_image: Path, target: Path,
                    tag: str) -> tuple[Path, Path, Path, dict]:
        """Rigid-register `target` (a T2w) to `src_image` (the same session's
        T1w run-01), initialised from the headers, then carry the masks back
        onto the target grid. Returns (face, brain, t1w_resampled, info)."""
        d = self.work / f"leg_{tag}"
        d.mkdir(parents=True, exist_ok=True)
        t2w2t1w = d / "t2w2t1w.mat"
        t1w2t2w = d / "t1w2t2w.mat"
        run(["flirt", "-in", target, "-ref", src_image, "-dof", "6", "-cost", "mutualinfo",
             "-usesqform", "-searchrx", "-20", "20", "-searchry", "-20", "20", "-searchrz", "-20", "20",
             "-omat", t2w2t1w, "-out", d / "t2w_in_t1w.nii.gz"], self.log)
        run(["convert_xfm", "-omat", t1w2t2w, "-inverse", t2w2t1w], self.log)
        shift_mm, rot_deg = fsl_mat_refinement(t2w2t1w, nib.load(target), nib.load(src_image))
        info = {"refinement_shift_mm": shift_mm, "refinement_rot_deg": rot_deg}
        if shift_mm > FLIRT_MAX_SHIFT_MM or rot_deg > FLIRT_MAX_ROT_DEG:
            raise RuntimeError(f"flirt refinement for {target.name} moved {shift_mm:.1f} mm / {rot_deg:.1f} deg "
                               f"from the header alignment (limits {FLIRT_MAX_SHIFT_MM} / {FLIRT_MAX_ROT_DEG}); "
                               f"registration is not trusted")
        face = d / "facemask.nii.gz"
        brain = d / "brainmask.nii.gz"
        t1 = d / "t1w_resampled.nii.gz"
        for src, out, interp in ((face_src, face, "nearestneighbour"), (brain_src, brain, "nearestneighbour"),
                                 (src_image, t1, "trilinear")):
            run(["flirt", "-in", src, "-ref", target, "-applyxfm", "-init", t1w2t2w,
                 "-interp", interp, "-out", out], self.log)
        return face, brain, t1, info


def _fsl_from_vox(img: nib.spatialimages.SpatialImage) -> np.ndarray:
    """FSL's internal scaled-mm coordinates from voxel indices (x flipped
    when the affine is neurological), the convention behind .mat files."""
    zooms = np.asarray(img.header.get_zooms()[:3], dtype=float)
    scale = np.diag(list(zooms) + [1.0])
    if np.linalg.det(img.affine[:3, :3]) > 0:
        flip = np.eye(4)
        flip[0, 0] = -1
        flip[0, 3] = img.shape[0] - 1
        return scale @ flip
    return scale


def fsl_mat_to_world(mat: np.ndarray, src: nib.spatialimages.SpatialImage,
                     ref: nib.spatialimages.SpatialImage) -> np.ndarray:
    """World(src) -> world(ref) affine implied by an FSL .mat."""
    return ref.affine @ np.linalg.inv(_fsl_from_vox(ref)) @ mat @ _fsl_from_vox(src) @ np.linalg.inv(src.affine)


def fsl_mat_refinement(mat_path: Path, src: nib.spatialimages.SpatialImage,
                       ref: nib.spatialimages.SpatialImage) -> tuple[float, float]:
    """How far the registration moved away from pure header alignment, as a
    translation at the source volume's centre (mm) and a rotation (deg)."""
    world = fsl_mat_to_world(np.loadtxt(mat_path), src, ref)
    centre_vox = np.array(list((np.asarray(src.shape[:3]) - 1) / 2.0) + [1.0])
    centre = src.affine @ centre_vox
    shift = float(np.linalg.norm((world @ centre)[:3] - centre[:3]))
    rot = world[:3, :3]
    rot = rot / np.cbrt(abs(np.linalg.det(rot)))
    angle = float(np.degrees(np.arccos(np.clip((np.trace(rot) - 1) / 2.0, -1.0, 1.0))))
    return shift, angle


# --------------------------------------------------------------------------- #
# target enumeration                                                          #
# --------------------------------------------------------------------------- #
@dataclass
class Target:
    path: Path
    kind: str            # raw_t1w | raw_t2w | fp_t1w | fp_mni_t1w | fp_t2w | fs_conformed | fs_own
    leg: str             # which mask/brain pair applies
    note: str = ""


@dataclass
class SubjectLayout:
    subject: str
    fp_anat: Path | None            # fMRIPrep subject- or session-level anat dir with the T1w
    fp_t1w: Path | None
    fp_mni_t1w: Path | None
    fp_brain: Path | None
    fp_mni_brain: Path | None
    xfm_t1w_to_mni: Path | None
    xfm_t1w_to_fsnative: Path | None
    fs_mri: Path | None
    raw_t1w: list[Path] = field(default_factory=list)
    raw_t2w: list[Path] = field(default_factory=list)
    fp_t2w: list[Path] = field(default_factory=list)
    orig_to_t1w: dict[str, Path] = field(default_factory=dict)   # raw T1w name -> ITK affine
    raw_direct: set = field(default_factory=set)                  # raw T1w keys with no transform yet
    targets: list[Target] = field(default_factory=list)


def _one(cands: list[Path], what: str, allow_none=False) -> Path | None:
    if len(cands) == 1:
        return cands[0]
    if not cands and allow_none:
        return None
    raise FileNotFoundError(f"expected exactly one {what}, found {len(cands)}: {[str(c) for c in cands]}")


def raw_t1w_key(path: Path) -> str:
    """The session/run part of a raw T1w name, which is also how fMRIPrep
    names its from-orig transform for that run."""
    return path.name.replace("_T1w.nii.gz", "")


def discover(roots: Roots, subject: str, raw_only: bool = False) -> SubjectLayout:
    sub = f"sub-{subject}"
    bids_sub = roots.bids / sub
    if not bids_sub.is_dir():
        raise FileNotFoundError(f"no BIDS subject directory at {bids_sub}")
    raw_t1w = sorted(bids_sub.glob("ses-*/anat/*_T1w.nii.gz"))
    raw_t2w = sorted(bids_sub.glob("ses-*/anat/*_T2w.nii.gz"))
    if not raw_t1w:
        raise FileNotFoundError(f"no raw T1w under {bids_sub}/ses-*/anat/")
    lay = SubjectLayout(subject, None, None, None, None, None, None, None, None, raw_t1w, raw_t2w)
    for t in raw_t1w:
        lay.targets.append(Target(t, "raw_t1w", f"raw:{raw_t1w_key(t)}"))
    for t in raw_t2w:
        lay.targets.append(Target(t, "raw_t2w", f"sess:{t.name}"))
    if raw_only:
        return lay

    fp_sub = roots.fmriprep / sub
    if not fp_sub.is_dir():
        raise FileNotFoundError(f"no fMRIPrep output at {fp_sub}; use --raw-only for a subject without preprocessing")
    t1w_cands = sorted(fp_sub.glob("anat/*_desc-preproc_T1w.nii.gz")) + \
        sorted(fp_sub.glob("ses-*/anat/*_desc-preproc_T1w.nii.gz"))
    t1w_cands = [c for c in t1w_cands if "space-" not in c.name]
    lay.fp_t1w = _one(t1w_cands, "fMRIPrep desc-preproc_T1w (no space entity)")
    lay.fp_anat = lay.fp_t1w.parent
    stem = lay.fp_t1w.name.replace("_desc-preproc_T1w.nii.gz", "")
    lay.fp_brain = _one(sorted(lay.fp_anat.glob(f"{stem}_desc-brain_mask.nii.gz")), "T1w brain mask")
    lay.fp_mni_t1w = _one(sorted(lay.fp_anat.glob(f"{stem}_space-MNI152NLin2009cAsym_res-2_desc-preproc_T1w.nii.gz")),
                          "MNI res-2 preproc T1w")
    lay.fp_mni_brain = _one(sorted(lay.fp_anat.glob(f"{stem}_space-MNI152NLin2009cAsym_res-2_desc-brain_mask.nii.gz")),
                            "MNI res-2 brain mask")
    lay.xfm_t1w_to_mni = _one(sorted(lay.fp_anat.glob(f"{stem}_from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5")),
                              "T1w->MNI transform")
    lay.xfm_t1w_to_fsnative = _one(sorted(lay.fp_anat.glob(f"{stem}_from-T1w_to-fsnative_mode-image_xfm.txt")),
                                   "T1w->fsnative transform")
    lay.fp_t2w = sorted(fp_sub.glob("ses-*/anat/*_desc-preproc_T2w.nii.gz"))
    for t in raw_t1w:
        key = raw_t1w_key(t)
        xfm = sorted(fp_sub.glob(f"ses-*/anat/{key}_from-orig_to-T1w_mode-image_xfm.txt"))
        if xfm:
            lay.orig_to_t1w[key] = _one(xfm, f"from-orig_to-T1w transform for {key}")
        else:
            # a session converted after fMRIPrep ran: no transform yet, so this
            # run gets its own pydeface registration (the converter-guard path)
            lay.raw_direct.add(key)

    lay.targets.append(Target(lay.fp_t1w, "fp_t1w", "T1w"))
    lay.targets.append(Target(lay.fp_mni_t1w, "fp_mni_t1w", "MNI"))
    for t in lay.fp_t2w:
        lay.targets.append(Target(t, "fp_t2w", "T1w", note="session T2w, asserted on the T1w grid"))

    fs_mri = roots.freesurfer / sub / "mri"
    if not fs_mri.is_dir():
        raise FileNotFoundError(f"no FreeSurfer subject at {fs_mri}")
    lay.fs_mri = fs_mri
    for name in FS_CONFORMED_WHOLE_HEAD:
        p = fs_mri / name
        if p.exists():
            lay.targets.append(Target(p, "fs_conformed", "fsnative"))
    # FreeSurfer inputs: orig/00?.mgz are byte-equal copies of the raw T1w runs
    # (asserted at run time), rawavg.mgz lives on run-01's grid, orig/T2raw.mgz
    # is a byte-equal copy of the T2w recon-all was given.
    inputs = sorted((fs_mri / "orig").glob("[0-9][0-9][0-9].mgz"))
    seen_by_fmriprep = [r for r in raw_t1w if raw_t1w_key(r) not in lay.raw_direct]
    if len(inputs) != len(seen_by_fmriprep):
        raise RuntimeError(f"{fs_mri}/orig has {len(inputs)} T1w inputs but fMRIPrep saw {len(seen_by_fmriprep)} raw T1w")
    matched: dict[Path, Path] = {}
    for inp in inputs:
        hits = [r for r in raw_t1w if voxels_equal(inp, r)]
        if len(hits) != 1:
            raise RuntimeError(f"{inp} is byte-equal to {len(hits)} raw T1w runs (expected exactly 1)")
        matched[inp] = hits[0]
        lay.targets.append(Target(inp, "fs_own", f"raw:{raw_t1w_key(hits[0])}", note=f"byte-equal to {hits[0].name}"))
    rawavg = fs_mri / "rawavg.mgz"
    if rawavg.exists():
        first = matched[inputs[0]]
        if not same_grid(nib.load(rawavg), nib.load(first)):
            raise RuntimeError(f"{rawavg} is not on the grid of {first.name}")
        lay.targets.append(Target(rawavg, "fs_own", f"raw:{raw_t1w_key(first)}", note=f"on the grid of {first.name}"))
    t2raw = fs_mri / "orig" / "T2raw.mgz"
    if t2raw.exists():
        hits = [r for r in raw_t2w if voxels_equal(t2raw, r)]
        if len(hits) != 1:
            raise RuntimeError(f"{t2raw} is byte-equal to {len(hits)} raw T2w (expected exactly 1)")
        lay.targets.append(Target(t2raw, "fs_own", f"sess:{hits[0].name}", note=f"byte-equal to {hits[0].name}"))
    audit_freesurfer_mri(fs_mri)
    return lay


def audit_freesurfer_mri(fs_mri: Path) -> None:
    """Every 3-D volume on the conformed grid that is mostly non-brain must be
    in FS_CONFORMED_WHOLE_HEAD; an unknown whole-head volume is an error."""
    bm = nib.load(fs_mri / "brainmask.mgz")
    brain = np.asanyarray(bm.dataobj).squeeze() > 0
    unknown = []
    for p in sorted(fs_mri.glob("*.mgz")):
        if p.name in FS_CONFORMED_WHOLE_HEAD:
            continue
        img = nib.load(p)
        if len(img.shape) > 3 and img.shape[3] > 1:
            continue
        if tuple(img.shape[:3]) != tuple(bm.shape[:3]):
            continue
        d = np.asanyarray(img.dataobj).squeeze()
        nz = d != 0
        if nz.sum() and (nz & ~brain).sum() / nz.sum() > FS_WHOLE_HEAD_FRACTION:
            unknown.append(p.name)
    if unknown:
        raise RuntimeError(f"unclassified whole-head volumes in {fs_mri}: {unknown} — add to FS_CONFORMED_WHOLE_HEAD "
                           f"or explain why they carry no face")


# --------------------------------------------------------------------------- #
# mask computation                                                            #
# --------------------------------------------------------------------------- #
@dataclass
class LegMasks:
    face: Path
    brain: Path | None
    direction_r: float | None = None
    info: dict = field(default_factory=dict)


def compute_masks(lay: SubjectLayout, roots: Roots, legs: Legs, raw_only: bool, log=print) -> dict[str, LegMasks]:
    masks: dict[str, LegMasks] = {}
    session_t1w: dict[str, Path] = {}          # ses-YY -> raw T1w run used as the T2w anchor
    for t in lay.raw_t1w:
        ses = t.name.split("_")[1]
        session_t1w.setdefault(ses, t)          # sorted: run-01 first

    if raw_only:
        for t in lay.raw_t1w:
            key = f"raw:{raw_t1w_key(t)}"
            log(f"[{key}] pydeface directly (no fMRIPrep yet; brain invariance deferred to verify)")
            masks[key] = LegMasks(legs.pydeface_mask(t, key.replace(":", "_")), None)
    else:
        log("[T1w] pydeface on the fMRIPrep T1w")
        face_t1w = legs.pydeface_mask(lay.fp_t1w, "T1w")
        masks["T1w"] = LegMasks(face_t1w, lay.fp_brain, 1.0)

        log("[MNI] carry via T1w->MNI h5")
        f, b, t1 = legs.carry_ants(face_t1w, lay.fp_brain, lay.fp_t1w, lay.fp_mni_t1w,
                                   [str(lay.xfm_t1w_to_mni)], "MNI")
        r = masked_corr(np.asanyarray(nib.load(t1).dataobj).squeeze(),
                        np.asanyarray(nib.load(lay.fp_mni_t1w).dataobj).squeeze(), load_bool(lay.fp_mni_brain))
        masks["MNI"] = LegMasks(f, lay.fp_mni_brain, r, {"carried_brain_vs_fmriprep_brain_dice": dice(load_bool(b), load_bool(lay.fp_mni_brain))})

        log("[fsnative] carry via T1w->fsnative ITK affine")
        orig = lay.fs_mri / "orig.mgz"
        ref = as_nifti(orig, legs.work / "leg_fsnative" / "orig_ref.nii.gz")
        f, b, t1 = legs.carry_ants(face_t1w, lay.fp_brain, lay.fp_t1w, ref, [str(lay.xfm_t1w_to_fsnative)], "fsnative")
        fs_brain = save_mask(np.asanyarray(nib.load(lay.fs_mri / "brainmask.mgz").dataobj).squeeze() > 0,
                             nib.load(ref), legs.work / "leg_fsnative" / "fs_brainmask.nii.gz")
        r = masked_corr(np.asanyarray(nib.load(t1).dataobj).squeeze(),
                        np.asanyarray(nib.load(orig).dataobj).squeeze(), load_bool(fs_brain))
        masks["fsnative"] = LegMasks(f, fs_brain, r, {"carried_brain_vs_freesurfer_brainmask_dice": dice(load_bool(b), load_bool(fs_brain))})

        for t in lay.raw_t1w:
            key = f"raw:{raw_t1w_key(t)}"
            if raw_t1w_key(t) in lay.raw_direct:
                log(f"[{key}] no fMRIPrep transform for this run: pydeface directly, brain invariance DEFERRED")
                masks[key] = LegMasks(legs.pydeface_mask(t, key.replace(":", "_")), None)
                continue
            log(f"[{key}] carry via inverse from-orig_to-T1w")
            f, b, t1 = legs.carry_ants(face_t1w, lay.fp_brain, lay.fp_t1w, t,
                                       [f"[{lay.orig_to_t1w[raw_t1w_key(t)]},1]"], key.replace(":", "_"))
            r = masked_corr(np.asanyarray(nib.load(t1).dataobj).squeeze(),
                            np.asanyarray(nib.load(t).dataobj).squeeze(), load_bool(b))
            masks[key] = LegMasks(f, b, r)

    for t in lay.raw_t2w:
        key = f"sess:{t.name}"
        ses = t.name.split("_")[1]
        anchor = session_t1w.get(ses)
        if anchor is None:
            raise RuntimeError(f"{t.name}: no T1w in the same session to anchor the T2w registration")
        src = masks[f"raw:{raw_t1w_key(anchor)}"]
        log(f"[{key}] flirt to {anchor.name}, masks carried back")
        brain_src = src.brain
        if brain_src is None:      # raw-only: no brain mask to carry; carry the face mask twice
            brain_src = src.face
        f, b, t1, info = legs.carry_flirt(src.face, brain_src, anchor, t, key.replace(":", "_").replace(".nii.gz", ""))
        r = None
        if src.brain is not None:
            r = masked_corr(np.asanyarray(nib.load(t1).dataobj).squeeze(),
                            np.asanyarray(nib.load(t).dataobj).squeeze(), load_bool(b))
            info["note"] = "direction r is T1w-vs-T2w, cross-modal: expect a NEGATIVE or weak value; the gate is the refinement bound"
        masks[key] = LegMasks(f, b if src.brain is not None else None, r, info)
    return masks


def dice(a: np.ndarray, b: np.ndarray) -> float:
    s = a.sum() + b.sum()
    return float(2 * (a & b).sum() / s) if s else float("nan")


# --------------------------------------------------------------------------- #
# QC montage                                                                  #
# --------------------------------------------------------------------------- #
def montage_stem(path: Path) -> str:
    """`T2.norm.mgz` and `T2.mgz` must not share a montage name; the FreeSurfer
    `orig/` inputs are prefixed so they do not collide with `orig.mgz`."""
    stem = path.name.replace(".nii.gz", "").replace(".mgz", "").replace(".", "_")
    if path.parent.name == "orig":
        stem = "orig-" + stem
    return stem


def montage(image: Path, face: np.ndarray, out: Path, title: str) -> Path:
    """Before/after panels: three sagittal cuts (midline and +-25 mm) and one
    axial cut through the lower third, in RAS with true voxel aspect so a
    thin oblique slab reads correctly."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from nibabel.orientations import io_orientation, apply_orientation
    img = nib.load(image)
    data = np.asanyarray(img.dataobj).reshape(img.shape[:3] + (-1,))[..., 0].astype(np.float32)
    ornt = io_orientation(img.affine)
    data = apply_orientation(data, ornt)
    face_ras = apply_orientation(face.astype(np.uint8), ornt) > 0
    zoom = np.abs(np.asarray(nib.as_closest_canonical(img).header.get_zooms()[:3], dtype=float))
    after = data.copy()
    after[face_ras] = 0
    vmax = float(np.percentile(data[data > 0], 99.5)) if (data > 0).any() else 1.0
    n = data.shape[0]
    step = int(round(25.0 / zoom[0]))
    cuts = [max(0, n // 2 - step), n // 2, min(n - 1, n // 2 + step)]
    fig, ax = plt.subplots(2, 4, figsize=(16, 8))
    for col, c in enumerate(cuts):
        for row, vol in enumerate((data, after)):
            ax[row, col].imshow(np.rot90(vol[c, :, :]), cmap="gray", vmin=0, vmax=vmax, aspect=zoom[2] / zoom[1])
            ax[row, col].set_title(f"{'before' if row == 0 else 'after'} sag {c}")
    zc = data.shape[2] // 3
    for row, vol in enumerate((data, after)):
        ax[row, 3].imshow(np.rot90(vol[:, :, zc]), cmap="gray", vmin=0, vmax=vmax, aspect=zoom[1] / zoom[0])
        ax[row, 3].set_title(f"{'before' if row == 0 else 'after'} axial {zc}")
    for a in ax.flat:
        a.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=60)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# provenance                                                                  #
# --------------------------------------------------------------------------- #
def load_provenance(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {"subject": None, "created": now(), "tool": tool_versions(), "files": {}, "legs": {}, "reports": {}}


def tool_versions() -> str:
    try:
        from importlib.metadata import version
        pyd = version("pydeface")
    except Exception:          # pragma: no cover - reports/audit need no pydeface
        pyd = "unknown"
    return f"pydeface {pyd}; antsApplyTransforms from {FMRIPREP_CONTAINER}; FSL flirt from PATH"


def save_provenance(path: Path, prov: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    prov["updated"] = now()
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(prov, indent=2, sort_keys=True))
    os.replace(tmp, path)


# --------------------------------------------------------------------------- #
# the run                                                                     #
# --------------------------------------------------------------------------- #
def backup(original: Path, mirror: Path, sha: str, dry_run: bool, log=print) -> None:
    if mirror.exists():
        if sha256(mirror) == sha:
            log(f"  mirror present, identical: {mirror}")
            return
        raise RuntimeError(f"REFUSING to overwrite an existing mirror with a different sha256: {mirror}")
    if dry_run:
        log(f"  would copy -> {mirror}")
        return
    mirror.parent.mkdir(parents=True, exist_ok=True)
    tmp = mirror.with_name(mirror.name + ".tmp")
    shutil.copy2(original, tmp)
    if sha256(tmp) != sha:
        tmp.unlink()
        raise RuntimeError(f"copy to {mirror} did not reproduce the sha256")
    os.replace(tmp, mirror)
    log(f"  copied -> {mirror}")


def cmd_run(args) -> int:
    roots = load_roots()
    subject = args.subject
    log = print
    ddir = roots.defacing_dir(subject)
    work = Path(args.work) if args.work else ddir / "work"
    prov_path = roots.provenance_path(subject)
    prov = load_provenance(prov_path)
    prov["subject"] = f"sub-{subject}"
    prov["raw_only"] = bool(args.raw_only)

    log(f"== sub-{subject}: discovering targets")
    lay = discover(roots, subject, raw_only=args.raw_only)
    for t in lay.targets:
        log(f"  {t.kind:12s} {t.leg:40s} {t.path}")
    log(f"  {len(lay.targets)} targets")

    log(f"== computing masks (work dir {work})")
    legs = Legs(roots, work, log)
    masks = compute_masks(lay, roots, legs, args.raw_only, log)

    # keep the masks beside the provenance, record the leg diagnostics
    mask_dir = ddir / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    for key, m in masks.items():
        safe = key.replace(":", "_").replace(".nii.gz", "")
        kept = mask_dir / f"{safe}_facemask.nii.gz"
        shutil.copy2(m.face, kept)
        entry = {"facemask": str(kept), "brainmask": str(m.brain) if m.brain else None,
                 "direction_r": m.direction_r, **m.info}
        if m.brain is not None:
            kept_b = mask_dir / f"{safe}_brainmask.nii.gz"
            shutil.copy2(m.brain, kept_b)
            entry["brainmask"] = str(kept_b)
        prov["legs"][key] = entry
        if m.direction_r is not None and not key.startswith("sess:") and not (m.direction_r >= DIRECTION_R_MIN):
            raise RuntimeError(f"leg {key}: direction check r={m.direction_r:.3f} < {DIRECTION_R_MIN}; "
                               f"the transform is inverted or wrong — nothing written")
        log(f"  leg {key}: direction r={m.direction_r} {m.info}")

    log("== pre-write invariants: face ∩ brain = ∅ in every grid")
    face_arr: dict[str, np.ndarray] = {}
    brain_arr: dict[str, np.ndarray | None] = {}
    for key, m in masks.items():
        face_arr[key] = load_bool(m.face)
        brain_arr[key] = load_bool(m.brain) if m.brain else None
        if brain_arr[key] is not None:
            overlap = int((face_arr[key] & brain_arr[key]).sum())
            frac = float(face_arr[key].mean())
            log(f"  {key}: face voxels {frac:.1%} of grid, overlap with brain = {overlap}")
            if overlap:
                raise RuntimeError(f"leg {key}: face mask intersects the brain mask in {overlap} voxels — nothing written")
        else:
            log(f"  {key}: face voxels {face_arr[key].mean():.1%} of grid, brain invariance DEFERRED (no brain mask)")

    log("== targets")
    qc_dir = ddir / "qc"
    n_done = n_skip = 0
    for t in lay.targets:
        face = face_arr[t.leg]
        brain = brain_arr[t.leg]
        img = nib.load(t.path)
        if tuple(img.shape[:3]) != face.shape:
            raise RuntimeError(f"{t.path}: grid {img.shape[:3]} vs mask {face.shape} for leg {t.leg}")
        sha_now = sha256(t.path)
        rec = prov["files"].get(str(t.path))
        if rec and rec.get("sha_after") == sha_now:
            log(f"  already defaced: {t.path}")
            n_skip += 1
            continue
        if rec and rec.get("sha_before") not in (None, sha_now):
            raise RuntimeError(f"{t.path}: sha256 matches neither the recorded original nor the defaced file")
        mirror = roots.mirror(t.path)
        log(f"- {t.path}")
        backup(t.path, mirror, sha_now, args.dry_run, log)
        png = montage(t.path, face, qc_dir / (montage_stem(t.path) + "_deface.png"), f"{t.path.name} [{t.leg}]")
        if args.dry_run:
            log(f"  DRY RUN: would zero {int(face.sum())} voxels; montage {png}")
            continue
        tmp_out = t.path.with_name(t.path.name + ".defaced_tmp" + "".join(t.path.suffixes))
        zero_voxels(t.path, tmp_out, face)
        checks = check_rewrite(t.path, tmp_out, face, brain)
        bad = [k for k, v in checks.items() if v is False or (k == "max_abs_diff_in_brain" and v not in (None, 0.0))]
        if bad:
            tmp_out.unlink()
            raise RuntimeError(f"{t.path}: post-write check failed {bad}: {checks}")
        os.replace(tmp_out, t.path)
        sha_after = sha256(t.path)
        prov["files"][str(t.path)] = {
            "kind": t.kind, "leg": t.leg, "note": t.note, "mirror": str(mirror),
            "sha_before": sha_now, "sha_after": sha_after, "voxels_zeroed": int(face.sum()),
            "fraction_zeroed": float(face.mean()), "checks": checks, "montage": str(png), "defaced_at": now(),
        }
        save_provenance(prov_path, prov)
        log(f"  defaced ({checks})")
        n_done += 1
    if not args.dry_run:
        save_provenance(prov_path, prov)
    log(f"== done: {n_done} defaced, {n_skip} already done, {len(lay.targets)} targets"
        + (" (DRY RUN — nothing written in the BIDS tree)" if args.dry_run else ""))
    return 0


# --------------------------------------------------------------------------- #
# verify                                                                      #
# --------------------------------------------------------------------------- #
def cmd_verify(args) -> int:
    roots = load_roots()
    prov_path = roots.provenance_path(args.subject)
    if not prov_path.exists():
        print(f"ERROR: no provenance at {prov_path} — run `deface_anat.py run --subject {args.subject}` first")
        return 2
    prov = json.loads(prov_path.read_text())
    failures = 0
    for path, rec in sorted(prov["files"].items()):
        p = Path(path)
        problems = []
        if not p.exists():
            problems.append("missing")
        elif sha256(p) != rec["sha_after"]:
            problems.append("sha256 differs from the recorded defaced file")
        m = Path(rec["mirror"])
        if not m.exists():
            problems.append(f"mirror missing: {m}")
        elif sha256(m) != rec["sha_before"]:
            problems.append("mirror sha256 differs from the recorded original")
        if not problems and args.deep:
            leg = prov["legs"][rec["leg"]]
            face = load_bool(Path(leg["facemask"]))
            brain = load_bool(Path(leg["brainmask"])) if leg.get("brainmask") else None
            checks = check_rewrite(m, p, face, brain)
            if not (checks["outside_face_identical"] and checks["face_zeroed"] and checks["header_preserved"]):
                problems.append(f"rewrite invariants: {checks}")
            if checks["max_abs_diff_in_brain"] not in (None, 0.0):
                problems.append(f"brain changed: max diff {checks['max_abs_diff_in_brain']}")
        status = "OK " if not problems else "BAD"
        print(f"{status} {p}" + ("" if not problems else "  <- " + "; ".join(problems)))
        failures += bool(problems)
    print(f"{len(prov['files'])} files, {failures} problems")
    return 1 if failures else 0


# --------------------------------------------------------------------------- #
# restore                                                                     #
# --------------------------------------------------------------------------- #
def restore_files(prov: dict, log=print) -> int:
    """Put every defaced file back from its mirror (the mirror must still hash
    to `sha_before`; the tree file must hash to `sha_after`). Entries are
    removed from the provenance as they are restored; legs and reports stay."""
    restored = 0
    for path, rec in sorted(prov["files"].items()):
        p, m = Path(path), Path(rec["mirror"])
        if not m.exists() or sha256(m) != rec["sha_before"]:
            raise RuntimeError(f"{m}: mirror missing or does not hash to the recorded original — not restoring {p}")
        if p.exists() and sha256(p) not in (rec["sha_after"], rec["sha_before"]):
            raise RuntimeError(f"{p}: hashes to neither the defaced nor the original file — not restoring")
        tmp = p.with_name(p.name + ".restore_tmp")
        shutil.copy2(m, tmp)
        os.replace(tmp, p)
        if sha256(p) != rec["sha_before"]:
            raise RuntimeError(f"{p}: restore did not reproduce the original sha256")
        del prov["files"][path]
        restored += 1
        log(f"  restored {p}")
    return restored


def cmd_restore(args) -> int:
    roots = load_roots()
    prov_path = roots.provenance_path(args.subject)
    if not prov_path.exists():
        print(f"ERROR: no provenance at {prov_path}")
        return 2
    prov = json.loads(prov_path.read_text())
    n = restore_files(prov)
    save_provenance(prov_path, prov)
    print(f"{n} files restored from their mirrors; mirrors kept")
    return 0


# --------------------------------------------------------------------------- #
# reports                                                                     #
# --------------------------------------------------------------------------- #
def report_files(roots: Roots, subject: str) -> list[Path]:
    sub = f"sub-{subject}"
    out: list[Path] = []
    out += sorted((roots.deriv / "mriqc").glob(f"{sub}_*_T1w.html"))
    out += sorted((roots.deriv / "mriqc").glob(f"{sub}_*_T2w.html"))
    figs = roots.fmriprep / sub / "figures"
    if figs.is_dir():
        out += sorted(p for p in figs.iterdir()
                      if "task-" not in p.name and (p.name.endswith("_T1w.svg") or p.name.endswith("_dseg.svg")))
    # volume viewer bundles embed a T1w or MNI underlay; surface bundles do not
    for qc in sorted(roots.deriv.glob(f"*/{sub}/qc/*_desc-viewer_*.html")):
        if "space-fsnative" not in qc.name:
            out.append(qc)
    return out


def cmd_reports(args) -> int:
    roots = load_roots()
    prov_path = roots.provenance_path(args.subject)
    prov = load_provenance(prov_path)
    prov["subject"] = f"sub-{args.subject}"
    files = report_files(roots, args.subject)
    moved = 0
    for f in files:
        mirror = roots.mirror(f)
        sha = sha256(f)
        print(f"- {f}\n  -> {mirror}")
        if args.dry_run:
            continue
        backup(f, mirror, sha, False)
        f.unlink()
        prov["reports"][str(f)] = {"mirror": str(mirror), "sha256": sha, "moved_at": now()}
        moved += 1
    if not args.dry_run:
        save_provenance(prov_path, prov)
    print(f"{len(files)} report files, {moved} moved" + (" (DRY RUN)" if args.dry_run else ""))
    if any("_desc-viewer_" in f.name for f in files):
        print("Rebuild the moved viewer bundles from the defaced volumes with build_brain_viewer.py "
              "(the moved originals embed the faced underlay).")
    return 0


# --------------------------------------------------------------------------- #
# audit                                                                       #
# --------------------------------------------------------------------------- #
def cmd_audit(args) -> int:
    roots = load_roots()
    subjects = sorted(p.name[4:] for p in roots.bids.glob("sub-*") if p.is_dir())
    bad = 0
    for s in subjects:
        prov_path = roots.provenance_path(s)
        prov = json.loads(prov_path.read_text()) if prov_path.exists() else None
        raw = sorted((roots.bids / f"sub-{s}").glob("ses-*/anat/*.nii.gz"))
        raw = [r for r in raw if r.name.endswith(("_T1w.nii.gz", "_T2w.nii.gz"))]
        for r in raw:
            rec = prov["files"].get(str(r)) if prov else None
            if rec is None:
                print(f"UNDEFACED  {r}  (no provenance entry in {prov_path})")
                bad += 1
            elif sha256(r) != rec["sha_after"]:
                print(f"CHANGED    {r}  (sha256 differs from the recorded defaced file)")
                bad += 1
            elif args.verbose:
                print(f"ok         {r}")
    print(f"{len(subjects)} subjects, {bad} problems")
    return 1 if bad else 0


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="deface one subject in place (originals mirrored to sourcedata first)")
    r.add_argument("--subject", required=True, help="bare label, e.g. 03")
    r.add_argument("--dry-run", action="store_true", help="compute masks, run every check, write montages; touch nothing in the BIDS tree")
    r.add_argument("--raw-only", action="store_true", help="converter guard: raw anat only, for a subject with no fMRIPrep")
    r.add_argument("--work", help="scratch dir for registrations (default: <source_dir>/anat_defacing/sub-XX/work)")
    r.set_defaults(func=cmd_run)
    v = sub.add_parser("verify", help="re-check a subject's provenance against the tree")
    v.add_argument("--subject", required=True)
    v.add_argument("--deep", action="store_true", help="also reload every volume and re-run the rewrite invariants")
    v.set_defaults(func=cmd_verify)
    rs = sub.add_parser("restore", help="put every defaced file of a subject back from its mirror")
    rs.add_argument("--subject", required=True)
    rs.set_defaults(func=cmd_restore)
    p = sub.add_parser("reports", help="move anat-rendering reports/figures/bundles out of the tree")
    p.add_argument("--subject", required=True)
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_reports)
    a = sub.add_parser("audit", help="every raw anat NIfTI in the tree has a defacing provenance entry")
    a.add_argument("--verbose", action="store_true")
    a.set_defaults(func=cmd_audit)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
