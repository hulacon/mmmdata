"""Face audit: no anatomical face anywhere in a dataset tree.

Read-only. It runs on the working tree, a collaborator share tree, or a
release stage, and needs the defacing records that ``scripts/deface_anat.py``
writes under the source-data directory
(``<source_dir>/anat_defacing/sub-<label>/provenance.json`` and the face and
brain masks it names). Two checks, for every ``sub-<label>`` directory found
anywhere under the root (raw tree and derivatives alike):

1. **Provenance (raw anatomy).** Every raw ``sub-*/ses-*/anat/*_T1w`` /
   ``*_T2w`` NIfTI at the top of the root has a provenance entry and hashes to
   the recorded defaced file. Entries are keyed by the working-tree path, so a
   copy is matched by its path relative to the configured BIDS root.

2. **Content (every volume).** Every 3-D NIfTI/MGH volume of the subject
   outside the EPI datatypes is scored for signal inside the subject's
   recorded face mask: the fraction of the eroded mask core whose magnitude
   exceeds ``SIGNAL_REL`` of the volume's 99th percentile. A defaced volume
   scores near 0 and a faced one well above ``FACE_SIGNAL_MAX``; the
   threshold and the deeper erosion of a carried mask were set by scoring
   every defaced volume against its faced original (the calibration record
   is kept with the de-identification effort, not in this repo). Limit:
   in the 2 mm ``MNI152NLin2009cAsym`` grid the face voxels can sit below
   the signal floor even in a faced original, so the MNI path may not tell
   faced from defaced there.
   The mask comes from

   * a recorded leg whose mask is on the volume's exact grid (``grid``);
   * otherwise, by the volume's ``space-`` entity: ``MNI152NLin2009cAsym``
     takes the MNI leg, a native space takes the T1w leg, carried by world
     coordinates (``world``). Native spaces share the scanner world: the
     fMRIPrep T1w, FreeSurfer's conformed grid, raw runs, hippocampal crops.
     A carried mask is eroded deeper, to absorb the carry error.

   A volume with no ``space-`` entity (FreeSurfer internals, unnamed
   outputs) is native unless its grid spans the MNI leg's bounding box
   (FreeSurfer 8 writes a head resampled to MNI152 under
   ``mri/transforms/synthmorph*/``). A subject defaced raw-only (no
   fMRIPrep at the time) has no T1w mask; its first raw T1w run's mask
   stands in for the native world.

   A volume in a space with no recorded mask (another template, ``ACPC``) is
   reported as ``unplaced``: the audit cannot vouch for it, so it fails until
   someone gives the space a mask. A mask that falls outside the volume's
   field of view leaves too small a core to hold a face; that volume passes.

The content check is what covers derivatives nobody listed: a new pipeline's
outputs (FastSurfer, YADO, HippUnfold, QSIPrep) are scored like any other
volume, with no table of file names to keep up to date.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .sweep import SKIP_DIRS, Finding

PROVENANCE_REL = Path("anat_defacing")
PROVENANCE_NAME = "provenance.json"

VOLUME_EXTENSIONS = (".nii", ".nii.gz", ".mgz", ".mgh")
# CIFTI grayordinates share the .nii extension but hold surface vertices and
# MNI subcortical voxels inside the brain: no voxel grid, and no face
CIFTI_EXTENSIONS = tuple(f".{k}.nii" for k in (
    "dtseries", "dscalar", "dlabel", "dconn", "ptseries", "pscalar", "plabel", "pconn", "pdconn",
    "dpconn", "dfan", "fiberTEMP"))
# EPI and non-image datatypes: no face is defaced in them, by field practice
EXCLUDED_DATATYPE_DIRS = {"func", "dwi", "fmap", "perf", "beh", "eeg", "meg", "ieeg", "pet", "nirs",
                          "motion", "micr"}
EXCLUDED_SUFFIXES = {"bold", "sbref", "boldref", "dwi", "epi", "cbv", "asl", "m0scan", "physio",
                     "phasediff", "magnitude", "magnitude1", "magnitude2", "phase1", "phase2", "fieldmap",
                     # statistical, label and mask maps: defacing practice covers intensity images
                     # only, and an UNMASKED map (GLMsingle fits) has values everywhere, which
                     # the content score cannot tell from a face
                     "statmap", "stat", "dseg", "probseg", "mask"}
TOP_LEVEL_SKIP = {"sourcedata", "stimuli", "code", "logs"}

MNI_SPACES = {"MNI152NLin2009cAsym"}
# spaces in the subject's native scanner world (no entity = native)
NATIVE_SPACES = {None, "T1w", "T2w", "fsnative", "orig", "anat", "cropT1w", "cropT2w", "corobl"}
# not anatomical coordinates at all: a face cannot be drawn in them
NON_ANATOMICAL_SPACES = {"unfold", "unfoldiso"}

SIGNAL_REL = 0.02          # a voxel "has signal" above this fraction of the 99th percentile
FACE_SIGNAL_MAX = 0.10     # face-core fraction above this -> face
ERODE_MM_GRID = 2.0        # erosion of a same-grid mask
ERODE_MM_WORLD = 6.0       # erosion of a world-carried mask (absorbs the carry error)
MIN_CORE = 1000            # fewer core voxels in view: the face is outside the field of view
MNI_BOX_TOL_MM = 2.5       # a volume with no space- entity spanning the MNI leg's box is in MNI

_SUB_RE = re.compile(r"^sub-([A-Za-z0-9]+)$")
_SPACE_RE = re.compile(r"(?:^|_)space-([A-Za-z0-9]+)")


@dataclass
class VolumeScore:
    path: str            # root-relative
    subject: str
    mask: str            # leg name, or "" when none applied
    method: str          # grid | world | outside-fov | skipped | unplaced | no-provenance
    core: int
    signal: float | None
    verdict: str         # ok | face | unplaced | no-provenance | unreadable

    def row(self) -> str:
        sig = "n/a" if self.signal is None else f"{self.signal:.4f}"
        return "\t".join([self.path, self.subject, self.mask or "n/a", self.method, str(self.core), sig,
                          self.verdict])


TSV_HEADER = "path\tsubject\tmask\tmethod\tcore_voxels\tface_signal\tverdict"


@dataclass
class AuditResult:
    findings: list[Finding] = field(default_factory=list)
    scores: list[VolumeScore] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# selection                                                                   #
# --------------------------------------------------------------------------- #
def _is_volume(name: str) -> bool:
    return name.endswith(VOLUME_EXTENSIONS) and not name.endswith(CIFTI_EXTENSIONS)


def _suffix(name: str) -> str:
    stem = name
    for ext in VOLUME_EXTENSIONS:
        if stem.endswith(ext):
            stem = stem[: -len(ext)]
            break
    return stem.rsplit("_", 1)[-1]


def space_of(name: str) -> str | None:
    m = _SPACE_RE.search(name)
    return m.group(1) if m else None


def subject_of(rel: Path) -> str | None:
    """The label of the first ``sub-<label>`` path component, if any."""
    for part in rel.parts[:-1]:
        m = _SUB_RE.match(part)
        if m:
            return m.group(1)
    return None


def candidate_volumes(root: Path) -> list[tuple[Path, str]]:
    """(root-relative path, subject) of every volume the content check scores."""
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        here = Path(dirpath)
        rel_dir = here.relative_to(root)
        if here == root:
            dirnames[:] = [d for d in dirnames if d not in TOP_LEVEL_SKIP]
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and d not in EXCLUDED_DATATYPE_DIRS)
        for name in sorted(filenames):
            if not _is_volume(name) or _suffix(name) in EXCLUDED_SUFFIXES:
                continue
            rel = rel_dir / name
            sub = subject_of(rel)
            if sub is not None:
                out.append((rel, sub))
    return out


# --------------------------------------------------------------------------- #
# masks                                                                       #
# --------------------------------------------------------------------------- #
@dataclass
class Leg:
    name: str
    face: np.ndarray       # bool
    affine: np.ndarray

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(self.face.shape[:3])


def provenance_path(source_dir: Path, subject: str) -> Path:
    return Path(source_dir) / PROVENANCE_REL / f"sub-{subject}" / PROVENANCE_NAME


class SubjectMasks:
    """A subject's recorded face masks, and carried copies cached per grid."""

    def __init__(self, subject: str, prov: dict):
        import nibabel as nib

        self.subject = subject
        self.prov = prov
        self.legs: dict[str, Leg] = {}
        for name, leg in prov.get("legs", {}).items():
            img = nib.load(leg["facemask"])
            self.legs[name] = Leg(name, np.asanyarray(img.dataobj).squeeze() > 0, img.affine)
        self._cache: dict[tuple, np.ndarray] = {}

    @classmethod
    def load(cls, source_dir: Path, subject: str) -> "SubjectMasks | None":
        p = provenance_path(source_dir, subject)
        if not p.exists():
            return None
        return cls(subject, json.loads(p.read_text()))

    def on_grid(self, shape, affine) -> Leg | None:
        for leg in self.legs.values():
            if leg.shape == tuple(shape[:3]) and np.allclose(leg.affine, affine, atol=1e-3):
                return leg
        return None

    def native(self) -> Leg | None:
        """The T1w leg, or for a raw-only subject its first raw T1w run's."""
        if "T1w" in self.legs:
            return self.legs["T1w"]
        raw = sorted(k for k in self.legs if k.startswith("raw:"))
        return self.legs[raw[0]] if raw else None

    def carried(self, leg: Leg, shape, affine) -> np.ndarray:
        key = (leg.name, tuple(shape[:3]), np.round(affine, 4).tobytes())
        if key not in self._cache:
            self._cache[key] = carry_mask(leg.face, leg.affine, tuple(shape[:3]), affine)
        return self._cache[key]


def carry_mask(mask: np.ndarray, mask_affine: np.ndarray, shape: tuple, affine: np.ndarray) -> np.ndarray:
    """Nearest-neighbour resample of a boolean mask onto another grid through
    world coordinates (both affines map voxels to the same world)."""
    ijk = np.indices(shape).reshape(3, -1)
    world = affine @ np.vstack([ijk, np.ones(ijk.shape[1])])
    src = np.rint((np.linalg.inv(mask_affine) @ world)[:3]).astype(int)
    ok = np.all((src >= 0) & (src < np.array(mask.shape[:3])[:, None]), axis=0)
    out = np.zeros(ijk.shape[1], bool)
    out[ok] = mask[src[0, ok], src[1, ok], src[2, ok]]
    return out.reshape(shape)


def _box(shape, affine) -> np.ndarray:
    """World coordinates of the first and last voxel centres."""
    lo = affine @ np.array([0, 0, 0, 1.0])
    hi = affine @ np.array([*(np.array(shape[:3]) - 1), 1.0])
    return np.sort(np.vstack([lo[:3], hi[:3]]), axis=0)


def spans_leg_box(shape, affine, leg: Leg) -> bool:
    return bool(np.all(np.abs(_box(shape, affine) - _box(leg.shape, leg.affine)) <= MNI_BOX_TOL_MM))


def face_signal(data: np.ndarray, face: np.ndarray, zooms, erode_mm: float) -> tuple[float | None, int]:
    """(fraction of the eroded face core with signal, core size). None when
    the volume is empty."""
    from scipy import ndimage

    mag = np.abs(np.asarray(data, dtype=np.float32))
    nz = mag[mag > 0]
    if nz.size == 0:
        return None, 0
    thr = SIGNAL_REL * float(np.percentile(nz, 99))
    it = max(1, int(round(erode_mm / float(np.mean(zooms[:3])))))
    core = ndimage.binary_erosion(face, iterations=it)
    n = int(core.sum())
    if n == 0:
        return None, 0
    return float((mag[core] > thr).mean()), n


def score_volume(path: Path, rel: Path, masks: SubjectMasks) -> VolumeScore:
    import nibabel as nib

    sub = masks.subject
    space = space_of(path.name)
    if space in NON_ANATOMICAL_SPACES:
        return VolumeScore(rel.as_posix(), sub, "", "skipped", 0, None, "ok")
    img = nib.load(str(path))
    shape = img.shape
    if len(shape) > 3 and any(s > 1 for s in shape[3:]):
        return VolumeScore(rel.as_posix(), sub, "", "skipped", 0, None, "ok")
    leg = masks.on_grid(shape, img.affine)
    if leg is not None:
        face, method, erode = leg.face, "grid", ERODE_MM_GRID
    else:
        mni = masks.legs.get("MNI")
        if space in MNI_SPACES:
            leg = mni
        elif space is None and mni is not None and spans_leg_box(shape, img.affine, mni):
            leg = mni
        elif space in NATIVE_SPACES:
            leg = masks.native()
        if leg is None:
            return VolumeScore(rel.as_posix(), sub, "", "unplaced", 0, None, "unplaced")
        face, method, erode = masks.carried(leg, shape, img.affine), "world", ERODE_MM_WORLD
    data = np.asanyarray(img.dataobj).reshape(shape[:3])
    sig, n = face_signal(data, face, img.header.get_zooms(), erode)
    if sig is None or n < MIN_CORE:
        return VolumeScore(rel.as_posix(), sub, leg.name, "outside-fov", n, sig, "ok")
    return VolumeScore(rel.as_posix(), sub, leg.name, method, n, sig, "face" if sig > FACE_SIGNAL_MAX else "ok")


# --------------------------------------------------------------------------- #
# provenance check of raw anatomy                                             #
# --------------------------------------------------------------------------- #
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def raw_anat(root: Path) -> list[Path]:
    out = []
    for sub in sorted(p for p in root.glob("sub-*") if p.is_dir()):
        for p in sorted(sub.glob("ses-*/anat/*.nii.gz")):
            if p.name.endswith(("_T1w.nii.gz", "_T2w.nii.gz")):
                out.append(p)
    return out


def check_raw_provenance(root: Path, bids_root: Path, masks: dict[str, SubjectMasks | None]) -> list[Finding]:
    findings = []
    for p in raw_anat(root):
        rel = p.relative_to(root)
        sub = subject_of(rel)
        m = masks.get(sub)
        rec = m.prov.get("files", {}).get(str(Path(bids_root) / rel)) if m else None
        if rec is None:
            findings.append(Finding(rel.as_posix(), 0, "undefaced", "no defacing provenance entry"))
        elif _sha256(p) != rec["sha_after"]:
            findings.append(Finding(rel.as_posix(), 0, "changed", "sha256 differs from the recorded defaced file"))
    return findings


# --------------------------------------------------------------------------- #
# the audit                                                                   #
# --------------------------------------------------------------------------- #
def audit(root: str | Path, bids_root: str | Path, source_dir: str | Path,
          subjects: set[str] | None = None, log=None) -> AuditResult:
    """Both checks over ``root``. Findings fail the audit; ``scores`` holds a
    row per scored volume for the record."""
    root = Path(root)
    res = AuditResult()
    vols = [(rel, s) for rel, s in candidate_volumes(root) if subjects is None or s in subjects]
    labels = sorted({s for _, s in vols} | {subject_of(p.relative_to(root)) for p in raw_anat(root)})
    if subjects is not None:
        labels = [s for s in labels if s in subjects]
    masks = {s: SubjectMasks.load(Path(source_dir), s) for s in labels}
    res.findings += [f for f in check_raw_provenance(root, Path(bids_root), masks)
                     if subjects is None or subject_of(Path(f.path)) in subjects]
    for k, (rel, sub) in enumerate(vols, 1):
        m = masks.get(sub)
        if m is None:
            sc = VolumeScore(rel.as_posix(), sub, "", "no-provenance", 0, None, "no-provenance")
        else:
            try:
                sc = score_volume(root / rel, rel, m)
            except Exception as exc:          # fail closed on this volume, keep auditing the rest
                sc = VolumeScore(rel.as_posix(), sub, "", f"unreadable: {type(exc).__name__}: {exc}"[:200],
                                 0, None, "unreadable")
        res.scores.append(sc)
        if sc.verdict == "face":
            res.findings.append(Finding(sc.path, 0, "face",
                                        f"face-core signal {sc.signal:.3f} > {FACE_SIGNAL_MAX} ({sc.mask}, {sc.method})"))
        elif sc.verdict == "unplaced":
            res.findings.append(Finding(sc.path, 0, "unplaced",
                                        f"space-{space_of(rel.name)} has no recorded face mask"))
        elif sc.verdict == "unreadable":
            res.findings.append(Finding(sc.path, 0, "unreadable", sc.method))
        elif sc.verdict == "no-provenance":
            res.findings.append(Finding(sc.path, 0, "no-provenance",
                                        f"no {provenance_path(Path(source_dir), sub)}"))
        if log and k % 500 == 0:
            log(f"  scored {k}/{len(vols)} volumes")
    return res
