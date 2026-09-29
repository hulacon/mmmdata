#!/usr/bin/env python3
"""
prf_splithalf_project.py — place a split-half pRF fit in its derivative tree,
project it to fsnative with the pooled product's own projector, and resample
fsnative -> fsaverage6.

Steps (for one subject and one half's session, or ``--pooled``):

1. **Place** (halves only): the assembler (``scripts/prf_analyzeprf_assemble.py``)
   writes ``sub-##_task-prf_space-T1w_*`` into a staging root; the files are
   moved to ``<derivatives>/<tree>/sub-##/ses-##/`` with a ``ses-`` entity
   (``sub-##_ses-##_task-prf_space-T1w_desc-<param>_prf.nii.gz``).
2. **fsnative**: ``scripts/project_prf_fsnative.project_unit``, unchanged —
   the same transform chain, 3-depth average, NaN-aware trilinear sampling
   and cos/sin angle handling as the pooled product. (``--pooled`` skips this:
   the pooled product already has fsnative maps.)
3. **fsaverage6**: nearest-neighbour resampling on the FreeSurfer spherical
   registration. Each fsaverage6 vertex takes the value of the subject vertex
   nearest to it on ``?h.sphere.reg`` (the subject's sphere registered to
   fsaverage), compared with fsaverage6's ``?h.sphere.reg``. Nearest neighbour
   rather than interpolation because the subject mesh is several times denser
   than fsaverage6 (the nearest vertex is well under a millimetre of cortex
   away) and because it never averages polar angles, so no cos/sin detour is
   needed at this step. Every parameter goes through the same mapping.

Outputs: ``..._space-fsaverage6_hemi-<L|R>_desc-<param>_prf.shape.gii`` beside
the fsnative maps (halves) or under ``<derivatives>/<tree>/sub-##/`` (pooled),
plus a sidecar naming the method.

Usage:
    python prf_splithalf_project.py --subject ## --session ## --staging <root>
    python prf_splithalf_project.py --subject ## --pooled
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import project_prf_fsnative as proj  # noqa: E402

TREE = "functional_space/prf_splithalf"
POLARITY = "prf"
FS_HEMI = {"L": "lh", "R": "rh"}


def fs_root() -> Path:
    return proj.DERIV_ROOT / "fmriprep" / "sourcedata" / "freesurfer"


def nn_index(subject: str, hemi: str) -> np.ndarray:
    """For each fsaverage6 vertex, the nearest subject vertex on sphere.reg."""
    import nibabel as nib
    from scipy.spatial import cKDTree

    src, _ = nib.freesurfer.read_geometry(str(fs_root() / f"sub-{subject}" / "surf" / f"{FS_HEMI[hemi]}.sphere.reg"))
    trg, _ = nib.freesurfer.read_geometry(str(fs_root() / "fsaverage6" / "surf" / f"{FS_HEMI[hemi]}.sphere.reg"))
    # Both spheres are radius-100 FreeSurfer spheres; normalise anyway so a
    # radius difference cannot bias the match.
    src = src / np.linalg.norm(src, axis=1, keepdims=True)
    trg = trg / np.linalg.norm(trg, axis=1, keepdims=True)
    dist, idx = cKDTree(src).query(trg)
    print(f"    hemi-{hemi}: {len(trg)} fsaverage6 <- {len(src)} fsnative vertices, "
          f"median NN chord {np.median(dist) * 100:.2f} (radius-100 units)")
    return idx


def place_half(staging: Path, subject: str, session: str, out_dir: Path) -> None:
    src_dir = staging / f"sub-{subject}"
    files = sorted(src_dir.glob(f"sub-{subject}_task-prf_space-T1w_*{POLARITY}.*"))
    if not files:
        sys.exit(f"ERROR: no assembled `{POLARITY}` files in {src_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        name = f.name.replace(f"sub-{subject}_task-prf", f"sub-{subject}_ses-{session}_task-prf", 1)
        shutil.move(str(f), str(out_dir / name))
    sc = out_dir / f"sub-{subject}_ses-{session}_task-prf_space-T1w_{POLARITY}.json"
    meta = json.loads(sc.read_text())
    meta["Maps"] = [m.replace(f"sub-{subject}_task-prf", f"sub-{subject}_ses-{session}_task-prf", 1)
                    for m in meta.get("Maps", [])]
    sc.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"  placed {len(files)} files in {out_dir}")


def resample_fsaverage6(subject: str, src_dir: Path, src_entities: str, out_dir: Path, out_entities: str) -> list:
    import nibabel as nib

    written = []
    for hemi in FS_HEMI:
        idx = nn_index(subject, hemi)
        for p in proj.PARAMS:
            src = src_dir / f"{src_entities}_task-prf_space-fsnative_hemi-{hemi}_desc-{p}_{POLARITY}.shape.gii"
            vals = np.asarray(nib.load(str(src)).darrays[0].data, dtype=np.float32)
            path = out_dir / f"{out_entities}_task-prf_space-fsaverage6_hemi-{hemi}_desc-{p}_{POLARITY}.shape.gii"
            proj.write_shape_gii(path, vals[idx], hemi)
            written.append(path.name)
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject", required=True)
    ap.add_argument("--session", help="the half's session (bare label)")
    ap.add_argument("--staging", type=Path, help="assembler --out-root for this half")
    ap.add_argument("--pooled", action="store_true", help="resample the pooled product's fsnative maps only")
    args = ap.parse_args()
    s = args.subject
    tree_root = proj.DERIV_ROOT / TREE

    if args.pooled:
        src_dir = proj.DERIV_ROOT / proj.TREES["pooled"][0] / f"sub-{s}"
        out_dir = tree_root / f"sub-{s}"
        out_dir.mkdir(parents=True, exist_ok=True)
        written = resample_fsaverage6(s, src_dir, f"sub-{s}", out_dir, f"sub-{s}")
        source = f"derivatives/{proj.TREES['pooled'][0]}/sub-{s} (pooled product, fsnative)"
        entities = f"sub-{s}"
    else:
        if not (args.session and args.staging):
            ap.error("--session and --staging are required unless --pooled")
        out_dir = tree_root / f"sub-{s}" / f"ses-{args.session}"
        if args.staging.exists():
            place_half(args.staging, s, args.session, out_dir)
        entities = f"sub-{s}_ses-{args.session}"
        unit = {"kind": "pooled", "subject": s, "session": args.session, "tree": TREE, "space": "T1w",
                "entities": entities, "label": f"sub-{s} half ses-{args.session}", "dir": out_dir}
        proj.project_unit(unit, POLARITY)
        written = resample_fsaverage6(s, out_dir, entities, out_dir, entities)
        source = f"{TREE}/sub-{s}/ses-{args.session} fsnative maps (project_prf_fsnative.project_unit)"

    meta = {
        "Description": "pRF parameters resampled fsnative -> fsaverage6",
        "Source": source,
        "Method": ("nearest neighbour on the FreeSurfer spherical registration: each fsaverage6 vertex takes "
                   "the value of the nearest subject vertex on ?h.sphere.reg (unit-normalised) vs fsaverage6 "
                   "?h.sphere.reg; no interpolation, so polar angle needs no cos/sin handling at this step"),
        "Polarity": POLARITY,
        "Maps": written,
        "Provenance": "mmmdata/scripts/functional_space/prf_splithalf_project.py",
    }
    (out_dir / f"{entities}_task-prf_space-fsaverage6_{POLARITY}.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"  wrote {len(written)} fsaverage6 maps to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
