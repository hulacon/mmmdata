#!/usr/bin/env python3
"""ROI ladder masks for the neural-rotation pilot, rungs (i)-(iii).

The *state-space ladder* makes the level at which an operator is fitted a
measured factor. This script cuts the anatomical rungs from the staged
Harvard-Oxford maxprob-thr25 atlas on the shared MNI152NLin2009cAsym res-2
grid (the GLMsingle fit grid):

  rung i    every Harvard-Oxford ROI in occipital, temporal and parietal
            cortex, plus hippocampus, each alone, bilateral; plus the
            control ROIs (mPFC = Frontal Medial Cortex, the benchmark's
            definition), each alone, which are fitted like any rung-(i)
            ROI but are NOT part of the rung-(iii) union
  rung ii   VTC (Ye et al. 2020: inferior temporal, parahippocampal, temporal
            fusiform) and VTC + angular gyrus
  rung iii  the union of every rung-(i) ROI ("Posterior"), plus a dseg volume
            carrying each voxel's rung-(i) ROI index for block bookkeeping;
            plus the remapping-vs-capacity controls: random size-matched
            subsets of Posterior (RandUnion<size>S<seed>), so a union rung's
            gain can be read against "the same number of voxels with no
            anatomical structure"

Rung (iv), the pRF-defined populations, is build_prf_masks.py.

Output tree (``<output_dir>/functional_rois/``):

  dataset_description.json
  ladder.tsv       rung, roi, hemi, atlas, labels, label_names, n_vox_atlas,
                   n_vox_grid, too_small, in_posterior, note, n_vox_brain_<sub> ...
  ladder.json      the spec the run used (labels, min voxels, reference grid)
  space-MNI152NLin2009cAsym_res-2/
      atlas-HOthr25_label-<ROI>_mask.nii.gz       one per ladder row
      atlas-HOthr25_label-Posterior_dseg.nii.gz   rung-(iii) block labels
      atlas-HOthr25_label-Posterior_dseg.tsv

ROI names are the atlas names in CamelCase with the parenthetical dropped
and "division"/"part" removed (Heschl's Gyrus -> HeschlsGyrus). A ROI under
``--min-vox`` voxels on the grid is still written but flagged ``too_small``
and reported as such downstream, never as a null.

The reference grid is the fMRIPrep MNI res-2 brain mask of the first
subject found (all must agree; asserted), and the staged atlas must already
be on it (asserted; nothing is resampled here). Per-subject counts inside
each brain mask are a proxy for the GLMsingle mask; the exact count comes
from the cache.

Usage:
    python build_roi_ladder.py [--subjects sub-## ...] [--min-vox 100] [--dry-run]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).resolve().parent.parent


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


tb = _load_module("glmsingle_tb", SCRIPTS / "glmsingle_tb.py")
ps = _load_module("pattern_similarity_shared", SCRIPTS / "pattern_similarity" / "shared.py")

TREE = "functional_rois"
SPACE_DIR = f"space-{tb.SPACE}"
ATLAS_TAG = "HOthr25"

# Rung (i): "every Harvard-Oxford ROI in occipital, temporal and parietal
# cortex, plus hippocampus" -- this script's reading of the decided rungs
# (a GUESS in the design record; rows with a note are the edge cases).
RUNG_I_CORTICAL = {
    "occipital": [22, 23, 24, 32, 36, 40, 47, 48],
    "temporal": [8, 9, 10, 11, 12, 13, 14, 15, 16, 34, 35, 37, 38, 39, 44, 45, 46],
    "parietal": [17, 18, 19, 20, 21, 31, 43],
}
RUNG_I_SUBCORTICAL = {"hippocampus": [9, 19]}
# Control ROIs: single ROIs outside the stated lobes, fitted at rung (i) but
# never pooled into the rung-(iii) union. mPFC is the settled benchmark's
# definition (pattern_similarity PATTERN_CORTICAL_ROIS), added 2026-09-23 so
# the pilot reads the benchmark's one enc<->word cell.
RUNG_I_CONTROL = {"frontal": [25]}
# Remapping-vs-capacity controls (DECIDED Ben 2026-09-24): uniform random
# subsets of the Posterior union, sizes matched to the union ROIs of sub-03
# (VTCAG 10,630; PrfPosThr2p5 33,637; PrfUnionThr2p5 58,535), two seeds each.
RANDOM_UNIONS = {"10k": 10630, "33k": 33637, "58k": 58535}
RANDOM_SEEDS = (0, 1)
RANDOM_NOTE = "random size-matched subset of Posterior (remapping-vs-capacity control); no blocks"
CONTROL_NOTE = "control ROI outside the stated lobes; fitted alone, NOT in the Posterior union"
EDGE_NOTE = "edge of the stated lobes; prune on Ben's call"
EDGE_LABELS = {17, 43, 44, 46}
MIN_VOX_DEFAULT = 100


def camel(name: str) -> str:
    """Atlas label name -> ROI token: drop parentheticals, 'division'/'part',
    apostrophes; CamelCase the words. Alphanumeric only (a BIDS label)."""
    name = re.sub(r"\(.*?\)", "", name)
    name = name.replace("'", "")
    parts = [p.strip() for p in name.split(",")]
    words = []
    for p in parts:
        words += [w for w in p.split() if w.lower() not in ("division", "part")]
    token = "".join(w[:1].upper() + w[1:] for w in words)
    if not re.fullmatch(r"[A-Za-z0-9]+", token):
        raise ValueError(f"ROI token {token!r} from {name!r} is not alphanumeric")
    return token


def reference_grid(fmriprep_dir: Path, subjects: list):
    """(ref_img, {sub: brain mask bool}) -- every subject's res-2 mask on one grid."""
    ref, masks = None, {}
    for sub in subjects:
        hits = sorted((fmriprep_dir / sub / "anat").glob(
            f"{sub}_*space-{tb.SPACE.replace('_res-2', '')}_res-2_desc-brain_mask.nii.gz"))
        if not hits:
            sys.exit(f"ERROR: no MNI res-2 brain mask for {sub} under {fmriprep_dir}")
        img = nib.load(str(hits[0]))
        if ref is None:
            ref = img
        elif img.shape != ref.shape or not np.allclose(img.affine, ref.affine):
            sys.exit(f"ERROR: {sub} brain-mask grid {img.shape} differs from "
                     f"{subjects[0]}'s {ref.shape}; the ladder needs one grid")
        masks[sub] = np.asarray(img.dataobj) > 0
    return ref, masks


def discover_subjects(fmriprep_dir: Path) -> list:
    return sorted(p.name for p in fmriprep_dir.glob("sub-*")
                  if any((p / "anat").glob("*_res-2_desc-brain_mask.nii.gz")))


def build_ladder(atlases: dict, min_vox: int, brain: dict):
    """-> (rows list, {roi: bool mask}, block dseg ndarray, block tsv rows)."""
    cort, cort_labels = atlases["HOCPA"]
    sub, sub_labels = atlases["HOSPA"]
    rows, masks = [], {}

    def add(rung, roi, atlas, labels, names, mask, note="", in_posterior=True):
        n = int(mask.sum())
        row = {"rung": rung, "roi": roi, "hemi": "bilateral", "atlas": atlas,
               "labels": "+".join(str(v) for v in labels),
               "label_names": " | ".join(names),
               "n_vox_atlas": n, "n_vox_grid": n,      # atlas already on the grid
               "too_small": n < min_vox, "in_posterior": in_posterior, "note": note}
        for s, b in brain.items():
            row[f"n_vox_brain_{s.replace('sub-', 'sub')}"] = int((mask & b).sum())
        rows.append(row)
        masks[roi] = mask

    rung_i = []
    for lobe, values in RUNG_I_CORTICAL.items():
        for v in values:
            name = cort_labels[v]
            mask = ps.checked_label_mask(cort, cort_labels, {v: name})
            roi = camel(name)
            add("i", roi, "HOCPA", [v], [name], mask,
                EDGE_NOTE if v in EDGE_LABELS else lobe)
            rung_i.append(roi)
    for lobe, values in RUNG_I_SUBCORTICAL.items():
        names = [sub_labels[v] for v in values]
        mask = ps.checked_label_mask(sub, sub_labels, dict(zip(values, names)))
        add("i", "Hippocampus", "HOSPA", values, names, mask, lobe)
        rung_i.append("Hippocampus")
    for lobe, values in RUNG_I_CONTROL.items():
        for v in values:
            name = cort_labels[v]
            mask = ps.checked_label_mask(cort, cort_labels, {v: name})
            roi = "mPFC" if v == 25 else camel(name)     # keep the benchmark's name
            add("i", roi, "HOCPA", [v], [name], mask, f"{CONTROL_NOTE} ({lobe})",
                in_posterior=False)

    vtc_spec = ps.PATTERN_CORTICAL_ROIS["VTC"]
    vtc = ps.checked_label_mask(cort, cort_labels, vtc_spec)
    ag = ps.checked_label_mask(cort, cort_labels, ps.PATTERN_CORTICAL_ROIS["AG"])
    add("ii", "VTC", "HOCPA", list(vtc_spec), [cort_labels[v] for v in vtc_spec], vtc,
        "Ye et al. 2020 VTC; label 39 excluded by decision", in_posterior=False)
    add("ii", "VTCAG", "HOCPA", list(vtc_spec) + [21],
        [cort_labels[v] for v in vtc_spec] + [cort_labels[21]], vtc | ag,
        "VTC + angular gyrus", in_posterior=False)

    # Block labels for rung (iii). Each atlas is a maxprob partition on its
    # own, but the cortical and subcortical atlases overlap each other (the
    # hippocampus meets the parahippocampal labels), so the cortical label
    # keeps those voxels in the dseg; the single-ROI masks are untouched.
    posterior = np.zeros(cort.shape, dtype=bool)
    dseg = np.zeros(cort.shape, dtype=np.int16)
    block_rows, n_overlap = [], 0
    for k, roi in enumerate(rung_i, start=1):
        m = masks[roi]
        taken = (dseg != 0) & m
        n_overlap += int(taken.sum())
        dseg[m & ~taken] = k
        posterior |= m
        block_rows.append({"index": k, "name": roi, "n_vox_block": int((dseg == k).sum())})
    add("iii", "Posterior", "HOCPA+HOSPA", ["rung-i"], rung_i, posterior,
        f"union of every rung-(i) ROI except the control ROIs; {n_overlap} voxels in two ROIs "
        "(subcortical vs cortical atlas) carry the cortical block in the dseg", in_posterior=False)
    post_idx = np.flatnonzero(posterior.ravel(order="C"))
    for tag, n in RANDOM_UNIONS.items():
        for seed in RANDOM_SEEDS:
            rng = np.random.default_rng(20260924 + 1000 * seed + n)
            pick = rng.choice(post_idx, size=min(n, len(post_idx)), replace=False)
            m = np.zeros(posterior.size, dtype=bool)
            m[pick] = True
            add("iii", f"RandUnion{tag}S{seed}", "HOCPA+HOSPA", ["random"], [f"{n} of Posterior, seed {seed}"],
                m.reshape(posterior.shape, order="C"), RANDOM_NOTE, in_posterior=False)
    return rows, masks, dseg, block_rows


def write_tree(out_root: Path, ref_img, rows, masks, dseg, block_rows, spec: dict,
               sources: list) -> None:
    space = out_root / SPACE_DIR
    space.mkdir(parents=True, exist_ok=True)
    dd = out_root / "dataset_description.json"
    if not dd.exists():
        with open(dd, "w") as f:
            json.dump({
                "Name": "Functional and anatomical ROI masks (neural-rotation ladder)",
                "BIDSVersion": "1.8.0",
                "DatasetType": "derivative",
                "GeneratedBy": [{"Name": "mmmdata/scripts/neural_rotation/build_roi_ladder.py",
                                 "Description": "state-space ladder rungs (i)-(iii); "
                                                "design record: mmmdata-agents "
                                                "docs/workbench/neural-rotation-pilot/"},
                                {"Name": "mmmdata/scripts/neural_rotation/build_prf_masks.py",
                                 "Description": "rung (iv), pRF-defined populations "
                                                "under sub-*/"}],
                "SourceDatasets": [{"URL": str(s)} for s in sources],
            }, f, indent=2)
    for roi, m in masks.items():
        nib.save(nib.Nifti1Image(m.astype(np.uint8), ref_img.affine),
                 space / f"atlas-{ATLAS_TAG}_label-{roi}_mask.nii.gz")
    nib.save(nib.Nifti1Image(dseg, ref_img.affine),
             space / f"atlas-{ATLAS_TAG}_label-Posterior_dseg.nii.gz")
    pd.DataFrame(block_rows).to_csv(space / f"atlas-{ATLAS_TAG}_label-Posterior_dseg.tsv",
                                    sep="\t", index=False)
    pd.DataFrame(rows).to_csv(out_root / "ladder.tsv", sep="\t", index=False)
    with open(out_root / "ladder.json", "w") as f:
        json.dump(spec, f, indent=2)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subjects", nargs="*", default=None,
                    help="subjects whose brain masks give the per-subject counts "
                         "(default: every sub-* with an MNI res-2 brain mask)")
    ap.add_argument("--min-vox", type=int, default=MIN_VOX_DEFAULT,
                    help="ROIs under this many grid voxels are flagged too_small")
    ap.add_argument("--fmriprep-dir", default=None)
    ap.add_argument("--atlases-dir", default=None, help="override derivatives/atlases")
    ap.add_argument("--out-root", default=None, help=f"override <output_dir>/{TREE}")
    ap.add_argument("--dry-run", action="store_true", help="build and print; write nothing")
    args = ap.parse_args()

    cfg = tb.load_config()
    bids_root = Path(cfg["bids_project_dir"])
    output_dir = Path(cfg["output_dir"])
    fmriprep_dir = Path(args.fmriprep_dir) if args.fmriprep_dir else output_dir / "fmriprep"
    atlases_dir = Path(args.atlases_dir) if args.atlases_dir else output_dir / "atlases"
    out_root = Path(args.out_root) if args.out_root else output_dir / TREE
    subjects = args.subjects or discover_subjects(fmriprep_dir)
    if not subjects:
        sys.exit(f"ERROR: no subjects with an MNI res-2 brain mask under {fmriprep_dir}")

    print(f"=== ROI ladder: rungs (i)-(iii) on {tb.SPACE}; subjects {subjects} ===")
    ref_img, brain = reference_grid(fmriprep_dir, subjects)
    atlases, affine = ps.load_ho_on_grid(source="staged", atlases_dir=atlases_dir)
    if atlases["HOCPA"][0].shape != ref_img.shape[:3] or not np.allclose(affine, ref_img.affine):
        sys.exit(f"ERROR: staged atlas grid {atlases['HOCPA'][0].shape} != reference "
                 f"grid {ref_img.shape[:3]} (or affines differ); re-run "
                 "scripts/stage_harvard_oxford.py against this reference")

    rows, masks, dseg, block_rows = build_ladder(atlases, args.min_vox, brain)
    df = pd.DataFrame(rows)
    print(df[["rung", "roi", "labels", "n_vox_grid", "too_small", "in_posterior", "note"]].to_string(index=False))
    print(f"{len(rows)} ladder rows; {int(df['too_small'].sum())} too small (< {args.min_vox})")

    spec = {"space": tb.SPACE, "atlas": "Harvard-Oxford maxprob-thr25 (staged)",
            "atlases_dir": str(atlases_dir), "reference_grid": {
                "shape": list(ref_img.shape[:3]), "affine": ref_img.affine.tolist()},
            "min_vox": args.min_vox, "rung_i_cortical": RUNG_I_CORTICAL,
            "rung_i_subcortical": RUNG_I_SUBCORTICAL, "rung_i_control": RUNG_I_CONTROL,
            "random_unions": RANDOM_UNIONS, "random_seeds": list(RANDOM_SEEDS),
            "vtc_labels": list(ps.PATTERN_CORTICAL_ROIS["VTC"]),
            "subjects_counted": subjects}
    if args.dry_run:
        print("DRY RUN — nothing written.")
        return
    write_tree(out_root, ref_img, rows, masks, dseg, block_rows, spec,
               [atlases_dir, fmriprep_dir])
    print(f"wrote {out_root}/ladder.tsv, ladder.json and {len(masks) + 1} volumes under {SPACE_DIR}/")


if __name__ == "__main__":
    main()
