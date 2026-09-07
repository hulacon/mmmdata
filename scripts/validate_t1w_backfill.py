#!/usr/bin/env python
"""Vet an fMRIPrep T1w-space backfill against the tree it reused.

Four checks on one session, each printed and written to a JSON record:

1. grid      — every ``space-T1w_desc-preproc_bold`` keeps the BOLD voxel size,
               adopts the T1w orientation (rotation part of the affine matches
               the subject's preproc T1w), and its FOV is the brain-mask box.
2. reuse     — the fMRIPrep log ran no head-motion / coregistration nodes
               (counts of node lines matching the fit workflows).
3. confounds — the backfill's regenerated confounds TSVs are byte-identical to
               the tree's (decides whether production can write in place).
4. maps      — a GLM z-map fitted on the fMRIPrep T1w-space BOLD vs the same
               model fitted in native ``func`` space and carried into the T1w
               grid with the stored ``from-boldref_to-T1w`` affine: Pearson r
               inside the T1w brain mask and Dice at |z| > 3.1. This is the
               cost of the after-the-fact resample the backfill replaces.

Usage:
  validate_t1w_backfill.py --subject 03 --session 30 --task motor \
      --tree <derivatives>/fmriprep --backfill <staging>/fmriprep \
      --glm-dir <staging>/derivatives/glm_localizer --log <sbatch .out> \
      --contrast handVsRest --out <staging>/validation.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import nibabel as nib
import numpy as np


def _md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def _rotation(aff: np.ndarray) -> np.ndarray:
    R = aff[:3, :3]
    return R / np.linalg.norm(R, axis=0)[None, :]


def check_grid(tree: Path, backfill: Path, sub: str, ses: str, task: str) -> dict:
    anat = next((tree / f"sub-{sub}" / "anat").glob(f"sub-{sub}*_desc-preproc_T1w.nii.gz"))
    t1 = nib.load(anat)
    out = {"t1w": str(anat), "runs": []}
    func = backfill / f"sub-{sub}" / f"ses-{ses}" / "func"
    for bold in sorted(func.glob(f"sub-{sub}_ses-{ses}_task-{task}_*space-T1w_desc-preproc_bold.nii.gz")):
        native = tree / f"sub-{sub}" / f"ses-{ses}" / "func" / bold.name.replace("_space-T1w", "")
        mask = bold.with_name(bold.name.replace("desc-preproc_bold", "desc-brain_mask"))
        b, n, m = nib.load(bold), nib.load(native), nib.load(mask)
        rot_dev = float(np.degrees(np.arccos(np.clip((np.trace(_rotation(b.affine) @ _rotation(t1.affine).T) - 1) / 2, -1, 1))))
        idx = np.argwhere(np.asanyarray(m.dataobj) > 0)
        fov_slack = (np.array(b.shape[:3]) - (idx.max(0) - idx.min(0) + 1)).tolist()
        out["runs"].append({
            "bold": bold.name, "shape": list(b.shape), "zooms": [round(float(z), 4) for z in b.header.get_zooms()[:3]],
            "native_zooms": [round(float(z), 4) for z in n.header.get_zooms()[:3]],
            "rotation_vs_T1w_deg": round(rot_dev, 4), "fov_minus_mask_bbox": fov_slack,
            "same_zooms_as_native": bool(np.allclose(b.header.get_zooms()[:3], n.header.get_zooms()[:3], atol=1e-3)),
        })
    out["ok"] = bool(out["runs"]) and all(r["same_zooms_as_native"] and r["rotation_vs_T1w_deg"] < 0.01 for r in out["runs"])
    return out


def check_reuse(log: Path) -> dict:
    text = log.read_text(errors="replace")
    pats = {"hmc_nodes": r"bold_hmc_wf", "bbreg_nodes": r"bbreg_wf|bold_reg_wf|coreg", "fmap_est": r"fmap_preproc_wf|sdcflows",
            "precomputed_msgs": r"[Pp]recomputed", "resample_nodes": r"bold_volumetric_resample_wf|resample", "errors": r"Error|Traceback|crash"}
    counts = {k: len(re.findall(p, text)) for k, p in pats.items()}
    m = re.search(r"Exit code: (\d+)", text)
    counts["exit_code"] = int(m.group(1)) if m else None
    counts["ok"] = counts["hmc_nodes"] == 0 and counts["bbreg_nodes"] == 0 and counts["exit_code"] == 0
    return counts


def check_confounds(tree: Path, backfill: Path, sub: str, ses: str) -> dict:
    out = {"runs": []}
    for new in sorted((backfill / f"sub-{sub}" / f"ses-{ses}" / "func").glob("*_desc-confounds_timeseries.tsv")):
        old = tree / f"sub-{sub}" / f"ses-{ses}" / "func" / new.name
        same = old.exists() and _md5(old) == _md5(new)
        out["runs"].append({"file": new.name, "identical": bool(same)})
    out["ok"] = bool(out["runs"]) and all(r["identical"] for r in out["runs"])
    return out


def check_maps(tree: Path, backfill: Path, glm_dir: Path, sub: str, ses: str, task: str, contrast: str, z_thr: float) -> dict:
    import nitransforms as nt
    d = glm_dir / f"sub-{sub}" / f"ses-{ses}" / "func"
    z_t1w = nib.load(d / f"sub-{sub}_ses-{ses}_task-{task}_space-T1w_contrast-{contrast}_stat-z_statmap.nii.gz")
    z_func = nib.load(d / f"sub-{sub}_ses-{ses}_task-{task}_space-func_contrast-{contrast}_stat-z_statmap.nii.gz")
    func = tree / f"sub-{sub}" / f"ses-{ses}" / "func"
    xfm_path = sorted(func.glob(f"sub-{sub}_ses-{ses}_task-{task}_run-01_from-boldref_to-T1w_mode-image_desc-coreg_xfm.txt"))[0]
    xfm = nt.linear.load(xfm_path, fmt="itk")
    carried = xfm.apply(z_func, reference=z_t1w, order=3)
    mask_path = sorted((backfill / f"sub-{sub}" / f"ses-{ses}" / "func").glob(f"sub-{sub}_ses-{ses}_task-{task}_run-01_space-T1w_desc-brain_mask.nii.gz"))[0]
    m = np.asanyarray(nib.load(mask_path).dataobj) > 0
    a, b = np.asanyarray(z_t1w.dataobj)[m], np.asanyarray(carried.dataobj)[m]
    good = np.isfinite(a) & np.isfinite(b)
    a, b = a[good], b[good]
    r = float(np.corrcoef(a, b)[0, 1])
    ta, tb = np.abs(a) > z_thr, np.abs(b) > z_thr
    dice = float(2 * (ta & tb).sum() / max(ta.sum() + tb.sum(), 1))
    out = {"contrast": contrast, "xfm": xfm_path.name, "n_vox": int(good.sum()), "pearson_r": round(r, 4),
           "z_threshold": z_thr, "n_supra_fmriprep_T1w": int(ta.sum()), "n_supra_carried_func": int(tb.sum()), "dice": round(dice, 4)}
    carried_path = d / f"sub-{sub}_ses-{ses}_task-{task}_space-T1w_contrast-{contrast}_stat-z_desc-carriedFromFunc_statmap.nii.gz"
    carried.to_filename(carried_path)
    out["carried_map"] = str(carried_path)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subject", required=True); p.add_argument("--session", required=True); p.add_argument("--task", default="motor")
    p.add_argument("--tree", type=Path, required=True, help="the reused fMRIPrep tree")
    p.add_argument("--backfill", type=Path, required=True, help="the backfill's fMRIPrep output dir")
    p.add_argument("--glm-dir", type=Path, default=None, help="glm_localizer dir holding space-T1w and space-func fits")
    p.add_argument("--log", type=Path, default=None, help="the backfill sbatch .out")
    p.add_argument("--contrast", default="handVsRest"); p.add_argument("--z-thr", type=float, default=3.1)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    sub, ses = a.subject.removeprefix("sub-"), a.session.removeprefix("ses-")
    rec = {"subject": sub, "session": ses, "task": a.task}
    rec["grid"] = check_grid(a.tree, a.backfill, sub, ses, a.task)
    rec["confounds"] = check_confounds(a.tree, a.backfill, sub, ses)
    if a.log: rec["reuse"] = check_reuse(a.log)
    if a.glm_dir: rec["maps"] = check_maps(a.tree, a.backfill, a.glm_dir, sub, ses, a.task, a.contrast, a.z_thr)
    a.out.write_text(json.dumps(rec, indent=2))
    for k in ("grid", "reuse", "confounds", "maps"):
        if k in rec:
            v = rec[k]
            flag = "OK " if v.get("ok", True) else "!! "
            brief = {kk: vv for kk, vv in v.items() if kk not in ("runs", "t1w")}
            print(f"{flag}{k}: {json.dumps(brief)}")
            for r in v.get("runs", []): print("     ", json.dumps(r))
    print("record:", a.out)


if __name__ == "__main__":
    sys.exit(main())
