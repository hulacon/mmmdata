#!/usr/bin/env python3
"""
localizer_ceiling.py — within-subject split-half reliability of localizer
contrast maps on fsaverage6, per ROI: the ceiling for a map-prediction metric.

Reads the split-half maps written by ``localizer_splithalf.py`` (one GIfTI
data array per split and half, order in the sidecar) and, for every subject x
task x contrast x stat x ROI, correlates the two halves of each split across
the ROI's vertices (both hemispheres pooled, NaN vertices dropped). Reports
the mean, min and max over splits of that half-data correlation, and the
Spearman-Brown projection to full data, 2r / (1 + r), since each half holds
half the runs.

ROIs:
* ``schaefer7n``: the 7 networks of Schaefer 2018 400-parcel 7-network
  (``derivatives/atlases/tpl-fsaverage``, den-41k), grouped by the network
  field of the current CBIG parcel names;
* ``familyB``: the Harvard-Oxford cortical ROIs of the stimulus-space pilot
  (HOCPA th25 on fsaverage6; label values as in the MNI volume);
* ``cortex``: every labelled Schaefer vertex.

``--derived`` adds between-condition contrasts built as linear combinations
of the per-run *effect* maps (``DERIVED``), for tasks whose model contrasts
all share one baseline. Each run is one half, so a 2-run task gives one
1-vs-1 split. Effect maps only: a derived t map would need the covariance
between the source contrasts, which the per-run outputs do not carry. Their
contrast names end in ``Derived``, and ``--append`` adds rows to an existing
table instead of replacing it.

``--prf`` scores the session split-half pRF fits written by
``prf_splithalf.sbatch`` (``derivatives/functional_space/prf_splithalf``):
polar angle by circular correlation (Jammalamadaka & SenGupta), eccentricity
and size by Pearson, on vertices where the POOLED fit's R2 exceeds each floor
in ``--prf-r2-floors`` and its eccentricity lies within the stimulus radius
(the product's ``StimulusRadiusDeg``; beyond it fitted centres are
unbounded extrapolations). Selecting on the pooled R2 uses data that contain
both halves; that is the usual convention and is recorded in ``desc``.
Spearman-Brown is exact for Pearson and an approximation for the circular
correlation.

Polar angle is scored WITHIN each hemisphere and the two coefficients are
averaged, weighted by vertex count. Pooled across hemispheres the angles are
bimodal (each hemisphere maps the contralateral hemifield), the circular
mean is ill-defined, and the circular correlation can come out near zero or
negative for halves that agree to within a few degrees. ``angleCosDiff`` adds
the mean cos(angle_a - angle_b), an agreement index that needs no circular
mean (1 = identical, 0 = unrelated); it is not a correlation and has no
Spearman-Brown value.

This is within-subject reliability only; nothing is compared across subjects.

Usage:
    python localizer_ceiling.py --subjects sub-## sub-## --tasks floc motor --out <table.tsv>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src" / "python") not in sys.path:
    sys.path.insert(0, str(_REPO / "src" / "python"))

HEMIS = ("L", "R")
SPACE = "fsaverage6"
#: Harvard-Oxford cortical label values (HOCPA th25) for the pilot's cortical
#: family-B ROIs; the same values as scripts/pattern_similarity/shared.py.
FAMILY_B = {"EVC": 24, "EAC": 45, "AG": 21, "Precuneus": 31, "mPFC": 25}
#: task -> derived contrast -> weights on the model's per-run contrasts. The
#: motor model's contrasts are all ``<effector>VsRest`` with rest modelled, so
#: differences cancel rest exactly and leave between-effector contrasts.
DERIVED = {
    "motor": {
        "handVsFootDerived": {"handVsRest": 1, "footVsRest": -1},
        "handVsMouthDerived": {"handVsRest": 1, "mouthVsRest": -1},
        "footVsMouthDerived": {"footVsRest": 1, "mouthVsRest": -1},
        "saccadeVsOthersDerived": {"saccadeVsRest": 1, "handVsRest": -1 / 3,
                                   "footVsRest": -1 / 3, "mouthVsRest": -1 / 3},
    },
}


def _config_derivatives() -> Path:
    from core.config import load_config

    cfg = load_config()
    bids_root = Path(cfg["paths"]["bids_project_dir"])
    return Path(cfg["paths"].get("output_dir", bids_root / "derivatives"))


def _label_gii(path: Path) -> np.ndarray:
    import nibabel as nib

    return np.asarray(nib.load(str(path)).darrays[0].data).astype(int).ravel()


def load_rois(atlases: Path) -> dict[tuple[str, str], np.ndarray]:
    """(family, roi) -> boolean over the L+R concatenated fsaverage6 vertices."""
    anat = atlases / "tpl-fsaverage" / "anat"
    stem = "den-41k_atlas-Schaefer2018_seg-7n_scale-400_dseg"
    sch = np.concatenate([_label_gii(anat / f"tpl-fsaverage_hemi-{h}_{stem}.label.gii") for h in HEMIS])
    names = pd.read_csv(anat / f"tpl-fsaverage_{stem}.tsv", sep="\t")
    network_of = {int(i): n.split("_")[2] for i, n in zip(names["index"], names["name"])}
    rois: dict[tuple[str, str], np.ndarray] = {}
    for net in sorted(set(network_of.values())):
        idx = [i for i, n in network_of.items() if n == net]
        rois[("schaefer7n", net)] = np.isin(sch, idx)
    rois[("cortex", "all")] = sch > 0
    ho_stem = "den-41k_atlas-HOCPA_desc-th25_dseg"
    ho = np.concatenate([_label_gii(anat / f"tpl-fsaverage_hemi-{h}_{ho_stem}.label.gii") for h in HEMIS])
    for roi, val in FAMILY_B.items():
        rois[("familyB", roi)] = ho == val
    return rois


def load_split_maps(func_dir: Path, subject: str, task: str, contrast: str, stat: str, desc: str) -> np.ndarray:
    """(n_splits*2, n_vertices L+R) split-half maps."""
    import nibabel as nib

    per_hemi = []
    for h in HEMIS:
        p = func_dir / f"sub-{subject}_task-{task}_hemi-{h}_space-{SPACE}_contrast-{contrast}_stat-{stat}_desc-{desc}_statmap.func.gii"
        per_hemi.append(np.stack([np.asarray(d.data, dtype=np.float64).ravel() for d in nib.load(str(p)).darrays]))
    return np.concatenate(per_hemi, axis=1)


def ceiling_rows(tree: Path, subject: str, task: str, rois: dict) -> list[dict]:
    func_dir = tree / f"sub-{subject}" / "func"
    sidecars = sorted(func_dir.glob(f"sub-{subject}_task-{task}_space-{SPACE}_desc-*Splithalf_statmap.json"))
    if len(sidecars) != 1:
        sys.exit(f"ERROR: expected one split-half sidecar for sub-{subject} task-{task} in {func_dir}, "
                 f"found {len(sidecars)}")
    meta = json.loads(sidecars[0].read_text())
    desc = meta["split_desc"]
    contrasts = sorted({p.name.split("_contrast-")[1].split("_")[0]
                        for p in func_dir.glob(f"*task-{task}_hemi-L_*desc-{desc}_statmap.func.gii")})
    rows = []
    for contrast in contrasts:
        for stat in ("t", "effect"):
            maps = load_split_maps(func_dir, subject, task, contrast, stat, desc)
            n_splits = maps.shape[0] // 2
            for (family, roi), sel in rois.items():
                rs = []
                for k in range(n_splits):
                    a, b = maps[2 * k, sel], maps[2 * k + 1, sel]
                    ok = np.isfinite(a) & np.isfinite(b)
                    rs.append(np.corrcoef(a[ok], b[ok])[0, 1] if ok.sum() > 2 else np.nan)
                rs = np.asarray(rs)
                r = float(np.nanmean(rs))
                rows.append({
                    "subject": f"sub-{subject}", "task": task, "contrast": contrast, "stat": stat,
                    "family": family, "roi": roi, "n_vertices": int(sel.sum()),
                    "n_vertices_finite": int((np.isfinite(maps[0, sel]) & np.isfinite(maps[1, sel])).sum()),
                    "n_splits": n_splits, "r_half_mean": r, "r_half_min": float(np.nanmin(rs)),
                    "r_half_max": float(np.nanmax(rs)), "r_full_sb": 2 * r / (1 + r),
                    "desc": desc,
                })
    return rows


def _run_effect(tree: Path, subject: str, task: str, session: str, run: str, contrast: str, desc: str) -> np.ndarray:
    import nibabel as nib

    d = tree / f"sub-{subject}" / f"ses-{session}" / "func"
    return np.concatenate([
        np.asarray(nib.load(str(d / f"sub-{subject}_ses-{session}_task-{task}_run-{run}_hemi-{h}_space-{SPACE}"
                                    f"_contrast-{contrast}_stat-effect_desc-{desc}_statmap.func.gii")).darrays[0].data,
                   dtype=np.float64).ravel()
        for h in HEMIS])


def derived_rows(tree: Path, subject: str, task: str, rois: dict) -> list[dict]:
    """Ceiling rows for ``DERIVED[task]``, from per-run effect maps, one split per run pair."""
    func_dir = tree / f"sub-{subject}" / "func"
    meta = json.loads(next(func_dir.glob(f"sub-{subject}_task-{task}_space-{SPACE}_desc-*Splithalf_statmap.json")).read_text())
    desc = meta["desc"]
    runs = []
    for prefix in meta["runs"]:
        ent = dict(part.split("-", 1) for part in prefix.split("_"))
        runs.append((ent["ses"], ent["run"]))
    rows = []
    for name, weights in DERIVED[task].items():
        per_run = {k: sum(w * _run_effect(tree, subject, task, ses, run, c, desc) for c, w in weights.items())
                   for k, (ses, run) in enumerate(runs)}
        splits = [([runs.index(_sr(x)) for x in sp["half_a"]], [runs.index(_sr(x)) for x in sp["half_b"]])
                  for sp in meta["splits"]]
        for (family, roi), sel in rois.items():
            rs = []
            for a_idx, b_idx in splits:
                a = np.mean([per_run[i] for i in a_idx], axis=0)[sel]
                b = np.mean([per_run[i] for i in b_idx], axis=0)[sel]
                ok = np.isfinite(a) & np.isfinite(b)
                rs.append(np.corrcoef(a[ok], b[ok])[0, 1] if ok.sum() > 2 else np.nan)
            rs = np.asarray(rs)
            r = float(np.nanmean(rs))
            rows.append({
                "subject": f"sub-{subject}", "task": task, "contrast": name, "stat": "effect",
                "family": family, "roi": roi, "n_vertices": int(sel.sum()),
                "n_vertices_finite": int(np.isfinite(per_run[0][sel]).sum()),
                "n_splits": len(splits), "r_half_mean": r, "r_half_min": float(np.nanmin(rs)),
                "r_half_max": float(np.nanmax(rs)), "r_full_sb": 2 * r / (1 + r),
                "desc": desc + "Derived",
            })
    return rows


def _sr(prefix: str) -> tuple[str, str]:
    ent = dict(part.split("-", 1) for part in prefix.split("_"))
    return ent["ses"], ent["run"]


PRF_PARAMS = ("angle", "eccentricity", "size")


def _prf_map(d: Path, entities: str, param: str) -> np.ndarray:
    import nibabel as nib

    return np.concatenate([
        np.asarray(nib.load(str(d / f"{entities}_task-prf_space-{SPACE}_hemi-{h}_desc-{param}_prf.shape.gii")
                            ).darrays[0].data, dtype=np.float64).ravel()
        for h in HEMIS])


def circ_corr(a_deg: np.ndarray, b_deg: np.ndarray) -> float:
    """Circular correlation (Jammalamadaka & SenGupta 2001) of two angle samples in degrees."""
    a, b = np.radians(a_deg), np.radians(b_deg)
    sa = np.sin(a - np.arctan2(np.sin(a).mean(), np.cos(a).mean()))
    sb = np.sin(b - np.arctan2(np.sin(b).mean(), np.cos(b).mean()))
    return float((sa * sb).sum() / np.sqrt((sa ** 2).sum() * (sb ** 2).sum()))


def prf_rows(derivatives: Path, tree: str, subject: str, rois: dict, floors: list[float]) -> list[dict]:
    root = derivatives / tree / f"sub-{subject}"
    halves = sorted(p.name for p in root.glob("ses-*") if p.is_dir())
    if len(halves) != 2:
        sys.exit(f"ERROR: expected two session halves under {root}, found {halves}")
    pooled_sc = json.loads((derivatives / "prf" / f"sub-{subject}" /
                            f"sub-{subject}_task-prf_space-T1w_prf.json").read_text())
    radius = float(pooled_sc["StimulusRadiusDeg"])
    r2 = _prf_map(root, f"sub-{subject}", "R2")
    ecc = _prf_map(root, f"sub-{subject}", "eccentricity")
    maps = {h: {p: _prf_map(root / h, f"sub-{subject}_{h}", p) for p in PRF_PARAMS} for h in halves}
    rows = []
    for floor in floors:
        keep = (r2 > floor) & (ecc <= radius)
        desc = f"prfSessionSplit_pooledR2gt{floor:g}_eccLe{radius:g}".replace(".", "p")
        for param in PRF_PARAMS:
            a_all, b_all = maps[halves[0]][param], maps[halves[1]][param]
            for (family, roi), sel in rois.items():
                m = sel & keep & np.isfinite(a_all) & np.isfinite(b_all)
                a, b = a_all[m], b_all[m]
                if m.sum() < 10:
                    r = np.nan
                elif param == "angle":
                    hemi = np.arange(m.size) >= m.size // 2
                    parts = [(circ_corr(a_all[m & (hemi == h)], b_all[m & (hemi == h)]), int((m & (hemi == h)).sum()))
                             for h in (False, True) if (m & (hemi == h)).sum() >= 10]
                    r = float(sum(c * n for c, n in parts) / sum(n for _, n in parts)) if parts else np.nan
                else:
                    r = float(np.corrcoef(a, b)[0, 1])
                rows.append({
                    "subject": f"sub-{subject}", "task": "prf", "contrast": param, "stat": "prfparam",
                    "family": family, "roi": roi, "n_vertices": int(sel.sum()),
                    "n_vertices_finite": int(m.sum()), "n_splits": 1, "r_half_mean": r,
                    "r_half_min": r, "r_half_max": r,
                    "r_full_sb": 2 * r / (1 + r) if np.isfinite(r) else np.nan, "desc": desc,
                })
                if param == "angle":
                    agree = float(np.mean(np.cos(np.radians(a - b)))) if m.sum() >= 10 else np.nan
                    rows.append({**rows[-1], "contrast": "angleCosDiff", "r_half_mean": agree,
                                 "r_half_min": agree, "r_half_max": agree, "r_full_sb": np.nan})
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subjects", nargs="+", required=True)
    p.add_argument("--tasks", nargs="+", default=["floc", "motor"])
    p.add_argument("--tree", default="functional_space/localizer_splithalf",
                   help="split-half tree, relative to derivatives/")
    p.add_argument("--out", type=Path, required=True, help="output TSV")
    p.add_argument("--derived", action="store_true", help="only the DERIVED contrasts of each task that has them")
    p.add_argument("--append", action="store_true", help="append rows to --out (refuses duplicate keys)")
    p.add_argument("--prf", action="store_true", help="score the session split-half pRF fits instead")
    p.add_argument("--prf-tree", default="functional_space/prf_splithalf")
    p.add_argument("--prf-r2-floors", type=float, nargs="+", default=[10.0, 2.5],
                   help="pooled-fit R2 floors in percent (default 10 2.5)")
    args = p.parse_args(argv)

    derivatives = _config_derivatives()
    rois = load_rois(derivatives / "atlases")
    rows = []
    for s in args.subjects:
        subject = s.split("-", 1)[1] if s.startswith("sub-") else s
        if args.prf:
            rows += prf_rows(derivatives, args.prf_tree, subject, rois, args.prf_r2_floors)
            continue
        for task in args.tasks:
            if args.derived:
                if task in DERIVED:
                    rows += derived_rows(derivatives / args.tree, subject, task, rois)
            else:
                rows += ceiling_rows(derivatives / args.tree, subject, task, rois)
    df = pd.DataFrame(rows)
    if args.append and args.out.exists():
        old = pd.read_csv(args.out, sep="\t")
        key = ["subject", "task", "contrast", "stat", "family", "roi", "desc"]
        clash = old.merge(df[key], on=key)
        if len(clash):
            sys.exit(f"ERROR: {len(clash)} rows already in {args.out}; refusing to duplicate them")
        df = pd.concat([old, df], ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, sep="\t", index=False, float_format="%.4f")
    print(f"wrote {len(df)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
