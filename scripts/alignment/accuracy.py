#!/usr/bin/env python3
"""Anatomical accuracy of one alignment arm for one subject: pial/CSF edge r and ribbon specificity.

  edge    pial/CSF edge r for every T1w boldref
  ribbon  ribbon specificity of one GLMsingle data-quality cell (T1w), over the signal
          parcels, which are fixed from arm A and passed to every other arm
  compare one arm against A: ribbon specificity gain (paired parcel bootstrap) and the
          ncsnr no-worse check, from both arms' ``ribbon`` counts and data-quality cells

The anatomy (ribbon, aseg, CSF probseg, Schaefer-400 on the BOLD grid) always
comes from the production fMRIPrep and data_quality trees; only the boldrefs
come from ``--bold-tree``, the arm's fMRIPrep output. Library and definitions:
src/python/neuroimaging/alignment_accuracy.py.

``--shift-check N`` also translates the first N runs' boldrefs by known
amounts along each world axis and records edge r at each shift; for ``ribbon``
it moves the anatomy against the fixed null-beating map instead (``shift_mm`` is
the map's shift relative to the anatomy), rebuilding the compartments from the
fine masks so that sub-voxel shifts count. A measure
that tracks the anatomy should peak near zero shift for a well-aligned arm; a
peak elsewhere is either a real offset or a bias of the measure, and is read
before any arm comparison.

Outputs under ``<out>/sub-##/``::

    sub-##_arm-<arm>_desc-pialedge_qc.tsv        one row per run: edge_r_pial, then one column per region
    sub-##_arm-<arm>_desc-pialedge_qc.json       provenance, compartment sizes, settings
    sub-##_arm-<arm>_desc-pialedgeshift_qc.tsv   (--shift-check) run x axis x shift -> edge r
    sub-##_desc-compartments_dseg.nii.gz         1 GM, 2 WM band, 3 other WM, 4 CSF side, 5 other outside
    sub-##_desc-pialband_mask.nii.gz
    sub-##_desc-compartments.png                 slices of the compartments over the template boldref
    sub-##_arm-<arm>_fit-<fit>_set-<set>_desc-ribbon_counts.tsv     (ribbon) per signal parcel: voxels, null-beating voxels
    sub-##_arm-<arm>_fit-<fit>_set-<set>_desc-ribbon_qc.json        (ribbon) specificity, bootstrap CI, settings
    sub-##_arm-<arm>_fit-<fit>_set-<set>_desc-ribbonshift_qc.tsv    (ribbon --shift-check)
    sub-##_arm-<arm>_fit-<fit>_set-<set>_desc-compare_qc.json       (compare: the arm against A)
    sub-##_fit-<fit>_set-<set>_desc-signalparcels.tsv               (ribbon, when computed from this cell)

Usage:
    python accuracy.py edge --sub ## --arm A --out <dir> [--bold-tree <fmriprep tree>] [--jobs 8] [--shift-check 6]
    python accuracy.py ribbon --sub ## --arm A --out <dir> --cell-tree <data_quality tree> --fit enc --set anchor \
        [--signal-parcels <tsv from arm A>] [--shift-check]
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src" / "python") not in sys.path:  # idempotent: tests import this module repeatedly
    sys.path.insert(0, str(REPO_ROOT / "src" / "python"))

from core.config import load_config  # noqa: E402
from neuroimaging import alignment_accuracy as aa  # noqa: E402
from neuroimaging import data_quality as dq  # noqa: E402
from neuroimaging import data_quality_glmsingle as dqgs  # noqa: E402
from neuroimaging import data_quality_voxelmaps as dqv  # noqa: E402

SCHAEFER_NAMES = "tpl-fsaverage/anat/tpl-fsaverage_den-41k_atlas-Schaefer2018_seg-17n_scale-400_dseg.tsv"
SHIFTS_MM = (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0)
AXES = ("x", "y", "z")
#: A signal parcel: more than this share of its floored voxels beat the shuffle null (TYPED).
SIGNAL_FRAC = 0.10
SIGNAL_BETA = "D"
SIGNAL_ATLAS = "Schaefer17n400"
#: Fewer signal parcels than this in arm A stops the comparison for that subject (see the design record).
MIN_SIGNAL_PARCELS = 20


def boldrefs(tree: Path, subject: str) -> list[Path]:
    hits = sorted(Path(tree).glob(f"sub-{subject}/ses-*/func/sub-{subject}_*_space-T1w_boldref.nii.gz"))
    if not hits:
        sys.exit(f"sub-{subject}: no space-T1w boldrefs under {tree}")
    return hits


def entities(p: Path) -> dict[str, str]:
    ent = dict(kv.split("-", 1) for kv in p.name.split("_") if "-" in kv)
    return {"ses": ent.get("ses", "n/a"), "task": ent.get("task", "n/a"), "run": ent.get("run", "n/a")}


def mask_for(boldref: Path) -> Path:
    m = Path(str(boldref).replace("_boldref.nii.gz", "_desc-brain_mask.nii.gz"))
    if not m.exists():
        raise FileNotFoundError(f"brain mask missing beside {boldref}")
    return m


def load_on_grid(boldref: Path, shape, affine, shift=None) -> np.ndarray:
    """Median-normalised boldref on the BOLD grid; ``shift`` (mm, world) translates it first."""
    import nibabel as nib

    img = nib.load(str(boldref))
    v = np.asarray(img.dataobj, dtype=np.float32)
    m = np.asarray(nib.load(str(mask_for(boldref))).dataobj) > 0
    v = v / dqv.run_scale(v, m)
    a = img.affine.copy()
    if shift is not None:
        a[:3, 3] += shift
    return dqv.onto_grid(v, np.ones(v.shape, bool), a, shape, affine)[0]


class Anatomy:
    """The subject's T1w BOLD grid, Schaefer-400 labels on it, the compartments and the CSF probseg."""

    def __init__(self, deriv: Path, sub: str):
        import nibabel as nib

        fmriprep = deriv / "fmriprep"
        seg_hits = sorted((deriv / "data_quality" / f"sub-{sub}" / "anat").glob(
            f"sub-{sub}_space-T1w_seg-Schaefer17n400_*dseg.nii.gz"))
        if len(seg_hits) != 1:
            sys.exit(f"sub-{sub}: expected one T1w Schaefer17n400 dseg under data_quality, found {seg_hits}")
        self.parcellation = seg_hits[0]
        seg_img = nib.load(str(seg_hits[0]))
        self.shape, self.affine = seg_img.shape, seg_img.affine
        self.labels = np.asarray(seg_img.dataobj).astype(np.int32)
        self.zooms = np.sqrt((self.affine[:3, :3] ** 2).sum(axis=0))

        sub_dir = fmriprep / f"sub-{sub}"
        self.files = {k: dqv._anat_one(sub_dir, sub, pat) for k, pat in
                      (("ribbon", "_desc-ribbon_mask.nii.gz"), ("aseg", "_desc-aseg_dseg.nii.gz"),
                       ("csf", "_label-CSF_probseg.nii.gz"))}
        rib = nib.load(str(self.files["ribbon"]))
        aseg = nib.load(str(self.files["aseg"]))
        self._fine = (np.asarray(rib.dataobj) > 0, rib.affine, np.asarray(aseg.dataobj).astype(np.int32), aseg.affine)
        grid = (self.shape, self.affine)
        self.comp = self.compartments()
        csf_img = nib.load(str(self.files["csf"]))
        csf_v = np.asarray(csf_img.dataobj, dtype=np.float32)
        self.csf = dqv.onto_grid(csf_v, np.ones(csf_v.shape, bool), csf_img.affine, *grid)[0]
        self.names = pd.read_csv(deriv / "atlases" / SCHAEFER_NAMES, sep="\t").set_index("index")["name"]
        print(f"sub-{sub}: compartments {self.comp.sizes()}", flush=True)

    def compartments(self, shift=None) -> aa.Compartments:
        """Compartments on the BOLD grid; ``shift`` = (axis, mm) moves the anatomy first."""
        return aa.anatomy_compartments(*self._fine, self.shape, self.affine, shift=shift)

    def provenance(self) -> dict:
        return {"anat": {k: str(v) for k, v in self.files.items()}, "parcellation": str(self.parcellation),
                "grid": {"shape": list(self.shape), "zooms": self.zooms.round(4).tolist()},
                "compartments": self.comp.sizes(),
                "settings": {"fraction_in": aa.FRACTION_IN, "supersample": aa.SUPERSAMPLE,
                             "wm_band_vox": aa.WM_BAND_VOX, "csf_side_vox": aa.CSF_SIDE_VOX,
                             "pial_band_vox": aa.PIAL_BAND_VOX, "edge_sigma_mm": aa.EDGE_SIGMA_MM,
                             "aseg_wm": aa.ASEG_WM, "aseg_outside": aa.ASEG_OUTSIDE}}

    def write_maps(self, out: Path, sub: str, refs: list[Path]) -> None:
        c = self.comp
        dseg = np.full(self.shape, 0, np.int16)
        dseg[c.outside] = 5
        dseg[c.csf_side] = 4
        dseg[c.wm] = 3
        dseg[c.wm_band] = 2
        dseg[c.gm] = 1
        dqv.save_volume(dseg, self.affine, out / f"sub-{sub}_desc-compartments_dseg.nii.gz")
        dqv.save_volume(c.pial_band.astype(np.int16), self.affine, out / f"sub-{sub}_desc-pialband_mask.nii.gz")
        template = dqv.nanmedian_stack([load_on_grid(p, self.shape, self.affine)
                                        for p in refs[:: max(1, len(refs) // 20)]])
        figure(template, dseg, c.pial_band, out / f"sub-{sub}_desc-compartments.png", f"sub-{sub}")


def _stamp(extra: dict) -> dict:
    return {"schema_version": aa.SCHEMA_VERSION, **extra, "code_version": dq.code_version(REPO_ROOT),
            "generated": _dt.datetime.now().isoformat(timespec="seconds")}


def cmd_edge(args, deriv: Path) -> None:
    sub = args.sub
    an = Anatomy(deriv, sub)
    bold_tree = args.bold_tree or deriv / "fmriprep"
    groups = {"all": sorted(int(i) for i in an.names.index), **aa.network_groups(an.names)}
    for c in args.component:
        groups[c] = aa.component_group(an.names, c)
    band_labels = aa.nearest_labels(an.labels, an.comp.pial_band)
    refs = boldrefs(bold_tree, sub)

    def one(p: Path) -> dict:
        v = load_on_grid(p, an.shape, an.affine)
        r = aa.region_edge_r(v, an.csf, an.comp.pial_band, an.zooms, band_labels, groups)
        return {**entities(p), "edge_r_pial": r.pop("all"), **{f"edge_r_{k}": x for k, x in r.items()}}

    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        table = pd.DataFrame(list(pool.map(one, refs)))
    out = args.out / f"sub-{sub}"
    out.mkdir(parents=True, exist_ok=True)
    stem = out / f"sub-{sub}_arm-{args.arm}"
    table.to_csv(f"{stem}_desc-pialedge_qc.tsv", sep="\t", index=False, na_rep="n/a", float_format="%.6g")
    print(f"sub-{sub}: {len(table)} runs, edge_r_pial median {table.edge_r_pial.median():.4f} "
          f"[{table.edge_r_pial.quantile(.05):.4f}, {table.edge_r_pial.quantile(.95):.4f}]", flush=True)

    if args.shift_check:
        def shifted(job):
            p, axis, d = job
            s = np.zeros(3)
            s[AXES.index(axis)] = d
            v = load_on_grid(p, an.shape, an.affine, shift=s)
            return {**entities(p), "axis": axis, "shift_mm": d,
                    "edge_r_pial": aa.pial_edge_r(v, an.csf, an.comp.pial_band, an.zooms)}
        jobs = [(p, ax, d) for p in refs[:args.shift_check] for ax in AXES for d in SHIFTS_MM]
        with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
            sh = pd.DataFrame(list(pool.map(shifted, jobs)))
        sh.to_csv(f"{stem}_desc-pialedgeshift_qc.tsv", sep="\t", index=False, float_format="%.6g")
        peaks = sh.loc[sh.groupby(["ses", "task", "run", "axis"])["edge_r_pial"].idxmax()]
        print("shift check, peak shift (mm) per run x axis:\n"
              + peaks.pivot_table(index=["ses", "task", "run"], columns="axis", values="shift_mm").to_string(),
              flush=True)

    an.write_maps(out, sub, refs)
    side = _stamp({"subject": sub, "arm": args.arm, "bold_tree": str(bold_tree), **an.provenance(),
                   "parcel_names": str(deriv / "atlases" / SCHAEFER_NAMES),
                   "regions": {k: len(v) for k, v in groups.items()},
                   "shift_check": {"runs": args.shift_check, "shifts_mm": SHIFTS_MM}})
    Path(f"{stem}_desc-pialedge_qc.json").write_text(json.dumps(side, indent=2))


def signal_parcels(parcels_tsv: Path, cond_set: str) -> pd.DataFrame:
    """Schaefer-400 parcels of a GLMsingle data-quality cell where > SIGNAL_FRAC of TYPED voxels beat the null."""
    t = pd.read_csv(parcels_tsv, sep="\t", na_values=["n/a"])
    col = f"{cond_set}_frac_exceed"
    if col not in t.columns:
        sys.exit(f"{parcels_tsv} has no {col}: this fit has no '{cond_set}' condition set")
    t = t[(t.atlas == SIGNAL_ATLAS) & (t.beta_type == SIGNAL_BETA)]
    return t.loc[t[col] > SIGNAL_FRAC, ["parcel", "name", col]].reset_index(drop=True)


def cmd_ribbon(args, deriv: Path) -> None:
    import nibabel as nib

    sub = args.sub
    an = Anatomy(deriv, sub)
    cell = dqgs.cell_paths(args.cell_tree, sub, args.fit, args.space)
    for k in ("maps", "sidecar", "parcels"):
        if not cell[k].exists():
            sys.exit(f"sub-{sub} {args.fit}: data-quality cell file missing: {cell[k]} "
                     f"(build it with tier1.py glmsingle --space {args.space} --tree-root {args.cell_tree})")
    vols = json.loads(cell["sidecar"].read_text())["volumes"]
    img = nib.load(str(cell["maps"]))
    if img.shape[:3] != tuple(an.shape) or not np.allclose(img.affine, an.affine, atol=1e-3):
        sys.exit(f"{cell['maps']} is not on the grid of {an.parcellation}")
    data = np.asarray(img.dataobj, dtype=np.float32)
    exceed = np.nan_to_num(data[..., vols.index(f"exceed_{args.set}_{SIGNAL_BETA}")]) > 0.5
    keep = np.nan_to_num(data[..., vols.index("floor")]) > 0.5
    del data

    out = args.out / f"sub-{sub}"
    out.mkdir(parents=True, exist_ok=True)
    tag = f"fit-{args.fit}_set-{args.set}"
    if args.signal_parcels:
        sig = pd.read_csv(args.signal_parcels, sep="\t")
    elif args.arm != "A":
        sys.exit("signal parcels are fixed from arm A: pass --signal-parcels <arm A's signalparcels.tsv>")
    else:
        sig = signal_parcels(cell["parcels"], args.set)
        sig.to_csv(out / f"sub-{sub}_{tag}_desc-signalparcels.tsv", sep="\t", index=False, float_format="%.4g")
    parcels = sorted(int(p) for p in sig["parcel"])
    print(f"sub-{sub} {args.fit}/{args.set}: {len(parcels)} signal parcels", flush=True)
    if len(parcels) < MIN_SIGNAL_PARCELS:
        print(f"  WARNING: fewer than {MIN_SIGNAL_PARCELS} signal parcels: stop and bring this subject "
              "to Ben before reading another arm.", flush=True)

    counts = aa.parcel_counts(exceed, keep, an.comp, an.labels, parcels)
    stem = out / f"sub-{sub}_arm-{args.arm}_{tag}"
    counts.to_csv(f"{stem}_desc-ribbon_counts.tsv", sep="\t", index=False)
    spec = aa.specificity(counts)
    boot = aa.bootstrap_specificity({args.arm: counts})[args.arm]
    ci = [float(np.nanpercentile(boot, 2.5)), float(np.nanpercentile(boot, 97.5))]
    print(f"  specificity {spec['specificity']:.4f} [{ci[0]:.4f}, {ci[1]:.4f}], rate ratio "
          f"{spec['rate_ratio']:.3f}, CSF-side share {spec['csf_share']:.4f}", flush=True)

    shift = None
    if args.shift_check:
        rows = []
        for ai, axis in enumerate(AXES):
            for d in SHIFTS_MM:
                # Moving the map +d against the anatomy = moving the anatomy -d. The anatomy moves
                # because its fine masks keep sub-voxel shifts; the thresholded map would snap them
                # to whole voxels.
                comp = an.comp if d == 0 else an.compartments(shift=(ai, -d))
                c = aa.parcel_counts(exceed, keep, comp, an.labels, parcels)
                rows.append({"axis": axis, "shift_mm": d, **aa.specificity(c)})
        shift = pd.DataFrame(rows)
        shift.to_csv(f"{stem}_desc-ribbonshift_qc.tsv", sep="\t", index=False, float_format="%.6g")
        peaks = shift.loc[shift.groupby("axis")["specificity"].idxmax(), ["axis", "shift_mm", "specificity"]]
        print("  shift check, peak per axis:\n" + peaks.to_string(index=False), flush=True)

    side = _stamp({"subject": sub, "arm": args.arm, "fit": args.fit, "set": args.set, "space": args.space,
                   "cell": str(cell["maps"]), "beta_type": SIGNAL_BETA,
                   "signal_parcels": {"source": str(args.signal_parcels or cell["parcels"]), "n": len(parcels),
                                      "rule": f"{SIGNAL_ATLAS} TYPE{SIGNAL_BETA} {args.set}_frac_exceed > {SIGNAL_FRAC}"},
                   **spec, "bootstrap": {"n": aa.N_BOOT, "seed": aa.RANDOM_SEED, "ci95": ci},
                   **an.provenance(),
                   "shift_check": None if shift is None else {"shifts_mm": SHIFTS_MM}})
    Path(f"{stem}_desc-ribbon_qc.json").write_text(json.dumps(side, indent=2))


def parcel_ncsnr(parcels_tsv: Path, cond_set: str) -> pd.Series:
    """Per-parcel median TYPED ncsnr (Schaefer-400) of one GLMsingle data-quality cell, indexed by parcel."""
    t = pd.read_csv(parcels_tsv, sep="\t", na_values=["n/a"])
    t = t[(t.atlas == SIGNAL_ATLAS) & (t.beta_type == SIGNAL_BETA)]
    return t.set_index("parcel")[f"{cond_set}_ncsnr_median"]


def cmd_compare(args, deriv: Path) -> None:
    """Arm vs A: ribbon specificity gain and ncsnr no-worse, from both arms' ``ribbon`` outputs and cells."""
    sub, tag = args.sub, f"fit-{args.fit}_set-{args.set}"
    d = args.out / f"sub-{sub}"
    counts = {}
    for arm in ("A", args.arm):
        f = d / f"sub-{sub}_arm-{arm}_{tag}_desc-ribbon_counts.tsv"
        if not f.exists():
            sys.exit(f"missing {f}: run `accuracy.py ribbon --arm {arm}` first")
        counts[arm] = pd.read_csv(f, sep="\t")
    ncsnr = {arm: parcel_ncsnr(dqgs.cell_paths(tree, sub, args.fit, args.space)["parcels"], args.set)
             for arm, tree in (("A", args.cell_tree_a), (args.arm, args.cell_tree))}
    res = aa.compare_arms(counts["A"], counts[args.arm], ncsnr["A"], ncsnr[args.arm])
    print(f"sub-{sub} {args.arm} vs A ({tag}, {res['n_parcels']} parcels): specificity "
          f"{res['specificity_A']:.4f} -> {res['specificity_B']:.4f}, gain {res['specificity_gain']:+.4f} "
          f"[{res['specificity_gain_ci95'][0]:+.4f}, {res['specificity_gain_ci95'][1]:+.4f}] "
          f"(needs >= {res['min_gain']:.3f}, CI > 0: {'pass' if res['specificity_passes'] else 'fail'}); "
          f"ncsnr median diff {res['ncsnr_median_diff']:+.4f} [{res['ncsnr_median_diff_ci95'][0]:+.4f}, "
          f"{res['ncsnr_median_diff_ci95'][1]:+.4f}] vs margin -{res['ncsnr_margin']:.4f} "
          f"({'no worse' if res['ncsnr_no_worse'] else 'WORSE'}); wins before visual review: {res['wins']}",
          flush=True)
    Path(d / f"sub-{sub}_arm-{args.arm}_{tag}_desc-compare_qc.json").write_text(json.dumps(_stamp({
        "subject": sub, "arm": args.arm, "against": "A", "fit": args.fit, "set": args.set, "space": args.space,
        "cells": {"A": str(args.cell_tree_a), args.arm: str(args.cell_tree)}, **res,
        "bootstrap": {"n": aa.N_BOOT, "seed": aa.RANDOM_SEED}}), indent=2))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    for name in ("edge", "ribbon", "compare"):
        p = sp.add_parser(name)
        p.add_argument("--sub", required=True, help="bare label, e.g. 04")
        p.add_argument("--arm", required=True, help="arm label written into the outputs, e.g. A")
        p.add_argument("--out", required=True, type=Path, help="output root (one sub-## directory per subject)")
        if name == "edge":
            p.add_argument("--bold-tree", type=Path, help="fMRIPrep tree holding the arm's boldrefs (default: production)")
            p.add_argument("--component", action="append", default=["PFCd"],
                           help="extra region: Schaefer component name, e.g. PFCd (repeatable)")
            p.add_argument("--jobs", type=int, default=1)
            p.add_argument("--shift-check", type=int, default=0, metavar="N", help="shift-check the first N runs")
        else:
            p.add_argument("--cell-tree", required=True, type=Path,
                           help="data_quality tree holding the arm's GLMsingle cell")
        if name == "compare":
            p.add_argument("--cell-tree-a", required=True, type=Path, help="data_quality tree holding arm A's cell")
            p.add_argument("--fit", required=True, choices=dqgs.ARMS, help="GLMsingle arm, e.g. enc")
            p.add_argument("--set", required=True, choices=("anchor", "repeat"), help="condition set")
            p.add_argument("--space", default="T1w")
        elif name == "ribbon":
            p.add_argument("--fit", required=True, choices=dqgs.ARMS, help="GLMsingle arm, e.g. enc")
            p.add_argument("--set", required=True, choices=("anchor", "repeat"), help="condition set")
            p.add_argument("--space", default="T1w")
            p.add_argument("--signal-parcels", type=Path, help="arm A's signalparcels.tsv (required for other arms)")
            p.add_argument("--shift-check", action="store_true")
    args = ap.parse_args(argv)
    deriv = Path(load_config()["paths"]["output_dir"])
    t0 = time.time()
    {"edge": cmd_edge, "ribbon": cmd_ribbon, "compare": cmd_compare}[args.cmd](args, deriv)
    print(f"sub-{args.sub}: {time.time() - t0:.0f} s", flush=True)


def figure(template: np.ndarray, dseg: np.ndarray, band: np.ndarray, path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    cmap = ListedColormap(["none", "#e4572e", "#29335c", "none", "#17bebb", "none"])
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for col, ax_i in enumerate((0, 1, 2)):
        k = int(np.argmax((dseg == 1).sum(axis=tuple(i for i in range(3) if i != ax_i))))
        take = lambda a: np.rot90(np.take(a, k, axis=ax_i))  # noqa: E731
        for row in range(2):
            ax = axes[row, col]
            ax.imshow(take(np.nan_to_num(template)), cmap="gray")
            if row == 0:
                ax.imshow(np.ma.masked_equal(take(dseg), 0), cmap=cmap, vmin=-0.5, vmax=5.5, alpha=0.45,
                          interpolation="nearest")
            else:
                ax.imshow(np.ma.masked_equal(take(band.astype(int)), 0), cmap="autumn", alpha=0.6,
                          interpolation="nearest")
            ax.set_axis_off()
    axes[0, 0].set_title(f"{title}: GM (red), WM band (navy), CSF side (teal)", loc="left", fontsize=10)
    axes[1, 0].set_title("pial band", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
