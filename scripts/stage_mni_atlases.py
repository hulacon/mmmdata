#!/usr/bin/env python3
"""Stage Schaefer2018 and Harvard-Oxford from their current upstream releases.

Neither atlas is published in MNI152NLin2009cAsym (the space of the dataset's
MNI BOLD). Each is published in FSL's MNI152, which is TemplateFlow's
MNI152NLin6Asym:

* Schaefer2018 -- CBIG, ``Parcellations/MNI`` (labels as of v0.14.3, which
  renamed parcels and, for 17n >= 500 and 7n 900, reordered them).
* Harvard-Oxford -- FSL ``data_atlases``.

So this writes two things:

1. ``tpl-MNI152NLin6Asym/anat/`` -- the upstream volumes **byte for byte**,
   under BIDS names, with label tables transcribed from the upstream
   lookup tables (CBIG freeview LUT; FSL atlas XML).
2. ``tpl-MNI152NLin2009cAsym/anat/`` -- the same atlases carried onto the
   res-2 grid of the dataset's MNI BOLD by ONE transform: TemplateFlow's
   ``from-MNI152NLin6Asym`` xfm, the one fMRIPrep itself uses.

   * Schaefer: the 1 mm label volume, ``antsApplyTransforms -n GenericLabel``.
   * Harvard-Oxford: the 1 mm probability maps, linear interpolation, then
     FSL's own maxprob rule (label = argmax + 1 where max >= threshold). At
     1 mm in NLin6 that rule reproduces FSL's ``maxprob-thr25`` with zero
     voxels different, so the staged file is FSL's definition applied in the
     target space rather than a resampled label image.

Label indices, names and colours are upstream's exactly. Every sidecar names
the pinned source URL and md5 of each input.

Compute nodes have no DNS, so this runs in two steps::

    python scripts/stage_mni_atlases.py fetch --cache DIR        # login node
    module load ants/2.5.2
    python scripts/stage_mni_atlases.py build --cache DIR --out DIR [--overwrite]

``--out`` defaults to nothing: point it at a staging directory, check the
result, then at ``derivatives/atlases`` (from config) to go live.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import nibabel as nib
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src" / "python") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src" / "python"))
from core.config import load_config  # noqa: E402

SRC_TPL = "MNI152NLin6Asym"
TGT_TPL = "MNI152NLin2009cAsym"

# ── pinned sources ───────────────────────────────────────────────────────
# CBIG and FSL URLs are pinned to a commit, so their bytes cannot change;
# the md5 of each download is recorded in the sidecars. TemplateFlow's S3
# objects are not versioned, so those carry an md5 that must match.

CBIG_COMMIT = "35b5664bec8822e2f77da5e090e96f91d0095be6"  # master, 2026-08-31
_CBIG = (f"https://raw.githubusercontent.com/ThomasYeoLab/CBIG/{CBIG_COMMIT}/stable_projects/"
         "brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/MNI")
FSL_ATLASES_TAG = "2103.0"
FSL_ATLASES_COMMIT = "b3ad6133f723052d8295c48c68bbc8ab05961874"
_FSL = f"https://git.fmrib.ox.ac.uk/fsl/data_atlases/-/raw/{FSL_ATLASES_COMMIT}"
_TF = f"https://templateflow.s3.amazonaws.com/tpl-{TGT_TPL}"

SCALES = (100, 200, 300, 400, 500, 600, 700, 800, 900, 1000)
NETWORKS = (7, 17)
HO = {  # atlas entity -> (FSL file stem, XML, human name)
    "HOCPA": ("cort", "HarvardOxford-Cortical.xml", "Harvard-Oxford cortical structural atlas"),
    "HOSPA": ("sub", "HarvardOxford-Subcortical.xml", "Harvard-Oxford subcortical structural atlas"),
}
HO_THRESHOLDS = (25,)

TF_PINNED = {
    "xfm": (f"{_TF}/tpl-{TGT_TPL}_from-{SRC_TPL}_mode-image_xfm.h5", "1f9b29f2d5cb898cc97243fe92ee9304"),
    "ref_res2": (f"{_TF}/tpl-{TGT_TPL}_res-02_T1w.nii.gz", "04e6e6eae189c060ba17478e097c6a24"),
}


def sources() -> dict[str, str]:
    """key -> URL for every input."""
    s = {k: u for k, (u, _) in TF_PINNED.items()}
    for n in NETWORKS:
        for k in SCALES:
            stem = f"Schaefer2018_{k}Parcels_{n}Networks_order"
            s[f"schaefer_{n}n_{k}_lut"] = f"{_CBIG}/freeview_lut/{stem}.txt"
            for mm in (1, 2):
                s[f"schaefer_{n}n_{k}_{mm}mm"] = f"{_CBIG}/{stem}_FSLMNI152_{mm}mm.nii.gz"
    for atlas, (stem, xml, _) in HO.items():
        s[f"{atlas}_xml"] = f"{_FSL}/{xml}"
        s[f"{atlas}_prob_1mm"] = f"{_FSL}/HarvardOxford/HarvardOxford-{stem}-prob-1mm.nii.gz"
        for thr in HO_THRESHOLDS:
            for mm in (1, 2):
                s[f"{atlas}_th{thr}_{mm}mm"] = f"{_FSL}/HarvardOxford/HarvardOxford-{stem}-maxprob-thr{thr}-{mm}mm.nii.gz"
    return s


def cached(cache: Path, key: str) -> Path:
    url = sources()[key]
    return cache / f"{key}{''.join(Path(url.rsplit('/', 1)[-1]).suffixes)}"


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def provenance(cache: Path, *keys: str) -> list[dict]:
    return [{"key": k, "url": sources()[k], "md5": md5(cached(cache, k))} for k in keys]


# ── fetch ────────────────────────────────────────────────────────────────


def fetch(cache: Path) -> int:
    cache.mkdir(parents=True, exist_ok=True)
    for key, url in sources().items():
        path = cached(cache, key)
        for attempt in range(5):  # Talapas DNS drops lookups intermittently
            if path.exists():
                break
            try:
                tmp = path.with_suffix(path.suffix + ".part")
                urllib.request.urlretrieve(url, tmp)
                tmp.rename(path)
            except OSError as err:
                print(f"retry {attempt + 1}: {key}: {err}", file=sys.stderr)
        if not path.exists():
            raise RuntimeError(f"could not fetch {key} ({url})")
        if key in TF_PINNED and md5(path) != TF_PINNED[key][1]:
            raise RuntimeError(f"{key}: md5 {md5(path)} != pinned {TF_PINNED[key][1]}")
    print(f"{len(sources())} sources in {cache}")
    return 0


# ── tables ───────────────────────────────────────────────────────────────


def schaefer_table(lut: Path) -> list[dict]:
    """CBIG freeview LUT -> rows of index, name, color (hex)."""
    rows = []
    for line in lut.read_text().splitlines():
        if not line.strip():
            continue
        idx, name, r, g, b = line.split("\t")[:5]
        rows.append({"index": int(idx), "name": name, "color": "#%02x%02x%02x" % (int(r), int(g), int(b))})
    return rows


def ho_table(xml: Path) -> list[dict]:
    """FSL atlas XML (0-based label index) -> rows of volume index (1-based), name."""
    labels = ET.parse(xml).getroot().find("data").findall("label")
    return [{"index": int(lab.get("index")) + 1, "name": lab.text} for lab in labels]


def write_tsv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, list(rows[0]), delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2) + "\n")


def name(tpl: str, atlas: str, res: int | None = None, suffix: str = "dseg", ext: str = ".nii.gz", **ents) -> str:
    """The live tree's order, which consumers hardcode: atlas, seg/scale, res, desc
    (``..._atlas-Schaefer2018_seg-17n_scale-400_res-2_dseg``, ``..._atlas-HOSPA_res-2_desc-th25_dseg``)."""
    desc = ents.pop("desc", None)
    parts = [f"tpl-{tpl}", f"atlas-{atlas}"] + [f"{k}-{v}" for k, v in ents.items()]
    if res is not None:
        parts.append(f"res-{res}")
    if desc is not None:
        parts.append(f"desc-{desc}")
    return "_".join(parts) + f"_{suffix}{ext}"


# ── build ────────────────────────────────────────────────────────────────


def ants(src: Path, ref: Path, xfm: Path, out: Path, interp: str, image_type: int = 0) -> None:
    cmd = ["antsApplyTransforms", "-d", "3", "-e", str(image_type), "-i", str(src), "-r", str(ref),
           "-t", str(xfm), "-n", interp, "-o", str(out)]
    subprocess.run(cmd, check=True)


def as_labels(img: nib.Nifti1Image, data: np.ndarray, dtype=np.int16) -> nib.Nifti1Image:
    out = nib.Nifti1Image(data.astype(dtype), img.affine)
    out.header.set_xyzt_units("mm")
    out.set_qform(img.affine, code=4)  # 4 = MNI152
    out.set_sform(img.affine, code=4)
    return out


def build_schaefer(cache: Path, src_dir: Path, tgt_dir: Path, ref: Path, xfm: Path, tmp: Path) -> list[dict]:
    report = []
    for n in NETWORKS:
        for k in SCALES:
            ents = {"seg": f"{n}n", "scale": k}
            key = f"schaefer_{n}n_{k}"
            rows = schaefer_table(cached(cache, f"{key}_lut"))
            if [r["index"] for r in rows] != list(range(1, k + 1)):
                raise RuntimeError(f"{key}: LUT indices are not 1..{k}")
            common = {
                "Name": f"Schaefer 2018, {k} parcels, {n}-network order",
                "Atlas": "Schaefer2018",
                "Version": f"CBIG {CBIG_COMMIT[:7]} (labels as of release v0.14.3, 2019-09-16)",
                "ReferencesAndLinks": ["https://doi.org/10.1093/cercor/bhx179",
                                       "https://github.com/ThomasYeoLab/CBIG/tree/"
                                       f"{CBIG_COMMIT}/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal"],
                "GeneratedBy": {"Name": Path(__file__).name},
            }
            # 1. upstream, byte for byte
            for mm in (1, 2):
                src = cached(cache, f"{key}_{mm}mm")
                data = np.asarray(nib.load(src).dataobj).astype(int)
                stray = set(np.unique(data)) - set(range(k + 1))
                if stray:
                    raise RuntimeError(f"{src.name}: labels outside 0..{k}: {sorted(stray)[:5]}")
                shutil.copyfile(src, src_dir / name(SRC_TPL, "Schaefer2018", mm, **ents))
                write_tsv(src_dir / name(SRC_TPL, "Schaefer2018", mm, ext=".tsv", **ents), rows)
                write_json(src_dir / name(SRC_TPL, "Schaefer2018", mm, ext=".json", **ents), {
                    **common,
                    "Description": "CBIG's FSLMNI152 volume, byte-identical to upstream; label table "
                                   "transcribed from CBIG's freeview LUT.",
                    "Sources": provenance(cache, f"{key}_{mm}mm", f"{key}_lut"),
                })
            # 2. onto the dataset's MNI152NLin2009cAsym res-2 grid
            warped = tmp / f"{key}_res2.nii.gz"
            ants(cached(cache, f"{key}_1mm"), ref, xfm, warped, "GenericLabel")
            wimg = nib.load(warped)
            wdata = np.rint(np.asarray(wimg.dataobj)).astype(int)
            counts = np.bincount(wdata.ravel(), minlength=k + 1)[1:]
            missing = [i + 1 for i in np.flatnonzero(counts == 0)]
            nib.save(as_labels(wimg, wdata), tgt_dir / name(TGT_TPL, "Schaefer2018", 2, **ents))
            write_tsv(tgt_dir / name(TGT_TPL, "Schaefer2018", 2, ext=".tsv", **ents), rows)
            write_json(tgt_dir / name(TGT_TPL, "Schaefer2018", 2, ext=".json", **ents), {
                **common,
                "Description": "CBIG's FSLMNI152 1 mm volume carried onto the MNI152NLin2009cAsym res-2 "
                               "grid by TemplateFlow's from-MNI152NLin6Asym transform "
                               "(antsApplyTransforms -n GenericLabel). Indices, names and colours are "
                               "CBIG's exactly; the tpl-MNI152NLin6Asym copy in this tree is the upstream file.",
                "SourceTemplate": SRC_TPL,
                "Transform": "antsApplyTransforms -d 3 -n GenericLabel -t <xfm>",
                "Sources": provenance(cache, f"{key}_1mm", f"{key}_lut", "xfm", "ref_res2"),
                "LabelsMissingAtRes2": missing,
            })
            report.append({"atlas": f"Schaefer2018 {n}n {k}", "missing_at_res2": missing,
                           "min_voxels_res2": int(counts[counts > 0].min())})
    return report


def build_ho(cache: Path, src_dir: Path, tgt_dir: Path, ref: Path, xfm: Path, tmp: Path) -> list[dict]:
    report = []
    for atlas, (_, _, human) in HO.items():
        rows = ho_table(cached(cache, f"{atlas}_xml"))
        n = len(rows)
        common = {
            "Name": human, "Atlas": atlas,
            "Version": f"FSL data_atlases {FSL_ATLASES_TAG} ({FSL_ATLASES_COMMIT[:8]})",
            "License": "FSL atlas licence (non-commercial); see FSL distribution",
            "ReferencesAndLinks": ["https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/Atlases",
                                   f"https://git.fmrib.ox.ac.uk/fsl/data_atlases/-/tree/{FSL_ATLASES_TAG}"],
            "GeneratedBy": {"Name": Path(__file__).name},
        }
        prob_src = cached(cache, f"{atlas}_prob_1mm")
        P = np.asarray(nib.load(prob_src).dataobj)
        if P.shape[3] != n:
            raise RuntimeError(f"{atlas}: {P.shape[3]} prob maps vs {n} XML labels")
        # 1. upstream, byte for byte; check FSL's maxprob rule on its own files
        for thr in HO_THRESHOLDS:
            for mm in (1, 2):
                src = cached(cache, f"{atlas}_th{thr}_{mm}mm")
                shutil.copyfile(src, src_dir / name(SRC_TPL, atlas, mm, desc=f"th{thr}"))
                write_tsv(src_dir / name(SRC_TPL, atlas, mm, ext=".tsv", desc=f"th{thr}"), rows)
                write_json(src_dir / name(SRC_TPL, atlas, mm, ext=".json", desc=f"th{thr}"), {
                    **common,
                    "Description": f"FSL maxprob-thr{thr}-{mm}mm, byte-identical to upstream; label table "
                                   "transcribed from FSL's atlas XML (volume value = XML index + 1).",
                    "Sources": provenance(cache, f"{atlas}_th{thr}_{mm}mm", f"{atlas}_xml"),
                })
            M = np.asarray(nib.load(cached(cache, f"{atlas}_th{thr}_1mm")).dataobj).astype(int)
            rule = np.where(P.max(3) >= thr, P.argmax(3) + 1, 0)
            if not np.array_equal(rule, M):
                raise RuntimeError(f"{atlas}: FSL's maxprob-thr{thr} is not argmax where max >= {thr} "
                                   f"({int((rule != M).sum())} voxels differ); the derivation below is invalid")
        shutil.copyfile(prob_src, src_dir / name(SRC_TPL, atlas, 1, suffix="probseg"))
        write_json(src_dir / name(SRC_TPL, atlas, 1, suffix="probseg", ext=".json"), {
            **common, "Description": "FSL prob-1mm, byte-identical to upstream (0-100, one volume per label).",
            "Sources": provenance(cache, f"{atlas}_prob_1mm", f"{atlas}_xml"),
        })
        write_tsv(src_dir / name(SRC_TPL, atlas, 1, suffix="probseg", ext=".tsv"), rows)
        # 2. prob maps onto the target grid, then FSL's rule
        warped = tmp / f"{atlas}_prob_res2.nii.gz"
        ants(prob_src, ref, xfm, warped, "Linear", image_type=3)
        wimg = nib.load(warped)
        W = np.asarray(wimg.dataobj, dtype=np.float32)
        nib.save(nib.Nifti1Image(W, wimg.affine), tgt_dir / name(TGT_TPL, atlas, 2, suffix="probseg"))
        write_tsv(tgt_dir / name(TGT_TPL, atlas, 2, suffix="probseg", ext=".tsv"), rows)
        write_json(tgt_dir / name(TGT_TPL, atlas, 2, suffix="probseg", ext=".json"), {
            **common, "Description": "FSL prob-1mm carried onto the MNI152NLin2009cAsym res-2 grid by "
                                     "TemplateFlow's from-MNI152NLin6Asym transform, linear interpolation.",
            "SourceTemplate": SRC_TPL, "Transform": "antsApplyTransforms -d 3 -e 3 -n Linear -t <xfm>",
            "Sources": provenance(cache, f"{atlas}_prob_1mm", f"{atlas}_xml", "xfm", "ref_res2"),
        })
        for thr in HO_THRESHOLDS:
            lab = np.where(W.max(3) >= thr, W.argmax(3) + 1, 0)
            counts = np.bincount(lab.ravel(), minlength=n + 1)[1:]
            missing = [i + 1 for i in np.flatnonzero(counts == 0)]
            nib.save(as_labels(wimg.slicer[..., 0], lab), tgt_dir / name(TGT_TPL, atlas, 2, desc=f"th{thr}"))
            write_tsv(tgt_dir / name(TGT_TPL, atlas, 2, ext=".tsv", desc=f"th{thr}"), rows)
            write_json(tgt_dir / name(TGT_TPL, atlas, 2, ext=".json", desc=f"th{thr}"), {
                **common,
                "Description": f"FSL's maxprob-thr{thr} rule (label = argmax + 1 where max >= {thr}) applied to "
                               "FSL prob-1mm after carrying it onto the MNI152NLin2009cAsym res-2 grid by "
                               "TemplateFlow's from-MNI152NLin6Asym transform (linear). The rule reproduces "
                               f"FSL's maxprob-thr{thr}-1mm exactly in the source space (checked at build).",
                "SourceTemplate": SRC_TPL,
                "Transform": "antsApplyTransforms -d 3 -e 3 -n Linear -t <xfm>; then maxprob rule",
                "Sources": provenance(cache, f"{atlas}_prob_1mm", f"{atlas}_xml", "xfm", "ref_res2"),
                "LabelsMissingAtRes2": missing,
            })
            report.append({"atlas": f"{atlas} th{thr}", "missing_at_res2": missing,
                           "min_voxels_res2": int(counts[counts > 0].min())})
    return report


def register(out: Path) -> None:
    """Atlas-level descriptions, and this script's line in dataset_description.json.

    Replaces the two GeneratedBy entries this script supersedes: the manual
    TemplateFlow install of the MNI Schaefer files, and stage_harvard_oxford.py.
    """
    write_json(out / "atlas-Schaefer2018_description.json", {
        "Name": "Schaefer 2018 Local-Global Parcellation of the Human Cerebral Cortex",
        "Description": ("Cortical parcellation from a gradient-weighted Markov random field model, 100-1000 "
                        "parcels in 7- and 17-network order. Labels as of CBIG release v0.14.3 "
                        "(2019-09-16; renamed parcels, reordered 17n >= 500 and 7n 900). Upstream files "
                        f"under tpl-{SRC_TPL}/ (CBIG's FSLMNI152) and tpl-fsaverage/; tpl-{TGT_TPL}/ is "
                        "derived from the former by TemplateFlow's transform, see each file's JSON sidecar."),
        "License": "MIT",
        "Authors": ["Alexander Schaefer", "Ru Kong", "Evan M. Gordon", "Timothy O. Laumann", "Xi-Nian Zuo",
                    "Avram J. Holmes", "Simon B. Eickhoff", "B. T. Thomas Yeo"],
        "ReferencesAndLinks": ["https://doi.org/10.1093/cercor/bhx179",
                               "https://github.com/ThomasYeoLab/CBIG/tree/"
                               f"{CBIG_COMMIT}/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal"],
        "Species": "homo sapiens",
        "DerivedFrom": "resting-state fMRI, 1489 subjects",
        "LevelType": "group",
        "SourceDistribution": f"CBIG {CBIG_COMMIT}",
    })
    write_json(out / "atlas-HarvardOxford_description.json", {
        "Name": "Harvard-Oxford cortical and subcortical structural atlases",
        "Description": ("Probabilistic atlases of 48 cortical and 21 subcortical structural areas from "
                        "segmentations by the Harvard Center for Morphometric Analysis. Upstream FSL files "
                        f"(maxprob-thr25 1/2 mm, prob 1 mm) under tpl-{SRC_TPL}/; tpl-{TGT_TPL}/ carries the "
                        "probability maps by TemplateFlow's transform and reapplies FSL's maxprob rule, see "
                        "each file's JSON sidecar. Label names are FSL's XML verbatim."),
        "License": "FSL atlas licence (non-commercial); see FSL distribution",
        "ReferencesAndLinks": ["https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/Atlases",
                               "https://doi.org/10.1016/j.schres.2005.11.020",
                               "https://doi.org/10.1016/j.biopsych.2006.06.027",
                               "https://doi.org/10.1016/j.neuroimage.2007.03.048"],
        "Species": "homo sapiens",
        "DerivedFrom": "T1w structural segmentations (37 subjects)",
        "LevelType": "group",
        "SourceDistribution": f"FSL data_atlases {FSL_ATLASES_TAG} ({FSL_ATLASES_COMMIT})",
    })
    path = out / "dataset_description.json"
    if not path.exists():
        return
    desc = json.loads(path.read_text())
    superseded = {"Manual", "stage_harvard_oxford.py", Path(__file__).name}
    gen = [g for g in desc.get("GeneratedBy", []) if g.get("Name") not in superseded]
    gen.insert(0, {
        "Name": Path(__file__).name,
        "Description": (f"Schaefer2018 (10 scales x 7n/17n) and Harvard-Oxford (HOCPA/HOSPA th25): upstream "
                        f"files byte-identical under tpl-{SRC_TPL}/, and on the {TGT_TPL} res-2 grid by "
                        "TemplateFlow's from-MNI152NLin6Asym transform (mmmdata scripts/stage_mni_atlases.py); "
                        "see each file's JSON sidecar"),
    })
    desc["GeneratedBy"] = gen
    write_json(path, desc)


def build(cache: Path, out: Path, overwrite: bool) -> int:
    if shutil.which("antsApplyTransforms") is None:
        raise RuntimeError("antsApplyTransforms not on PATH (module load ants/2.5.2)")
    missing = [k for k in sources() if not cached(cache, k).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} sources not in {cache} (e.g. {missing[0]}); run fetch first")
    src_dir, tgt_dir = out / f"tpl-{SRC_TPL}" / "anat", out / f"tpl-{TGT_TPL}" / "anat"
    ours = [p for d in (src_dir, tgt_dir) if d.exists() for p in d.glob("*atlas-Schaefer2018*")]
    ours += [p for d in (src_dir, tgt_dir) if d.exists() for p in d.glob("*atlas-HO[CS]PA*")]
    if ours and not overwrite:
        raise FileExistsError(f"{len(ours)} outputs exist (e.g. {ours[0]}); pass --overwrite")
    for d in (src_dir, tgt_dir):
        d.mkdir(parents=True, exist_ok=True)
    ref, xfm = cached(cache, "ref_res2"), cached(cache, "xfm")
    with tempfile.TemporaryDirectory() as tmp:
        report = build_schaefer(cache, src_dir, tgt_dir, ref, xfm, Path(tmp))
        report += build_ho(cache, src_dir, tgt_dir, ref, xfm, Path(tmp))
    register(out)
    print(json.dumps(report, indent=1))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--cache", type=Path, required=True)
    b = sub.add_parser("build")
    b.add_argument("--cache", type=Path, required=True)
    b.add_argument("--out", type=Path, required=True,
                   help="output root; the live tree is <output_dir>/atlases from config")
    b.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)
    if args.cmd == "fetch":
        return fetch(args.cache)
    live = Path(load_config(config_dir=_REPO_ROOT / "config")["paths"]["output_dir"]) / "atlases"
    if args.out.resolve() == live.resolve():
        print(f"writing into the live atlases tree {live}", file=sys.stderr)
    return build(args.cache, args.out, args.overwrite)


if __name__ == "__main__":
    raise SystemExit(main())
