#!/usr/bin/env python3
"""Stage cortical atlases on fsaverage6 (``den-41k``), plus HCP-MMP1 in MNI.

Writes into ``derivatives/atlases`` beside the MNI volumes staged earlier:

* **Schaefer 2018, 400 parcels, 7- and 17-network**: CBIG's own fsaverage6
  ``.annot`` files, converted to GIfTI label files.
* **HCP-MMP1 (Glasser 2016, 360 areas)**: the fsaverage ``.annot`` projection
  (Mills 2016, figshare), cut to fsaverage6 by taking the first 40,962 vertices.
  Also a 22-section grouping (Glasser 2016 supplementary table, via a lookup
  table) as its own label file.
* **Harvard-Oxford cortical maxprob-thr25**: the same FSL 2 mm volume that
  ``scripts/stage_harvard_oxford.py`` stages, projected to fsaverage with
  CBIG's registration-fusion mapping (Wu 2018, RF-ANTs, MNI152 -> fsaverage),
  sampled nearest-neighbour, then cut to fsaverage6.
* **HCP-MMP1 in MNI152NLin2009cAsym res-2**: the volumetric projection by
  Horn 2016 (figshare, MNI152 ICBM2009a nlin), split into hemispheres at the
  midline and resampled once, nearest-neighbour, onto the grid of the Schaefer
  ``dseg`` files already staged.

Label values. Surface parcellations use one index space across hemispheres,
the same one as the matching volume (Schaefer: LH 1-200, RH 201-400;
HCP-MMP1: LH 1-180, RH 181-360), so one ``dseg.tsv`` serves both hemispheres
and the volume. Harvard-Oxford is bilateral (1-48, as in the volume). The
HCP-MMP1 sections use 1-22 in each hemisphere; the hemisphere is the file.

Why fsaverage6 is a vertex subset of fsaverage. The fsaverage family is an
icosahedral hierarchy: fsaverage6's 40,962 vertices are the first 40,962 of
fsaverage's 163,842 at identical sphere coordinates. The script checks this
against TemplateFlow's spheres before relying on it, and cross-checks it on
CBIG's own fsaverage and fsaverage6 Schaefer files.

Every source is downloaded from a pinned URL and checked against a pinned
MD5; a mismatch is a hard failure.

Usage::

    python scripts/functional_space/stage_fsaverage6_atlases.py [--overwrite]
"""

from __future__ import annotations

import argparse
import colorsys
import hashlib
import json
import sys
import tempfile
import urllib.request
from pathlib import Path

import nibabel as nib
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "src" / "python") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src" / "python"))
from core.config import load_config  # noqa: E402

N_FS6 = 40962           # vertices per hemisphere, fsaverage6 (ico6)
N_FS7 = 163842          # vertices per hemisphere, fsaverage (ico7)
DEN = "41k"             # TemplateFlow's density label for fsaverage6
HEMIS = {"L": "lh", "R": "rh"}
SURF_DIR = "tpl-fsaverage/anat"
MNI = "MNI152NLin2009cAsym"
MNI_DIR = f"tpl-{MNI}/anat"
MIN_HO_COVERAGE = 0.85  # fraction of non-medial-wall vertices given a label

# ── pinned sources ───────────────────────────────────────────────────────

CBIG_COMMIT = "35b5664bec8822e2f77da5e090e96f91d0095be6"
_CBIG = f"https://raw.githubusercontent.com/ThomasYeoLab/CBIG/{CBIG_COMMIT}/stable_projects"
_SCHAEFER = f"{_CBIG}/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/FreeSurfer5.3"
_RF = f"{_CBIG}/registration/Wu2017_RegistrationFusion/bin/final_warps_FS5.3"
_TF = "https://templateflow.s3.amazonaws.com/tpl-fsaverage"
REGIONLIST_COMMIT = "1a414bc38e0bc49d324890e835446abc9131ca98"

# key -> (url, md5)
SOURCES = {
    "schaefer_fs6_lh_7": (f"{_SCHAEFER}/fsaverage6/label/lh.Schaefer2018_400Parcels_7Networks_order.annot", "f10b4f896ea61fc238e72a729a09e363"),
    "schaefer_fs6_rh_7": (f"{_SCHAEFER}/fsaverage6/label/rh.Schaefer2018_400Parcels_7Networks_order.annot", "a9f2840b95c5b6c7e29d1c4cc36ddcbc"),
    "schaefer_fs6_lh_17": (f"{_SCHAEFER}/fsaverage6/label/lh.Schaefer2018_400Parcels_17Networks_order.annot", "4c7e6cab9e27138a4d100250e007198c"),
    "schaefer_fs6_rh_17": (f"{_SCHAEFER}/fsaverage6/label/rh.Schaefer2018_400Parcels_17Networks_order.annot", "29f664977a59f7a1c315b1a0098fb69b"),
    "schaefer_fs7_lh_7": (f"{_SCHAEFER}/fsaverage/label/lh.Schaefer2018_400Parcels_7Networks_order.annot", "ec29d243d5745418fb5dcb0997e1765e"),
    "schaefer_fs7_rh_7": (f"{_SCHAEFER}/fsaverage/label/rh.Schaefer2018_400Parcels_7Networks_order.annot", "f0a415f7aa960e489d17a68f11807af0"),
    "schaefer_fs7_lh_17": (f"{_SCHAEFER}/fsaverage/label/lh.Schaefer2018_400Parcels_17Networks_order.annot", "845d4ec726e1238a9d017305d7cc8e7e"),
    "schaefer_fs7_rh_17": (f"{_SCHAEFER}/fsaverage/label/rh.Schaefer2018_400Parcels_17Networks_order.annot", "d01b71dda255c82cd73a67447b68edab"),
    "rf_lh": (f"{_RF}/lh.avgMapping_allSub_RF_ANTs_MNI152_orig_to_fsaverage.mat", "7f31cd9a05c5b644241685c0ad12afaf"),
    "rf_rh": (f"{_RF}/rh.avgMapping_allSub_RF_ANTs_MNI152_orig_to_fsaverage.mat", "9eb70e0e35ee8a9078029e605ef6a604"),
    "mmp_lh": ("https://ndownloader.figshare.com/files/5528816", "46a102b59b2fb1bb4bd62d51bf02e975"),
    "mmp_rh": ("https://ndownloader.figshare.com/files/5528819", "75e96b331940227bbcb07c1c791c2463"),
    "mmp_regionlist": (f"https://bitbucket.org/dpat/tools/raw/{REGIONLIST_COMMIT}/REF/ATLASES/HCP-MMP1_UniqueRegionList.csv", "62ed4338082215ed39fc3667f839af09"),
    "mmp_mni": ("https://ndownloader.figshare.com/files/5594363", "4a6a53f08e56413cddf56f9629a17bf1"),
    "sphere_L_164k": (f"{_TF}/tpl-fsaverage_hemi-L_den-164k_sphere.surf.gii", "576153844fb020fcf0865873b53f9085"),
    "sphere_R_164k": (f"{_TF}/tpl-fsaverage_hemi-R_den-164k_sphere.surf.gii", "7fdd2d156df4264b47cc706462d2c5be"),
    "sphere_L_41k": (f"{_TF}/tpl-fsaverage_hemi-L_den-41k_sphere.surf.gii", "2af0c9e2e5e5bc73f812f7a0f7c93483"),
    "sphere_R_41k": (f"{_TF}/tpl-fsaverage_hemi-R_den-41k_sphere.surf.gii", "dd84f143c1993446954dbf1ac41ef92f"),
}

EXT = {"mmp_lh": ".annot", "mmp_rh": ".annot", "mmp_mni": ".nii.gz"}

REFS = {
    "Schaefer2018": ["https://doi.org/10.1093/cercor/bhx179",
                     f"https://github.com/ThomasYeoLab/CBIG/tree/{CBIG_COMMIT}/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal"],
    "HCPMMP1": ["https://doi.org/10.1038/nature18933",
                "https://doi.org/10.6084/m9.figshare.3498446.v2",
                "https://doi.org/10.6084/m9.figshare.3501911.v5",
                f"https://bitbucket.org/dpat/tools/src/{REGIONLIST_COMMIT}/REF/ATLASES/HCP-MMP1_UniqueRegionList.csv"],
    "RF": ["https://doi.org/10.1002/hbm.24213",
           f"https://github.com/ThomasYeoLab/CBIG/tree/{CBIG_COMMIT}/stable_projects/registration/Wu2017_RegistrationFusion"],
}

# ── helpers ──────────────────────────────────────────────────────────────


def fetch(key: str, cache: Path) -> Path:
    url, md5 = SOURCES[key]
    # Keep the source's extension: nibabel picks a reader by it. figshare
    # URLs have none, so their keys carry it.
    suffix = "".join(Path(url.rsplit("/", 1)[-1]).suffixes) or EXT.get(key, "")
    path = cache / f"{key}{suffix}"
    if not path.exists():
        urllib.request.urlretrieve(url, path)
    got = hashlib.md5(path.read_bytes()).hexdigest()
    if got != md5:
        raise RuntimeError(f"{key}: md5 {got} != pinned {md5} ({url})")
    return path


def provenance(*keys: str) -> list[dict]:
    return [{"key": k, "url": SOURCES[k][0], "md5": SOURCES[k][1]} for k in keys]


def distinct_rgba(n: int) -> list[tuple]:
    """n well-separated opaque colours, deterministic (golden-ratio hues)."""
    out = []
    for i in range(n):
        h = (i * 0.618033988749895) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 0.65, 0.9)
        out.append((r, g, b, 1.0))
    return out


def write_label_gii(path: Path, data: np.ndarray, table: dict, hemi: str) -> None:
    """GIfTI label file; ``table`` maps key -> (name, (r, g, b, a) in 0-1)."""
    lt = nib.gifti.GiftiLabelTable()
    for key, (name, rgba) in sorted(table.items()):
        lab = nib.gifti.GiftiLabel(key=int(key), red=rgba[0], green=rgba[1],
                                   blue=rgba[2], alpha=rgba[3])
        lab.label = name
        lt.labels.append(lab)
    da = nib.gifti.GiftiDataArray(
        data.astype(np.int32), intent="NIFTI_INTENT_LABEL", datatype="NIFTI_TYPE_INT32",
        meta=nib.gifti.GiftiMetaData({"AnatomicalStructurePrimary":
                                      "CortexLeft" if hemi == "L" else "CortexRight"}),
    )
    img = nib.gifti.GiftiImage(darrays=[da], labeltable=lt)
    nib.save(img, path)
    back = nib.load(path)
    if not np.array_equal(back.agg_data(), data.astype(np.int32)):
        raise RuntimeError(f"{path.name}: label data does not round-trip")
    keys = {lab.key for lab in back.labeltable.labels}
    missing = set(np.unique(data).tolist()) - keys
    if missing:
        raise RuntimeError(f"{path.name}: values without a label-table entry: {sorted(missing)}")


def write_tsv(path: Path, rows: list[dict]) -> None:
    cols = list(rows[0])
    with open(path, "w") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in rows:
            fh.write("\t".join(str(r[c]) for c in cols) + "\n")


def write_json(path: Path, obj: dict) -> None:
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2)


def hexcolor(rgba) -> str:
    return "#" + "".join(f"{int(round(c * 255)):02x}" for c in rgba[:3])


def surf_name(hemi: str | None, atlas: str, ext: str, **ents) -> str:
    parts = ["tpl-fsaverage"]
    if hemi:
        parts.append(f"hemi-{hemi}")
    parts.append(f"den-{DEN}")
    parts.append(f"atlas-{atlas}")
    parts += [f"{k}-{v}" for k, v in ents.items()]
    return "_".join(parts) + f"_dseg{ext}"


def read_annot(path: Path):
    labels, ctab, names = nib.freesurfer.read_annot(str(path))
    return labels, ctab, [n.decode() for n in names]


# ── checks ───────────────────────────────────────────────────────────────


def check_nesting(cache: Path) -> dict:
    """fsaverage6 vertices == first N_FS6 fsaverage vertices, on both hemis."""
    out = {}
    for hemi in HEMIS:
        big = nib.load(fetch(f"sphere_{hemi}_164k", cache)).agg_data("pointset")
        small = nib.load(fetch(f"sphere_{hemi}_41k", cache)).agg_data("pointset")
        if big.shape != (N_FS7, 3) or small.shape != (N_FS6, 3):
            raise RuntimeError(f"hemi-{hemi}: sphere shapes {big.shape} / {small.shape}")
        diff = float(np.abs(big[:N_FS6] - small).max())
        if diff > 1e-4:
            raise RuntimeError(f"hemi-{hemi}: fsaverage6 is not the first {N_FS6} "
                               f"fsaverage vertices (max |diff| {diff} mm)")
        out[hemi] = diff
    return out


# ── stages ───────────────────────────────────────────────────────────────


def stage_schaefer(cache: Path, surf_dir: Path, mni_tsv_dir: Path) -> dict:
    report = {}
    for nets in (7, 17):
        seg = f"{nets}n"
        rows, datas = [], {}
        for hemi, hh in HEMIS.items():
            lab6, ctab, names = read_annot(fetch(f"schaefer_fs6_{hh}_{nets}", cache))
            lab7, _, names7 = read_annot(fetch(f"schaefer_fs7_{hh}_{nets}", cache))
            if lab6.shape != (N_FS6,) or names7 != names:
                raise RuntimeError(f"Schaefer {seg} {hemi}: unexpected annot shape or names")
            agree = float((lab7[:N_FS6] == lab6).mean())
            if agree != 1.0:
                raise RuntimeError(f"Schaefer {seg} {hemi}: fsaverage[:{N_FS6}] vs "
                                   f"fsaverage6 agree on only {agree:.4f} of vertices")
            n_parc = len(np.unique(lab6[lab6 > 0]))
            if n_parc != 200 or len(names) != 201:
                raise RuntimeError(f"Schaefer {seg} {hemi}: {n_parc} parcels, {len(names)} names")
            offset = 0 if hemi == "L" else 200
            data = np.where(lab6 > 0, lab6 + offset, 0)
            table = {0: ("???", (0, 0, 0, 0))}
            for i in range(1, 201):
                rgba = tuple(ctab[i, :3] / 255.0) + (1.0,)
                table[i + offset] = (names[i], rgba)
                rows.append({"index": i + offset, "name": names[i], "color": hexcolor(rgba)})
            write_label_gii(surf_dir / surf_name(hemi, "Schaefer2018", ".label.gii",
                                                  seg=seg, scale=400), data, table, hemi)
            datas[hemi] = data
            report[f"{seg}_{hemi}"] = {"parcels": n_parc, "medial_wall_vertices": int((lab6 == 0).sum()),
                                      "fsaverage_first_N_agreement": agree}
        rows.sort(key=lambda r: r["index"])
        write_tsv(surf_dir / surf_name(None, "Schaefer2018", ".tsv", seg=seg, scale=400), rows)

        # The MNI dseg.tsv already staged came from TemplateFlow; say whether
        # its names match CBIG's current release (they differ if it predates
        # CBIG v0.14.3's label renaming).
        mni_tsv = mni_tsv_dir / f"tpl-{MNI}_atlas-Schaefer2018_seg-{seg}_scale-400_res-2_dseg.tsv"
        name_match = None
        if mni_tsv.exists():
            mni_names = [l.rstrip("\n").split("\t")[1] for l in open(mni_tsv)][1:]
            name_match = sum(a == b["name"] for a, b in zip(mni_names, rows))
            report[f"{seg}_mni_tsv_names_matching"] = f"{name_match}/400"
        write_json(surf_dir / surf_name(None, "Schaefer2018", ".json", seg=seg, scale=400), {
            "Name": f"Schaefer 2018, 400 parcels, {nets}-network order, on fsaverage6",
            "Atlas": "Schaefer2018",
            "Template": "fsaverage", "Density": DEN, "DensityAlias": "fsaverage6",
            "Description": ("CBIG's fsaverage6 .annot converted to GIfTI label files, one per "
                            "hemisphere. Label values: LH 1-200, RH 201-400 (the order of the "
                            "annot names, LH then RH); 0 is the FreeSurfer-defined medial wall."),
            "Sources": provenance(f"schaefer_fs6_lh_{nets}", f"schaefer_fs6_rh_{nets}",
                                  f"schaefer_fs7_lh_{nets}", f"schaefer_fs7_rh_{nets}"),
            "Checks": {"fsaverage_first_40962_vertices_equal_fsaverage6_labels": True,
                       "parcels_per_hemisphere": 200},
            "Caveat": (None if name_match in (None, 400) else
                       f"Only {name_match}/400 names match the MNI dseg.tsv staged from "
                       "TemplateFlow in this tree; that table predates CBIG's label renaming. "
                       "Match volume and surface parcels by index only after checking."),
            "ReferencesAndLinks": REFS["Schaefer2018"],
            "GeneratedBy": {"Name": Path(__file__).name},
        })
    return report


def mmp_table(cache: Path):
    """Region list rows keyed by global index (LH 1-180, RH 181-360)."""
    import csv
    with open(fetch("mmp_regionlist", cache), encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    # One section is spelt two ways in the source table; take the canonical
    # spelling per Cortex_ID from Glasser 2016 (hyphenated).
    canonical = {}
    for r in rows:
        canonical.setdefault(int(r["Cortex_ID"]), set()).add(r["cortex"])
    sections = {}
    for cid, spellings in canonical.items():
        sections[cid] = sorted(spellings, key=lambda s: (s.count("-"), s))[-1]
    if len(sections) != 22:
        raise RuntimeError(f"HCP-MMP1 region list has {len(sections)} sections, expected 22")
    table = {}
    for r in rows:
        rid = int(r["regionID"])
        idx = rid if r["LR"] == "L" else rid - 200 + 180   # R regionIDs are 201-380
        table[idx] = {"region": r["region"], "long": r["regionLongName"], "hemi": r["LR"],
                      "section_id": int(r["Cortex_ID"]), "section": sections[int(r["Cortex_ID"])]}
    if sorted(table) != list(range(1, 361)):
        raise RuntimeError("HCP-MMP1 region list does not cover indices 1-360")
    return table, sections


def stage_mmp_surface(cache: Path, surf_dir: Path, table: dict, sections: dict) -> dict:
    report = {}
    rows = []
    sec_colors = dict(zip(sorted(sections), distinct_rgba(len(sections))))
    for hemi, hh in HEMIS.items():
        lab7, ctab, names = read_annot(fetch(f"mmp_{hh}", cache))
        if lab7.shape != (N_FS7,) or len(names) != 181:
            raise RuntimeError(f"HCP-MMP1 {hemi}: annot shape {lab7.shape}, {len(names)} names")
        offset = 0 if hemi == "L" else 180
        for i in range(1, 181):
            expect = f"{hemi}_{table[i + offset]['region']}_ROI"
            if names[i].lower() != expect.lower():
                raise RuntimeError(f"HCP-MMP1 {hemi} label {i} is {names[i]!r}, "
                                   f"region list says {expect!r}")
        lab6 = lab7[:N_FS6]
        n_parc = len(np.unique(lab6[lab6 > 0]))
        if n_parc != 180:
            raise RuntimeError(f"HCP-MMP1 {hemi}: {n_parc} areas survive on fsaverage6")
        data = np.where(lab6 > 0, lab6 + offset, 0)
        parc_table = {0: ("???", (0, 0, 0, 0))}
        for i in range(1, 181):
            rgba = tuple(ctab[i, :3] / 255.0) + (1.0,)
            parc_table[i + offset] = (names[i], rgba)
            t = table[i + offset]
            rows.append({"index": i + offset, "name": names[i], "hemi": hemi,
                         "region": t["region"], "region_long_name": t["long"],
                         "section_id": t["section_id"], "section": t["section"],
                         "color": hexcolor(rgba)})
        write_label_gii(surf_dir / surf_name(hemi, "HCPMMP1", ".label.gii"), data, parc_table, hemi)

        lut = np.zeros(361, dtype=int)
        for idx, t in table.items():
            lut[idx] = t["section_id"]
        sec = lut[data]
        sec_table = {0: ("???", (0, 0, 0, 0))}
        sec_table.update({cid: (sections[cid], sec_colors[cid]) for cid in sections})
        write_label_gii(surf_dir / surf_name(hemi, "HCPMMP1", ".label.gii", seg="sections"),
                        sec, sec_table, hemi)
        report[hemi] = {"areas": n_parc, "sections": len(np.unique(sec[sec > 0])),
                        "unlabelled_vertices": int((lab6 == 0).sum())}
    rows.sort(key=lambda r: r["index"])
    write_tsv(surf_dir / surf_name(None, "HCPMMP1", ".tsv"), rows)
    write_tsv(surf_dir / surf_name(None, "HCPMMP1", ".tsv", seg="sections"),
              [{"index": c, "name": sections[c], "color": hexcolor(sec_colors[c])}
               for c in sorted(sections)])
    common = {
        "Atlas": "HCPMMP1", "Template": "fsaverage", "Density": DEN, "DensityAlias": "fsaverage6",
        "Sources": provenance("mmp_lh", "mmp_rh", "mmp_regionlist", "sphere_L_164k",
                              "sphere_R_164k", "sphere_L_41k", "sphere_R_41k"),
        "Method": ("fsaverage .annot (Mills 2016) restricted to its first 40,962 vertices per "
                   "hemisphere, which are the fsaverage6 vertices (checked against "
                   "TemplateFlow's spheres: identical coordinates)."),
        "ReferencesAndLinks": REFS["HCPMMP1"],
        "GeneratedBy": {"Name": Path(__file__).name},
    }
    write_json(surf_dir / surf_name(None, "HCPMMP1", ".json"), {
        "Name": "HCP-MMP1.0 (Glasser 2016), 360 areas, on fsaverage6", **common,
        "Description": ("Label values: LH 1-180, RH 181-360, the same index space as "
                        "the MNI152NLin2009cAsym dseg in this tree; 0 is unlabelled "
                        "(medial wall). Columns section_id/section give the 22-section "
                        "grouping of Glasser 2016's supplementary neuroanatomical results.")})
    write_json(surf_dir / surf_name(None, "HCPMMP1", ".json", seg="sections"), {
        "Name": "HCP-MMP1.0 grouped into Glasser 2016's 22 sections, on fsaverage6", **common,
        "Description": ("Label values 1-22 in each hemisphere (the hemisphere is the file). "
                        "Area-to-section lookup from the HCP-MMP1_UniqueRegionList table "
                        "(Cortex_ID/cortex), transcribed from Glasser 2016's supplementary "
                        "neuroanatomical results. One section is spelt two ways in that "
                        "table; the hyphenated spelling is used.")})
    return report


def stage_mmp_volume(cache: Path, mni_dir: Path, table: dict) -> dict:
    from nilearn.image import resample_to_img

    ref_files = sorted(mni_dir.glob(f"tpl-{MNI}_atlas-Schaefer2018_*_res-2_dseg.nii.gz"))
    if not ref_files:
        raise FileNotFoundError(f"no Schaefer res-2 dseg under {mni_dir}; nothing to resample onto")
    ref = nib.load(ref_files[0])
    src = nib.load(fetch("mmp_mni", cache))
    src_data = np.rint(np.asarray(src.dataobj)).astype(np.int16)
    src_int = nib.Nifti1Image(src_data, src.affine)
    out = resample_to_img(src_int, ref, interpolation="nearest",
                          force_resample=True, copy_header=True)
    data = np.rint(np.asarray(out.dataobj)).astype(np.int16)
    # The source uses 1-180 for both hemispheres. Split at x = 0 in world
    # coordinates; the res-2 grid has no voxel centre on x = 0.
    i = np.arange(data.shape[0])
    x = ref.affine[0, 0] * i + ref.affine[0, 3]
    if np.any(np.isclose(x, 0)):
        raise RuntimeError("res-2 grid has a voxel centre on x = 0; midline split is ambiguous")
    right = (x > 0)[:, None, None] & (data > 0)
    data = np.where(right, data + 180, data).astype(np.int16)
    img = nib.Nifti1Image(data, ref.affine)
    img.set_data_dtype(np.int16)
    stem = f"tpl-{MNI}_atlas-HCPMMP1_res-2_dseg"
    nib.save(img, mni_dir / f"{stem}.nii.gz")
    present = set(np.unique(data[data > 0]).tolist())
    missing = sorted(set(range(1, 361)) - present)
    write_tsv(mni_dir / f"{stem}.tsv",
              [{"index": k, "name": f"{t['hemi']}_{t['region']}_ROI", "section_id": t["section_id"],
                "section": t["section"]} for k, t in sorted(table.items())])
    write_json(mni_dir / f"{stem}.json", {
        "Name": "HCP-MMP1.0 (Glasser 2016), volumetric projection", "Atlas": "HCPMMP1",
        "Description": ("Horn 2016's projection of HCP-MMP1.0 onto the MNI152 ICBM2009a "
                        "nonlinear template (1 mm), split into hemispheres at world x = 0 "
                        "(the source labels both hemispheres 1-180; here LH 1-180, RH 181-360, "
                        "matching the fsaverage6 files) and resampled once with "
                        "nearest-neighbour interpolation onto the res-2 grid of the Schaefer "
                        "dseg files in this tree."),
        "SourceTemplate": "MNI152NLin2009aAsym", "TargetTemplate": MNI,
        "SourceGrid": {"shape": list(src.shape), "affine": src.affine.tolist()},
        "TargetGrid": {"shape": list(ref.shape), "affine": ref.affine.tolist()},
        "Interpolation": "nearest", "CrossTemplateRegistration": "none",
        "Caveat": ("Labels are defined on ICBM2009a; no 2009a->2009c warp was applied (the two "
                   "share a coordinate frame and differ in intensity processing). A surface "
                   "atlas projected to a volume is approximate in folded cortex, and the "
                   "1 mm -> 2 mm nearest-neighbour step drops small areas; see "
                   "AreasMissingAtRes2."),
        "AreasMissingAtRes2": missing,
        "Sources": provenance("mmp_mni", "mmp_regionlist"),
        "ReferencesAndLinks": REFS["HCPMMP1"],
        "GeneratedBy": {"Name": Path(__file__).name},
    })
    return {"areas_present": len(present), "areas_missing": missing,
            "voxels_L": int(((data > 0) & (data <= 180)).sum()), "voxels_R": int((data > 180).sum())}


def stage_ho_surface(cache: Path, surf_dir: Path) -> dict:
    """Harvard-Oxford cortical maxprob-thr25 -> fsaverage6 via RF-ANTs."""
    import scipy.io as sio
    from nilearn.datasets import fetch_atlas_harvard_oxford

    atlas = fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
    img = atlas.maps if hasattr(atlas.maps, "affine") else nib.load(atlas.maps)
    src_path = getattr(atlas, "filename", None) or img.get_filename()
    labels = list(atlas.labels)
    vol = np.asarray(img.dataobj).astype(int)
    inv = np.linalg.inv(img.affine)
    colors = distinct_rgba(len(labels) - 1)
    table = {0: ("Background", (0, 0, 0, 0))}
    table.update({i: (labels[i], colors[i - 1]) for i in range(1, len(labels))})

    # Family-B cortical ROIs, reported for the record (label values as in
    # scripts/pattern_similarity/shared.py).
    family_b = {"EVC": 24, "EAC": 45, "AG": 21, "Precuneus": 31, "mPFC": 25}
    report = {}
    for hemi, hh in HEMIS.items():
        ras = sio.loadmat(fetch(f"rf_{hh}", cache))["ras"]
        if ras.shape != (3, N_FS7):
            raise RuntimeError(f"RF mapping {hemi}: shape {ras.shape}")
        ras = ras[:, :N_FS6]
        ijk = np.rint((inv @ np.vstack([ras, np.ones(N_FS6)]))[:3]).astype(int)
        inside = np.all((ijk >= 0) & (ijk < np.array(vol.shape)[:, None]), axis=0)
        data = np.zeros(N_FS6, dtype=int)
        data[inside] = vol[ijk[0, inside], ijk[1, inside], ijk[2, inside]]
        # Medial wall = Schaefer's FreeSurfer-defined medial wall (label 0).
        medial = read_annot(fetch(f"schaefer_fs6_{hh}_7", cache))[0] == 0
        data[medial] = 0
        coverage = float((data[~medial] > 0).mean())
        if coverage < MIN_HO_COVERAGE:
            raise RuntimeError(f"Harvard-Oxford {hemi}: only {coverage:.3f} of cortex labelled")
        write_label_gii(surf_dir / surf_name(hemi, "HOCPA", ".label.gii", desc="th25"),
                        data, table, hemi)
        report[hemi] = {"coverage_non_medial_wall": round(coverage, 4),
                        "labels_present": len(np.unique(data[data > 0])),
                        "family_b_vertices": {k: int((data == v).sum()) for k, v in family_b.items()}}
    write_tsv(surf_dir / surf_name(None, "HOCPA", ".tsv", desc="th25"),
              [{"index": i, "name": labels[i]} for i in range(1, len(labels))])
    write_json(surf_dir / surf_name(None, "HOCPA", ".json", desc="th25"), {
        "Name": "Harvard-Oxford cortical structural atlas, maxprob-thr25, on fsaverage6",
        "Atlas": "HOCPA", "Template": "fsaverage", "Density": DEN, "DensityAlias": "fsaverage6",
        "Description": ("FSL cort-maxprob-thr25-2mm (MNI152NLin6Asym, the file the MNI "
                        "HOCPA dseg in this tree was staged from) projected to fsaverage with "
                        "the registration-fusion mapping of Wu 2018 (RF-ANTs, FSL MNI152 -> "
                        "fsaverage, average of 745 subjects): each fsaverage vertex takes the "
                        "label of the voxel nearest its mapped MNI coordinate, as CBIG's "
                        "projection does for label volumes. Restricted to the first 40,962 "
                        "vertices (fsaverage6). The FreeSurfer-defined medial wall (Schaefer "
                        "label 0) is set to 0. Label values are the volume's (1-48, bilateral)."),
        "SourceTemplate": "MNI152NLin6Asym", "SourceFile": str(src_path),
        "Projection": "registration fusion (RF-ANTs), nearest voxel",
        "Caveat": ("Vertices whose mapped coordinate falls outside every label at the 25% "
                   "threshold stay 0; see Coverage. Volume-defined gyral labels are "
                   "approximate on the surface."),
        "Coverage": {h: r["coverage_non_medial_wall"] for h, r in report.items()},
        "Sources": provenance("rf_lh", "rf_rh", "schaefer_fs6_lh_7", "schaefer_fs6_rh_7"),
        "ReferencesAndLinks": REFS["RF"] + ["https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/Atlases"],
        "GeneratedBy": {"Name": Path(__file__).name},
    })
    return report


def register_in_dataset_description(atlases_dir: Path) -> None:
    write_json(atlases_dir / "atlas-HCPMMP1_description.json", {
        "Name": "HCP multi-modal parcellation, version 1.0 (HCP-MMP1.0)",
        "Description": ("180 areas per hemisphere delineated from multi-modal MRI (architecture, "
                        "function, connectivity, topography) in 210 HCP subjects. Staged here on "
                        "fsaverage6 (den-41k, from Mills 2016's fsaverage projection) and on the "
                        "MNI152NLin2009cAsym res-2 grid (from Horn 2016's ICBM2009a volumetric "
                        "projection), plus Glasser 2016's 22-section grouping; see each file's "
                        "JSON sidecar."),
        "License": ("HCP Open Access Data Use Terms (https://www.humanconnectome.org/study/"
                    "hcp-young-adult/document/wu-minn-hcp-consortium-open-access-data-use-terms)"),
        "ReferencesAndLinks": REFS["HCPMMP1"][:3],
        "Species": "homo sapiens",
        "DerivedFrom": "multi-modal MRI, 210 HCP young adults",
        "LevelType": "group",
    })
    path = atlases_dir / "dataset_description.json"
    desc = json.loads(path.read_text())
    name = Path(__file__).name
    if not any(g.get("Name") == name for g in desc.get("GeneratedBy", [])):
        desc.setdefault("GeneratedBy", []).append({
            "Name": name,
            "Description": ("fsaverage6 (den-41k) Schaefer-400 7n/17n, HCP-MMP1 (+22 sections) and "
                            "Harvard-Oxford cortical label files under tpl-fsaverage/, and "
                            "HCP-MMP1 on the MNI152NLin2009cAsym res-2 grid (mmmdata "
                            "scripts/functional_space/stage_fsaverage6_atlases.py); see each "
                            "file's JSON sidecar"),
        })
        write_json(path, desc)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)

    config = load_config(config_dir=_REPO_ROOT / "config")
    atlases_dir = Path(config["paths"]["output_dir"]) / "atlases"
    if not (atlases_dir / "dataset_description.json").exists():
        raise FileNotFoundError(f"{atlases_dir} is not the staged atlases derivative")
    surf_dir = atlases_dir / SURF_DIR
    mni_dir = atlases_dir / MNI_DIR
    existing = list(surf_dir.glob("*")) + list(mni_dir.glob("*atlas-HCPMMP1*"))
    if existing and not args.overwrite:
        raise FileExistsError(f"{len(existing)} outputs exist (e.g. {existing[0]}); "
                              "pass --overwrite to replace them")
    surf_dir.mkdir(parents=True, exist_ok=True)

    report = {}
    with tempfile.TemporaryDirectory() as tmp:
        cache = Path(tmp)
        report["nesting_max_abs_diff_mm"] = check_nesting(cache)
        report["schaefer"] = stage_schaefer(cache, surf_dir, mni_dir)
        table, sections = mmp_table(cache)
        report["hcpmmp1_surface"] = stage_mmp_surface(cache, surf_dir, table, sections)
        report["hcpmmp1_volume"] = stage_mmp_volume(cache, mni_dir, table)
        report["hocpa_surface"] = stage_ho_surface(cache, surf_dir)
    register_in_dataset_description(atlases_dir)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
