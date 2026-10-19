"""A synthetic dataset for the mmmget tests: catalog, QC decisions, confounds.

Everything is made up: participant labels 01/02, a task called ``memory``
and a single-run task called ``rest``. Layout under ``tmp_path``::

    bids/                          the configured BIDS root
      inventory/catalog.duckdb     the catalog (files + datasets tables)
      derivatives/fmriprep/...     one real confounds TSV (sub-01 run-01)
      derivatives/atlases/...      a tiny Schaefer-shaped atlas
      derivatives/duckbrain/qc/decisions/   current QC decisions
      derivatives/preprocessing_qc/         a legacy-location, old-schema one
    config/base.toml               points the loader at bids/

Catalog paths are dataset-relative, as the bids2table sweep writes them;
only the files a test reads exist on disk, which is the point: runs()
answers from the catalog, never from a directory walk.

QC record, by run (task memory, ses-01):

    sub-01 run-01   keep, by a person
    sub-01 run-02   exclude, by a person              -> excluded
    sub-01 run-03   pending, automated, recommends exclude  -> kept
    sub-02 run-01   exclude, old {latest, history} schema, legacy dir -> excluded
    sub-02 run-02   exclude by a person, then a newer domain note -> excluded
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

FMRIPREP = "derivatives/fmriprep"
HIPPO = "derivatives/hipposeg"
MNI = ("MNI152NLin2009cAsym", "2")
CONFOUND_COLS = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z",
                 "framewise_displacement", "non_steady_state_outlier00",
                 *[f"a_comp_cor_{i:02d}" for i in range(6)],
                 "cosine00", "cosine01"]
N_VOL = 12
PERSON = "reviewer-a"


def _func_rows(sub, ses, task, run):
    r = f"_run-{run:02d}" if run else ""
    stem = f"sub-{sub}_ses-{ses}_task-{task}{r}"
    d = f"sub-{sub}/ses-{ses}/func/"
    mni = f"_space-{MNI[0]}_res-{MNI[1]}"
    base = dict(dataset_relpath=FMRIPREP, sub=sub, ses=ses, datatype="func",
                task=task, run=run, space=None, res=None, hemi=None)
    rows = [
        {**base, "desc": "confounds", "suffix": "timeseries", "ext": ".tsv",
         "path": d + stem + "_desc-confounds_timeseries.tsv"},
        {**base, "desc": "confounds", "suffix": "timeseries", "ext": ".json",
         "path": d + stem + "_desc-confounds_timeseries.json"},
        {**base, "space": MNI[0], "res": MNI[1], "desc": "preproc", "suffix": "bold",
         "ext": ".nii.gz", "path": d + stem + mni + "_desc-preproc_bold.nii.gz"},
        {**base, "space": MNI[0], "res": MNI[1], "desc": "brain", "suffix": "mask",
         "ext": ".nii.gz", "path": d + stem + mni + "_desc-brain_mask.nii.gz"},
        {**base, "space": MNI[0], "res": MNI[1], "desc": None, "suffix": "boldref",
         "ext": ".nii.gz", "path": d + stem + mni + "_boldref.nii.gz"},
        {**base, "desc": "preproc", "suffix": "bold", "ext": ".nii.gz",
         "path": d + stem + "_desc-preproc_bold.nii.gz"},
        {**base, "desc": "coreg", "suffix": "boldref", "ext": ".nii.gz",
         "path": d + stem + "_desc-coreg_boldref.nii.gz"},
        {**base, "hemi": "L", "space": "fsaverage6", "desc": None, "suffix": "bold",
         "ext": ".func.gii", "path": d + stem + "_hemi-L_space-fsaverage6_bold.func.gii"},
        {**base, "hemi": "R", "space": "fsaverage6", "desc": None, "suffix": "bold",
         "ext": ".func.gii", "path": d + stem + "_hemi-R_space-fsaverage6_bold.func.gii"},
        # the raw dataset's events file
        {**base, "dataset_relpath": ".", "desc": None, "suffix": "events",
         "ext": ".tsv", "path": d + stem + "_events.tsv"},
    ]
    return rows


def build_catalog(db: Path, extra_rows=()) -> None:
    rows = []
    for sub in ("01", "02"):
        for run in (1, 2, 3):
            rows += _func_rows(sub, "01", "memory", run)
        rows += _func_rows(sub, "01", "rest", None)
    # a raw BOLD with no fMRIPrep output: not a run to runs()
    rows.append(dict(dataset_relpath=".", sub="01", ses="02", datatype="func",
                     task="memory", run=1, space=None, res=None, hemi=None,
                     desc=None, suffix="bold", ext=".nii.gz",
                     path="sub-01/ses-02/func/sub-01_ses-02_task-memory_run-01_bold.nii.gz"))
    # hippocampal subfield segmentations (only sub-01)
    for hemi in ("L", "R"):
        rows.append(dict(dataset_relpath=HIPPO, sub="01", ses="01", datatype="anat",
                         task=None, run=None, space="T1w", res=None, hemi=hemi,
                         desc="subfields", suffix="dseg", ext=".nii.gz",
                         path=f"sub-01/ses-01/anat/sub-01_ses-01_hemi-{hemi}"
                              "_space-T1w_desc-subfields_dseg.nii.gz"))
    rows += list(extra_rows)
    df = pd.DataFrame(rows).astype({"run": "Int64"})
    db.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db))
    con.register("df", df)
    con.execute("CREATE TABLE files AS SELECT dataset_relpath, sub, ses, datatype, task, "
                "CAST(run AS INTEGER) AS run, space, res, hemi, \"desc\", suffix, ext, path "
                "FROM df")
    con.execute("CREATE TABLE datasets (relpath VARCHAR, kind VARCHAR)")
    con.execute("INSERT INTO datasets VALUES ('.', 'canonical'), (?, 'canonical')", [FMRIPREP])
    con.close()


def _decision(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _entry(decision, reviewer=PERSON, automated=False, **extra):
    return {"decision": decision, "reason": "fixture", "reviewer": reviewer,
            "automated": automated, "timestamp": "2000-01-01T00:00:00+00:00", **extra}


def write_decisions(bids: Path) -> None:
    cur = bids / "derivatives" / "duckbrain" / "qc" / "decisions"
    legacy = bids / "derivatives" / "preprocessing_qc"

    def key(sub, run):
        return f"sub-{sub}_ses-01_task-memory_run-{run:02d}_bold"

    _decision(cur / f"{key('01', 1)}_decision.json",
              {"run_key": key("01", 1), "decisions": [_entry("keep")]})
    _decision(cur / f"{key('01', 2)}_decision.json",
              {"run_key": key("01", 2),
               "decisions": [_entry("pending", "auto-stub", True), _entry("exclude")]})
    _decision(cur / f"{key('01', 3)}_decision.json",
              {"run_key": key("01", 3),
               "decisions": [_entry("pending", "auto-stub", True, recommendation="exclude")]})
    _decision(legacy / "sub-02" / f"{key('02', 1)}_decision.json",
              {"latest": _entry("exclude"), "history": [_entry("pending", "auto-stub", True)]})
    _decision(cur / f"{key('02', 2)}_decision.json",
              {"run_key": key("02", 2),
               "decisions": [_entry("exclude"), _entry("keep", domain="physio")]})


def write_confounds(bids: Path, sub="01", run=1) -> Path:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(N_VOL, len(CONFOUND_COLS))), columns=CONFOUND_COLS)
    df["non_steady_state_outlier00"] = [1] + [0] * (N_VOL - 1)
    df.loc[0, "framewise_displacement"] = np.nan
    p = (bids / FMRIPREP / f"sub-{sub}" / "ses-01" / "func"
         / f"sub-{sub}_ses-01_task-memory_run-{run:02d}_desc-confounds_timeseries.tsv")
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, sep="\t", index=False, na_rep="n/a")
    return p


def write_atlas(atlases: Path) -> None:
    import nibabel as nib

    anat = atlases / "tpl-MNI152NLin2009cAsym" / "anat"
    anat.mkdir(parents=True, exist_ok=True)
    stem = "tpl-MNI152NLin2009cAsym_atlas-Schaefer2018_seg-7n_scale-100_res-2_dseg"
    data = np.zeros((4, 4, 4), dtype=np.int16)
    data[0, :, :] = 1
    data[1, :, :] = 2
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(anat / f"{stem}.nii.gz"))
    pd.DataFrame({"index": [1, 2], "name": ["7Networks_LH_Vis_1", "7Networks_RH_Vis_1"]}) \
        .to_csv(anat / f"{stem}.tsv", sep="\t", index=False)


def write_hippo(bids: Path) -> None:
    import nibabel as nib

    d = bids / HIPPO / "sub-01" / "ses-01" / "anat"
    d.mkdir(parents=True, exist_ok=True)
    data = np.zeros((3, 3, 3), dtype=np.int16)
    data[1, 1, 1] = 2
    nib.save(nib.Nifti1Image(data, np.eye(4)),
             str(d / "sub-01_ses-01_hemi-L_space-T1w_desc-subfields_dseg.nii.gz"))


SUBFIELD_TOML = f"""
[mmmget.hippocampal_subfields]
dataset = "{HIPPO}"
desc = "subfields"
space = "T1w"
labels = {{ Sub = 1, CA1 = 2, DG = 5 }}
"""


def make_env(tmp_path: Path, monkeypatch, extra_toml: str = "") -> Path:
    """Build the dataset and point config at it. Returns the BIDS root."""
    import mmmget

    bids = tmp_path / "bids"
    build_catalog(bids / "inventory" / "catalog.duckdb")
    write_decisions(bids)
    write_confounds(bids)
    write_atlas(bids / "derivatives" / "atlases")
    write_hippo(bids)
    cfg = tmp_path / "config"
    cfg.mkdir()
    (cfg / "base.toml").write_text(
        f'[paths]\nbids_project_dir = "{bids}"\n'
        f'output_dir = "{bids / "derivatives"}"\n' + extra_toml)
    monkeypatch.setenv("MMMDATA_CONFIG_DIR", str(cfg))
    mmmget.reset_provenance()
    return bids
