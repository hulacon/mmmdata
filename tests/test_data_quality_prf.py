"""data-quality pRF rows (T1.11): R² summaries on surface cortex, the fit mask and warped parcels."""

from __future__ import annotations

import json
import shutil

import numpy as np
import pandas as pd
import pytest

nib = pytest.importorskip("nibabel")

from neuroimaging import data_quality as dq  # noqa: E402
from neuroimaging import data_quality_prf as dqp  # noqa: E402

SHAPE = (6, 6, 4)
N_VERT = 50


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------

def test_r2_summary_counts_nan_and_never_fills():
    s = dqp.r2_summary(np.array([5.0, 15.0, 25.0, 35.0, np.nan]))
    assert s["n"] == 5 and s["n_nan"] == 1
    assert s["r2_median"] == pytest.approx(20.0)
    assert s["frac_r2_gt10"] == pytest.approx(.75)
    assert s["frac_r2_gt20"] == pytest.approx(.5)
    assert s["frac_r2_gt30"] == pytest.approx(.25)


def test_r2_summary_all_nan_is_nan_not_zero():
    s = dqp.r2_summary(np.full(3, np.nan))
    assert s["n_nan"] == 3
    assert np.isnan(s["r2_median"]) and np.isnan(s["frac_r2_gt20"])


def test_parcel_rows_coverage_and_mask():
    r2 = np.arange(24, dtype=float).reshape(2, 3, 4)
    labels = np.zeros(r2.shape, dtype=int)
    labels[0] = 1
    mask = np.ones(r2.shape, dtype=bool)
    mask[0, 0] = False  # 4 of parcel 1's 12 voxels out of the mask
    table = pd.DataFrame({"index": [1, 2], "name": ["a", "b"]})
    rows = dqp.parcel_rows(r2, labels, mask, table, "X", "prf")
    assert rows[0]["n_voxels_atlas"] == 12 and rows[0]["n_voxels_mask"] == 8
    assert rows[0]["coverage"] == pytest.approx(8 / 12)
    assert rows[0]["r2_median"] == pytest.approx(np.median(r2[0, 1:]))
    assert rows[1]["n_voxels_atlas"] == 0 and np.isnan(rows[1]["coverage"]) and np.isnan(rows[1]["r2_median"])


# ---------------------------------------------------------------------------
# Cell: build, collect, staleness
# ---------------------------------------------------------------------------

@pytest.fixture
def atlases(tmp_path, monkeypatch):
    root = tmp_path / "atlases"
    root.mkdir()
    lab = np.zeros(SHAPE, dtype=np.int16)
    lab[:3] = 1
    lab[3:] = 2
    nib.Nifti1Image(lab, np.eye(4)).to_filename(str(root / "sch.nii.gz"))
    pd.DataFrame({"index": [1, 2], "name": ["17Networks_LH_VisCent_ExStr_1", "17Networks_RH_DefaultA_IPL_1"]}
                 ).to_csv(root / "sch.tsv", sep="\t", index=False)
    lab2 = np.zeros(SHAPE, dtype=np.int16)
    lab2[:, :2] = 9
    lab2[:, 5] = 4
    nib.Nifti1Image(lab2, np.eye(4)).to_filename(str(root / "ho.nii.gz"))
    pd.DataFrame({"index": [4, 9], "name": ["Left Cerebral White Matter", "Left Hippocampus"]}
                 ).to_csv(root / "ho.tsv", sep="\t", index=False)
    monkeypatch.setattr(dq, "PARCELLATIONS", {
        "Schaefer17n400": {"stem": "sch", "exclude_substrings": ()},
        "HOSPA": {"stem": "ho", "exclude_substrings": ("White Matter",)},
    })
    # Identity grid: the "warp" is a copy, so parcels land where the fixture put them.
    monkeypatch.setattr(dqp, "warp_labels", lambda src, ref, xfm, out: shutil.copy(src, out))
    return root


def _write_label(path, vertices):
    lines = ["#!ascii label", str(len(vertices))] + [f"{v} 0.0 0.0 0.0 0.0" for v in vertices]
    path.write_text("\n".join(lines) + "\n")


def _fake_inputs(tmp_path, sub="03", seed=0):
    """A pRF fit (two runs, one off-grid) and the fMRIPrep files the cell reads."""
    rng = np.random.default_rng(seed)
    prf = tmp_path / "prf" / f"sub-{sub}"
    prf.mkdir(parents=True)
    fp = tmp_path / "fmriprep"
    for pol in dqp.POLARITIES:
        r2 = rng.uniform(0, 40, SHAPE)
        r2[0, 0, 0] = np.nan  # an unfit in-mask voxel
        r2[5, 5, :] = np.nan  # outside the fit mask
        nib.Nifti1Image(r2.astype(np.float32), np.eye(4)).to_filename(
            str(prf / f"sub-{sub}_task-prf_space-T1w_desc-R2_{pol}.nii.gz"))
        (prf / f"sub-{sub}_task-prf_space-T1w_{pol}.json").write_text(
            json.dumps({"Runs": ["ses-02_run-01", "ses-03_run-01"]}))
        for h in dqp.HEMIS:
            v = rng.uniform(0, 40, N_VERT).astype(np.float32)
            v[-1] = np.nan  # medial wall, outside cortex
            img = nib.gifti.GiftiImage(darrays=[nib.gifti.GiftiDataArray(v)])
            nib.save(img, str(prf / f"sub-{sub}_task-prf_space-fsnative_hemi-{h}_desc-R2_{pol}.shape.gii"))
    mask = np.ones(SHAPE, dtype=np.uint8)
    mask[5, 5, :] = 0
    for ses, shape in (("ses-02", (5, 5, 5)), ("ses-03", SHAPE)):  # ses-02 resampled: own grid
        d = fp / f"sub-{sub}" / ses / "func"
        d.mkdir(parents=True)
        m = mask if shape == SHAPE else np.ones(shape, dtype=np.uint8)
        nib.Nifti1Image(m, np.eye(4)).to_filename(
            str(d / f"sub-{sub}_{ses}_task-prf_run-01_space-T1w_desc-brain_mask.nii.gz"))
    anat = fp / f"sub-{sub}" / "anat"
    anat.mkdir(parents=True)
    (anat / f"sub-{sub}_acq-X_from-MNI152NLin2009cAsym_to-T1w_mode-image_xfm.h5").write_bytes(b"xfm")
    lab = fp / "sourcedata" / "freesurfer" / f"sub-{sub}" / "label"
    lab.mkdir(parents=True)
    for fs in ("lh", "rh"):
        _write_label(lab / f"{fs}.cortex.label", list(range(N_VERT - 1)))
    return tmp_path / "prf", fp


def test_build_cell_collect_and_staleness(tmp_path, atlases):
    prf_root, fp = _fake_inputs(tmp_path)
    tree = tmp_path / "dq"
    side = dqp.build_cell(tree, prf_root, fp, atlases, "03", {"code_version": "x"}, log=lambda m: None)
    assert side["mask_runs"] == ["ses-03_run-01"]  # the off-grid run is left out of the mask
    assert side["n_voxels_mask"] == int(np.prod(SHAPE)) - 4

    rows, parcels = dqp.collect(tree)
    assert len(rows) == len(dqp.POLARITIES) * (len(dqp.HEMIS) + 1)
    surf = rows[rows["domain"] == "fsnative_hemi-L_cortex"].iloc[0]
    assert surf["n"] == N_VERT - 1 and surf["n_nan"] == 0  # the NaN medial-wall vertex is not cortex
    vol = rows[rows["domain"] == "T1w_mask"].iloc[0]
    assert vol["n_nan"] == 1  # the unfit voxel is counted, not filled
    assert set(parcels["atlas"]) == {"Schaefer17n400", "HOSPA"}
    assert "Left Cerebral White Matter" not in set(parcels["name"])  # exclusions follow the collection
    assert len(parcels) == len(dqp.POLARITIES) * 3
    for p in dqp.cell_paths(tree, "03").values():
        assert p.exists()

    assert dqp.build_cell(tree, prf_root, fp, atlases, "03", {}, log=lambda m: None) is None  # current
    gii = prf_root / "sub-03" / "sub-03_task-prf_space-fsnative_hemi-L_desc-R2_negprf.shape.gii"
    img = nib.load(str(gii))
    img.darrays[0].data[0] = 99.0
    nib.save(img, str(gii))
    assert dqp.build_cell(tree, prf_root, fp, atlases, "03", {}, log=lambda m: None) is not None  # input changed


def test_missing_inputs_are_loud(tmp_path, atlases):
    prf_root, fp = _fake_inputs(tmp_path)
    next((prf_root / "sub-03").glob("*hemi-R_desc-R2_prf.shape.gii")).unlink()
    with pytest.raises(FileNotFoundError, match="incomplete"):
        dqp.fit_files(prf_root, "03")
    (fp / "sourcedata" / "freesurfer" / "sub-03" / "label" / "rh.cortex.label").unlink()
    with pytest.raises(FileNotFoundError, match="cortex label"):
        dqp.subject_inputs(fp, "03")


def test_session_level_anat_transform_is_found(tmp_path):
    fp = tmp_path / "fmriprep"
    anat = fp / "sub-06" / "ses-01" / "anat"
    anat.mkdir(parents=True)
    xfm = anat / "sub-06_ses-01_acq-X_from-MNI152NLin2009cAsym_to-T1w_mode-image_xfm.h5"
    xfm.write_bytes(b"xfm")
    lab = fp / "sourcedata" / "freesurfer" / "sub-06" / "label"
    lab.mkdir(parents=True)
    for fs in ("lh", "rh"):
        _write_label(lab / f"{fs}.cortex.label", [0])
    assert dqp.subject_inputs(fp, "06")["xfm"] == xfm
    (fp / "sub-06" / "ses-02" / "anat").mkdir(parents=True)
    (fp / "sub-06" / "ses-02" / "anat" / "sub-06_ses-02_acq-X_from-MNI152NLin2009cAsym_to-T1w_mode-image_xfm.h5").write_bytes(b"x")
    with pytest.raises(FileNotFoundError, match="found 2"):
        dqp.subject_inputs(fp, "06")


def test_cortex_label_past_surface_is_refused(tmp_path, atlases):
    r2_surf = {f"{h}_{p}": np.zeros(5) for h in dqp.HEMIS for p in dqp.POLARITIES}
    cortex = {"L": np.arange(10), "R": np.arange(5)}
    with pytest.raises(ValueError, match="cortex label indexes vertex"):
        dqp.write_cell(tmp_path, "03", {}, r2_surf, cortex, np.ones(SHAPE, bool), {}, np.eye(4), atlases, {})


def test_warp_without_ants_names_the_module(monkeypatch, tmp_path):
    monkeypatch.setattr(dqp.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="module load ants"):
        dqp.warp_labels(tmp_path / "a", tmp_path / "b", tmp_path / "c", tmp_path / "d")
