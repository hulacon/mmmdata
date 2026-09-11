"""deface_anat.py — the data-free logic.

Byte-exact rewriting, the post-write invariants, mirror-path mapping, the
backup refusal, FSL .mat interpretation, and target discovery on a synthetic
tree shaped like fMRIPrep + FreeSurfer output. The registrations themselves
(pydeface, ANTs, flirt) need the real tools and are exercised by the pilot run
logged in mmmdata-agents/docs/workbench/anat-defacing/.
"""

import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import deface_anat as da  # noqa: E402


# --------------------------------------------------------------------------- #
# fixtures                                                                    #
# --------------------------------------------------------------------------- #
def _vol(shape=(12, 14, 10), seed=0, dtype=np.int16, lo=1, hi=1000):
    rng = np.random.default_rng(seed)
    return rng.integers(lo, hi, size=shape).astype(dtype)


def _face(shape=(12, 14, 10)):
    face = np.zeros(shape, dtype=bool)
    face[:, :4, :3] = True          # a slab in the anterior-inferior corner
    return face


def _brain(shape=(12, 14, 10)):
    brain = np.zeros(shape, dtype=bool)
    brain[2:10, 6:13, 4:9] = True
    return brain


def _write_nifti(path, data, slope=None, inter=None, affine=None):
    affine = np.eye(4) if affine is None else affine
    img = nib.Nifti1Image(data, affine)
    if slope is not None:
        img.header.set_slope_inter(slope, inter)
    img.to_filename(str(path))
    return path


# --------------------------------------------------------------------------- #
# byte-exact rewriting                                                        #
# --------------------------------------------------------------------------- #
def test_zero_voxels_nifti_preserves_header_and_outside_bytes(tmp_path):
    data = _vol()
    src = _write_nifti(tmp_path / "a.nii.gz", data, slope=2.0, inter=1.0)
    face = _face()
    dst = tmp_path / "b.nii.gz"
    da.zero_voxels_nifti(src, dst, face)
    a, b = nib.load(src), nib.load(dst)
    assert a.header.binaryblock == b.header.binaryblock
    assert a.get_data_dtype() == b.get_data_dtype() == np.int16
    ra = np.asanyarray(a.dataobj)
    rb = np.asanyarray(b.dataobj)
    assert np.array_equal(ra[~face], rb[~face])
    # scaled zero is the intercept, and that is what a raw 0 means on disk
    unscaled = np.asanyarray(b.dataobj.get_unscaled())
    assert (unscaled[face] == 0).all()


def test_zero_voxels_nifti_big_endian_and_trailing_dim(tmp_path):
    data = _vol(dtype=np.dtype(">i2"))[..., None]      # 4-D with a singleton
    img = nib.Nifti1Image(data, np.eye(4))
    src = tmp_path / "a.nii"
    img.to_filename(str(src))
    face = _face()
    dst = tmp_path / "b.nii"
    da.zero_voxels_nifti(src, dst, face)
    a, b = nib.load(src), nib.load(dst)
    assert a.header.binaryblock == b.header.binaryblock
    ra, rb = np.asanyarray(a.dataobj), np.asanyarray(b.dataobj)
    assert np.array_equal(ra[~face], rb[~face])
    assert (rb[face] == 0).all()


def test_zero_voxels_nifti_rejects_wrong_mask_shape(tmp_path):
    src = _write_nifti(tmp_path / "a.nii.gz", _vol())
    with pytest.raises(ValueError):
        da.zero_voxels_nifti(src, tmp_path / "b.nii.gz", np.zeros((3, 3, 3), bool))


def test_zero_voxels_mgh_preserves_header_fields(tmp_path):
    data = _vol(dtype=np.uint8, hi=255)
    img = nib.MGHImage(data, np.diag([1.0, 1.0, 1.0, 1.0]))
    img.header["tr"] = 2300.0
    img.header["flip_angle"] = 9.0
    src = tmp_path / "a.mgz"
    img.to_filename(str(src))
    face = _face()
    dst = tmp_path / "b.mgz"
    da.zero_voxels_mgh(src, dst, face)
    a, b = nib.load(src), nib.load(dst)
    for k in a.header.keys():
        assert np.array_equal(a.header[k], b.header[k]), k
    ra, rb = np.asanyarray(a.dataobj), np.asanyarray(b.dataobj)
    assert np.array_equal(ra[~face], rb[~face])
    assert (rb[face] == 0).all()


def test_check_rewrite_passes_when_face_and_brain_disjoint(tmp_path):
    src = _write_nifti(tmp_path / "a.nii.gz", _vol())
    face, brain = _face(), _brain()
    dst = tmp_path / "b.nii.gz"
    da.zero_voxels(src, dst, face)
    checks = da.check_rewrite(src, dst, face, brain)
    assert checks == {"outside_face_identical": True, "face_zeroed": True,
                      "header_preserved": True, "max_abs_diff_in_brain": 0.0}


def test_check_rewrite_reports_brain_change_when_mask_leaks(tmp_path):
    src = _write_nifti(tmp_path / "a.nii.gz", _vol())
    face = _face()
    face[3, 7, 5] = True                      # one voxel inside the brain box
    brain = _brain()
    dst = tmp_path / "b.nii.gz"
    da.zero_voxels(src, dst, face)
    checks = da.check_rewrite(src, dst, face, brain)
    assert checks["max_abs_diff_in_brain"] > 0
    assert checks["outside_face_identical"] and checks["face_zeroed"]


# --------------------------------------------------------------------------- #
# mirror paths and backups                                                    #
# --------------------------------------------------------------------------- #
def _roots(tmp_path):
    bids = tmp_path / "bids"
    return da.Roots(bids=bids, source=tmp_path / "src", deriv=bids / "derivatives", containers=tmp_path / "c")


def test_mirror_paths(tmp_path):
    r = _roots(tmp_path)
    raw = r.bids / "sub-01" / "ses-01" / "anat" / "sub-01_ses-01_T1w.nii.gz"
    assert r.mirror(raw) == r.source / "sub-01" / "ses-01" / "anat_faced" / "sub-01_ses-01_T1w.nii.gz"
    fs = r.freesurfer / "sub-01" / "mri" / "orig.mgz"
    assert r.mirror(fs) == r.source / "derivatives_faced" / "fmriprep" / "sourcedata" / "freesurfer" / "sub-01" / "mri" / "orig.mgz"
    fig = r.fmriprep / "sub-01" / "figures" / "sub-01_desc-reconall_T1w.svg"
    assert r.mirror(fig) == r.source / "derivatives_faced" / "fmriprep" / "sub-01" / "figures" / "sub-01_desc-reconall_T1w.svg"
    with pytest.raises(ValueError):
        r.mirror(tmp_path / "elsewhere.nii.gz")


def test_backup_copies_then_refuses_a_different_mirror(tmp_path):
    original = tmp_path / "o.bin"
    original.write_bytes(b"faced")
    mirror = tmp_path / "m" / "o.bin"
    da.backup(original, mirror, da.sha256(original), dry_run=False, log=lambda *_: None)
    assert mirror.read_bytes() == b"faced"
    da.backup(original, mirror, da.sha256(original), dry_run=False, log=lambda *_: None)   # idempotent
    mirror.write_bytes(b"something else")
    with pytest.raises(RuntimeError, match="REFUSING"):
        da.backup(original, mirror, da.sha256(original), dry_run=False, log=lambda *_: None)


def test_backup_dry_run_writes_nothing(tmp_path):
    original = tmp_path / "o.bin"
    original.write_bytes(b"faced")
    mirror = tmp_path / "m" / "o.bin"
    da.backup(original, mirror, da.sha256(original), dry_run=True, log=lambda *_: None)
    assert not mirror.exists()


# --------------------------------------------------------------------------- #
# FSL .mat interpretation                                                     #
# --------------------------------------------------------------------------- #
def test_fsl_identity_mat_is_no_refinement(tmp_path):
    affine = np.diag([-1.0, 1.0, 1.0, 1.0])           # radiological, no flip needed
    img = nib.Nifti1Image(np.zeros((10, 10, 10), np.float32), affine)
    mat = tmp_path / "id.mat"
    np.savetxt(mat, np.eye(4))
    shift, rot = da.fsl_mat_refinement(mat, img, img)
    assert shift == pytest.approx(0.0, abs=1e-9)
    assert rot == pytest.approx(0.0, abs=1e-6)


def test_fsl_translation_mat_is_measured_in_mm(tmp_path):
    affine = np.diag([-2.0, 2.0, 2.0, 1.0])           # 2 mm voxels
    img = nib.Nifti1Image(np.zeros((10, 10, 10), np.float32), affine)
    m = np.eye(4)
    m[:3, 3] = [3.0, 4.0, 0.0]                        # FSL .mat translations are in mm
    mat = tmp_path / "t.mat"
    np.savetxt(mat, m)
    shift, rot = da.fsl_mat_refinement(mat, img, img)
    assert shift == pytest.approx(5.0, abs=1e-6)
    assert rot == pytest.approx(0.0, abs=1e-6)


def test_fsl_flip_convention_for_neurological_grids(tmp_path):
    """A neurological (det>0) grid gets FSL's x flip; an identity .mat between
    two such grids on the same geometry must still be no refinement."""
    affine = np.diag([1.0, 1.0, 1.0, 1.0])
    img = nib.Nifti1Image(np.zeros((10, 12, 14), np.float32), affine)
    mat = tmp_path / "id.mat"
    np.savetxt(mat, np.eye(4))
    shift, rot = da.fsl_mat_refinement(mat, img, img)
    assert shift == pytest.approx(0.0, abs=1e-9) and rot == pytest.approx(0.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# discovery on a synthetic tree                                               #
# --------------------------------------------------------------------------- #
def _fake_tree(tmp_path, subject="01", mismatch_orig=False):
    r = _roots(tmp_path)
    sub = f"sub-{subject}"
    anat = r.bids / sub / "ses-01" / "anat"
    anat.mkdir(parents=True)
    shape = (12, 14, 10)
    run1 = _write_nifti(anat / f"{sub}_ses-01_acq-MPR_run-01_T1w.nii.gz", _vol(shape, 1))
    run2 = _write_nifti(anat / f"{sub}_ses-01_acq-MPR_run-02_T1w.nii.gz", _vol(shape, 2))
    t2w = _write_nifti(anat / f"{sub}_ses-01_acq-SPC_T2w.nii.gz", _vol(shape, 3))
    fp = r.fmriprep / sub
    (fp / "anat").mkdir(parents=True)
    (fp / "ses-01" / "anat").mkdir(parents=True)
    stem = f"{sub}_acq-MPR"
    _write_nifti(fp / "anat" / f"{stem}_desc-preproc_T1w.nii.gz", _vol(shape, 4))
    _write_nifti(fp / "anat" / f"{stem}_desc-brain_mask.nii.gz", _brain(shape).astype(np.uint8))
    _write_nifti(fp / "anat" / f"{stem}_space-MNI152NLin2009cAsym_res-2_desc-preproc_T1w.nii.gz", _vol((8, 9, 8), 5))
    _write_nifti(fp / "anat" / f"{stem}_space-MNI152NLin2009cAsym_res-2_desc-brain_mask.nii.gz", np.ones((8, 9, 8), np.uint8))
    (fp / "anat" / f"{stem}_from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5").write_bytes(b"")
    (fp / "anat" / f"{stem}_from-T1w_to-fsnative_mode-image_xfm.txt").write_text("#Insight Transform File V1.0\n")
    for run in ("run-01", "run-02"):
        (fp / "ses-01" / "anat" / f"{sub}_ses-01_acq-MPR_{run}_from-orig_to-T1w_mode-image_xfm.txt").write_text("#\n")
    _write_nifti(fp / "ses-01" / "anat" / f"{sub}_ses-01_desc-preproc_T2w.nii.gz", _vol(shape, 6))
    mri = r.freesurfer / sub / "mri"
    (mri / "orig").mkdir(parents=True)
    conformed = (16, 16, 16)
    brain = np.zeros(conformed, np.uint8)
    brain[4:12, 4:12, 4:12] = 1
    nib.MGHImage(brain * 100, np.eye(4)).to_filename(str(mri / "brainmask.mgz"))
    for name in ("orig.mgz", "T1.mgz", "nu.mgz"):
        nib.MGHImage(_vol(conformed, 7, np.uint8, 1, 255), np.eye(4)).to_filename(str(mri / name))
    nib.MGHImage(brain * 90, np.eye(4)).to_filename(str(mri / "brain.mgz"))       # brain-only, not a target
    inputs = [(np.asanyarray(nib.load(run1).dataobj)), (np.asanyarray(nib.load(run2).dataobj))]
    if mismatch_orig:
        inputs[0] = inputs[0] + 1
    for k, arr in enumerate(inputs, start=1):
        nib.MGHImage(arr, np.eye(4)).to_filename(str(mri / "orig" / f"{k:03d}.mgz"))
    nib.MGHImage(_vol(shape, 8), np.eye(4)).to_filename(str(mri / "rawavg.mgz"))
    nib.MGHImage(np.asanyarray(nib.load(t2w).dataobj), np.eye(4)).to_filename(str(mri / "orig" / "T2raw.mgz"))
    return r


def test_discover_full_layout(tmp_path):
    r = _fake_tree(tmp_path)
    lay = da.discover(r, "01")
    kinds = sorted((t.kind, t.path.name, t.leg) for t in lay.targets)
    assert ("raw_t1w", "sub-01_ses-01_acq-MPR_run-01_T1w.nii.gz", "raw:sub-01_ses-01_acq-MPR_run-01") in kinds
    assert ("raw_t2w", "sub-01_ses-01_acq-SPC_T2w.nii.gz", "sess:sub-01_ses-01_acq-SPC_T2w.nii.gz") in kinds
    assert ("fp_t1w", "sub-01_acq-MPR_desc-preproc_T1w.nii.gz", "T1w") in kinds
    assert ("fp_mni_t1w", "sub-01_acq-MPR_space-MNI152NLin2009cAsym_res-2_desc-preproc_T1w.nii.gz", "MNI") in kinds
    assert ("fp_t2w", "sub-01_ses-01_desc-preproc_T2w.nii.gz", "T1w") in kinds
    assert ("fs_conformed", "orig.mgz", "fsnative") in kinds
    assert ("fs_conformed", "T1.mgz", "fsnative") in kinds
    assert ("fs_own", "001.mgz", "raw:sub-01_ses-01_acq-MPR_run-01") in kinds
    assert ("fs_own", "002.mgz", "raw:sub-01_ses-01_acq-MPR_run-02") in kinds
    assert ("fs_own", "rawavg.mgz", "raw:sub-01_ses-01_acq-MPR_run-01") in kinds
    assert ("fs_own", "T2raw.mgz", "sess:sub-01_ses-01_acq-SPC_T2w.nii.gz") in kinds
    assert not any(t.path.name == "brain.mgz" for t in lay.targets)
    assert len(lay.targets) == 13      # 2 raw T1w + 1 raw T2w + 3 fMRIPrep + 3 conformed FS + 4 own-grid FS
    assert set(lay.orig_to_t1w) == {"sub-01_ses-01_acq-MPR_run-01", "sub-01_ses-01_acq-MPR_run-02"}


def test_discover_refuses_an_unmatched_freesurfer_input(tmp_path):
    r = _fake_tree(tmp_path, mismatch_orig=True)
    with pytest.raises(RuntimeError, match="byte-equal"):
        da.discover(r, "01")


def test_discover_raw_only_needs_no_derivatives(tmp_path):
    r = _roots(tmp_path)
    anat = r.bids / "sub-08" / "ses-01" / "anat"
    anat.mkdir(parents=True)
    _write_nifti(anat / "sub-08_ses-01_acq-MPR_run-01_T1w.nii.gz", _vol())
    _write_nifti(anat / "sub-08_ses-01_acq-SPC_T2w.nii.gz", _vol())
    lay = da.discover(r, "08", raw_only=True)
    assert [t.kind for t in lay.targets] == ["raw_t1w", "raw_t2w"]
    with pytest.raises(FileNotFoundError, match="raw-only"):
        da.discover(r, "08")


def test_audit_freesurfer_flags_an_unknown_whole_head_volume(tmp_path):
    r = _fake_tree(tmp_path)
    mri = r.freesurfer / "sub-01" / "mri"
    nib.MGHImage(_vol((16, 16, 16), 9, np.uint8, 1, 255), np.eye(4)).to_filename(str(mri / "mystery.mgz"))
    with pytest.raises(RuntimeError, match="mystery.mgz"):
        da.audit_freesurfer_mri(mri)


# --------------------------------------------------------------------------- #
# reports                                                                     #
# --------------------------------------------------------------------------- #
def test_report_files_selects_anat_renderers_only(tmp_path):
    r = _roots(tmp_path)
    mriqc = r.deriv / "mriqc"
    mriqc.mkdir(parents=True)
    for n in ("sub-01_ses-01_acq-MPR_run-01_T1w.html", "sub-01_ses-01_acq-SPC_T2w.html", "group_T1w.html",
              "sub-01_ses-02_task-rest_bold.html"):
        (mriqc / n).write_text("")
    mfigs = mriqc / "sub-01" / "figures"
    mfigs.mkdir(parents=True)
    for n in ("sub-01_ses-01_acq-MPR_run-01_desc-background_T1w.svg", "sub-01_ses-01_acq-SPC_desc-zoomed_T2w.svg",
              "sub-01_ses-02_task-rest_desc-carpet_bold.svg"):
        (mfigs / n).write_text("")
    figs = r.fmriprep / "sub-01" / "figures"
    figs.mkdir(parents=True)
    for n in ("sub-01_acq-MPR_desc-reconall_T1w.svg", "sub-01_acq-MPR_dseg.svg",
              "sub-01_acq-MPR_space-MNI152NLin2009cAsym_desc-preproc_T1w.svg",
              "sub-01_ses-01_task-rest_desc-coreg_bold.svg", "sub-01_ses-01_desc-summary_T1w.html"):
        (figs / n).write_text("")
    qc = r.deriv / "prf" / "sub-01" / "qc"
    qc.mkdir(parents=True)
    for n in ("sub-01_task-prf_space-T1w_desc-viewer_prf.html", "sub-01_task-prf_space-fsnative_hemi-L_desc-viewer_prf.html"):
        (qc / n).write_text("")
    names = sorted(p.name for p in da.report_files(r, "01"))
    assert names == sorted([
        "sub-01_ses-01_acq-MPR_run-01_T1w.html", "sub-01_ses-01_acq-SPC_T2w.html",
        "sub-01_ses-01_acq-MPR_run-01_desc-background_T1w.svg", "sub-01_ses-01_acq-SPC_desc-zoomed_T2w.svg",
        "sub-01_acq-MPR_dseg.svg",
        "sub-01_task-prf_space-T1w_desc-viewer_prf.html",
    ])


# --------------------------------------------------------------------------- #
# provenance round trip                                                       #
# --------------------------------------------------------------------------- #
def test_provenance_round_trip(tmp_path):
    p = tmp_path / "prov.json"
    prov = da.load_provenance(p)
    prov["files"]["x"] = {"sha_before": "a", "sha_after": "b", "checks": {"max_abs_diff_in_brain": 0.0}}
    da.save_provenance(p, prov)
    again = json.loads(p.read_text())
    assert again["files"]["x"]["sha_after"] == "b"
    assert "updated" in again


# --------------------------------------------------------------------------- #
# restore                                                                     #
# --------------------------------------------------------------------------- #
def test_restore_puts_originals_back_and_refuses_a_bad_mirror(tmp_path):
    original = _write_nifti(tmp_path / "t.nii.gz", _vol())
    face = _face()
    mirror = tmp_path / "m" / "t.nii.gz"
    sha_before = da.sha256(original)
    da.backup(original, mirror, sha_before, dry_run=False, log=lambda *_: None)
    da.zero_voxels(original, original, face)
    prov = da.load_provenance(tmp_path / "p.json")
    prov["files"][str(original)] = {"mirror": str(mirror), "sha_before": sha_before, "sha_after": da.sha256(original)}
    assert da.restore_files(prov, log=lambda *_: None) == 1
    assert da.sha256(original) == sha_before and prov["files"] == {}
    prov["files"][str(original)] = {"mirror": str(mirror), "sha_before": "not-the-hash", "sha_after": "x"}
    with pytest.raises(RuntimeError, match="mirror"):
        da.restore_files(prov, log=lambda *_: None)


def test_montage_handles_anisotropic_non_ras_volume(tmp_path):
    affine = np.array([[0, 0, 1.8, 0], [0.43, 0, 0, 0], [0, -0.43, 0, 0], [0, 0, 0, 1.0]])
    img = nib.Nifti1Image(_vol((20, 18, 6), dtype=np.float32), affine)
    src = tmp_path / "slab.nii.gz"
    img.to_filename(str(src))
    out = da.montage(src, np.zeros((20, 18, 6), bool), tmp_path / "m.png", "slab")
    assert out.exists() and out.stat().st_size > 0


def test_discover_falls_back_to_direct_pydeface_for_a_new_session(tmp_path):
    """A session converted after fMRIPrep ran has a raw T1w with no
    from-orig transform: it must be planned as a direct-pydeface leg, and
    the FreeSurfer input count is checked against what fMRIPrep saw."""
    r = _fake_tree(tmp_path)
    anat = r.bids / "sub-01" / "ses-09" / "anat"
    anat.mkdir(parents=True)
    _write_nifti(anat / "sub-01_ses-09_acq-MPR_T1w.nii.gz", _vol(seed=99))
    lay = da.discover(r, "01")
    assert lay.raw_direct == {"sub-01_ses-09_acq-MPR"}
    assert any(t.leg == "raw:sub-01_ses-09_acq-MPR" and t.kind == "raw_t1w" for t in lay.targets)
    assert len(lay.targets) == 14


def test_mirror_falls_back_when_the_session_dir_is_not_writable(tmp_path):
    r = _roots(tmp_path)
    raw = r.bids / "sub-07" / "ses-01" / "anat" / "sub-07_ses-01_T1w.nii.gz"
    locked = r.source / "sub-07" / "ses-01"
    locked.mkdir(parents=True)
    locked.chmod(0o555)
    try:
        assert r.mirror(raw) == r.source / "anat_defacing" / "sub-07" / "anat_faced" / "ses-01" / "sub-07_ses-01_T1w.nii.gz"
    finally:
        locked.chmod(0o755)
    assert r.mirror(raw) == r.source / "sub-07" / "ses-01" / "anat_faced" / "sub-07_ses-01_T1w.nii.gz"


def test_reports_skip_files_already_recorded_as_moved(tmp_path, monkeypatch):
    """A viewer bundle rebuilt after its faced original was moved must not be
    moved again (its mirror holds the faced one and is never overwritten)."""
    r = _roots(tmp_path)
    qc = r.deriv / "prf" / "sub-01" / "qc"
    qc.mkdir(parents=True)
    bundle = qc / "sub-01_task-prf_space-T1w_desc-viewer_prf.html"
    bundle.write_text("faced")
    prov_path = r.provenance_path("01")
    prov = da.load_provenance(prov_path)
    monkeypatch.setattr(da, "load_roots", lambda: r)
    class A: subject = "01"; dry_run = False
    assert da.cmd_reports(A()) == 0
    assert not bundle.exists() and r.mirror(bundle).read_text() == "faced"
    bundle.write_text("defaced rebuild")
    assert da.cmd_reports(A()) == 0            # second pass: skipped, not refused
    assert bundle.read_text() == "defaced rebuild"
