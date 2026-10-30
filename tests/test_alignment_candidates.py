"""Pure helpers of neuroimaging.alignment_candidates (no ANTs, no FreeSurfer)."""

import numpy as np
import pytest

from neuroimaging import alignment_candidates as ac


def _rot_z(deg):
    t = np.radians(deg)
    M = np.eye(4)
    M[:2, :2] = [[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]]
    return M


def test_read_itk_txt_matches_centre_convention(tmp_path):
    # ITK: p -> A (p - c) + c + t
    A = _rot_z(10)[:3, :3]
    c = np.array([5.0, -3.0, 2.0])
    t = np.array([1.0, 2.0, -0.5])
    f = tmp_path / "x.txt"
    f.write_text("#Insight Transform File V1.0\n#Transform 0\nTransform: AffineTransform_float_3_3\n"
                 f"Parameters: {' '.join(map(str, [*A.ravel(), *t]))}\nFixedParameters: {' '.join(map(str, c))}\n")
    M = ac.read_itk_txt(f)
    p = np.array([[10.0, 0.0, 0.0]])
    np.testing.assert_allclose(ac.apply_affine(M, p)[0], A @ (p[0] - c) + c + t, atol=1e-9)


def test_read_itk_txt_rejects_two_transforms(tmp_path):
    f = tmp_path / "x.txt"
    f.write_text("Transform: AffineTransform_float_3_3\nTransform: AffineTransform_float_3_3\n"
                 "Parameters: 1 0 0 0 1 0 0 0 1 0 0 0\nFixedParameters: 0 0 0\n")
    with pytest.raises(ValueError):
        ac.read_itk_txt(f)


def test_composition_order_is_right_to_left():
    shift = np.eye(4)
    shift[0, 3] = 2.0
    rot = _rot_z(90)
    p = np.array([[1.0, 0.0, 0.0]])
    # rot after shift: (1,0,0) -> (3,0,0) -> (0,3,0)
    np.testing.assert_allclose(ac.apply_affine(rot @ shift, p)[0], [0.0, 3.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(ac.apply_affine(rot, ac.apply_affine(shift, p))[0], [0.0, 3.0, 0.0], atol=1e-9)


def test_summarise_pure_translation():
    pts = np.random.default_rng(0).normal(size=(50, 3))
    out = ac.summarise(pts + [0.3, 0.4, 0.0], pts, "B")
    assert out["B_mean_mm"] == pytest.approx(0.5)
    assert out["B_max_mm"] == pytest.approx(0.5)


def test_mask_points_stride_and_lps():
    mask = np.zeros((6, 6, 6), bool)
    mask[2:5, 2:5, 2:5] = True
    aff = np.diag([2.0, 2.0, 2.0, 1.0])
    pts, ijk = ac.mask_points_lps(mask, aff, stride=2)
    assert {tuple(x) for x in ijk} == {(i, j, k) for i in (2, 4) for j in (2, 4) for k in (2, 4)}
    # RAS (4, 4, 4) for voxel (2, 2, 2) -> LPS (-4, -4, 4)
    row = np.flatnonzero((ijk == [2, 2, 2]).all(axis=1))[0]
    np.testing.assert_allclose(pts[row], [-4.0, -4.0, 4.0])


def test_sample_field_handles_itk_5d():
    field = np.zeros((3, 3, 3, 1, 3), np.float32)
    field[1, 2, 0, 0] = [0.5, -1.0, 2.0]
    out = ac.sample_field_at_voxels(field, np.array([[1, 2, 0], [0, 0, 0]]))
    np.testing.assert_allclose(out, [[0.5, -1.0, 2.0], [0.0, 0.0, 0.0]])


def test_reference_and_followers():
    assert ac.reference_and_followers(["02", "05", "04", "07", "06", "08", "09"], "04", 5) == \
        ["04", "05", "06", "07", "08"]
    with pytest.raises(ValueError):
        ac.reference_and_followers(["04", "05"], "04", 5)
    with pytest.raises(ValueError):
        ac.reference_and_followers(["05", "06"], "04", 1)


def test_write_itk_txt_round_trips(tmp_path):
    M = _rot_z(7)
    M[:3, 3] = [0.3, -1.2, 2.5]
    np.testing.assert_allclose(ac.read_itk_txt(ac.write_itk_txt(M, tmp_path / "m.txt")), M, atol=1e-9)


def test_arm_b_coreg_chains_template_then_run_then_a():
    C, D, T = _rot_z(3), _rot_z(-5), _rot_z(11)
    C[:3, 3], D[:3, 3], T[:3, 3] = [1, 0, 0], [0, 2, 0], [0, 0, 3]
    p = np.array([[4.0, -2.0, 7.0]])
    want = ac.apply_affine(C, ac.apply_affine(D, ac.apply_affine(T, p)))
    np.testing.assert_allclose(ac.apply_affine(ac.arm_b_coreg(C, D, T), p), want, atol=1e-9)


def _arm_tree_world(tmp_path, stage0_mean="1.0000"):
    """Production: anatomy + ses-04 with a TBencoding and a TBretrieval run; stage 0: arm B moves 1 mm in x."""
    nib = pytest.importorskip("nibabel")
    from scipy.io import savemat

    prod, stage0 = tmp_path / "fmriprep", tmp_path / "stage0"
    anat, func = prod / "sub-99" / "anat", prod / "sub-99" / "ses-04" / "func"
    for d in (anat, func, prod / "sub-99" / "figures"):
        d.mkdir(parents=True)
    (prod / "dataset_description.json").write_text("{}")
    mask = np.zeros((8, 8, 8), np.uint8)
    mask[2:6, 2:6, 2:6] = 1
    nib.save(nib.Nifti1Image(mask, np.eye(4)), str(anat / "sub-99_desc-brain_mask.nii.gz"))
    nib.save(nib.Nifti1Image(mask.astype(np.float32), np.eye(4)), str(anat / "sub-99_desc-preproc_T1w.nii.gz"))
    ac.write_itk_txt(np.eye(4), anat / "sub-99_from-fsnative_to-T1w_mode-image_xfm.txt")
    for task in ("TBencoding", "TBretrieval"):
        pre = f"sub-99_ses-04_task-{task}_run-01"
        ac.write_itk_txt(np.eye(4), func / f"{pre}{ac.COREG_SUFFIX}")
        (func / f"{pre}_desc-hmc_boldref.nii.gz").write_bytes(b"x")
    (stage0 / "sub-99" / "xfm" / "B_bbreg").mkdir(parents=True)
    ac.write_itk_txt(np.eye(4), stage0 / "sub-99" / "xfm" / "B_bbreg" / "bbreg_itk.txt")
    params = np.r_[np.eye(3).ravel(), [1.0, 0.0, 0.0]][:, None]
    savemat(str(stage0 / "sub-99" / "xfm" / "B_run_sub-99_ses-04_task-TBencoding_run-01_0GenericAffine.mat"),
            {"AffineTransform_double_3_3": params, "fixed": np.zeros((3, 1))})
    (stage0 / "sub-99" / "sub-99_desc-stage0_deltas.tsv").write_text(
        f"sub\tses\ttask\trun\tB_mean_mm\n99\t04\tTBencoding\t01\t{stage0_mean}\n99\t04\tTBretrieval\t01\t0.5\n")
    return prod, stage0, func


def test_arm_tree_replaces_only_the_chosen_runs_coreg(tmp_path):
    prod, stage0, func = _arm_tree_world(tmp_path)
    out = tmp_path / "armB" / "fmriprep"
    t = ac.build_arm_tree(prod, stage0, "99", "B", ["04"], ["TBencoding"], out, log=lambda m: None)
    assert t[["task", "run"]].values.tolist() == [["TBencoding", "01"]]
    assert t.disp_mean_mm.iloc[0] == pytest.approx(1.0)
    f = out / "sub-99" / "ses-04" / "func"
    enc = f / f"sub-99_ses-04_task-TBencoding_run-01{ac.COREG_SUFFIX}"
    assert not enc.is_symlink() and ac.read_itk_txt(enc)[0, 3] == pytest.approx(1.0)
    assert (f / f"sub-99_ses-04_task-TBretrieval_run-01{ac.COREG_SUFFIX}").is_symlink()
    assert (f / "sub-99_ses-04_task-TBencoding_run-01_desc-hmc_boldref.nii.gz").is_symlink()
    assert (out / "sub-99" / "anat" / "sub-99_desc-brain_mask.nii.gz").is_symlink()
    assert not (out / "sub-99" / "figures").exists()
    assert ac.read_itk_txt(func / f"sub-99_ses-04_task-TBencoding_run-01{ac.COREG_SUFFIX}")[0, 3] == 0  # production untouched
    # A rebuild over an existing tree gives the same xfm, composed from production's, not from its own last output.
    ac.build_arm_tree(prod, stage0, "99", "B", ["04"], ["TBencoding"], out, log=lambda m: None)
    assert ac.read_itk_txt(enc)[0, 3] == pytest.approx(1.0)


def test_arm_tree_control_and_stage0_mismatch(tmp_path):
    prod, stage0, _ = _arm_tree_world(tmp_path, stage0_mean="0.5000")
    a = ac.build_arm_tree(prod, stage0, "99", "A", ["04"], ["TBencoding"], tmp_path / "A", log=lambda m: None)
    assert a.disp_mean_mm.iloc[0] == 0
    with pytest.raises(ValueError, match="stage 0 says 0.5000"):
        ac.build_arm_tree(prod, stage0, "99", "B", ["04"], ["TBencoding"], tmp_path / "B", log=lambda m: None)
    with pytest.raises(ValueError, match="must not be the production tree"):
        ac.build_arm_tree(prod, stage0, "99", "B", ["04"], ["TBencoding"], prod, log=lambda m: None)
