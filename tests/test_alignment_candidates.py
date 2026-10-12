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
