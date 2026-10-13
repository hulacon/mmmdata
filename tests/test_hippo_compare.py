"""Pure helpers of neuroimaging.hippo_compare (no ANTs)."""

import math

import nibabel as nib
import numpy as np
import pytest

from neuroimaging import hippo_compare as hc


def test_dice_basic_and_empty():
    a = np.array([1, 1, 0, 0], bool)
    b = np.array([1, 0, 1, 0], bool)
    assert hc.dice(a, b) == pytest.approx(0.5)
    assert hc.dice(a, a) == 1.0
    assert math.isnan(hc.dice(np.zeros(4, bool), np.zeros(4, bool)))


def test_to_canon_merges_ca4_into_dg_and_drops_cyst():
    native = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
    canon = hc.to_canon(native, hc.NATIVE_TO_CANON["hippunfold"])
    names = [hc.CANON[c] for c in canon]
    assert names == ["bg", "Sub", "CA1", "CA2", "CA3", "DG", "DG", "SRLM", "bg"]


def test_to_canon_hsf_order_and_fs_hemi():
    assert [hc.CANON[c] for c in hc.to_canon(np.arange(6), hc.NATIVE_TO_CANON["hsf"])] == \
        ["bg", "DG", "CA1", "CA2", "CA3", "Sub"]
    aseg = np.array([0, 17, 53, 2000])
    assert list(hc.to_canon(aseg, hc.NATIVE_TO_CANON["fs_L"])) == [0, 7, 0, 0]
    assert list(hc.to_canon(aseg, hc.NATIVE_TO_CANON["fs_R"])) == [0, 0, 7, 0]


def test_box_affine_covers_mask_with_padding():
    aff = np.diag([1.0, 1.0, 1.0, 1.0])
    aff[:3, 3] = [-10, -20, -30]
    mask = np.zeros((20, 20, 20), bool)
    mask[5:8, 10:12, 3:4] = True
    box, shape = hc.box_affine(mask, aff, spacing=0.5, pad=2.0)
    np.testing.assert_allclose(box[:3, :3], np.eye(3) * 0.5)
    # mask voxel faces in world mm, plus pad
    lo = nib.affines.apply_affine(aff, [4.5, 9.5, 2.5]) - 2.0
    hi = nib.affines.apply_affine(aff, [7.5, 11.5, 3.5]) + 2.0
    np.testing.assert_allclose(box[:3, 3], lo)
    far = nib.affines.apply_affine(box, np.array(shape) - 1)
    assert np.all(far >= hi - 1e-9)


def test_box_affine_rejects_empty():
    with pytest.raises(ValueError):
        hc.box_affine(np.zeros((3, 3, 3), bool), np.eye(4), 0.5, 1.0)


def test_plurality_on_2x_blocks():
    # coarse 2 mm grid, fine 1 mm grid with voxel centres offset so each coarse voxel holds 2x2x2 fine voxels
    coarse_aff = np.diag([2.0, 2.0, 2.0, 1.0])
    fine_aff = np.diag([1.0, 1.0, 1.0, 1.0])
    fine_aff[:3, 3] = -0.5
    fine = np.zeros((4, 2, 2), np.uint8)
    fine[0:2, :, :] = 2          # coarse voxel 0: all CA1
    fine[2:4, :, :] = 0
    fine[2, 0, :] = 1            # coarse voxel 1: 2 Sub, 2 CA3, 4 bg -> bg wins
    fine[3, 0, :] = 4
    cmap = hc.coarse_map(fine.shape, fine_aff, (2, 1, 1), coarse_aff)
    lab = hc.plurality(fine, cmap)
    out, aff = hc.crop_coarse(lab, cmap, coarse_aff)
    assert out.shape == (2, 1, 1)
    assert list(out.ravel()) == [2, 0]
    np.testing.assert_allclose(aff, coarse_aff)


def test_plurality_tie_goes_to_lower_class():
    coarse_aff = np.diag([2.0, 2.0, 2.0, 1.0])
    fine_aff = np.eye(4)
    fine_aff[:3, 3] = -0.5
    fine = np.zeros((2, 2, 2), np.uint8)
    fine[0] = 3
    fine[1] = 2                  # 4 CA2 vs 4 CA1 -> CA1 (2)
    cmap = hc.coarse_map(fine.shape, fine_aff, (1, 1, 1), coarse_aff)
    assert list(hc.plurality(fine, cmap)) == [2]


def test_coarse_map_drops_fine_voxels_outside():
    cmap = hc.coarse_map((4, 1, 1), np.eye(4), (2, 1, 1), np.eye(4))
    assert list(cmap.keep) == [0, 1]
    assert list(cmap.flat) == [0, 1]


def test_crop_coarse_shifts_origin():
    coarse_aff = np.diag([2.0, 2.0, 2.0, 1.0])
    cmap = hc.CoarseMap(keep=np.array([0]), group=np.array([0]),
                        flat=np.array([np.ravel_multi_index((3, 4, 5), (10, 10, 10))]), coarse_shape=(10, 10, 10))
    out, aff = hc.crop_coarse(np.array([5], np.uint8), cmap, coarse_aff)
    assert out.shape == (1, 1, 1) and out[0, 0, 0] == 5
    np.testing.assert_allclose(aff[:3, 3], [6.0, 8.0, 10.0])


def _arms():
    return [hc.Arm("A", "hippunfold", "a", "SPC"), hc.Arm("B", "hsf", "b", "SPC"), hc.Arm("F", "fs", "f", None)]


def test_pair_rows_scores_subfields_only_between_subfield_arms():
    canon = {"A": np.array([1, 2, 6, 0]), "B": np.array([1, 2, 0, 0]), "F": np.array([7, 7, 7, 0])}
    rows = hc.pair_rows(canon, _arms())
    regions = {(r["arm_a"], r["arm_b"]): {x["region"] for x in rows
                                          if (x["arm_a"], x["arm_b"]) == (r["arm_a"], r["arm_b"])} for r in rows}
    assert regions[("A", "B")] == {"hipp", "hippNoSRLM", *hc.SUBFIELDS}
    assert regions[("A", "F")] == {"hipp", "hippNoSRLM"}
    by = {(r["arm_a"], r["arm_b"], r["region"]): r for r in rows}
    assert by[("A", "F", "hipp")]["dice"] == pytest.approx(1.0)          # SRLM counts as hippocampus
    assert by[("A", "F", "hippNoSRLM")]["dice"] == pytest.approx(0.8)    # 2*2 / (2+3)
    assert by[("A", "B", "Sub")]["dice"] == 1.0


def test_assignment_rows_either_and_both():
    canon = {"A": np.array([1, 2, 3, 0, 6]), "B": np.array([1, 3, 0, 5, 1]), "F": np.zeros(5, np.uint8)}
    (row,) = hc.assignment_rows(canon, _arms())
    assert (row["arm_a"], row["arm_b"]) == ("A", "B")
    # A subfields at 0,1,2 (SRLM at 4 is not a subfield); B at 0,1,3,4 -> either = 5, both = 2, same = 1
    assert row["n_either"] == 5 and row["n_both"] == 2
    assert row["agree_either"] == pytest.approx(0.2)
    assert row["agree_both"] == pytest.approx(0.5)
