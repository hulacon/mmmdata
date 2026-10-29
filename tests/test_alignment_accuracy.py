"""neuroimaging.alignment_accuracy on synthetic slabs: coverage fractions, compartments, pial edge r, ribbon specificity."""

import numpy as np
import pandas as pd
import pytest

from neuroimaging import alignment_accuracy as aa


def test_coverage_fraction_counts_fine_voxels_inside_each_coarse_voxel():
    fine = np.zeros((8, 8, 8), bool)
    fine[:3] = True
    # 2 mm voxels whose edges fall on fine voxel edges: coarse i covers fine 2i and 2i+1.
    ref = np.diag([2.0, 2.0, 2.0, 1.0])
    ref[:3, 3] = 0.5
    f = aa.coverage_fraction(fine, np.eye(4), (4, 4, 4), ref)
    np.testing.assert_allclose(f[:, 1, 1], [1.0, 0.5, 0.0, 0.0])
    # Outside the fine grid counts as uncovered, not as an error.
    assert aa.coverage_fraction(np.ones((2, 2, 2), bool), np.eye(4), (4, 4, 4), ref)[3, 3, 3] == 0


def _slab(shape=(30, 6, 6)):
    """WM for x < 10, cortex at x 10-12, outside-brain from x 13: a flat cortical sheet."""
    x = np.arange(shape[0])[:, None, None] * np.ones(shape)
    wm = (x < 10).astype(float)
    gm = ((x >= 10) & (x <= 12)).astype(float)
    return gm, wm, 1.0 - gm - wm, x


def test_compartments_on_a_flat_sheet():
    gm, wm, out, x = _slab()
    c = aa.compartments(gm, wm, out)
    col = lambda m: sorted(set(x[m].astype(int)))  # noqa: E731
    assert col(c.gm) == [10, 11, 12]
    assert col(c.wm_band) == [8, 9]                 # WM within 2 voxels of GM
    assert col(c.csf_side) == [13, 14]
    assert col(c.pial_band) == [11, 12, 13]         # outer GM row dilated by 1, never into WM
    assert not (c.pial_band & c.wm).any()


def test_anatomy_compartments_register_sub_voxel_shifts():
    # 1 mm anatomy along x: WM below 16, a 4 mm ribbon 16-19, outside from 20; 2 mm BOLD grid.
    fine = np.zeros((48, 6, 6), np.int32)
    fine[:16] = 2
    ribbon = np.zeros(fine.shape, bool)
    ribbon[16:20] = True
    fine[16:20] = 3
    eye, bold = np.eye(4), np.diag([2.0, 2.0, 2.0, 1.0])
    gm_x = lambda c: sorted(set(np.nonzero(c.gm)[0]))  # noqa: E731
    base = aa.anatomy_compartments(ribbon, eye, fine, eye, (24, 3, 3), bold)
    assert (aa.anatomy_compartments(ribbon, eye, fine, eye, (24, 3, 3), bold, shift=(0, 0.0)).gm == base.gm).all()
    half = aa.anatomy_compartments(ribbon, eye, fine, eye, (24, 3, 3), bold, shift=(0, 0.5))
    assert gm_x(half) != gm_x(base)                 # a quarter-voxel move changes the boundary voxels
    whole = aa.anatomy_compartments(ribbon, eye, fine, eye, (24, 3, 3), bold, shift=(0, 2.0))
    assert gm_x(whole) == [x + 1 for x in gm_x(base)]


def test_pial_edge_r_falls_as_the_run_shifts_off_the_boundary():
    gm, wm, out, x = _slab()
    c = aa.compartments(gm, wm, out)
    # A run whose only edge is the CSF edge. (With EPI's WM/GM contrast too, a
    # 3-voxel sheet under a 2-voxel sigma blends both edges into the band, so the
    # toy says nothing about real data; the driver's shift check does that.)
    run = 2.0 * out + 0.1
    r0 = aa.pial_edge_r(run, out, c.pial_band, np.ones(3))
    r1 = aa.pial_edge_r(np.roll(run, 1, axis=0), out, c.pial_band, np.ones(3))
    r2 = aa.pial_edge_r(np.roll(run, 2, axis=0), out, c.pial_band, np.ones(3))
    assert r0 == pytest.approx(1.0) and r0 > r1 > r2


def test_region_edge_r_matches_the_whole_band_when_one_group_covers_it():
    gm, wm, out, x = _slab()
    c = aa.compartments(gm, wm, out)
    run = 1.0 * out + 0.6 * gm + 0.3 * wm + 0.05 * np.sin(np.arange(out.size)).reshape(out.shape)
    labels = np.where(c.gm, 1, 0)
    band_labels = aa.nearest_labels(labels, c.pial_band)
    r = aa.region_edge_r(run, out, c.pial_band, np.ones(3), band_labels, {"all": [1], "none": [2]})
    assert r["all"] == pytest.approx(aa.pial_edge_r(run, out, c.pial_band, np.ones(3)))
    assert np.isnan(r["none"])


def test_ribbon_specificity_counts_and_paired_bootstrap():
    gm, wm, out, x = _slab()
    c = aa.compartments(gm, wm, out)
    labels = np.zeros(gm.shape, int)
    labels[c.gm & (np.indices(gm.shape)[1] < 3)] = 1
    labels[c.gm & (np.indices(gm.shape)[1] >= 3)] = 2
    keep = np.ones(gm.shape, bool)
    exceed = c.gm.copy()                       # every GM voxel beats the null...
    exceed[(x == 9)] = True                    # ...and one WM-band row of two
    counts = aa.parcel_counts(exceed, keep, c, labels)
    assert counts["parcel"].tolist() == [1, 2]
    s = aa.specificity(counts)
    assert s["x_gm"] == 3 * 36 and s["x_wm_band"] == 36 and s["n_wm_band"] == 2 * 36
    assert s["specificity"] == pytest.approx(0.75)
    assert s["rate_ratio"] == pytest.approx(1.0 / 0.5)
    assert s["csf_share"] == 0
    # Voxels outside keep count nowhere.
    assert aa.parcel_counts(exceed, np.zeros_like(keep), c, labels)[aa.COUNT_COLUMNS].to_numpy().sum() == 0

    better = aa.parcel_counts(c.gm, keep, c, labels)
    boot = aa.bootstrap_specificity({"A": counts, "B": better}, n_boot=50)
    assert list(boot.columns) == ["A", "B"] and (boot["B"] == 1.0).all()
    np.testing.assert_allclose(boot["A"], 0.75)
    with pytest.raises(ValueError, match="paired"):
        aa.bootstrap_specificity({"A": counts, "B": better.iloc[::-1]})


def test_network_and_component_groups_from_schaefer_names():
    names = pd.Series({1: "17Networks_LH_SomMotA_1", 2: "17Networks_LH_DefaultB_PFCd_1",
                       3: "17Networks_RH_DefaultB_PFCd_2", 4: "17Networks_RH_SomMotA_2"})
    assert aa.network_groups(names) == {"SomMotA": [1, 4], "DefaultB": [2, 3]}
    assert aa.component_group(names, "PFCd") == [2, 3]
    with pytest.raises(ValueError):
        aa.network_groups(pd.Series({1: "Background"}))
