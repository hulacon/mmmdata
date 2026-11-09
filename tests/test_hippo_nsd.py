"""Pure helpers of neuroimaging.hippo_nsd (no ANTs)."""

import math

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from neuroimaging import hippo_compare as hc
from neuroimaging import hippo_nsd as hn

U = next(a for a in hn.ARMS if a.name == "U2iso")
FSSF = next(a for a in hn.ARMS if a.name == "FSsf")
ASEG = next(a for a in hn.ARMS if a.name == "FSaseg")


def test_decode_manual_parity_and_ctab():
    # raw 7/8 = SUB (ctab 4), 9/10 = CA1, 19/20 = HT, 1/2 = ERC (background); odd = R, even = L
    raw = np.array([0, 7, 8, 9, 10, 19, 20, 1, 2])
    out = hn.decode_manual(raw)
    names = lambda v: [hn.CANON[int(c)] for c in v]  # noqa: E731
    assert names(out["R"]) == ["bg", "Sub", "bg", "CA1", "bg", "hipp", "bg", "bg", "bg"]
    assert names(out["L"]) == ["bg", "bg", "Sub", "bg", "CA1", "bg", "hipp", "bg", "bg"]


def test_decode_manual_all_hippocampal_codes():
    raw = np.arange(1, 21)
    both = hn.decode_manual(raw)["L"] + hn.decode_manual(raw)["R"]
    # ctab 4..8 -> Sub CA1 CA2 CA3 DG, 10 -> hipp; ctab 1,2,3,9 -> bg
    expect = {4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 10: hn.HIPP}
    for k in range(1, 21):
        assert both[k - 1] == expect.get((k + 1) // 2, 0)


def test_decode_manual_midline_overrides_parity():
    # x = i (identity affine). Right cluster (odd codes) at x 20-22, left (even) at x 0-2.
    raw = np.zeros((23, 1, 1), np.int64)
    raw[20:23, 0, 0] = 9          # CA1 right
    raw[0:3, 0, 0] = 10           # CA1 left
    raw[1, 0, 0] = 13             # CA3 coded right (odd) but lying in the left cluster: a parity slip
    aff = np.eye(4)
    is_right, n = hn.manual_hemisphere(raw, aff)
    assert n == 1 and not is_right[1, 0, 0] and is_right[20:23, 0, 0].all()
    out = hn.decode_manual(raw, aff)
    assert out["L"][1, 0, 0] == 4 and out["R"][1, 0, 0] == 0      # CA3 stays CA3, moves to the left
    # without the affine the old parity rule applies
    par = hn.decode_manual(raw)
    assert par["R"][1, 0, 0] == 4 and par["L"][1, 0, 0] == 0


def test_manual_hemisphere_follows_orientation_and_needs_both_parities():
    raw = np.zeros((10, 1, 1), np.int64)
    raw[0, 0, 0], raw[9, 0, 0] = 7, 8                       # right at low i, left at high i
    flip = np.diag([-1.0, 1, 1, 1])                         # x = -i: right is still +x
    is_right, n = hn.manual_hemisphere(raw, flip)
    assert is_right[0, 0, 0] and not is_right[9, 0, 0] and n == 0
    with pytest.raises(ValueError):
        hn.manual_hemisphere(np.where(raw == 7, 7, 0), np.eye(4))


def test_collapse_primary_and_alt():
    canon = np.zeros((7, 1, 1), np.uint8)
    canon[:, 0, 0] = [0, 2, hn.SRLM, hn.SRLM, 5, hn.CA4, 0]
    p = hn.collapse(canon, "primary")[:, 0, 0]
    assert list(p) == [0, 2, hn.SRLM, hn.SRLM, 5, 5, 0]
    a = hn.collapse(canon, "alt")[:, 0, 0]
    # SRLM at index 2 is nearest CA1 (index 1), at index 3 nearest DG (index 4); CA4 -> CA3
    assert list(a) == [0, 2, 2, 5, 5, 4, 0]
    with pytest.raises(ValueError):
        hn.collapse(canon, "nope")


def test_express_merges_ca2_ca3_only_for_arms_without_them():
    lab = np.array([1, 3, 4, 9])
    assert list(hn.express(lab, U)) == [1, 3, 4, 9]
    assert list(hn.express(lab, FSSF)) == [1, 9, 9, 9]


def test_subfield_domain_slices_minus_tail():
    man = np.zeros((2, 2, 5), np.uint8)
    man[0, 0, 1] = 2           # CA1 in slice 1
    man[1, 1, 2] = 5           # DG in slice 2
    man[0, 1, 2] = hn.HIPP     # tail voxel in a domain slice
    man[0, 0, 4] = hn.HIPP     # tail-only slice
    dom = hn.subfield_domain(man, slice_axis=2)
    assert dom[:, :, 1].all() and not dom[:, :, 0].any() and not dom[:, :, 4].any()
    assert not dom[0, 1, 2] and dom[1, 1, 2]


def test_supersample_nests_exactly_and_block_counts_agree_with_coarse_map():
    aff = np.array([[-0.4, 0, 0, 10], [0, 0, 1.5, -20], [0, -0.4, 0, 5], [0, 0, 0, 1]], float)
    shape, sub = (3, 4, 2), (2, 2, 5)
    faff, fshape = hn.supersample(aff, shape, sub)
    assert fshape == (6, 8, 10)
    cm = hc.coarse_map(fshape, faff, shape, aff)
    assert len(cm.flat) == np.prod(shape)
    assert np.all(np.bincount(cm.group) == np.prod(sub))
    rng = np.random.default_rng(0)
    fine = rng.integers(0, 4, fshape).astype(np.uint8)
    bc = hn.block_counts(fine, sub, n_classes=4)
    mc = hn.map_counts(fine, cm, n_classes=4)
    assert np.array_equal(bc.reshape(-1, 4)[cm.flat], mc)
    # plurality ties go to the lower class
    assert hn.counts_plurality(np.array([[0, 3, 3, 1]])).tolist() == [1]


def test_score_rows_whole_subfield_and_srlm_drop():
    man = np.array([1, 1, 2, 2, 5, 0, hn.HIPP])
    arm = np.array([1, 2, 2, hn.SRLM, 5, 5, 0])
    dom = np.array([1, 1, 1, 1, 1, 1, 0], bool)
    rows = {(r["metric"], r["region"]): r for r in hn.score_rows(man, arm, dom, U)}
    assert rows[("dice", "hipp")]["value"] == pytest.approx(2 * 5 / (6 + 6))
    assert rows[("dice", "hippNoSRLM")]["value"] == pytest.approx(2 * 4 / (5 + 6))
    # index 3 (arm SRLM) dropped: CA1 arm {1,2}, manual {2} -> 2*1/3
    assert rows[("dice", "CA1")]["value"] == pytest.approx(2 / 3)
    assert rows[("dice", "Sub")]["value"] == pytest.approx(2 / 3)
    assert rows[("sub_coverage", "Sub")]["value"] == pytest.approx(0.5)
    assert math.isnan(rows[("dice", "CA2")]["value"])
    aseg_rows = hn.score_rows(man, np.where(arm > 0, hn.HIPP, 0), dom, ASEG)
    assert {r["region"] for r in aseg_rows} == {"hipp", "hippNoSRLM"}


def test_score_rows_fov_restricts_whole_only():
    man = np.array([1, 1, 0])
    arm = np.array([1, 1, 1])
    fov = np.array([1, 1, 0], bool)
    rows = {(r["metric"], r["region"]): r for r in hn.score_rows(man, arm, np.ones(3, bool), U, fov=fov)}
    assert rows[("dice", "hipp")]["value"] == 1.0


def test_gate_row_tissue_share():
    man = np.array([1, 2, 0, 5, 2, 3])
    arm = np.array([2, 0, 1, hn.SRLM, 2, 4])
    dom = np.ones(6, bool)
    g = hn.gate_row(man, arm, dom, U)
    # differ: idx0 (Sub vs CA1, boundary), idx1 (tissue), idx2 (tissue), idx5 (CA2 vs CA3, boundary); idx3 dropped
    assert (g["n_differ"], g["n_tissue"]) == (4, 2)
    assert g["tissue_share"] == pytest.approx(0.5)
    # FS-sf expresses CA23 only: CA2 vs CA3 is agreement
    g2 = hn.gate_row(man, np.where(arm == 4, 9, arm), dom, FSSF)
    assert g2["n_differ"] == 3


def test_assignment_and_purity():
    man_co = np.array([1, 2, 2, 0])
    arm_co = np.array([1, 2, 5, 5])
    keep = np.ones(4, bool)
    r = hn.assignment_row(man_co, arm_co, keep, U)
    assert (r["n_either"], r["agree"]) == (4, pytest.approx(0.5))
    counts = np.zeros((4, hn.N_CLASSES), int)
    counts[0, 1], counts[0, 0] = 3, 1      # voxel 0: 3/4 Sub
    counts[1, 2] = 4                       # voxel 1: all CA1
    counts[2, 2], counts[2, 5] = 3, 1      # voxel 2: 1/4 DG
    counts[3, 0] = 4                       # voxel 3: background
    rows = {x["region"]: x for x in hn.purity_rows(counts, arm_co, man_co, keep, U)}
    assert rows["Sub"]["purity"] == pytest.approx(0.75)
    assert rows["CA1"]["purity"] == 1.0
    assert rows["DG"]["purity"] == pytest.approx((0.25 + 0.0) / 2)
    assert rows["DG"]["n_vox_arm"] == 2 and rows["DG"]["n_vox_manual"] == 0
    assert math.isnan(rows["CA2"]["purity"])


def test_bold_affine_offsets_and_cover():
    t1 = np.diag([0.8, 0.8, 0.8, 1.0])
    t1[:3, 3] = [-100, -120, -90]
    b0 = hn.bold_affine(t1, (1.8, 1.8, 2.0), (0, 0, 0))
    bh = hn.bold_affine(t1, (1.8, 1.8, 2.0), (0.5, 0.5, 0.5))
    assert np.allclose(np.diag(b0)[:3], [1.8, 1.8, 2.0])
    assert np.allclose(bh[:3, 3] - b0[:3, 3], [0.9, 0.9, 1.0])
    faff, fshape = hc.box_affine(np.ones((2, 2, 2), bool), t1, 0.3, 3.0)
    caff, cshape = hn.coarse_grid_over(fshape, faff, bh)
    cm = hc.coarse_map(fshape, faff, cshape, caff)
    assert len(cm.keep) == np.prod(fshape)          # every fine voxel lands inside the coarse grid
    # whole-voxel shift only: the lattice phase is preserved
    shift = np.linalg.inv(bh) @ caff
    assert np.allclose(shift[:3, 3], np.round(shift[:3, 3]))


def test_nsd_subject_label():
    assert hn.nsd_subject("01") == "subj01"
    assert hn.nsd_subject("8") == "subj08"


# --- group read (rules 6-8) ---------------------------------------------------

HEMIS16 = [(f"{s:02d}", h) for s in range(1, 9) for h in ("L", "R")]


def _long(values: dict, variant="primary", metric="dice_hrt2_hipp", grid="hrt2"):
    """values: arm -> 16 numbers (None = missing)."""
    rows = []
    for arm, vs in values.items():
        for (s, h), v in zip(HEMIS16, vs):
            if v is not None:
                rows.append({"subject": s, "hemi": h, "arm": arm, "variant": variant, "grid": grid, "metric": metric, "value": v})
    return pd.DataFrame(rows)


def test_pairwise_counts_strict_wins_and_statuses():
    a = [0.9] * 13 + [0.1] * 3
    b = [0.5] * 16
    c = [0.9] * 12 + [0.1] * 4
    pw = hn.pairwise(_long({"A": a, "B": b, "C": c})).set_index(["a", "b"])
    assert pw.loc[("A", "B"), "wins"] == 13 and pw.loc[("A", "B"), "status"] == "beats"
    assert pw.loc[("C", "B"), "wins"] == 12 and pw.loc[("C", "B"), "status"] == "tied"
    # equal values are not wins for either side
    assert pw.loc[("A", "C"), "wins"] == 1 and pw.loc[("C", "A"), "wins"] == 0


def test_pairwise_interim_open_and_cannot():
    a = [0.9] * 10 + [None] * 6
    b = [0.5] * 10 + [None] * 6
    c = [0.1] * 4 + [0.9] * 6 + [None] * 6
    pw = hn.pairwise(_long({"A": a, "B": b, "C": c})).set_index(["a", "b"])
    assert pw.loc[("A", "B"), "status"] == "open"     # 10 wins + 6 outstanding can reach 13
    assert pw.loc[("B", "A"), "status"] == "cannot"   # 0 + 6 < 13
    assert pw.loc[("C", "B"), "status"] == "cannot"   # 6 wins + 6 outstanding = 12 < 13
    assert pw.loc[("B", "C"), "wins"] == 4


def test_leaders_rule8_margin_and_reversal():
    lead = [0.9] * 13 + [0.4] * 3       # beats B by exactly 13 -> flagged
    run = [0.5] * 16
    prim = _long({"A": lead, "B": run})
    alt = _long({"A": [0.5] * 16, "B": [0.6] * 16}, variant="alt")  # B ahead under alt -> reversed
    long = pd.concat([prim, alt], ignore_index=True)
    out = hn.leaders(long, hn.pairwise(long)).iloc[0]
    assert out["leader_primary"] == "A" and out["runner_up_primary"] == "B"
    assert out["min_wins_primary"] == 13 and out["min_wins_vs_primary"] == "B"
    assert out["beats_all"] and out["rule8_flag"]
    assert out["top2_reversed"] and out["leader_changed"]


def test_leaders_clear_lead_not_flagged():
    long = pd.concat([_long({"A": [0.9] * 16, "B": [0.5] * 16}),
                      _long({"A": [0.9] * 16, "B": [0.5] * 16}, variant="alt")], ignore_index=True)
    out = hn.leaders(long, hn.pairwise(long)).iloc[0]
    assert out["min_wins_primary"] == 16 and not out["rule8_flag"] and not out["top2_reversed"]


def _tables():
    sc, asg, pur, gate = [], [], [], []
    for s, h in HEMIS16:
        for arm, base in (("Hhr", 0.8), ("U1iso", 0.7)):
            for var in hn.VARIANTS:
                sc.append({"subject": s, "hemi": h, "arm": arm, "variant": var, "grid": "hrt2", "offset": "", "metric": "dice",
                               "region": "hipp", "value": base})
                for k, off in enumerate(("000", "001")):  # bold: mean over offsets
                    sc.append({"subject": s, "hemi": h, "arm": arm, "variant": var, "grid": "bold", "offset": off, "metric": "dice",
                                   "region": "CA1", "value": base + (0.1 if k else -0.1)})
                    asg.append({"subject": s, "hemi": h, "arm": arm, "variant": var, "grid": "bold", "offset": off, "n_either": 10,
                                    "agree": base})
                    pur.append({"subject": s, "hemi": h, "arm": arm, "variant": var, "offset": off, "region": "DG", "n_vox_arm": 5,
                                    "n_vox_manual": 5, "purity": base})
                sc.append({"subject": s, "hemi": h, "arm": arm, "variant": var, "grid": "hrt2", "offset": "",
                               "metric": "sub_coverage", "region": "Sub", "value": base})
                gate.append({"subject": s, "hemi": h, "arm": arm, "variant": var, "n_differ": 10, "n_tissue": 6, "tissue_share": 0.6})
    return {"scores": pd.DataFrame(sc), "assignment": pd.DataFrame(asg), "purity": pd.DataFrame(pur),
            "gate": pd.DataFrame(gate)}


def test_per_hemisphere_metric_names_and_offset_mean():
    long = hn.per_hemisphere(_tables())
    assert set(long["metric"]) == {"dice_hrt2_hipp", "dice_bold_CA1", "assign_bold", "purity_bold_DG",
                                   "sub_coverage_hrt2"}
    one = long[(long["metric"] == "dice_bold_CA1") & (long["arm"] == "Hhr") & (long["variant"] == "primary")]
    assert len(one) == 16 and np.allclose(one["value"], 0.8)


def test_group_read_and_figures(tmp_path):
    read = hn.group_read(_tables())
    assert set(read) >= {"perhemi", "summary", "pairwise", "leaders", "gate"}
    assert (read["leaders"]["leader_primary"] == "Hhr").all()
    assert read["gate"].set_index("arm").loc["Hhr", "median"] == pytest.approx(0.6)
    figs = hn.group_figures(read, tmp_path, headline=("dice_hrt2_hipp", "assign_bold"),
                            paired=(("Hhr", "U1iso"),), paired_metrics=("dice_hrt2_hipp",),
                            rule7_metrics=("dice_hrt2_hipp", "assign_bold"))
    assert [f.name for f in figs] == ["group_desc-headline_hippnsd.png", "group_desc-rule7_hippnsd.png",
                                      "group_desc-paired_hippnsd.png"]
    assert all(f.stat().st_size > 0 for f in figs)
