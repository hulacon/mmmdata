"""data-quality views: every tier-2 part gets specs mmmview accepts, and the summaries are right."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from neuroimaging import confounds
from neuroimaging import data_quality_views as dqv
from resultsview import page

REPO = Path(__file__).resolve().parents[1]
REGIMES = list(dqv.REGIME_ORDER)
PARCELS = ["17Networks_LH_VisCent_ExStr_1", "17Networks_RH_DefaultA_PFCm_1", "17Networks_LH_DefaultB_Temp_1",
           "17Networks_RH_TempPar_2", "Left Hippocampus"]

# Headers of the tier-2 tables the views read (the columns every spec may name).
HEADERS = {
    "snr/task_summary": "scope task regime provisional n_runs n_absent tsnr_median tsnr_q25 tsnr_q75 dof_resid_median "
                        "dof_resid_min dof_loss_median n_regressors_median",
    "snr/sessions": "sub ses regime provisional tasks n_runs n_absent tsnr_median tsnr_rel_median n_runs_rel "
                    "dof_loss_median fd_mean_median fd_mean_max fdf_mean_median fd_frac_gt_0.2_max "
                    "mask_n_voxels_median cov_median_min n_cov_below_floor_max hipp_cov_min",
    "connectivity/qcfc": "regime fd_kind scope subjects n_runs dof n_edges median_abs_qcfc frac_p05 "
                         "distance_dependence provisional",
    "connectivity/fingerprint": "regime test scope subjects n_targets chance id_rate margin_median provisional",
    "connectivity/hipp_similarity": "sub seed regime_a regime_b n_parcels r",
    "univariate/task_r2": "scope task regime provisional n_runs n_absent frac_p001_median frac_p001_q25 "
                          "frac_p001_q75 r2adj_median r2adj_median_q25 r2adj_median_q75 r2adj_p99 r2adj_p99_q25 "
                          "r2adj_p99_q75 r2raw_median dof_resid_median n_regressors_median motion_task_r_median "
                          "motion_task_r_max",
    "univariate/split_half": "sub task contrast regime n_runs n_half1 n_half2 runs_half1 n_mask n_valid r dice@z3.1 "
                             "n@z3.1_half1 n@z3.1_half2 dice@250 dice@500 dice@1000 dice@2000 dice@4000",
    "glmsingle/fit_summary": "sub arm beta_type regime absent absent_reason provisional n_trials repeat_n_conditions "
                             "repeat_rep_scope r2_median r2_p90 repeat_frac_exceed anchor_frac_exceed",
    "glmsingle/networks": "sub arm beta_type atlas region n_parcels n_voxels_floor r2_median repeat_frac_exceed",
    "naturalistic/wsc": "stimulus_id regime sub parcel n_showings n_pairs n_vol z_mean r",
    "naturalistic/wsc_pairs": "stimulus_id regime sub ses_a ses_b n_vol r_median",
    "naturalistic/discriminability": "ses regime parcel within_r between_r discriminability n_within n_between",
    "naturalistic/loo_isc": "stimulus_id regime sub parcel isc",
    "naturalistic/envelope_lag": "stimulus_id regime sub ses run envelope roi n_parcels n_vol n_window lag r",
}

# Values for the columns the views filter or group on; every other column is numeric filler.
LABELS = {
    "scope": ["pooled_confirmed", "sub-03"], "task": ["all", "floc"], "regime": REGIMES, "sub": ["03", "04"],
    "ses": ["01", "02"], "fd_kind": ["raw", "filtered"], "test": ["subject", "session"], "seed": ["both"],
    "regime_a": REGIMES, "regime_b": REGIMES, "contrast": ["faceVsObject"], "arm": ["enc", "ret-word"],
    "beta_type": ["B", "D"], "atlas": ["Schaefer17n400", "HOSPA"], "region": ["LH_VisCent", "Left Hippocampus"],
    "stimulus_id": ["bench"], "parcel": PARCELS, "envelope": ["linear", "log"], "roi": ["SomMotB_Aud"],
    "lag": [-3, 0], "tasks": ["TBencoding"], "subjects": ["03"], "provisional": ["False"], "absent": ["False"],
    "absent_reason": ["n/a"], "repeat_rep_scope": ["within-session"], "runs_half1": ["03:01"], "run": ["01"],
    "ses_a": ["19"], "ses_b": ["21"],
}


def _write_part_tables(root: Path) -> Path:
    rng = np.random.default_rng(0)
    for name, header in HEADERS.items():
        cols = header.split()
        label_cols = [c for c in cols if c in LABELS]
        grid = pd.MultiIndex.from_product([LABELS[c] for c in label_cols], names=label_cols).to_frame(index=False)
        for c in cols:
            if c not in LABELS:
                grid[c] = rng.uniform(0.01, 0.9, len(grid)).round(4)
        path = root / f"{name}.tsv"
        path.parent.mkdir(parents=True, exist_ok=True)
        grid[cols].to_csv(path, sep="\t", index=False, na_rep="n/a")
    return root


@pytest.fixture
def tier2(tmp_path):
    return _write_part_tables(tmp_path / "tier2")


# ---------------------------------------------------------------------------
# Small pieces
# ---------------------------------------------------------------------------

def test_regime_order_covers_the_registry_exactly():
    assert sorted(dqv.REGIME_ORDER) == sorted(confounds.load_regimes())


@pytest.mark.parametrize("name,network", [
    ("17Networks_LH_VisCent_ExStr_1", "VisCent"),
    ("17Networks_RH_DefaultA_PFCm_1", "Default"),
    ("17Networks_LH_SalVentAttnB_PFCl_1", "SalVentAttn"),
    ("17Networks_RH_TempPar_2", "TempPar"),
    ("LH_ContC", "Cont"),                   # glmsingle networks.tsv region names
    ("Left Hippocampus", None),
    ("17Networks_LH_Unknown_1", None),
])
def test_network_of(name, network):
    assert dqv.network_of(name) == network


def test_wsc_networks_pools_abc_and_reports_every_subject_and_the_pool():
    wsc = pd.DataFrame({"stimulus_id": "bench", "regime": "none",
                        "sub": ["03", "03", "04", "04", "04"],
                        "parcel": ["17Networks_LH_DefaultA_x_1", "17Networks_LH_DefaultB_x_1",
                                   "17Networks_LH_DefaultC_x_1", "17Networks_LH_DefaultA_x_2", "Left Thalamus"],
                        "r": [0.1, 0.3, 0.2, 0.6, 0.9]})
    out = dqv.wsc_networks(wsc).set_index("scope")
    assert list(out.index) == ["pooled", "sub-03", "sub-04"]
    assert set(out["network"]) == {"Default"}                  # the subcortical row is not a network
    assert out.loc["pooled", "r_median"] == pytest.approx(0.25)
    assert out.loc["sub-04", "r_median"] == pytest.approx(0.4)
    assert out.loc["pooled", "n"] == 4


def test_envelope_lag_summary_is_median_and_iqr_over_showings():
    lag = pd.DataFrame({"stimulus_id": ["a", "b", "c", "d"], "regime": "none", "sub": ["03", "03", "04", "04"],
                        "envelope": "linear", "roi": "SomMotB_Aud", "lag": -3, "r": [0.1, 0.2, 0.3, 0.4]})
    out = dqv.envelope_lag_summary(lag).set_index("scope")
    assert out.loc["pooled", "r_median"] == pytest.approx(0.25)
    assert out.loc["pooled", "r_q25"] == pytest.approx(0.175)
    assert out.loc["pooled", "n_showings"] == 4
    assert out.loc["sub-04", "r_median"] == pytest.approx(0.35)


# ---------------------------------------------------------------------------
# Specs against the results-page contract
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("part", dqv.PARTS)
def test_every_spec_passes_the_results_contract(tier2, part):
    specs = dqv.write_views(part, tier2 / part)
    assert specs, part
    for s in specs:
        item = page.load_spec(s)                                # raises on a missing column or data
        assert item["rows"], s.name
        spec = json.loads(s.read_text())
        assert spec["title"] and spec["usermeta"]["mmmview"]["caption"]


def test_specs_are_numbered_in_page_order_and_rewrites_drop_stale_specs(tier2):
    d = tier2 / "snr"
    (d / "99_retired.vl.json").write_text("{}")
    names = [p.name for p in dqv.write_views("snr", d)]
    assert names == sorted(names) and names[0] == "00_tsnr_by_regime.vl.json"
    assert sorted(p.name for p in d.glob("*.vl.json")) == names


def test_tooltip_shorthand_becomes_field_definitions(tier2):
    for s in dqv.write_views("univariate", tier2 / "univariate"):
        text = s.read_text()
        spec = json.loads(text)

        def check(node):
            if isinstance(node, dict):
                for k, v in node.items():
                    if k == "tooltip" and isinstance(v, list):
                        assert all(isinstance(t, dict) for t in v), s.name
                    check(v)
            elif isinstance(node, list):
                for v in node:
                    check(v)
        check(spec)


def test_naturalistic_writes_its_summary_tables(tier2):
    dqv.write_views("naturalistic", tier2 / "naturalistic")
    for name in dqv.NATURALISTIC_SUMMARIES:
        assert (tier2 / "naturalistic" / f"{name}.tsv").exists()


def test_selector_defaults_fall_back_to_a_present_value(tier2):
    # the fixture has no `pooled` scope in task_r2; the univariate default is pooled_confirmed, present
    spec = json.loads(dqv.write_views("univariate", tier2 / "univariate")[0].read_text())
    (param,) = spec["params"]
    assert param["value"] in param["bind"]["options"]
    assert dqv._select("x", ["a", "b"], "zzz", "x")["value"] == "a"


# ---------------------------------------------------------------------------
# The driver's views verb
# ---------------------------------------------------------------------------

def _tier2_script():
    spec = importlib.util.spec_from_file_location("tier2_script", REPO / "scripts" / "data_quality" / "tier2.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_views_verb_writes_every_part(tier2, capsys):
    _tier2_script().main(["views", "--out-dir", str(tier2)])
    out = capsys.readouterr().out
    for part in dqv.PARTS:
        assert f"mmmview {tier2 / part}" in out
        assert list((tier2 / part).glob("*.vl.json"))


def test_views_verb_names_the_build_for_a_missing_part(tmp_path):
    with pytest.raises(SystemExit, match="tier2.py build --parts snr"):
        _tier2_script().main(["views", "--parts", "snr", "--out-dir", str(tmp_path)])
