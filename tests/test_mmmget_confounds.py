"""mmmget.confounds(): regimes through the registry; the D3 comparison set; escape hatch marked."""

import dataclasses

import numpy as np
import pandas as pd
import pytest

import mmmget
from neuroimaging import confounds as registry
from tests.mmmget_fixtures import N_VOL, make_env


@pytest.fixture
def run(tmp_path, monkeypatch):
    make_env(tmp_path, monkeypatch)
    (r,) = mmmget.runs(subject="01", task="memory", run=1)
    mmmget.reset_provenance()
    return r


def test_regime_goes_through_registry(run):
    design = mmmget.confounds(run, regime="base")
    assert isinstance(design, registry.RegimeDesign)
    assert list(design.columns) == ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y",
                                    "rot_z", "poly_1", "poly_2"]
    assert design.n_vol == N_VOL and design.n_nss == 1
    (rec,) = mmmget.provenance_records()
    assert rec == {"getter": "confounds", "standard": True, "regime": "base",
                   "regime_version": registry.get_regime("base").version,
                   "regime_status": "defined"}


def test_several_regimes_by_name(run):
    out = mmmget.confounds(run, regimes=["base", "baseacc6"])
    assert set(out) == {"base", "baseacc6"}
    assert out["baseacc6"].n_regressors == 6 + 6 + 2


STANDARD = ["none", "base", "basecsfwm", "baseacc6", "base12fdacc20", "gsr"]


def _full_confounds(run, n_acompcor, n_vol=80):
    """Every column the standard set reads, with *n_acompcor* aCompCor components."""
    motion = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]
    cols = (motion + [f"{m}_derivative1" for m in motion]
            + ["framewise_displacement", "csf", "white_matter", "global_signal",
               "non_steady_state_outlier00"]
            + [f"a_comp_cor_{i:02d}" for i in range(n_acompcor)])
    df = pd.DataFrame(np.random.default_rng(0).normal(size=(n_vol, len(cols))), columns=cols)
    df["non_steady_state_outlier00"] = [1] + [0] * (n_vol - 1)
    df.to_csv(run.confounds, sep="\t", index=False, na_rep="n/a")


def test_standard_set_is_the_d3_decision():
    cset = registry.get_comparison_set("standard")
    assert list(cset.regimes) == STANDARD


def test_standard_set_returns_every_member(run):
    _full_confounds(run, n_acompcor=20)
    out = mmmget.confounds(run, regimes="standard")
    assert list(out) == STANDARD and out.absent == {}
    assert out.comparison_set == "standard"
    assert out.version == registry.get_comparison_set("standard").version
    set_rec = [r for r in mmmget.provenance_records() if "comparison_set" in r]
    assert set_rec == [{"getter": "confounds", "standard": True, "comparison_set": "standard",
                        "comparison_set_version": out.version}]
    assert {r.get("regime") for r in mmmget.provenance_records()} >= set(STANDARD)


def test_member_the_run_cannot_carry_is_declared_absent(run):
    _full_confounds(run, n_acompcor=6)  # base12fdacc20 needs 20
    out = mmmget.confounds(run, regimes="standard")
    assert "base12fdacc20" not in out and list(out.absent) == ["base12fdacc20"]
    assert "20" in out.absent["base12fdacc20"]
    assert out.members == tuple(STANDARD)
    (set_rec,) = [r for r in mmmget.provenance_records() if "comparison_set" in r]
    assert set_rec["standard"] is True and set_rec["absent_regimes"] == ["base12fdacc20"]


def test_unknown_comparison_set(run):
    with pytest.raises(KeyError, match="comparison set 'nope'"):
        mmmget.confounds(run, regimes="nope")


def test_columns_escape_hatch_works_and_is_non_standard(run):
    df = mmmget.confounds(run, columns=["trans_x", "framewise_displacement"])
    assert list(df.columns) == ["trans_x", "framewise_displacement"]
    (rec,) = mmmget.provenance_records()
    assert rec["standard"] is False and rec["columns"] == ["trans_x", "framewise_displacement"]
    assert "raw columns" in rec["reason"]


def test_provisional_regime_is_non_standard(run, monkeypatch):
    prov = dataclasses.replace(registry.get_regime("base"), name="trial", status="provisional")
    monkeypatch.setattr(registry, "get_regime", lambda name: prov)
    mmmget.confounds(run, regime="trial")
    (rec,) = mmmget.provenance_records()
    assert rec["standard"] is False and rec["reason"] == "provisional regime"


def test_exactly_one_selector(run):
    with pytest.raises(TypeError):
        mmmget.confounds(run)
    with pytest.raises(TypeError):
        mmmget.confounds(run, regime="base", columns=["trans_x"])


def test_unknown_regime(run):
    with pytest.raises(KeyError, match="registry has"):
        mmmget.confounds(run, regime="nope")
