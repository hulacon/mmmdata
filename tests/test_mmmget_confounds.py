"""mmmget.confounds(): regimes through the registry; D3 surfaced; escape hatch marked."""

import dataclasses

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


def test_standard_set_raises_naming_d3(run):
    with pytest.raises(mmmget.OpenDecisionError, match="open decision D3") as exc:
        mmmget.confounds(run, regimes="standard")
    assert exc.value.decision == "D3"
    assert "base" in str(exc.value)  # names what the registry does have


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
