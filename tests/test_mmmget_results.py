"""mmmget.write_result(): mmmview-valid output carrying getter provenance."""

import json
import re

import pandas as pd
import pytest

import mmmget
from resultsview import page
from tests.mmmget_fixtures import make_env

SPEC = {
    "mark": "point",
    "encoding": {"x": {"field": "run", "type": "nominal"},
                 "y": {"field": "n_regressors", "type": "quantitative"}},
    "usermeta": {"mmmview": {"err_over": "none"}},
}


@pytest.fixture
def run(tmp_path, monkeypatch):
    make_env(tmp_path, monkeypatch)
    (r,) = mmmget.runs(subject="01", task="memory", run=1)
    return r


def _table(run, design):
    return pd.DataFrame([{"sub": run.subject, "run": run.run,
                          "n_regressors": design.n_regressors, "note": None}])


def test_output_is_accepted_by_resultsview(run, tmp_path):
    design = mmmget.confounds(run, regime="base")
    tsv, vl = mmmget.write_result(_table(run, design), SPEC, tmp_path / "out", "regs")
    assert (tsv.name, vl.name) == ("regs.tsv", "regs.vl.json")
    loaded = page.load_spec(vl)
    assert loaded["columns"] == ["sub", "run", "n_regressors", "note"]
    # labels with leading zeros stay strings; missing reads as missing
    assert loaded["rows"] == [{"sub": "01", "run": "01", "n_regressors": 8, "note": None}]
    if all((page.VENDOR_DIR / n).is_file() for n in page.VENDOR_JS):
        page.build_results_page([vl], tmp_path / "out" / "page.html")


def test_standard_provenance_is_stamped(run, tmp_path):
    mmmget.confounds(run, regime="base")
    _, vl = mmmget.write_result([{"run": "01", "n_regressors": 8}], SPEC, tmp_path, "regs")
    prov = json.loads(vl.read_text())["usermeta"]["mmmview"]["provenance"]
    assert prov["mmmget_version"] == mmmget.__version__
    assert re.fullmatch(r"\d+\.\d+\.\d+", mmmget.__version__)
    assert prov["standard"] is True and prov["non_standard"] == []
    assert [g["getter"] for g in prov["getters"]] == ["runs", "confounds"]
    assert prov["getters"][1]["regime"] == "base"


def test_escape_hatch_marks_result_non_standard(run, tmp_path):
    mmmget.confounds(run, columns=["trans_x"])
    _, vl = mmmget.write_result([{"run": "01", "n_regressors": 1}], SPEC, tmp_path, "regs")
    prov = json.loads(vl.read_text())["usermeta"]["mmmview"]["provenance"]
    assert prov["standard"] is False
    assert prov["non_standard"] == ["confounds: raw columns instead of a registry regime"]


def test_no_getters_is_not_standard(tmp_path, monkeypatch):
    make_env(tmp_path, monkeypatch)
    _, vl = mmmget.write_result([{"run": "01", "n_regressors": 1}], SPEC, tmp_path, "regs")
    prov = json.loads(vl.read_text())["usermeta"]["mmmview"]["provenance"]
    assert prov["standard"] is False and "no mmmget getter" in prov["non_standard"][0]


def test_caller_spec_is_not_mutated(run, tmp_path):
    before = json.dumps(SPEC, sort_keys=True)
    mmmget.write_result([{"run": "01", "n_regressors": 1}], SPEC, tmp_path, "regs")
    assert json.dumps(SPEC, sort_keys=True) == before


@pytest.mark.parametrize("spec, match", [
    ({**SPEC, "data": {"values": [{"run": 1}]}}, "single source"),
    ({**SPEC, "usermeta": {"mmmview": {}}}, "err_over is required"),
    ({**SPEC, "encoding": {"x": {"field": "rn", "type": "nominal"}}}, "rn"),
    ({**SPEC, "usermeta": {"mmmview": {"err_over": "none", "table": "other.tsv"}}}, "other.tsv"),
])
def test_contract_violation_raises_and_writes_nothing(run, tmp_path, spec, match):
    out = tmp_path / "out"
    with pytest.raises(page.SpecError, match=match):
        mmmget.write_result([{"run": "01", "n_regressors": 1}], spec, out, "regs")
    assert not out.exists() or not list(out.iterdir())


def test_failed_rewrite_keeps_the_previous_result(run, tmp_path):
    tsv, vl = mmmget.write_result([{"run": "01", "n_regressors": 1}], SPEC, tmp_path, "regs")
    before = (tsv.read_text(), vl.read_text())
    with pytest.raises(page.SpecError):
        mmmget.write_result([{"run": "01", "n_regressors": 2}],
                            {**SPEC, "data": {"values": []}}, tmp_path, "regs")
    assert (tsv.read_text(), vl.read_text()) == before


def test_tab_in_a_cell_is_refused(run, tmp_path):
    with pytest.raises(page.SpecError, match="tab or newline"):
        mmmget.write_result([{"run": "a\tb", "n_regressors": 1}], SPEC, tmp_path, "regs")


def test_stem_must_be_bare(run, tmp_path):
    with pytest.raises(ValueError, match="bare name"):
        mmmget.write_result([{"run": "01", "n_regressors": 1}], SPEC, tmp_path, "regs.tsv")
