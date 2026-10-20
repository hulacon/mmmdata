"""`mmmview check` (src/python/resultsview/check.py): the page contract and
result provenance, checked without rendering."""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

import mmmget
from resultsview import check
from tests.mmmget_fixtures import make_env
from tests.test_mmmget_results import SPEC
from tests.test_resultsview import spec, write_pair

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def run(tmp_path, monkeypatch):
    make_env(tmp_path, monkeypatch)
    (r,) = mmmget.runs(subject="01", task="memory", run=1)
    return r


def _write(run, out, stem, **confounds_kw):
    mmmget.reset_provenance()
    design = mmmget.confounds(run, **confounds_kw)
    # the columns= escape hatch returns a plain DataFrame, a regime a design
    n = design.shape[1] if isinstance(design, pd.DataFrame) else design.n_regressors
    table = pd.DataFrame([{"run": run.run, "n_regressors": n}])
    return mmmget.write_result(table, SPEC, out, stem)


def test_write_result_output_passes_as_standard(run, tmp_path):
    _write(run, tmp_path / "out", "regs", regime="base")
    report = check.check(tmp_path / "out")
    assert report["exit_status"] == check.EXIT_OK
    (r,) = report["results"]
    assert (r["ok"], r["provenance"], r["reasons"]) == (True, "standard", [])
    assert r["table"].endswith("regs.tsv")


def test_escape_hatch_is_non_standard_and_strict_fails(run, tmp_path):
    _write(run, tmp_path / "out", "regs", columns=["trans_x"])
    report = check.check(tmp_path / "out")
    (r,) = report["results"]
    assert r["ok"] and r["provenance"] == "non_standard" and r["reasons"]
    assert report["exit_status"] == check.EXIT_OK
    assert check.check(tmp_path / "out", strict=True)["exit_status"] == check.EXIT_FAIL


def test_hand_written_spec_has_absent_provenance(tmp_path):
    write_pair(tmp_path)
    (r,) = check.check(tmp_path)["results"]
    assert (r["ok"], r["provenance"]) == (True, "absent")


@pytest.mark.parametrize("broken, needle", [
    (spec(err_over=""), "err_over is required"),
    (spec(data={"values": [{"a": 1}]}), "carries its own data"),
    (spec(encoding={"x": {"field": "nope", "type": "nominal"}}), "nope"),
])
def test_contract_violation_fails_naming_the_fix(tmp_path, broken, needle):
    write_pair(tmp_path / "res", sp=broken)
    report = check.check(tmp_path)
    assert report["exit_status"] == check.EXIT_FAIL
    (r,) = report["results"]
    assert not r["ok"] and needle in r["error"]


def test_missing_table_fails(tmp_path):
    (tmp_path / "acc.vl.json").write_text(json.dumps(spec()))
    (r,) = check.check(tmp_path)["results"]
    assert not r["ok"] and "does not exist" in r["error"]


def test_one_bad_spec_fails_the_directory(tmp_path):
    write_pair(tmp_path, stem="good")
    write_pair(tmp_path, stem="bad", sp=spec(err_over=""))
    report = check.check(tmp_path)
    assert report["summary"] == {"specs": 2, "invalid": 1, "non_standard_provenance": 2}
    assert report["exit_status"] == check.EXIT_FAIL


def test_table_path_finds_its_spec(tmp_path):
    write_pair(tmp_path)
    (r,) = check.check(tmp_path / "acc.tsv")["results"]
    assert r["spec"].endswith("acc.vl.json")


def test_hidden_directories_are_skipped(tmp_path):
    write_pair(tmp_path / ".mmmget-tmp", sp=spec(err_over=""))
    write_pair(tmp_path / "res")
    assert check.check(tmp_path)["summary"]["specs"] == 1


def test_nothing_to_check_is_a_usage_error(tmp_path):
    assert check.check(tmp_path)["exit_status"] == check.EXIT_USAGE
    assert check.check(tmp_path / "absent")["exit_status"] == check.EXIT_USAGE
    assert "no .vl.json spec found" in check.render(check.check(tmp_path), "text")


def test_github_format_annotates_file(tmp_path):
    write_pair(tmp_path, sp=spec(err_over=""))
    out = check.render(check.check(tmp_path), "github")
    assert out.startswith(f"::error file={tmp_path / 'acc.vl.json'},title=mmmview contract::")
    assert "::warning" in out and "provenance" in out


def test_json_report_is_self_describing(tmp_path):
    write_pair(tmp_path)
    report = json.loads(check.render(check.check(tmp_path), "json"))
    assert report["schema_version"] == check.REPORT_SCHEMA_VERSION
    assert report["contract_schema_version"] == 1


def test_module_runs_with_only_src_on_the_path(tmp_path):
    """CI runs it from a bare checkout: `PYTHONPATH=src/python python -m resultsview.check`."""
    write_pair(tmp_path, sp=spec(err_over=""))
    proc = subprocess.run(
        [sys.executable, "-I", "-c",
         f"import sys; sys.path.insert(0, {str(REPO_ROOT / 'src/python')!r}); "
         "from resultsview import check; sys.exit(check.main(sys.argv[1:]))",
         str(tmp_path)], capture_output=True, text=True)
    assert proc.returncode == check.EXIT_FAIL, proc.stderr
    assert "FAIL" in proc.stdout and "err_over is required" in proc.stdout


def test_mmmview_dispatches_the_check_verb(tmp_path):
    write_pair(tmp_path)
    proc = subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "mmmview.py"),
                           "check", str(tmp_path)], capture_output=True, text=True)
    assert proc.returncode == check.EXIT_OK, proc.stderr
    assert "1 spec(s), 0 invalid" in proc.stdout
