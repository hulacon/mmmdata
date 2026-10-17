"""run_all.py --validate: convert, then validate, with the right arguments.

validate.main() used to parse run_all's own sys.argv, so --validate (and
any --dry-run or --tasks beside it) reached a parser that rejects them and
every --validate run exited with argparse status 2, doing neither step.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CONVERTERS = REPO_ROOT / "src" / "python" / "raw2bids_converters"
if str(CONVERTERS) not in sys.path:
    sys.path.insert(0, str(CONVERTERS))

from raw2bids_converters import run_all  # noqa: E402

validate = run_all.validate


@pytest.fixture
def no_inventory(monkeypatch):
    monkeypatch.setattr(run_all, "load_inventory", lambda: [])


def test_validate_runs_after_conversion_with_its_own_arguments(no_inventory, monkeypatch):
    calls = []
    monkeypatch.setattr(validate, "main", lambda argv=None: calls.append(argv) or 0)

    rc = run_all.main(["--dry-run", "--validate", "--tasks", "timed_events",
                       "--subjects", "sub-##"])

    assert rc == 0
    assert calls == [["--subjects", "sub-##"]]


def test_verbose_is_forwarded(no_inventory, monkeypatch):
    calls = []
    monkeypatch.setattr(validate, "main", lambda argv=None: calls.append(argv) or 0)

    run_all.main(["--validate", "--verbose", "--dry-run"])

    assert calls == [["--verbose"]]


def test_a_failed_validation_fails_the_run(no_inventory, monkeypatch):
    monkeypatch.setattr(validate, "main", lambda argv=None: 1)
    assert run_all.main(["--dry-run", "--validate"]) == 1


def test_no_validation_without_the_flag(no_inventory, monkeypatch):
    def fail(argv=None):
        raise AssertionError("validate ran without --validate")

    monkeypatch.setattr(validate, "main", fail)
    assert run_all.main(["--dry-run"]) == 0


def test_validate_main_takes_an_argv(monkeypatch):
    """The parser reads the list it is given, not the caller's sys.argv."""
    seen = []
    monkeypatch.setattr(validate, "validate_subject",
                        lambda sub, tol, verbose: seen.append(sub) or [_result("MATCH")])
    monkeypatch.setattr(sys, "argv", ["run_all.py", "--validate", "--dry-run"])

    assert validate.main(["--subjects", "sub-07"]) == 0
    assert seen == [7]


# validate.main used to return 0 unless a pair MISMATCHED. Its filename
# patterns can miss every generated file (task labels renamed), and then it
# compared nothing and passed, which made `run_all --validate` a false pass.

def _result(status):
    return {"generated": "gen.tsv" if status != "MISSING_GENERATED" else None,
            "reference": "ref.tsv" if status != "MISSING_REFERENCE" else None,
            "status": status, "issues": []}


def test_validate_fails_when_nothing_was_compared(monkeypatch, capsys):
    monkeypatch.setattr(validate, "validate_subject", lambda sub, tol, verbose: [])

    assert validate.main(["--subjects", "07"]) != 0
    assert "compared no files" in capsys.readouterr().out


def test_validate_fails_when_only_unmatched_references_were_found(monkeypatch):
    monkeypatch.setattr(validate, "validate_subject",
                        lambda sub, tol, verbose: [_result("MISSING_REFERENCE")])

    assert validate.main(["--subjects", "07"]) != 0


def test_validate_fails_when_a_reference_has_no_generated_file(monkeypatch):
    monkeypatch.setattr(validate, "validate_subject",
                        lambda sub, tol, verbose: [_result("MATCH"),
                                                   _result("MISSING_GENERATED")])

    assert validate.main(["--subjects", "07"]) != 0


def test_validate_passes_when_every_compared_pair_matches(monkeypatch):
    monkeypatch.setattr(validate, "validate_subject",
                        lambda sub, tol, verbose: [_result("MATCH"),
                                                   _result("MISSING_REFERENCE")])

    assert validate.main(["--subjects", "07"]) == 0
