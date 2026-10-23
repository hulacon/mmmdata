"""mmmget.testing: the requires_dataset plugin project repos load.

Each case runs a throwaway project's tests in an inner pytest (pytester), with
the dataset root pointed at a directory that does or does not exist, the way
CI and a cluster session differ.
"""

import pytest

pytest_plugins = ["pytester"]

PROJECT_TESTS = """
import pytest

@pytest.mark.requires_dataset
def test_reads_data():
    pass

def test_pure():
    pass
"""


@pytest.fixture
def project(pytester):
    pytester.makepyfile(test_project=PROJECT_TESTS)
    return pytester


def test_marked_tests_skip_without_a_dataset(project, tmp_path, monkeypatch):
    monkeypatch.setenv("MMMDATA_BIDS_ROOT", str(tmp_path / "absent"))
    result = project.runpytest("-p", "mmmget.testing", "--strict-markers", "-rs")
    result.assert_outcomes(passed=1, skipped=1)
    result.stdout.fnmatch_lines(["*no MMMData dataset reachable here*"])


def test_marked_tests_run_with_a_dataset(project, tmp_path, monkeypatch):
    (tmp_path / "copy").mkdir()
    monkeypatch.setenv("MMMDATA_BIDS_ROOT", str(tmp_path / "copy"))
    result = project.runpytest("-p", "mmmget.testing", "--strict-markers")
    result.assert_outcomes(passed=2)


def test_loading_from_conftest_and_command_line_is_harmless(project, tmp_path,
                                                             monkeypatch):
    """The shared checks pass -p; a project's conftest may also name it."""
    project.makeconftest('pytest_plugins = ["mmmget.testing"]')
    monkeypatch.setenv("MMMDATA_BIDS_ROOT", str(tmp_path / "absent"))
    result = project.runpytest("-p", "mmmget.testing", "--strict-markers")
    result.assert_outcomes(passed=1, skipped=1)
