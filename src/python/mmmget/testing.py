"""A pytest plugin for project repos that test code built on mmmget.

A project's tests run in two places: next to the dataset (on the cluster, or
on a copy) and in CI, where there is no dataset at all. A test that reads the
data says so with a marker, and skips itself, with the reason, wherever the
dataset is not reachable:

    import pytest

    @pytest.mark.requires_dataset
    def test_every_run_has_confounds():
        for run in mmmget.runs(task="<label>"):
            ...

Load the plugin from the project's ``conftest.py``:

    pytest_plugins = ["mmmget.testing"]

or on the command line (``pytest -p mmmget.testing``), which is how the lab's
shared checks run it; loading it both ways is harmless. "Reachable" means the
configured BIDS root (``MMMDATA_BIDS_ROOT``, ``config/local.toml``, or
``base.toml``) is a directory, so a copy of the dataset counts. Unmarked tests
always run, and a test that reads the data without the marker fails where the
data is absent: the marker is the only thing that turns that into a skip.
"""

from __future__ import annotations

from pathlib import Path

import pytest

MARKER = "requires_dataset"


def dataset_root() -> Path | None:
    """The configured BIDS root when it is a directory, else None."""
    from . import _config

    try:
        root = _config.bids_root()
    except (FileNotFoundError, KeyError):
        return None
    return root if root.is_dir() else None


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        f"{MARKER}: reads the MMMData dataset; skipped where it is not reachable",
    )


def pytest_collection_modifyitems(config: pytest.Config,
                                  items: list[pytest.Item]) -> None:
    if dataset_root() is not None:
        return
    skip = pytest.mark.skip(
        reason="no MMMData dataset reachable here (set MMMDATA_BIDS_ROOT to a "
               "copy, or run on the cluster)")
    for item in items:
        if MARKER in item.keywords:
            item.add_marker(skip)
