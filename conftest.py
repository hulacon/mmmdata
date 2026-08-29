"""Repo-root pytest configuration.

Two things a bare checkout needs and did not have.

**Imports.** The test suite imports the packages under ``src/python`` by
their bare names (``from core import catalog``, ``from behavioral.io
import ...``), which only resolves if ``src/python`` is on ``sys.path``.
Nothing in the repo put it there: it worked because the shared Talapas
env happens to provide it, so a fresh clone anywhere else collected
zero tests with a ``ModuleNotFoundError``. Bootstrapping it here means
``pytest`` works from a clean checkout with no environment setup.

**The dataset.** Tests that read the real dataset on Talapas GPFS are
marked ``requires_dataset`` and skip themselves with a reason when it is
not mounted, rather than failing an assertion that reads like a data
problem when it is really an absent filesystem.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent
SRC_PYTHON = REPO_ROOT / "src" / "python"

if str(SRC_PYTHON) not in sys.path:
    sys.path.insert(0, str(SRC_PYTHON))


def dataset_available() -> bool:
    """Whether the configured dataset roots are actually mounted.

    Answered from the config rather than a hardcoded path, so a local
    ``config/local.toml`` pointing at a copy of the tree counts as
    available.
    """
    try:
        from core.config import load_config

        cfg = load_config()
        paths = cfg["paths"]
        return Path(paths["bids_project_dir"]).is_dir()
    except Exception:
        return False


def pytest_collection_modifyitems(config, items):
    """Skip the dataset tier when the dataset is not reachable."""
    if dataset_available():
        return

    skip_no_dataset = pytest.mark.skip(
        reason="dataset not reachable from this host (off-cluster); "
               "these tests read the real tree on Talapas GPFS"
    )
    for item in items:
        if "requires_dataset" in item.keywords:
            item.add_marker(skip_no_dataset)
