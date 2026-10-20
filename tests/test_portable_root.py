"""The getters and viewers work on a copy of the dataset, not only on GPFS.

Someone who copies the dataset elsewhere (Globus) and clones the repos sets
one thing, the root, in local.toml or MMMDATA_BIDS_ROOT. Everything here
uses the tracked base.toml unchanged, then asserts that no path under
base.toml's dataset tree is opened or listed. The repo and the running
interpreter's environment are exempt: on the cluster both can sit inside
the dataset's code/ directory, and a lazy import is not a data read.
"""

import os
import shutil
import subprocess
import sys
import tomllib
from contextlib import contextmanager
from pathlib import Path

import pytest

import mmmget
from tests.mmmget_fixtures import make_env
from tests.test_mmmget_results import SPEC

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE = tomllib.loads((REPO_ROOT / "config" / "base.toml").read_text())["paths"]
#: The dataset tree base.toml points at; nothing under it may be touched.
FORBIDDEN = [Path(BASE[k]) for k in ("bids_project_dir", "output_dir", "inventory_dir")]
#: Code, not data: the checkout and the interpreter's environment.
EXEMPT = {REPO_ROOT, Path(sys.prefix).absolute(), Path(sys.exec_prefix).absolute()}

_touched: list[str] = []
_watching = False


def _hook(event, args):
    if not _watching or event not in ("open", "os.listdir", "os.scandir"):
        return
    if not args or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    p = Path(os.fsdecode(args[0])).absolute()
    if any(p == e or e in p.parents for e in EXEMPT):
        return
    if any(p == f or f in p.parents for f in FORBIDDEN):
        _touched.append(str(p))


sys.addaudithook(_hook)  # cannot be removed; inert unless _watching


@contextmanager
def no_dataset_paths():
    global _watching
    _touched.clear()
    _watching = True
    try:
        yield
    finally:
        _watching = False
    assert not _touched, f"opened paths from base.toml's dataset tree: {_touched[:5]}"


@pytest.fixture
def copy_root(tmp_path, monkeypatch):
    """The synthetic dataset, with the tracked base.toml and only the root moved."""
    monkeypatch.delenv("MMMDATA_BIDS_ROOT", raising=False)
    bids = make_env(tmp_path, monkeypatch)
    cfg = tmp_path / "config"
    shutil.copy(REPO_ROOT / "config" / "base.toml", cfg / "base.toml")
    (cfg / "local.toml").write_text(f'[paths]\nbids_project_dir = "{bids}"\n')
    return bids


def test_getters_read_only_the_copy(copy_root, tmp_path):
    with no_dataset_paths():
        (run,) = mmmget.runs(subject="01", task="memory", run=1)
        design = mmmget.confounds(run, regime="base")
        mmmget.roi("01", "schaefer100-7:LH_Vis_1")
        tsv, _ = mmmget.write_result(
            [{"run": run.run, "n_regressors": design.n_regressors}],
            SPEC, tmp_path / "out", "regs")
    assert copy_root in run.confounds.parents
    assert tsv.exists()


def test_env_var_alone_relocates(tmp_path, monkeypatch):
    bids = make_env(tmp_path, monkeypatch)
    cfg = tmp_path / "config"
    shutil.copy(REPO_ROOT / "config" / "base.toml", cfg / "base.toml")
    monkeypatch.setenv("MMMDATA_BIDS_ROOT", str(bids))
    with no_dataset_paths():
        assert len(mmmget.runs(subject="01", task="memory")) == 2


def test_mmmview_roots_follow_the_env_root(tmp_path, monkeypatch):
    if (REPO_ROOT / "config" / "local.toml").exists():
        pytest.skip("a local.toml in this checkout may set output_dir explicitly")
    monkeypatch.setenv("MMMDATA_BIDS_ROOT", str(tmp_path))
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    try:
        import mmmview
    finally:
        sys.path.remove(str(REPO_ROOT / "scripts"))
    roots = mmmview.load_roots()
    assert roots.deriv == tmp_path / "derivatives"
    assert roots.bids == tmp_path


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_shell_loader_follows_the_env_root(tmp_path):
    if (REPO_ROOT / "config" / "local.toml").exists():
        pytest.skip("a local.toml in this checkout may set output_dir explicitly")
    env = {**os.environ, "MMMDATA_BIDS_ROOT": str(tmp_path), "REPO_ROOT": str(REPO_ROOT),
           "PATH": f"{Path(sys.executable).parent}:{os.environ.get('PATH', '')}"}
    out = subprocess.run(
        ["bash", "-c", f'source "{REPO_ROOT}/scripts/load_config.sh" >/dev/null '
                       '&& echo "$BIDS_DIR|$OUTPUT_DIR|$INVENTORY_DIR"'],
        env=env, capture_output=True, text=True, check=True).stdout.strip()
    assert out == f"{tmp_path}|{tmp_path / 'derivatives'}|{tmp_path / 'inventory'}"



def test_catalog_built_for_another_root_is_refused(copy_root, tmp_path):
    """A copy carrying the cluster's catalog: its file list is the cluster's."""
    import duckdb
    from core.catalog import CatalogRootMismatch

    con = duckdb.connect(str(copy_root / "inventory" / "catalog.duckdb"))
    con.execute("UPDATE sweep_meta SET value = ?",
                ['{"root": "/cluster/mmmdata", "started_at": "x"}'])
    con.close()
    from tests.mmmget_fixtures import SUBFIELD_TOML
    local = tmp_path / "config" / "local.toml"
    local.write_text(local.read_text() + SUBFIELD_TOML)
    with pytest.raises(CatalogRootMismatch, match="duckbrain.catalog rebuild --root"):
        mmmget.runs(subject="01", task="memory")
    with pytest.raises(CatalogRootMismatch):
        mmmget.roi("sub-01", "hipp:LH_CA1", space="T1w")


def test_runs_records_when_the_catalog_was_swept(copy_root):
    from tests.mmmget_fixtures import SWEPT_AT

    mmmget.runs(subject="01", task="memory")
    (rec,) = mmmget.provenance_records()
    assert rec["catalog_swept_at"] == SWEPT_AT
