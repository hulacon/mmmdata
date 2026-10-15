"""A broken config must raise, not fall back to the hardcoded GPFS root.

Three resolvers caught every Exception from load_config() and returned the
production dataset path. A local.toml that points at a scratch tree but has a
syntax error then sent converters (which write) and readers to the shared
dataset without a word. Only an unreachable config system (no config dir,
config module not importable) keeps the fallback.
"""

import sys
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CONVERTERS = REPO_ROOT / "src" / "python" / "raw2bids_converters"
if str(CONVERTERS) not in sys.path:
    sys.path.insert(0, str(CONVERTERS))

import common  # noqa: E402
from behavioral import io as beh_io  # noqa: E402
from neuroimaging import io as nimg_io  # noqa: E402


@pytest.fixture
def broken_config(tmp_path, monkeypatch):
    cfg = tmp_path / "config"
    cfg.mkdir()
    (cfg / "base.toml").write_text(
        '[paths]\nbids_project_dir = "/x"\nsource_dir = "/y"\n')
    (cfg / "local.toml").write_text('[paths]\nbids_project_dir = /scratch\n')
    monkeypatch.setenv("MMMDATA_CONFIG_DIR", str(cfg))
    return cfg


@pytest.fixture
def missing_config(tmp_path, monkeypatch):
    import core.config as cc
    def boom():
        raise FileNotFoundError("no config dir")
    monkeypatch.setattr(cc, "_find_config_dir", boom)


@pytest.mark.parametrize("resolve", [
    lambda: common._load_paths(),
    lambda: beh_io._resolve_bids_root(),
    lambda: nimg_io._resolve_bids_root(),
])
def test_malformed_local_toml_raises(broken_config, resolve):
    with pytest.raises(tomllib.TOMLDecodeError):
        resolve()


def test_unreachable_config_still_falls_back(missing_config):
    assert common._load_paths() == (common._FALLBACK_BIDS_ROOT,
                                    common._FALLBACK_SOURCE_DIR)
    assert beh_io._resolve_bids_root() == beh_io.DEFAULT_BIDS_ROOT
    assert nimg_io._resolve_bids_root() == nimg_io.DEFAULT_BIDS_ROOT
