"""Tests for config loading functionality."""

import pytest
from pathlib import Path
import tempfile
import tomllib

from src.python.core.config import load_config, _deep_update, _find_config_dir


class TestDeepUpdate:
    """Tests for the _deep_update function."""

    def test_simple_update(self):
        """Test simple key-value update."""
        base = {'a': 1, 'b': 2}
        override = {'b': 3, 'c': 4}
        result = _deep_update(base, override)
        assert result == {'a': 1, 'b': 3, 'c': 4}

    def test_nested_update(self):
        """Test nested dictionary update."""
        base = {'paths': {'dir1': '/path1', 'dir2': '/path2'}, 'other': 'value'}
        override = {'paths': {'dir2': '/newpath2', 'dir3': '/path3'}}
        result = _deep_update(base, override)
        assert result == {
            'paths': {'dir1': '/path1', 'dir2': '/newpath2', 'dir3': '/path3'},
            'other': 'value'
        }

    def test_deep_nested_update(self):
        """Test deeply nested dictionary update."""
        base = {'level1': {'level2': {'level3': 'old'}}}
        override = {'level1': {'level2': {'level3': 'new', 'extra': 'value'}}}
        result = _deep_update(base, override)
        assert result == {'level1': {'level2': {'level3': 'new', 'extra': 'value'}}}


class TestLoadConfig:
    """Tests for the load_config function."""

    def test_load_base_config_only(self, tmp_path):
        """Test loading only base config when local doesn't exist."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()

        base_config = {
            'paths': {'bids_project_dir': '/test/path'},
            'slurm': {'partition': 'compute'}
        }

        with open(config_dir / 'base.toml', 'wb') as f:
            import tomli_w
            tomli_w.dump(base_config, f)

        config = load_config(config_dir)
        assert config['paths']['bids_project_dir'] == '/test/path'
        assert config['slurm']['partition'] == 'compute'

    def test_load_with_local_override(self, tmp_path):
        """Test loading base config with local overrides."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()

        base_config = {
            'paths': {'bids_project_dir': '/test/path', 'output_dir': '/output'},
            'slurm': {'partition': 'compute'}
        }

        local_config = {
            'paths': {'bids_project_dir': '/local/path'},
            'slurm': {'email': 'test@example.com'}
        }

        with open(config_dir / 'base.toml', 'wb') as f:
            import tomli_w
            tomli_w.dump(base_config, f)

        with open(config_dir / 'local.toml', 'wb') as f:
            import tomli_w
            tomli_w.dump(local_config, f)

        config = load_config(config_dir)
        # Local should override base
        assert config['paths']['bids_project_dir'] == '/local/path'
        # Base values not overridden should remain
        assert config['paths']['output_dir'] == '/output'
        # New local values should be added
        assert config['slurm']['email'] == 'test@example.com'
        # Base values not overridden should remain
        assert config['slurm']['partition'] == 'compute'

    def test_missing_base_config_raises_error(self, tmp_path):
        """Test that missing base config raises FileNotFoundError."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()

        with pytest.raises(FileNotFoundError, match="Base config file not found"):
            load_config(config_dir)


class TestFindConfigDir:
    """Tests for the _find_config_dir function."""

    def test_find_config_with_env_var(self, tmp_path, monkeypatch):
        """Test finding config directory using environment variable."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / 'base.toml').touch()

        monkeypatch.setenv('MMMDATA_CONFIG_DIR', str(config_dir))
        found_dir = _find_config_dir()
        assert found_dir == config_dir

    def test_env_var_takes_precedence(self, tmp_path, monkeypatch):
        """Test that environment variable takes precedence."""
        # Create two config directories
        config_dir1 = tmp_path / "config1"
        config_dir1.mkdir()
        (config_dir1 / 'base.toml').touch()

        config_dir2 = tmp_path / "config2"
        config_dir2.mkdir()
        (config_dir2 / 'base.toml').touch()

        # Set env var to config_dir1
        monkeypatch.setenv('MMMDATA_CONFIG_DIR', str(config_dir1))

        found_dir = _find_config_dir()
        assert found_dir == config_dir1


# Run tests with pytest if executed directly
if __name__ == '__main__':
    pytest.main([__file__, '-v'])


class TestDatasetRoot:
    """One setting relocates the dataset: the dataset-internal paths follow."""

    BASE = (
        '[paths]\n'
        'bids_project_dir = "/cluster/data"\n'
        'output_dir = "/cluster/data/derivatives"\n'
        'inventory_dir = "/cluster/data/inventory"\n'
        'source_dir = "/cluster/source"\n'
    )

    @pytest.fixture
    def config_dir(self, tmp_path, monkeypatch):
        monkeypatch.delenv("MMMDATA_BIDS_ROOT", raising=False)
        d = tmp_path / "config"
        d.mkdir()
        (d / "base.toml").write_text(self.BASE)
        return d

    def test_unmoved_root_changes_nothing(self, config_dir):
        paths = load_config(config_dir)["paths"]
        assert paths["output_dir"] == "/cluster/data/derivatives"

    def test_local_root_moves_dataset_paths(self, config_dir):
        (config_dir / "local.toml").write_text('[paths]\nbids_project_dir = "/copy"\n')
        paths = load_config(config_dir)["paths"]
        assert paths["output_dir"] == str(Path("/copy/derivatives"))
        assert paths["inventory_dir"] == str(Path("/copy/inventory"))
        # outside the dataset: not moved
        assert paths["source_dir"] == "/cluster/source"

    def test_explicit_local_key_wins(self, config_dir):
        (config_dir / "local.toml").write_text(
            '[paths]\nbids_project_dir = "/copy"\noutput_dir = "/elsewhere"\n')
        paths = load_config(config_dir)["paths"]
        assert paths["output_dir"] == "/elsewhere"
        assert paths["inventory_dir"] == str(Path("/copy/inventory"))

    def test_env_root_wins_over_local(self, config_dir, monkeypatch):
        (config_dir / "local.toml").write_text('[paths]\nbids_project_dir = "/copy"\n')
        monkeypatch.setenv("MMMDATA_BIDS_ROOT", "/env-copy")
        paths = load_config(config_dir)["paths"]
        assert paths["bids_project_dir"] == "/env-copy"
        assert paths["output_dir"] == str(Path("/env-copy/derivatives"))

    def test_unconventional_base_value_is_left_alone(self, config_dir, monkeypatch):
        (config_dir / "base.toml").write_text(
            self.BASE.replace('output_dir = "/cluster/data/derivatives"',
                              'output_dir = "/scratch/derivs"'))
        monkeypatch.setenv("MMMDATA_BIDS_ROOT", "/copy")
        assert load_config(config_dir)["paths"]["output_dir"] == "/scratch/derivs"
