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
