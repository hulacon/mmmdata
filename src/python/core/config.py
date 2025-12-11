"""Configuration loading utilities."""

import tomllib
from pathlib import Path
from typing import Dict, Any


def load_config(config_dir: str | Path = None) -> Dict[str, Any]:
    """
    Load configuration from TOML files.
    
    Loads base.toml first, then overlays local.toml settings on top,
    allowing local overrides of base configuration.
    
    Parameters
    ----------
    config_dir : str or Path, optional
        Path to the config directory. If None, uses the default location
        relative to this file (../../../config).
    
    Returns
    -------
    dict
        Merged configuration dictionary with local settings overriding base.
    
    Examples
    --------
    >>> config = load_config()
    >>> bids_dir = config['bids_project_dir']
    """
    if config_dir is None:
        # Default to config directory relative to this file
        config_dir = Path(__file__).parent.parent.parent.parent / 'config'
    else:
        config_dir = Path(config_dir)
    
    base_config_path = config_dir / 'base.toml'
    local_config_path = config_dir / 'local.toml'
    
    # Load base configuration
    config = {}
    if base_config_path.exists():
        with open(base_config_path, 'rb') as f:
            config = tomllib.load(f)
    else:
        raise FileNotFoundError(f"Base config file not found: {base_config_path}")
    
    # Overlay local configuration (if exists)
    if local_config_path.exists():
        with open(local_config_path, 'rb') as f:
            local_config = tomllib.load(f)
            config.update(local_config)
    
    return config
