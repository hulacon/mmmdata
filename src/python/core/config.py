"""Configuration loading utilities."""

import os
import tomllib
from pathlib import Path
from typing import Dict, Any

#: ``[paths]`` keys that live inside the dataset, as their place under
#: ``bids_project_dir``. When ``local.toml`` or ``MMMDATA_BIDS_ROOT`` moves
#: the root and leaves one of these unset, it moves with the root, so one
#: setting points everything at a copy of the dataset.
DATASET_PATHS = {"output_dir": "derivatives", "inventory_dir": "inventory"}

#: Environment variable that sets ``[paths] bids_project_dir``. It wins over
#: local.toml, so a shell or a job can point at a copy without editing files.
ROOT_ENV = "MMMDATA_BIDS_ROOT"


def _find_config_dir() -> Path:
    """
    Find the config directory using multiple strategies.

    Tries in order:
    1. MMMDATA_CONFIG_DIR environment variable
    2. Walking up from current file to find 'config' directory
    3. Relative path from this file (legacy, fragile)

    Returns:
        Path to the config directory

    Raises:
        FileNotFoundError: If config directory cannot be found
    """
    # Strategy 1: Environment variable
    if 'MMMDATA_CONFIG_DIR' in os.environ:
        config_dir = Path(os.environ['MMMDATA_CONFIG_DIR'])
        if config_dir.exists() and config_dir.is_dir():
            return config_dir

    # Strategy 2: Walk up from current file to find config directory
    current = Path(__file__).resolve().parent
    for _ in range(5):  # Limit depth to prevent infinite loop
        config_candidate = current / 'config'
        if config_candidate.exists() and config_candidate.is_dir():
            if (config_candidate / 'base.toml').exists():
                return config_candidate
        current = current.parent

    # Strategy 3: Legacy relative path (fragile but backward compatible)
    legacy_path = Path(__file__).parent.parent.parent.parent / 'config'
    if legacy_path.exists() and (legacy_path / 'base.toml').exists():
        return legacy_path

    raise FileNotFoundError(
        "Could not locate config directory. "
        "Set MMMDATA_CONFIG_DIR environment variable or ensure config/ exists in repository root."
    )


def _deep_update(base: Dict, override: Dict) -> Dict:
    """
    Recursively merge two dictionaries.

    Args:
        base: Base dictionary
        override: Dictionary with override values

    Returns:
        Merged dictionary with nested updates
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = value
    return result


def _follow_root(config: Dict, base_paths: Dict, explicit: set) -> None:
    """Move the DATASET_PATHS keys with a relocated ``bids_project_dir``.

    A key follows only when nothing set it explicitly (local.toml) and its
    base.toml value sits at the conventional place under base.toml's root;
    a base value elsewhere is a deliberate choice and is left alone.
    """
    paths = config.get("paths", {})
    old_root, new_root = base_paths.get("bids_project_dir"), paths.get("bids_project_dir")
    if old_root is None or new_root is None or Path(new_root) == Path(old_root):
        return
    for key, rel in DATASET_PATHS.items():
        if key in explicit or key not in base_paths:
            continue
        if Path(base_paths[key]) == Path(old_root) / rel:
            paths[key] = str(Path(new_root) / rel)


def load_config(config_dir: str | Path = None) -> Dict[str, Any]:
    """
    Load configuration from TOML files.

    Loads base.toml first, then overlays local.toml settings on top,
    allowing local overrides of base configuration. Nested dictionaries
    are merged recursively. ``MMMDATA_BIDS_ROOT``, when set, then replaces
    ``[paths] bids_project_dir``. If the root moved, the dataset-internal
    paths (``DATASET_PATHS``) move with it unless local.toml sets them.

    Parameters
    ----------
    config_dir : str or Path, optional
        Path to the config directory. If None, automatically discovers
        the config directory using environment variables or by searching
        up the directory tree.

    Returns
    -------
    dict
        Merged configuration dictionary with local settings overriding base.
        Config uses nested structure: config['paths']['bids_project_dir']

    Examples
    --------
    >>> config = load_config()
    >>> bids_dir = config['paths']['bids_project_dir']
    >>> email = config['slurm']['email']

    Environment Variables
    ---------------------
    MMMDATA_CONFIG_DIR : Path to config directory (optional)
    MMMDATA_BIDS_ROOT : Path to a copy of the dataset (optional)
    """
    if config_dir is None:
        config_dir = _find_config_dir()
    else:
        config_dir = Path(config_dir)

    base_config_path = config_dir / 'base.toml'
    local_config_path = config_dir / 'local.toml'

    # Load base configuration
    if not base_config_path.exists():
        raise FileNotFoundError(f"Base config file not found: {base_config_path}")

    with open(base_config_path, 'rb') as f:
        config = tomllib.load(f)
    base_paths = dict(config.get('paths', {}))
    explicit = set()

    # Overlay local configuration (if exists) with deep merge
    if local_config_path.exists():
        with open(local_config_path, 'rb') as f:
            local_config = tomllib.load(f)
            config = _deep_update(config, local_config)
        explicit = set(local_config.get('paths', {})) - {'bids_project_dir'}

    if os.environ.get(ROOT_ENV):
        config.setdefault('paths', {})['bids_project_dir'] = os.environ[ROOT_ENV]

    _follow_root(config, base_paths, explicit)
    return config
