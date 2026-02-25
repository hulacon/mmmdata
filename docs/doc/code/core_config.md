---
title: config
parent: Core
grand_parent: Code Documentation
nav_order: 51.02
---

# config

Configuration loading utilities.

**Source:** `src/python/core/config.py`
{: .fs-3 .text-grey-dk-000 }

---

## Functions

### `load_config`

Load configuration from TOML files.

Loads base.toml first, then overlays local.toml settings on top,
allowing local overrides of base configuration. Nested dictionaries
are merged recursively.

```python
load_config(config_dir: str | Path = None) -> Dict[str, Any]
```

**Parameters**

- **`config_dir`** (`str or Path, optional`) — Path to the config directory. If None, automatically discovers the config directory using environment variables or by searching up the directory tree.

**Returns**

dict
    Merged configuration dictionary with local settings overriding base.
    Config uses nested structure: config['paths']['bids_project_dir']

**Examples**

```python
>>> config = load_config()
>>> bids_dir = config['paths']['bids_project_dir']
>>> email = config['slurm']['email']

Environment Variables
MMMDATA_CONFIG_DIR : Path to config directory (optional)
```

---

