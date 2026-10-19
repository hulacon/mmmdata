"""Where the dataset is, from config (never a hardcoded path)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from core.config import load_config


def config() -> dict[str, Any]:
    """The merged mmmdata config (base.toml + local.toml).

    ``MMMDATA_CONFIG_DIR`` points it elsewhere, which is how tests and other
    machines supply their own paths.
    """
    return load_config()


def bids_root(override: Path | str | None = None) -> Path:
    """The BIDS root: *override* when given, else ``paths.bids_project_dir``."""
    if override is not None:
        return Path(override)
    return Path(config()["paths"]["bids_project_dir"])


def mmmget_section(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """The optional ``[mmmget]`` config table ({} when absent)."""
    cfg = config() if cfg is None else cfg
    section = cfg.get("mmmget", {})
    return section if isinstance(section, dict) else {}
