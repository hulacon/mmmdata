"""Path resolution for the MMMData MCP server.

Every path comes from the mmmdata config (``config/base.toml`` with a
``config/local.toml`` overlay, via ``core.config.load_config``). Nothing
here names a host, so the same server works against the dataset on
Talapas and against a copy anywhere else: point ``[paths]`` in
``local.toml`` at the copy.

A configured root that does not exist raises :class:`DatasetUnavailable`,
which the server turns into a JSON error naming the path, the config key
and the fix. That is the expected state on a laptop or CI runner, not a
fault, and it must never read as "no data".
"""

import sys
from functools import cached_property
from pathlib import Path
from typing import Any

#: Repository root: src/python/mmmdata_mcp/paths.py -> three levels up.
REPO_ROOT = Path(__file__).resolve().parents[3]

#: ProjectPaths attribute -> the config key it is read from, so an
#: error can name the key a user has to change.
_CONFIG_KEYS = {
    "bids_root": "[paths] bids_project_dir",
    "output_dir": "[paths] output_dir",
    "inventory_dir": "[paths] inventory_dir",
    "sourcedata_dir": "[paths] source_dir",
}


class DatasetUnavailable(FileNotFoundError):
    """A configured dataset root does not exist on this host.

    Subclasses FileNotFoundError so callers that already catch that keep
    working, and carries the path and config key so a caller can name them.
    """

    def __init__(self, path: str | Path, config_key: str):
        self.path = Path(path)
        self.config_key = config_key
        super().__init__(
            f"MMMData dataset not found: {self.path} (from {config_key}).\n"
            "This is expected off-cluster: config/base.toml points at "
            "Talapas GPFS.\n"
            "To use a copy of the dataset, set that key in "
            "config/local.toml (see config/local.toml.example), or point "
            "MMMDATA_CONFIG_DIR at a config dir whose base.toml does."
        )


class ProjectPaths:
    """Lazily-resolved dataset paths derived from the mmmdata config."""

    def __init__(self, cfg: dict[str, Any] | None = None):
        if cfg is None:
            from core.config import load_config
            cfg = load_config()
        self._cfg = cfg

    @property
    def raw_config(self) -> dict[str, Any]:
        """The underlying config dict, for passing to libraries that need it."""
        return self._cfg

    def _path(self, key: str) -> Path:
        return Path(self._cfg["paths"][key])

    # -- Roots (direct from config) --

    @cached_property
    def bids_root(self) -> Path:
        """BIDS dataset root."""
        return self._path("bids_project_dir")

    @cached_property
    def output_dir(self) -> Path:
        """Derivatives directory."""
        return self._path("output_dir")

    @cached_property
    def inventory_dir(self) -> Path:
        """Inventory directory (holds the Contract A catalog)."""
        return self._path("inventory_dir")

    @cached_property
    def sourcedata_dir(self) -> Path:
        """Raw source data root, outside the BIDS tree (holds PII)."""
        return self._path("source_dir")

    @cached_property
    def viz2psy_dir(self) -> str | None:
        """viz2psy checkout, for feature-name metadata in stimulus plots.

        Optional: the plots work without it, with less descriptive labels.
        """
        value = self._cfg["paths"].get("viz2psy_dir")
        return str(value) if value else None

    # -- Code locations (from this checkout, not config) --

    @cached_property
    def mmmdata_root(self) -> Path:
        """Root of the mmmdata code repository this server runs from."""
        return REPO_ROOT

    @cached_property
    def converters_dir(self) -> Path:
        """raw2bids_converters package directory."""
        return REPO_ROOT / "src" / "python" / "raw2bids_converters"

    @cached_property
    def mmmdata_venv_python(self) -> str:
        """Python for converter subprocesses: the repo venv if present."""
        venv_python = REPO_ROOT / ".venv" / "bin" / "python"
        if venv_python.exists():
            return str(venv_python)
        return sys.executable

    @cached_property
    def figures_dir(self) -> Path:
        """Where plot tools write HTML: ``[mcp] figures_dir``.

        Relative values resolve against the repository root. Created on
        first use.
        """
        raw = self._cfg.get("mcp", {}).get("figures_dir", "output/figures")
        p = Path(raw)
        if not p.is_absolute():
            p = REPO_ROOT / p
        p.mkdir(parents=True, exist_ok=True)
        return p

    # -- Derived paths --

    @cached_property
    def catalog_db(self) -> Path:
        """Contract A catalog database."""
        return self.inventory_dir / "catalog.duckdb"

    @cached_property
    def stimuli_dir(self) -> Path:
        """BIDS stimuli directory."""
        return self.bids_root / "stimuli"

    @cached_property
    def stimuli_features_dir(self) -> Path:
        """Contract B feature store (derivatives/stimuli_features)."""
        return self.bids_root / "derivatives" / "stimuli_features"

    @cached_property
    def stimuli_features_tables_dir(self) -> Path:
        """The psytwill aggregate tables inside the Contract B store."""
        return self.stimuli_features_dir / "psytwill"

    @cached_property
    def mriqc_dir(self) -> Path:
        """MRIQC output directory."""
        return self.output_dir / "mriqc"

    @cached_property
    def fmriprep_dir(self) -> Path:
        """fMRIPrep derivatives directory (standard variant)."""
        return self.output_dir / "fmriprep"

    #: Preprocessing variants -> fMRIPrep derivatives directory names.
    FMRIPREP_VARIANTS = {"standard": "fmriprep", "nordic": "fmriprep_nordic"}

    def fmriprep_dir_for(self, variant: str | None = None) -> Path:
        """fMRIPrep derivatives directory for a preprocessing variant.

        'standard' (or None) -> fmriprep/; 'nordic' -> fmriprep_nordic/
        (NORDIC-denoised inputs).
        """
        key = (variant or "standard").lower()
        if key not in self.FMRIPREP_VARIANTS:
            raise ValueError(
                f"Unknown fMRIPrep variant '{variant}'; "
                f"expected one of {sorted(self.FMRIPREP_VARIANTS)}"
            )
        return self.output_dir / self.FMRIPREP_VARIANTS[key]

    @cached_property
    def atlases_dir(self) -> Path:
        """Brain atlases directory."""
        return self.output_dir / "atlases"

    # -- Guards --

    def require_dir(self, name: str) -> Path:
        """Return root ``name``, raising DatasetUnavailable if it is absent."""
        p = getattr(self, name)
        if not Path(p).is_dir():
            raise DatasetUnavailable(p, _CONFIG_KEYS.get(name, name))
        return p


# Module-level singleton (lazy -- config loaded on first access)
_paths: ProjectPaths | None = None


def get_paths(cfg: dict[str, Any] | None = None) -> ProjectPaths:
    """Get a ProjectPaths instance.

    With no arguments, returns a cached singleton. Passing cfg creates a
    fresh instance (useful for testing).
    """
    global _paths
    if cfg is not None:
        return ProjectPaths(cfg)
    if _paths is None:
        _paths = ProjectPaths()
    return _paths


def dataset_available() -> bool:
    """Whether the configured BIDS root exists on this host."""
    try:
        return get_paths().bids_root.is_dir()
    except (KeyError, OSError):
        return False
