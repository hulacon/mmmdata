"""dcm2bids config generation for the MMMData project.

Programmatic generation of dcm2bids configuration files from structured
session definitions, replacing the previous template + string-replacement
approach.

Legacy configs at ``dcm2bids_configfiles/`` can be used for validation::

    LEGACY_DIR = Path("/gpfs/projects/hulacon/shared/mmmdata/code/dcm2bids_configfiles")

"""

from .config_builder import build_config
from .overrides import OverrideResult
from .session_defs import SESSION_SCHEDULE, SessionDef, TaskDef

__all__ = ["build_config", "OverrideResult", "SESSION_SCHEDULE", "SessionDef", "TaskDef"]
