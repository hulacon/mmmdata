"""MMMData behavioral analysis library.

Provides tools for loading, analyzing, and visualizing trial-based
behavioral data from the MMMData longitudinal memory study.

Typical usage::

    from behavioral import io, accuracy, plotting

    df = io.load_tb2afc()
    acc = accuracy.accuracy_by_condition(df, group_cols=["subject", "enCon"])
    plotting.plot_accuracy_by_condition(acc)
"""

from . import io
from . import constants
from . import preprocessing
from . import accuracy
from . import rt
from . import learning
from . import encoding
from . import final_session
from . import plotting

__all__ = [
    "io",
    "constants",
    "preprocessing",
    "accuracy",
    "rt",
    "learning",
    "encoding",
    "final_session",
    "plotting",
]
