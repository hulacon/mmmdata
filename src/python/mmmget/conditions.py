"""events(): condition-labelled events for a run (waits on decision D6)."""

from __future__ import annotations

from typing import Optional

from neuroimaging.io import FmriprepRun

from .errors import OpenDecisionError


def events(run: FmriprepRun, scheme: Optional[str] = None):
    """A run's events under a named condition scheme.

    Not available yet: where the lab's condition definitions live is open
    decision D6, so there is no scheme to read. Raises
    :class:`~mmmget.errors.OpenDecisionError` rather than guessing one.
    """
    where = f" (raw events file: {run.events})" if getattr(run, "events", None) else ""
    raise OpenDecisionError(
        "D6",
        f"where the lab's condition definitions live is not decided, so "
        f"events(run, scheme={scheme!r}) has no scheme to apply. Until it is, "
        f"read the run's BIDS events file yourself{where}; a result built on "
        f"it is not on a standard scheme.")
