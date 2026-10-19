"""What the getters were asked for in this process, for write_result to stamp.

Every getter call appends one record. A record names the getter, the
arguments that decide *which standard* was followed (a regime name and its
version, an ROI name and its source, whether QC exclusions were applied),
and whether the call stayed on the standard path. Per-file details are not
recorded: a result's sidecar should say what it was built on, not repeat
the inputs.

The log is process-wide and accumulates until :func:`reset_provenance`, so
two results written from the same inputs carry the same record. A script
that builds unrelated results in one process calls ``reset_provenance()``
between them.
"""

from __future__ import annotations

import copy
from typing import Any

_LOG: list[dict[str, Any]] = []


def record(getter: str, *, standard: bool, reason: str | None = None,
           **details: Any) -> dict[str, Any]:
    """Append one getter call to the log. ``reason`` says why a call is
    non-standard; it is required when ``standard`` is False."""
    if not standard and not reason:
        raise ValueError("a non-standard record must say why (reason=...)")
    entry: dict[str, Any] = {"getter": getter, "standard": bool(standard)}
    if reason:
        entry["reason"] = reason
    entry.update({k: v for k, v in details.items() if v is not None})
    if entry not in _LOG:  # a loop over runs asks the same question many times
        _LOG.append(entry)
    return entry


def provenance_records() -> list[dict[str, Any]]:
    """A copy of every distinct getter call recorded since the last reset."""
    return copy.deepcopy(_LOG)


def reset_provenance() -> None:
    """Forget every recorded call (start of a new, unrelated result)."""
    _LOG.clear()
