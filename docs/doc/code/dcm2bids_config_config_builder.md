---
title: config_builder
parent: DCM2BIDS Config
grand_parent: Code Documentation
nav_order: 52.02
---

# config_builder

Build dcm2bids configuration dicts from structured session definitions.

This module contains pure functions with no filesystem I/O.  All inputs are
dataclasses or simple dicts; the output is a plain ``dict`` ready for
``json.dumps()``.

**Source:** `src/python/dcm2bids_config/config_builder.py`
{: .fs-3 .text-grey-dk-000 }

---

## Functions

### `build_config`

Build a complete dcm2bids config dict.

This is a pure function: same inputs always produce the same output.

```python
build_config(subject: str, session: str, session_def: SessionDef, fmap_info: FieldmapInfo | None = None, *, run_protocols: dict[str, dict[int, str]] | None = None, run_series: dict[str, dict[int, dict[str, int]]] | None = None, fmap_desc_map: dict[str, str] | None = None) -> dict
```

**Parameters**

- **`subject`** (`str`) — Subject ID including prefix (e.g. ``"sub-03"``).
- **`session`** (`str`) — Session ID including prefix (e.g. ``"ses-06"``).
- **`session_def`** (`SessionDef`) — The session type definition (from ``session_defs``).
- **`fmap_info`** (`dict, optional`) — Fieldmap series numbers, keyed by group name.  Required when ``session_def.fmap_strategy == "series_number"``.  Example:: {"encoding": {"ap": 10, "pa": 11}, "retrieval": {"ap": 28, "pa": 30}}
- **`run_protocols`** (`dict, optional`) — Per-task, per-run ProtocolName overrides.  Example:: {"FINretrieval": {1: "free_recall_retrieval_run1_attempt2"}}
- **`run_series`** (`dict, optional`) — Per-task, per-run SeriesNumber constraints for BOLD/SBRef.  Example:: {"FINretrieval": {2: {"bold": 45, "sbref": 44}}}

**Returns**

dict
    A dcm2bids-compatible config dict with a ``"descriptions"`` key.

---

