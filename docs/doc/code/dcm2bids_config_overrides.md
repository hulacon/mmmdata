---
title: overrides
parent: DCM2BIDS Config
grand_parent: Code Documentation
nav_order: 52.04
---

# overrides

Override file parsing for per-subject session idiosyncrasies.

Each subject may have an optional ``overrides.toml`` file that documents
and handles sessions deviating from the canonical schedule.  Example::

    [ses-10]
    note = "Re-entry after encoding run 1; 3 fieldmap pairs"
    fmap_groups = ["encoding_r1", "encoding_r2", "retrieval"]
    [ses-10.fmap_series.encoding_r1]
    ap = 5
    pa = 7
    [ses-10.fmap_series.encoding_r2]
    ap = 21
    pa = 23
    [ses-10.fmap_series.retrieval]
    ap = 37
    pa = 39

    [ses-02]
    note = "Localizer: PRF (3 runs) + auditory + tone"
    session_type = "localizer"
    [[ses-02.tasks]]
    task_label = "prf"
    protocol_base = "localizer_prf_run{n}"
    fmap_group = "first"
    runs = 3
    has_sbref = false
    [[ses-02.tasks]]
    task_label = "auditory"
    protocol_base = "localizer_auditory"
    fmap_group = "second"
    [[ses-02.tasks]]
    task_label = "tone"
    protocol_base = "localizer_tone"
    fmap_group = "second"

**Source:** `src/python/dcm2bids_config/overrides.py`
{: .fs-3 .text-grey-dk-000 }

---

## Classes

### `OverrideResult` (dataclass)

Result of applying overrides to a session definition.

**Fields**

- **`session_def`** (`SessionDef`)
- **`fmap_info`** (`dict[str, dict[str, int]] | None`) = `None`
- **`run_protocols`** (`dict[str, dict[int, str]] | None`) = `None`
- **`run_series`** (`dict[str, dict[int, dict[str, int]]] | None`) = `None`
- **`fmap_desc_map`** (`dict[str, str] | None`) = `None`

---

## Functions

### `load_overrides`

Load a subject's override TOML file.

```python
load_overrides(overrides_path: Path) -> dict[str, Any]
```

**Parameters**

- **`overrides_path`** (`Path`) — Path to the ``overrides.toml`` file.

**Returns**

dict
    Parsed TOML, keyed by session ID (e.g. ``"ses-10"``).
    Returns empty dict if file doesn't exist.

---

### `apply_overrides`

Apply overrides to a session definition.

```python
apply_overrides(session_id: str, session_def: SessionDef, overrides: dict[str, Any]) -> OverrideResult
```

**Parameters**

- **`session_id`** (`str`) — Session identifier (e.g. ``"ses-10"``).
- **`session_def`** (`SessionDef`) — Base session definition from the schedule.
- **`overrides`** (`dict`) — Full overrides dict (all sessions for a subject).

**Returns**

OverrideResult
    Modified session definition and optional override metadata.

---

