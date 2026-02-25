---
title: session_defs
parent: DCM2BIDS Config
grand_parent: Code Documentation
nav_order: 52.05
---

# session_defs

Session and task definitions for dcm2bids config generation.

This module is the single source of truth for:
- What tasks exist and how they map to DICOM protocol names
- What each session type contains
- Which session numbers map to which session type

All data here is pure Python with no I/O.

**Source:** `src/python/dcm2bids_config/session_defs.py`
{: .fs-3 .text-grey-dk-000 }

---

## Classes

### `TaskDef` (dataclass)

Definition of a single functional task.

**Fields**

- **`task_label`** (`str`)
- **`protocol_base`** (`str`)
- **`fmap_group`** (`str`)
- **`runs`** (`int | tuple[int, ...]`) = `1`
- **`has_sbref`** (`bool`) = `False`

**Methods**

#### `run_numbers`

Return the explicit run numbers for this task.

```python
run_numbers() -> tuple[int, ...]
```

#### `is_multi_run`

Whether filenames should include a ``run-XX`` entity.

```python
is_multi_run() -> bool
```

#### `protocol_name`

Return the DICOM ProtocolName for a specific run number.

```python
protocol_name(run: int) -> str
```

#### `sbref_description`

Return the DICOM SeriesDescription for the SBRef of a run.

```python
sbref_description(run: int) -> str
```

---

### `AnatDef` (dataclass)

Definition of an anatomical or DWI acquisition.

**Fields**

- **`suffix`** (`str`)
- **`acq`** (`str`)
- **`series_description`** (`str`)
- **`datatype`** (`str`) = `'anat'`
- **`custom_entities`** (`str`) = `''`

---

### `SessionDef` (dataclass)

Complete definition of a session type.

**Fields**

- **`session_type`** (`str`)
- **`tasks`** (`tuple[TaskDef, ...]`) = `()`
- **`fmap_strategy`** (`str`) = `'series_number'`
- **`anat`** (`tuple[AnatDef, ...]`) = `()`
- **`fmap_groups`** (`tuple[str, ...]`) = `()`

**Methods**

#### `task_ids_for_fmap_group`

Return dcm2bids description IDs for all tasks in a fieldmap group.

```python
task_ids_for_fmap_group(group: str) -> list[str]
```

---

## Functions

### `get_session_def`

Look up the SessionDef for a session ID.

```python
get_session_def(session_id: str) -> SessionDef
```

**Parameters**

- **`session_id`** (`str`) — Session identifier (e.g. ``"ses-06"``).

**Returns**

SessionDef

---

