---
title: dicom_inspect
parent: DCM2BIDS Config
grand_parent: Code Documentation
nav_order: 52.03
---

# dicom_inspect

Inspect DICOM directories to extract fieldmap series information.

This is the only module that touches the filesystem.  It scans the DICOM
directory structure (``Series_##_<description>/``) to determine:
- Which fieldmap strategy applies (series_description vs series_number)
- The actual series numbers for each AP/PA pair

**Source:** `src/python/dcm2bids_config/dicom_inspect.py`
{: .fs-3 .text-grey-dk-000 }

---

## Classes

### `FieldmapDetection` (dataclass)

Result of inspecting a DICOM session directory for fieldmaps.

**Fields**

- **`strategy`** (`str`) = `'none'`
- **`groups`** (`dict[str, dict[str, int]]`) = `field(default_factory=dict)`
- **`warnings`** (`list[str]`) = `field(default_factory=list)`

---

## Functions

### `inspect_fieldmaps`

Scan a DICOM session directory for spin-echo fieldmap series.

```python
inspect_fieldmaps(dicom_dir: Path) -> FieldmapDetection
```

**Parameters**

- **`dicom_dir`** (`Path`) — Path to the session DICOM directory containing ``Series_*/`` dirs.

**Returns**

FieldmapDetection
    Detected strategy, groups, and any warnings.

---

