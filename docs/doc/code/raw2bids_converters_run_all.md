---
title: run_all
parent: Raw-to-BIDS Converters
grand_parent: Code Documentation
nav_order: 53.09
---

# run_all

Orchestrator: read file_inventory.csv and run all converters.

Reads the inventory, groups files by conversion_type, and dispatches each
to the appropriate converter module. Supports dry-run, subject filtering,
and task filtering.

Usage:
    python run_all.py [--dry-run] [--subjects sub-03,sub-04] [--tasks encoding,math]
    python run_all.py --validate [--subjects sub-03]

**Source:** `src/python/raw2bids_converters/run_all.py`
{: .fs-3 .text-grey-dk-000 }

---

## Functions

### `load_inventory`

Load and return all rows from file_inventory.csv.

```python
load_inventory()
```

---

### `filter_rows`

Filter inventory rows by subject and/or conversion type.

```python
filter_rows(rows, subjects = None, conversion_types = None)
```

---

### `process_timed_events`

Process a timed_events file (behavioral + timing -> events.tsv).

```python
process_timed_events(row, dry_run = False)
```

---

### `process_psychopy_encoding`

Process a psychopy_encoding file.

```python
process_psychopy_encoding(row, dry_run = False)
```

---

### `process_psychopy_retrieval`

Process a psychopy_retrieval file.

```python
process_psychopy_retrieval(row, dry_run = False)
```

---

### `process_localizer`

Process a localizer_events file.

```python
process_localizer(row, dry_run = False)
```

---

### `process_behavioral`

Process a behavioral_to_beh file.

```python
process_behavioral(row, dry_run = False)
```

---

### `process_edf`

Process an edf_to_physio file.

```python
process_edf(row, dry_run = False)
```

---

### `process_physio_dcm`

Process a physio_dcm file (PhysioLog DICOM -> BIDS physio).

```python
process_physio_dcm(row, dry_run = False)
```

---

