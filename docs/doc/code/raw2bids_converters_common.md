---
title: common
parent: Raw-to-BIDS Converters
grand_parent: Code Documentation
nav_order: 53.02
---

# common

Shared utilities for raw-to-BIDS behavioral data converters.

**Source:** `src/python/raw2bids_converters/common.py`
{: .fs-3 .text-grey-dk-000 }

---

## Functions

### `bids_sub`

Return BIDS subject label, e.g. 'sub-03'.

```python
bids_sub(num)
```

---

### `bids_ses`

Return BIDS session label, e.g. 'ses-04'.

```python
bids_ses(num)
```

---

### `bids_ses_fr`

Map free recall session number to BIDS session label.

```python
bids_ses_fr(sess_num)
```

---

### `bids_ses_cr`

Map cued recall session number to BIDS session label.

```python
bids_ses_cr(sess_num)
```

---

### `na_value`

Replace NaN/None/empty with BIDS 'n/a'.

```python
na_value(val)
```

---

### `write_events_tsv`

Write a BIDS events TSV file.

- Tab-separated, no index column
- NaN/missing replaced with 'n/a'
- Creates parent directories if needed

```python
write_events_tsv(df, output_path, dry_run = False)
```

---

### `write_beh_tsv`

Write a BIDS behavioral TSV file (same format as events).

```python
write_beh_tsv(df, output_path, dry_run = False)
```

---

### `write_json_sidecar`

Write a BIDS JSON sidecar file with column descriptions.

```python
write_json_sidecar(descriptions, output_path, dry_run = False)
```

**Parameters**

- **`descriptions`** (`dict`) — Maps column names to description dicts, e.g.: {"onset": {"Description": "Event onset time", "Units": "s"}}
- **`output_path`** (`str`) — Path for JSON file.

---

### `bids_output_path`

Build a full BIDS output path.

```python
bids_output_path(sub_num, ses_num, modality, filename)
```

**Parameters**

- **`sub_num`** (`int`) — Subject number (e.g. 3)
- **`ses_num`** (`int`) — BIDS session number (already mapped, e.g. 19 for free recall session 1)
- **`modality`** (`str`) — 'func' or 'beh'
- **`filename`** (`str`) — BIDS filename (e.g. 'sub-03_ses-19_task-encoding_run-01_events.tsv')

---

### `int_or_na`

Convert to int if numeric, else return 'n/a'.

```python
int_or_na(val)
```

---

### `float_or_na`

Convert to float if numeric, else return 'n/a'.

```python
float_or_na(val)
```

---

