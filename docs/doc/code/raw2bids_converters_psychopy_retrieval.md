---
title: psychopy_retrieval
parent: Raw-to-BIDS Converters
grand_parent: Code Documentation
nav_order: 53.08
---

# psychopy_retrieval

Convert PsychoPy free recall retrieval CSVs into BIDS _events.tsv files.

Handles files with conversion_type='psychopy_retrieval' (30 files total).
These are PsychoPy output CSVs from the free recall retrieval (recall) task,
sessions 1-10 (BIDS ses-19 to ses-28), 1 run per session.

PsychoPy CSV structure (11 rows, ~89 columns):
  - Rows with trials_recall.thisN not NaN are trial rows (typically 4-6 per run)
  - Scanner reference time: use_row.started on first trial row
  - Trial onset: recall1.started
  - Duration: time of recall period (key_resp_recall.rt or next trial onset)

Usage:
    python psychopy_retrieval.py <psychopy_csv> [<output_events_tsv>] [--dry-run]

**Source:** `src/python/raw2bids_converters/psychopy_retrieval.py`
{: .fs-3 .text-grey-dk-000 }

---

## Functions

### `parse_filename`

Extract subject and session from PsychoPy retrieval filename.

Pattern: {subj}_{sess}_free_recall_recall_{timestamp}.csv
Example: 3_1_free_recall_recall_2025-04-01_13h31.05.742.csv

```python
parse_filename(csv_path)
```

---

### `parse_recall_rt`

Parse key_resp_recall.rt which may be a string like '[119.86]'.

```python
parse_recall_rt(val)
```

---

### `convert_file`

Convert a PsychoPy retrieval CSV to BIDS events TSV+JSON.

```python
convert_file(psychopy_csv, output_tsv, dry_run = False)
```

**Parameters**

- **`psychopy_csv`** (`str`) — Path to the PsychoPy output CSV.
- **`output_tsv`** (`str`) — Destination path for the BIDS events TSV.
- **`dry_run`** (`bool`) — If True, print what would be done without writing files.

**Returns**

bool
    True if conversion succeeded.

---

