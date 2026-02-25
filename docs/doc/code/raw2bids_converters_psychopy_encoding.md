---
title: psychopy_encoding
parent: Raw-to-BIDS Converters
grand_parent: Code Documentation
nav_order: 53.07
---

# psychopy_encoding

Convert PsychoPy free recall encoding CSVs into BIDS _events.tsv files.

Handles files with conversion_type='psychopy_encoding' (60 files total).
These are PsychoPy output CSVs from the free recall encoding (movie watching)
task, sessions 1-10 (BIDS ses-19 to ses-28), 2 runs per session.

PsychoPy CSV structure (9 rows, ~77 columns):
  - Rows with movie_loop.thisN not NaN are trial rows (4 movies per run)
  - Scanner reference time: use_row.started on first trial row
  - Movie onset: movies.started
  - Movie duration: mov_len (seconds)

Usage:
    python psychopy_encoding.py <psychopy_csv> [<output_events_tsv>] [--dry-run]

**Source:** `src/python/raw2bids_converters/psychopy_encoding.py`
{: .fs-3 .text-grey-dk-000 }

---

## Functions

### `parse_filename`

Extract subject, session, run from PsychoPy encoding filename.

Pattern: {subj}_{sess}_{run}_mem_search_recall_{timestamp}.csv
Example: 3_1_1_mem_search_recall_2025-04-01_12h39.55.925.csv

```python
parse_filename(csv_path)
```

---

### `convert_file`

Convert a PsychoPy encoding CSV to BIDS events TSV+JSON.

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

