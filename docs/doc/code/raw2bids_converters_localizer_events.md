---
title: localizer_events
parent: Raw-to-BIDS Converters
grand_parent: Code Documentation
nav_order: 53.05
---

# localizer_events

Convert localizer timing CSVs into BIDS _events.tsv files.

Handles files with conversion_type='localizer_events' (9 files total):
  - Auditory localizer (3 files, 1 per subject)
  - Motor localizer (6 files, 2 runs per subject)

All are final session files -> BIDS ses-30.

Auditory localizer format:
  Columns: sub_id, task_id, sess_id, run_id, trial_id, stim_start, stim_end,
           stim_fixation_start, stim_fixation_end
  Single trial with long auditory stimulus (~562s).

Motor localizer format:
  Columns: sub_id, task, onset, offset
  Block design with conditions: foot, mouth, saccade, hand, rest (20s blocks).

Usage:
    python localizer_events.py <timing_csv> [<output_events_tsv>] [--dry-run]

**Source:** `src/python/raw2bids_converters/localizer_events.py`
{: .fs-3 .text-grey-dk-000 }

---

## Functions

### `detect_localizer_type`

Detect whether this is an auditory or motor localizer.

```python
detect_localizer_type(csv_path)
```

---

### `parse_subj_run`

Extract subject and run numbers from localizer filename.

```python
parse_subj_run(csv_path)
```

---

### `convert_auditory`

Convert auditory localizer timing CSV -> BIDS events TSV.

```python
convert_auditory(csv_path, output_tsv, dry_run = False)
```

---

### `convert_motor`

Convert motor localizer timing CSV -> BIDS events TSV.

```python
convert_motor(csv_path, output_tsv, dry_run = False)
```

---

### `convert_file`

Convert a localizer timing CSV to BIDS events TSV+JSON.

```python
convert_file(csv_path, output_tsv, dry_run = False)
```

---

