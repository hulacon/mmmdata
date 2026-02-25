---
title: generate_inventory
parent: Raw-to-BIDS Converters
grand_parent: Code Documentation
nav_order: 53.04
---

# generate_inventory

Generate file_inventory.csv from current sourcedata contents.

Walks sourcedata/{subject}/{session}/behavioral/ and eyetracking/ directories,
classifies each file, and maps it to a BIDS destination with the correct
prefixed task names (TBencoding, NATencoding, FINretrieval, etc.).

Usage:
    python generate_inventory.py [--output file_inventory.csv]

**Source:** `src/python/raw2bids_converters/generate_inventory.py`
{: .fs-3 .text-grey-dk-000 }

---

## Functions

### `bids_sub`

```python
bids_sub(num)
```

---

### `bids_ses`

```python
bids_ses(num)
```

---

### `classify_cued_recall_file`

Classify a cued recall behavioral file.

```python
classify_cued_recall_file(filename, subj_num, bids_ses_num)
```

---

### `classify_free_recall_behavioral`

Classify a free recall session behavioral file.

```python
classify_free_recall_behavioral(filename, subj_num, bids_ses_num)
```

---

### `classify_eyetracking_file`

Classify an eyetracking file (EDF, AOI, or audio).

```python
classify_eyetracking_file(filename, subj_num, bids_ses_num, subdir = '')
```

---

### `classify_final_session_file`

Classify a final session (ses-30) file.

```python
classify_final_session_file(filename, subj_num, subdir)
```

---

### `determine_bids_session`

Determine session type from BIDS session number.

```python
determine_bids_session(ses_num)
```

---

### `walk_subject`

Walk all sessions for a subject and classify files.

```python
walk_subject(subj_num)
```

---

