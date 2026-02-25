---
title: edf_to_physio
parent: Raw-to-BIDS Converters
grand_parent: Code Documentation
nav_order: 53.03
---

# edf_to_physio

Convert EyeLink EDF files into BIDS _recording-eye_physio.tsv.gz files.

Handles files with conversion_type='edf_to_physio' (87 complete EDF files).
These are EyeLink eyetracking recordings from the free recall sessions
(encoding and retrieval), BIDS ses-19 to ses-28.

Requires: eyelinkio (pip install eyelinkio)

EDF naming: s{Subj}s{Sess}r{Run}{m|r}_YYYY_MM_DD_HH_MM.EDF
  m = encoding (memory), r = recall (retrieval)

Output format:
  - _recording-eye_physio.tsv.gz: continuous eye position + pupil samples
    (no header, tab-separated, gzipped)
  - _recording-eye_physio.json: sampling rate, start time, column names

Usage:
    python edf_to_physio.py <edf_file> [<output_physio_tsv_gz>] [--dry-run]

**Source:** `src/python/raw2bids_converters/edf_to_physio.py`
{: .fs-3 .text-grey-dk-000 }

---

## Functions

### `parse_edf_filename`

Extract subject, session, run, phase from EDF filename.

Returns (subj, sess, run, phase) where phase is 'encoding' or 'retrieval'.

```python
parse_edf_filename(edf_path)
```

---

### `convert_file`

Convert an EDF file to BIDS physio TSV.GZ + JSON sidecar.

```python
convert_file(edf_path, output_tsv_gz, dry_run = False)
```

**Parameters**

- **`edf_path`** (`str`) — Path to the EDF file.
- **`output_tsv_gz`** (`str`) — Destination path for the physio TSV.GZ file.
- **`dry_run`** (`bool`) — If True, print what would be done without writing files.

**Returns**

bool
    True if conversion succeeded.

---

