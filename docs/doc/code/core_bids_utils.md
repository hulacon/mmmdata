---
title: bids_utils
parent: Core
grand_parent: Code Documentation
nav_order: 51.01
---

# bids_utils

BIDS dataset utilities using pybids.

**Source:** `src/python/core/bids_utils.py`
{: .fs-3 .text-grey-dk-000 }

---

## Functions

### `summarize_bids_dataset`

Summarize the contents of a BIDS dataset using pybids.

If bids_dir is not provided, loads it from the configuration file.

```python
summarize_bids_dataset(bids_dir: Optional[str | Path] = None, config: Optional[Dict[str, Any]] = None, verbose: bool = True) -> Dict[str, Any]
```

**Parameters**

- **`bids_dir`** (`str or Path, optional`) — Path to the BIDS dataset directory. If None, loads from config.
- **`config`** (`dict, optional`) — Pre-loaded configuration dictionary. If None, loads from config files.
- **`verbose`** (`bool, default=True`) — If True, prints summary information to stdout.

**Returns**

dict
    Dictionary containing dataset summary with keys:
    - 'dataset_path': Path to the dataset
    - 'n_subjects': Number of subjects
    - 'subjects': List of subject IDs
    - 'n_sessions': Number of sessions (total across all subjects)
    - 'sessions': List of unique session IDs
    - 'datatypes': List of datatypes (e.g., 'anat', 'func')
    - 'modalities': List of modalities (e.g., 'T1w', 'bold')
    - 'tasks': List of task names (for func data)
    - 'layout': BIDSLayout object for further querying

**Examples**

```python
>>> # Use config file
>>> summary = summarize_bids_dataset()
>>> print(f"Found {summary['n_subjects']} subjects")

>>> # Specify dataset path directly
>>> summary = summarize_bids_dataset('/path/to/bids/dataset')
>>> layout = summary['layout']
>>> files = layout.get(subject='01', suffix='T1w')
```

---

### `get_subject_summary`

Get a detailed summary of all files for a specific subject.

```python
get_subject_summary(subject_id: str, layout: Optional[BIDSLayout] = None, bids_dir: Optional[str | Path] = None) -> pd.DataFrame
```

**Parameters**

- **`subject_id`** (`str`) — Subject ID (without 'sub-' prefix)
- **`layout`** (`BIDSLayout, optional`) — Pre-initialized BIDSLayout object. If None, creates one from bids_dir.
- **`bids_dir`** (`str or Path, optional`) — Path to BIDS dataset. Required if layout is None.

**Returns**

pd.DataFrame
    DataFrame with one row per file, containing BIDS entities and file paths.

---

