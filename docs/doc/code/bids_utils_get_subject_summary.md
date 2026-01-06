---
title: get_subject_summary
parent: Bids Utils
grand_parent: Code Documentation
nav_order: 51.02
---

# `get_subject_summary`

Get a detailed summary of all files for a specific subject.

## Signature

```python
get_subject_summary(subject_id: str, layout: Optional[BIDSLayout] = None, bids_dir: Optional[str | Path] = None) -> pd.DataFrame
```

## Parameters

**`subject_id`** : `str`
  
Subject ID (without 'sub-' prefix)

**`layout`** : `BIDSLayout, optional`
  
Pre-initialized BIDSLayout object. If None, creates one from bids_dir.

**`bids_dir`** : `str or Path, optional`
  
Path to BIDS dataset. Required if layout is None.

## Returns

-------
pd.DataFrame
    DataFrame with one row per file, containing BIDS entities and file paths.

## Source

Defined in `bids_utils.py` at line 148
