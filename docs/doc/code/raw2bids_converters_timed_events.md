---
title: timed_events
parent: Raw-to-BIDS Converters
grand_parent: Code Documentation
nav_order: 53.1
---

# timed_events

Convert behavioral CSV + timing CSV pairs into BIDS _events.tsv files.

Handles these file types (conversion_type='timed_events'):
  - Cued recall encoding  (ses-04 to ses-18)
  - Cued recall math      (ses-04 to ses-18)
  - Cued recall retrieval (ses-04 to ses-18)
  - Free recall math      (ses-19 to ses-28)
  - Final cued recall     (ses-30)

Each source behavioral CSV is paired with a timing CSV that provides
scanner-relative onset/offset values. The behavioral CSV provides trial-level
stimulus and response data.

Usage:
    python timed_events.py <behavioral_csv> <timing_csv> <output_events_tsv> [--dry-run]

**Source:** `src/python/raw2bids_converters/timed_events.py`
{: .fs-3 .text-grey-dk-000 }

---

## Functions

### `detect_task`

Detect the task type from the behavioral CSV filename.

Returns one of: 'cued_recall_encoding', 'cued_recall_math',
'cued_recall_retrieval', 'free_recall_math', 'final_cued_recall'

```python
detect_task(behavioral_path)
```

---

### `parse_subj_sess_run`

Extract subject, session, run numbers from behavioral filename.

```python
parse_subj_sess_run(behavioral_path)
```

---

### `convert_cued_recall_encoding`

Cued recall encoding -> BIDS events.

Behavioral columns: subjId, session, run, trial, pairId, mmmId, nsdId,
    itmno, word, voiceId, sharedId, enCon, reCon, voice, resp, resp_RT
Timing columns: sub_id, task_id, sess_id, run_id, trial_id,
    stim_image_start, stim_image_end, stim_word_start, stim_word_end,
    stim_fixation_start, stim_fixation_end

onset = stim_image_start (scanner-relative)
duration = stim_image_end - stim_image_start
trial_type = "image"
modality = "visual"

```python
convert_cued_recall_encoding(beh_df, timing_df, subj, sess, run)
```

---

### `convert_cued_recall_math`

Cued recall math -> BIDS events.

Behavioral columns: problem, answer, trial, resp, resp_RT
Timing columns: sub_id, task_id, sess_id, run_id, trial_id,
    stim_image_start, stim_image_end, stim_fixation_start, stim_fixation_end

onset = stim_image_start
duration = stim_image_end - stim_image_start

```python
convert_cued_recall_math(beh_df, timing_df, subj, sess, run)
```

---

### `convert_cued_recall_retrieval`

Cued recall retrieval -> BIDS events.

Behavioral columns: subjId, session, run, trial, cueId, pairId, mmmId,
    nsdId, itmno, word, voiceId, voice, sharedId, enCon, reCon, resp, resp_RT
Timing columns: sub_id, task_id, sess_id, run_id, trial_id,
    stim_image_start, stim_image_end, stim_word_start, stim_word_end,
    stim_fixation_start, stim_fixation_end

onset = stim_word_start (word cue; stim_image columns are empty)
duration = stim_word_end - stim_word_start

Rest trials: behavioral fields are empty, timing duplicates previous trial.
For rest trials, use fixation_end of previous trial as onset_actual,
and scheduled duration (0.54s) as duration_actual.

```python
convert_cued_recall_retrieval(beh_df, timing_df, subj, sess, run)
```

---

### `convert_free_recall_math`

Free recall math -> BIDS events.

Same structure as cued recall math. Behavioral columns: problem, answer,
trial, resp, resp_RT.

```python
convert_free_recall_math(beh_df, timing_df, subj, sess, run)
```

---

### `convert_final_cued_recall`

Final cued recall -> BIDS events.

Behavioral columns: subjId, session, run, trial, enSession, enRun,
    enTrial, pairId, mmmId, nsdId, itmno, word, voiceId, sharedId,
    enCon, reCon, voice, trial_accuracy, cueId, resp, resp_RT
Timing columns: same as cued recall retrieval (stim_image empty, uses stim_word)

Same structure as cued recall retrieval but with extra columns
(enSession, enRun, enTrial, trial_accuracy).

```python
convert_final_cued_recall(beh_df, timing_df, subj, sess, run)
```

---

### `get_sidecar`

Return the appropriate JSON sidecar definition for a task.

```python
get_sidecar(task)
```

---

### `convert_file`

Convert a behavioral+timing CSV pair to a BIDS events TSV+JSON.

```python
convert_file(behavioral_csv, timing_csv, output_tsv, dry_run = False)
```

**Parameters**

- **`behavioral_csv`** (`str`) — Path to the behavioral data CSV.
- **`timing_csv`** (`str`) — Path to the companion timing CSV.
- **`output_tsv`** (`str`) — Destination path for the BIDS events TSV.
- **`dry_run`** (`bool`) — If True, print what would be done without writing files.

**Returns**

bool
    True if conversion succeeded.

---

### `find_timing_csv`

Locate the companion timing CSV for a behavioral CSV.

Searches in timing_data/ or timing/ subdirectory relative to the
behavioral file, or in parent directories.

```python
find_timing_csv(behavioral_csv)
```

---

