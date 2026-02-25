---
title: behavioral_to_beh
parent: Raw-to-BIDS Converters
grand_parent: Code Documentation
nav_order: 53.01
---

# behavioral_to_beh

Convert out-of-scanner behavioral CSVs into BIDS _beh.tsv files.

Handles files with conversion_type='behavioral_to_beh' (51 files total):
  - Cued recall outscan recognition (45 files, 15 per subject -> ses-04 to ses-18)
  - Final recognition (3 files, 1 per subject -> ses-30)
  - Final timeline sequence (3 files, 1 per subject -> ses-30)

These are self-paced tasks performed outside the scanner, so there are no
scanner-relative timings. The output goes in beh/ subdirectories.

For recognition tasks, onset is computed as cumulative response time
(each trial starts when the previous response ends).

Usage:
    python behavioral_to_beh.py <behavioral_csv> [<output_beh_tsv>] [--dry-run]

**Source:** `src/python/raw2bids_converters/behavioral_to_beh.py`
{: .fs-3 .text-grey-dk-000 }

---

## Functions

### `detect_task`

Detect behavioral task type from filename.

```python
detect_task(csv_path)
```

---

### `parse_subj_sess_run`

Extract subject, session, run from behavioral filename.

```python
parse_subj_sess_run(csv_path)
```

---

### `convert_outscan_recognition`

Convert cued recall outscan recognition -> BIDS beh TSV.

Columns: subjId, session, run, trial, cueId, pairId, mmmId, nsdId, itmno,
word, voiceId, voice, sharedId, enCon, reCon, mmmId_lure, nsdId_lure,
image1, image2, correct_resp, resp, resp_RT, recog, trial_accuracy

onset = cumulative resp_RT (self-paced)
duration = resp_RT for current trial

```python
convert_outscan_recognition(csv_path, output_tsv, dry_run = False)
```

---

### `convert_final_recognition`

Convert final recognition -> BIDS beh TSV.

Columns: subjId, session, run, trial, enSession, enRun, enTrial, pairId,
mmmId, nsdId, itmno, word, voiceId, sharedId, enCon, reCon, voice,
trial_accuracy, cueId, mmmId_lure, nsdId_lure, image1, image2,
ans, resp, resp_RT, recog, accuracy

```python
convert_final_recognition(csv_path, output_tsv, dry_run = False)
```

---

### `convert_final_timeline`

Convert final timeline sequence -> BIDS beh TSV.

Columns: subjId, session, run, trial, enSession, enRun, enTrial, pairId,
mmmId, nsdId, itmno, word, voiceId, sharedId, enCon, reCon, voice,
trial_accuracy, cueId, timeline_RT, timeline_resp

```python
convert_final_timeline(csv_path, output_tsv, dry_run = False)
```

---

### `convert_file`

Convert a behavioral CSV to BIDS beh TSV+JSON.

```python
convert_file(csv_path, output_tsv, dry_run = False)
```

---

