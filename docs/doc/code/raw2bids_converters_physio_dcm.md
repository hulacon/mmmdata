---
title: physio_dcm
parent: Raw-to-BIDS Converters
grand_parent: Code Documentation
nav_order: 53.06
---

# physio_dcm

Convert Siemens PhysioLog DICOMs into BIDS _physio.tsv.gz files.

Handles files with conversion_type='physio_dcm'. These are Siemens PMU
physiological recordings embedded in DICOM private tag (7FE1,1010).

Each PhysioLog DICOM contains 5 concatenated log sections:
  ECG  (SampleTime=1ms,  4 channels: ECG1-ECG4)
  PULS (SampleTime=2ms,  1 channel: pulse oximetry)
  RESP (SampleTime=8ms,  1 channel: respiratory belt)
  EXT  (SampleTime=8ms,  external trigger)
  ACQUISITION_INFO (volume/slice timing)

Output per BIDS spec for physio data:
  - _recording-cardiac_physio.tsv.gz  + .json  (ECG channel 1)
  - _recording-pulse_physio.tsv.gz    + .json  (PULS)
  - _recording-respiratory_physio.tsv.gz + .json (RESP)

Requires: pydicom, numpy

Usage:
    python physio_dcm.py <dicom_dir> [<output_dir>] [--dry-run]

**Source:** `src/python/raw2bids_converters/physio_dcm.py`
{: .fs-3 .text-grey-dk-000 }

---

## Functions

### `parse_pmu_text`

Parse the full PMU text into per-section data.

```python
parse_pmu_text(text)
```

**Returns**

sections : dict
    {section_name: {"sample_time": int, "timestamps": ndarray, "values": ndarray}}
acq_info : dict
    {"num_volumes": int, "num_slices": int, "vol_start_tics": dict}

---

### `convert_file`

Convert a PhysioLog DICOM directory to BIDS physio files.

```python
convert_file(dicom_dir, output_base, dry_run = False)
```

**Parameters**

- **`dicom_dir`** (`str`) — Path to the PhysioLog DICOM directory (contains one .dcm file).
- **`output_base`** (`str`) — Base path for output files. The recording entity and suffix will be
- **`appended`** (`e.g. if output_base is`) — sub-03/ses-04/func/sub-03_ses-04_task-CRencoding_run-01 then output files will be: ..._recording-cardiac_physio.tsv.gz ..._recording-pulse_physio.tsv.gz ..._recording-respiratory_physio.tsv.gz
- **`dry_run`** (`bool`) — If True, print what would be done without writing.

**Returns**

bool
    True if conversion succeeded.

---

