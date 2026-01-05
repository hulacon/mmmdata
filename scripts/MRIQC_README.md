# MRIQC Setup and Usage Guide

This directory contains scripts to run MRIQC (MRI Quality Control) on the MMMData BIDS dataset.

## Overview

MRIQC extracts no-reference image quality metrics (IQMs) from structural (T1w, T2w) and functional (BOLD) MRI data. It provides:
- Individual participant HTML reports with quality metrics
- Group-level reports for comparing quality across subjects/sessions
- JSON files with quantitative metrics for further analysis

## Installation

MRIQC will be automatically downloaded as a Singularity image when you first run the script. The image will be stored in:
```
/projects/hulacon/shared/mmmdata/singularity_images/mriqc-24.0.0.simg
```

**No manual installation required!** The script handles everything.

## Usage

### Option 1: Python Script (Recommended)

The Python script integrates with the existing MMMData codebase and configuration.

#### Basic Usage - Process All Subjects/Sessions
```bash
cd /gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/scripts
python run_mriqc.py
```

#### Process Specific Subjects
```bash
python run_mriqc.py --subjects sub-01 sub-02
```

#### Adjust Resources
```bash
python run_mriqc.py --nprocs 8 --mem-gb 32
```

#### Group-Level Analysis (after participant-level completes)
```bash
python run_mriqc.py --analysis-level group
```

#### All Options
```bash
python run_mriqc.py --help
```

Available options:
- `--analysis-level`: `participant` (default) or `group`
- `--subjects`: Space-separated list of subjects (e.g., `sub-01 sub-02`)
- `--nprocs`: Number of parallel processes (default: 4)
- `--mem-gb`: Memory limit in GB (default: 16)
- `--fs-license`: Path to FreeSurfer license file (optional)
- `--mriqc-version`: MRIQC version (default: 24.0.0)
- `--bids-dir`: Override BIDS directory from config
- `--output-dir`: Override output directory

### Option 2: Bash Script

A standalone bash script is also available:

```bash
cd /gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/scripts
./run_mriqc.sh participant  # Process all participants
./run_mriqc.sh group        # Generate group reports
```

## Output Location

Results will be saved to:
```
/projects/hulacon/shared/mmmdata/derivatives/mriqc/
```

### Output Structure
```
derivatives/mriqc/
├── sub-01_ses-01_T1w.html              # Individual anatomical reports
├── sub-01_ses-01_task-*_bold.html      # Individual functional reports
├── sub-01_ses-01_T1w.json              # Quantitative metrics (JSON)
├── sub-01_ses-01_task-*_bold.json      # Quantitative metrics (JSON)
├── group_T1w.html                      # Group anatomical report
├── group_bold.html                     # Group functional report
├── dataset_description.json            # BIDS metadata
└── work/                               # Temporary working directory
```

## What MRIQC Will Process

Based on your dataset, MRIQC will generate quality metrics for:

### Anatomical Data (T1w)
- One report per session that has anatomical data
- Metrics: SNR, CNR, tissue contrast, artifacts, etc.

### Functional Data (BOLD)
- One report per run for each task:
  - Localizer tasks: PRF, Floc, Motor, Auditory, Tonotopy
  - Memory tasks: Encoding, Cued Recall, 2-AFC, Final Cued Recall
  - Naturalistic: Movie Viewing, Free Recall
  - Resting state, Math task
- Metrics: tSNR, framewise displacement, motion parameters, artifacts, etc.

## Expected Runtime

- **First run**: Image download (~5-10 minutes) + processing
- **Participant-level**: ~10-30 minutes per subject-session (varies by number of runs)
- **Group-level**: ~5-10 minutes

For 3 subjects with multiple sessions, expect several hours for complete participant-level processing.

## Recommended Workflow

1. **Start with a single subject** to test:
   ```bash
   python run_mriqc.py --subjects sub-01
   ```

2. **Review the output** to ensure everything looks correct

3. **Run on all subjects**:
   ```bash
   python run_mriqc.py
   ```

4. **Generate group reports** after participant-level completes:
   ```bash
   python run_mriqc.py --analysis-level group
   ```

## Customization Options

### Adjust Computational Resources

If you have access to more cores/memory:
```bash
python run_mriqc.py --nprocs 16 --mem-gb 64
```

### Process Only Anatomical or Functional Data

Edit the script and add to the MRIQC command:
- `--modalities T1w` - Only anatomical
- `--modalities bold` - Only functional
- `--modalities T1w bold` - Both (default)

### FreeSurfer Metrics (Optional)

If you have a FreeSurfer license, you can enable additional metrics:
```bash
python run_mriqc.py --fs-license /path/to/license.txt
```

### Use Different MRIQC Version

```bash
python run_mriqc.py --mriqc-version 23.2.0
```

## Troubleshooting

### Out of Memory
Reduce `--nprocs` or `--mem-gb`:
```bash
python run_mriqc.py --nprocs 2 --mem-gb 8
```

### Singularity Errors
Ensure Singularity/Apptainer is available:
```bash
singularity --version
```

### Disk Space
MRIQC generates substantial temporary files. Ensure adequate space in:
- Output directory: `derivatives/mriqc/`
- Work directory: `derivatives/mriqc/work/`

You can clean up the work directory after successful completion:
```bash
rm -rf /projects/hulacon/shared/mmmdata/derivatives/mriqc/work/
```

## Integration with Existing Pipeline

The MRIQC outputs can be integrated with your existing fMRIPrep derivatives:

```python
from core.bids_utils import get_subject_summary
import pandas as pd
import json

# Get MRIQC metrics for a subject
subject = 'sub-01'
mriqc_dir = '/projects/hulacon/shared/mmmdata/derivatives/mriqc'

# Load BOLD metrics
bold_metrics = []
for json_file in Path(mriqc_dir).glob(f'{subject}_*_bold.json'):
    with open(json_file) as f:
        metrics = json.load(f)
        bold_metrics.append(metrics)

df = pd.DataFrame(bold_metrics)
print(df[['subject_id', 'session_id', 'task_id', 'fd_mean', 'tsnr']])
```

## Documentation

- MRIQC Documentation: https://mriqc.readthedocs.io/
- MRIQC GitHub: https://github.com/nipreps/mriqc
- Output Metrics: https://mriqc.readthedocs.io/en/latest/measures.html

## Support

For issues with:
- **This script**: Contact the MMMData team
- **MRIQC itself**: https://github.com/nipreps/mriqc/issues
- **NeuroStars forum**: https://neurostars.org/ (tag: mriqc)
