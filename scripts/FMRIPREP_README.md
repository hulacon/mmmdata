# fMRIPrep Setup and Usage Guide

This directory contains scripts to run fMRIPrep preprocessing on the MMMData BIDS dataset.

## Overview

fMRIPrep is a robust preprocessing pipeline for fMRI data that produces analysis-ready derivatives including:
- Preprocessed BOLD data in standard space (MNI152NLin2009cAsym)
- Surface-projected data (fsaverage6)
- Confound timeseries (framewise displacement, DVARS, motion parameters, CompCor, etc.)
- Brain masks, tissue segmentations, and spatial transforms
- FreeSurfer cortical surface reconstructions
- HTML quality reports per subject/session

## Prerequisites

### Container Image

The fMRIPrep Singularity image should already be available at:
```
/gpfs/projects/hulacon/shared/mmmdata/code/containers/fmriprep-24.1.1.simg
```

If not, download it:
```bash
singularity pull /gpfs/projects/hulacon/shared/mmmdata/code/containers/fmriprep-24.1.1.simg \
    docker://nipreps/fmriprep:24.1.1
```

### FreeSurfer License

A FreeSurfer license is required and is auto-detected at:
```
/gpfs/projects/hulacon/shared/mmmdata/licenses/fz_unlabeled_license.txt
```

## Usage

### Option 1: Submission Helper (Recommended)

```bash
cd /gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/scripts

# Test with one subject first (recommended)
./submit_fmriprep.sh test

# Process all subjects in parallel (after test succeeds)
./submit_fmriprep.sh array

# Process all subjects sequentially
./submit_fmriprep.sh participant
```

### Option 2: Python Script Directly

```bash
cd /gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/scripts

# Process specific subjects
python run_fmriprep.py --subjects sub-03

# All subjects with custom resources
python run_fmriprep.py --nprocs 16 --mem-gb 64

# Custom output spaces
python run_fmriprep.py --output-spaces MNI152NLin2009cAsym:res-2 fsaverage6 fsLR
```

### Option 3: Direct SLURM Submission

```bash
# Single subject test (array task 1 only)
sbatch --array=1-1 fmriprep_array.sbatch

# All subjects in parallel
sbatch fmriprep_array.sbatch

# All subjects sequentially
sbatch fmriprep_participant.sbatch
```

### All Python Script Options

```
python run_fmriprep.py --help
```

| Option | Default | Description |
|--------|---------|-------------|
| `--subjects` | all | Space-separated subject IDs (e.g., `sub-03 sub-04`) |
| `--nprocs` | 8 | Number of parallel processes |
| `--mem-gb` | 32 | Memory limit in GB |
| `--output-spaces` | `MNI152NLin2009cAsym:res-2 fsaverage6` | fMRIPrep output spaces |
| `--fs-license` | auto-detect | Path to FreeSurfer license |
| `--fs-subjects-dir` | none | Reuse existing FreeSurfer recon-all output |
| `--fmriprep-version` | 24.1.1 | fMRIPrep version for image lookup |
| `--bids-dir` | from config | Override BIDS directory |
| `--output-dir` | `derivatives/fmriprep` | Override output directory |

## Output Location

Results are saved to:
```
/projects/hulacon/shared/mmmdata/derivatives/fmriprep/
```

### Output Structure
```
derivatives/fmriprep/
├── dataset_description.json
├── sub-03_anat.html                     # Anatomical QC report
├── sub-03_ses-01_func.html              # Functional QC reports
├── sub-03/
│   ├── anat/
│   │   ├── sub-03_desc-preproc_T1w.nii.gz          # Preprocessed T1w
│   │   ├── sub-03_desc-brain_mask.nii.gz            # Brain mask
│   │   ├── sub-03_dseg.nii.gz                       # Tissue segmentation
│   │   ├── sub-03_space-MNI152NLin2009cAsym_...     # MNI-space outputs
│   │   ├── sub-03_hemi-L_*.surf.gii                 # Cortical surfaces
│   │   └── sub-03_from-T1w_to-MNI152NLin2009cAsym_*.h5  # Transforms
│   ├── ses-01/
│   │   └── func/
│   │       ├── *_desc-confounds_timeseries.tsv       # Confound regressors
│   │       ├── *_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz
│   │       ├── *_desc-brain_mask.nii.gz
│   │       └── *_from-boldref_to-T1w_*.txt           # Coregistration
│   └── figures/                                       # QC visualizations (SVGs)
├── sourcedata/
│   └── freesurfer/                                    # FreeSurfer recon-all output
└── work/                                              # Temp files (can be deleted)
```

## Reusing FreeSurfer Output

FreeSurfer recon-all takes 6-8 hours per subject. If you have existing output (e.g., from a prior fMRIPrep run), you can reuse it:

```bash
python run_fmriprep.py \
    --fs-subjects-dir /projects/hulacon/shared/mmmdata/derivatives/fmriprep_pre022426/sourcedata/freesurfer
```

This skips the recon-all step and significantly reduces processing time.

## Resource Estimates

| Scenario | Time | CPUs | Memory | Notes |
|----------|------|------|--------|-------|
| 1 subject, all sessions | 12-48h | 8 | 48 GB | Depends on number of sessions |
| 1 subject, reusing FreeSurfer | 6-24h | 8 | 48 GB | Skips recon-all (~6-8h savings) |
| 3 subjects, parallel array | 12-48h total | 8×3 | 48×3 GB | Wall time same as 1 subject |

For 3 subjects with ~29 sessions each, expect the array job to complete in 24-48 hours.

## Recommended Workflow

1. **Test with one subject**:
   ```bash
   ./submit_fmriprep.sh test
   ```

2. **Review the output** — check HTML reports and confound files:
   ```bash
   ls derivatives/fmriprep/sub-03/
   ls derivatives/fmriprep/sub-03/ses-01/func/*confounds*
   ```

3. **Run all subjects** after test succeeds:
   ```bash
   ./submit_fmriprep.sh array
   ```

4. **Clean up work directory** after successful completion:
   ```bash
   rm -rf /projects/hulacon/shared/mmmdata/derivatives/fmriprep/work/
   ```

## Troubleshooting

### Out of Memory
Increase memory allocation:
```bash
python run_fmriprep.py --mem-gb 64
```
Or edit the sbatch file to request more memory (`--mem=64G`).

### Disk Space
fMRIPrep generates substantial temporary files. Ensure adequate space:
- Output: ~100-120 GB per subject
- Work directory: up to 50 GB per subject during processing

### Container Issues
Verify Singularity/Apptainer is available:
```bash
singularity --version
```

### FreeSurfer License Errors
Ensure the license file is readable:
```bash
cat /gpfs/projects/hulacon/shared/mmmdata/licenses/fz_unlabeled_license.txt
```

### Checking Job Status
```bash
squeue -u $USER           # List your running jobs
sacct -j <JOB_ID>         # Detailed job info
tail -f logs/fmriprep_*   # Follow log output
```

## Documentation

- fMRIPrep Documentation: https://fmriprep.org/
- fMRIPrep GitHub: https://github.com/nipreps/fmriprep
- Output Spaces: https://fmriprep.org/en/stable/spaces.html
- Confound Regressors: https://fmriprep.org/en/stable/outputs.html#confounds
