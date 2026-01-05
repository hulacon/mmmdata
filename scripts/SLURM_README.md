# MRIQC SLURM Batch Job Guide

This guide explains how to run MRIQC on the MMMData dataset using SLURM batch jobs on the HPC cluster.

## Overview

Three SLURM batch scripts are provided:

1. **`mriqc_participant.sbatch`** - Process all subjects sequentially in a single job
2. **`mriqc_array.sbatch`** - Process subjects in parallel using SLURM array jobs (RECOMMENDED)
3. **`mriqc_group.sbatch`** - Generate group-level reports after participant-level processing

## Cluster Configuration

Based on your HPC cluster, the available partitions are:

| Partition | Max Time | Default Memory | Best For |
|-----------|----------|----------------|----------|
| `compute` | 1 day | 4GB/CPU | Standard jobs |
| `computelong` | 14 days | 4GB/CPU | Long-running jobs |
| `compute_intel` | 1 day | 4GB/CPU | Intel-specific jobs |
| `computelong_intel` | 14 days | 4GB/CPU | Long Intel jobs |
| `interactive` | 12 hours | 4GB/CPU | Testing/debugging |

## Quick Start

### Step 1: Update Email Notifications

Edit each `.sbatch` file and replace `your-email@uoregon.edu` with your email address:

```bash
cd /gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/scripts
sed -i 's/your-email@uoregon.edu/your.email@uoregon.edu/g' *.sbatch
```

### Step 2: Submit Participant-Level Analysis

**Option A: Array Job (RECOMMENDED - Fastest)**

Process subjects in parallel:

```bash
sbatch mriqc_array.sbatch
```

This submits 3 jobs (one per subject) that run simultaneously.

**Option B: Single Sequential Job**

Process all subjects in one job:

```bash
sbatch mriqc_participant.sbatch
```

This takes longer but uses only one job slot.

### Step 3: Monitor Job Progress

Check job status:
```bash
squeue -u $USER
```

View detailed job info:
```bash
scontrol show job <JOB_ID>
```

Check array job status:
```bash
squeue -u $USER -r
```

### Step 4: Check Logs

Logs are saved to:
```
/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/
```

View real-time output:
```bash
tail -f logs/mriqc_participant_<JOB_ID>.out
```

For array jobs:
```bash
tail -f logs/mriqc_array_<JOB_ID>_<TASK_ID>.out
```

### Step 5: Generate Group Reports

After all participant-level jobs complete successfully:

```bash
sbatch mriqc_group.sbatch
```

## Detailed Usage

### mriqc_participant.sbatch

Processes all subjects and sessions sequentially in a single job.

**Default Resources:**
- Partition: `compute`
- Time: 24 hours
- CPUs: 8
- Memory: 32 GB

**Modify resources:**
```bash
# Edit the SBATCH headers in the file
#SBATCH --cpus-per-task=16    # Use more CPUs
#SBATCH --mem=64G             # Use more memory
#SBATCH --time=48:00:00       # Extend time limit
#SBATCH --partition=computelong  # Use long partition
```

**Submit:**
```bash
sbatch mriqc_participant.sbatch
```

### mriqc_array.sbatch (RECOMMENDED)

Processes each subject in parallel as separate array tasks. Much faster than sequential processing.

**Default Resources (per task):**
- Partition: `compute`
- Time: 12 hours per subject
- CPUs: 8 per subject
- Memory: 32 GB per subject
- Array size: 1-3 (for 3 subjects)

**How it works:**
- Creates one job per subject automatically
- Subjects are processed simultaneously (if resources available)
- Each subject gets its own log file

**Modify array size:**

The script automatically detects subjects, but you can customize:

```bash
# Edit the SBATCH header
#SBATCH --array=1-3           # For subjects sub-03, sub-04, sub-05
#SBATCH --array=1-10          # If you have 10 subjects
#SBATCH --array=1-5%2         # Process 5 subjects, max 2 running simultaneously
```

**Submit:**
```bash
sbatch mriqc_array.sbatch
```

**Monitor array jobs:**
```bash
# View all array tasks
squeue -u $USER -r

# Cancel specific array task
scancel <JOB_ID>_<TASK_ID>

# Cancel all tasks in array
scancel <JOB_ID>
```

### mriqc_group.sbatch

Generates group-level summary reports. Run AFTER participant-level analysis completes.

**Default Resources:**
- Partition: `compute`
- Time: 2 hours
- CPUs: 4
- Memory: 16 GB

**Submit:**
```bash
sbatch mriqc_group.sbatch
```

## Resource Optimization

### Choosing CPU Count

MRIQC scales with more CPUs but with diminishing returns:

- **4 CPUs**: Minimal, slower processing
- **8 CPUs**: Recommended default (good balance)
- **16 CPUs**: Faster, but only ~1.5x speedup vs 8 CPUs
- **32+ CPUs**: Little additional benefit

### Memory Requirements

Typical memory needs per subject:
- **Anatomical only**: 8-16 GB
- **Functional**: 16-32 GB
- **Both**: 32 GB (recommended)

If you encounter out-of-memory errors:
```bash
#SBATCH --mem=64G  # Increase memory
```

### Time Estimates

Per subject (all sessions):
- **Anatomical**: ~30 minutes
- **Functional**: ~2-4 hours (depends on number of runs)
- **Both**: ~3-5 hours

For 3 subjects with dense design:
- **Sequential job**: 10-15 hours total
- **Array job**: 3-5 hours (if all run in parallel)

## Job Dependency Chains

Run group analysis automatically after participant analysis completes:

### Option 1: Manual Dependency
```bash
# Submit participant job
PART_JOB=$(sbatch --parsable mriqc_participant.sbatch)

# Submit group job to run after participant completes
sbatch --dependency=afterok:${PART_JOB} mriqc_group.sbatch
```

### Option 2: Array Job Dependency
```bash
# Submit array job
ARRAY_JOB=$(sbatch --parsable mriqc_array.sbatch)

# Submit group job after all array tasks complete
sbatch --dependency=afterok:${ARRAY_JOB} mriqc_group.sbatch
```

## Troubleshooting

### Job Fails Immediately

Check the error log:
```bash
cat logs/mriqc_participant_<JOB_ID>.err
```

Common issues:
- Python virtual environment not activated
- BIDS directory path incorrect
- Insufficient permissions

### Out of Memory

Increase memory allocation:
```bash
#SBATCH --mem=64G
```

Or reduce parallel processes:
```bash
#SBATCH --cpus-per-task=4
```

### Job Timeout

Increase time limit:
```bash
#SBATCH --time=48:00:00
```

Or use `computelong` partition:
```bash
#SBATCH --partition=computelong
#SBATCH --time=7-00:00:00  # 7 days
```

### Array Job Issues

**Wrong array size:**
```bash
# Check number of subjects
ls -d /projects/hulacon/shared/mmmdata/sub-* | wc -l

# Update array size to match
#SBATCH --array=1-N  # N = number of subjects
```

**Tasks fail individually:**

Check individual task logs:
```bash
cat logs/mriqc_array_<JOB_ID>_<TASK_ID>.err
```

Resubmit failed tasks only:
```bash
#SBATCH --array=2,5,7  # Only rerun tasks 2, 5, and 7
```

### Singularity Image Download Fails

The first run downloads the MRIQC container. If this fails:

1. Check network connectivity
2. Manually download:
   ```bash
   mkdir -p /projects/hulacon/shared/mmmdata/singularity_images
   cd /projects/hulacon/shared/mmmdata/singularity_images
   singularity pull mriqc-24.0.0.simg docker://nipreps/mriqc:24.0.0
   ```

## Advanced Usage

### Process Specific Subjects in Array Job

Modify the `mriqc_array.sbatch` script to define specific subjects:

```bash
# Replace the automatic subject detection with manual list
SUBJECTS=(sub-03 sub-05)  # Only process these subjects

#SBATCH --array=1-2  # Update array size
```

### Test on Interactive Node

For testing/debugging:

```bash
srun --partition=interactive \
     --cpus-per-task=4 \
     --mem=16G \
     --time=2:00:00 \
     --pty bash

# Then run the Python script directly
cd /gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/scripts
source ../.venv/bin/activate
python run_mriqc.py --subjects sub-03
```

### Custom MRIQC Options

Edit `run_mriqc.py` to add custom MRIQC flags. For example, to process only BOLD data:

In the `run_mriqc()` function, add:
```python
cmd.extend(['--modalities', 'bold'])
```

## Checking Results

After jobs complete, check outputs:

```bash
# List all MRIQC outputs
ls -lh /projects/hulacon/shared/mmmdata/derivatives/mriqc/

# Count individual reports
ls /projects/hulacon/shared/mmmdata/derivatives/mriqc/*.html | wc -l

# View group reports
firefox /projects/hulacon/shared/mmmdata/derivatives/mriqc/group_*.html
```

## Best Practices

1. **Start with array job** - Fastest for multiple subjects
2. **Test on one subject first** - Verify setup before processing all
3. **Use job dependencies** - Automate the workflow
4. **Monitor disk space** - MRIQC generates large temporary files
5. **Clean up work directory** - Remove after successful completion
   ```bash
   rm -rf /projects/hulacon/shared/mmmdata/derivatives/mriqc/work/
   ```
6. **Keep logs organized** - Archive old logs periodically
7. **Email notifications** - Ensure you get notified of job completion/failure

## Example Workflows

### Workflow 1: Quick Parallel Processing
```bash
# Process all subjects in parallel, then generate group reports
sbatch mriqc_array.sbatch
# Wait for completion, then:
sbatch mriqc_group.sbatch
```

### Workflow 2: Automated Pipeline
```bash
# Submit everything with dependencies
ARRAY_JOB=$(sbatch --parsable mriqc_array.sbatch)
sbatch --dependency=afterok:${ARRAY_JOB} mriqc_group.sbatch
```

### Workflow 3: Conservative Sequential
```bash
# Process everything in one job (slower but simpler)
sbatch mriqc_participant.sbatch
# After completion:
sbatch mriqc_group.sbatch
```

## Additional Resources

- **SLURM Documentation**: `man sbatch`
- **Check partition info**: `scontrol show partition`
- **View your jobs**: `squeue -u $USER`
- **Job history**: `sacct -u $USER`
- **Cancel job**: `scancel <JOB_ID>`
- **MRIQC Documentation**: https://mriqc.readthedocs.io/

## Support

For cluster-specific issues, contact your HPC support team.
For MRIQC issues, see the MRIQC documentation or NeuroStars forum.
