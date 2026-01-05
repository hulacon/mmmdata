# MRIQC SLURM Quick Start

## TL;DR - Fastest Way to Run MRIQC

```bash
cd /gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/scripts

# 1. Update your email address (one time only)
sed -i 's/your-email@uoregon.edu/your.actual.email@uoregon.edu/g' *.sbatch

# 2. Submit the full pipeline (recommended)
./submit_mriqc.sh array-pipeline

# Done! Monitor with:
squeue -u $USER
```

## What This Does

1. Downloads MRIQC container (first run only, ~5-10 min)
2. Processes all subjects in parallel (3-5 hours)
3. Automatically generates group reports when complete

## Results Location

```
/projects/hulacon/shared/mmmdata/derivatives/mriqc/
```

## Alternative Methods

### Process subjects one at a time (slower)
```bash
./submit_mriqc.sh pipeline
```

### Process only participant-level (no group reports)
```bash
./submit_mriqc.sh array
```

### Generate group reports only (after participant-level done)
```bash
./submit_mriqc.sh group
```

## Monitor Jobs

```bash
# View running jobs
squeue -u $USER

# View job details
scontrol show job <JOB_ID>

# View logs in real-time
tail -f logs/mriqc_array_*.out

# Cancel a job
scancel <JOB_ID>
```

## Customization

Edit the `.sbatch` files to change:
- CPUs: `#SBATCH --cpus-per-task=8`
- Memory: `#SBATCH --mem=32G`
- Time: `#SBATCH --time=24:00:00`
- Partition: `#SBATCH --partition=compute`

## Troubleshooting

### Job fails immediately
```bash
cat logs/mriqc_*_<JOB_ID>.err
```

### Out of memory
Increase memory in the `.sbatch` file:
```bash
#SBATCH --mem=64G
```

### Need more time
Use the `computelong` partition:
```bash
#SBATCH --partition=computelong
#SBATCH --time=7-00:00:00
```

## Full Documentation

See `SLURM_README.md` for complete documentation.
