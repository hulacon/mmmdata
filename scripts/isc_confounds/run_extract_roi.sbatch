#!/bin/bash
#SBATCH --job-name=extract_roi_ts
#SBATCH --output=extract_roi_ts_%j.out
#SBATCH --error=extract_roi_ts_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --partition=compute

PYTHON=/home/bhutch/.conda/envs/neuroconda3/bin/python
SCRIPT_DIR=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/scripts/isc_confounds

echo "Start: $(date)"
echo "Node: $(hostname)"
echo "Script: ${SCRIPT_DIR}/extract_roi_timeseries.py"
echo "Dataset: ${DATASET:-both}"

PYTHONUNBUFFERED=1 ${PYTHON} "${SCRIPT_DIR}/extract_roi_timeseries.py" \
    --dataset "${DATASET:-both}"

echo "End: $(date)"
