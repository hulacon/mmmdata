#!/bin/bash
#SBATCH --job-name=isc_benchmark
#SBATCH --output=isc_benchmark_%j.out
#SBATCH --error=isc_benchmark_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=compute

PYTHON=/home/bhutch/.conda/envs/neuroconda3/bin/python
SCRIPT_DIR=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/scripts/isc_confounds

echo "Start: $(date)"
echo "Node: $(hostname)"
echo "Script: ${SCRIPT_DIR}/isc_benchmark.py"

PYTHONUNBUFFERED=1 ${PYTHON} "${SCRIPT_DIR}/isc_benchmark.py"

echo "End: $(date)"
