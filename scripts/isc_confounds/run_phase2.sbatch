#!/bin/bash
#SBATCH --job-name=isc_confound_ph2
#SBATCH --output=isc_confound_ph2_%j.out
#SBATCH --error=isc_confound_ph2_%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --partition=compute

PYTHON=/home/bhutch/.conda/envs/neuroconda3/bin/python
SCRIPT_DIR=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/scripts/isc_confounds

echo "Start: $(date)"
echo "Node: $(hostname)"

PYTHONUNBUFFERED=1 ${PYTHON} "${SCRIPT_DIR}/isc_confound_comparison.py" --all-sessions

echo "End: $(date)"
