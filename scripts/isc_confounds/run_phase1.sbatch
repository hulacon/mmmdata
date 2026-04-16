#!/bin/bash
#SBATCH --job-name=isc_confound_ph1
#SBATCH --output=isc_confound_ph1_%j.out
#SBATCH --error=isc_confound_ph1_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=compute

PYTHON=/home/bhutch/.conda/envs/neuroconda3/bin/python
SCRIPT_DIR=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/scripts/isc_confounds

echo "Start: $(date)"
echo "Node: $(hostname)"
echo "Script: ${SCRIPT_DIR}/isc_confound_comparison.py"

PYTHONUNBUFFERED=1 ${PYTHON} "${SCRIPT_DIR}/isc_confound_comparison.py" --sessions ses-19

echo "End: $(date)"
