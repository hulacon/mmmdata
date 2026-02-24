#!/bin/bash
#SBATCH --job-name=edf_triage
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=edf_triage_%j.out
#SBATCH --error=edf_triage_%j.err

# EDF triage: scan-window quality audit for all 87 EDF files
# Expected: ~10 min, <200 MB RAM

PYTHON=/home/bhutch/.local/envs/viz2psy/bin/python3
SCRIPT_DIR=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata-agents/scripts
SCRIPT="$SCRIPT_DIR/edf_triage.py"

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "Python: $PYTHON"
echo ""

$PYTHON "$SCRIPT"

echo ""
echo "Job finished: $(date)"
