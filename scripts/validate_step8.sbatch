#!/bin/bash
#SBATCH --job-name=validate_s8
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=validate_step8_%j.out
#SBATCH --error=validate_step8_%j.err

# Step 8: Validate behavioral pipeline output against reference
# Compares new BIDS files in sub-*/ses-*/ against derivatives/bids_validation/eventfiles/

PYTHON=/home/bhutch/.local/envs/viz2psy/bin/python3
SCRIPT_DIR=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata-agents/scripts

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "Python: $PYTHON"
echo ""

$PYTHON "$SCRIPT_DIR/validate_step8.py" --verbose

echo ""
echo "Exit code: $?"
echo "Job finished: $(date)"
