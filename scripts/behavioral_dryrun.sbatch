#!/bin/bash
#SBATCH --job-name=beh_dryrun
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:45:00
#SBATCH --output=behavioral_dryrun_%j.out
#SBATCH --error=behavioral_dryrun_%j.err

# Step 6: Full dry-run of all behavioral/physio BIDS converters
# Runs all 7 converter types against file_inventory.csv in dry-run mode.
# Expected: ~15 min (mostly physio_dcm DICOM parsing), <500 MB RAM.

PYTHON=/home/bhutch/.local/envs/viz2psy/bin/python3
CONVERTER_DIR=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/raw2bids_converters

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "Python: $PYTHON"
echo "Working dir: $CONVERTER_DIR"
echo ""

cd "$CONVERTER_DIR" && $PYTHON run_all.py --dry-run --verbose

echo ""
echo "Exit code: $?"
echo "Job finished: $(date)"
