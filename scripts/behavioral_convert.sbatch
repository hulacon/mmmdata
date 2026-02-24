#!/bin/bash
#SBATCH --job-name=beh_convert
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:30:00
#SBATCH --output=behavioral_convert_%j.out
#SBATCH --error=behavioral_convert_%j.err

# Step 7: Full behavioral/physio BIDS conversion
# Runs all 7 converter types (no --dry-run).
# Expected: ~45 min (73 EDFs ~29 min, 232 PhysioLogs ~6 min, rest fast).

PYTHON=/home/bhutch/.local/envs/viz2psy/bin/python3
CONVERTER_DIR=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/raw2bids_converters

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "Python: $PYTHON"
echo "Working dir: $CONVERTER_DIR"
echo ""

cd "$CONVERTER_DIR"
$PYTHON run_all.py --verbose
RC=$?

echo ""
echo "run_all.py exit code: $RC"
echo "Job finished: $(date)"
exit $RC
