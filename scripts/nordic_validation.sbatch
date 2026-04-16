#!/bin/bash
#SBATCH --job-name=nordic_val
#SBATCH --partition=compute
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/nordic_val_%A_%a.out
#SBATCH --error=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/nordic_val_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --exclude=n0119,n0123,n0128,n0130,n0131,n0190
#SBATCH --array=1-6

#==============================================================================
# NORDIC Validation — SLURM Array Job
#==============================================================================
# Runs nordic_validation.py for each subject/session combination.
# Array task mapping:
#   1: sub-03/ses-04   2: sub-03/ses-19
#   3: sub-04/ses-04   4: sub-04/ses-19
#   5: sub-05/ses-04   6: sub-05/ses-19
#
# Usage:
#   sbatch nordic_validation.sbatch
#==============================================================================

# Define subject/session pairs (1-indexed to match SLURM array)
SUBJECTS=(sub-03 sub-03 sub-04 sub-04 sub-05 sub-05)
SESSIONS=(ses-04 ses-19 ses-04 ses-19 ses-04 ses-19)

IDX=$((SLURM_ARRAY_TASK_ID - 1))
SUBJECT=${SUBJECTS[$IDX]}
SESSION=${SESSIONS[$IDX]}

echo "=========================================="
echo "NORDIC Validation — SLURM Job Info"
echo "=========================================="
echo "Job ID: ${SLURM_ARRAY_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Subject: ${SUBJECT}"
echo "Session: ${SESSION}"
echo "Start Time: $(date)"
echo "=========================================="
echo ""

export REPO_ROOT="/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata"
source "${REPO_ROOT}/.venv/bin/activate"

python ${REPO_ROOT}/scripts/nordic_validation.py \
    --subject ${SUBJECT} \
    --session ${SESSION}

EXIT_STATUS=$?

echo ""
echo "=========================================="
echo "End Time: $(date)"
echo "Exit Status: ${EXIT_STATUS}"
echo "=========================================="

exit ${EXIT_STATUS}
