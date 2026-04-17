#!/bin/bash
#SBATCH --job-name=nordic_isfc
#SBATCH --partition=compute
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/nordic_isfc_%j.out
#SBATCH --error=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/nordic_isfc_%j.err
#SBATCH --exclude=n0119,n0123,n0128,n0130,n0131,n0190

#==============================================================================
# NORDIC ISFC Validation
#==============================================================================
# Computes inter-subject functional correlation within ROIs for NATencoding
# movie clips, comparing original vs NORDIC-denoised fMRIPrep outputs.
#
# Usage:
#   sbatch nordic_isfc_validation.sbatch
#==============================================================================

echo "=========================================="
echo "NORDIC ISFC Validation"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Start Time: $(date)"
echo "=========================================="
echo ""

export REPO_ROOT="/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata"
source "${REPO_ROOT}/.venv/bin/activate"

python ${REPO_ROOT}/scripts/nordic_isfc_validation.py

EXIT_STATUS=$?

echo ""
echo "=========================================="
echo "End Time: $(date)"
echo "Exit Status: ${EXIT_STATUS}"
echo "=========================================="

exit ${EXIT_STATUS}
