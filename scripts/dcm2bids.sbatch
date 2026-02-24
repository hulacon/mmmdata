#!/bin/bash
#SBATCH --job-name=dcm2bids
#SBATCH --partition=compute
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/dcm2bids_%j.out
#SBATCH --error=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/dcm2bids_%j.err
#SBATCH --mail-type=END,FAIL

#==============================================================================
# dcm2bids DICOM-to-BIDS Conversion
#==============================================================================
# Generates dcm2bids config JSON and runs the conversion for a single
# subject/session pair.
#
# Usage:
#   sbatch --export=SUBJECT=sub-03,SESSION=ses-06 dcm2bids.sbatch
#   sbatch --export=SUBJECT=sub-03,SESSION=all dcm2bids.sbatch
#
# Environment variables:
#   SUBJECT  - Subject ID (required, e.g. sub-03)
#   SESSION  - Session ID (required, e.g. ses-06 or "all")
#==============================================================================

# Validate required variables
if [ -z "${SUBJECT}" ]; then
    echo "ERROR: SUBJECT not set. Use: sbatch --export=SUBJECT=sub-03,SESSION=ses-06 dcm2bids.sbatch"
    exit 1
fi
if [ -z "${SESSION}" ]; then
    echo "ERROR: SESSION not set. Use: sbatch --export=SUBJECT=sub-03,SESSION=ses-06 dcm2bids.sbatch"
    exit 1
fi

# Print job information
echo "=========================================="
echo "SLURM Job Information"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Job Name: ${SLURM_JOB_NAME}"
echo "Node: ${SLURMD_NODENAME}"
echo "Subject: ${SUBJECT}"
echo "Session: ${SESSION}"
echo "Start Time: $(date)"
echo "=========================================="
echo ""

# Load configuration from config files
SCRIPT_DIR_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR_HERE}/load_config.sh"

# Activate Python virtual environment
source ${VENV_DIR}/bin/activate

# Change to script directory
cd ${SCRIPT_DIR}

# Run dcm2bids (config generation + conversion)
python run_dcm2bids.py \
    --subject "${SUBJECT}" \
    --session "${SESSION}" \
    --force

# Capture exit status
EXIT_STATUS=$?

# Print completion information
echo ""
echo "=========================================="
echo "Job Completion Information"
echo "=========================================="
echo "End Time: $(date)"
echo "Exit Status: ${EXIT_STATUS}"
echo "=========================================="

exit ${EXIT_STATUS}
