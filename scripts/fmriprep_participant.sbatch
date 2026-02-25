#!/bin/bash
#SBATCH --job-name=fmriprep_participant
#SBATCH --partition=computelong
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/fmriprep_participant_%j.out
#SBATCH --error=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/fmriprep_participant_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your-email@uoregon.edu

#==============================================================================
# fMRIPrep Participant-Level Preprocessing (All Subjects)
#==============================================================================
# This script runs fMRIPrep preprocessing on all subjects and sessions
# in the MMMData BIDS dataset.
#
# Usage:
#   sbatch fmriprep_participant.sbatch
#
# Output:
#   - Logs: /gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/
#   - Results: /projects/hulacon/shared/mmmdata/derivatives/fmriprep/
#==============================================================================

# Print job information
echo "=========================================="
echo "SLURM Job Information"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Job Name: ${SLURM_JOB_NAME}"
echo "Partition: ${SLURM_JOB_PARTITION}"
echo "Node: ${SLURMD_NODENAME}"
echo "CPUs: ${SLURM_CPUS_PER_TASK}"
echo "Memory: ${SLURM_MEM_PER_NODE}M"
echo "Start Time: $(date)"
echo "=========================================="
echo ""

# Activate Python virtual environment first (needed for tomllib in config loader)
export REPO_ROOT="/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata"
source "${REPO_ROOT}/.venv/bin/activate"

# Load configuration from config files
source "${REPO_ROOT}/scripts/load_config.sh"

# Set up environment using config values
export SINGULARITYENV_TEMPLATEFLOW_HOME=${BIDS_DIR}/.cache/templateflow
export SINGULARITY_TMPDIR=${CODE_ROOT}/tmp
mkdir -p ${SINGULARITY_TMPDIR}

# Change to script directory
cd ${SCRIPT_DIR}

# Run fMRIPrep on all subjects
echo "Running fMRIPrep participant-level preprocessing..."
echo ""

python run_fmriprep.py \
    --nprocs ${SLURM_CPUS_PER_TASK} \
    --mem-gb $((${SLURM_MEM_PER_NODE} / 1024))

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

# Clean up temporary files
if [ -d "${SINGULARITY_TMPDIR}" ]; then
    echo "Cleaning up temporary files..."
    rm -rf ${SINGULARITY_TMPDIR}/*
fi

exit ${EXIT_STATUS}
