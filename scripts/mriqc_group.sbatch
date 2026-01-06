#!/bin/bash
#SBATCH --job-name=mriqc_group
#SBATCH --partition=compute
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/mriqc_group_%j.out
#SBATCH --error=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/mriqc_group_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your-email@uoregon.edu

#==============================================================================
# MRIQC Group-Level Analysis
#==============================================================================
# This script generates group-level MRIQC reports that summarize quality
# metrics across all subjects and sessions.
#
# IMPORTANT: Run this AFTER participant-level analysis is complete.
#
# Usage:
#   sbatch mriqc_group.sbatch
#
# Output:
#   - Logs: /gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/
#   - Results: /projects/hulacon/shared/mmmdata/derivatives/mriqc/
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

# Load required modules (if any)
# module load singularity  # Uncomment if singularity is a module

# Load configuration from config files
SCRIPT_DIR_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR_HERE}/load_config.sh"

# Set up environment using config values
export SINGULARITYENV_TEMPLATEFLOW_HOME=${BIDS_DIR}/.cache/templateflow
export SINGULARITY_TMPDIR=${CODE_ROOT}/tmp
mkdir -p ${SINGULARITY_TMPDIR}

# Activate Python virtual environment
source ${VENV_DIR}/bin/activate

# Change to script directory
cd ${SCRIPT_DIR}

# Run MRIQC group-level analysis
echo "Running MRIQC group-level analysis..."
echo ""

python run_mriqc.py \
    --analysis-level group \
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
