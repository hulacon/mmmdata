#!/bin/bash
#SBATCH --job-name=mriqc_array
#SBATCH --partition=compute
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=1-3
#SBATCH --output=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/mriqc_array_%A_%a.out
#SBATCH --error=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/mriqc_array_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your-email@uoregon.edu

#==============================================================================
# MRIQC Participant-Level Analysis (SLURM Array Job)
#==============================================================================
# This script runs MRIQC on individual subjects in parallel using SLURM
# array jobs. Each array task processes one subject.
#
# IMPORTANT: Update the --array parameter to match your number of subjects
#            (e.g., --array=1-3 for 3 subjects: sub-03, sub-04, sub-05)
#
# Usage:
#   sbatch mriqc_array.sbatch
#
# Output:
#   - Logs: /gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/
#   - Results: /projects/hulacon/shared/mmmdata/derivatives/mriqc/
#==============================================================================

# Print job information
echo "=========================================="
echo "SLURM Job Information"
echo "=========================================="
echo "Job ID: ${SLURM_ARRAY_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
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

# Set up environment
export SINGULARITYENV_TEMPLATEFLOW_HOME=/projects/hulacon/shared/mmmdata/.cache/templateflow
export SINGULARITY_TMPDIR=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/tmp/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p ${SINGULARITY_TMPDIR}

# Project directories
BIDS_DIR="/projects/hulacon/shared/mmmdata"
SCRIPT_DIR="/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/scripts"
VENV_DIR="/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/.venv"

# Define subject list
# Automatically populate from BIDS directory
SUBJECTS=($(ls -d ${BIDS_DIR}/sub-* | xargs -n 1 basename | sort))

# Check if SLURM_ARRAY_TASK_ID is within bounds
if [ ${SLURM_ARRAY_TASK_ID} -gt ${#SUBJECTS[@]} ]; then
    echo "ERROR: Array task ID ${SLURM_ARRAY_TASK_ID} exceeds number of subjects (${#SUBJECTS[@]})"
    exit 1
fi

# Get subject for this array task (array is 1-indexed)
SUBJECT=${SUBJECTS[$((SLURM_ARRAY_TASK_ID - 1))]}

echo "Processing subject: ${SUBJECT}"
echo "Subject index: ${SLURM_ARRAY_TASK_ID} of ${#SUBJECTS[@]}"
echo ""

# Activate Python virtual environment
source ${VENV_DIR}/bin/activate

# Change to script directory
cd ${SCRIPT_DIR}

# Run MRIQC on this subject
echo "Running MRIQC participant-level analysis for ${SUBJECT}..."
echo ""

python run_mriqc.py \
    --analysis-level participant \
    --subjects ${SUBJECT} \
    --nprocs ${SLURM_CPUS_PER_TASK} \
    --mem-gb $((${SLURM_MEM_PER_NODE} / 1024))

# Capture exit status
EXIT_STATUS=$?

# Print completion information
echo ""
echo "=========================================="
echo "Job Completion Information"
echo "=========================================="
echo "Subject: ${SUBJECT}"
echo "End Time: $(date)"
echo "Exit Status: ${EXIT_STATUS}"
echo "=========================================="

# Clean up temporary files for this task
if [ -d "${SINGULARITY_TMPDIR}" ]; then
    echo "Cleaning up temporary files..."
    rm -rf ${SINGULARITY_TMPDIR}
fi

exit ${EXIT_STATUS}
