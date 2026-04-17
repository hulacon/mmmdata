#!/bin/bash
#SBATCH --job-name=fmriprep_func
#SBATCH --partition=compute
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/fmriprep_func_%A_%a.out
#SBATCH --error=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/fmriprep_func_%A_%a.err
#SBATCH --mail-user=your-email@uoregon.edu
#SBATCH --exclude=n0119,n0123,n0128,n0130,n0131,n0190

#==============================================================================
# fMRIPrep Functional Preprocessing - Per Session (SLURM Array Job)
#==============================================================================
# Stage 2 of the split fMRIPrep pipeline. Runs functional preprocessing for
# one session at a time, reusing cached anatomical results from Stage 1
# (fmriprep_anat.sbatch).
#
# IMPORTANT: This script requires FMRIPREP_SUBJECT to be set. Submit with:
#   sbatch --export=ALL,FMRIPREP_SUBJECT=sub-03 --array=1-29 fmriprep_func.sbatch
#
# The --array range should match the number of sessions for the subject.
# Sessions are discovered automatically from the BIDS directory.
#==============================================================================

# Validate required environment variable
if [ -z "${FMRIPREP_SUBJECT}" ]; then
    echo "ERROR: FMRIPREP_SUBJECT not set."
    echo "Submit with: sbatch --export=ALL,FMRIPREP_SUBJECT=sub-XX --array=1-N fmriprep_func.sbatch"
    exit 1
fi

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
echo "Subject: ${FMRIPREP_SUBJECT}"
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
export SINGULARITY_TMPDIR=${CODE_ROOT}/tmp/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p ${SINGULARITY_TMPDIR}

# Discover sessions for this subject
SESSIONS=($(ls -d ${BIDS_DIR}/${FMRIPREP_SUBJECT}/ses-* | xargs -n 1 basename | sort))

if [ ${#SESSIONS[@]} -eq 0 ]; then
    echo "ERROR: No sessions found for ${FMRIPREP_SUBJECT} in ${BIDS_DIR}"
    exit 1
fi

# Check if SLURM_ARRAY_TASK_ID is within bounds
if [ ${SLURM_ARRAY_TASK_ID} -gt ${#SESSIONS[@]} ]; then
    echo "ERROR: Array task ID ${SLURM_ARRAY_TASK_ID} exceeds number of sessions (${#SESSIONS[@]})"
    exit 1
fi

# Get session for this array task (array is 1-indexed)
SESSION=${SESSIONS[$((SLURM_ARRAY_TASK_ID - 1))]}

echo "Processing: ${FMRIPREP_SUBJECT} / ${SESSION}"
echo "Session index: ${SLURM_ARRAY_TASK_ID} of ${#SESSIONS[@]}"
echo ""

# Change to script directory
cd ${SCRIPT_DIR}

# Run fMRIPrep functional preprocessing for this session
echo "Running fMRIPrep functional preprocessing for ${FMRIPREP_SUBJECT} ${SESSION}..."
echo ""

# Determine fmriprep derivatives dir for pre-computed anat results
FMRIPREP_DERIVATIVES="${BIDS_DIR}/derivatives/fmriprep"

python run_fmriprep.py \
    --subjects ${FMRIPREP_SUBJECT} \
    --session ${SESSION} \
    --derivatives ${FMRIPREP_DERIVATIVES} \
    --nprocs ${SLURM_CPUS_PER_TASK} \
    --mem-gb $((${SLURM_MEM_PER_NODE} / 1024))

# Capture exit status
EXIT_STATUS=$?

# Print completion information
echo ""
echo "=========================================="
echo "Job Completion Information"
echo "=========================================="
echo "Subject: ${FMRIPREP_SUBJECT}"
echo "Session: ${SESSION}"
echo "End Time: $(date)"
echo "Exit Status: ${EXIT_STATUS}"
echo "=========================================="

# Clean up temporary files for this task
if [ -d "${SINGULARITY_TMPDIR}" ]; then
    echo "Cleaning up temporary files..."
    rm -rf ${SINGULARITY_TMPDIR}
fi

exit ${EXIT_STATUS}
