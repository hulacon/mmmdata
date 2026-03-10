#!/bin/bash
#SBATCH --job-name=nordic
#SBATCH --partition=compute
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/nordic_%A_%a.out
#SBATCH --error=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/nordic_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --exclude=n0119,n0123,n0128,n0130,n0131,n0190

#==============================================================================
# NORDIC Denoising — SLURM Array Job
#==============================================================================
# Applies magnitude-only NORDIC PCA denoising to BOLD NIfTIs for one session.
# Each array task processes one BOLD run.
#
# Usage:
#   sbatch --export=ALL,NORDIC_SUBJECT=sub-03,NORDIC_SESSION=ses-04 \
#          --array=1-7 nordic_denoise.sbatch
#
# The --array range should match the number of BOLD runs in the session.
# Runs are discovered automatically from the BIDS func/ directory.
#==============================================================================

# Validate required environment variables
if [ -z "${NORDIC_SUBJECT}" ] || [ -z "${NORDIC_SESSION}" ]; then
    echo "ERROR: NORDIC_SUBJECT and NORDIC_SESSION must be set."
    echo "Submit with: sbatch --export=ALL,NORDIC_SUBJECT=sub-XX,NORDIC_SESSION=ses-YY --array=1-N nordic_denoise.sbatch"
    exit 1
fi

# Print job information
echo "=========================================="
echo "NORDIC Denoising — SLURM Job Info"
echo "=========================================="
echo "Job ID: ${SLURM_ARRAY_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "CPUs: ${SLURM_CPUS_PER_TASK}"
echo "Memory: ${SLURM_MEM_PER_NODE}M"
echo "Subject: ${NORDIC_SUBJECT}"
echo "Session: ${NORDIC_SESSION}"
echo "Start Time: $(date)"
echo "=========================================="
echo ""

# Activate Python venv (needed for load_config.sh)
export REPO_ROOT="/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata"
source "${REPO_ROOT}/.venv/bin/activate"

# Load configuration
source "${REPO_ROOT}/scripts/load_config.sh"

# Paths
FUNC_DIR="${BIDS_DIR}/${NORDIC_SUBJECT}/${NORDIC_SESSION}/func"
OUT_DIR="${BIDS_DIR}/derivatives/nordic/${NORDIC_SUBJECT}/${NORDIC_SESSION}/func"
NORDIC_TOOLBOX="/gpfs/projects/hulacon/shared/mmmdata/code/NORDIC_Raw"

mkdir -p "${OUT_DIR}"

# Discover BOLD runs for this session
BOLD_FILES=($(ls "${FUNC_DIR}"/*_bold.nii.gz 2>/dev/null | sort))

if [ ${#BOLD_FILES[@]} -eq 0 ]; then
    echo "ERROR: No BOLD files found in ${FUNC_DIR}"
    exit 1
fi

echo "Found ${#BOLD_FILES[@]} BOLD runs in ${FUNC_DIR}"

# Check array task bounds
if [ ${SLURM_ARRAY_TASK_ID} -gt ${#BOLD_FILES[@]} ]; then
    echo "Array task ${SLURM_ARRAY_TASK_ID} exceeds run count (${#BOLD_FILES[@]}). Exiting cleanly."
    exit 0
fi

# Get the BOLD file for this array task (1-indexed)
BOLD_IN="${BOLD_FILES[$((SLURM_ARRAY_TASK_ID - 1))]}"
BOLD_BASENAME=$(basename "${BOLD_IN}")

echo "Processing: ${BOLD_BASENAME}"
echo ""

# Check if output already exists
if [ -f "${OUT_DIR}/${BOLD_BASENAME}" ]; then
    echo "Output already exists: ${OUT_DIR}/${BOLD_BASENAME}"
    echo "Skipping. Delete existing output to re-run."
    exit 0
fi

# Load MATLAB and run NORDIC
module load matlab/R2024a

matlab -nodisplay -nosplash -nodesktop -r "\
    addpath('${NORDIC_TOOLBOX}'); \
    addpath('${REPO_ROOT}/scripts'); \
    nordic_denoise('${BOLD_IN}', '${OUT_DIR}'); \
    exit" 2>&1

EXIT_STATUS=$?

# Print completion
echo ""
echo "=========================================="
echo "Job Completion"
echo "=========================================="
echo "File: ${BOLD_BASENAME}"
echo "End Time: $(date)"
echo "Exit Status: ${EXIT_STATUS}"
echo "=========================================="

exit ${EXIT_STATUS}
