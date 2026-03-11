#!/bin/bash
#SBATCH --job-name=hippunfold
#SBATCH --partition=compute
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/hippunfold_%A_%a.out
#SBATCH --error=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/hippunfold_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your-email@uoregon.edu

#==============================================================================
# HippUnfold: Hippocampal Subfield Segmentation & Unfolding (SLURM Job)
#==============================================================================
# Runs HippUnfold v1.5.2 on BIDS data using T1w + T2w (acq-SPC) inputs.
# Uses the combined T1T2w nnU-net model for best segmentation quality.
#
# Usage:
#   Single subject:   sbatch --export=SUBJECT=sub-03 hippunfold.sbatch
#   Array (all 3):    sbatch --array=1-3 hippunfold.sbatch
#
# Output goes to: derivatives/hippunfold/
#==============================================================================

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

# Activate Python virtual environment (needed for config loader)
export REPO_ROOT="/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata"
source "${REPO_ROOT}/.venv/bin/activate"

# Load configuration from config files
source "${REPO_ROOT}/scripts/load_config.sh"

# --- Determine subject ---
if [ -n "${SUBJECT}" ]; then
    # Explicit subject passed via --export=SUBJECT=sub-XX
    SUB="${SUBJECT}"
elif [ -n "${SLURM_ARRAY_TASK_ID}" ]; then
    # Array mode: map task ID to subject
    SUBJECTS=($(ls -d ${BIDS_DIR}/sub-* | xargs -n 1 basename | sort))
    if [ ${SLURM_ARRAY_TASK_ID} -gt ${#SUBJECTS[@]} ]; then
        echo "ERROR: Array task ID ${SLURM_ARRAY_TASK_ID} exceeds number of subjects (${#SUBJECTS[@]})"
        exit 1
    fi
    SUB=${SUBJECTS[$((SLURM_ARRAY_TASK_ID - 1))]}
else
    echo "ERROR: Specify subject via --export=SUBJECT=sub-XX or use --array"
    exit 1
fi

# Extract participant label (number only, no "sub-" prefix)
PARTICIPANT_LABEL="${SUB#sub-}"

echo "Processing: ${SUB} (participant label: ${PARTICIPANT_LABEL})"
echo ""

# --- Container setup ---
CONTAINER="${SINGULARITY_DIR}/hippunfold-v1.5.2.sif"
if [ ! -f "${CONTAINER}" ]; then
    echo "ERROR: HippUnfold container not found at ${CONTAINER}"
    exit 1
fi

# Set up temporary directory for Singularity
export SINGULARITY_TMPDIR="${CODE_ROOT}/tmp/hippunfold_${SLURM_JOB_ID}"
mkdir -p "${SINGULARITY_TMPDIR}"

# Bind paths for container access
export SINGULARITY_BINDPATH="/gpfs:/gpfs"

# HippUnfold model cache (downloaded on first run)
export HIPPUNFOLD_CACHE_DIR="${CODE_ROOT}/.cache/hippunfold"
mkdir -p "${HIPPUNFOLD_CACHE_DIR}"

# T2w acquisition: default SPC, override via --export=T2W_ACQ=oblcor
T2W_ACQ="${T2W_ACQ:-SPC}"

# Output directory: separate dirs per T2w acquisition for comparison
if [ "${T2W_ACQ}" = "SPC" ]; then
    OUTPUT_DIR="${BIDS_DIR}/derivatives/hippunfold"
else
    OUTPUT_DIR="${BIDS_DIR}/derivatives/hippunfold_${T2W_ACQ}"
fi
mkdir -p "${OUTPUT_DIR}"

# --- Run HippUnfold ---
echo "Running HippUnfold with T1T2w model..."
echo "  Input:  ${BIDS_DIR}"
echo "  Output: ${OUTPUT_DIR}"
echo "  T2w acquisition: ${T2W_ACQ}"
echo "  Container: ${CONTAINER}"
echo ""

singularity run -e \
    --env HIPPUNFOLD_CACHE_DIR="${HIPPUNFOLD_CACHE_DIR}" \
    "${CONTAINER}" \
    "${BIDS_DIR}" \
    "${OUTPUT_DIR}" \
    participant \
    --participant-label "${PARTICIPANT_LABEL}" \
    --modality T2w \
    --force-nnunet-model T1T2w \
    --t1-reg-template \
    --output-spaces T1w \
    --filter_T1w acquisition=MPR run=01 \
    --filter_T2w acquisition="${T2W_ACQ}" \
    --cores ${SLURM_CPUS_PER_TASK} \
    -p

# Capture exit status
EXIT_STATUS=$?

# --- Completion ---
echo ""
echo "=========================================="
echo "Job Completion Information"
echo "=========================================="
echo "Subject: ${SUB}"
echo "End Time: $(date)"
echo "Exit Status: ${EXIT_STATUS}"
echo "=========================================="

# Clean up temporary files
if [ -d "${SINGULARITY_TMPDIR}" ]; then
    echo "Cleaning up temporary files..."
    rm -rf "${SINGULARITY_TMPDIR}"
fi

exit ${EXIT_STATUS}
