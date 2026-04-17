#!/bin/bash
#SBATCH --job-name=resample_func
#SBATCH --partition=compute
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/resample_func_%j.out
#SBATCH --error=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/resample_func_%j.err
#SBATCH --exclude=n0119,n0123,n0128,n0130,n0131,n0190

#==============================================================================
# Post-hoc native-space BOLD resampling (Option B)
#==============================================================================
# Applies cached HMC + SDC transforms to NORDIC-denoised BOLD data,
# producing func-space output equivalent to fMRIPrep's --output-spaces func
# but without re-running the full pipeline.
#
# STC is intentionally skipped (see docs/func-space-plan.md).
#
# Usage:
#   sbatch --export=ALL,FPREP_SUBJECT=sub-03,FPREP_SESSION=ses-04 resample_bold_to_func.sbatch
#
# To process all sessions for a subject:
#   for ses in $(seq -f "ses-%02g" 1 30); do
#     sbatch --export=ALL,FPREP_SUBJECT=sub-03,FPREP_SESSION=$ses resample_bold_to_func.sbatch
#   done
#==============================================================================

if [ -z "${FPREP_SUBJECT}" ] || [ -z "${FPREP_SESSION}" ]; then
    echo "ERROR: FPREP_SUBJECT and FPREP_SESSION must be set."
    echo "Submit with: sbatch --export=ALL,FPREP_SUBJECT=sub-XX,FPREP_SESSION=ses-YY resample_bold_to_func.sbatch"
    exit 1
fi

echo "=========================================="
echo "resample_bold_to_func — SLURM Job Info"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "CPUs: ${SLURM_CPUS_PER_TASK}"
echo "Memory: ${SLURM_MEM_PER_NODE}M"
echo "Subject: ${FPREP_SUBJECT}"
echo "Session: ${FPREP_SESSION}"
echo "Start Time: $(date)"
echo "=========================================="
echo ""

export REPO_ROOT="/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata"
source "${REPO_ROOT}/scripts/load_config.sh"

FMRIPREP_IMAGE="${SINGULARITY_DIR}/fmriprep-24.1.1.simg"
if [ ! -f "${FMRIPREP_IMAGE}" ]; then
    echo "ERROR: fMRIPrep container not found: ${FMRIPREP_IMAGE}"
    exit 1
fi

WORK_DIR="${REPO_ROOT}/work_resample_func/${FPREP_SUBJECT}_${FPREP_SESSION}_${SLURM_JOB_ID}"
mkdir -p "${WORK_DIR}"

singularity exec \
    -B "${BIDS_DIR}:${BIDS_DIR}" \
    -B "${REPO_ROOT}:${REPO_ROOT}" \
    -B "${WORK_DIR}:${WORK_DIR}" \
    "${FMRIPREP_IMAGE}" \
    python3 "${REPO_ROOT}/scripts/resample_bold_to_func.py" \
        --subject "${FPREP_SUBJECT}" \
        --session "${FPREP_SESSION}" \
        --bids-dir "${BIDS_DIR}" \
        --work-dir "${WORK_DIR}"

EXIT_STATUS=$?

echo ""
echo "=========================================="
echo "Job Completion"
echo "=========================================="
echo "End Time: $(date)"
echo "Exit Status: ${EXIT_STATUS}"
echo "=========================================="

# Clean up work directory on success
if [ ${EXIT_STATUS} -eq 0 ] && [ -d "${WORK_DIR}" ]; then
    echo "Cleaning up work directory: ${WORK_DIR}"
    rm -rf "${WORK_DIR}"
else
    echo "Preserving work directory for debugging: ${WORK_DIR}"
fi

exit ${EXIT_STATUS}
