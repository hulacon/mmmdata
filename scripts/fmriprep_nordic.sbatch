#!/bin/bash
#SBATCH --job-name=fmriprep_nordic
#SBATCH --partition=compute
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/fmriprep_nordic_%j.out
#SBATCH --error=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/fmriprep_nordic_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --exclude=n0119,n0123,n0128,n0130,n0131,n0190

#==============================================================================
# fMRIPrep on NORDIC-denoised data — Pilot (sub-03, ses-04)
#==============================================================================
# Runs fMRIPrep functional preprocessing on NORDIC-denoised BOLD data,
# reusing cached anatomical results from the existing fMRIPrep run.
#
# Usage:
#   sbatch fmriprep_nordic.sbatch
#==============================================================================

echo "=========================================="
echo "fMRIPrep (NORDIC) — SLURM Job Info"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "CPUs: ${SLURM_CPUS_PER_TASK}"
echo "Memory: ${SLURM_MEM_PER_NODE}M"
echo "Start Time: $(date)"
echo "=========================================="
echo ""

# Activate Python venv
export REPO_ROOT="/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata"
source "${REPO_ROOT}/.venv/bin/activate"

# Load configuration
source "${REPO_ROOT}/scripts/load_config.sh"

# Paths — use a separate work dir so fMRIPrep does NOT reuse cached HMC/motion
# transforms from the original (non-NORDIC) run
NORDIC_BIDS="${BIDS_DIR}/derivatives/nordic/bids_input"
NORDIC_OUTPUT="${BIDS_DIR}/derivatives/fmriprep_nordic"
NORDIC_WORK="${CODE_ROOT}/work_nordic"
FMRIPREP_DERIVATIVES="${BIDS_DIR}/derivatives/fmriprep"

export SINGULARITYENV_TEMPLATEFLOW_HOME=${BIDS_DIR}/.cache/templateflow
export SINGULARITY_TMPDIR=${CODE_ROOT}/tmp/${SLURM_JOB_ID}
mkdir -p ${SINGULARITY_TMPDIR}

# The NORDIC BIDS tree uses symlinks pointing to files under BIDS_DIR.
# Ensure the main BIDS root is bind-mounted so Singularity can follow them.
export SINGULARITY_BIND="${BIDS_DIR}:${BIDS_DIR}:ro"

cd ${REPO_ROOT}/scripts

python run_fmriprep.py \
    --bids-dir "${NORDIC_BIDS}" \
    --output-dir "${NORDIC_OUTPUT}" \
    --work-dir "${NORDIC_WORK}" \
    --subjects sub-03 \
    --session ses-04 \
    --derivatives "${FMRIPREP_DERIVATIVES}" \
    --nprocs ${SLURM_CPUS_PER_TASK} \
    --mem-gb $((${SLURM_MEM_PER_NODE} / 1024))

EXIT_STATUS=$?

echo ""
echo "=========================================="
echo "Job Completion"
echo "=========================================="
echo "End Time: $(date)"
echo "Exit Status: ${EXIT_STATUS}"
echo "=========================================="

# Clean up
if [ -d "${SINGULARITY_TMPDIR}" ]; then
    rm -rf ${SINGULARITY_TMPDIR}
fi

exit ${EXIT_STATUS}
