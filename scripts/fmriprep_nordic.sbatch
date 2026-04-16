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
# fMRIPrep on NORDIC-denoised data
#==============================================================================
# Runs fMRIPrep functional preprocessing on NORDIC-denoised BOLD data,
# reusing cached anatomical results from the existing fMRIPrep run.
#
# Usage:
#   sbatch --export=ALL,FPREP_SUBJECT=sub-03,FPREP_SESSION=ses-04 fmriprep_nordic.sbatch
#==============================================================================

# Validate required environment variables
if [ -z "${FPREP_SUBJECT}" ] || [ -z "${FPREP_SESSION}" ]; then
    echo "ERROR: FPREP_SUBJECT and FPREP_SESSION must be set."
    echo "Submit with: sbatch --export=ALL,FPREP_SUBJECT=sub-XX,FPREP_SESSION=ses-YY fmriprep_nordic.sbatch"
    exit 1
fi

echo "=========================================="
echo "fMRIPrep (NORDIC) — SLURM Job Info"
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
# Use anat-only trimmed derivatives so fMRIPrep does NOT reuse
# functional HMC/coreg from the original (non-NORDIC) run
FMRIPREP_DERIVATIVES="${BIDS_DIR}/derivatives/nordic/fmriprep_anat"

export SINGULARITYENV_TEMPLATEFLOW_HOME=${BIDS_DIR}/.cache/templateflow
export SINGULARITY_TMPDIR=${CODE_ROOT}/tmp/${SLURM_JOB_ID}
mkdir -p ${SINGULARITY_TMPDIR}

# No SINGULARITY_BIND needed — all NORDIC inputs are real copies (no symlinks)
# and run_fmriprep.py handles bind mounts for bids_dir, output_dir, work_dir

cd ${REPO_ROOT}/scripts

python run_fmriprep.py \
    --bids-dir "${NORDIC_BIDS}" \
    --output-dir "${NORDIC_OUTPUT}" \
    --work-dir "${NORDIC_WORK}" \
    --subjects ${FPREP_SUBJECT} \
    --session ${FPREP_SESSION} \
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

# Clean up Singularity temp
if [ -d "${SINGULARITY_TMPDIR}" ]; then
    rm -rf ${SINGULARITY_TMPDIR}
fi

# Clean up fMRIPrep work directory for this session on success.
# run_fmriprep.py isolates work per sub/ses (work_nordic/fmriprep/sub-XX_ses-YY),
# so this only removes the current session — no multi-session side effects.
if [ ${EXIT_STATUS} -eq 0 ]; then
    SUBJ_ID="${FPREP_SUBJECT#sub-}"
    SES_ID="${FPREP_SESSION#ses-}"
    SESSION_WORK="${NORDIC_WORK}/fmriprep/sub-${SUBJ_ID}_ses-${SES_ID}"
    if [ -d "${SESSION_WORK}" ]; then
        echo "Cleaning up work directory: ${SESSION_WORK}"
        rm -rf "${SESSION_WORK}"
    fi
else
    echo "fMRIPrep failed — preserving work directory for debugging."
fi

exit ${EXIT_STATUS}
