#!/bin/bash
#SBATCH --job-name=dcm2bids
#SBATCH --partition=compute
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/dcm2bids_%j.out
#SBATCH --error=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/dcm2bids_%j.err

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
#   SUBJECT   - Subject ID (required, e.g. sub-03)
#   SESSION   - Session ID (required, e.g. ses-06 or "all")
#   REPO_ROOT - mmmdata repo root (set by submit_dcm2bids.sh)
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
if [ -z "${REPO_ROOT}" ]; then
    echo "ERROR: REPO_ROOT not set. Use submit_dcm2bids.sh or pass --export=REPO_ROOT=/path/to/mmmdata"
    exit 1
fi

# Resolve REPO_ROOT to absolute path
REPO_ROOT="$(cd "${REPO_ROOT}" && pwd)"

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

# Activate Python virtual environment first (load_config.sh needs Python 3.11+ for tomllib)
source "${REPO_ROOT}/.venv/bin/activate"

# Load configuration from config files
source "${REPO_ROOT}/scripts/load_config.sh"

# Change to repo root (run_dcm2bids.py expects this)
cd "${REPO_ROOT}"

# Run dcm2bids (config generation + conversion)
python scripts/run_dcm2bids.py \
    --subject "${SUBJECT}" \
    --session "${SESSION}" \
    --force

# Capture exit status
EXIT_STATUS=$?

# Defacing guard (anat-defacing, DECIDED 2026-09-11): no faced anatomy may
# stay under <bids_root>/sub-*/. If this session brought anat NIfTIs, deface
# them now. A subject without fMRIPrep gets the raw-only path (pydeface on
# each T1w, same-session T2w carried by flirt); one with fMRIPrep gets the
# full run, which is idempotent for the volumes already done. The mirror of
# every original lands in mmmsourcedata before the in-place write.
if [ "${EXIT_STATUS}" -eq 0 ]; then
    SUB_LABEL="${SUBJECT#sub-}"
    ANAT_COUNT=$(find "${BIDS_DIR}/${SUBJECT}" -path "*/anat/*" \( -name "*_T1w.nii.gz" -o -name "*_T2w.nii.gz" \) 2>/dev/null | wc -l)
    if [ "${ANAT_COUNT}" -gt 0 ]; then
        DEFACE_FLAGS=""
        if [ ! -d "${OUTPUT_DIR}/fmriprep/${SUBJECT}" ]; then
            DEFACE_FLAGS="--raw-only"
        fi
        echo "Defacing guard: ${ANAT_COUNT} anat NIfTIs under ${SUBJECT}; running deface_anat.py run ${DEFACE_FLAGS}"
        module load fsl/6.0.7.9 2>/dev/null || true
        export FSLOUTPUTTYPE=NIFTI_GZ
        python scripts/deface_anat.py run --subject "${SUB_LABEL}" ${DEFACE_FLAGS}
        EXIT_STATUS=$?
        python scripts/deface_anat.py audit || EXIT_STATUS=1
    fi
fi

# Print completion information
echo ""
echo "=========================================="
echo "Job Completion Information"
echo "=========================================="
echo "End Time: $(date)"
echo "Exit Status: ${EXIT_STATUS}"
echo "=========================================="

exit ${EXIT_STATUS}
