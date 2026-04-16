#!/bin/bash
#==============================================================================
# NORDIC Full-Scale Pipeline — Master Submission Script
#==============================================================================
# Submits the complete NORDIC → BIDS tree → fMRIPrep pipeline for all
# subjects and sessions, with proper SLURM dependency chains.
#
# Architecture:
#   For each subject (in parallel across subjects):
#     For each session (serialized fMRIPrep within subject):
#       1. NORDIC denoising (array job, 1 task per BOLD run)
#       2. BIDS input tree builder (depends on NORDIC completion)
#       3. fMRIPrep (depends on BIDS tree + previous session's fMRIPrep)
#
# Usage:
#   bash nordic_fullscale_submit.sh                    # dry run (default)
#   bash nordic_fullscale_submit.sh --submit           # actually submit
#   bash nordic_fullscale_submit.sh --submit --wave 1  # submit wave 1 only
#
# Waves: sessions are grouped into waves of WAVE_SIZE (default 10) to avoid
# flooding the scheduler. Run the next wave after the previous completes.
#==============================================================================

set -uo pipefail

# --- Configuration ---
WAVE_SIZE=10          # sessions per wave per subject
DRY_RUN=true
WAVE_NUM=0            # 0 = all waves

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --submit)  DRY_RUN=false; shift ;;
        --wave)    WAVE_NUM=$2; shift 2 ;;
        --wave-size) WAVE_SIZE=$2; shift 2 ;;
        -h|--help)
            echo "Usage: bash nordic_fullscale_submit.sh [--submit] [--wave N] [--wave-size N]"
            echo "  --submit     Actually submit jobs (default: dry run)"
            echo "  --wave N     Submit only wave N (default: all)"
            echo "  --wave-size  Sessions per wave per subject (default: 10)"
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# --- Paths ---
export REPO_ROOT="/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata"
source "${REPO_ROOT}/.venv/bin/activate"
source "${REPO_ROOT}/scripts/load_config.sh"

SCRIPTS="${REPO_ROOT}/scripts"
LOGS="${REPO_ROOT}/logs"
NORDIC_DIR="${BIDS_DIR}/derivatives/nordic"
FPREP_NORDIC_DIR="${BIDS_DIR}/derivatives/fmriprep_nordic"
SLURM_ACCOUNT="${SLURM_ACCOUNT:-hulacon}"

mkdir -p "${LOGS}"

# --- Subjects and sessions ---
SUBJECTS=(sub-03 sub-04 sub-05)

# All sessions with functional data (ses-01 through ses-28 + ses-30)
ALL_SESSIONS=()
for i in $(seq -w 1 28); do
    ALL_SESSIONS+=("ses-${i}")
done
ALL_SESSIONS+=("ses-30")

# --- Helper: check if a session is already fully processed ---
session_needs_nordic() {
    local sub=$1 ses=$2
    local raw_count nordic_count
    raw_count=$(ls "${BIDS_DIR}/${sub}/${ses}/func/"*_bold.nii.gz 2>/dev/null | wc -l)
    nordic_count=$(ls "${NORDIC_DIR}/${sub}/${ses}/func/"*_bold.nii.gz 2>/dev/null | wc -l)
    [ "${raw_count}" -gt 0 ] && [ "${nordic_count}" -lt "${raw_count}" ]
}

session_needs_bids_input() {
    local sub=$1 ses=$2
    local nordic_count bids_count
    nordic_count=$(ls "${NORDIC_DIR}/${sub}/${ses}/func/"*_bold.nii.gz 2>/dev/null | wc -l)
    bids_count=$(ls "${NORDIC_DIR}/bids_input/${sub}/${ses}/func/"*_bold.nii.gz 2>/dev/null | wc -l)
    [ "${nordic_count}" -gt 0 ] && [ "${bids_count}" -lt "${nordic_count}" ]
}

session_needs_fmriprep() {
    local sub=$1 ses=$2
    # Check for fMRIPrep HTML report as completion indicator
    local sub_id="${sub#sub-}"
    local ses_id="${ses#ses-}"
    [ ! -f "${FPREP_NORDIC_DIR}/${sub}/${ses}/${sub}_${ses}_*.html" ] 2>/dev/null
    # Fallback: check for func/ output directory
    [ ! -d "${FPREP_NORDIC_DIR}/${sub}/${ses}/func" ]
}

# --- Build session list per subject, filtering already-done ---
echo "=========================================="
echo "NORDIC Full-Scale Pipeline"
echo "=========================================="
echo "Mode: $([ "${DRY_RUN}" = true ] && echo 'DRY RUN' || echo 'SUBMITTING')"
echo "Wave size: ${WAVE_SIZE} sessions/subject"
[ "${WAVE_NUM}" -gt 0 ] && echo "Wave: ${WAVE_NUM}" || echo "Wave: ALL"
echo "=========================================="
echo ""

declare -A SUBJECT_SESSIONS
TOTAL_JOBS=0

for sub in "${SUBJECTS[@]}"; do
    sessions=()
    for ses in "${ALL_SESSIONS[@]}"; do
        # Skip if no BOLD data in raw
        raw_count=$(ls "${BIDS_DIR}/${sub}/${ses}/func/"*_bold.nii.gz 2>/dev/null | wc -l)
        if [ "${raw_count}" -eq 0 ]; then
            continue
        fi

        # Check if fully processed (fMRIPrep output exists)
        if [ -d "${FPREP_NORDIC_DIR}/${sub}/${ses}/func" ]; then
            continue
        fi

        sessions+=("${ses}")
    done

    # Apply wave filtering
    if [ "${WAVE_NUM}" -gt 0 ]; then
        start=$(( (WAVE_NUM - 1) * WAVE_SIZE ))
        wave_sessions=()
        for (( idx=start; idx < start + WAVE_SIZE && idx < ${#sessions[@]}; idx++ )); do
            wave_sessions+=("${sessions[idx]}")
        done
        sessions=("${wave_sessions[@]+"${wave_sessions[@]}"}")
    fi

    SUBJECT_SESSIONS[${sub}]="${sessions[*]:-}"
    n=${#sessions[@]}
    TOTAL_JOBS=$((TOTAL_JOBS + n))
    echo "${sub}: ${n} sessions to process"
    if [ ${n} -gt 0 ]; then
        echo "  ${sessions[*]}"
    fi
done

echo ""
echo "Total: ${TOTAL_JOBS} subject/session combinations"
echo ""

if [ "${TOTAL_JOBS}" -eq 0 ]; then
    echo "Nothing to submit. All sessions are processed."
    exit 0
fi

if [ "${DRY_RUN}" = true ]; then
    echo "--- DRY RUN complete. Re-run with --submit to submit jobs. ---"
    echo ""
    echo "Estimated resources per session:"
    echo "  NORDIC:   ~2h, 4 CPU, 32G (array: 1 task per BOLD run)"
    echo "  BIDS tree: ~5min, 1 CPU, 4G"
    echo "  fMRIPrep: ~4-8h, 8 CPU, 48G"
    exit 0
fi

# --- Submit jobs ---
echo "Submitting jobs..."
echo ""

# Track all job IDs for summary
declare -a ALL_JOB_IDS

for sub in "${SUBJECTS[@]}"; do
    sessions_str="${SUBJECT_SESSIONS[${sub}]:-}"
    if [ -z "${sessions_str}" ]; then
        continue
    fi
    read -ra sessions <<< "${sessions_str}"

    echo "=== ${sub} (${#sessions[@]} sessions) ==="

    PREV_FPREP_JOB=""

    for ses in "${sessions[@]}"; do
        raw_count=$(ls "${BIDS_DIR}/${sub}/${ses}/func/"*_bold.nii.gz 2>/dev/null | wc -l)

        echo ""
        echo "--- ${sub}/${ses} (${raw_count} BOLD runs) ---"

        # Step 1: NORDIC denoising (array job)
        # Check if NORDIC is already complete for this session
        nordic_count=$(ls "${NORDIC_DIR}/${sub}/${ses}/func/"*_bold.nii.gz 2>/dev/null | wc -l)
        NORDIC_DEP=""

        if [ "${nordic_count}" -ge "${raw_count}" ]; then
            echo "  NORDIC: already complete (${nordic_count}/${raw_count} runs)"
        else
            NORDIC_JOB=$(sbatch --parsable \
                --account="${SLURM_ACCOUNT}" \
                --export=ALL,NORDIC_SUBJECT="${sub}",NORDIC_SESSION="${ses}" \
                --array="1-${raw_count}" \
                "${SCRIPTS}/nordic_denoise.sbatch")
            echo "  NORDIC: job ${NORDIC_JOB} (array 1-${raw_count})"
            NORDIC_DEP="--dependency=afterok:${NORDIC_JOB}"
            ALL_JOB_IDS+=("${NORDIC_JOB}")
        fi

        # Step 2: BIDS input tree builder
        BIDS_TREE_DEPS="${NORDIC_DEP}"
        BIDS_TREE_JOB=$(sbatch --parsable \
            --account="${SLURM_ACCOUNT}" \
            --job-name="bids_tree_${sub}_${ses}" \
            --partition=compute \
            --mem=4G --time=00:30:00 --cpus-per-task=1 \
            --output="${LOGS}/bids_tree_${sub}_${ses}_%j.out" \
            --error="${LOGS}/bids_tree_${sub}_${ses}_%j.err" \
            ${BIDS_TREE_DEPS} \
            --export=ALL,NORDIC_SUBJECT="${sub}",NORDIC_SESSION="${ses}" \
            "${SCRIPTS}/nordic_build_bids_input.sh")
        echo "  BIDS tree: job ${BIDS_TREE_JOB}"
        ALL_JOB_IDS+=("${BIDS_TREE_JOB}")

        # Step 3: fMRIPrep (depends on BIDS tree + previous session's fMRIPrep)
        FPREP_DEPS="--dependency=afterok:${BIDS_TREE_JOB}"
        if [ -n "${PREV_FPREP_JOB}" ]; then
            FPREP_DEPS="--dependency=afterok:${BIDS_TREE_JOB},afterok:${PREV_FPREP_JOB}"
        fi

        FPREP_JOB=$(sbatch --parsable \
            --account="${SLURM_ACCOUNT}" \
            ${FPREP_DEPS} \
            --export=ALL,FPREP_SUBJECT="${sub}",FPREP_SESSION="${ses}" \
            "${SCRIPTS}/fmriprep_nordic.sbatch")
        echo "  fMRIPrep: job ${FPREP_JOB} (deps: ${FPREP_DEPS})"
        ALL_JOB_IDS+=("${FPREP_JOB}")

        PREV_FPREP_JOB="${FPREP_JOB}"
    done

    echo ""
done

# --- Summary ---
echo "=========================================="
echo "Submission Summary"
echo "=========================================="
echo "Total jobs submitted: ${#ALL_JOB_IDS[@]}"
echo ""
echo "Monitor with:"
echo "  squeue -u \$(whoami) --format='%.10i %.9P %.30j %.8T %.10M %.6D %R'"
echo ""
echo "Check specific job:"
echo "  sacct -j <JOBID> --format=JobID,State,Elapsed,ExitCode,MaxRSS -X"
echo ""
echo "After all jobs complete, verify:"
echo "  bash ${SCRIPTS}/nordic_fullscale_submit.sh  # dry run shows remaining"
