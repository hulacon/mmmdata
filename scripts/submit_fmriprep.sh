#!/bin/bash
#
# submit_fmriprep.sh - Helper script to submit fMRIPrep jobs to SLURM
#
# This script provides a simple interface for submitting fMRIPrep jobs.
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Color output for better readability
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored messages
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to display usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Submit fMRIPrep jobs to SLURM cluster

OPTIONS:
    participant     Submit preprocessing for all subjects (sequential)
    array           Submit preprocessing for all subjects (parallel array job)
    test            Submit test run for first subject only (sub-03)
    anat            Submit anat-only preprocessing for all subjects (Stage 1)
    func            Submit per-session functional jobs for all subjects (Stage 2)
    split           Submit anat first, then func with dependency (RECOMMENDED)

    -h, --help      Show this help message

EXAMPLES:
    # Recommended: split into anat + per-session func (handles large datasets)
    $0 split

    # Just run anat-only stage
    $0 anat

    # Submit func jobs after anat has completed
    $0 func

    # Process all subjects in parallel (all-at-once, for small datasets)
    $0 array

NOTES:
    - fMRIPrep is participant-level only (no group stage)
    - For datasets with many sessions, use 'split' to avoid wall time limits
    - The split pipeline: anat (48hr) -> pilot session 1 (8hr computelong) -> remaining sessions (4hr compute, %6 throttle)
    - The pilot session establishes anat postprocessing outputs (ribbon, probseg
      warps, MNI transforms) before parallel sessions start, avoiding write races
    - Output: /projects/hulacon/shared/mmmdata/derivatives/fmriprep/

EOF
}

# Function to check if email is configured
check_email() {
    local batch_file=$1
    if grep -q "your-email@uoregon.edu" "$batch_file"; then
        print_warning "Email not configured in $batch_file"
        print_warning "Edit the file and replace 'your-email@uoregon.edu' with your email"
        return 1
    fi
    return 0
}

# Function to submit participant job
submit_participant() {
    print_info "Submitting fMRIPrep for all subjects (sequential)..."
    check_email "${SCRIPT_DIR}/fmriprep_participant.sbatch"

    JOB_ID=$(sbatch --parsable "${SCRIPT_DIR}/fmriprep_participant.sbatch")

    if [ $? -eq 0 ]; then
        print_success "Job submitted successfully!"
        print_info "Job ID: ${JOB_ID}"
        print_info "Monitor with: squeue -u \$USER"
        print_info "View logs: tail -f ${SCRIPT_DIR}/../logs/fmriprep_participant_${JOB_ID}.out"
        echo "$JOB_ID"
        return 0
    else
        print_error "Job submission failed"
        return 1
    fi
}

# Function to submit array job
submit_array() {
    print_info "Submitting fMRIPrep for all subjects (parallel array)..."
    check_email "${SCRIPT_DIR}/fmriprep_array.sbatch"

    # Count subjects
    N_SUBJECTS=$(ls -d /projects/hulacon/shared/mmmdata/sub-* 2>/dev/null | wc -l)
    print_info "Found ${N_SUBJECTS} subjects"

    JOB_ID=$(sbatch --parsable "${SCRIPT_DIR}/fmriprep_array.sbatch")

    if [ $? -eq 0 ]; then
        print_success "Array job submitted successfully!"
        print_info "Job ID: ${JOB_ID}"
        print_info "Array tasks: 1-${N_SUBJECTS}"
        print_info "Monitor with: squeue -u \$USER -r"
        print_info "View logs: tail -f ${SCRIPT_DIR}/../logs/fmriprep_array_${JOB_ID}_*.out"
        echo "$JOB_ID"
        return 0
    else
        print_error "Job submission failed"
        return 1
    fi
}

# Function to submit test job (single subject)
submit_test() {
    print_info "Submitting fMRIPrep test run (first subject only)..."
    check_email "${SCRIPT_DIR}/fmriprep_array.sbatch"

    # Override array to just task 1 (first subject)
    JOB_ID=$(sbatch --parsable --array=1-1 "${SCRIPT_DIR}/fmriprep_array.sbatch")

    if [ $? -eq 0 ]; then
        # Determine which subject is first
        FIRST_SUBJECT=$(ls -d /projects/hulacon/shared/mmmdata/sub-* 2>/dev/null | head -1 | xargs basename)
        print_success "Test job submitted successfully!"
        print_info "Job ID: ${JOB_ID}"
        print_info "Processing: ${FIRST_SUBJECT}"
        print_info "Monitor with: squeue -u \$USER"
        print_info "View logs: tail -f ${SCRIPT_DIR}/../logs/fmriprep_array_${JOB_ID}_1.out"
        echo "$JOB_ID"
        return 0
    else
        print_error "Job submission failed"
        return 1
    fi
}

# Function to submit anat-only job
submit_anat() {
    print_info "Submitting fMRIPrep anat-only for all subjects..."
    check_email "${SCRIPT_DIR}/fmriprep_anat.sbatch"

    N_SUBJECTS=$(ls -d /projects/hulacon/shared/mmmdata/sub-* 2>/dev/null | wc -l)
    print_info "Found ${N_SUBJECTS} subjects"

    JOB_ID=$(sbatch --parsable "${SCRIPT_DIR}/fmriprep_anat.sbatch")

    if [ $? -eq 0 ]; then
        print_success "Anat-only job submitted successfully!"
        print_info "Job ID: ${JOB_ID}"
        print_info "Array tasks: 1-${N_SUBJECTS}"
        print_info "Monitor with: squeue -u \$USER -r"
        print_info "View logs: tail -f ${SCRIPT_DIR}/../logs/fmriprep_anat_${JOB_ID}_*.out"
        echo "$JOB_ID"
        return 0
    else
        print_error "Job submission failed"
        return 1
    fi
}

# Function to submit per-session functional jobs for all subjects
#
# Uses a pilot-then-parallel strategy to avoid race conditions:
#   1. Session 1 runs first (alone) to establish anat postprocessing outputs
#      (ribbon mask, MNI probseg warps, etc.) in the shared derivatives dir.
#   2. Remaining sessions launch after the pilot completes, with %6 throttle.
#
# This prevents 29 concurrent sessions from racing on DerivativesDataSink
# writes to the same anat output files.
#
# TODO (future): Fold anat postprocessing into the anat-only stage or a
# dedicated "stage 1.5" so the pilot session is no longer needed. The
# --anat-only flag stops before postprocessing (ribbon, probseg warps,
# MNI transforms), so currently the func stage must redo it.
submit_func() {
    local dependency=${1:-}

    BIDS_DIR="/projects/hulacon/shared/mmmdata"
    SUBJECTS=($(ls -d ${BIDS_DIR}/sub-* | xargs -n 1 basename | sort))
    print_info "Found ${#SUBJECTS[@]} subjects"

    for SUBJECT in "${SUBJECTS[@]}"; do
        # Count sessions for this subject
        N_SESSIONS=$(ls -d ${BIDS_DIR}/${SUBJECT}/ses-* 2>/dev/null | wc -l)

        if [ ${N_SESSIONS} -eq 0 ]; then
            print_warning "No sessions found for ${SUBJECT}, skipping"
            continue
        fi

        print_info "Submitting func jobs for ${SUBJECT} (${N_SESSIONS} sessions)..."

        # Stage 2a (stage 1.5): Pilot session runs first to establish anat
        # postprocessing outputs (ribbon, probseg warps, MNI transforms).
        # Gets extra time on computelong since it must redo anat postprocessing
        # that --anat-only skips, in addition to func processing.
        PILOT_ARGS="--parsable --export=ALL,FMRIPREP_SUBJECT=${SUBJECT} --array=1"
        PILOT_ARGS="${PILOT_ARGS} --partition=computelong --time=08:00:00"
        if [ -n "${dependency}" ]; then
            PILOT_ARGS="${PILOT_ARGS} --dependency=afterok:${dependency}"
        fi

        PILOT_JOB_ID=$(sbatch ${PILOT_ARGS} "${SCRIPT_DIR}/fmriprep_func.sbatch")

        if [ $? -eq 0 ]; then
            print_success "${SUBJECT}: pilot session submitted as job ${PILOT_JOB_ID}"
        else
            print_error "${SUBJECT}: pilot session submission failed"
            continue
        fi

        # Stage 2b: Remaining sessions after pilot completes, with throttle
        if [ ${N_SESSIONS} -gt 1 ]; then
            REMAINING_ARGS="--parsable --export=ALL,FMRIPREP_SUBJECT=${SUBJECT} --array=2-${N_SESSIONS}%6"
            REMAINING_ARGS="${REMAINING_ARGS} --dependency=afterok:${PILOT_JOB_ID}"

            REMAINING_JOB_ID=$(sbatch ${REMAINING_ARGS} "${SCRIPT_DIR}/fmriprep_func.sbatch")

            if [ $? -eq 0 ]; then
                print_success "${SUBJECT}: sessions 2-${N_SESSIONS} submitted as job ${REMAINING_JOB_ID} (after pilot, %6 throttle)"
            else
                print_error "${SUBJECT}: remaining sessions submission failed"
            fi
        fi
    done
}

# Function to submit split pipeline (anat -> func with dependency)
submit_split() {
    print_info "Submitting split pipeline: anat-only -> per-session func"
    echo ""

    # Stage 1: anat-only
    print_info "=== Stage 1: Anatomical preprocessing ==="
    check_email "${SCRIPT_DIR}/fmriprep_anat.sbatch"
    check_email "${SCRIPT_DIR}/fmriprep_func.sbatch"

    N_SUBJECTS=$(ls -d /projects/hulacon/shared/mmmdata/sub-* 2>/dev/null | wc -l)
    ANAT_JOB_ID=$(sbatch --parsable "${SCRIPT_DIR}/fmriprep_anat.sbatch")

    if [ $? -ne 0 ]; then
        print_error "Anat job submission failed"
        return 1
    fi

    print_success "Anat job submitted: ${ANAT_JOB_ID} (${N_SUBJECTS} subjects)"
    echo ""

    # Stage 2: per-session func (depends on anat completing)
    # Uses pilot-then-parallel: session 1 first (anat postprocessing), then rest
    print_info "=== Stage 2: Functional preprocessing (pilot + parallel) ==="
    submit_func "${ANAT_JOB_ID}"

    echo ""
    print_success "Split pipeline submitted!"
    print_info "Anat job: ${ANAT_JOB_ID}"
    print_info "Pilot sessions will start after anat completes"
    print_info "Remaining sessions will start after each pilot completes (%6 throttle)"
    print_info "Monitor with: squeue -u \$USER -r"
}

# Main script
if [ $# -eq 0 ]; then
    print_error "No option specified"
    usage
    exit 1
fi

case "$1" in
    participant)
        submit_participant
        ;;

    array)
        submit_array
        ;;

    test)
        submit_test
        ;;

    anat)
        submit_anat
        ;;

    func)
        submit_func
        ;;

    split)
        submit_split
        ;;

    -h|--help)
        usage
        exit 0
        ;;

    *)
        print_error "Unknown option: $1"
        usage
        exit 1
        ;;
esac
