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
    array           Submit preprocessing for all subjects (parallel array job) [RECOMMENDED]
    test            Submit test run for first subject only (sub-03)

    -h, --help      Show this help message

EXAMPLES:
    # Test with one subject first (recommended)
    $0 test

    # Process all subjects in parallel
    $0 array

    # Process subjects sequentially
    $0 participant

NOTES:
    - fMRIPrep is participant-level only (no group stage)
    - Estimated runtime: 12-48 hours per subject (depends on number of sessions)
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
